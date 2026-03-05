from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from .layouts import *
from .flow import *
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
from graphviz import Digraph

@dataclass(frozen=True)
class VersionedTile:
    tile: Tile
    version: int

    def __str__(self) -> str:
        return f"{self.tile.name}@{self.version}"

    def __hash__(self):
        return hash(str(self))
    
    def __eq__(self, other):
        if not isinstance(other, VersionedTile):
            return False
        return str(self) == str(other)

@dataclass(frozen=True)
class Instruction:
    stmt: Stmt
    reads: List[VersionedTile]
    writes: List[VersionedTile]
    name: str

    def __hash__(self):
        return hash(str(self) + ",".join([str(r) for r in self.reads]) + "|".join([str(w) for w in self.writes]))

    def __eq__(self, other):
        if not isinstance(other, Instruction):
            return False
        return (self.name, self.reads, self.writes) == (self.name, self.reads, self.writes)

    def __str__(self) -> str:
        return f"{self.name}"

class DataflowGraph:
    def __init__(self, kernel_globals: KernelGlobals = KernelGlobals("globals")):
        self.G = nx.DiGraph()
        self.kernel_globals = kernel_globals
        self.instructions = []

        self.versioned_tiles: Dict[Tile, VersionedTile] = {}
        self.tiles: List[Tile] = []

        self.parent_tile_group: Dict[Tile, TileGroup] = {}

        self.instructions = []

    def increment_version(self, tile: Tile) -> VersionedTile:
        if not tile in self.tiles:
            self.tiles.append(tile)
            self.versioned_tiles[tile] = VersionedTile(
                tile=tile,
                version=0
            )
            self.G.add_node(self.versioned_tiles[tile])
        
        self.versioned_tiles[tile] = VersionedTile(
            tile=tile,
            version=self.versioned_tiles[tile].version + 1
        )
        return self.versioned_tiles[tile]
    
    def current_version(self, tile: Tile) -> VersionedTile:
        if tile not in self.versioned_tiles:
            self.increment_version(tile)
        return self.versioned_tiles.get(tile)

    def add_instruction(self, instruction: Instruction):
        self.instructions.append(instruction)
        for read in instruction.reads:
            self.G.add_edge(read, instruction)
        for write in instruction.writes:
            self.G.add_edge(instruction, write)
    
    def get_tile_group_from_list(self, tiles: List[Tile]):
        name = "_".join(sorted(tile.name for tile in tiles))
        return TileGroup(name, tiles)

    def tma_produce(self, load: List[MemLoad]):
        tiles = [l.dest for l in load]
        tile_group = self.get_tile_group_from_list(tiles)

        new_version_tiles = [self.increment_version(tile) for tile in tiles]
        
        for tile in tiles:
            self.parent_tile_group[tile] = tile_group

        instruction = Instruction(
            stmt=SeqStmt([
                tile_group.wait_empty(Level.warpgroup),
                tile_group.async_load_global(load)
            ]),
            reads=[],
            writes=new_version_tiles,
            name=f"tma_produce_{[str(t) for t in new_version_tiles]}"
        )

        self.add_instruction(instruction)
        return instruction

    def subscope(self) -> DataflowGraph:
        child = DataflowGraph()
        child.versioned_tiles = self.versioned_tiles
        child.tiles = self.tiles
        return child
    
    def tma_consume(self, tiles: List[Tile]) -> Instruction:
        tile_group = self.get_tile_group_from_list(tiles)
        for tile in tiles:
            assert self.parent_tile_group.get(tile) == tile_group, f"TileGroup {tile_group} and parent of tile {tile} ({self.parent_tile_group.get(tile)}) must match on synchronization"
        versioned_tiles = [self.current_version(tile) for tile in tiles]
        instruction = Instruction(
            stmt=SeqStmt([
                tile_group.wait_full(Level.warpgroup)
            ]),
            reads=versioned_tiles,
            writes=[],
            name=f"tma_consume_{[str(t) for t in versioned_tiles]}"
        )
        self.add_instruction(instruction)
        return instruction

    def tma_done(self, tiles: List[Tile], parent_instrs: List[Instruction] = []) -> Instruction:
        tile_group = self.get_tile_group_from_list(tiles)
        for tile in tiles:
            assert self.parent_tile_group.get(tile) == tile_group, f"TileGroup {tile_group} and parent of tile {tile} ({self.parent_tile_group.get(tile)}) must match on synchronization"
        versioned_tiles = [self.current_version(tile) for tile in tiles]
        instruction = Instruction(
            stmt=SeqStmt([
                tile_group.arrive_empty()
            ]),
            reads=versioned_tiles,
            writes=[],
            name=f"tma_done_{[str(t) for t in versioned_tiles]}"
        )
        self.add_instruction(instruction)
        for parent_instr in parent_instrs:
            self.G.add_edge(parent_instr, instruction)
        return instruction

    def mma_issue(self, out_var: Tile, a: Tile, b: Tile, parent_instrs: List[Instruction] = []) -> Instruction:
        versioned_a, versioned_b = self.current_version(a), self.current_version(b)
        versioned_out, next_versioned_out = self.current_version(out_var), self.increment_version(out_var)

        instruction = Instruction(
            stmt=SeqStmt([
                OpCall("warpgroup::mma_AB", [out_var, a, b]).to_stmt() 
            ]),
            reads=[versioned_a, versioned_b, versioned_out],
            writes=[next_versioned_out],
            name=f"mma_issue_{out_var}"
        )
        self.add_instruction(instruction)
        for parent_instr in parent_instrs:
            self.G.add_edge(parent_instr, instruction)
        return instruction
    
    def mma_wait(self, out_var: Tile, parent_instrs: List[Instruction] = []) -> Instruction:
        versioned_out = self.current_version(out_var)
        instruction = Instruction(
            stmt=SeqStmt([
                OpCall("warpgroup::mma_async_wait", []).to_stmt()
            ]),
            reads=[versioned_out],
            writes=[],
            name=f"mma_wait_{out_var}"
        )
        self.add_instruction(instruction)
        for parent_instr in parent_instrs:
            self.G.add_edge(parent_instr, instruction)
        return instruction

    def op(self, name: str, parameters: List[Var], out: Tile, ins: List[Tile], parent_instrs: List[Instruction] = []) -> Instruction:
        versioned_ins = [self.current_version(iv) for iv in ins]
        versioned_out = self.increment_version(out)
        instruction = Instruction(
            stmt=SeqStmt([
                OpCall(name, parameters).to_stmt()
            ]),
            reads=versioned_ins,
            writes=[versioned_out],
            name=f"{name}_{versioned_out}"
        )
        self.add_instruction(instruction)
        for parent_instr in parent_instrs:
            self.G.add_edge(parent_instr, instruction)
        return instruction

    def loop(self, i_var: Var, bound: Expr, body: DataflowGraph, reads: List[Tile], writes: List[Tiles], increment_expr: Optional[Expr] = None):
        for tile in body.tiles:
            if tile not in self.tiles:
                self.tiles.append(tile)
        self.versioned_tiles.update(body.versioned_tiles)
        self.parent_tile_group.update(body.parent_tile_group)

        read_versions = [self.current_version(t) for t in reads]

        self.G = nx.compose(self.G, body.G)
        instruction = Instruction(
            stmt=SeqStmt([
                i_var.declare(),
                ForStmt(
                    AssignExpr(i_var, zero),
                    BinaryOp(i_var, bound, "<"),
                    AssignExpr(i_var, BinaryOp(i_var, RawExpr(1), "+")) if not increment_expr else increment_expr,
                    body.emit_stmt()
                )
            ]),
            reads=read_versions,
            writes=[self.increment_version(t) for t in writes],
            name=f"for_loop_{i_var.name}"
        )

        self.add_instruction(instruction)
        return instruction

    def emit_stmt(self):
        stmt_list = []
        for instruction in self.instructions:
            stmt_list.append(instruction.stmt)
        return SeqStmt(stmt_list)

    def emit_program(self) -> Program:
        stmt_list = []

        tg_seen = []
        for tile in self.tiles:
            stmt_list.append(tile.declare())
            parent_tg = self.parent_tile_group.get(tile)
            if parent_tg and parent_tg not in tg_seen:
                tg_seen.append(parent_tg)

        for tile_group in tg_seen:
            stmt_list.append(tile_group.initialize())

        for instruction in self.instructions:
            stmt_list.append(instruction.stmt)
        
        return Program(
            kernel_vars=self.kernel_globals,
            kernel_stmt=SeqStmt(stmt_list)
        )

def draw_graph(graph: DataflowGraph, outpath: str):
    dot = Digraph(graph_attr={'rankdir': 'TB', 'splines': 'ortho', 'nodesep': '0.5'})
    dot.attr('node', shape='ellipse', style='filled', fillcolor='steelblue', 
             fontcolor='white', fontsize='10')
    
    for node in graph.G.nodes():
        dot.node(str(id(node)), label=str(node))
    
    for u, v in graph.G.edges():
        dot.edge(str(id(u)), str(id(v)))
    
    dot.render(outpath.replace('.png', ''), format='png', cleanup=True)