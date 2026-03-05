from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from .layouts import *
from .flow import *
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
from graphviz import Digraph
import copy


# ---------------------------------------------------------------------------
# Operation abstraction
# ---------------------------------------------------------------------------

class Operation(ABC):
    """Standard lifecycle interface for GPU async operations (TMA, MMA, CUDA).

    Every operation follows a three-phase model:
      produce  – issue the async work (start a TMA load, launch an MMA, etc.)
      consume  – wait for the work to complete and acquire the result
      done     – release resources (signal buffer empty / semaphore arrive)

    This uniform interface lets the optimizer reason about every operation
    type identically when building the pipelined schedule.
    """

    @abstractmethod
    def produce(self, graph: 'DataflowGraph',
                parent_instrs: List['Instruction'] = []) -> 'Instruction':
        """Issue the async operation and return the resulting Instruction."""

    @abstractmethod
    def consume(self, graph: 'DataflowGraph',
                parent_instrs: List['Instruction'] = []) -> 'Instruction':
        """Wait for the operation to complete and return the resulting Instruction."""

    @abstractmethod
    def done(self, graph: 'DataflowGraph',
             parent_instrs: List['Instruction'] = []) -> 'Instruction':
        """Release resources after the result has been consumed."""

    @property
    @abstractmethod
    def latency(self) -> int:
        """Approximate pipeline latency (used by the cost-model scheduler)."""


class TMAOperation(Operation):
    """Tensor Memory Accelerator async load operation.

    produce  – issue async DMA from global → shared (wait_empty + async_load)
    consume  – wait for the tile to be ready (wait_full barrier)
    done     – release the buffer (arrive_empty signal)
    """

    def __init__(self, loads: List['MemLoad']):
        self.loads = loads
        self.tiles = [l.dest for l in loads]

    @property
    def latency(self) -> int:
        return 12  # TMA loads have ~3x longer latency than MMA

    def produce(self, graph: 'DataflowGraph',
                parent_instrs: List['Instruction'] = []) -> 'Instruction':
        tile_group = graph.get_tile_group_from_list(self.tiles)
        new_version_tiles = [graph.increment_version(t) for t in self.tiles]
        for t in self.tiles:
            graph.parent_tile_group[t] = tile_group
        instruction = Instruction(
            stmt=SeqStmt([
                tile_group.wait_empty(Level.warpgroup),
                tile_group.async_load_global(self.loads),
            ]),
            reads=[],
            writes=new_version_tiles,
            name=f"tma_produce_{[str(t) for t in new_version_tiles]}",
            source_op=self,
        )
        graph.add_instruction(instruction)
        for parent_instr in parent_instrs:
            graph.G.add_edge(parent_instr, instruction)
        return instruction

    def consume(self, graph: 'DataflowGraph',
                parent_instrs: List['Instruction'] = []) -> 'Instruction':
        tile_group = graph.get_tile_group_from_list(self.tiles)
        for t in self.tiles:
            assert graph.parent_tile_group.get(t) == tile_group, (
                f"TileGroup {tile_group} and parent of tile {t} "
                f"({graph.parent_tile_group.get(t)}) must match on synchronization"
            )
        versioned_tiles = [graph.current_version(t) for t in self.tiles]
        instruction = Instruction(
            stmt=SeqStmt([tile_group.wait_full(Level.warpgroup)]),
            reads=versioned_tiles,
            writes=[],
            name=f"tma_consume_{[str(t) for t in versioned_tiles]}",
        )
        graph.add_instruction(instruction)
        for parent_instr in parent_instrs:
            graph.G.add_edge(parent_instr, instruction)
        return instruction

    def done(self, graph: 'DataflowGraph',
             parent_instrs: List['Instruction'] = []) -> 'Instruction':
        tile_group = graph.get_tile_group_from_list(self.tiles)
        for t in self.tiles:
            assert graph.parent_tile_group.get(t) == tile_group, (
                f"TileGroup {tile_group} and parent of tile {t} "
                f"({graph.parent_tile_group.get(t)}) must match on synchronization"
            )
        versioned_tiles = [graph.current_version(t) for t in self.tiles]
        instruction = Instruction(
            stmt=SeqStmt([tile_group.arrive_empty()]),
            reads=versioned_tiles,
            writes=[],
            name=f"tma_done_{[str(t) for t in versioned_tiles]}",
        )
        graph.add_instruction(instruction)
        for parent_instr in parent_instrs:
            graph.G.add_edge(parent_instr, instruction)
        return instruction


class MMAOperation(Operation):
    """Warpgroup async Matrix-Multiply-Accumulate operation.

    produce  – issue async tensor-core op (warpgroup::mma_AB)
    consume  – wait for completion (warpgroup::mma_async_wait)
    done     – same as consume (MMA has no separate release signal)
    """

    def __init__(self, out_var: 'Tile', a: 'Tile', b: 'Tile'):
        self.out_var = out_var
        self.a = a
        self.b = b

    @property
    def latency(self) -> int:
        return 4  # MMA is the most expensive operation

    def produce(self, graph: 'DataflowGraph',
                parent_instrs: List['Instruction'] = []) -> 'Instruction':
        versioned_a  = graph.current_version(self.a)
        versioned_b  = graph.current_version(self.b)
        versioned_out     = graph.current_version(self.out_var)
        next_versioned_out = graph.increment_version(self.out_var)
        instruction = Instruction(
            stmt=SeqStmt([
                OpCall("warpgroup::mma_AB", [self.out_var, self.a, self.b]).to_stmt()
            ]),
            reads=[versioned_a, versioned_b, versioned_out],
            writes=[next_versioned_out],
            name=f"mma_issue_{self.out_var}",
        )
        graph.add_instruction(instruction)
        for parent_instr in parent_instrs:
            graph.G.add_edge(parent_instr, instruction)
        return instruction

    def consume(self, graph: 'DataflowGraph',
                parent_instrs: List['Instruction'] = []) -> 'Instruction':
        versioned_out = graph.current_version(self.out_var)
        instruction = Instruction(
            stmt=SeqStmt([
                OpCall("warpgroup::mma_async_wait", []).to_stmt()
            ]),
            reads=[versioned_out],
            writes=[],
            name=f"mma_wait_{self.out_var}",
        )
        graph.add_instruction(instruction)
        for parent_instr in parent_instrs:
            graph.G.add_edge(parent_instr, instruction)
        return instruction

    def done(self, graph: 'DataflowGraph',
             parent_instrs: List['Instruction'] = []) -> 'Instruction':
        # MMA has no separate "done" signal; consume == done
        return self.consume(graph, parent_instrs=parent_instrs)


class CUDAOperation(Operation):
    """Generic synchronous CUDA operation (produce == consume == done).

    For operations with no async latency (e.g. warp::zero, warp::store)
    all three phases are the same: call the op and continue.
    """

    def __init__(self, name: str, parameters: List, out: 'Tile',
                 ins: List['Tile']):
        self.name = name
        self.parameters = parameters
        self.out = out
        self.ins = ins

    @property
    def latency(self) -> int:
        return 1

    def produce(self, graph: 'DataflowGraph',
                parent_instrs: List['Instruction'] = []) -> 'Instruction':
        versioned_ins = [graph.current_version(iv) for iv in self.ins]
        versioned_out = graph.increment_version(self.out)
        instruction = Instruction(
            stmt=SeqStmt([OpCall(self.name, self.parameters).to_stmt()]),
            reads=versioned_ins,
            writes=[versioned_out],
            name=f"{self.name}_{versioned_out}",
        )
        graph.add_instruction(instruction)
        for parent_instr in parent_instrs:
            graph.G.add_edge(parent_instr, instruction)
        return instruction

    def consume(self, graph: 'DataflowGraph',
                parent_instrs: List['Instruction'] = []) -> 'Instruction':
        return self.produce(graph, parent_instrs=parent_instrs)

    def done(self, graph: 'DataflowGraph',
             parent_instrs: List['Instruction'] = []) -> 'Instruction':
        return self.produce(graph, parent_instrs=parent_instrs)

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
    # Loop annotation — set to True by DataflowGraph.loop(); lets the
    # optimizer locate and inspect loop instructions without name-matching.
    is_loop: bool = False
    # Optional loop metadata stored for the optimizer.
    body_graph: Optional[Any] = field(default=None, compare=False, hash=False)
    loop_var: Optional[Any]   = field(default=None, compare=False, hash=False)
    loop_bound: Optional[Any] = field(default=None, compare=False, hash=False)
    # The Operation object that issued this instruction.  Stored so the
    # pipeline optimizer can recreate the operation on different tiles/coords.
    source_op: Optional[Any]  = field(default=None, compare=False, hash=False)

    def __hash__(self):
        return hash(str(self) + ",".join([str(r) for r in self.reads]) + "|".join([str(w) for w in self.writes]))

    def __eq__(self, other):
        if not isinstance(other, Instruction):
            return False
        return (self.name, self.reads, self.writes) == (other.name, other.reads, other.writes)

    def __str__(self) -> str:
        return f"{self.name}"

def substitute_expr(expr: Expr, find: Expr, replace: Expr) -> Expr:
    if str(expr) == str(find):
        return replace
    elif isinstance(expr, BinaryOp):
        return BinaryOp(
            substitute_expr(expr.a, find, replace),
            substitute_expr(expr.b, find, replace),
            expr.op_type
        )
    elif isinstance(expr, Coord):
        return Coord([substitute_expr(e, find, replace) for e in expr.elements])
    elif isinstance(expr, OpCall):
        return OpCall(expr.function_name, [substitute_expr(e, find, replace) for e in expr.inputs])
    elif isinstance(expr, AssignExpr):
        return AssignExpr(
            substitute_expr(expr.lhs, find, replace),
            substitute_expr(expr.rhs, find, replace)
        )
    else:
        return expr

def substitute_stmt(stmt: Stmt, find: Expr, replace: Expr) -> Stmt:
    if isinstance(stmt, ExprStmt):
        return ExprStmt(substitute_expr(stmt.expr, find, replace))
    elif isinstance(stmt, SeqStmt):
        return SeqStmt([substitute_stmt(s, find, replace) for s in stmt.stmts])
    elif isinstance(stmt, ForStmt):
        return ForStmt(
            substitute_expr(stmt.init, find, replace),
            substitute_expr(stmt.cond, find, replace),
            substitute_expr(stmt.step, find, replace),
            substitute_stmt(stmt.body, find, replace)
        )
    elif isinstance(stmt, IfStmt):
        return IfStmt(
            substitute_expr(stmt.cond, find, replace),
            substitute_stmt(stmt.then_stmt, find, replace),
            substitute_stmt(stmt.else_stmt, find, replace) if stmt.else_stmt else None
        )
    elif isinstance(stmt, AssignExpr):
        return AssignExpr(
            substitute_expr(stmt.lhs, find, replace),
            substitute_expr(stmt.rhs, find, replace)
        )
    else:
        return stmt

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

    def subscope(self) -> DataflowGraph:
        child = DataflowGraph()
        child.versioned_tiles = self.versioned_tiles
        child.tiles = self.tiles
        return child

    def loop(self, i_var: Var, bound: Expr, body: 'DataflowGraph',
             reads: List[Tile], writes: List[Tile],
             increment_expr: Optional[Expr] = None) -> Instruction:
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
            name=f"for_loop_{i_var.name}",
            # Annotate as loop so the optimizer can find and inspect this node.
            is_loop=True,
            body_graph=body,
            loop_var=i_var,
            loop_bound=bound,
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

    def clone_with_substitution(self, find: Expr, replace: Expr) -> 'DataflowGraph':
        new_graph = copy.deepcopy(self)
        new_graph.instructions = []
        for instr in self.instructions:
            new_graph.instructions.append(Instruction(
                stmt=substitute_stmt(instr.stmt, find, replace),
                reads=instr.reads,
                writes=instr.writes,
                name=instr.name + "_shifted",
                source_op=instr.source_op,
            ))
        return new_graph

    def get_instruction_graph(self) -> nx.DiGraph:
        """Return a DiGraph over Instruction nodes only.

        Collapses VersionedTile intermediaries: an edge A → B is added whenever
        instruction A writes a VersionedTile that instruction B reads, i.e. B has
        a data dependency on A.  Use this instead of re-deriving deps from the
        reads/writes fields so the optimizer uses the same ground-truth graph
        that DataflowGraph.G already encodes.
        """
        G_i = nx.DiGraph()
        instrs = [n for n in self.G.nodes() if isinstance(n, Instruction)]
        G_i.add_nodes_from(instrs)
        for instr in instrs:
            # predecessors of instr in G are the VersionedTiles it reads
            for vt in self.G.predecessors(instr):
                if isinstance(vt, VersionedTile):
                    # predecessors of vt are the Instructions that wrote it
                    for writer in self.G.predecessors(vt):
                        if isinstance(writer, Instruction) and writer is not instr:
                            G_i.add_edge(writer, instr)
        return G_i

def draw_graph(graph: DataflowGraph, outpath: str):
    dot = Digraph(graph_attr={'rankdir': 'TB', 'splines': 'ortho', 'nodesep': '0.5'})
    dot.attr('node', shape='ellipse', style='filled', fillcolor='steelblue', 
             fontcolor='white', fontsize='10')
    
    for node in graph.G.nodes():
        dot.node(str(id(node)), label=str(node))
    
    for u, v in graph.G.edges():
        dot.edge(str(id(u)), str(id(v)))
    
    dot.render(outpath.replace('.png', ''), format='png', cleanup=True)