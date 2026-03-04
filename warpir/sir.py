from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
from .layouts import *
from .flow import *
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

@dataclass
class VersionedTile:
    tile: Tile
    version: int
    context: SIR

class VersionedEdge:
    def __init__(self, vtiles: List[VersionedTile]):
        self.vtiles     = vtiles
        self._produced  = False
        self._done:     set = set()
        self._consumed  = False

    def produce(self) -> Stmt:
        if self._produced: return NoStmt()
        self._produced = True
        return self._produce_stmt()

    def consume(self, stmt: Stmt) -> Stmt:
        first = not self._consumed
        self._consumed = True
        return self._consume_stmt(stmt, first)

    def done(self, vtile: VersionedTile) -> Stmt:
        self._done.add(vtile)
        if self._done == set(self.vtiles):
            return self._done_stmt()
        return NoStmt()

    def _produce_stmt(self) -> Stmt:  return NoStmt()
    def _consume_stmt(self, stmt: Stmt, first: bool) -> Stmt: return stmt
    def _done_stmt(self) -> Stmt:     return NoStmt()


class VersionedTMAEdge(VersionedEdge):
    def __init__(self, group: TileGroup, vtiles: List[VersionedTile], loads: List[MemLoad]):
        super().__init__(vtiles)
        self.group = group
        self.loads = loads

    def _produce_stmt(self) -> Stmt:
        return SeqStmt([
            self.group.wait_empty(Level.warpgroup),
            self.group.async_load_global(self.loads),
        ])

    def _consume_stmt(self, stmt: Stmt, first: bool) -> Stmt:
        return SeqStmt([self.group.wait_full(Level.warpgroup), stmt]) if first else stmt

    def _done_stmt(self) -> Stmt:
        return self.group.arrive_empty()


class VersionedMMAEdge(VersionedEdge):
    def __init__(self, a: Tile, b: Tile, out: Tile):
        super().__init__([VersionedTile(out, -1)])  # single output
        self.a   = a
        self.b   = b
        self.out = out

    def _produce_stmt(self) -> Stmt:
        return OpCall("warpgroup::mma_AB", [self.out, self.a, self.b]).to_stmt()

    def _consume_stmt(self, stmt: Stmt, first: bool) -> Stmt:
        return SeqStmt([OpCall("warpgroup::mma_async_wait", []).to_stmt(), stmt]) if first else stmt

class SIR:
    def __init__(self, parent: SIR = None, tiles: List[Tile]):
        self.kernel_globals: KernelGlobals = KernelGlobals()        
        self.G = nx.DiGraph()
        self.parent = parent

        self.tiles = [VersionedTile(tile=tile, version=0) for tile in tiles]
        self.vars = 

    def write_tile(self, tile: Tile) -> VersionedTile:
        for vtile in self.vtiles:
            if vtile.tile == tile:
                vtile.version += 1
                return vtile

    def read_tile(self, tile: Tile) -> VersionedTile:
        for vtile in self.vtiles:
            if vtile.tile == tile:
                return vtile

    def set_kernel_globals(self, kernel_globals: KernelGlobals):
        self.kernel_globals = kernel_globals

    def tma(self, loads: List[MemLoad]) -> List[VersionedTile]:
        # replace each of these instructions with versioned tile groups?

    def mma(self, a: Tile, b: Tile, out: Tile) -> VersionedTile:
        instruction = SeqStmt([
            
            
        ])
        versioned_tile_group = ...
        
        versioned_a = read_tile(a)
        versioned_b = read_tile(b)
        versioned_out_prev = read_tile(out)
        versioned_out_next = read_tile(out_next)

        G.add_edge(versioned_a, versioned_out_next, instruction) # replace each of these instructions with versioned tile groups?
        G.add_edge(versioned_b, versioned_out_next, instruction)
        G.add_edge(versioned_out_pred, versioned_out_next, instruction)

        return out
        
    def tile_op(self, op_name: str, parameters: List[Tile], output_var: Tile) -> VersionedTile:
        versioned_parameters = [read_tile(parameter) for parameter in parameters]
        versioned_out_prev = write_tile(output_var)
        instruction = OpCall(op_name, parameters, output_var)
        for vparam in versioned_parameters:
            G.add_edge(vparam, versioned_out, instruction) # out is one of the parameters
        

    def op(self, op_name: str, parameters: List[Var], output_var: Var) -> VersionedTile:
        ...

    def loop(self, size: Expr, body: SIR):
        ...
    
    def get_stmt(self):
        # produce a topological sort of the graph

        # where TMA values with a particular value are produced, and where they are consumed, you should have different values
            # when do you signal arrive(empty)? when every node of that version has already been used (you can just do TileGroup.arrive_empty())
            # when do you do wait full? the first time that somethign from that tilegroup is touched (you can just do TileGroup.wait_full())

        # what does this imply? every versioned tile requires:
            # a produce instruction -> will just be the operation that does the thing in most cases
            # a consume instruction -> will just be the variable name in most cases
            # and a done instruction -> will be nothing in most cases