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

class VersionedAwaitedEdge(VersionedEdge):
    def __init__(self, versioned_edge: VersionedEdge):
        ...
    
    # Basically, take a standard vanilla versioned edge and add semaphores and waiting and what not
    # This will be used to wrap an edge if we want to move it cross-block

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
        self.vars = ...

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
        # have some way for people to say that a given operation doesn't depend on the previuous version
        # just add it to the graph with only a producer
        ...

    def loop(self, size: Expr, body: SIR):
        # find some way to feed in "i" as a variable to the body
        # the version of the elements on this need to be dependent on the variable "i" so that when i pipeline this, I can run a body creation method with i and i + 1 and get two separate versions
        # then, once i produce those two versions, i can figure out how to color them in a way that properly manages the dependencies

    def get_stmt(self):
        ...
        # produce a topological sort of the graph
            # there needs to be an indicator that states that you must wait for a variable to be "freed" (regardless of version) before you move on to the next version
        
        # add all of the new definitions that are being made in this staement
        # create a topological sort of the instructions
        # write a counter for each versioned tile of how many instructions feed into it (for keeping track of done)
        
        # for each 
            # then, produce a list of edges which represent what to do
            # for each operation that is being performed, consume the incoming edges and produce the outgoing edge
            # if this is the last consumer of this verison of the tile, mark done
                # mark this version as done, and so you can traverse the new edges and add it to your topological sort to make sure that you aren't overwriting values in the middle
    

        # where TMA values with a particular value are produced, and where they are consumed, you should have different values
            # when do you signal arrive(empty)? when every node of that version has already been used (you can just do TileGroup.arrive_empty())
            # when do you do wait full? the first time that somethign from that tilegroup is touched (you can just do TileGroup.wait_full())

        # what does this imply? every versioned tile requires:
            # a produce instruction -> will just be the operation that does the thing in most cases
            # a consume instruction -> will just be the variable name in most cases
            # and a done instruction -> will be nothing in most cases
    
