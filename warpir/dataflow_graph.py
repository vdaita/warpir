from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from .layouts import *
from .flow import *
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

@dataclass(frozen=True)
class VersionedTile:
    tile: Tile
    version: int
    context: DataflowGraph

class Op(ABC):
    @abstractmethod
    def produce(self) -> Stmt:
        pass

    @abstractmethod
    def consume(self, stmt: Stmt) -> Stmt:
        pass

    @abstractmethod
    def done(self) -> Stmt:
        pass

class NormalOp(Op):
    def __init__(self, produce_stmt: Stmt):
        self.produce_stmt = produce_stmt
        self.was_produced = False

    def produce(self):
        if self.was_produced:
            return NoStmt()
        self.was_produced = True
        return self.produce_stmt

    def consume(self, stmt: Stmt):
        return stmt

    def done(self):
        return NoStmt()

# class AwaitedOp(Op):
#     def __init__(self, normal_Op: Op, DataflowGraph: DataflowGraph):
#         self.DataflowGraph.declarations.append(
#             # add in the semaphore and tic/toc for this variable
#         )

class TMAOp(Op):
    def __init__(self, group: TileGroup, loads: List[MemLoad]):
        self.group = group
        self.loads = loads

        self.was_produced = False
        self.was_consumed = False
        self.was_done = False

    def produce(self) -> Stmt:
        if self.was_produced:
            return NoStmt()
        
        self.was_produced = True
        return SeqStmt([
            self.group.wait_empty(Level.warpgroup),
            self.group.async_load_global(self.loads)
        ])
    
    def consume(self, stmt: Stmt) -> Stmt:
        if self.was_consumed:
            return stmt
        self.was_consumed = True
        return SeqStmt([
            self.group.wait_full(Level.warpgroup),
            stmt
        ])
    
    def done(self) -> Stmt:
        if self.was_done:
            return NoStmt()
        self.was_done = True
        return self.group.arrive_empty()

class MMAOp(Op):
    def __init__(self, a: Tile, b: Tile, out: Tile):
        self.a, self.b, self.out = a, b, out
        
        self.was_produced = False
        self.was_consumed = False

    def produce(self) -> Stmt:
        if self.was_produced:
            return NoStmt()
        return OpCall("warpgroup::mma_AB", [self.out, self.a, self.b], to_stmt())
    
    def consume(self, stmt: Stmt) -> Stmt:
        if self.was_consumed:
            return NoStmt()
        return SeqStmt([
            OpCall("warpgroup::mma_AB", [self.out, self.a, self.b]).to_stmt(),
            stmt
        ])

    def done(self) -> Stmt:
        return NoStmt()

class ForLoopOp(Op):
    def __init__(self, name: str, bound: Expr, body: DataflowGraph):
        self.name = name
        self.bound = bound
        self.body = body
    
    def produce(self) -> Stmt:
        i = Var("i", ScalarType("int"))
        return ForStmt(
            AssignExpr(i, zero),
            BinaryOp(i, self.bound, "<"),
            AssignExpr(i, BinaryOp(i, one, "+")),
            self.body.get_stmt()
        )

    def consume(self, stmt: Stmt) -> Stmt:
        return stmt

    def done(self) -> Stmt:
        return NoStmt()

class DataflowGraph:
    def __init__(self, tiles: List[Tile], kernel_vars: List[Var], parent: DataflowGraph = None):
        self.kernel_globals: KernelGlobals = KernelGlobals("globals")        
        self.G = nx.DiGraph()
        self.parent = parent

        self.vtiles = [VersionedTile(tile=tile, version=0, context=self) for tile in tiles]
        self.vars = kernel_vars

        self.tile_groups = []

    def write_tile(self, tile: Tile) -> VersionedTile:
        for i, vtile in enumerate(self.vtiles):
            if vtile.tile == tile:
                new_vtile = VersionedTile(tile=vtile.tile, version=vtile.version + 1, context=vtile.context)
                self.vtiles[i] = new_vtile
                return new_vtile

    def read_tile(self, tile: Tile) -> VersionedTile:
        for vtile in self.vtiles:
            if vtile.tile == tile:
                return vtile

    def set_kernel_globals(self, kernel_globals: KernelGlobals):
        self.kernel_globals = kernel_globals

    def tma(self, loads: List[MemLoad]) -> List[VersionedTile]:
        joint_name = "_".join(sorted(load.dest.name for load in loads))
        tile_group = TileGroup(joint_name, [load.dest for load in loads])
        if not tile_group in self.tile_groups:
            self.tile_groups.append(tile_group)

        Op = TMAOp(tile_group, loads)
        versioned_outputs = [self.write_tile(load.dest) for load in loads]
        for versioned_output in versioned_outputs:
            self.G.add_Op(versioned_output, Op)
        return [load.dest for load in loads]

    def mma(self, a: Tile, b: Tile, out: Tile) -> Tile:
        versioned_a = self.read_tile(a)
        versioned_b = self.read_tile(b)
        versioned_out_prev = self.read_tile(out)
        versioned_out_next = self.write_tile(out)

        Op = MMAOp(a, b, out)
        for versioned_input in [versioned_a, versioned_b, versioned_out_prev]:
            self.G.add_Op(versioned_input, Op)
        self.G.add_Op(Op, versioned_out_next)
        return out
        
    def tile_op(self, op_name: str, parameters: List[Tile], output_var: Tile, input_vars: Optional[List[Tile]] = None) -> VersionedTile:
        if not input_vars:
            input_vars = parameters

        versioned_inputs = [(self.read_tile(iparam) if (type(iparam) == Tile) else iparam) for iparam in input_vars]

        versioned_out = self.write_tile(output_var)

        Op = NormalOp(
            OpCall(op_name, parameters).to_stmt()
        )

        for vparam in versioned_inputs:
            if type(vparam) == VersionedTile:
                self.G.add_Op(vparam, Op)
        self.G.add_Op(Op, versioned_out)

        return output_var
        
    def inputs(self) -> List[VersionedTile]:
        return [
            self.G.nodes[nid]["vtile"]
            for nid in self.G.nodes
            if "vtile" in self.G.nodes[nid]
            and self.G.in_degree(nid) == 0
        ]

    def outputs(self) -> List[VersionedTile]:
        return [
            self.G.nodes[nid]["vtile"]
            for nid in self.G.nodes
            if "vtile" in self.G.nodes[nid]
            and self.G.out_degree(nid) == 0
        ]

    def loop(self, iterator_name: str, bound: Expr, body: DataflowGraph) -> List[VersionedTile]:
        body_inputs  = body.inputs()
        body_outputs = body.outputs()

        loop_Op = ForLoopOp(iterator_name, bound, body)
        loop_nid  = f"loop_{id(loop_Op)}"
        self.G.add_node(loop_nid, Op=loop_Op)

        # wire parent's current tile versions into the loop
        for vt in body_inputs:
            parent_vt = self._read(vt.tile)
            if parent_vt.nid not in self.G:
                self.G.add_node(parent_vt.nid, vtile=parent_vt)
            self.G.add_Op(parent_vt.nid, loop_nid)

        # wire loop outputs to new versions in parent
        written = []
        for vt in body_outputs:
            _, nxt = self._write(vt.tile)
            self.G.add_node(nxt.nid, vtile=nxt)
            self.G.add_Op(loop_nid, nxt.nid)
            written.append(nxt)

        return written

    def get_stmt(self) -> Stmt:
        stmts: List[Stmt] = [vt.tile.var.declare() 
                            for vt in self.vtiles]
        stmts += [tg.initialize() for tg in self.tile_groups]
        seen:  set = set()

        # track how many consumers each VersionedTile has
        consumer_counts: dict[str, int] = {
            nid: self.G.out_degree(nid)
            for nid in self.G.nodes
            if "vtile" in self.G.nodes[nid]
        }
        consumed_so_far: dict[str, int] = defaultdict(int)

        def emit(s: Stmt):
            k = str(s).strip()
            if k and k not in seen:
                stmts.append(s); seen.add(k)

        for nid in nx.topological_sort(self.G):
            node = self.G.nodes[nid]
            if "Op" not in node:
                continue
            Op = node["Op"]

            # produce: fire the op
            emit(Op.produce())

            # consume: for each input VersionedTile to this Op,
            # emit the consume stmt and track done
            for pred_nid in self.G.predecessors(nid):
                pred_node = self.G.nodes[pred_nid]
                if "vtile" not in pred_node:
                    continue
                pred_vt = pred_node["vtile"]
                emit(Op.consume(NoStmt()))
                consumed_so_far[pred_nid] += 1
                if consumed_so_far[pred_nid] == consumer_counts[pred_nid]:
                    emit(Op.done())

        return SeqStmt(stmts)
        # find some way to feed in "i" as a variable to the body
        # the version of the elements on this need to be dependent on the variable "i" so that when i pipeline this, I can run a body creation method with i and i + 1 and get two separate versions
        # then, once i produce those two versions, i can figure out how to color them in a way that properly manages the dependencies

    # def get_stmt(self):
        ...
        # produce a topological sort of the graph
            # there needs to be an indicator that states that you must wait for a variable to be "freed" (regardless of version) before you move on to the next version
        
        # add all of the new definitions that are being made in this staement
        # create a topological sort of the instructions
        # write a counter for each versioned tile of how many instructions feed into it (for keeping track of done)
        
        # for each 
            # then, produce a list of Ops which represent what to do
            # for each operation that is being performed, consume the incoming Ops and produce the outgoing Op
            # if this is the last consumer of this verison of the tile, mark done
                # mark this version as done, and so you can traverse the new Ops and add it to your topological sort to make sure that you aren't overwriting values in the middle
    

        # where TMA values with a particular value are produced, and where they are consumed, you should have different values
            # when do you signal arrive(empty)? when every node of that version has already been used (you can just do TileGroup.arrive_empty())
            # when do you do wait full? the first time that somethign from that tilegroup is touched (you can just do TileGroup.wait_full())

        # what does this imply? every versioned tile requires:
            # a produce instruction -> will just be the operation that does the thing in most cases
            # a consume instruction -> will just be the variable name in most cases
            # and a done instruction -> will be nothing in most cases