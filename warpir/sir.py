from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
from .layouts import *
from .flow import *
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


@dataclass
class MMAState:
    nid:      str
    consumed: bool = False

    def consume(self) -> None:
        assert not self.consumed, "MMAState already consumed"
        self.consumed = True


class TMA:
    def expect(self, mgr: MultiTileLoadManager, loads: List[MemLoad]) -> Stmt:
        return mgr.async_load_global(loads)

    def wait(self, mgr: MultiTileLoadManager, level: Level = Level.warpgroup) -> Stmt:
        return mgr.wait_full(level)

    def prime(self, mgr: MultiTileLoadManager) -> Stmt:
        return mgr.arrive_empty()

    def release(self, mgr: MultiTileLoadManager) -> Stmt:
        return mgr.arrive_empty()


class MMA:
    def __init__(self):
        self._last: Optional[MMAState] = None

    def issue(self, scope: Scope, a: Tile, b: Tile, out: Tile) -> MMAState:
        if self._last and not self._last.consumed:
            scope._emit(OpCall("warpgroup::mma_async_wait", []).to_stmt())
            self._last.consume()
        scope._emit(OpCall("warpgroup::mma_AB", [out, a, b]).to_stmt())
        state = MMAState(nid=f"mma_{a.name}x{b.name}")
        self._last = state
        return state

    def sync(self, scope: Scope) -> None:
        assert self._last and not self._last.consumed, "no pending MMA to sync"
        scope._emit(OpCall("warpgroup::mma_async_wait", []).to_stmt())
        self._last.consume()


class Scope(Stmt):
    def __init__(self, sir: SIR):
        self._sir       = sir
        self._children: List[Stmt] = []

    def _emit(self, stmt: Stmt) -> Stmt:
        self._children.append(stmt)
        return stmt

    def __str__(self) -> str:
        return str(SeqStmt(self._children))

    def tma_load(self, loads: List[MemLoad]) -> Tuple[List[Tile], ConsumeScope]:
        tiles = [l.dest for l in loads]
        name  = "_".join(t.name for t in tiles)
        mgr   = MultiTileLoadManager(f"lm_{name}", tiles)
        self._sir._root_decls.append(mgr.initialize())
        self._sir._pre_loop.append(self._sir.tma.prime(mgr))
        self._emit(mgr.wait_empty(Level.warpgroup))
        self._emit(self._sir.tma.expect(mgr, loads))
        nid = f"tma_{name}"
        self._sir.G.add_node(nid, label=f"TMA\n{[t.name for t in tiles]}", color="lightblue")
        for t in tiles: self._sir.G.add_edge(nid, f"tile_{t.name}")
        return tiles, ConsumeScope(mgr, self._sir)

    def mma(self, a: Tile, b: Tile, out: Tile) -> MMAState:
        state = self._sir.mma.issue(self, a, b, out)
        prev  = self._sir.mma._last
        self._sir.G.add_node(state.nid, label=f"MMA\n{a.name}x{b.name}->{out.name}", color="orange")
        for t in [a, b]: self._sir.G.add_edge(f"tile_{t.name}", state.nid)
        self._sir.G.add_edge(state.nid, f"tile_{out.name}")
        return state

    def op(self, fn: str, *args) -> None:
        self._emit(OpCall(fn, list(args)).to_stmt())
        nid = f"op_{fn.split('::')[-1]}_{len(self._sir.G)}"
        self._sir.G.add_node(nid, label=fn.split("::")[-1], color="lightgrey")
        for a in args:
            if isinstance(a, Tile) and f"tile_{a.name}" in self._sir.G:
                self._sir.G.add_edge(f"tile_{a.name}", nid)


class ConsumeScope(Scope):
    def __init__(self, mgr: MultiTileLoadManager, sir: SIR):
        super().__init__(sir)
        self._mgr = mgr

    def __str__(self) -> str:
        return str(SeqStmt([
            self._sir.tma.wait(self._mgr),
            SeqStmt(self._children),
            self._sir.tma.release(self._mgr),
        ]))


class LoopScope(Scope):
    def __init__(self, var: Var, bound: Expr, sir: SIR):
        super().__init__(sir)
        self.i      = var
        self._bound = bound

    def __str__(self) -> str:
        return str(ForStmt(
            AssignExpr(self.i, RawExpr(0)),
            BinaryOp(self.i, self._bound, "<"),
            AssignExpr(self.i, BinaryOp(self.i, RawExpr(1), "+")),
            SeqStmt(self._children),
        ))


class SIR(Scope):
    def __init__(self, tma: TMA = None, mma: MMA = None):
        super().__init__(self)
        self.tma          = tma or TMA()
        self.mma          = mma or MMA()
        self._root_decls: List[Stmt] = []
        self._pre_loop:   List[Stmt] = []
        self.G            = nx.DiGraph()

    def tile(self, name: str, tile_type) -> Tile:
        t = Tile(name, tile_type)
        self._root_decls.append(t.var.declare())
        self.G.add_node(f"tile_{name}", label=f"{type(tile_type).__name__}\n{name}", color="lightyellow")
        return t

    def add_loop(self, bound: Expr) -> Tuple[Var, LoopScope]:
        loop = LoopScope(Var("i", ScalarType("int")), bound, self)
        for s in self._pre_loop: self._emit(s)
        self._pre_loop = []
        self._emit(loop)
        return loop.i, loop

    def add_store(self, tile: Tile, dst: Var, coord: Coord) -> None:
        self._emit(tile.warpgroup_store_global(dst, coord))
        nid = f"store_{tile.name}"
        self.G.add_node(nid, label=f"store\n{tile.name}", color="lightgrey")
        self.G.add_edge(f"tile_{tile.name}", nid)

    def emit(self) -> Stmt:
        return SeqStmt(self._root_decls + self._children)

    def save_graph(self, path: str = "outputs/sir_graph.png") -> None:
        import os; os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        G     = self.G
        order = list(nx.topological_sort(G))
        depth: dict = {}
        for n in order:
            depth[n] = max((depth[p]+1 for p in G.predecessors(n)), default=0)
        by_layer: dict = defaultdict(list)
        for n, d in depth.items(): by_layer[d].append(n)
        pos = {n: (d*280, -i*150) for d, ns in by_layer.items() for i, n in enumerate(ns)}
        fig, ax = plt.subplots(figsize=(max(14, len(by_layer)*3), 8))
        ax.axis("off")
        nx.draw_networkx(G, pos, ax=ax, arrows=True, font_size=8, node_size=2200,
                         node_color=[G.nodes[n].get("color", "white") for n in G.nodes],
                         labels={n: G.nodes[n].get("label", n) for n in G.nodes},
                         arrowsize=20, width=1.5, connectionstyle="arc3,rad=0.08")
        plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
        print(f"saved -> {path}")