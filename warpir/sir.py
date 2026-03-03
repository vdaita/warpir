from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from .layouts import *
from .flow import *


# ── graph nodes (for visualization only) ─────────────────────────────────────

@dataclass
class GraphNode:
    id:    str
    label: str
    color: str
    edges: List[str] = field(default_factory=list)


# ── async handles ─────────────────────────────────────────────────────────────

@dataclass
class TMAGroup:
    """Handle returned by TMA.load — represents in-flight tiles."""
    tiles:    List[Tile]
    manager:  MultiTileLoadManager
    node_id:  str
    consumed: bool = False


@dataclass
class MMAResult:
    """Handle returned by MMA.produce — represents an in-flight accumulation."""
    tile:       Tile
    produce_id: str
    consume_id: str
    consumed:   bool = False


# ── async units ───────────────────────────────────────────────────────────────

class TMA:
    def __init__(self, sir: SIR):
        self._sir     = sir
        self._history: List[Tuple[TMAGroup, Optional[List[Tile]]]] = []

    def load(self, loads: List[MemLoad], num_consumers: int = 1) -> TMAGroup:
        """
        Primes empty semaphore on first use, then:
        wait(empty) -> expect_bytes -> load_async.
        """
        tiles = [load.dest for load in loads]
        mgr   = MultiTileLoadManager(self._sir._fresh("lm"), tiles, num_consumers)

        self._sir._decls.append(mgr.initialize())
        self._sir._managers.append(mgr)

        # prime empty semaphore once before the loop so first load can proceed
        self._sir._init_stmts.append(mgr.arrive_empty())

        # wait for buffer to be empty before loading into it
        self._sir._stmts.append(mgr.wait_empty(Level.warpgroup))

        # fire the async loads
        self._sir._stmts.append(mgr.async_load_global(loads))

        # graph
        tile_names = ", ".join(t.name for t in tiles)
        node_id    = self._sir._fresh("tma")
        self._sir._graph[node_id] = GraphNode(
            id    = node_id,
            label = f"TMA.load\\n[{tile_names}]",
            color = "lightblue",
        )
        for t in tiles:
            t_id = f"tile_{t.name}"
            if t_id not in self._sir._graph:
                self._sir._graph[t_id] = GraphNode(
                    id    = t_id,
                    label = f"shared\\n{t.name}",
                    color = "lightyellow",
                )
            self._sir._graph[node_id].edges.append(t_id)

        group = TMAGroup(tiles, mgr, node_id)
        for t in tiles:
            self._sir._pending[id(t)] = group
        self._history.append((group, None))
        return group

    def consume(self, group: TMAGroup, level: Level = Level.warpgroup) -> List[Tile]:
        """Emit wait_full — tiles are now ready for use."""
        assert not group.consumed, "TMAGroup already consumed"
        self._sir._stmts.append(group.manager.wait_full(level))
        group.consumed = True
        for i, (g, _) in enumerate(self._history):
            if g is group:
                self._history[i] = (g, group.tiles)
                break
        return group.tiles

    def arrive_empty(self, group: TMAGroup) -> None:
        """Signal that consumer is done — buffer can be reused by producer."""
        self._sir._stmts.append(group.manager.arrive_empty())


class MMA:
    def __init__(self, sir: SIR):
        self._sir     = sir
        self._last:    Optional[MMAResult] = None
        self._history: List[Tuple[MMAResult, Optional[MMAResult]]] = []

    def produce(self, a: Tile, b: Tile, out: Tile,
                release: Optional[List[TMAGroup]] = None) -> MMAResult:
        """
        Emit mma_AB + async_wait.
        release: TMAGroups to arrive_empty after MMA completes,
                 releasing those buffers back to the producer.
        """
        assert self._last is None or self._last.consumed, \
            "previous MMAResult must be consumed before producing again"

        self._sir._stmts.append(OpCall("warpgroup::mma_AB", [out, a, b]).to_stmt())
        self._sir._stmts.append(OpCall("warpgroup::mma_async_wait", []).to_stmt())

        if release:
            for grp in release:
                self._sir._stmts.append(grp.manager.arrive_empty())

        # graph
        prod_id = self._sir._fresh("mma_prod")
        cons_id = self._sir._fresh("mma_cons")
        acc_id  = f"tile_{out.name}"

        self._sir._graph[prod_id] = GraphNode(
            id    = prod_id,
            label = f"MMA.produce\\n→ {out.name}",
            color = "orange",
        )
        self._sir._graph[cons_id] = GraphNode(
            id    = cons_id,
            label = f"MMA.consume\\n← {out.name}",
            color = "lightgreen",
        )

        # input tiles -> produce
        for t in [a, b]:
            t_id = f"tile_{t.name}"
            if t_id in self._sir._graph:
                self._sir._graph[t_id].edges.append(prod_id)

        # produce -> accum tile -> consume
        if acc_id not in self._sir._graph:
            self._sir._graph[acc_id] = GraphNode(
                id    = acc_id,
                label = f"accum\\n{out.name}",
                color = "lightyellow",
            )
        self._sir._graph[prod_id].edges.append(acc_id)
        self._sir._graph[acc_id].edges.append(cons_id)

        # previous consume -> this produce (serialization dependency)
        if self._last is not None:
            self._sir._graph[self._last.consume_id].edges.append(prod_id)

        result = MMAResult(out, prod_id, cons_id)
        self._last = result
        self._sir._pending[id(out)] = result
        self._history.append((result, None))
        return result

    def consume(self, result: MMAResult) -> Tile:
        """Returns the accumulated tile — marks result as consumed."""
        assert not result.consumed, "MMAResult already consumed"
        result.consumed = True
        for i, (prod, _) in enumerate(self._history):
            if prod is result:
                self._history[i] = (prod, result)
                break
        return result.tile


# ── SIR ───────────────────────────────────────────────────────────────────────

class SIR:
    def __init__(self):
        self._stmts:      List[Stmt] = []
        self._decls:      List[Stmt] = []
        self._init_stmts: List[Stmt] = []
        self._counter:    int        = 0
        self._managers:   List[MultiTileLoadManager] = []
        self._pending:    dict       = {}
        self._graph:      dict[str, GraphNode] = {}
        self._loop_nodes: List[str]  = []

        self.tma = TMA(self)
        self.mma = MMA(self)

    def _fresh(self, prefix="t") -> str:
        self._counter += 1
        return f"{prefix}{self._counter}"

    def tile(self, tile_type) -> Tile:
        t    = Tile(self._fresh("t"), tile_type)
        t_id = f"tile_{t.name}"
        self._decls.append(t.var.declare())
        self._graph[t_id] = GraphNode(
            id    = t_id,
            label = f"{type(tile_type).__name__}\\n{t.name}",
            color = "lightyellow",
        )
        return t

    def op(self, fn: str, *args) -> None:
        """Pass-through for all synchronous ops."""
        resolved = [a.tile if isinstance(a, MMAResult) else a for a in args]
        self._stmts.append(OpCall(fn, list(resolved)).to_stmt())

        op_id = self._fresh("op")
        short = fn.split("::")[-1]
        self._graph[op_id] = GraphNode(
            id    = op_id,
            label = f"{short}",
            color = "lightgrey",
        )
        for a in args:
            if isinstance(a, MMAResult):
                self._graph[a.consume_id].edges.append(op_id)
            elif isinstance(a, Tile):
                t_id = f"tile_{a.name}"
                if t_id in self._graph:
                    self._graph[t_id].edges.append(op_id)

    def loop(self, var: Var, bound: Expr, body_fn) -> None:
        """Emit a for loop. body_fn receives a child SIR sharing state."""
        inner = SIR()
        inner._counter    = self._counter
        inner._managers   = self._managers
        inner._pending    = self._pending
        inner._graph      = self._graph
        inner._init_stmts = self._init_stmts
        inner.mma._last    = self.mma._last
        inner.mma._history = self.mma._history
        inner.tma._history = self.tma._history

        before = set(self._graph.keys())
        body_fn(inner)
        after  = set(inner._graph.keys())
        self._loop_nodes = list(after - before)

        self._counter  = inner._counter
        self.mma._last = inner.mma._last
        self._decls.extend(inner._decls)

        # emit init stmts (e.g. arrive_empty priming) before the loop
        self._stmts.extend(self._init_stmts)
        self._init_stmts = []

        self._stmts.append(ForStmt(
            AssignExpr(var, RawExpr(0)),
            BinaryOp(var, bound, "<"),
            AssignExpr(var, BinaryOp(var, RawExpr(1), "+")),
            SeqStmt(inner._stmts)
        ))

    def emit(self) -> Stmt:
        return SeqStmt(self._decls + self._init_stmts + self._stmts)