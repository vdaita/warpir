from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# --- Core IR for scheduling ---

@dataclass(frozen=True)
class Op:
    name: str
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    kind: str
    # Optional metadata to guide cost model and constraints
    latency_cycles: Optional[int] = None
    smem_bytes: int = 0
    regs: int = 0
    pipe: Optional[str] = None


@dataclass
class CostModel:
    # Conservative Hopper-ish defaults; replace with real estimates.
    default_latency: int = 4
    smem_cycle_per_byte: float = 0.02
    pipe_latency: Dict[str, int] = field(default_factory=dict)

    def op_latency(self, op: Op) -> int:
        if op.latency_cycles is not None:
            return op.latency_cycles
        if op.pipe and op.pipe in self.pipe_latency:
            return self.pipe_latency[op.pipe]
        return self.default_latency

    def op_smem_cost(self, op: Op) -> int:
        return int(op.smem_bytes * self.smem_cycle_per_byte)


@dataclass
class SchedulerConfig:
    num_warpgroups: int = 2
    max_schedule_len: int = 10_000
    # Placeholder for extra constraints (bandwidth, regs, etc.)
    max_regs: Optional[int] = None
    max_smem_per_cycle: Optional[int] = None


@dataclass
class DependencyGraph:
    # Edges: u -> v (u must finish before v starts)
    edges: Dict[int, List[int]]
    # Mapping from op index to list of preds for convenience
    preds: Dict[int, List[int]]


@dataclass
class WarpSchedule:
    ops: List[Op]
    start_times: Dict[int, int]
    warpgroups: Dict[int, int]


# --- Graph construction ---

def build_dependency_graph(ops: Sequence[Op]) -> DependencyGraph:
    producers: Dict[str, int] = {}
    edges: Dict[int, List[int]] = {i: [] for i in range(len(ops))}
    preds: Dict[int, List[int]] = {i: [] for i in range(len(ops))}
    for i, op in enumerate(ops):
        for inp in op.inputs:
            if inp in producers:
                u = producers[inp]
                edges[u].append(i)
                preds[i].append(u)
        for out in op.outputs:
            producers[out] = i
    return DependencyGraph(edges=edges, preds=preds)


# --- Partitioning ---

def split_warpgroups_round_robin(ops: Sequence[Op], num_warpgroups: int) -> Dict[int, int]:
    assignment: Dict[int, int] = {}
    for i in range(len(ops)):
        assignment[i] = i % max(1, num_warpgroups)
    return assignment


# --- CP-SAT scheduling ---

def _import_cp_sat():
    try:
        from ortools.sat.python import cp_model  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError("ortools is required for CP-SAT scheduling") from exc
    return cp_model


def solve_warp_schedules(
    ops: Sequence[Op],
    cfg: SchedulerConfig,
    cost: Optional[CostModel] = None,
    initial_assignment: Optional[Dict[int, int]] = None,
) -> WarpSchedule:
    cost = cost or CostModel()
    dep = build_dependency_graph(ops)
    cp_model = _import_cp_sat()

    model = cp_model.CpModel()
    horizon = cfg.max_schedule_len

    # Decision vars
    start = [model.NewIntVar(0, horizon, f"t_{i}") for i in range(len(ops))]
    dur = [cost.op_latency(op) + cost.op_smem_cost(op) for op in ops]

    # Warpgroup assignment
    if initial_assignment is None:
        wg = [
            model.NewIntVar(0, cfg.num_warpgroups - 1, f"wg_{i}")
            for i in range(len(ops))
        ]
    else:
        wg = [model.NewIntVar(0, cfg.num_warpgroups - 1, f"wg_{i}") for i in range(len(ops))]
        for i, g in initial_assignment.items():
            model.Add(wg[i] == g)

    # Dependency constraints
    for u, vs in dep.edges.items():
        for v in vs:
            model.Add(start[v] >= start[u] + dur[u])

    # Optional resource constraints
    # Placeholder: limit total smem per cycle or registers per warpgroup.
    # Extend here with cumulative constraints if needed.

    # Objective: minimize makespan
    makespan = model.NewIntVar(0, horizon, "makespan")
    for i in range(len(ops)):
        model.Add(makespan >= start[i] + dur[i])
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible schedule found")

    start_times = {i: solver.Value(start[i]) for i in range(len(ops))}
    warpgroups = {i: solver.Value(wg[i]) for i in range(len(ops))}
    return WarpSchedule(ops=list(ops), start_times=start_times, warpgroups=warpgroups)
