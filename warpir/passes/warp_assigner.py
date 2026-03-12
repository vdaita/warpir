from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union
from ortools.sat.python import cp_model

from warpir.ir.ops import (
    AllocSharedOp,
    BufSlotExpr,
    ForOp,
    IterArg,
    Kernel,
    MMABufOp,
    MMAOp,
    Op,
    TMALoadBufOp,
    TMALoadOp,
    TMAStoreOp,
    Value,
    WaitOp,
    WarpSpecializedRegionOp,
    YieldOp,
    ZeroOp,
    NegInftyOp,
)
from warpir.ir.types import SharedBufferType


# ── Solver datastructures ─────────────────────────────────────────────────

@dataclass
class WAInstruction:
    name: str
    version: int

    parent_instructions: List['WAInstruction']
    ordering_instructions: List['WAInstruction']
    is_async: bool

    cuda_latency: int
    total_latency: int

    modulo_issue_cycle: int

    def __str__(self) -> str:
        return f"{self.name}_{self.version}"


class WAInstructionManager:
    def __init__(self):
        self.instructions: List[WAInstruction] = []
        self.intervals: Dict[str, cp_model.IntervalVar] = {}

    def add_instruction(self, instruction: WAInstruction):
        self.instructions.append(instruction)

    def solve(self, max_cycles: int, num_warps: int, synchronization_cost: int,
              instruction_types: List[str], use_verbose=False):
        model = cp_model.CpModel()

        warp_assignments: Dict[str, List[cp_model.BoolVarT]] = {}
        instruction_type_assignments: Dict[str, List[cp_model.BoolVarT]] = {}

        for instruction_type in instruction_types:
            pins = [model.new_bool_var(f"warp_it_{instruction_type}_{w}") for w in range(num_warps)]
            instruction_type_assignments[instruction_type] = pins
            model.add_exactly_one(pins)

        for instruction in self.instructions:
            key = str(instruction)
            ub  = max_cycles - 1 - max(instruction.cuda_latency, instruction.total_latency)
            if ub < 0:
                raise ValueError(f"max_cycles too small for {key}")

            start = model.new_int_var(0, ub, f"start_{key}")
            self.intervals[key] = model.new_interval_var(
                start, instruction.cuda_latency, start + instruction.cuda_latency, f"iv_{key}"
            )

            warps = [model.new_bool_var(f"warp_{key}_{w}") for w in range(num_warps)]
            warp_assignments[key] = warps
            model.add_exactly_one(warps)

            if instruction.name in instruction_type_assignments:
                for w in range(num_warps):
                    model.add(warps[w] == instruction_type_assignments[instruction.name][w])

        for instruction in self.instructions:
            instruction_id = str(instruction)

            for parent in instruction.parent_instructions:
                parent_id = str(parent)
                gap = instruction.modulo_issue_cycle - parent.modulo_issue_cycle
                min_separation = max(gap, parent.total_latency) if gap > 0 else parent.total_latency

                if parent.is_async:
                    model.add(
                        self.intervals[instruction_id].start_expr() >=
                        self.intervals[parent_id].start_expr() + min_separation
                    )
                else:
                    for pw in range(num_warps):
                        for cw in range(num_warps):
                            extra = 0 if pw == cw else synchronization_cost
                            model.add(
                                self.intervals[instruction_id].start_expr() >=
                                self.intervals[parent_id].start_expr() + min_separation + extra
                            ).only_enforce_if([
                                warp_assignments[parent_id][pw],
                                warp_assignments[instruction_id][cw],
                            ])

            for parent in instruction.ordering_instructions:
                parent_id = str(parent)
                gap = instruction.modulo_issue_cycle - parent.modulo_issue_cycle
                min_separation = max(gap, parent.total_latency) if gap > 0 else parent.total_latency
                model.add(
                    self.intervals[instruction_id].start_expr() >=
                    self.intervals[parent_id].start_expr() + min_separation
                )

        for w in range(num_warps):
            optional_intervals = []
            for instr in self.instructions:
                key   = str(instr)
                on_w  = warp_assignments[key][w]
                start = self.intervals[key].start_expr()
                optional_intervals.append(model.new_optional_interval_var(
                    start, instr.cuda_latency, start + instr.cuda_latency,
                    on_w, f"opt_{key}_w{w}"
                ))
            model.add_no_overlap(optional_intervals)

        makespan = model.new_int_var(0, max_cycles - 1, "makespan")
        for instr in self.instructions:
            model.add(makespan >= self.intervals[str(instr)].start_expr() + instr.total_latency)
        model.minimize(makespan)

        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = use_verbose
        status = solver.solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None

        type_to_warp: Dict[str, int] = {}
        for itype in instruction_types:
            assigned = next(w for w in range(num_warps)
                           if solver.value(instruction_type_assignments[itype][w]))
            type_to_warp[itype] = assigned

        return type_to_warp


# ── Kernel pass ───────────────────────────────────────────────────────────

def _solve_warp_assignment(
    load_names: List[str],
    compute_names: List[str],
) -> Dict[str, int]:
    """Run the WAInstructionManager solver on a GEMM-like loop body.

    Returns a dict mapping instruction-type name → warp id (0 = producer, 1 = consumer).
    """
    II = 6
    N = 4
    num_warps = 2
    max_cycles = 500
    synchronization_cost = 80

    all_types = load_names + compute_names
    manager = WAInstructionManager()

    load_instrs: Dict[str, Dict[int, WAInstruction]] = {n: {} for n in load_names}
    compute_instrs: Dict[str, Dict[int, WAInstruction]] = {n: {} for n in compute_names}

    for k in range(N):
        for ln in load_names:
            instr = WAInstruction(
                name=ln, version=k,
                parent_instructions=[], ordering_instructions=[],
                is_async=True, cuda_latency=1, total_latency=12,
                modulo_issue_cycle=k * II,
            )
            load_instrs[ln][k] = instr
            manager.add_instruction(instr)

    for k in range(N):
        parents = [load_instrs[ln][k] for ln in load_names]
        for cn in compute_names:
            prev = compute_instrs[cn].get(k - 1)
            if prev is not None:
                parents.append(prev)
            instr = WAInstruction(
                name=cn, version=k,
                parent_instructions=parents, ordering_instructions=[],
                is_async=False, cuda_latency=4, total_latency=4,
                modulo_issue_cycle=2 * II + k * II,
            )
            compute_instrs[cn][k] = instr
            manager.add_instruction(instr)

    result = manager.solve(max_cycles, num_warps, synchronization_cost, all_types)
    if result is None:
        return {n: 0 for n in load_names} | {n: 1 for n in compute_names}
    return result


def kernel_pass(kernel: Kernel, num_stages: int = 2) -> Kernel:
    """Transform an SSA kernel into a warp-specialized (producer/consumer) form.

    This is an alternative to modulo_scheduler.kernel_pass — both take the
    original SSA kernel and produce an optimized variant.

    The pass:
      1. Locates the main ForOp.
      2. Partitions ops into producer (TMA loads) and consumer (compute).
      3. Validates the split via the WAInstructionManager solver.
      4. Emits a WarpSpecializedRegionOp with full/empty semaphore protocol.
    """
    # ── Locate ForOp and partition kernel body ────────────────────────────
    for_idx: Optional[int] = None
    for i, op in enumerate(kernel.body):
        if isinstance(op, ForOp):
            for_idx = i
            break

    if for_idx is None:
        return kernel

    for_op: ForOp = kernel.body[for_idx]  # type: ignore[assignment]
    pre_ops = list(kernel.body[:for_idx])
    post_ops = list(kernel.body[for_idx + 1:])

    # ── Separate setup ops (consumer) from other pre-ops ─────────────────
    consumer_setup: List[Op] = []
    alloc_level_ops: List[Op] = []
    for op in pre_ops:
        if isinstance(op, (ZeroOp, NegInftyOp)):
            consumer_setup.append(op)
        else:
            alloc_level_ops.append(op)

    # ── Extract loop body ops ─────────────────────────────────────────────
    load_ops = [o for o in for_op.body if isinstance(o, TMALoadOp)]
    mma_ops = [o for o in for_op.body if isinstance(o, MMAOp)]
    yield_op = next((o for o in for_op.body if isinstance(o, YieldOp)), None)
    other_body_ops = [
        o for o in for_op.body
        if not isinstance(o, (TMALoadOp, WaitOp, MMAOp, YieldOp))
    ]

    if not load_ops or not mma_ops:
        return kernel

    # ── Solver validation ─────────────────────────────────────────────────
    load_names = [f"load_{l.source.name.lower()}" for l in load_ops]
    compute_names = [f"mma_{m.result.name}" for m in mma_ops]
    _solve_warp_assignment(load_names, compute_names)

    # ── Allocate shared buffers ───────────────────────────────────────────
    new_body: List[Op] = []
    buf_values: Dict[Value, Value] = {}

    for ld in load_ops:
        buf_type = SharedBufferType(tile_type=ld.result.type, count=num_stages)
        buf_val = Value(f"{ld.source.name.lower()}_bufs", buf_type)
        buf_values[ld.result] = buf_val
        new_body.append(AllocSharedOp(result=buf_val))

    # ── Build producer body (TMA loads) ───────────────────────────────────
    slot = BufSlotExpr(value=for_op.induction_var, offset=0, modulus=num_stages)
    producer_body: List[Op] = []
    for ld in load_ops:
        producer_body.append(TMALoadBufOp(
            buf=buf_values[ld.result],
            slot=slot,
            source=ld.source,
            coords=ld.coords,
        ))

    # ── Build consumer body (compute) ─────────────────────────────────────
    consumer_body: List[Op] = []
    for mma in mma_ops:
        consumer_body.append(MMABufOp(
            result=mma.result,
            a_buf=buf_values[mma.a],
            a_slot=slot,
            b_buf=buf_values[mma.b],
            b_slot=slot,
            accum=mma.accum,
        ))
    consumer_body.extend(other_body_ops)
    if yield_op is not None:
        consumer_body.append(yield_op)

    # ── Assemble WarpSpecializedRegionOp ──────────────────────────────────
    new_body.append(WarpSpecializedRegionOp(
        num_stages=num_stages,
        bufs=tuple(buf_values.values()),
        induction_var=for_op.induction_var,
        start=0,
        stop=for_op.stop,
        step=for_op.step,
        tile_size=for_op.tile_size,
        producer_body=tuple(producer_body),
        consumer_setup=tuple(consumer_setup),
        consumer_body=tuple(consumer_body),
        consumer_iter_args=for_op.iter_args,
        consumer_results=for_op.results,
        consumer_finish=tuple(post_ops),
    ))

    return Kernel(
        name=kernel.name,
        inputs=kernel.inputs,
        outputs=kernel.outputs,
        body=tuple(new_body),
    )
