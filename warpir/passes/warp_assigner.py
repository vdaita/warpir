from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from ortools.sat.python import cp_model
from rich import print

@dataclass
class WAInstruction:
    name: str
    version: int

    parent_instructions: List['WAInstruction']   # data deps — pay sync cost if cross-warp
    ordering_instructions: List['WAInstruction'] # ordering deps — no sync cost, just sequencing
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

    def solve(self, max_cycles: int, num_warps: int, synchronization_cost: int, instruction_types: List[str], use_verbose=False):
        model = cp_model.CpModel()

        warp_assignments: Dict[str, List[cp_model.BoolVarT]] = {}
        instruction_type_assignments: Dict[str, List[cp_model.BoolVarT]] = {}

        # type-level warp pins
        for instruction_type in instruction_types:
            pins = [model.new_bool_var(f"warp_it_{instruction_type}_{w}") for w in range(num_warps)]
            instruction_type_assignments[instruction_type] = pins
            model.add_exactly_one(pins)

        # per-instruction vars
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

            # ── data deps: pay sync cost if cross-warp ─────────────────
            for parent in instruction.parent_instructions:
                parent_id = str(parent)
                gap = instruction.modulo_issue_cycle - parent.modulo_issue_cycle
                min_separation = max(gap, parent.total_latency) if gap > 0 else parent.total_latency

                if parent.is_async:
                    # async: result available after total_latency regardless of warp
                    model.add(
                        self.intervals[instruction_id].start_expr() >=
                        self.intervals[parent_id].start_expr() + min_separation
                    )
                else:
                    # sync: cross-warp pays hard latency penalty
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

            # ── ordering deps: pure sequencing, no sync cost ever ──────
            for parent in instruction.ordering_instructions:
                parent_id = str(parent)
                gap = instruction.modulo_issue_cycle - parent.modulo_issue_cycle
                min_separation = max(gap, parent.total_latency) if gap > 0 else parent.total_latency
                # unconditional — warp placement irrelevant
                model.add(
                    self.intervals[instruction_id].start_expr() >=
                    self.intervals[parent_id].start_expr() + min_separation
                )

        # no overlap per warp
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

        # makespan
        makespan = model.new_int_var(0, max_cycles - 1, "makespan")
        for instr in self.instructions:
            model.add(makespan >= self.intervals[str(instr)].start_expr() + instr.total_latency)
        model.minimize(makespan)

        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = use_verbose
        status = solver.solve(model)

        warp_instructions = [[] for _ in range(num_warps)]
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for itype in instruction_types:
                assigned = next(w for w in range(num_warps) if solver.value(instruction_type_assignments[itype][w]))
                warp_instructions[assigned].append((f"[type: {itype}]", -1))

            for instruction in self.instructions:
                key      = str(instruction)
                assigned = next(w for w in range(num_warps) if solver.value(warp_assignments[key][w]))
                start    = solver.value(self.intervals[key].start_expr())
                warp_instructions[assigned].append((f"{instruction.name}@{instruction.version}", start))
        else:
            print("INFEASIBLE!")

        for warp_id in range(num_warps):
            print(f"warp {warp_id}:")
            warp_instructions[warp_id].sort(key=lambda x: x[1])
            for entry in warp_instructions[warp_id]:
                print("\t", entry)


if __name__ == "__main__":
    II                   = 6
    N                    = 6
    max_cycles           = 500
    num_warps            = 2
    synchronization_cost = 80   # realistic hard latency penalty for cross-warp data dep
    use_verbose          = False

    LOAD_MODULO = 0
    MAC_MODULO  = 2 * II  # macs consume data from 2 iters ago

    def make_load(name: str, k: int) -> WAInstruction:
        return WAInstruction(name=name, version=k,
            parent_instructions=[], ordering_instructions=[],
            is_async=True,
            cuda_latency=1, total_latency=12,
            modulo_issue_cycle=LOAD_MODULO + k * II)

    def make_mac(k: int, parents: List[WAInstruction]) -> WAInstruction:
        return WAInstruction(name="mac", version=k,
            parent_instructions=parents, ordering_instructions=[],
            is_async=False,
            cuda_latency=1, total_latency=4,
            modulo_issue_cycle=MAC_MODULO + k * II)

    manager = WAInstructionManager()

    load_as: Dict[int, WAInstruction] = {}
    load_bs: Dict[int, WAInstruction] = {}
    macs:    Dict[int, WAInstruction] = {}

    # prologue: iter 0 and 1 — loads only
    for k in range(2):
        load_as[k] = make_load("load_a", k)
        load_bs[k] = make_load("load_b", k)

    # steady state
    for k in range(2, N + 2):
        load_as[k] = make_load("load_a", k)
        load_bs[k] = make_load("load_b", k)
        mac_parents = [load_as[k - 2], load_bs[k - 2]]
        if k - 3 in macs:
            mac_parents.append(macs[k - 3])  # accumulator chain
        macs[k - 2] = make_mac(k - 2, mac_parents)

        # back-pressure via ordering dep (no sync cost) — spaces loads out by II
        if k - 2 in macs:
            load_as[k].ordering_instructions.append(macs[k - 2])
            load_bs[k].ordering_instructions.append(macs[k - 2])

    # epilogue
    for k in [N, N + 1]:
        mac_parents = [load_as[k], load_bs[k]]
        if k - 1 in macs:
            mac_parents.append(macs[k - 1])
        macs[k] = make_mac(k, mac_parents)

    all_instructions = (
        [load_as[k] for k in sorted(load_as)] +
        [load_bs[k] for k in sorted(load_bs)] +
        [macs[k]    for k in sorted(macs)]
    )
    for instr in all_instructions:
        manager.add_instruction(instr)

    print(f"II={II}  N={N}  instructions={len(all_instructions)}")
    manager.solve(max_cycles, num_warps, synchronization_cost, ["load_a", "load_b", "mac"], use_verbose)