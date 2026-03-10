from dataclasses import dataclass
from typing import List, Dict, Tuple
from ortools.sat.python import cp_model
from rich import print
@dataclass
class WAInstruction:
    name: str
    version: int # this will help me manage different stages in my pipeline
    
    parent_instructions: List['WAInstruction']
    is_async: bool # if this is already an async function then you don't incur an additional cost from a producer-consumer relationships across warps (ie warps)
    
    cuda_latency: int # this is how much of the cuda resource is used on that warp
    total_latency: int # this the data that should be used for the data dependency
    
    def __str__(self) -> str:
        return f"{self.name}_{self.version}"
        
class WAInstructionManager:
    def __init__(self):
        self.instructions: List[WAInstruction] = []
        self.intervals: Dict[str, cp_model.IntervalVar] = {}
    
    def add_instruction(self, instruction: WAInstruction):
        self.instructions.append(instruction)
    def solve(self, max_cycles: int, num_warps: int, synchronization_cost: int, instructions: List[WAInstruction], use_verbose=False):
        model = cp_model.CpModel()
    
        warp_assignments: Dict[str, List[cp_model.BoolVarT]] = {}
        
        for instruction in self.instructions:
            instruction_start = model.new_int_var(0, max_cycles - 1 - max(instruction.cuda_latency, instruction.total_latency), f"start_{str(instruction)}")
            instruction_interval = model.new_interval_var(
                instruction_start,
                instruction.cuda_latency,
                instruction_start + instruction.cuda_latency,
                f"interval_{str(instruction)}"
            ) # you need to account for how long something takes?
            self.intervals[str(instruction)] = instruction_interval
            
            warp_assignments[str(instruction)] = []
            for warp in range(num_warps):
                warp_assignment = model.new_bool_var(
                    f"warp_{str(instruction)}_{warp}"   
                )
                warp_assignments[str(instruction)].append(warp_assignment)
            model.add_exactly_one(warp_assignments[str(instruction)]) # warp uniqueness
                    
        for instruction in self.instructions:
            for parent in instruction.parent_instructions:
                # if this is an async operation, 
                parent_id = str(parent)
                instruction_id = str(instruction)
                if parent.is_async:
                    model.add(self.intervals[instruction_id].start_expr() >= self.intervals[parent_id].start_expr() + parent.total_latency) # type: ignore
                else:
                    for parent_warp in range(num_warps):
                        for child_warp in range(num_warps):
                            this_warp_assignment = model.new_bool_var(f"instruction_{instruction_id}_parent_{parent_warp}_child_{child_warp}")
                            model.add_bool_and([
                                warp_assignments[parent_id][parent_warp],
                                warp_assignments[instruction_id][child_warp]
                            ]).only_enforce_if(this_warp_assignment)
                            model.add_bool_or([
                                warp_assignments[parent_id][parent_warp].negated(),
                                warp_assignments[instruction_id][child_warp].negated()
                            ]).only_enforce_if(this_warp_assignment.negated())
                            
                            if parent_warp == child_warp:
                                model.add(self.intervals[instruction_id].start_expr() >= self.intervals[parent_id].start_expr() + parent.total_latency).only_enforce_if(this_warp_assignment) # cross-warp spills + concurrency
                            else:
                                model.add(self.intervals[instruction_id].start_expr() >= self.intervals[parent_id].start_expr() + parent.total_latency + synchronization_cost).only_enforce_if(this_warp_assignment) # handles parent-child dependencies
            
        for w in range(num_warps):
            optional_intervals = []
            for instr in self.instructions:
                key = str(instr)
                on_w = warp_assignments[key][w]
                start = self.intervals[key].start_expr()
                
                optional_intervals.append(
                    model.new_optional_interval_var(
                        start, instr.cuda_latency,
                        start + instr.cuda_latency,
                        on_w, f"opt_{key}_w{w}"
                    )
                )
            model.add_no_overlap(optional_intervals)
            # if both are active, one must start after the other
            # for interval_index in range(len(optional_intervals) - 1):
            #     curr, next = optional_intervals[interval_index], optional_intervals[interval_index + 1]
            #     model.add(curr.start_expr() > next.start_expr()).only_enforce_if(curr.presence_literals() + next.presence_literals())
                
            
        makespan = model.new_int_var(0, max_cycles - 1, "makespan")
        for instr in self.instructions:
            key = str(instr)
            start = self.intervals[key].start_expr()
            model.add(makespan >= start + instr.total_latency)

        model.minimize(makespan)
            
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = use_verbose
        status = solver.solve(model)
        
        
        warp_instructions = [[] for _ in range(num_warps)]
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            for instruction in self.instructions:
                instruction_id = str(instruction)
                warp_assignment_values = [solver.value(warp_assignments[instruction_id][i]) for i in range(num_warps)]
                assigned_warp = warp_assignment_values.index(True)
                
                start_time = solver.value(self.intervals[instruction_id].start_expr())
                
                warp_instructions[assigned_warp].append((f"{instruction.name}@{instruction.version}", start_time))
        else:
            print("INFEASIBLE!")
        
        for warp_id in range(num_warps):
            print("Warp id: ", warp_id)
            warp_instructions[warp_id].sort(key=lambda x: x[1])
            for instruction in warp_instructions[warp_id]:
                print("\t", instruction)
            
if __name__ == "__main__":
    manager = WAInstructionManager()
    max_cycles = 100
    num_warps = 2
    synchronization_cost = 100
    use_verbose = False
    
    def generate_load_instruction(name: str, version: int):
        return WAInstruction(
            name=name,
            version=version,
            parent_instructions=[],
            is_async=True,
            cuda_latency=1,
            total_latency=12
        )
    
    def generate_mul_instruction(name: str, version: int, parents: List[WAInstruction]):
        return WAInstruction(
            name=name,
            version=version,
            parent_instructions=parents,
            is_async=False,
            cuda_latency=1,
            total_latency=4
        )
    
    # write out your pipeline here
    load_as = [generate_load_instruction("load_a", i) for i in range(4)]
    load_bs = [generate_load_instruction("load_b", i) for i in range(4)]
    mul0 = generate_mul_instruction("mac", 0, [load_as[0], load_bs[0]])
    mul1 = generate_mul_instruction("mac", 1, [load_as[1], load_bs[1], mul0])
    mul2 = generate_mul_instruction("mac", 2, [load_as[2], load_bs[2], mul1])
    mul3 = generate_mul_instruction("mac", 3, [load_as[3], load_bs[3], mul2])

<<<<<<< HEAD:warpir/passes/warp_assigner.py
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

# TODO: convert instructions into WAInstructions
=======
    instructions = load_as + load_bs + [mul0, mul1, mul2, mul3]
    for instruction in instructions:
        manager.add_instruction(instruction)
    
    manager.solve(max_cycles, num_warps, synchronization_cost, instructions, use_verbose=use_verbose)
>>>>>>> parent of 5b96398 (warp assignments may be cooked):warpir_v2/passes/warp_assigner.py
