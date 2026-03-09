from dataclasses import dataclass, field
from pickletools import string1
from typing import List, Dict, Tuple
from math import ceil
from ortools.sat.python import cp_model
from rich import print

@dataclass
class MSInstructionEdge:
    source: 'MSInstruction'
    dest: 'MSInstruction'
    distance: int

@dataclass
class MSInstruction:
    resource: str
    latency: int
    name: str
    
    parent_edges: List[MSInstructionEdge]
    

class MSInstructionManager:
    def __init__(self):
        self.instructions = []
        self.child_edges: Dict[str, List[MSInstructionEdge]] = {}
        self.intervals: Dict[Tuple[int, str], cp_model.IntervalVar] = {}
            
    def add_instruction(self, instruction: MSInstruction):
        self.instructions.append(instruction)
        for parent_edge in instruction.parent_edges:
            self.child_edges[parent_edge.source.name] = self.child_edges.get(parent_edge.source.name, []) + [parent_edge]

    def solve(self, max_cycles: int, depth: int, resource_counts: Dict[str, int], use_verbose=False):
        model = cp_model.CpModel()
        
        print("child edges: ", self.child_edges)
        
        for current_depth in range(depth):
            for instruction in self.instructions:
                instruction_start = model.new_int_var(0, max_cycles - 1 - instruction.latency, f"start_{instruction.name}_{current_depth}")
                instruction_interval = model.new_interval_var(
                    instruction_start,
                    instruction.latency,
                    instruction_start + instruction.latency,
                    f"interval_{instruction.name}_{current_depth}"
                )
                self.intervals[(current_depth, instruction.name)] = instruction_interval
    
        print("intervals: ", self.intervals)
            
        # parent-child relationships. 
        for current_depth in range(depth):
            for instruction in self.instructions:
                for parent_edge in instruction.parent_edges:
                    if current_depth - parent_edge.distance >= 0:
                        model.add(self.intervals[(current_depth, instruction.name)].start_expr() >= self.intervals[(current_depth - parent_edge.distance, parent_edge.source.name)].end_expr()) # type: ignore
                        print(f"{(current_depth - parent_edge.distance, parent_edge.source.name)} -> {(current_depth, instruction.name)}")
                        
        # resource cumulative
        for resource in resource_counts.keys():
            resource_instructions = []
            for current_depth in range(depth):
                for instruction in self.instructions:
                    if instruction.resource == resource:
                        resource_instructions.append(self.intervals[(current_depth, instruction.name)])
            model.add_cumulative(
                resource_instructions,
                [1 for _ in range(len(resource_instructions))],
                resource_counts[resource]
            )
                    
        ii = model.new_int_var(0, max_cycles - 1, "ii")
        for instruction in self.instructions:
            for i in range(0, depth - 1):
                model.add(
                    self.intervals[(i + 1, instruction.name)].start_expr() ==
                    self.intervals[(i, instruction.name)].start_expr() + ii # type: ignore
                ) 
                    
                    
        makespan = model.new_int_var(0, max_cycles - 1, "makespan")
        model.add_max_equality(makespan, [interval.end_expr() for interval in self.intervals.values()]) # type: ignore
        
        model.minimize(makespan)
            
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = use_verbose
        
        status = solver.solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            time_assignments = [[] for _ in range(max_cycles)]
            for current_depth in range(depth):
                for instruction in self.instructions:
                    start_time = solver.value(self.intervals[(current_depth, instruction.name)].start_expr()) # type: ignore
                    end_time = solver.value(self.intervals[(current_depth, instruction.name)].end_expr()) # type: ignore
                    print("current depth: ", current_depth, " instruction name: ", instruction.name, start_time, end_time)
                    
                    time_assignments[start_time].append(f"start {current_depth}_{instruction.name}")
                    time_assignments[end_time].append(f"end {current_depth}_{instruction.name}")

            for i in range(max_cycles):
                print(f"cycle {i}: {time_assignments[i]}") 
                
            ii_value = solver.value(ii)
            print("II value: ", ii_value)
            kernel_start: Dict[str, int] = {
                instruction.name: solver.value(self.intervals[(0, instruction.name)].start_expr()) # type: ignore
                for instruction in self.instructions
            }
            stage_of: Dict[str, int] = {
                instruction.name: kernel_start[instruction.name] // ii_value
                for instruction in self.instructions
            }
            num_stages = max(
                solver.value(self.intervals[(0, instruction.name)].start_expr()) // ii_value # type: ignore
                for instruction in self.instructions
            ) + 1
            
            prologue = []
            for i in range(num_stages - 1):
                step = []
                for instruction in self.instructions:
                    if stage_of[instruction.name] <= i:
                        step.append((i - stage_of[instruction.name], instruction.name))
                prologue.append(step)
        
            steady_state = []
            for k in range(num_stages - 1, depth):
                for instruction in self.instructions:
                    steady_state.append((instruction.name, k - stage_of[instruction.name]))
            
            epilogue = []
            for i in range(1, num_stages):
                step = []
                for instruction in self.instructions:
                    if stage_of[instruction.name] >= i:
                        step.append((instruction.name, stage_of[instruction.name]))
                epilogue.append(step)
                
            print("// Prologue")
            for i, step in enumerate(prologue):
                print(f"// iteration {i}")
                for (logical_iter, inst_name) in step:
                    inst = next(x for x in self.instructions if x.name == inst_name)
                    print(f"  {inst_name}(buf\[{logical_iter}])   // stage {stage_of[inst_name]}, cycle {kernel_start[inst_name]}")
                print()
            
            print(f"// Steady State - repeats every {ii_value} cycles")
            print(f"for k in range({num_stages - 1}, N):")
            for instruction in self.instructions:
                offset = stage_of[instruction.name]
                if offset == 0:
                    print(f"    {instruction.name}(buf\[k])   // loads fresh data")
                else:
                    print(f"    {instruction.name}(buf\[k - {offset}])   // consuming data from {offset} iters ago")
            print()
            
            print("// Epilogue")
            for i, step in enumerate(epilogue):
                print(f"// drain {i}")
                for (inst_name, stage) in step:
                    print(f"  {inst_name}(buf\[N - {stage - i}])")
                print()
                
        else:
            print("No solution found.") 

if __name__ == "__main__":
    manager = MSInstructionManager()
    max_cycles = 100
    depth = 5
    resource_counts = {"tma": 4, "tc": 1}
    use_verbose = False
    
    load_a_instruction = MSInstruction("tma", 12, "load_a", [])
    load_b_instruction = MSInstruction("tma", 12, "load_b", [])
    mac_instruction = MSInstruction("tc", 4, "mac", [])
    
    for load_instruction in [load_a_instruction, load_b_instruction]:
        mac_instruction.parent_edges.append(
            MSInstructionEdge(
                source=load_instruction, 
                dest=mac_instruction,
                distance=0
            )
        )
    mac_instruction.parent_edges.append(
        MSInstructionEdge(
            source=mac_instruction,
            dest=mac_instruction,
            distance=1
        )
    )
    
    for instruction in [load_a_instruction, load_b_instruction, mac_instruction]:
        manager.add_instruction(instruction)
    manager.solve(max_cycles, depth, resource_counts, use_verbose=use_verbose)