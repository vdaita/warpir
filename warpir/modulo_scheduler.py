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
            
    def find_cycles_with_start(self, instruction: MSInstruction) -> List[List[MSInstructionEdge]]:
        cycles = []
        
        def dfs(current: MSInstruction, path: List[MSInstructionEdge]):
            for child_edge in self.child_edges.get(current.name, []):
                if child_edge.dest == instruction:
                    cycles.append(path + [child_edge])
                else:
                    if not any(child_edge.dest == path_edge.dest for path_edge in path):
                        dfs(child_edge.dest, path + [child_edge])
                        
        dfs(instruction, [])
        return cycles
    
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
                    if instruction.resource == resource_instructions:
                        resource_instructions.append(self.intervals[(current_depth, instruction.name)])
            model.add_cumulative(
                resource_instructions,
                [1 for _ in range(len(resource_instructions))],
                resource_counts[resource]
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
        else:
            print("No solution found.")    
            
# def resource_mii(manager: MSInstructionManager, capacity: Dict[str, int]):
#     usage: Dict[str, int] = {}
#     for resource in capacity:
#         usage[resource] = sum(1 for instruction in manager.instructions if instruction.resource == resource)
#     return max(ceil(usage[resource] / capacity[resource]) for resource in capacity)
    
# def recurrence_mii(manager: MSInstructionManager):
#     best = 1
#     for root_instruction in manager.instructions:
#         cycles = manager.find_cycles_with_start(root_instruction)
#         for cycle in cycles:
#             lat_sum = sum(edge.source.latency for edge in cycle)
#             dist_sum = sum(edge.distance for edge in cycle)
#             if dist_sum > 0:
#                 best = max(best, ceil(lat_sum / dist_sum))
#     return best
# 

if __name__ == "__main__":
    manager = MSInstructionManager()
    max_cycles = 48
    depth = 2
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