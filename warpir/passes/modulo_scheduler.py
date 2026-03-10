from dataclasses import dataclass
from typing import List, Dict, Tuple, Union
from ortools.sat.python import cp_model
from rich import print

from warpir.ir.ops import Kernel, ForOp, TMALoadOp, TMAStoreOp, MMAOp, Value, IterArg, YieldOp

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
            
            # prologue: list of steps, each step is list of (logical_iter, inst_name)
            prologue = []
            for i in range(num_stages - 1):
                step = []
                for instruction in self.instructions:
                    if stage_of[instruction.name] <= i:
                        step.append((i - stage_of[instruction.name], instruction.name))
                prologue.append(step)

            # steady state: grouped by k, each entry is list of (inst_name, buffer_offset)
            steady_state = []
            for k in range(num_stages - 1, depth):
                step = []
                for instruction in self.instructions:
                    step.append((instruction.name, k - stage_of[instruction.name]))
                steady_state.append(step)

            # epilogue: list of drain steps, each is list of (inst_name, buffer_offset)
            epilogue = []
            for i in range(1, num_stages):
                step = []
                for instruction in self.instructions:
                    if stage_of[instruction.name] >= i:
                        step.append((instruction.name, stage_of[instruction.name] - i))
                epilogue.append(step)

            return prologue, steady_state, epilogue, ii_value, num_stages, stage_of

        else:
            print("No solution found.")
            return None, None, None, None, None, None

def kernel_pass(kernel: Kernel) -> Kernel:
    resource_counts = {"tma": 4, "tc": 1}
    max_cycles = 100
    depth = 5
    use_verbose = False

    new_body = []
    for op in kernel.body:
        if not isinstance(op, ForOp) or op.step != 1:
            new_body.append(op)
            continue

        manager = MSInstructionManager()
        instructions: Dict[str, MSInstruction] = {}

        for body_op in op.body:
            instr = None
            if type(body_op) == TMALoadOp:
                instr = MSInstruction(resource="tma", latency=12, name=str(body_op), parent_edges=[])
            elif type(body_op) == MMAOp:
                instr = MSInstruction(resource="tc", latency=4, name=str(body_op), parent_edges=[])
            elif type(body_op) == TMAStoreOp:
                instr = MSInstruction(resource="tma", latency=12, name=str(body_op), parent_edges=[])
            if instr:
                instructions[str(body_op)] = instr

        emitter_map: Dict[Value, MSInstruction] = {}
        for body_op in op.body:
            if type(body_op) in (TMALoadOp, MMAOp):
                emitter_map[body_op.result] = instructions[str(body_op)]

        yielded_values: set = {ia.block_arg for ia in op.iter_args}
        yield_op = next(o for o in op.body if isinstance(o, YieldOp))
        iter_arg_to_yield: Dict[Value, Value] = {
            ia.block_arg: yield_op.values[i]
            for i, ia in enumerate(op.iter_args)
        }

        for body_op in op.body:
            if str(body_op) not in instructions:
                continue
            instr = instructions[str(body_op)]
            input_fields = {k: v for k, v in vars(body_op).items() if k != 'result'}
            for operand in (v for v in input_fields.values() if isinstance(v, Value)):
                distance = 1 if operand in yielded_values else 0
                source_val = iter_arg_to_yield.get(operand, operand)
                if source_val in emitter_map:
                    instr.parent_edges.append(MSInstructionEdge(
                        source=emitter_map[source_val], dest=instr, distance=distance
                    ))

        for instr in instructions.values():
            manager.add_instruction(instr)

        prologue, steady_state, epilogue, ii_value, num_stages, stage_of = manager.solve(
            max_cycles=max_cycles,
            depth=depth,
            resource_counts=resource_counts,
            use_verbose=use_verbose,
        )

        if prologue is None:
            new_body.append(op)
            continue

        instr_to_op: Dict[str, object] = {str(b): b for b in op.body}

        # staged Values: num_stages fresh SSA values per result-producing op
        staged: Dict[str, List[Value]] = {
            inst_name: [
                Value(f"{instr_to_op[inst_name].result.name}_s{s}", instr_to_op[inst_name].result.type)
                for s in range(num_stages)
            ]
            for inst_name in stage_of
            if hasattr(instr_to_op[inst_name], 'result')
        }

        # Prologue: induction var → concrete iteration, result → staged[logical_iter] slot
        for step in prologue:
            for (logical_iter, inst_name) in step:
                source_op = instr_to_op[inst_name]
                subs = {op.induction_var: logical_iter}
                if hasattr(source_op, 'result'):
                    subs[source_op.result] = staged[inst_name][logical_iter]
                new_body.append(source_op.substitute(subs))

        # _cur iter_args: one per staged result, init from the last prologue-filled staged value
        staged_iter_args = tuple(
            IterArg(
                block_arg=Value(f"{instr_to_op[inst_name].result.name}_cur", instr_to_op[inst_name].result.type),
                init=staged[inst_name][num_stages - 1],
            )
            for inst_name in stage_of
            if hasattr(instr_to_op[inst_name], 'result')
        )
        # inst_name → the _cur Value carried through the loop
        staged_cur_vals: Dict[str, Value] = {
            inst_name: staged_iter_args[i].block_arg
            for i, inst_name in enumerate(n for n in stage_of if hasattr(instr_to_op[n], 'result'))
        }
        # original result → _cur Value, for remapping input operands inside the loop body
        result_to_cur: Dict[Value, Value] = {
            instr_to_op[inst_name].result: cur_val
            for inst_name, cur_val in staged_cur_vals.items()
        }

        # _final Values: what the ForOp yields for each staged buffer
        staged_final_vals: Dict[str, Value] = {
            inst_name: Value(f"{instr_to_op[inst_name].result.name}_final", instr_to_op[inst_name].result.type)
            for inst_name in stage_of
            if hasattr(instr_to_op[inst_name], 'result')
        }
        # original result → _final Value, for remapping epilogue operands
        result_to_final: Dict[Value, Value] = {
            instr_to_op[inst_name].result: final_val
            for inst_name, final_val in staged_final_vals.items()
        }

        # Steady state body: remap all inputs via result_to_cur, keep induction var symbolic,
        # remap result to _cur so the op writes into the right slot
        steady_body = []
        for (inst_name, stage_offset) in steady_state[0]:
            source_op = instr_to_op[inst_name]
            subs = {**result_to_cur, op.induction_var: op.induction_var}
            if hasattr(source_op, 'result'):
                subs[source_op.result] = staged_cur_vals[inst_name]
            steady_body.append(source_op.substitute(subs))

        steady_body.append(YieldOp(values=
            tuple(staged_cur_vals[n] for n in stage_of if hasattr(instr_to_op[n], 'result'))
            + tuple(result_to_cur.get(v, v) for v in yield_op.values)
        ))

        new_body.append(ForOp(
            induction_var=op.induction_var,
            start=num_stages - 1,
            stop=op.stop,
            step=op.step,
            tile_size=op.tile_size,
            iter_args=staged_iter_args + op.iter_args,
            body=tuple(steady_body),
            results=tuple(staged_final_vals.values()) + op.results,
        ))

        # Epilogue: remap staged results to _final, iter_args to their loop results
        iter_arg_final: Dict[Value, Value] = {
            ia.block_arg: op.results[i]
            for i, ia in enumerate(op.iter_args)
        }
        for step in epilogue:
            for (inst_name, buf_offset) in step:
                source_op = instr_to_op[inst_name]
                subs = {**result_to_final, **iter_arg_final, op.induction_var: buf_offset}
                new_body.append(source_op.substitute(subs))

    return Kernel(name=kernel.name, inputs=kernel.inputs, outputs=kernel.outputs, body=tuple(new_body))

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