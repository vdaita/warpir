from dataclasses import dataclass
from typing import List, Dict, Tuple, Union
from ortools.sat.python import cp_model
# from rich import print

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
    WaitBufOp,
    WaitOp,
    YieldOp,
)
from warpir.ir.types import SharedBufferType, SharedTileType

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

    def solve(
        self,
        max_cycles: int,
        depth: int,
        resource_counts: Dict[str, int],
        use_verbose: bool = False,
        max_pipeline_depth: Union[int, None] = None,
    ):
        model = cp_model.CpModel()
        
        
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
    
        # print("intervals: ", self.intervals)
            
        # parent-child relationships. 
        for current_depth in range(depth):
            for instruction in self.instructions:
                for parent_edge in instruction.parent_edges:
                    if current_depth - parent_edge.distance >= 0:
                        model.add(self.intervals[(current_depth, instruction.name)].start_expr() >= self.intervals[(current_depth - parent_edge.distance, parent_edge.source.name)].end_expr()) # type: ignore
                        # print(f"{(current_depth - parent_edge.distance, parent_edge.source.name)} -> {(current_depth, instruction.name)}")
                        
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
                    
        ii = model.new_int_var(1, max_cycles - 1, "ii")
        for instruction in self.instructions:
            for i in range(0, depth - 1):
                model.add(
                    self.intervals[(i + 1, instruction.name)].start_expr() ==
                    self.intervals[(i, instruction.name)].start_expr() + ii # type: ignore
                ) 

        if max_pipeline_depth is not None:
            for instruction in self.instructions:
                model.add(
                    self.intervals[(0, instruction.name)].start_expr() <=
                    max_pipeline_depth * ii  # type: ignore
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
                    # print("current depth: ", current_depth, " instruction name: ", instruction.name, start_time, end_time)
                    
                    time_assignments[start_time].append(f"start {current_depth}_{instruction.name}")
                    time_assignments[end_time].append(f"end {current_depth}_{instruction.name}")

            # for i in range(max_cycles):
                # print(f"cycle {i}: {time_assignments[i]}") 
                
            ii_value = solver.value(ii)
            kernel_start: Dict[str, int] = {
                instruction.name: solver.value(self.intervals[(0, instruction.name)].start_expr()) # type: ignore
                for instruction in self.instructions
            }
            base_start = min(kernel_start.values()) if kernel_start else 0
            stage_of: Dict[str, int] = {
                instruction.name: (kernel_start[instruction.name] - base_start) // ii_value
                for instruction in self.instructions
            }
            num_stages = max(stage_of.values()) + 1
            
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

            return prologue, steady_state, epilogue, ii_value, num_stages, stage_of, kernel_start

        else:
            raise ValueError("No solution found!")


def _iter_value_operands(x):
    if isinstance(x, Value):
        yield x
        return
    if isinstance(x, BufSlotExpr):
        if x.value is not None:
            yield x.value
        return
    if isinstance(x, tuple):
        for e in x:
            yield from _iter_value_operands(e)
        return


def _op_schedule_info(op: Op) -> Union[Tuple[str, int], None]:
    if isinstance(op, TMALoadOp):
        return ("tma", 12)
    if isinstance(op, TMAStoreOp):
        return ("tma", 12)
    if isinstance(op, MMAOp):
        return ("tc", 4)
    if isinstance(op, WaitOp):
        return ("cuda", 1)
    if isinstance(op, YieldOp):
        return None
    return ("cuda", 1)

def kernel_pass(kernel: Kernel, max_pipeline_depth: Union[int, None] = None) -> Kernel:
    resource_counts = {"tma": 4, "tc": 1, "cuda": 1}
    max_cycles = 5000
    default_depth = 10
    use_verbose = False

    solve_depth = default_depth
    if max_pipeline_depth is not None:
        solve_depth = max(2, max_pipeline_depth + 1)

    new_body = []
    for op in kernel.body:
        if not isinstance(op, ForOp) or op.step != 1:
            new_body.append(op)
            continue

        manager = MSInstructionManager()
        instructions: Dict[str, MSInstruction] = {}
        instr_order: Dict[str, int] = {}

        for idx, body_op in enumerate(op.body):
            sched = _op_schedule_info(body_op)
            if sched is None:
                continue
            resource, latency = sched
            instr = MSInstruction(
                resource=resource,
                latency=latency,
                name=str(body_op),
                parent_edges=[],
            )
            instructions[str(body_op)] = instr
            instr_order[str(body_op)] = idx

        emitter_map: Dict[Value, MSInstruction] = {}
        for body_op in op.body:
            if hasattr(body_op, "result") and isinstance(getattr(body_op, "result"), Value):
                op_key = str(body_op)
                if op_key in instructions:
                    emitter_map[getattr(body_op, "result")] = instructions[op_key]

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
            for operand_field in input_fields.values():
                for operand in _iter_value_operands(operand_field):
                    distance = 1 if operand in yielded_values else 0
                    source_val = iter_arg_to_yield.get(operand, operand)
                    if source_val in emitter_map:
                        instr.parent_edges.append(MSInstructionEdge(
                            source=emitter_map[source_val], dest=instr, distance=distance
                        ))

        for instr in instructions.values():
            manager.add_instruction(instr)

        (
            prologue,
            steady_state,
            epilogue,
            ii_value,
            num_stages,
            stage_of,
            kernel_start,
        ) = manager.solve(
            max_cycles=max_cycles,
            depth=solve_depth,
            resource_counts=resource_counts,
            use_verbose=use_verbose,
            max_pipeline_depth=max_pipeline_depth,
        )

        if prologue is None:
            new_body.append(op)
            continue

        instr_to_op: Dict[str, Op] = {str(b): b for b in op.body}
        load_ops: List[TMALoadOp] = [b for b in op.body if isinstance(b, TMALoadOp)]
        scheduled_names = sorted(
            instructions.keys(),
            key=lambda n: (kernel_start[n], instr_order[n]),
        )

        # ---- Build pipelined IR using buffer ops ----
        pipeline_depth = num_stages - 1

        # ---- ALLOC SHARED BUFFERS ----
        buf_values: Dict[Value, Value] = {}
        for ld in load_ops:
            if not isinstance(ld.result.type, SharedTileType):
                continue
            buf_type = SharedBufferType(
                tile_type=ld.result.type,
                count=num_stages,
            )
            buf_val = Value(f"{ld.source.name.lower()}_bufs", buf_type)
            buf_values[ld.result] = buf_val
            new_body.append(AllocSharedOp(result=buf_val))

        # ---- PROLOGUE: fill pipeline with pipeline_depth load steps ----
        for step_idx, step_ops in enumerate(prologue):
            for (logical_iter, inst_name) in step_ops:
                source_op = instr_to_op[inst_name]
                if not isinstance(source_op, TMALoadOp):
                    continue
                coords_op = source_op.substitute(
                    {op.induction_var: logical_iter}
                )
                coords = source_op.coords
                if isinstance(coords_op, TMALoadOp):
                    coords = coords_op.coords
                new_body.append(TMALoadBufOp(
                    buf=buf_values[source_op.result],
                    slot=step_idx,
                    source=source_op.source,
                    coords=coords,
                ))

        # ---- STEADY STATE LOOP ----
        body_ops: List[Op] = []
        if load_ops:
            first_consumer_stage = min(
                stage_of[name]
                for name in scheduled_names
                for v in _iter_value_operands(vars(instr_to_op[name]))
                if isinstance(instr_to_op[name], Op)
                if v in buf_values
            ) if any(
                v in buf_values
                for name in scheduled_names
                for v in _iter_value_operands(vars(instr_to_op[name]))
            ) else 0

            compute_slot = BufSlotExpr(
                value=op.induction_var,
                offset=-first_consumer_stage,
                modulus=num_stages,
            )
            body_ops.append(WaitBufOp(
                bufs=tuple(buf_values[ld.result] for ld in load_ops),
                slot=compute_slot,
            ))

        for inst_name in scheduled_names:
            source_op = instr_to_op[inst_name]
            op_stage = stage_of[inst_name]
            slot = BufSlotExpr(
                value=op.induction_var,
                offset=-op_stage,
                modulus=num_stages,
            )

            if isinstance(source_op, TMALoadOp):
                body_ops.append(TMALoadBufOp(
                    buf=buf_values[source_op.result],
                    slot=slot,
                    source=source_op.source,
                    coords=source_op.coords,
                ))
            elif isinstance(source_op, WaitOp):
                continue
            elif isinstance(source_op, MMAOp):
                body_ops.append(MMABufOp(
                    result=source_op.result,
                    a_buf=buf_values.get(source_op.a, source_op.a),
                    a_slot=slot if source_op.a in buf_values else 0,
                    b_buf=buf_values.get(source_op.b, source_op.b),
                    b_slot=slot if source_op.b in buf_values else 0,
                    accum=source_op.accum,
                    transpose_b=source_op.transpose_b,
                ))
            elif not isinstance(source_op, YieldOp):
                body_ops.append(source_op)

        body_ops.append(yield_op)

        new_body.append(ForOp(
            induction_var=op.induction_var,
            start=pipeline_depth,
            stop=op.stop,
            step=op.step,
            tile_size=op.tile_size,
            iter_args=op.iter_args,
            body=tuple(body_ops),
            results=op.results,
        ))

        # ---- EPILOGUE: drain remaining pipeline_depth compute steps ----
        for ep_step, ep_ops in enumerate(epilogue):
            if load_ops:
                buf_list = tuple(buf_values[ld.result] for ld in load_ops)
                drain_slot = ep_step % num_stages
                new_body.append(WaitBufOp(bufs=buf_list, slot=drain_slot))
            for (inst_name, _buf_offset) in ep_ops:
                source_op = instr_to_op[inst_name]
                drain_slot = ep_step % num_stages
                if isinstance(source_op, (TMALoadOp, WaitOp, YieldOp)):
                    continue
                if isinstance(source_op, MMAOp):
                    new_body.append(MMABufOp(
                        result=source_op.result,
                        a_buf=buf_values.get(source_op.a, source_op.a),
                        a_slot=drain_slot if source_op.a in buf_values else 0,
                        b_buf=buf_values.get(source_op.b, source_op.b),
                        b_slot=drain_slot if source_op.b in buf_values else 0,
                        accum=source_op.accum,
                        transpose_b=source_op.transpose_b,
                    ))
                else:
                    new_body.append(source_op)

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