from copy import deepcopy
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union
from ortools.sat.python import cp_model
from rich import print

from warpir import *
from warpir.lowering import lower_stmt, lower_program

@dataclass
class MSInstructionEdge:
    source: 'MSInstruction'
    dest: 'MSInstruction'
    distance: int

@dataclass
class MSInstruction:
    resource: Union[str, Tuple[str, ...]]
    latency: int
    
    name: str
    output_var: Var
    
    parent_edges: List[MSInstructionEdge]
    instruction: Stmt

    @property
    def resource_keys(self) -> Tuple[str, ...]:
        if isinstance(self.resource, (list, tuple)):
            return tuple(self.resource)
        return (self.resource,)
    

class MSInstructionManager:
    def __init__(self):
        self.instructions: List[MSInstruction] = []
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
                    if resource in instruction.resource_keys:
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

if __name__ == "__main__":
    manager = MSInstructionManager()
    max_cycles = 100
    depth = 5
    resource_counts = {"tma": 4, "tc": 1}
    use_verbose = False

    BLOCK_SIZE = 16
    shared_tile_type = SharedTileType(
        GPUType.bf16,
        BLOCK_SIZE,
        BLOCK_SIZE,
        SharedTileLayout.row_major,
    )
    accum_tile_type = RegTileType(
        GPUType.fp32,
        16,
        16,
        RegTileLayout.row_major,
    )
    global_type = GlobalType(GPUType.bf16, shared_tile_type, 1, 1, -1, -1)

    A = Var("A", global_type)
    B = Var("B", global_type)
    C = Var("C", global_type)
    N = Var("N", ScalarType("int"))

    a_tile = Tile("a_tile", shared_tile_type)
    b_tile = Tile("b_tile", shared_tile_type)
    c_tile = Tile("c_tile", accum_tile_type)

    row = RawExpr("blockIdx.y")
    col = RawExpr("blockIdx.x")
    tile_idx = Var("tile_idx", ScalarType("int"))
    tile_limit = BinaryOp(
        BinaryOp(N, RawExpr(BLOCK_SIZE - 1), "+"),
        RawExpr(BLOCK_SIZE),
        "/",
    )

    load_a_stmt = TileLoadOp(
        a_tile,
        A,
        Coord([zero, zero, row, tile_idx]),
    )
    load_b_stmt = TileLoadOp(
        b_tile,
        B,
        Coord([zero, zero, tile_idx, col]),
    )
    mac_stmt = SeqStmt([
        ExprStmt(OpCall("warpgroup::mma_AB", [c_tile, a_tile, b_tile])),
        ExprStmt(OpCall("warpgroup::mma_async_wait", [])),
    ])

    load_a_instruction = MSInstruction("tma", 12, "load_a", a_tile.var, [], load_a_stmt)
    load_b_instruction = MSInstruction("tma", 12, "load_b", b_tile.var, [], load_b_stmt)
    mac_instruction = MSInstruction("tc", 4, "mac", c_tile.var, [], mac_stmt)

    for load_instr in [load_a_instruction, load_b_instruction]:
        mac_instruction.parent_edges.append(
            MSInstructionEdge(source=load_instr, dest=mac_instruction, distance=0)
        )
    mac_instruction.parent_edges.append(
        MSInstructionEdge(source=mac_instruction, dest=mac_instruction, distance=1)
    )

    for instr in [load_a_instruction, load_b_instruction, mac_instruction]:
        manager.add_instruction(instr)

    prologue, steady_state, epilogue, ii_value, num_stages, stage_of = manager.solve(
        max_cycles,
        depth,
        resource_counts,
        use_verbose=use_verbose,
    )

    print("II value: ", ii_value)
    print("Num stages: ", num_stages)
    print("Stage of: ", stage_of)

    variable_buffer_sizes: Dict[Var, int] = {
        a_tile.var: 1,
        b_tile.var: 1,
        c_tile.var: 1,
    }
    for instruction in manager.instructions:
        for parent_edge in instruction.parent_edges:
            if parent_edge.source.output_var in variable_buffer_sizes:
                variable_buffer_sizes[parent_edge.source.output_var] = max(
                    variable_buffer_sizes[parent_edge.source.output_var],
                    stage_of[instruction.name] - stage_of[parent_edge.source.name] + 1,
                )

    print("Variable buffer sizes: ", variable_buffer_sizes)

    new_variables: List[Var] = []
    for var in [a_tile.var, b_tile.var, c_tile.var]:
        for num_occ in range(variable_buffer_sizes[var]):
            new_variables.append(
                Var(f"{var.name}_{num_occ}", var.var_type)
            )

    print("new variables: ", [f"{var.name} - {var.var_type}" for var in new_variables])

    instruction_lookup = {instr.name: instr for instr in manager.instructions}

    def format_stage_entries(entries, stage_kind: str) -> List[str]:
        lines: List[str] = []
        if stage_kind == "prologue":
            for offset, inst_name in entries:
                inst = instruction_lookup[inst_name]
                lines.append(
                    f"{inst_name} (iter delta {offset}): {inst.instruction}"
                )
        elif stage_kind == "steady":
            for inst_name, buffer_offset in entries:
                inst = instruction_lookup[inst_name]
                lines.append(
                    f"{inst_name} (buffer offset {buffer_offset}): {inst.instruction}"
                )
        else:
            for inst_name, drain_offset in entries:
                inst = instruction_lookup[inst_name]
                lines.append(
                    f"{inst_name} (drain offset {drain_offset}): {inst.instruction}"
                )
        return lines

    def dump_schedule(label: str, stages, stage_kind: str):
        for idx, stage_entries in enumerate(stages):
            print(f"{label} stage {idx}:")
            for line in format_stage_entries(stage_entries, stage_kind):
                print("  ", line)

    dump_schedule("Prologue", prologue, "prologue")
    dump_schedule("Steady", steady_state, "steady")
    dump_schedule("Epilogue", epilogue, "epilogue")

    kernel_globals = KernelGlobals("globals")
    for shared in [A, B, C, N]:
        kernel_globals.add_var(shared)

    loop_body = SeqStmt([load_a_stmt, load_b_stmt, mac_stmt])
    tile_loop = ForStmt(
        AssignExpr(tile_idx, zero),
        BinaryOp(tile_idx, tile_limit, "<"),
        AssignExpr(tile_idx, BinaryOp(tile_idx, RawExpr(1), "+")),
        loop_body,
        inputs=[tile_idx],
        yields=[c_tile.var],
    )
    kernel_stmt = SeqStmt([
        a_tile.declare(),
        b_tile.declare(),
        c_tile.declare(),
        tile_idx.declare(),
        tile_loop,
    ])

    sample_program = Program(kernel_globals, kernel_stmt)
    print("Sample IR (before lowering):")
    print(sample_program.kernel_stmt)

    lowered_sample_program = lower_program(sample_program)
    print("Sample IR (after lowering):")
    print(lowered_sample_program.kernel_stmt)

    
    # Algorithm to convert 

    # first generate all of the buffered variables (tiles and their lowered vars)
    var_to_tile = {
        a_tile.var: a_tile,
        b_tile.var: b_tile,
        c_tile.var: c_tile,
    }

    buffered_variables: Dict[Var, List[Var]] = {}
    tile_buffer_map: Dict[Tile, List[Tile]] = {}
    for var, size in variable_buffer_sizes.items():
        tile = var_to_tile.get(var)
        buffered_vars: List[Var] = []
        for buf_id in range(size):
            if tile is not None:
                clone = Tile(
                    f"{tile.name}_{buf_id}",
                    tile.var_type,
                    tile.use_semaphores,
                    tile.num_consumers,
                )
                buffered_vars.append(clone)
                tile_buffer_map.setdefault(tile, []).append(clone)
            else:
                clone_var = deepcopy(var)
                clone_var.name = f"{var.name}_{buf_id}"
                buffered_vars.append(clone_var)
        buffered_variables[var] = buffered_vars

    def buffer_index(offset: int, var: Var) -> int:
        candidates = buffered_variables.get(var, [])
        return offset % len(candidates) if candidates else 0

    def instantiate_instruction(
        inst_name: str,
        iteration_expr: Expr,
        buffer_offset: int
    ) -> Stmt:
        instruction = deepcopy(instruction_lookup[inst_name].instruction)
        replacements: Dict[Var, Expr] = {tile_idx: iteration_expr}
        output_var = instruction_lookup[inst_name].output_var
        candidates = buffered_variables.get(output_var, [])
        if candidates:
            idx = buffer_index(buffer_offset, output_var)
            replacements[output_var] = candidates[idx]
            output_tile = var_to_tile.get(output_var)
            if output_tile is not None and output_tile in tile_buffer_map:
                clones = tile_buffer_map[output_tile]
                replacements[output_tile] = clones[idx]
                replacements[output_tile.var] = clones[idx]

        for buffered_var, clones in buffered_variables.items():
            if not clones or buffered_var in replacements:
                continue
            idx = buffer_index(buffer_offset, buffered_var)
            replacements[buffered_var] = clones[idx]
            tile = var_to_tile.get(buffered_var)
            if tile is not None and tile in tile_buffer_map:
                clones_for_tile = tile_buffer_map[tile]
                replacements[tile] = clones_for_tile[idx]
                replacements[tile.var] = clones_for_tile[idx]

        return instruction.replace_vars(replacements)

    prologue_stmts: List[Stmt] = []
    for stage_idx, stage_entries in enumerate(prologue):
        for iter_delta, inst_name in stage_entries:
            iteration_value = max(stage_idx - iter_delta, 0)
            iteration_expr = RawExpr(str(iteration_value))
            prologue_stmts.append(
                instantiate_instruction(inst_name, iteration_expr, iter_delta)
            )

    steady_stmts: List[Stmt] = []
    for entries in steady_state:
        for inst_name, buffer_offset in entries:
            steady_stmts.append(
                instantiate_instruction(inst_name, tile_idx, buffer_offset)
            )

    def make_epilogue_expr(drain_offset: int) -> Expr:
        return BinaryOp(
            BinaryOp(tile_limit, RawExpr(1), "-"),
            RawExpr(drain_offset),
            "-"
        )

    epilogue_stmts: List[Stmt] = []
    for entries in epilogue:
        for inst_name, drain_offset in entries:
            epilogue_stmts.append(
                instantiate_instruction(inst_name, make_epilogue_expr(drain_offset), drain_offset)
            )

    buffered_declarations: List[Stmt] = []
    for clones in buffered_variables.values():
        buffered_declarations.extend([clone.declare() for clone in clones])

    steady_body = SeqStmt(steady_stmts)
    pipelined_tile_loop = ForStmt(
        AssignExpr(tile_idx, zero),
        BinaryOp(tile_idx, tile_limit, "<"),
        AssignExpr(tile_idx, BinaryOp(tile_idx, RawExpr(1), "+")),
        steady_body,
        inputs=[tile_idx],
        yields=[buffered_variables[c_tile.var][0]],
    )

    pipelined_kernel_stmt = SeqStmt(
        buffered_declarations
        + [tile_idx.declare()]
        + prologue_stmts
        + [pipelined_tile_loop]
        + epilogue_stmts
    )

    pipelined_program = Program(kernel_globals, pipelined_kernel_stmt)
    print("Pipelined IR:")
    print(pipelined_program.kernel_stmt)

