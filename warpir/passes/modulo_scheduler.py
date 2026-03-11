from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union, cast

from ortools.sat.python import cp_model

from warpir import *
from warpir.lowering import lower_program


@dataclass
class MSInstructionEdge:
    source: "MSInstruction"
    dest: "MSInstruction"
    distance: int


@dataclass
class MSInstruction:
    resource: Union[str, Tuple[str, ...]]
    latency: int
    name: str
    output_var: Optional[Var]
    parent_edges: List[MSInstructionEdge]
    instruction: Stmt
    input_vars: List[Var]

    @property
    def resource_keys(self) -> Tuple[str, ...]:
        if isinstance(self.resource, (list, tuple)):
            return tuple(self.resource)
        return (self.resource,)


@dataclass
class MSScheduleResult:
    prologue: List[List[Tuple[int, MSInstruction]]]
    steady_state: List[List[Tuple[int, MSInstruction]]]
    epilogue: List[List[Tuple[int, MSInstruction]]]
    ii_value: int
    num_stages: int
    stage_of: Dict[str, int]


class MSInstructionManager:
    def __init__(self):
        self.instructions: List[MSInstruction] = []
        self.child_edges: Dict[str, List[MSInstructionEdge]] = {}
        self.intervals: Dict[Tuple[int, str], cp_model.IntervalVar] = {}

    def add_instruction(self, instruction: MSInstruction):
        self.instructions.append(instruction)
        for parent_edge in instruction.parent_edges:
            self.child_edges[parent_edge.source.name] = self.child_edges.get(parent_edge.source.name, []) + [parent_edge]

    def solve(self, max_cycles: int, depth: int, resource_counts: Dict[str, int], use_verbose: bool = False) -> MSScheduleResult:
        model = cp_model.CpModel()
        self.intervals = {}

        for current_depth in range(depth):
            for instruction in self.instructions:
                instruction_start = model.new_int_var(
                    0,
                    max_cycles - 1 - instruction.latency,
                    f"start_{instruction.name}_{current_depth}",
                )
                instruction_interval = model.new_interval_var(
                    instruction_start,
                    instruction.latency,
                    instruction_start + instruction.latency,
                    f"interval_{instruction.name}_{current_depth}",
                )
                self.intervals[(current_depth, instruction.name)] = instruction_interval

        for current_depth in range(depth):
            for instruction in self.instructions:
                for parent_edge in instruction.parent_edges:
                    parent_depth = current_depth - parent_edge.distance
                    if parent_depth >= 0:
                        child_start = self.intervals[(current_depth, instruction.name)].start_expr()
                        parent_end = self.intervals[(parent_depth, parent_edge.source.name)].end_expr()
                        model.add(
                            cast(Any, child_start) >= cast(Any, parent_end)
                        )

        for resource in resource_counts.keys():
            resource_instructions: List[cp_model.IntervalVar] = []
            for current_depth in range(depth):
                for instruction in self.instructions:
                    if resource in instruction.resource_keys:
                        resource_instructions.append(self.intervals[(current_depth, instruction.name)])
            if resource_instructions:
                model.add_cumulative(
                    resource_instructions,
                    [1 for _ in range(len(resource_instructions))],
                    resource_counts[resource],
                )

        ii = model.new_int_var(1, max_cycles - 1, "ii")
        for instruction in self.instructions:
            for i in range(0, depth - 1):
                model.add(
                    self.intervals[(i + 1, instruction.name)].start_expr()
                    == self.intervals[(i, instruction.name)].start_expr() + ii  # type: ignore[operator]
                )

        makespan = model.new_int_var(0, max_cycles - 1, "makespan")
        model.add_max_equality(makespan, [interval.end_expr() for interval in self.intervals.values()])  # type: ignore[arg-type]
        model.minimize(makespan)

        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = use_verbose
        status = solver.solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise ValueError("Modulo scheduler pass cannot be applied. Please remove this pass.")

        ii_value = solver.value(ii)
        if ii_value <= 0:
            raise ValueError("Scheduler produced non-positive II.")

        kernel_start: Dict[str, int] = {
            instruction.name: solver.value(self.intervals[(0, instruction.name)].start_expr())  # type: ignore[arg-type]
            for instruction in self.instructions
        }
        stage_of: Dict[str, int] = {
            instruction.name: kernel_start[instruction.name] // ii_value
            for instruction in self.instructions
        }
        num_stages = max(stage_of.values()) + 1

        prologue: List[List[Tuple[int, MSInstruction]]] = []
        for i in range(num_stages - 1):
            step: List[Tuple[int, MSInstruction]] = []
            for instruction in self.instructions:
                inst_stage = stage_of[instruction.name]
                if inst_stage <= i:
                    step.append((i - inst_stage, instruction))
            prologue.append(step)

        steady_state: List[List[Tuple[int, MSInstruction]]] = []
        for k in range(num_stages - 1, min(depth, num_stages - 1 + num_stages)):
            step: List[Tuple[int, MSInstruction]] = []
            for instruction in self.instructions:
                step.append((k - stage_of[instruction.name], instruction))
            steady_state.append(step)

        epilogue: List[List[Tuple[int, MSInstruction]]] = []
        for i in range(1, num_stages):
            step: List[Tuple[int, MSInstruction]] = []
            for instruction in self.instructions:
                inst_stage = stage_of[instruction.name]
                if inst_stage >= i:
                    step.append((inst_stage - 1, instruction))
            epilogue.append(step)

        return MSScheduleResult(
            prologue=prologue,
            steady_state=steady_state,
            epilogue=epilogue,
            ii_value=ii_value,
            num_stages=num_stages,
            stage_of=stage_of,
        )


def _vars_in_expr(expr: Expr) -> List[Var]:
    if isinstance(expr, Var):
        return [expr]
    if isinstance(expr, BinaryOp):
        return _vars_in_expr(expr.a) + _vars_in_expr(expr.b)
    if isinstance(expr, AssignExpr):
        return _vars_in_expr(expr.lhs) + _vars_in_expr(expr.rhs)
    if isinstance(expr, OpCall):
        vars_found: List[Var] = []
        for inp in expr.inputs:
            vars_found.extend(_vars_in_expr(inp))
        return vars_found
    if isinstance(expr, Coord):
        vars_found: List[Var] = []
        for elem in expr.elements:
            vars_found.extend(_vars_in_expr(elem))
        return vars_found
    if isinstance(expr, SizeBytesExpr):
        return _vars_in_expr(expr.expr)
    return []


def _vars_in_stmt(stmt: Stmt) -> List[Var]:
    if isinstance(stmt, TileLoadOp):
        # TileLoadOp writes to tile; only source/coords are true read dependencies.
        return [stmt.source] + _vars_in_expr(stmt.coord)
    if isinstance(stmt, ExprStmt):
        return _vars_in_expr(stmt.call)
    if isinstance(stmt, SeqStmt):
        vars_found: List[Var] = []
        for child in stmt.stmts:
            vars_found.extend(_vars_in_stmt(child))
        return vars_found
    if isinstance(stmt, IfStmt):
        vars_found = _vars_in_expr(stmt.cond)
        vars_found.extend(_vars_in_stmt(stmt.then_stmt))
        if stmt.else_stmt is not None:
            vars_found.extend(_vars_in_stmt(stmt.else_stmt))
        return vars_found
    if isinstance(stmt, ForStmt):
        vars_found = _vars_in_expr(stmt.init)
        vars_found.extend(_vars_in_expr(stmt.cond))
        vars_found.extend(_vars_in_expr(stmt.step))
        vars_found.extend(_vars_in_stmt(stmt.body))
        return vars_found
    return []


def _op_calls_in_stmt(stmt: Stmt) -> List[OpCall]:
    if isinstance(stmt, ExprStmt) and isinstance(stmt.call, OpCall):
        return [stmt.call]
    if isinstance(stmt, SeqStmt):
        op_calls: List[OpCall] = []
        for child in stmt.stmts:
            op_calls.extend(_op_calls_in_stmt(child))
        return op_calls
    if isinstance(stmt, IfStmt):
        op_calls = _op_calls_in_stmt(stmt.then_stmt)
        if stmt.else_stmt is not None:
            op_calls.extend(_op_calls_in_stmt(stmt.else_stmt))
        return op_calls
    return []


def _side_effect_outputs(stmt: Stmt) -> List[Var]:
    if isinstance(stmt, TileLoadOp):
        return [stmt.output] if stmt.output is not None else []

    outputs: List[Var] = []
    for call in _op_calls_in_stmt(stmt):
        if call.function_name in {"warpgroup::mma_AB", "warpgroup::zero"} and call.inputs:
            first_arg = call.inputs[0]
            if isinstance(first_arg, Var):
                outputs.append(first_arg)
    return outputs


def _instruction_signature(stmt: Stmt) -> Optional[Tuple[Union[str, Tuple[str, ...]], int]]:
    if isinstance(stmt, TileLoadOp):
        return ("tma", 12)

    call_names = {call.function_name for call in _op_calls_in_stmt(stmt)}
    if "warpgroup::mma_AB" in call_names:
        return ("tc", 4)

    return None


def _dedup_vars(vars_list: Iterable[Var]) -> List[Var]:
    out: List[Var] = []
    seen: Set[str] = set()
    for var in vars_list:
        key = str(var)
        if key in seen:
            continue
        seen.add(key)
        out.append(var)
    return out


def build_instruction_manager(loop_stmt: ForStmt) -> MSInstructionManager:
    manager = MSInstructionManager()
    producer_of: Dict[Var, MSInstruction] = {}

    for idx, stmt in enumerate(loop_stmt.body.stmts):
        signature = _instruction_signature(stmt)
        if signature is None:
            continue

        resource, latency = signature
        side_outputs = _side_effect_outputs(stmt)
        output_var = side_outputs[0] if side_outputs else None

        instruction = MSInstruction(
            resource=resource,
            latency=latency,
            name=f"{stmt.name}_{idx}",
            output_var=output_var,
            parent_edges=[],
            instruction=stmt,
            input_vars=_dedup_vars(_vars_in_stmt(stmt)),
        )

        manager.add_instruction(instruction)
        if output_var is not None:
            producer_of[output_var] = instruction

    if not manager.instructions:
        raise ValueError("No schedulable instructions found in loop body.")

    yield_vars = set(loop_stmt.yields)
    for instruction in manager.instructions:
        for input_var in instruction.input_vars:
            parent = producer_of.get(input_var)
            if parent is None:
                continue

            distance = 1 if input_var in yield_vars else 0
            edge = MSInstructionEdge(source=parent, dest=instruction, distance=distance)
            instruction.parent_edges.append(edge)
            manager.child_edges[parent.name] = manager.child_edges.get(parent.name, []) + [edge]

    return manager


def compute_variable_buffer_sizes(manager: MSInstructionManager, stage_of: Dict[str, int]) -> Dict[Var, int]:
    variable_buffer_sizes: Dict[Var, int] = {}

    for instruction in manager.instructions:
        if instruction.output_var is not None:
            variable_buffer_sizes.setdefault(instruction.output_var, 1)

    for instruction in manager.instructions:
        for parent_edge in instruction.parent_edges:
            source_var = parent_edge.source.output_var
            if source_var is None:
                continue
            live_span = stage_of[instruction.name] - stage_of[parent_edge.source.name] + 1
            variable_buffer_sizes[source_var] = max(variable_buffer_sizes.get(source_var, 1), live_span)

    return variable_buffer_sizes


def build_buffered_variables(variable_buffer_sizes: Dict[Var, int]) -> Dict[Var, List[Var]]:
    buffered_variables: Dict[Var, List[Var]] = {}

    for var, size in variable_buffer_sizes.items():
        buffered_variables[var] = []
        for buf_id in range(size):
            clone = deepcopy(var)
            clone.name = f"{var.name}_{buf_id}"
            if isinstance(clone, Tile):
                clone.var.name = clone.name
            buffered_variables[var].append(clone)

    return buffered_variables


def _replace_tile_idx(base_idx: Var, iter_delta: int, mode: str) -> Expr:
    if mode == "prologue":
        return RawExpr(iter_delta)
    if mode == "steady":
        return BinaryOp(base_idx, RawExpr(iter_delta), "+")
    if mode == "epilogue":
        return BinaryOp(base_idx, RawExpr(iter_delta), "-")
    raise ValueError(f"unknown replacement mode: {mode}")


def _iter_guard(loop_stmt: ForStmt, tile_idx: Var, iter_delta: int) -> Expr:
    idx_expr = BinaryOp(tile_idx, RawExpr(iter_delta), "+")
    return loop_stmt.cond.replace_vars({tile_idx: idx_expr})


def _build_schedule_stmts(
    schedule_steps: List[List[Tuple[int, MSInstruction]]],
    tile_idx: Var,
    buffered_variables: Dict[Var, List[Var]],
    loop_stmt: ForStmt,
    mode: str,
) -> List[Stmt]:
    output_stmts: List[Stmt] = []
    for stage_entries in schedule_steps:
        for iter_delta, instruction in stage_entries:
            variable_replacements: Dict[Var, Expr] = {tile_idx: _replace_tile_idx(tile_idx, iter_delta, mode)}
            for var, buffers in buffered_variables.items():
                idx = iter_delta % len(buffers)
                variable_replacements[var] = buffers[idx]

            replaced_stmt = instruction.instruction.replace_vars(variable_replacements)
            if mode == "steady":
                replaced_stmt = IfStmt(_iter_guard(loop_stmt, tile_idx, iter_delta), replaced_stmt)
            output_stmts.append(replaced_stmt)
    return output_stmts


def _declared_var(stmt: Stmt) -> Optional[Var]:
    if isinstance(stmt, DeclStmt):
        return stmt.var
    if isinstance(stmt, SeqStmt) and len(stmt.stmts) == 1 and isinstance(stmt.stmts[0], DeclStmt):
        return stmt.stmts[0].var
    return None


def _flatten_stmt(stmt: Stmt) -> List[Stmt]:
    if isinstance(stmt, SeqStmt):
        return list(stmt.stmts)
    return [stmt]


def _find_loop_index_var(loop_stmt: ForStmt) -> Var:
    if isinstance(loop_stmt.init, AssignExpr) and isinstance(loop_stmt.init.lhs, Var):
        return loop_stmt.init.lhs
    raise ValueError("ForStmt init must be AssignExpr to a Var for modulo scheduling.")


def pipeline_loop_stmt(
    loop_stmt: ForStmt,
    max_cycles: int,
    depth: int,
    resource_counts: Dict[str, int],
    use_verbose: bool = False,
) -> Tuple[SeqStmt, Dict[Var, List[Var]], MSScheduleResult]:
    manager = build_instruction_manager(loop_stmt)
    schedule = manager.solve(max_cycles, depth, resource_counts, use_verbose=use_verbose)

    variable_buffer_sizes = compute_variable_buffer_sizes(manager, schedule.stage_of)
    buffered_variables = build_buffered_variables(variable_buffer_sizes)

    tile_idx = _find_loop_index_var(loop_stmt)

    prologue_stmts = _build_schedule_stmts(
        schedule.prologue,
        tile_idx,
        buffered_variables,
        loop_stmt,
        mode="prologue",
    )
    steady_state_stmts = _build_schedule_stmts(
        schedule.steady_state,
        tile_idx,
        buffered_variables,
        loop_stmt,
        mode="steady",
    )
    epilogue_stmts = _build_schedule_stmts(
        schedule.epilogue,
        tile_idx,
        buffered_variables,
        loop_stmt,
        mode="epilogue",
    )

    step_expr = AssignExpr(tile_idx, BinaryOp(tile_idx, RawExpr(schedule.num_stages), "+"))
    for_inputs = _dedup_vars([tile_idx] + [buf for buf_list in buffered_variables.values() for buf in buf_list])
    pipelined_loop = ForStmt(
        loop_stmt.init,
        loop_stmt.cond,
        step_expr,
        SeqStmt(steady_state_stmts),
        inputs=for_inputs,
        yields=[],
    )

    return SeqStmt(prologue_stmts + [pipelined_loop] + epilogue_stmts), buffered_variables, schedule


def apply_modulo_scheduling_to_seqstmt(
    stmt: SeqStmt,
    max_cycles: int,
    depth: int,
    resource_counts: Dict[str, int],
    use_verbose: bool = False,
) -> SeqStmt:
    variable_map: Dict[Var, Expr] = {}
    for_stmt_map: Dict[ForStmt, SeqStmt] = {}
    buffered_decls: Dict[str, List[Stmt]] = {}

    for op in stmt.stmts:
        if not isinstance(op, ForStmt):
            continue

        pipelined_stmt, buffered_variables, _ = pipeline_loop_stmt(
            op,
            max_cycles=max_cycles,
            depth=depth,
            resource_counts=resource_counts,
            use_verbose=use_verbose,
        )
        for_stmt_map[op] = pipelined_stmt

        for original_var, buffers in buffered_variables.items():
            variable_map[original_var] = buffers[0]
            if isinstance(original_var, Tile):
                variable_map[original_var.var] = buffers[0]

            original_decl_name = original_var.var.name if isinstance(original_var, Tile) else original_var.name
            buffered_decls[original_decl_name] = []
            for instance in buffers:
                decl = instance.declare()
                buffered_decls[original_decl_name].extend(_flatten_stmt(decl))

    new_sequence: List[Stmt] = []
    for op in stmt.stmts:
        if isinstance(op, ForStmt) and op in for_stmt_map:
            new_sequence.extend(for_stmt_map[op].stmts)
            continue

        declared = _declared_var(op)
        if declared is not None and declared.name in buffered_decls:
            new_sequence.extend(buffered_decls[declared.name])
            continue

        new_sequence.append(op.replace_vars(variable_map))

    return SeqStmt(new_sequence)


if __name__ == "__main__":
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
    mac_stmt = SeqStmt(
        [
            ExprStmt(OpCall("warpgroup::mma_AB", [c_tile, a_tile, b_tile])),
            ExprStmt(OpCall("warpgroup::mma_async_wait", [])),
        ]
    )

    zero_c_op = ExprStmt(OpCall("warpgroup::zero", [c_tile]))
    store_op = ExprStmt(OpCall("warpgroup::store", [c_tile, Coord([RawExpr(0), RawExpr(0), row, col])]))

    loop = ForStmt(
        AssignExpr(tile_idx, RawExpr(0)),
        BinaryOp(tile_idx, tile_limit, "<"),
        AssignExpr(tile_idx, BinaryOp(tile_idx, RawExpr(1), "+")),
        SeqStmt([load_a_stmt, load_b_stmt, mac_stmt]),
        inputs=[c_tile],
        yields=[c_tile],
    )

    kernel_stmt = SeqStmt(
        [
            a_tile.declare(),
            b_tile.declare(),
            c_tile.declare(),
            zero_c_op,
            loop,
            store_op,
        ]
    )

    kernel_globals = KernelGlobals("globals")
    for shared in [A, B, C, N]:
        kernel_globals.add_var(shared)

    sample_program = Program(kernel_globals, kernel_stmt)
    print("Sample IR (before lowering):")
    print(sample_program.kernel_stmt)

    pipelined_kernel_stmt = apply_modulo_scheduling_to_seqstmt(
        kernel_stmt,
        max_cycles=max_cycles,
        depth=depth,
        resource_counts=resource_counts,
        use_verbose=use_verbose,
    )

    pipelined_program = Program(kernel_globals, pipelined_kernel_stmt)
    print("Pipelined IR (before lowering):")
    print(pipelined_program.kernel_stmt)

    lowered_pipelined_program = lower_program(pipelined_program)
    print("Pipelined program (after lowering):")
    print(lowered_pipelined_program.kernel_stmt)