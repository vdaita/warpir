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

def stringify_dict(in_dict):
    return {str(k): str(v) for k, v in variable_replacements.items()}

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
            prologue: List[List[Tuple[int, MSInstruction]]] = []
            for i in range(num_stages - 1):
                step: List[Tuple[int, MSInstruction]] = []
                for instruction in self.instructions:
                    if stage_of[instruction.name] <= i:
                        step.append((i - stage_of[instruction.name], instruction))
                prologue.append(step)

            # steady state: grouped by k, each entry is list of (inst_name, buffer_offset)
            steady_state: List[List[Tuple[int, MSInstruction]]] = []
            for k in range(num_stages - 1, min(depth, num_stages - 1 + num_stages)):
                step = []
                for instruction in self.instructions:
                    step.append(((k - stage_of[instruction.name]), instruction))
                steady_state.append(step)

            # epilogue: list of drain steps, each is list of (inst_name, buffer_offset)
            epilogue: List[List[Tuple[int, MSInstruction]]] = []
            for i in range(1, num_stages):
                step = []
                for instruction in self.instructions:
                    if stage_of[instruction.name] >= i:
                        step.append((stage_of[instruction.name] - 1, instruction))
                epilogue.append(step)

            return prologue, steady_state, epilogue, ii_value, num_stages, stage_of

        else:
            print("No solution found.")
            raise ValueError("Modulo scheduler pass cannot be applied. Please remove this pass.")

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

    kernel_globals = KernelGlobals("globals")
    for shared in [A, B, C, N]:
        kernel_globals.add_var(shared)

    loop_body = SeqStmt([load_a_stmt, load_b_stmt, mac_stmt])
    tile_loop = ForStmt(
        AssignExpr(tile_idx, zero),
        BinaryOp(tile_idx, tile_limit, "<"),
        AssignExpr(tile_idx, BinaryOp(tile_idx, RawExpr(1), "+")),
        loop_body,
        inputs=[tile_idx, c_tile.var],
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

    buffered_variables: Dict[Var, List[Var]] = {}
    for var, size in variable_buffer_sizes.items():
        buffered_variables[var] = []
        for buf_id in range(size):
            clone_var = deepcopy(var)
            clone_var.name = f"{var.name}_{buf_id}"
            buffered_variables[var].append(clone_var)

    prologue_stmts: List[Stmt] = []
    for stage_idx, stage_entries in enumerate(prologue):
        for iter_delta, instruction in stage_entries:
            # all of the input variables that go into this are handled b
            variable_replacements: Dict[Var, Expr] = {
                # tile_idx: BinaryOp(tile_idx, RawExpr(iter_delta), "+"),
                tile_idx: RawExpr(stage_idx) # right?
            }
            for var in buffered_variables:
                variable_replacements[var] = buffered_variables[var][iter_delta % variable_buffer_sizes[var]]
        
            print("In prologue instruction ", instruction.instruction, " with stage ", stage_idx, " and iter_delta ", iter_delta, " using following map: ", stringify_dict(variable_replacements))

            prologue_stmts.append(
                instruction.instruction.replace_vars(variable_replacements)
            )
    
    print("=========")

    steady_state_stmts: List[Stmt] = []
    for stage_idx, stage_entries in enumerate(steady_state):
        for iter_delta, instruction in stage_entries:
            variable_replacements: Dict[Var, Expr] = {
                tile_idx: BinaryOp(tile_idx, RawExpr(iter_delta), "+")
            }
            for var in buffered_variables:
                variable_replacements[var] = buffered_variables[var][iter_delta % variable_buffer_sizes[var]]
            if_wrapper = IfStmt(BinaryOp(BinaryOp(tile_idx, RawExpr(iter_delta), "+"), N, "<"), instruction.instruction.replace_vars(variable_replacements))
            print("In steady state instruction ", instruction.instruction, " with stage ", stage_idx, " and iter_delta ", iter_delta, " using following map: ", stringify_dict(variable_replacements))
            steady_state_stmts.append(if_wrapper)

    print("========")
    
    epilogue_stmts: List[Stmt] = []
    for stage_idx, stage_entries in enumerate(epilogue):
        for iter_delta, instruction in stage_entries:
            variable_replacements: Dict[Var, Expr] = {
                tile_idx: BinaryOp(tile_idx, RawExpr(iter_delta), "-")
            }
            for var in buffered_variables:
                variable_replacements[var] = buffered_variables[var][iter_delta % variable_buffer_sizes[var]]
            print("In epilogue instruction ", instruction.instruction, " with stage ", stage_idx, " and iter_delta ", iter_delta, " using following map: ", stringify_dict(variable_replacements))
            epilogue_stmts.append(instruction.instruction.replace_vars(variable_replacements))
    
    final_kernel = []
    for stmt in kernel_stmt.stmts:
        if type(stmt) == DeclStmt:
            if stmt.var in buffered_variables:
                for instance in buffered_variables[stmt.var]:
                    final_kernel.append(instance.declare())
        else:
            # figure out if there is a command or somethign that concerns another member
                # will need to deal with this for the zero operation
            pass
    
    final_kernel.extend(prologue_stmts)
    for_inputs = [tile_idx]
    for var in buffered_variables:
        for x in buffered_variables[var]:
            for_inputs.append(x)
    new_for_loop = ForStmt(
        tile_loop.init,
        tile_loop.cond,
        AssignExpr(tile_idx, BinaryOp(tile_idx, RawExpr(num_stages), "+")),
        SeqStmt(steady_state_stmts),
        inputs=for_inputs,
        yields=[]
    )
    final_kernel.append(new_for_loop)
    final_kernel.extend(epilogue_stmts)

    final_kernel_stmt = SeqStmt(final_kernel)

    pipelined_program = Program(kernel_globals, final_kernel_stmt)
    print("Pipelined IR:")
    print(pipelined_program.kernel_stmt)

