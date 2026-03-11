from typing import Dict, Optional

from warpir import *
from warpir.lowering import lower_program
from warpir.passes.modulo_scheduler import apply_modulo_scheduling_to_seqstmt


def build_base_gemm_program() -> Program:
    block_size = 64

    shared_tile_type = SharedTileType(
        GPUType.bf16,
        block_size,
        block_size,
        SharedTileLayout.row_major,
    )
    accum_tile_type = RegTileType(
        GPUType.fp32,
        16,
        block_size,
        RegTileLayout.row_major,
    )
    global_type = GlobalType(GPUType.bf16, shared_tile_type, 1, 1, -1, -1)

    A = Var("A", global_type)
    B = Var("B", global_type)
    C = Var("C", global_type)
    N = Var("N", ScalarType("int"))

    kernel_globals = KernelGlobals("globals")
    for var in [A, B, C, N]:
        kernel_globals.add_var(var)

    a_tile = Tile("a_tile", shared_tile_type)
    b_tile = Tile("b_tile", shared_tile_type)
    c_accum = Tile("c_accum", accum_tile_type)

    # Use one-manager-per-tile so async ops each have a single clear output tile.
    lm_a = TileGroup("lm_a", [a_tile], num_consumers=1)
    lm_b = TileGroup("lm_b", [b_tile], num_consumers=1)

    tile = Var("tile", ScalarType("int"))
    zero = RawExpr(0)
    one = RawExpr(1)
    row = RawExpr("blockIdx.y")
    col = RawExpr("blockIdx.x")
    num_tiles = BinaryOp(BinaryOp(N, RawExpr(block_size - 1), "+"), RawExpr(block_size), "/")

    loop_body = SeqStmt(
        [
            lm_a.wait_empty(Level.block),
            lm_b.wait_empty(Level.block),
            lm_a.async_load_global(
                [MemLoad(source=A, dest=a_tile, coord=Coord([zero, zero, row, tile]))]
            ),
            lm_b.async_load_global(
                [MemLoad(source=B, dest=b_tile, coord=Coord([zero, zero, tile, col]))]
            ),
            lm_a.wait_full(Level.block),
            lm_b.wait_full(Level.block),
            SeqStmt([
                OpCall("warpgroup::mma_AB", [c_accum, a_tile, b_tile]).to_stmt(),
                OpCall("warpgroup::mma_async_wait", []).to_stmt(),
            ]),
            lane0_if(ExprStmt(OpCall("arrive", [lm_a.empty_sem, one]))),
            lane0_if(ExprStmt(OpCall("arrive", [lm_b.empty_sem, one]))),
        ]
    )

    tile_loop = ForStmt(
        AssignExpr(tile, zero),
        BinaryOp(tile, num_tiles, "<"),
        AssignExpr(tile, BinaryOp(tile, one, "+")),
        loop_body,
        inputs=[tile, c_accum, a_tile, b_tile],
        yields=[c_accum],
    )

    kernel_stmt = SeqStmt(
        [
            a_tile.declare(),
            b_tile.declare(),
            c_accum.declare(),
            tile.declare(),
            lm_a.initialize(),
            lm_b.initialize(),
            lm_a.arrive_empty(),
            lm_b.arrive_empty(),
            OpCall("kittens::warp::zero", [c_accum]).to_stmt(),
            tile_loop,
            c_accum.warpgroup_store_global(C, Coord([zero, zero, row, col])),
        ]
    )

    return Program(kernel_vars=kernel_globals, kernel_stmt=kernel_stmt)


def build_autopipelined_program(
    max_cycles: int = 128,
    depth: int = 8,
    resource_counts: Optional[Dict[str, int]] = None,
) -> Program:
    if resource_counts is None:
        resource_counts = {"tma": 2, "tc": 1, "barrier": 1}

    base_program = build_base_gemm_program()
    if not isinstance(base_program.kernel_stmt, SeqStmt):
        raise TypeError("Expected SeqStmt kernel body for modulo scheduling.")
    pipelined_stmt = apply_modulo_scheduling_to_seqstmt(
        base_program.kernel_stmt,
        max_cycles=max_cycles,
        depth=depth,
        resource_counts=resource_counts,
        use_verbose=False,
    )
    pipelined_program = Program(base_program.kernel_vars, pipelined_stmt)
    return lower_program(pipelined_program)


if __name__ == "__main__":
    base = build_base_gemm_program()
    print("=== Base Program (IR) ===")
    print(base.kernel_stmt)
    print()

    auto = build_autopipelined_program()
    print("=== Auto-Pipelined Program (Lowered IR) ===")
    print(auto.kernel_stmt)
    print()

    print("=== Auto-Pipelined Program (Emitted C++) ===")
    print(emit_cpp(auto))
