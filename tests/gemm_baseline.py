from warpir import *

def build_gemm_kernel() -> Program:
    BLOCK_SIZE = 32

    shared_tile_type = SharedTileType(GPUType.bf16, BLOCK_SIZE, BLOCK_SIZE, SharedTileLayout.row_major)
    As = Tile("As", shared_tile_type)
    Bs = Tile("Bs", shared_tile_type)

    global_type = GlobalType(GPUType.bf16, shared_tile_type, 1, 1, -1, -1)
    A = Var("A", global_type)
    B = Var("B", global_type)
    C = Var("C", global_type)
    N = Var("N", ScalarType("int"))

    kernel_globals = KernelGlobals("globals")
    for var in [A, B, C, N]:
        kernel_globals.add_var(var)

    A_reg = Tile("A_reg", RegTileType(GPUType.bf16, BLOCK_SIZE, BLOCK_SIZE, RegTileLayout.row_major))
    B_reg = Tile("B_reg", RegTileType(GPUType.bf16, BLOCK_SIZE, BLOCK_SIZE, RegTileLayout.row_major))
    B_reg_col = Tile("B_reg_col", RegTileType(GPUType.bf16, BLOCK_SIZE, BLOCK_SIZE, RegTileLayout.col_major))
    C_accum = Tile("C_accum", RegTileType(GPUType.fp32, BLOCK_SIZE, BLOCK_SIZE, RegTileLayout.row_major))

    zero_op_stmt = OpCall("kittens::warp::zero", [C_accum]).to_stmt()
    col = RawExpr("blockIdx.x")
    row = RawExpr("blockIdx.y")

    num_tiles = BinaryOp(BinaryOp(BinaryOp(N, RawExpr(BLOCK_SIZE), "+"), RawExpr(1), "+"), RawExpr(BLOCK_SIZE), "/")

    tile = Var("tile", ScalarType("int"))

    body = SeqStmt([
        As.load_global(A, Coord([RawExpr(0), RawExpr(0), row, tile])),
        Bs.load_global(B, Coord([RawExpr(0), RawExpr(0), row, tile])),
        A_reg.load_shared(As),
        B_reg.load_shared(Bs),
        OpCall("kittens::warp::swap_layout", [B_reg_col, B_reg]).to_stmt(),
        OpCall("__syncthreads", []).to_stmt(),
        OpCall("kittens::warp::mma_AB", [C_accum, A_reg, B_reg_col, C_accum]).to_stmt(),
        OpCall("__syncthreads", []).to_stmt(),
    ])
    loop = ForStmt(
        AssignExpr(tile, RawExpr("0")),
        BinaryOp(tile, num_tiles, "<"),
        AssignExpr(tile, BinaryOp(tile, RawExpr("1"), "+")),
        body
    )

    store_stmt = C_accum.warp_store_global(C, Coord([RawExpr(0), RawExpr(0), row, col]))
    program_body = SeqStmt(
        [var.declare() for var in [As, Bs, A_reg, B_reg, B_reg_col, C_accum, tile]] +
        [
            zero_op_stmt,
            loop,
            store_stmt
        ]
    )

    return Program(
        kernel_vars=kernel_globals,
        kernel_stmt=program_body,
    )


if __name__ == "__main__":
    print(emit_cpp(build_gemm_kernel()))
