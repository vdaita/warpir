from warpir import *

def build_gemm_kernel() -> Program:
    block_size, qsize = 64, 2
    ab_type = SharedTileType(GPUType.bf16, block_size, block_size, SharedTileLayout.row_major)
    accum_type = RegTileType(GPUType.fp32, 16, block_size, RegTileLayout.row_major)

    g = KernelGlobals(A=GlobalType(GPUType.bf16, ab_type), B=GlobalType(GPUType.bf16, ab_type),
                      C=GlobalType(GPUType.bf16, ab_type), N=ScalarType("int"))
    As, Bs = [Tile(f"As_{i}", ab_type) for i in range(2)], [Tile(f"Bs_{i}", ab_type) for i in range(2)]
    C_accum = Tile("C_accum", accum_type)
    
    row, col = Var("row", ScalarType("int")), Var("col", ScalarType("int"))
    num_tiles = Var("num_tiles", ScalarType("int"))

    k_index = Var("k_index", ScalarType("int"))
    for_stmt = ForStmt(
        AssignStmt(k_index, 0),
        k_index < num_tiles,
        AssignStmt(k_index, k_index + 1),
        SeqStmt([
            g.A.load(As[0], Coord(0, 0, row, k_index)),
            g.B.load(Bs[0], Coord(0, 0, k_index, col)),
            g.A.load(As[1], Coord(0, 0, row, k_index + 1)),
            g.B.load(Bs[1], Coord(0, 0, k_index + 1, col)),
            MMAOp(C_accum, As[0], Bs[0]),
            MMAWaitOp(),
            MMAOp(C_accum, As[1], Bs[1]),
            MMAWaitOp()
        ])
    )
    store_stmt = g.C.store(C_accum, Coord(0, 0, row, col))

    return Program(input_vars=[], kernel_vars=g, kernel_stmt=SeqStmt([
        DeclStmt(row, getConst("blockIdx.y")),
        DeclStmt(col, getConst("blockIdx.x")),
        DeclStmt(num_tiles, (g.N + block_size - 1) / block_size),
        *[Ai.def_() for Ai in As],
        *[Bi.def_() for Bi in Bs],
        C_accum.def_(),
        OpCall("kittens::warp::zero", C_accum),
        DeclStmt(k_index),
        for_stmt,
        store_stmt
    ]))

if __name__ == "__main__":
    print(emit_cpp(build_gemm_kernel()))
