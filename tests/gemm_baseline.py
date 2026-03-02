from warpir import *

def build_gemm_kernel() -> Program:
    block_size = 32
    ab_type = SharedTileType(GPUType.bf16, block_size, block_size, SharedTileLayout.row_major)
    # Output accumulates locally to a float tile
    accum_type = RegTileType(GPUType.fp32, block_size, block_size, RegTileLayout.row_major)
    
    # Global types have depth=1, height=1
    g = KernelGlobals(A=GlobalType(GPUType.bf16, ab_type, batch_dim=1, depth_dim=1), 
                      B=GlobalType(GPUType.bf16, ab_type, batch_dim=1, depth_dim=1),
                      C=GlobalType(GPUType.bf16, ab_type, batch_dim=1, depth_dim=1), 
                      N=ScalarType("int"))
    
    As = Tile("As", ab_type)
    Bs = Tile("Bs", ab_type)
    
    A_reg = Tile("A_reg", RegTileType(GPUType.bf16, block_size, block_size, RegTileLayout.row_major))
    B_reg = Tile("B_reg", RegTileType(GPUType.bf16, block_size, block_size, RegTileLayout.row_major))
    B_reg_col = Tile("B_reg_col", RegTileType(GPUType.bf16, block_size, block_size, RegTileLayout.col_major))
    
    C_accum = Tile("C_accum", accum_type)

    row, col = Var("row", ScalarType("int")), Var("col", ScalarType("int"))
    num_tiles = Var("num_tiles", ScalarType("int"))

    tile = Var("tile", ScalarType("int"))
    for_stmt = ForStmt(
        AssignStmt(tile, 0),
        tile < num_tiles,
        BuiltinExpr("++tile"),
        SeqStmt([
            OpCall("kittens::warp::load", As, g.A, Coord(0, 0, row, tile)),
            OpCall("kittens::warp::load", Bs, g.B, Coord(0, 0, tile, col)),
            OpCall("__syncthreads"),
            OpCall("kittens::warp::load", A_reg, As),
            OpCall("kittens::warp::load", B_reg, Bs),
            OpCall("kittens::warp::swap_layout", B_reg_col, B_reg),
            OpCall("__syncthreads"),
            OpCall("kittens::warp::mma_AB", C_accum, A_reg, B_reg_col, C_accum),
            OpCall("__syncthreads")
        ])
    )
    store_stmt = OpCall("kittens::warp::store", g.C, C_accum, Coord(0, 0, row, col))

    return Program(input_vars=[], kernel_vars=g, kernel_stmt=SeqStmt([
        As.def_(),
        Bs.def_(),
        A_reg.def_(),
        B_reg.def_(),
        B_reg_col.def_(),
        C_accum.def_(),
        DeclStmt(col, getConst("blockIdx.x")),
        DeclStmt(row, getConst("blockIdx.y")),
        OpCall("kittens::warp::zero", C_accum),
        DeclStmt(num_tiles, (g.N + block_size - 1) / block_size),
        DeclStmt(tile),
        for_stmt,
        store_stmt
    ]))

if __name__ == "__main__":
    print(emit_cpp(build_gemm_kernel()))
