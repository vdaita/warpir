from warpir import *
from warpir.ops import BuiltinExpr

def build_gemm_kernel() -> Program:
    block_size_val = 32
    BLOCK_SIZE = BuiltinExpr("BLOCK_SIZE")
    
    constants = f"""
static constexpr int BLOCK_SIZE = {block_size_val};
static constexpr int NUM_WORKERS =  (1);
static constexpr int NUM_THREADS = (NUM_WORKERS*kittens::WARP_THREADS);
"""

    # Define types with programmatic aliases
    sub_tile = SharedTileType(GPUType.bf16, BLOCK_SIZE, BLOCK_SIZE, SharedTileLayout.row_major, alias_name="sub_tile")
    tile_gl = GlobalType(GPUType.bf16, sub_tile, w=1, x=1, y=-1, z=-1, alias_name="tile_gl")

    # Use aliases in KernelGlobals
    g = KernelGlobals(
        name="kernel_globals",
        A=tile_gl,
        B=tile_gl,
        C=tile_gl,
        N=ScalarType("int")
    )
    
    ab_type = sub_tile
    # Output accumulates locally to a float tile
    accum_type = RegTileType(GPUType.fp32, BLOCK_SIZE, BLOCK_SIZE, RegTileLayout.row_major)
    
    As = Tile("As", ab_type)
    Bs = Tile("Bs", ab_type)
    
    A_reg = Tile("A_reg", RegTileType(GPUType.bf16, BLOCK_SIZE, BLOCK_SIZE, RegTileLayout.row_major))
    B_reg = Tile("B_reg", RegTileType(GPUType.bf16, BLOCK_SIZE, BLOCK_SIZE, RegTileLayout.row_major))
    B_reg_col = Tile("B_reg_col", RegTileType(GPUType.bf16, BLOCK_SIZE, BLOCK_SIZE, RegTileLayout.col_major))
    
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

    launch_code = """
// launch kernel
void matmul(bf16* A, bf16* B, bf16* C, size_t N) { 

    // global pointers
    using a_gl = kernel_globals::tile_gl;
    using b_gl = kernel_globals::tile_gl; 
    using c_gl = kernel_globals::tile_gl;
    a_gl  a_arg{A, nullptr, nullptr, (int)N, (int)N};
    b_gl  b_arg{B, nullptr, nullptr, (int)N, (int)N};
    c_gl  c_arg{C, nullptr, nullptr, (int)N, (int)N};
    kernel_globals g{a_arg, b_arg, c_arg, (int)N}; 

    // launch
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);  // Watch out for requesting too many!
    unsigned long mem_size = 100000;
    cudaDeviceSynchronize();
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    kernel<<<blocks, NUM_THREADS, mem_size>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();
}
"""

    return Program(
        input_vars=[], 
        kernel_vars=g, 
        kernel_stmt=SeqStmt([
            As.def_(),
            Bs.def_(),
            A_reg.def_(),
            B_reg.def_(),
            B_reg_col.def_(),
            C_accum.def_(),
            DeclStmt(col, getConst("blockIdx.x")),
            DeclStmt(row, getConst("blockIdx.y")),
            OpCall("kittens::warp::zero", C_accum),
            DeclStmt(num_tiles, (g.N + BLOCK_SIZE - 1) / BLOCK_SIZE),
            DeclStmt(tile),
            for_stmt,
            store_stmt
        ]),
        constants=constants,
        launch_code=launch_code
    )

if __name__ == "__main__":
    print(emit_cpp(build_gemm_kernel()))
