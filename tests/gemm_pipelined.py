from warpir import *
from warpir.ops import BuiltinExpr

def build_gemm_kernel() -> Program:
    block_size_val = 64
    BLOCK_SIZE = BuiltinExpr("BLOCK_SIZE")
    
    constants = f"""
static constexpr int BLOCK_SIZE = {block_size_val};
"""

    # Define types with programmatic aliases
    sub_tile = SharedTileType(GPUType.bf16, BLOCK_SIZE, BLOCK_SIZE, SharedTileLayout.row_major, alias_name="sub_tile")
    tile_gl = GlobalType(GPUType.bf16, sub_tile, w=1, x=1, y=-1, z=-1, alias_name="tile_gl")

    g = KernelGlobals(
        name="matmul_globals",
        A=tile_gl,
        B=tile_gl,
        C=tile_gl, 
        N=ScalarType("int")
    )
    
    ab_type = sub_tile
    accum_type = RegTileType(GPUType.fp32, 16, BLOCK_SIZE, RegTileLayout.row_major)

    As = Tile("As", ab_type)
    Bs = Tile("Bs", ab_type)
    
    C_accum = Tile("C_accum", accum_type)
    C_accum_cpy = Tile("C_accum_cpy", accum_type)

    row, col = Var("row", ScalarType("int")), Var("col", ScalarType("int"))
    num_tiles = Var("num_tiles", ScalarType("int"))
    
    tic = Var("tic", ScalarType("int"))
    toc = Var("toc", ScalarType("int"))
    
    bar = Symbol("bar")

    tile = Var("tile", ScalarType("int"))
    for_stmt = ForStmt(
        AssignStmt(tile, 0),
        tile < num_tiles,
        BuiltinExpr("++tile, tic^=1, toc^=1"),
        SeqStmt([
            OpCall("wait", bar, tic),
            OpCall("__syncthreads"),
            IfStmt(
                BuiltinExpr("threadIdx.x == 0 && tile+1 < num_tiles"),
                SeqStmt([
                    OpCall("tma::expect_bytes", bar, BuiltinExpr("size_bytes<sub_tile> * 2")),
                    OpCall("tma::load_async", "As[toc]", g.A, Coord(0, 0, row, BuiltinExpr("tile+1")), bar),
                    OpCall("tma::load_async", "Bs[toc]", g.B, Coord(0, 0, BuiltinExpr("tile+1"), col), bar)
                ])
            ),
            OpCall("warpgroup::mma_AB", C_accum, "As[tic]", "Bs[tic]"),
            OpCall("warpgroup::mma_async_wait"),
            OpCall("kittens::warp::add", C_accum_cpy, C_accum_cpy, C_accum),
            OpCall("kittens::warp::zero", C_accum),
            OpCall("__syncthreads")
        ])
    )
    store_stmt = OpCall("warpgroup::store", g.C, C_accum_cpy, Coord(0, 0, row, col))

    return Program(
        input_vars=[], 
        kernel_vars=g, 
        kernel_stmt=SeqStmt([
            SharedAllocStmt("As", ab_type, count=2),
            SharedAllocStmt("Bs", ab_type, count=2),
            DeclStmt(tic, 0),
            DeclStmt(toc, 1),
            C_accum.def_(),
            C_accum_cpy.def_(),
            DeclStmt(row, getConst("blockIdx.y")),
            DeclStmt(col, getConst("blockIdx.x")),
            BuiltinExpr("__shared__ semaphore bar;"),
            IfStmt(
                BuiltinExpr("threadIdx.x == 0"),
                SeqStmt([
                    OpCall("init_semaphore", bar, 0, 1),
                    OpCall("tma::expect_bytes", bar, BuiltinExpr("size_bytes<sub_tile> * 2")),
                    OpCall("tma::load_async", "As[tic]", g.A, Coord(0, 0, row, 0), bar),
                    OpCall("tma::load_async", "Bs[tic]", g.B, Coord(0, 0, 0, col), bar)
                ])
            ),
            OpCall("__syncthreads"),
            OpCall("kittens::warp::zero", C_accum_cpy),
            DeclStmt(num_tiles, (g.N + BLOCK_SIZE - 1) / BLOCK_SIZE),
            DeclStmt(tile),
            for_stmt,
            store_stmt
        ]),
        constants=constants
    )

if __name__ == "__main__":
    print(emit_cpp(build_gemm_kernel()))
