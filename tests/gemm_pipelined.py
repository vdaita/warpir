from warpir import *
from warpir.ops import BuiltinExpr

def build_gemm_kernel() -> Program:
    block_size = 64
    ab_type = SharedTileType(GPUType.bf16, block_size, block_size, SharedTileLayout.row_major)
    accum_type = RegTileType(GPUType.fp32, 16, block_size, RegTileLayout.row_major)

    g = KernelGlobals(A=GlobalType(GPUType.bf16, ab_type, batch_dim=1, depth_dim=1),
                      B=GlobalType(GPUType.bf16, ab_type, batch_dim=1, depth_dim=1),
                      C=GlobalType(GPUType.bf16, ab_type, batch_dim=1, depth_dim=1), 
                      N=ScalarType("int"))
    
    # We will just declare these manually for exact array layout matching instead of lists of tiles
    # In kitten it was `st_bf<BLOCK_SIZE,BLOCK_SIZE> (&As)[2] = al.allocate<st_bf<BLOCK_SIZE,BLOCK_SIZE>, 2>();`
    # Let's use SharedAllocStmt count=2.
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
                    OpCall("tma::expect_bytes", bar, BuiltinExpr("size_bytes<typeof(As[0])> + size_bytes<typeof(Bs[0])>")),
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

    return Program(input_vars=[], kernel_vars=g, kernel_stmt=SeqStmt([
        SharedAllocStmt("As", ab_type, count=2),
        SharedAllocStmt("Bs", ab_type, count=2),
        DeclStmt(tic, 0),
        DeclStmt(toc, 1),
        C_accum.def_(),
        C_accum_cpy.def_(),
        DeclStmt(row, getConst("blockIdx.y")),
        DeclStmt(col, getConst("blockIdx.x")),
        DeclStmt(Var("condition", ScalarType("int")), BuiltinExpr("(threadIdx.x == 0 && threadIdx.y == 0 & blockIdx.x == 0)")),
        BuiltinExpr("__shared__ semaphore bar;"),
        IfStmt(
            BuiltinExpr("threadIdx.x == 0"),
            SeqStmt([
                OpCall("init_semaphore", bar, 0, 1),
                OpCall("tma::expect_bytes", bar, BuiltinExpr("size_bytes<typeof(As[0])> + size_bytes<typeof(Bs[0])>")),
                OpCall("tma::load_async", "As[tic]", g.A, Coord(0, 0, row, 0), bar),
                OpCall("tma::load_async", "Bs[tic]", g.B, Coord(0, 0, 0, col), bar)
            ])
        ),
        OpCall("__syncthreads"),
        OpCall("kittens::warp::zero", C_accum_cpy),
        DeclStmt(num_tiles, (g.N + block_size - 1) / block_size),
        DeclStmt(tile),
        for_stmt,
        store_stmt
    ]))

if __name__ == "__main__":
    print(emit_cpp(build_gemm_kernel()))
