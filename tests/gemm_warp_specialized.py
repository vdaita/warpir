from warpir import *
from warpir.ops import BuiltinExpr

def build_gemm_kernel() -> Program:
    block_size_val, qsize = 64, 2
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
    
    row = getConst("blockIdx.y")
    col = getConst("blockIdx.x")

    num_tiles = Var("num_tiles", ScalarType("int"))
    warpid = Var("warpid", ScalarType("int"))
    warpgroupid = Var("warpgroupid", ScalarType("int"))
    num_consumers = Var("num_consumers", ScalarType("int"))
    num_consumers_expr = BuiltinExpr("(NUM_THREADS / 128) - 1")
    
    tile = Var("tile", ScalarType("int"))
    p = Var("p", ScalarType("int"))
    qidx = Var("qidx", ScalarType("int"))

    producer_loop = ForStmt(
        AssignStmt(tile, 0),
        tile < num_tiles,
        BuiltinExpr("++tile, ++qidx"),
        SeqStmt([
            IfStmt(BuiltinExpr("qidx == 2"), BuiltinExpr("qidx = 0; p ^= 1;")),
            OpCall("wait", BuiltinExpr(f"empty[qidx]"), p),
            OpCall("tma::expect_bytes", BuiltinExpr(f"full[qidx]"), BuiltinExpr("size_bytes<sub_tile> * 2")),
            OpCall("tma::load_async", BuiltinExpr(f"As[qidx]"), g.A, Coord(0, 0, row, tile), BuiltinExpr("full[qidx]")),
            OpCall("tma::load_async", BuiltinExpr(f"Bs[qidx]"), g.B, Coord(0, 0, tile, col), BuiltinExpr("full[qidx]"))
        ])
    )
    
    producer = SeqStmt([RawStmt("warpgroup::decrease_registers<32>();"),
                        IfStmt(BuiltinExpr("warpgroup::laneid() == 0"),
                               SeqStmt([DeclStmt(p, 0), DeclStmt(qidx, 0), producer_loop]))
                        ])

    C_accum = Var("C_accum", RegTileType(GPUType.fp32, 16, BLOCK_SIZE, RegTileLayout.row_major))
    
    init_arrive = ForStmt(
        BuiltinExpr("int i = 0"),
        BuiltinExpr(f"i < {qsize}"),
        BuiltinExpr("++i"),
        OpCall("arrive", BuiltinExpr("empty[i]"), 1)
    )

    consumer_loop = ForStmt(
        AssignStmt(tile, 0),
        tile < num_tiles,
        BuiltinExpr("++tile, ++qidx"),
        SeqStmt([
            IfStmt(BuiltinExpr("qidx == 2"), BuiltinExpr("qidx = 0; p ^= 1;")),
            OpCall("wait", BuiltinExpr(f"full[qidx]"), p),
            OpCall("warpgroup::mma_AB", C_accum, BuiltinExpr(f"As[qidx]"), BuiltinExpr(f"Bs[qidx]")),
            OpCall("warpgroup::mma_async_wait"),
            IfStmt(BuiltinExpr("warpgroup::laneid() == 0"), OpCall("arrive", BuiltinExpr("empty[qidx]"), 1))
        ])
    )
    consumer = SeqStmt([
        RawStmt("warpgroup::increase_registers<256>();"), DeclStmt(C_accum),
        OpCall("kittens::warp::zero", C_accum),
        IfStmt(BuiltinExpr("warpgroup::laneid() == 0"), init_arrive),
        DeclStmt(p, 0), DeclStmt(qidx, 0),
        consumer_loop, 
        OpCall("warpgroup::store", g.C, C_accum, Coord(0, 0, row, col)),
    ])

    kernel_stmt = SeqStmt([
        SharedAllocStmt("As", sub_tile, count=qsize),
        SharedAllocStmt("Bs", sub_tile, count=qsize),
        DeclStmt(Var("row", ScalarType("int")), row),
        DeclStmt(Var("col", ScalarType("int")), col),
        DeclStmt(num_tiles, (g.N + BLOCK_SIZE - 1) / BLOCK_SIZE),
        DeclStmt(warpid, BuiltinExpr("kittens::warpid()")), 
        DeclStmt(warpgroupid, BuiltinExpr("warpid/4")), 
        DeclStmt(num_consumers, num_consumers_expr),
        BuiltinExpr(f"__shared__ semaphore full[{qsize}], empty[{qsize}];"),
        IfStmt(BuiltinExpr("threadIdx.x == 0"), SeqStmt([
            ForStmt(BuiltinExpr("int i = 0"), BuiltinExpr(f"i < {qsize}"), BuiltinExpr("++i"),
                    SeqStmt([
                        OpCall("init_semaphore", BuiltinExpr("full[i]"), 0, 1),
                        OpCall("init_semaphore", BuiltinExpr("empty[i]"), num_consumers, 0)
                    ]))
        ])),
        RawStmt("__syncthreads();"),
        # Instead of generic dispatch, use an If/Else for warp specialised struct
        IfStmt(BuiltinExpr("warpgroupid == 0"), producer, consumer)
    ])
    return Program(
        input_vars=[Var("A", ScalarType("bf16*")), Var("B", ScalarType("bf16*")), Var("C", ScalarType("bf16*")), Var("N", ScalarType("size_t"))], 
        kernel_vars=g, 
        kernel_stmt=kernel_stmt, 
        constants=constants,
        grid_dims="(N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE",
        block_dims="NUM_THREADS",
        shared_mem="102400",
        launch_name="matmul"
    )

if __name__ == "__main__":
    print(emit_cpp(build_gemm_kernel()))