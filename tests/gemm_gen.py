from warpir import (
    Program,
    SeqStmt,
    RawStmt,
    ExprStmt,
    ForStmt,
    IfStmt,
    Tile,
    TileQueue,
    DeclStmt,
    SharedTileType,
    SharedTileLayout,
    GlobalType,
    GPUType,
    Var,
    ScalarType,
    MMAOp,
    StoreOp,
    BlockIdx,
    ThreadIdx,
    WarpId,
    WarpGroupId,
    Symbol,
    Coord,
    ArriveOp,
    format_cpp
)


def build_gemm_kernel() -> Program:
    block_size = 64
    qsize = 2

    sub_tile = SharedTileType(GPUType.bf16, block_size, block_size, SharedTileLayout.row_major)
    tile_gl = GlobalType(GPUType.bf16, sub_tile)

    A = Var("A", tile_gl)
    B = Var("B", tile_gl)
    C = Var("C", tile_gl)
    N = Var("N", ScalarType("int"))
    g = Symbol("g")
    num_tiles = Symbol("num_tiles")

    As = [Tile(f"As_{i}", sub_tile) for i in range(qsize)]
    Bs = [Tile(f"Bs_{i}", sub_tile) for i in range(qsize)]
    q0 = TileQueue("q0", [As[0], Bs[0]])
    q1 = TileQueue("q1", [As[1], Bs[1]])

    def on_qidx(stmt0, stmt1):
        return IfStmt("qidx == 0", stmt0, stmt1)

    row = Var("row", ScalarType("int"))
    col = Var("col", ScalarType("int"))
    tile = Var("tile", ScalarType("int"))
    qidx = Var("qidx", ScalarType("int"))
    p = Var("p", ScalarType("int"))

    producer_loop = ForStmt(
        init="int tile = 0, qidx = 0",
        cond=f"{tile} < {num_tiles}",
        step="++tile, ++qidx",
        body=SeqStmt([
            IfStmt("qidx == QSIZE", RawStmt("qidx = 0; p ^= 1;")),
            on_qidx(
                q0.produce(
                    [(As[0], g.A, Coord(0, 0, row, tile)), (Bs[0], g.B, Coord(0, 0, tile, col))],
                    phase=p,
                ),
                q1.produce(
                    [(As[1], g.A, Coord(0, 0, row, tile)), (Bs[1], g.B, Coord(0, 0, tile, col))],
                    phase=p,
                ),
            ),
        ]),
    )

    producer = SeqStmt([
        RawStmt("warpgroup::decrease_registers<32>();"),
        IfStmt("warpgroup::laneid() == 0", SeqStmt([
            RawStmt("int p = 0;"),
            producer_loop,
        ])),
    ])

    consumer_loop = ForStmt(
        init="int tile = 0, qidx = 0",
        cond=f"{tile} < {num_tiles}",
        step="++tile, ++qidx",
        body=SeqStmt([
            IfStmt("qidx == QSIZE", RawStmt("qidx = 0; p ^= 1;")),
            on_qidx(
                q0.consume(p, SeqStmt([
                    ExprStmt(MMAOp("C_accum", "As_0", "Bs_0")),
                    RawStmt("warpgroup::mma_async_wait();"),
                ]), arrive=False),
                q1.consume(p, SeqStmt([
                    ExprStmt(MMAOp("C_accum", "As_1", "Bs_1")),
                    RawStmt("warpgroup::mma_async_wait();"),
                ]), arrive=False),
            ),
            IfStmt(
                "warpgroup::laneid() == 0",
                on_qidx(
                    ExprStmt(ArriveOp(q0.empty_sem, 1)),
                    ExprStmt(ArriveOp(q1.empty_sem, 1)),
                ),
            ),
        ]),
    )

    consumer = SeqStmt([
        RawStmt("warpgroup::increase_registers<256>();"),
        RawStmt("rt_fl<16, BLOCK_SIZE> C_accum;"),
        RawStmt("kittens::warp::zero(C_accum);"),
        IfStmt("warpgroup::laneid() == 0", SeqStmt([
            ExprStmt(ArriveOp(q0.empty_sem, 1)),
            ExprStmt(ArriveOp(q1.empty_sem, 1)),
        ])),
        RawStmt("int p = 0;"),
        consumer_loop,
        ExprStmt(StoreOp("g.C", "C_accum", "{0, 0, row, col}")),
    ])

    kernel_stmt = SeqStmt([
        RawStmt(f"static constexpr int BLOCK_SIZE = {block_size};"),
        RawStmt(f"static constexpr int QSIZE = {qsize};"),
        RawStmt("static constexpr int NUM_THREADS = 8 * kittens::WARP_THREADS;"),
        RawStmt("extern __shared__ alignment_dummy __shm[];"),
        RawStmt("shared_allocator al((int*)&__shm[0]);"),
        DeclStmt(row, BlockIdx.y),
        DeclStmt(col, BlockIdx.x),
        DeclStmt(Var("num_tiles", ScalarType("int")), f"({g}.N + BLOCK_SIZE - 1) / BLOCK_SIZE"),
        DeclStmt(Var("warpid", ScalarType("int")), WarpId),
        DeclStmt(Var("warpgroupid", ScalarType("int")), WarpGroupId),
        DeclStmt(Var("num_consumers", ScalarType("int")), "(NUM_THREADS / 128) - 1"),
        *[t.declare() for t in As],
        *[t.declare() for t in Bs],
        q0.declare_semaphores(),
        q1.declare_semaphores(),
        IfStmt(str(ThreadIdx.x) + " == 0", SeqStmt([
            q0.init_semaphores(0, "num_consumers"),
            q1.init_semaphores(0, "num_consumers"),
        ])),
        RawStmt("__syncthreads();"),
        IfStmt("warpgroupid == 0", producer, consumer),
    ])

    return Program(
        input_vars=[],
        kernel_vars=[A, B, C, N],
        kernel_stmt=kernel_stmt,
    )


if __name__ == "__main__":
    print(format_cpp(str(build_gemm_kernel())))
