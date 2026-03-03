from warpir import *

def build_gemm_kernel() -> Program:
    BLOCK_SIZE = 64
    QSIZE = 2
    NUM_WORKERS = 8
    NUM_THREADS = NUM_WORKERS * 32
    NUM_CONSUMERS = (NUM_THREADS // 128) - 1

    shared_tile_type = SharedTileType(GPUType.bf16, BLOCK_SIZE, BLOCK_SIZE, SharedTileLayout.row_major)
    As = [Tile(f"As{i}", shared_tile_type) for i in range(QSIZE)]
    Bs = [Tile(f"Bs{i}", shared_tile_type) for i in range(QSIZE)]

    global_type = GlobalType(GPUType.bf16, shared_tile_type, 1, 1, -1, -1)
    A = Var("A", global_type)
    B = Var("B", global_type)
    C = Var("C", global_type)
    N = Var("N", ScalarType("int"))

    kernel_globals = KernelGlobals("globals")
    for var in [A, B, C, N]:
        kernel_globals.add_var(var)

    lms = [
        MultiTileLoadManager(f"lm{i}", [As[i], Bs[i]], num_consumers=NUM_CONSUMERS)
        for i in range(QSIZE)
    ]

    accum_tile_type = RegTileType(GPUType.fp32, 16, BLOCK_SIZE, RegTileLayout.row_major)
    C_accum = Tile("C_accum", accum_tile_type)

    zero = RawExpr(0)
    col = RawExpr("blockIdx.x")
    row = RawExpr("blockIdx.y")
    num_tiles = BinaryOp(BinaryOp(N, RawExpr(BLOCK_SIZE - 1), "+"), RawExpr(BLOCK_SIZE), "/")
    tile = Var("tile", ScalarType("int"))

    def producer_stmt(i: int) -> Stmt:
        return SeqStmt([
            lms[i].wait_empty(Level.warpgroup),
            lms[i].async_load_global([
                MemLoad(source=A, dest=As[i], coord=Coord([zero, zero, row, tile])),
                MemLoad(source=B, dest=Bs[i], coord=Coord([zero, zero, tile, col]))
            ]),
        ])

    def consumer_stmt(i: int) -> Stmt:
        return SeqStmt([
            lms[i].wait_full(Level.warpgroup),
            OpCall("warpgroup::mma_AB", [C_accum, As[i], Bs[i]]).to_stmt(),
            OpCall("warpgroup::mma_async_wait", []).to_stmt(),
            lane0_if(ExprStmt(OpCall("arrive", [lms[i].empty_sem, RawExpr(1)]))),
        ])

    producer_loop = ForStmt(
        AssignExpr(tile, zero),
        BinaryOp(tile, num_tiles, "<"),
        AssignExpr(tile, BinaryOp(tile, RawExpr(1), "+")),
        IfStmt(
            BinaryOp(BinaryOp(tile, RawExpr(QSIZE), "%"), zero, "=="),
            producer_stmt(0),
            producer_stmt(1),
        )
    )

    consumer_loop = ForStmt(
        AssignExpr(tile, zero),
        BinaryOp(tile, num_tiles, "<"),
        AssignExpr(tile, BinaryOp(tile, RawExpr(1), "+")),
        IfStmt(
            BinaryOp(BinaryOp(tile, RawExpr(QSIZE), "%"), zero, "=="),
            consumer_stmt(0),
            consumer_stmt(1),
        )
    )

    prime_empty = SeqStmt([
        lms[i].arrive_empty()
        for i in range(QSIZE)
    ])

    producer_block = SeqStmt([
        OpCall("warpgroup::decrease_registers<32>", []).to_stmt(),
        producer_loop,
    ])

    consumer_block = SeqStmt([
        OpCall("warpgroup::increase_registers<256>", []).to_stmt(),
        OpCall("kittens::warp::zero", [C_accum]).to_stmt(),
        prime_empty,
        consumer_loop,
        C_accum.warpgroup_store_global(C, Coord([zero, zero, row, col])),
    ])

    body = SeqStmt(
        [var.declare() for var in [*As, *Bs, C_accum]] +
        [lm.initialize() for lm in lms] +
        [
            ExprStmt(OpCall("__syncthreads", [])),
            tile.declare(),
            IfStmt(
                BinaryOp(RawExpr("warpgroupid"), zero, "=="),
                producer_block,
                consumer_block,
            )
        ]
    )

    return Program(kernel_vars=kernel_globals, kernel_stmt=body)

if __name__ == "__main__":
    print(emit_cpp(build_gemm_kernel()))