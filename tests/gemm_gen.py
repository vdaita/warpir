from warpir import *
def build_gemm_kernel() -> Program:
    block_size, qsize = 64, 2
    sub_tile = SharedTileType(GPUType.bf16, block_size, block_size, SharedTileLayout.row_major)
    g = KernelGlobals(A=GlobalType(GPUType.bf16, sub_tile), B=GlobalType(GPUType.bf16, sub_tile),
                      C=GlobalType(GPUType.bf16, sub_tile), N=ScalarType("int"))
    As, Bs = [Tile(f"As_{i}", sub_tile) for i in range(qsize)], [Tile(f"Bs_{i}", sub_tile) for i in range(qsize)]
    q0, q1 = TileQueue("q0", [As[0], Bs[0]]), TileQueue("q1", [As[1], Bs[1]])
    row, col, tile, qidx, p = (Var("row", ScalarType("int")), Var("col", ScalarType("int")),
                               Var("tile", ScalarType("int")), Var("qidx", ScalarType("int")), Var("p", ScalarType("int")))
    C_accum = Var("C_accum", RegTileType(GPUType.fp32, 16, block_size, RegTileLayout.row_major))
    on_qidx = lambda s0, s1: IfStmt(BinaryOp("==", qidx, 0), s0, s1)
    toggle = lambda: IfStmt(BinaryOp("==", qidx, "QSIZE"),
                            SeqStmt([AssignStmt(qidx, 0), AssignStmt(p, BinaryOp("^", p, 1))]))
    lane0 = BinaryOp("==", CallExpr("warpgroup::laneid"), 0)
    producer_loop = ForStmt("int tile = 0, qidx = 0", f"{tile} < num_tiles", "++tile, ++qidx", SeqStmt([
        toggle(),
        on_qidx(
            q0.produce([(As[0], g.A, Coord(0, 0, row, tile)), (Bs[0], g.B, Coord(0, 0, tile, col))], phase=p),
            q1.produce([(As[1], g.A, Coord(0, 0, row, tile)), (Bs[1], g.B, Coord(0, 0, tile, col))], phase=p),
        ),
    ]))
    producer = SeqStmt([RawStmt("warpgroup::decrease_registers<32>();"),
                        IfStmt(lane0, SeqStmt([DeclStmt(p, 0), producer_loop]))])

    consumer_loop = ForStmt("int tile = 0, qidx = 0", f"{tile} < num_tiles", "++tile, ++qidx", SeqStmt([
        toggle(),
        on_qidx(
            q0.consume(p, SeqStmt([ExprStmt(MMAOp(C_accum, As[0], Bs[0])),
                                   RawStmt("warpgroup::mma_async_wait();")]), arrive=False),
            q1.consume(p, SeqStmt([ExprStmt(MMAOp(C_accum, As[1], Bs[1])),
                                   RawStmt("warpgroup::mma_async_wait();")]), arrive=False),
        ),
        IfStmt(lane0, on_qidx(ExprStmt(ArriveOp(q0.empty_sem, 1)), ExprStmt(ArriveOp(q1.empty_sem, 1)))),
    ]))
    consumer = SeqStmt([
        RawStmt("warpgroup::increase_registers<256>();"), DeclStmt(C_accum),
        ExprStmt(CallExpr("kittens::warp::zero", C_accum)),
        IfStmt(lane0, SeqStmt([ExprStmt(ArriveOp(q0.empty_sem, 1)), ExprStmt(ArriveOp(q1.empty_sem, 1))])),
        DeclStmt(p, 0), consumer_loop, ExprStmt(StoreOp(g.C, C_accum, Coord(0, 0, row, col))),
    ])

    kernel_stmt = SeqStmt([
        kernel_prelude(block_size, qsize, g, row, col, As + Bs, [q0, q1]),
        IfStmt("warpgroupid == 0", producer, consumer),
    ])
    return Program(input_vars=[], kernel_vars=g, kernel_stmt=kernel_stmt)
if __name__ == "__main__":
    print(format_cpp(str(build_gemm_kernel())))
