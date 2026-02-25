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
    qsize_sym = Symbol("QSIZE")
    num_tiles = Var("num_tiles", ScalarType("int"))
    warpid = Var("warpid", ScalarType("int"))
    warpgroupid = Var("warpgroupid", ScalarType("int"))
    num_consumers = Var("num_consumers", ScalarType("int"))
    pipeline = Pipeline(qsize_sym, tile, qidx, p)
    producer_loop = pipeline.loop(
        num_tiles,
        pipeline_select(qidx, [
            q0.produce([(As[0], g.A, Coord(0, 0, row, tile)), (Bs[0], g.B, Coord(0, 0, tile, col))], phase=p),
            q1.produce([(As[1], g.A, Coord(0, 0, row, tile)), (Bs[1], g.B, Coord(0, 0, tile, col))], phase=p),
        ]),
    )
    producer = SeqStmt([WarpgroupRegs(32, "decrease"),
                        lane0_if(SeqStmt([DeclStmt(p, 0), producer_loop]))])

    consumer_loop = pipeline.loop(
        num_tiles,
        SeqStmt([
            pipeline_select(qidx, [
                q0.consume(p, SeqStmt([ExprStmt(MMAOp(C_accum, As[0], Bs[0])),
                                       ExprStmt(MMAWaitOp())]), arrive=False),
                q1.consume(p, SeqStmt([ExprStmt(MMAOp(C_accum, As[1], Bs[1])),
                                       ExprStmt(MMAWaitOp())]), arrive=False),
            ]),
            lane0_if(pipeline_select(qidx, [ExprStmt(ArriveOp(q0.empty_sem, 1)), ExprStmt(ArriveOp(q1.empty_sem, 1))])),
        ]),
    )
    consumer = SeqStmt([
        WarpgroupRegs(256, "increase"), DeclStmt(C_accum),
        ExprStmt(UnaryOp("kittens::warp::zero", C_accum)),
        lane0_if(SeqStmt([ExprStmt(ArriveOp(q0.empty_sem, 1)), ExprStmt(ArriveOp(q1.empty_sem, 1))])),
        DeclStmt(p, 0), consumer_loop, ExprStmt(StoreOp(g.C, C_accum, Coord(0, 0, row, col))),
    ])

    kernel_stmt = SeqStmt([
        kernel_prelude(block_size, qsize, g, row, col, As + Bs, [q0, q1],
                       num_tiles, warpid, warpgroupid, num_consumers),
        WarpgroupDispatch(warpgroupid, [(0, producer)], default=consumer),
    ])
    return Program(input_vars=[], kernel_vars=g, kernel_stmt=kernel_stmt)
if __name__ == "__main__":
    print(emit_cpp(build_gemm_kernel()))
