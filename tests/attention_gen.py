from warpir import *


def build_attention_kernel() -> Program:
    block, stages = 64, 2
    q_tile = SharedTileType(GPUType.bf16, block, block, SharedTileLayout.row_major)
    k_tile = SharedTileType(GPUType.bf16, block, block, SharedTileLayout.row_major)
    v_tile = SharedTileType(GPUType.bf16, block, block, SharedTileLayout.row_major)
    g = KernelGlobals(
        Q=GlobalType(GPUType.bf16, q_tile),
        K=GlobalType(GPUType.bf16, k_tile),
        V=GlobalType(GPUType.bf16, v_tile),
        O=GlobalType(GPUType.bf16, q_tile),
        N=ScalarType("int"),
    )
    Qs = [Tile(f"Qs_{i}", q_tile) for i in range(stages)]
    Ks = [Tile(f"Ks_{i}", k_tile) for i in range(stages)]
    Vs = [Tile(f"Vs_{i}", v_tile) for i in range(stages)]
    q0, q1 = TileQueue("q0", [Qs[0], Ks[0], Vs[0]]), TileQueue("q1", [Qs[1], Ks[1], Vs[1]])
    row, col, tile, qidx, p = (Var("row", ScalarType("int")), Var("col", ScalarType("int")),
                               Var("tile", ScalarType("int")), Var("qidx", ScalarType("int")), Var("p", ScalarType("int")))
    num_tiles = Var("num_tiles", ScalarType("int"))
    warpid = Var("warpid", ScalarType("int"))
    warpgroupid = Var("warpgroupid", ScalarType("int"))
    num_consumers = Var("num_consumers", ScalarType("int"))
    stages_sym = Symbol("STAGES")
    S_accum = Var("S_accum", RegTileType(GPUType.fp32, 16, block, RegTileLayout.row_major))
    O_accum = Var("O_accum", RegTileType(GPUType.fp32, 16, block, RegTileLayout.row_major))
    max_vec = Var("max_vec", RegVecType(GPUType.fp32, 16))
    norm_vec = Var("norm_vec", RegVecType(GPUType.fp32, 16))
    pipeline = Pipeline(stages_sym, tile, qidx, p)

    producer_loop = pipeline.loop(
        num_tiles,
        pipeline_select(qidx, [
            q0.produce([(Qs[0], g.Q, Coord(0, 0, row, tile)),
                        (Ks[0], g.K, Coord(0, 0, tile, col)),
                        (Vs[0], g.V, Coord(0, 0, tile, col))], phase=p),
            q1.produce([(Qs[1], g.Q, Coord(0, 0, row, tile)),
                        (Ks[1], g.K, Coord(0, 0, tile, col)),
                        (Vs[1], g.V, Coord(0, 0, tile, col))], phase=p),
        ]),
    )
    producer = SeqStmt([WarpgroupRegs(32, "decrease"),
                        lane0_if(SeqStmt([DeclStmt(p, 0), producer_loop]))])

    consumer_loop = pipeline.loop(
        num_tiles,
        SeqStmt([
            pipeline_select(qidx, [
                q0.consume(p, SeqStmt([ExprStmt(MMAOp(S_accum, Qs[0], Ks[0])),
                                       ExprStmt(MMAWaitOp())]), arrive=False),
                q1.consume(p, SeqStmt([ExprStmt(MMAOp(S_accum, Qs[1], Ks[1])),
                                       ExprStmt(MMAWaitOp())]), arrive=False),
            ]),
            lane0_if(pipeline_select(qidx, [ExprStmt(ArriveOp(q0.empty_sem, 1)), ExprStmt(ArriveOp(q1.empty_sem, 1))])),
        ]),
    )
    consumer = SeqStmt([
        WarpgroupRegs(256, "increase"), DeclStmt(O_accum),
        DeclStmt(S_accum),
        DeclStmt(max_vec),
        DeclStmt(norm_vec),
        ExprStmt(UnaryOp("kittens::warp::zero", O_accum)),
        ExprStmt(UnaryOp("kittens::warp::zero", S_accum)),
        lane0_if(SeqStmt([ExprStmt(ArriveOp(q0.empty_sem, 1)), ExprStmt(ArriveOp(q1.empty_sem, 1))])),
        DeclStmt(p, 0),
        consumer_loop,
        ExprStmt(BinaryOp("kittens::warp::row_max", max_vec, S_accum)),
        ExprStmt(BinaryOp("kittens::warp::sub_row", S_accum, max_vec)),
        ExprStmt(UnaryOp("kittens::warp::exp", S_accum)),
        ExprStmt(BinaryOp("kittens::warp::row_sum", norm_vec, S_accum)),
        AssignStmt(norm_vec, UnaryOp("kittens::warp::rcp", norm_vec)),
        ExprStmt(BinaryOp("kittens::warp::mul_row", S_accum, norm_vec)),
        pipeline_select(qidx, [
            ExprStmt(MMAOp(O_accum, S_accum, Vs[0])),
            ExprStmt(MMAOp(O_accum, S_accum, Vs[1])),
        ]),
        ExprStmt(StoreOp(g.O, O_accum, Coord(0, 0, row, col))),
    ])

    kernel_stmt = SeqStmt([
        kernel_prelude(block, stages, g, row, col, Qs + Ks + Vs, [q0, q1],
                       num_tiles, warpid, warpgroupid, num_consumers),
        WarpgroupDispatch(warpgroupid, [(0, producer)], default=consumer),
    ])
    return Program(input_vars=[], kernel_vars=g, kernel_stmt=kernel_stmt)


if __name__ == "__main__":
    print(emit_cpp(build_attention_kernel()))
