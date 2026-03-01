from warpir import *
def build_gemm_kernel() -> Program:
    block_size, qsize = 64, 2
    sub_tile = SharedTileType(GPUType.bf16, block_size, block_size, SharedTileLayout.row_major)
    g = KernelGlobals(A=GlobalType(GPUType.bf16, sub_tile), B=GlobalType(GPUType.bf16, sub_tile),
                      C=GlobalType(GPUType.bf16, sub_tile), N=ScalarType("int"))
    As, Bs = [Tile(f"As_{i}", sub_tile) for i in range(qsize)], [Tile(f"Bs_{i}", sub_tile) for i in range(qsize)]
    row, col, tile, p = getConst("blockIdx.x"), getConst("blockIdx.y"), getConst("threadIdx.x"), getConst("threadIdx.y")

    C_accum = Var("C_accum", RegTileType(GPUType.fp32, 16, block_size, RegTileLayout.row_major))
    num_tiles = Var("num_tiles", ScalarType("int"))
    warpid = Var("warpid", ScalarType("int"))
    warpgroupid = Var("warpgroupid", ScalarType("int"))
    num_consumers = Var("num_consumers", ScalarType("int"))
    num_consumers_expr = getConst("(NUM_THREADS / 128) - 1")
    k_index = Var("k_index", ScalarType("int"))

    producer_loop = ForStmt(
        AssignStmt(k_index, 0),
        k_index < num_tiles,
        AssignStmt(k_index, k_index + 1),
        SeqStmt([
            As[0].wait_empty(),
            g.A.load_async(As[0], Coord(0, 0, row, k_index), As[0].full_sem),
            As[0].expect_bytes(),
            Bs[0].wait_empty(),
            g.B.load_async(Bs[0], Coord(0, 0, k_index, col), Bs[0].full_sem),
            Bs[0].expect_bytes(),
            As[1].wait_empty(),
            g.A.load_async(As[1], Coord(0, 0, row, k_index + 1), As[1].full_sem),
            As[1].expect_bytes(),
            Bs[1].wait_empty(),
            g.B.load_async(Bs[1], Coord(0, 0, k_index + 1, col), Bs[1].full_sem),
            Bs[1].expect_bytes(),
        ])
    )
    producer = SeqStmt([RawStmt("warpgroup::decrease_registers<32>();"),
                        producer_loop])

    consumer_loop = ForStmt(
        AssignStmt(k_index, 0),
        k_index < num_tiles,
        AssignStmt(k_index, k_index + 1),
        SeqStmt([
            As[0].wait_full(), Bs[0].wait_full(),
            MMAOp(C_accum, As[0].ref(), Bs[0].ref()),
            MMAWaitOp(),
            As[0].arrive_empty(), Bs[0].arrive_empty(),
            
            As[1].wait_full(), Bs[1].wait_full(),
            MMAOp(C_accum, As[1].ref(), Bs[1].ref()),
            MMAWaitOp(),
            As[1].arrive_empty(), Bs[1].arrive_empty(),
        ])
    )
    consumer = SeqStmt([
        RawStmt("warpgroup::increase_registers<256>();"), DeclStmt(C_accum),
        OpCall("kittens::warp::zero", C_accum),
        consumer_loop, 
        g.C.store(C_accum, Coord(0, 0, row, col)),
    ])

    kernel_stmt = SeqStmt([
        DeclStmt(k_index),
        DeclStmt(num_tiles), DeclStmt(warpid), DeclStmt(warpgroupid), DeclStmt(num_consumers, num_consumers_expr),
        RawStmt("extern __shared__ alignment_dummy __shm[];\nshared_allocator al((int*)&__shm[0]);"),
        *[Ai.def_() for Ai in As],
        *[Bi.def_() for Bi in Bs],
        *[Ai.declare_semaphores() for Ai in As],
        *[Bi.declare_semaphores() for Bi in Bs],
        IfStmt(ThreadIdx.x == 0, SeqStmt([
            *[Ai.init_semaphores(0, "num_consumers") for Ai in As],
            *[Bi.init_semaphores(0, "num_consumers") for Bi in Bs],
        ])),
        RawStmt("__syncthreads();"),
        WarpgroupDispatch(warpgroupid, [(0, producer), (1, consumer)])
    ])
    return Program(input_vars=[], kernel_vars=g, kernel_stmt=kernel_stmt)
if __name__ == "__main__":
    print(emit_cpp(build_gemm_kernel()))