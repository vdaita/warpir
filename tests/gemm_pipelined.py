from warpir import RegTileLayout
from warpir import *

def build_gemm_kernel() -> Program:
    BLOCK_SIZE = 64
    NUM_WORKERS = 4
    NUM_THREADS = 4 * 32 # the number of threads in a warp is 32 anyways

    shared_tile_type = SharedTileType(GPUType.bf16, BLOCK_SIZE, BLOCK_SIZE, SharedTileLayout.row_major)
    As = [Tile("As0", shared_tile_type),  Tile("As1", shared_tile_type)]
    Bs = [Tile("Bs0", shared_tile_type), Tile("Bs1", shared_tile_type,)]
    
    global_type = GlobalType(GPUType.bf16, shared_tile_type, 1, 1, -1, -1)
    A = Var("A", global_type)
    B = Var("B", global_type)
    C = Var("C", global_type)
    N = Var("N", ScalarType("int"))

    kernel_globals = KernelGlobals("globals")
    for var in [A, B, C, N]:
        kernel_globals.add_var(var)

    accum_tile_type = RegTileType(GPUType.fp32, 16, BLOCK_SIZE, RegTileLayout.row_major)
    C_accum = Tile("C_accum", accum_tile_type)
    C_accum_cpy = Tile("C_accum_cpy", accum_tile_type)

    num_tiles = BinaryOp(BinaryOp(N, RawExpr(BLOCK_SIZE - 1), "+"), RawExpr(BLOCK_SIZE), "/")
    col = RawExpr("blockIdx.x")
    row = RawExpr("blockIdx.y")

    tile = Var("tile", ScalarType("int"))
    tile_plus_1 = BinaryOp(tile, RawExpr(1), "+")

    load_manager = [TileGroup("lm0", [As[0], Bs[0]]), TileGroup("lm1", (As[1], Bs[1]))]

    def statement_idx(i: int):
        return SeqStmt([
            # wait for 0,
            load_manager[i].wait_full(Level.block),
            # load 1,
            IfStmt(
                BinaryOp(tile_plus_1, num_tiles, "<"),
                load_manager[i ^ 1].async_load_global([
                    MemLoad(
                        source=A,
                        dest=As[i ^ 1],
                        coord=Coord([zero, zero, row, tile_plus_1])
                    ),
                    MemLoad(
                        source=B,
                        dest=Bs[i ^ 1],
                        coord=Coord([zero, zero, tile_plus_1, col])
                    )
                ])
            ),
            # compute 0,
            OpCall("warpgroup::mma_AB", [C_accum, As[i], Bs[i]]).to_stmt(),
            OpCall("warpgroup::mma_async_wait", []).to_stmt(),
            # store accumulation 
            OpCall("kittens::warp::add", [C_accum_cpy, C_accum_cpy, C_accum]).to_stmt(),
            OpCall("kittens::warp::zero", [C_accum]).to_stmt(),
            OpCall("__syncthreads", []).to_stmt(), 
        ])

    # load the first elements
    zero = RawExpr(0)
    body = SeqStmt(
    [var.declare() for var in [As[0], As[1], Bs[0], Bs[1], C_accum, C_accum_cpy, tile]] +
    [manager.initialize() for manager in load_manager] +
    [
        load_manager[0].async_load_global([
            MemLoad(
                source=A,
                dest=As[0],
                coord=Coord([zero, zero, row, zero])
            ),
            MemLoad(
                source=B,
                dest=Bs[0],
                coord=Coord([zero, zero, zero, col])
            )
        ]),
        OpCall("__syncthreads", []).to_stmt(),
        OpCall("kittens::warp::zero", [C_accum]).to_stmt(),
        OpCall("kittens::warp::zero", [C_accum_cpy]).to_stmt(),
        ForStmt(
            AssignExpr(tile, zero),
            BinaryOp(tile, num_tiles, "<"),
            AssignExpr(tile, BinaryOp(tile, RawExpr(1), "+")),
            SeqStmt([
                IfStmt(
                    BinaryOp(BinaryOp(tile, RawExpr(2), "%"), zero, "=="),
                    statement_idx(0),
                    statement_idx(1)
                )    
            ])
        ),
        C_accum_cpy.warpgroup_store_global(C, Coord([zero, zero, row, col]))
    ])

    return Program(
        kernel_vars=kernel_globals,
        kernel_stmt=body
    )

if __name__ == "__main__":
    print(emit_cpp(build_gemm_kernel()))
