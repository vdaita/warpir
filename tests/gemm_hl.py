from warpir import *
from warpir.dataflow_graph import DataflowGraph, MemLoad

def build_gemm_kernel() -> tuple[Program, DataflowGraph]:
    BLOCK_SIZE  = 64
    shared_type = SharedTileType(GPUType.bf16, BLOCK_SIZE, BLOCK_SIZE, SharedTileLayout.row_major)
    accum_type  = RegTileType(GPUType.fp32, 16, BLOCK_SIZE, RegTileLayout.row_major)
    global_type = GlobalType(GPUType.bf16, shared_type, 1, 1, -1, -1)

    A, B, C, N = Var("A", global_type), Var("B", global_type), Var("C", global_type), Var("N", ScalarType("int"))
    g = KernelGlobals("globals")
    for v in [A, B, C, N]: g.add_var(v)

    row, col  = RawExpr("blockIdx.y"), RawExpr("blockIdx.x")
    num_tiles = BinaryOp(BinaryOp(N, RawExpr(BLOCK_SIZE - 1), "+"), RawExpr(BLOCK_SIZE), "/")

    As  = Tile("As",  shared_type)
    Bs  = Tile("Bs",  shared_type)
    acc = Tile("acc", accum_type)

    outer = DataflowGraph(tiles=[As, Bs, acc], kernel_vars=[A, B, C, N])
    outer.tile_op("kittens::warp::zero", parameters=[acc], output_var=acc, input_vars=[])

    i = Var("i", ScalarType("int"))

    body = DataflowGraph(tiles=[As, Bs, acc], parent=outer, kernel_vars=[])
    body.tma([
        MemLoad(A, As, Coord([zero, zero, row, i])),
        MemLoad(B, Bs, Coord([zero, zero, i,   col])),
    ])
    body.mma(As, Bs, acc)

    outer.loop("tile", num_tiles, body)
    outer.tile_op("warpgroup::store", parameters=[C, acc, Coord([zero, zero, row, col])],
                  output_var=acc, input_vars=[acc])

    return Program(kernel_vars=g, kernel_stmt=outer.get_stmt()), outer


if __name__ == "__main__":
    from warpir.compiler import emit_cpp
    p, DataflowGraph = build_gemm_kernel()
    print(emit_cpp(p))