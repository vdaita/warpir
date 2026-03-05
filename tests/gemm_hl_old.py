from warpir import *
from warpir.dataflow_graph import DataflowGraph, MemLoad, draw_graph
import os
import networkx as nx


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
    tile = Var("tile", ScalarType("int"))

    outer = DataflowGraph(kernel_globals=g)
    outer.op("kittens::warp::zero", parameters=[acc], out=acc, ins=[])

    body = outer.subscope()
    body.tma_produce([
        MemLoad(
            source=A,
            dest=As,
            coord=Coord([zero, zero, row, tile])
        ),
        MemLoad(
            source=B,
            dest=Bs,
            coord=Coord([zero, zero, tile, col])
        )
    ])
    tma_consume_inst = body.tma_consume([As, Bs])
    mma_issue_inst = body.mma_issue(acc, As, Bs, parent_instrs=[tma_consume_inst])
    mma_wait_inst = body.mma_wait(acc)
    tma_done_inst = body.tma_done([As, Bs], [tma_consume_inst, mma_wait_inst]) # technically just needs to rely on the wait

    outer.loop(tile, num_tiles, body, [acc], [acc, As, Bs]) # reads -> [acc], writes -> [acc]
    outer.op("warpgroup::store", parameters=[C, acc, Coord([zero, zero, row, col])], out=acc, ins=[acc])

    return outer.emit_program(), outer


if __name__ == "__main__":
    p, dataflow = build_gemm_kernel()
    print(emit_cpp(p))
    import matplotlib.pyplot as plt

    print("cycles: ", list(nx.simple_cycles(dataflow.G)))

    os.makedirs("outputs", exist_ok=True)
    draw_graph(dataflow, "outputs/graph.png")