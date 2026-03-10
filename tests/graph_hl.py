import os
import sys
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from warpir import *
from warpir.dataflow_graph import (
    DataflowGraph, MemLoad, draw_graph,
    TMAOperation, MMAOperation, CUDAOperation,
)

def build_gemm_kernel() -> tuple[Program, DataflowGraph]:
    BLOCK_SIZE = 64

    shared_type = SharedTileType(GPUType.bf16, BLOCK_SIZE, BLOCK_SIZE,
                                 SharedTileLayout.row_major)
    accum_type  = RegTileType(GPUType.fp32, 16, BLOCK_SIZE,
                              RegTileLayout.row_major)
    global_type = GlobalType(GPUType.bf16, shared_type, 1, 1, -1, -1)

    A = Var("A", global_type)
    B = Var("B", global_type)
    C = Var("C", global_type)
    N = Var("N", ScalarType("int"))

    g = KernelGlobals("globals")
    for v in [A, B, C, N]:
        g.add_var(v)

    row      = RawExpr("blockIdx.y")
    col      = RawExpr("blockIdx.x")
    num_tiles = BinaryOp(BinaryOp(N, RawExpr(BLOCK_SIZE - 1), "+"),
                         RawExpr(BLOCK_SIZE), "/")

    As  = Tile("As",  shared_type)
    Bs  = Tile("Bs",  shared_type)
    acc = Tile("acc", accum_type)
    tile = Var("tile", ScalarType("int"))

    outer = DataflowGraph(kernel_globals=g)
    CUDAOperation(
        name="kittens::warp::zero",
        parameters=[acc],
        out=acc,
        ins=[],
    ).produce(outer)
    body = outer.subscope()

    tma = TMAOperation(loads=[
        MemLoad(source=A, dest=As, coord=Coord([zero, zero, row, tile])),
        MemLoad(source=B, dest=Bs, coord=Coord([zero, zero, tile, col])),
    ])
    mma = MMAOperation(out_var=acc, a=As, b=Bs)

    tma_produce_inst  = tma.produce(body)
    tma_consume_inst  = tma.consume(body)
    mma_issue_inst    = mma.produce(body, parent_instrs=[tma_consume_inst])
    mma_wait_inst     = mma.consume(body)
    tma_done_inst     = tma.done(body, parent_instrs=[tma_consume_inst,
                                                      mma_wait_inst])

    outer.loop(tile, num_tiles, body, reads=[acc], writes=[acc, As, Bs])
    store_op = CUDAOperation(
        name="warpgroup::store",
        parameters=[C, acc, Coord([zero, zero, row, col])],
        out=acc,
        ins=[acc],
    )
    store_op.produce(outer)

    return outer.emit_program(), outer

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    print("Building un-pipelined GEMM DataflowGraph...")
    baseline_prog, baseline_graph = build_gemm_kernel()

    print(f"  Instructions in loop body: "
          f"{sum(1 for i in baseline_graph.instructions if i.is_loop)}"
          f" loop(s), "
          f"{sum(1 for i in baseline_graph.instructions if not i.is_loop)}"
          f" non-loop op(s)")
    print(f"  Dependency cycles: {list(nx.simple_cycles(baseline_graph.G))}")

    draw_graph(baseline_graph, "outputs/graph_before.png")
    print("  → outputs/graph_before.png")

    baseline_cu = emit_cpp(baseline_prog)
    with open("outputs/hl_baseline.cu", "w") as f:
        f.write(baseline_cu)
    print("  → outputs/hl_baseline.cu\n")