from warpir import *
from warpir.sir import SIR, TMA, MMA, MemLoad

def build_gemm_kernel() -> tuple[Program, SIR]:
    BLOCK_SIZE  = 64
    shared_type = SharedTileType(GPUType.bf16, BLOCK_SIZE, BLOCK_SIZE, SharedTileLayout.row_major)
    accum_type  = RegTileType(GPUType.fp32, 16, BLOCK_SIZE, RegTileLayout.row_major)
    global_type = GlobalType(GPUType.bf16, shared_type, 1, 1, -1, -1)

    A, B, C, N = Var("A", global_type), Var("B", global_type), Var("C", global_type), Var("N", ScalarType("int"))
    g = KernelGlobals("globals")
    for v in [A, B, C, N]: g.add_var(v)

    row, col  = RawExpr("blockIdx.y"), RawExpr("blockIdx.x")
    zero      = RawExpr(0)
    num_tiles = BinaryOp(BinaryOp(N, RawExpr(BLOCK_SIZE - 1), "+"), RawExpr(BLOCK_SIZE), "/")

    sir = SIR()
    As  = sir.tile("As", shared_type)
    Bs  = sir.tile("Bs", shared_type)
    acc = sir.tile("acc", accum_type)
    sir.op("kittens::warp::zero", acc)

    i, loop = sir.add_loop(num_tiles)

    [As_loaded, Bs_loaded], ab_scope = loop.tma_load([
        MemLoad(A, As, Coord([zero, zero, row, i])),
        MemLoad(B, Bs, Coord([zero, zero, i, col])),
    ])
    loop._emit(ab_scope)
    ab_scope.mma(As_loaded, Bs_loaded, acc)

    sir.add_store(acc, C, Coord([zero, zero, row, col]))

    return Program(kernel_vars=g, kernel_stmt=sir.emit()), sir


if __name__ == "__main__":
    from warpir.compiler import emit_cpp
    p, sir = build_gemm_kernel()
    print(emit_cpp(p))
    sir.save_graph("outputs/sir_graph.png")