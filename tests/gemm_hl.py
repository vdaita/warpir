from warpir import *
from warpir.sir import SIR, MemLoad

def build_gemm_kernel() -> Program:
    BLOCK_SIZE  = 64
    shared_type = SharedTileType(GPUType.bf16, BLOCK_SIZE, BLOCK_SIZE, SharedTileLayout.row_major)
    accum_type  = RegTileType(GPUType.fp32, 16, BLOCK_SIZE, RegTileLayout.row_major)
    global_type = GlobalType(GPUType.bf16, shared_type, 1, 1, -1, -1)

    A, B, C, N = Var("A", global_type), Var("B", global_type), Var("C", global_type), Var("N", ScalarType("int"))
    g = KernelGlobals("globals")
    for v in [A, B, C, N]: g.add_var(v)

    row, col  = RawExpr("blockIdx.y"), RawExpr("blockIdx.x")
    tile      = Var("tile", ScalarType("int"))
    zero      = RawExpr(0)
    num_tiles = BinaryOp(BinaryOp(N, RawExpr(BLOCK_SIZE - 1), "+"), RawExpr(BLOCK_SIZE), "/")

    sir = SIR()
    As  = sir.tile(shared_type)
    Bs  = sir.tile(shared_type)
    acc = sir.tile(accum_type)
    sir.op("kittens::warp::zero", acc)

    def loop_body(s: SIR):
        group          = s.tma.load([
            MemLoad(A, As, Coord([zero, zero, row, tile])),
            MemLoad(B, Bs, Coord([zero, zero, tile, col])),
        ])
        As_rdy, Bs_rdy = s.tma.consume(group)
        result         = s.mma.produce(As_rdy, Bs_rdy, acc)
        s.mma.consume(result)

    sir.loop(tile, num_tiles, loop_body)
    sir.op("warpgroup::store", acc, C, Coord([zero, zero, row, col]))

    return Program(kernel_vars=g, kernel_stmt=sir.emit()), sir


if __name__ == "__main__":
    import subprocess, os
    os.makedirs("outputs", exist_ok=True)
    
    p, sir = build_gemm_kernel()
    print(emit_cpp(p))
