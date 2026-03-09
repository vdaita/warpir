import sys
import os

sys.path.insert(0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from warpir.ir import (
    ForOp,
    GPUType,
    GlobalType,
    IntType,
    IterArg,
    Kernel,
    MMAOp,
    RegTileType,
    SharedTileType,
    TMALoadOp,
    TMAStoreOp,
    Value,
    WaitOp,
    YieldOp,
    ZeroOp,
)


def build_gemm_kernel() -> Kernel:
    BLOCK_SIZE = 64
    WARPGROUP_ROWS = 16  # each warp holds 1/4 of the tile

    shared_tile = SharedTileType(GPUType.bf16, BLOCK_SIZE, BLOCK_SIZE)
    global_type = GlobalType(
        data_type=GPUType.bf16,
        sub_tile_type=shared_tile,
        batch=1, depth=1, rows=-1, cols=-1,
    )
    accum_reg = RegTileType(GPUType.fp32, WARPGROUP_ROWS, BLOCK_SIZE)

    # Kernel I/O
    A = Value("A", global_type)
    B = Value("B", global_type)
    C_global = Value("C_global", global_type)
    N = Value("N", IntType())

    # Grid built-ins
    row = Value("blockIdx.y", IntType())
    col = Value("blockIdx.x", IntType())

    # SSA values
    c_init = Value("c_init", accum_reg)
    c_accum = Value("c_accum", accum_reg)    # block arg inside loop
    c_final = Value("c_final", accum_reg)    # result after loop
    i = Value("i", IntType())
    a_shared = Value("a_shared", shared_tile)
    b_shared = Value("b_shared", shared_tile)
    c_new = Value("c_new", accum_reg)

    return Kernel(
        name="gemm",
        inputs=(A, B, N),
        outputs=(C_global,),
        body=(
            ZeroOp(result=c_init),
            ForOp(
                induction_var=i,
                start=0,
                stop=N,
                step=1,
                tile_size=BLOCK_SIZE,
                iter_args=(
                    IterArg(block_arg=c_accum, init=c_init),
                ),
                body=(
                    TMALoadOp(result=a_shared, source=A, coords=(0, 0, row, i)),
                    TMALoadOp(result=b_shared, source=B, coords=(0, 0, i, col)),
                    WaitOp(values=(a_shared, b_shared)),
                    MMAOp(result=c_new, a=a_shared, b=b_shared, accum=c_accum),
                    YieldOp(values=(c_new,)),
                ),
                results=(c_final,),
            ),
            TMAStoreOp(source=c_final, dest=C_global, coords=(0, 0, row, col)),
        ),
    )


if __name__ == "__main__":
    from pathlib import Path
    from warpir.printing import print_kernel
    from warpir.lowering import ThunderKittensLowerer

    kernel = build_gemm_kernel()

    print("=== IR ===")
    print_kernel(kernel)
    print()

    lowerer = ThunderKittensLowerer()
    cuda_src = lowerer.lower(kernel)

    print("=== ThunderKittens CUDA ===")
    print(cuda_src)

    out_path = Path(__file__).parent / "outputs" / "gemm_baseline.cu"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(cuda_src)
    print(f"Wrote {out_path}")
