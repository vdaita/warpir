import sys
import os

sys.path.insert(0, 
  os.path.dirname(
    os.path.dirname(
      os.path.abspath(__file__)
    )
  )
)

from warpir_v2.ir import (
    CallOp,
    ExprArg,
    ForOp,
    GPUType,
    GlobalType,
    RegTileLayout,
    RegTileType,
    Kernel,
    Param,
    IntType,
    ScalarType,
    SeqOp,
    SharedTileLayout,
    SharedTileType,
    Value,
)
from warpir_v2.lowering import ThunderKittensLowerer


def gen_gemm_baseline() -> Kernel:
    block_size = 32
    shared_bf16_tile = SharedTileType(
        data_type=GPUType.bf16,
        rows=block_size,
        cols=block_size,
        layout=SharedTileLayout.row_major,
    )
    global_type = GlobalType(
        data_type=GPUType.bf16,
        sub_tile_type=shared_bf16_tile,
        batch=1,
        depth=1,
        rows=-1,
        cols=-1,
    )
    reg_bf16_row = RegTileType(GPUType.bf16, block_size, block_size, RegTileLayout.row_major)
    reg_bf16_col = RegTileType(GPUType.bf16, block_size, block_size, RegTileLayout.col_major)
    reg_fp32_row = RegTileType(GPUType.fp32, block_size, block_size, RegTileLayout.row_major)
    coord_type = ScalarType("coord")

    as_val = Value("As", shared_bf16_tile)
    bs_val = Value("Bs", shared_bf16_tile)
    a_reg_val = Value("A_reg", reg_bf16_row)
    b_reg_val = Value("B_reg", reg_bf16_row)
    b_reg_col_val = Value("B_reg_col", reg_bf16_col)
    c_accum_val = Value("C_accum", reg_fp32_row)
    tile_iter = Value("tile", IntType())

    return Kernel(
        name="kernel",
        params=(
            Param("A", global_type),
            Param("B", global_type),
            Param("C", global_type),
            Param("N", IntType()),
        ),
        body=SeqOp(
            (
                CallOp("kittens::warp::zero", (c_accum_val,)),
                ForOp(
                    iter_value=tile_iter,
                    start=ExprArg("0", IntType()),
                    stop=ExprArg("(((globals.N + 32) + 1) / 32)", IntType()),
                    step=ExprArg("1", IntType()),
                    body=SeqOp(
                        (
                            CallOp(
                                "kittens::warp::load",
                                (
                                    as_val,
                                    ExprArg("globals.A", global_type),
                                    ExprArg("{0, 0, blockIdx.y, tile}", coord_type),
                                ),
                            ),
                            CallOp("__syncthreads", ()),
                            CallOp(
                                "kittens::warp::load",
                                (
                                    bs_val,
                                    ExprArg("globals.B", global_type),
                                    ExprArg("{0, 0, blockIdx.y, tile}", coord_type),
                                ),
                            ),
                            CallOp("__syncthreads", ()),
                            CallOp("kittens::warp::load", (a_reg_val, as_val)),
                            CallOp("__syncthreads", ()),
                            CallOp("kittens::warp::load", (b_reg_val, bs_val)),
                            CallOp("__syncthreads", ()),
                            CallOp("kittens::warp::swap_layout", (b_reg_col_val, b_reg_val)),
                            CallOp("__syncthreads", ()),
                            CallOp("kittens::warp::mma_AB", (c_accum_val, a_reg_val, b_reg_col_val, c_accum_val)),
                            CallOp("__syncthreads", ()),
                        )
                    ),
                ),
                CallOp(
                    "kittens::warp::store",
                    (
                        c_accum_val,
                        ExprArg("globals.C", global_type),
                        ExprArg("{0, 0, blockIdx.y, blockIdx.x}", coord_type),
                    ),
                ),
            )
        ),
    )


def main():
    kernel = gen_gemm_baseline()
    print(ThunderKittensLowerer().lower(kernel))


if __name__ == "__main__":
    main()