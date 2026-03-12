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
    ColVecType,
    CopyOp,
    DivRowOp,
    Exp2Op,
    ForOp,
    GPUType,
    GlobalType,
    IntType,
    IterArg,
    Kernel,
    MMAOp,
    MulOp,
    MulRowOp,
    MulScalarOp,
    NegInftyOp,
    RegTileType,
    RowMaxOp,
    RowSumOp,
    SharedTileType,
    SubOp,
    SubRowOp,
    TMALoadOp,
    TMAStoreOp,
    Value,
    WaitOp,
    YieldOp,
    ZeroOp,
)

# D=64: 1/sqrt(64) * log2(e) for exp2-based softmax
SCALE = 0.125 * 1.44269504089


def build_attention_kernel() -> Kernel:
    D = 64
    BLOCK = 64
    WARP_ROWS = 16

    shared_tile = SharedTileType(GPUType.bf16, BLOCK, D)
    global_type = GlobalType(
        data_type=GPUType.bf16,
        sub_tile_type=shared_tile,
        batch=1, depth=1, rows=-1, cols=-1,
    )
    # att scores: WARP_ROWS × BLOCK (Q rows per warp × K rows per tile)
    att_fp32 = RegTileType(GPUType.fp32, WARP_ROWS, BLOCK)
    att_bf16 = RegTileType(GPUType.bf16, WARP_ROWS, BLOCK)
    # output accumulator: WARP_ROWS × D
    o_reg = RegTileType(GPUType.fp32, WARP_ROWS, D)
    # column vectors for online softmax (one element per row of att block)
    cvec = ColVecType(GPUType.fp32, WARP_ROWS, BLOCK)

    # ---- Kernel I/O ----
    Q = Value("Q", global_type)
    K = Value("K", global_type)
    V = Value("V", global_type)
    O_global = Value("O", global_type)
    N = Value("N", IntType())

    # Grid builtins
    seq_idx = Value("blockIdx.x", IntType())

    # ---- Q load (stays resident in shared memory) ----
    q_shared = Value("q_shared", shared_tile)

    # ---- Accumulator init ----
    o_init     = Value("o_init", o_reg)
    max_init   = Value("max_init", cvec)
    norm_init  = Value("norm_init", cvec)

    # ---- Loop ----
    i          = Value("i", IntType())
    o_accum    = Value("o_accum", o_reg)       # block arg
    max_vec    = Value("max_vec", cvec)         # block arg
    norm_vec   = Value("norm_vec", cvec)        # block arg

    # Loop body values
    k_shared       = Value("k_shared", shared_tile)
    v_shared       = Value("v_shared", shared_tile)
    att_zero       = Value("att_zero", att_fp32)
    att            = Value("att", att_fp32)
    max_prev       = Value("max_prev", cvec)
    max_new        = Value("max_new", cvec)
    att_scaled     = Value("att_scaled", att_fp32)
    max_new_sc     = Value("max_new_sc", cvec)
    max_prev_sc    = Value("max_prev_sc", cvec)
    att_shifted    = Value("att_shifted", att_fp32)
    P_fp32         = Value("P_fp32", att_fp32)
    corr_diff      = Value("corr_diff", cvec)
    correction     = Value("correction", cvec)
    norm_rescaled  = Value("norm_rescaled", cvec)
    norm_new       = Value("norm_new", cvec)
    o_rescaled     = Value("o_rescaled", o_reg)
    P_bf16         = Value("P_bf16", att_bf16)
    o_new          = Value("o_new", o_reg)

    # Post-loop
    o_final    = Value("o_final", o_reg)
    max_final  = Value("max_final", cvec)
    norm_final = Value("norm_final", cvec)
    o_norm     = Value("o_norm", o_reg)

    return Kernel(
        name="attention_fwd",
        inputs=(Q, K, V, N),
        outputs=(O_global,),
        body=(
            # Load Q tile for this block's sequence position
            TMALoadOp(result=q_shared, source=Q, coords=(0, 0, seq_idx, 0)),
            WaitOp(values=(q_shared,)),

            # Initialise accumulators
            ZeroOp(result=o_init),
            NegInftyOp(result=max_init),
            ZeroOp(result=norm_init),

            # Loop over KV tiles
            ForOp(
                induction_var=i,
                start=0,
                stop=N,
                step=1,
                tile_size=BLOCK,
                iter_args=(
                    IterArg(block_arg=o_accum,  init=o_init),
                    IterArg(block_arg=max_vec,  init=max_init),
                    IterArg(block_arg=norm_vec, init=norm_init),
                ),
                body=(
                    # Load K and V tiles
                    TMALoadOp(result=k_shared, source=K, coords=(0, 0, i, 0)),
                    TMALoadOp(result=v_shared, source=V, coords=(0, 0, i, 0)),
                    WaitOp(values=(k_shared, v_shared)),

                    # S = Q @ K^T
                    ZeroOp(result=att_zero),
                    MMAOp(result=att, a=q_shared, b=k_shared,
                          accum=att_zero, transpose_b=True),

                    # Online softmax ----------------------------------
                    # Save old max before updating
                    CopyOp(result=max_prev, input=max_vec),
                    # Running row-wise max
                    RowMaxOp(result=max_new, tile=att, prev=max_vec),
                    # Scale scores and max vectors for exp2
                    MulScalarOp(result=att_scaled,  input=att,      scalar=SCALE),
                    MulScalarOp(result=max_new_sc,  input=max_new,  scalar=SCALE),
                    MulScalarOp(result=max_prev_sc, input=max_prev, scalar=SCALE),
                    # P = exp2(scores * scale - max_new * scale)
                    SubRowOp(result=att_shifted, tile=att_scaled, vec=max_new_sc),
                    Exp2Op(result=P_fp32, input=att_shifted),
                    # Correction factor = exp2(old_max_scaled - new_max_scaled)
                    SubOp(result=corr_diff, a=max_prev_sc, b=max_new_sc),
                    Exp2Op(result=correction, input=corr_diff),
                    # Update running normalizer
                    MulOp(result=norm_rescaled, a=norm_vec, b=correction),
                    RowSumOp(result=norm_new, tile=P_fp32, prev=norm_rescaled),
                    # Rescale previous O accumulator
                    MulRowOp(result=o_rescaled, tile=o_accum, vec=correction),
                    # O += P @ V  (P cast to bf16 for MMA)
                    CopyOp(result=P_bf16, input=P_fp32),
                    MMAOp(result=o_new, a=P_bf16, b=v_shared,
                          accum=o_rescaled),

                    YieldOp(values=(o_new, max_new, norm_new)),
                ),
                results=(o_final, max_final, norm_final),
            ),

            # Normalise output
            DivRowOp(result=o_norm, tile=o_final, vec=norm_final),
            TMAStoreOp(source=o_norm, dest=O_global, coords=(0, 0, seq_idx, 0)),
        ),
    )


if __name__ == "__main__":
    from pathlib import Path
    from warpir.printing import print_kernel
    from warpir.lowering import ThunderKittensLowerer

    kernel = build_attention_kernel()

    print("=== Attention IR ===")
    print_kernel(kernel)
    print()

    lowerer = ThunderKittensLowerer()
    cuda_src = lowerer.lower(kernel)

    print("=== ThunderKittens CUDA ===")
    print(cuda_src)

    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "attention_baseline.cu"
    out_path.write_text(cuda_src)
    print(f"\nWrote {out_path}")
