"""Test WarpIR attention baseline kernel on an H100 via Modal.

Usage:
    uv run modal run tests/test_modal_attention.py
"""
import modal

BLOCK_SIZE = 64
D = 64
NUM_THREADS = 128

EXTRA_CUDA_CFLAGS = [
    "-std=c++20", "-O3", "--use_fast_math",
    "--expt-extended-lambda", "--expt-relaxed-constexpr",
    "-DKITTENS_HOPPER",
    "-gencode", "arch=compute_90a,code=sm_90a",
    "-I/root/warpir/thunderkittens/include",
    "-DNDEBUG",
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
]

LAUNCH_WRAPPER = f"""
void launch_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                      torch::Tensor O) {{
    size_t N = Q.size(0);
    size_t head_dim = Q.size(1);
    using tile_gl = gl<bf16, 1, 1, -1, -1, st_bf<{BLOCK_SIZE}, {D}>>;
    tile_gl q_gl{{(bf16*)Q.data_ptr(), nullptr, nullptr, N, head_dim}};
    tile_gl k_gl{{(bf16*)K.data_ptr(), nullptr, nullptr, N, head_dim}};
    tile_gl v_gl{{(bf16*)V.data_ptr(), nullptr, nullptr, N, head_dim}};
    tile_gl o_gl{{(bf16*)O.data_ptr(), nullptr, nullptr, N, head_dim}};
    global_vars g{{q_gl, k_gl, v_gl, (int)N, o_gl}};
    dim3 grid(N / {BLOCK_SIZE}, 1, 1);
    attention_fwd<<<grid, {NUM_THREADS}>>>(g);
}}
"""

CPP_DECL = """
#include <torch/extension.h>
void launch_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                      torch::Tensor O);
"""

cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .pip_install("torch", "ninja", "setuptools")
    .add_local_dir(
        ".",
        remote_path="/root/warpir",
        ignore=[
            ".venv/",
            ".git/",
            "__pycache__/",
            "*.pyc",
            "papers/",
            "v2_tests/build/",
        ],
    )
)

app = modal.App("warpir-attention-test")


def _compile(cuda_kernel: str):
    """JIT-compile the attention kernel via torch load_inline."""
    import os
    from torch.utils.cpp_extension import load_inline

    cuda_src = "#include <torch/extension.h>\n" + cuda_kernel + LAUNCH_WRAPPER
    build_dir = "/tmp/warpir_build/attention_baseline"
    os.makedirs(build_dir, exist_ok=True)

    print("Compiling attention_baseline ...")
    return load_inline(
        name="attention_baseline",
        cpp_sources=[CPP_DECL],
        cuda_sources=[cuda_src],
        functions=["launch_attention"],
        extra_cuda_cflags=EXTRA_CUDA_CFLAGS,
        extra_ldflags=["-lcuda"],
        build_directory=build_dir,
        verbose=True,
    )


def _ref_attention(Q, K, V):
    """Standard single-head scaled-dot-product attention in float32."""
    import torch, math
    q, k, v = Q.float(), K.float(), V.float()
    scores = q @ k.T / math.sqrt(q.size(-1))
    P = torch.softmax(scores, dim=-1)
    return (P @ v).bfloat16()


def _test_correctness(mod, sizes: list[int]) -> bool:
    """Run correctness checks against PyTorch reference for each N."""
    import torch

    passed = True
    for N in sizes:
        Q = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)
        O = torch.zeros(N, D, device="cuda", dtype=torch.bfloat16)

        mod.launch_attention(Q, K, V, O)
        torch.cuda.synchronize()

        O_ref = _ref_attention(Q, K, V)
        max_err = (O.float() - O_ref.float()).abs().max().item()
        avg_err = (O.float() - O_ref.float()).abs().mean().item()
        ok = max_err < 1.0
        status = "OK" if ok else "FAIL"
        print(f"  N={N:5d}  max_err={max_err:.4f}  avg_err={avg_err:.6f}  {status}")
        if not ok:
            passed = False
    return passed


def _benchmark(mod, N: int, warmup: int = 10, iters: int = 50) -> float:
    """Benchmark using CUDA events.  Returns median ms."""
    import torch

    Q = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)
    O = torch.zeros(N, D, device="cuda", dtype=torch.bfloat16)

    for _ in range(warmup):
        mod.launch_attention(Q, K, V, O)
    torch.cuda.synchronize()

    timings: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        mod.launch_attention(Q, K, V, O)
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))

    timings.sort()
    median = timings[len(timings) // 2]
    # FLOPs: 2*N*N*D (QK^T) + 4*N*N (softmax) + 2*N*N*D (PV)
    flops = 4 * N * N * D + 4 * N * N
    tflops = flops / (median / 1000) / 1e12
    print(f"  N={N}  median={median:.3f} ms  ({tflops:.2f} TFLOP/s)")
    return median


@app.function(gpu="H100", image=cuda_image, timeout=600)
def test_attention():
    import sys
    sys.path.insert(0, "/root/warpir")

    from tests.attention_baseline import build_attention_kernel
    from warpir.lowering import ThunderKittensLowerer

    # ---- Build IR & lower ----
    kernel = build_attention_kernel()
    cuda_src = ThunderKittensLowerer().lower(kernel)

    print("=== Generated CUDA ===")
    print(cuda_src)
    print()

    # ---- Compile ----
    mod = _compile(cuda_src)

    # ---- Correctness ----
    test_sizes = [64, 128, 256, 512, 1024]
    print("=== Correctness ===")
    ok = _test_correctness(mod, test_sizes)

    # ---- Benchmark ----
    bench_sizes = [128, 256, 512, 1024]
    print()
    print("=== Benchmark ===")
    for N in bench_sizes:
        _benchmark(mod, N)

    print()
    print("PASSED" if ok else "FAILED")
    return ok


@app.local_entrypoint()
def main():
    result = test_attention.remote()
    raise SystemExit(0 if result else 1)
