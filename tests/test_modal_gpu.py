"""Test WarpIR GEMM kernels (baseline + pipelined) on an H100 via Modal.

Usage:
    uv run modal run tests/test_modal_gpu.py
"""
import modal

BLOCK_SIZE = 64
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

LAUNCH_WRAPPER_BASELINE = f"""
void launch_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C) {{
    size_t N = A.size(0);
    using tile_gl = gl<bf16, 1, 1, -1, -1, st_bf<{BLOCK_SIZE}, {BLOCK_SIZE}>>;
    tile_gl a_gl{{(bf16*)A.data_ptr(), nullptr, nullptr, N, N}};
    tile_gl b_gl{{(bf16*)B.data_ptr(), nullptr, nullptr, N, N}};
    tile_gl c_gl{{(bf16*)C.data_ptr(), nullptr, nullptr, N, N}};
    global_vars g{{a_gl, b_gl, (int)N, c_gl}};
    dim3 grid(N / {BLOCK_SIZE}, N / {BLOCK_SIZE});
    gemm<<<grid, {NUM_THREADS}>>>(g);
}}
"""

LAUNCH_WRAPPER_PIPELINED = f"""
void launch_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C) {{
    size_t N = A.size(0);
    using tile_gl = gl<bf16, 1, 1, -1, -1, st_bf<{BLOCK_SIZE}, {BLOCK_SIZE}>>;
    tile_gl a_gl{{(bf16*)A.data_ptr(), nullptr, nullptr, N, N}};
    tile_gl b_gl{{(bf16*)B.data_ptr(), nullptr, nullptr, N, N}};
    tile_gl c_gl{{(bf16*)C.data_ptr(), nullptr, nullptr, N, N}};
    global_vars g{{a_gl, b_gl, (int)N, c_gl}};
    dim3 grid(N / {BLOCK_SIZE}, N / {BLOCK_SIZE});
    unsigned long mem_size = 100000;
    cudaFuncSetAttribute(gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    gemm<<<grid, {NUM_THREADS}, mem_size>>>(g);
}}
"""

CPP_DECL = """
#include <torch/extension.h>
void launch_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C);
"""

cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .pip_install("torch", "ninja", "setuptools", "ortools")
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

app = modal.App("warpir-gemm-test")


def _compile(name: str, cuda_kernel: str, wrapper: str):
    """JIT-compile a TK kernel via torch load_inline."""
    import os
    from torch.utils.cpp_extension import load_inline

    cuda_src = "#include <torch/extension.h>\n" + cuda_kernel + wrapper
    build_dir = f"/tmp/warpir_build/{name}"
    os.makedirs(build_dir, exist_ok=True)

    print(f"Compiling {name}...")
    return load_inline(
        name=name,
        cpp_sources=[CPP_DECL],
        cuda_sources=[cuda_src],
        functions=["launch_gemm"],
        extra_cuda_cflags=EXTRA_CUDA_CFLAGS,
        extra_ldflags=["-lcuda"],
        build_directory=build_dir,
        verbose=True,
    )


def _test_correctness(mod, label: str, sizes: list[int]) -> bool:
    """Run correctness checks against torch.mm for each N."""
    import torch

    passed = True
    for N in sizes:
        A = torch.randn(N, N, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(N, N, device="cuda", dtype=torch.bfloat16)
        C = torch.zeros(N, N, device="cuda", dtype=torch.bfloat16)

        mod.launch_gemm(A, B, C)
        torch.cuda.synchronize()

        ref = torch.mm(A.float(), B.float()).bfloat16()
        max_err = (C.float() - ref.float()).abs().max().item()
        avg_err = (C.float() - ref.float()).abs().mean().item()
        ok = max_err < 1.0
        status = "OK" if ok else "FAIL"
        print(f"  [{label}] N={N:5d}  max_err={max_err:.4f}  avg_err={avg_err:.6f}  {status}")
        if not ok:
            passed = False
    return passed


def _benchmark(mod, label: str, N: int, warmup: int = 10, iters: int = 50) -> float:
    """Benchmark using CUDA events.  Returns median ms."""
    import torch

    A = torch.randn(N, N, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, N, device="cuda", dtype=torch.bfloat16)
    C = torch.zeros(N, N, device="cuda", dtype=torch.bfloat16)

    for _ in range(warmup):
        mod.launch_gemm(A, B, C)
    torch.cuda.synchronize()

    timings: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        mod.launch_gemm(A, B, C)
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))

    timings.sort()
    median = timings[len(timings) // 2]
    tflops = (2 * N * N * N) / (median / 1000) / 1e12
    print(f"  [{label}] N={N}  median={median:.3f} ms  ({tflops:.2f} TFLOP/s)")
    return median


@app.function(gpu="H100", image=cuda_image, timeout=600)
def test_gemm():
    import sys
    sys.path.insert(0, "/root/warpir")

    from tests.gemm_baseline_v2 import build_gemm_kernel
    from warpir.lowering import ThunderKittensLowerer
    from warpir.passes.modulo_scheduler import kernel_pass

    # ---- Build IR ----
    kernel = build_gemm_kernel()
    pipelined_kernel = kernel_pass(kernel)

    # ---- Lower to CUDA ----
    baseline_cuda = ThunderKittensLowerer().lower(kernel)
    pipelined_cuda = ThunderKittensLowerer().lower(pipelined_kernel)

    print("=== Baseline CUDA ===")
    print(baseline_cuda)
    print()
    print("=== Pipelined CUDA ===")
    print(pipelined_cuda)
    print()

    # ---- Compile ----
    baseline_mod = _compile("gemm_baseline", baseline_cuda, LAUNCH_WRAPPER_BASELINE)
    pipelined_mod = _compile("gemm_pipelined", pipelined_cuda, LAUNCH_WRAPPER_PIPELINED)

    # ---- Correctness ----
    test_sizes = [128, 256, 512, 1024]
    print("=== Correctness ===")
    ok_baseline = _test_correctness(baseline_mod, "baseline", test_sizes)
    ok_pipelined = _test_correctness(pipelined_mod, "pipelined", test_sizes)

    # ---- Benchmark ----
    bench_sizes = [256, 512, 1024, 2048, 4096]
    print()
    print("=== Benchmark ===")
    for N in bench_sizes:
        _benchmark(baseline_mod, "baseline", N)
        _benchmark(pipelined_mod, "pipelined", N)
        print()

    # ---- Summary ----
    all_passed = ok_baseline and ok_pipelined
    print("PASSED" if all_passed else "FAILED")
    return all_passed


@app.local_entrypoint()
def main():
    result = test_gemm.remote()
    raise SystemExit(0 if result else 1)
