"""Build GEMM IR -> lower to ThunderKittens -> compile -> run -> verify against PyTorch."""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.cpp_extension import load_inline

from tests.gemm_baseline_v2 import build_gemm_kernel
from warpir.lowering import ThunderKittensLowerer

WARPIR_ROOT = Path(__file__).resolve().parent.parent
TK_INCLUDE = str(WARPIR_ROOT / "thunderkittens" / "include")

BLOCK_SIZE = 64
NUM_THREADS = 128  # 1 warpgroup = 4 warps * 32 threads

LAUNCH_WRAPPER = f"""
void launch_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C) {{
    int N = A.size(0);
    using tile_gl = gl<bf16, 1, 1, -1, -1, st_bf<{BLOCK_SIZE}, {BLOCK_SIZE}>>;
    tile_gl a_gl{{(bf16*)A.data_ptr(), nullptr, nullptr, N, N}};
    tile_gl b_gl{{(bf16*)B.data_ptr(), nullptr, nullptr, N, N}};
    tile_gl c_gl{{(bf16*)C.data_ptr(), nullptr, nullptr, N, N}};
    global_vars g{{a_gl, b_gl, (int)N, c_gl}};
    dim3 grid(N / {BLOCK_SIZE}, N / {BLOCK_SIZE});
    gemm<<<grid, {NUM_THREADS}>>>(g);
}}
"""

CPP_DECL = """
#include <torch/extension.h>
void launch_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C);
"""

EXTRA_CUDA_CFLAGS = [
    "-std=c++20", "-O3", "--use_fast_math",
    "--expt-extended-lambda", "--expt-relaxed-constexpr",
    "-DKITTENS_HOPPER",
    "-gencode", "arch=compute_90a,code=sm_90a",
    f"-I{TK_INCLUDE}",
    "-DNDEBUG",
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
]


def compile_kernel():
    kernel_ir = build_gemm_kernel()
    cuda_kernel = ThunderKittensLowerer().lower(kernel_ir)

    cuda_src = (
        '#include <torch/extension.h>\n'
        + cuda_kernel
        + LAUNCH_WRAPPER
    )

    build_dir = str(WARPIR_ROOT / "v2_tests" / "build")
    os.makedirs(build_dir, exist_ok=True)

    print("Compiling kernel...")
    return load_inline(
        name="gemm_baseline",
        cpp_sources=[CPP_DECL],
        cuda_sources=[cuda_src],
        functions=["launch_gemm"],
        extra_cuda_cflags=EXTRA_CUDA_CFLAGS,
        build_directory=build_dir,
        verbose=True,
    )


def test_gemm(mod, N=256):
    A = torch.randn(N, N, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, N, device="cuda", dtype=torch.bfloat16)
    C = torch.zeros(N, N, device="cuda", dtype=torch.bfloat16)

    mod.launch_gemm(A, B, C)
    torch.cuda.synchronize()

    ref = torch.mm(A.float(), B.float()).bfloat16()
    max_err = (C.float() - ref.float()).abs().max().item()
    avg_err = (C.float() - ref.float()).abs().mean().item()

    print(f"  N={N}  max_err={max_err:.4f}  avg_err={avg_err:.6f}")
    return max_err


def main():
    mod = compile_kernel()

    passed = True
    for N in [64, 128, 256, 512]:
        err = test_gemm(mod, N)
        if err > 1.0:
            print(f"  FAIL at N={N}")
            passed = False

    print()
    print("PASSED" if passed else "FAILED")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
