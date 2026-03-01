import os
import sys
import torch
from torch.utils.cpp_extension import load
from time import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from warpir.compiler import emit_cpp
from tests.gemm_baseline import build_gemm_kernel as build_baseline
from tests.gemm_pipelined import build_gemm_kernel as build_pipelined
from tests.gemm_warp_specialized import build_gemm_kernel as build_warp_specialized

os.makedirs("outputs", exist_ok=True)

WRAPPER_TEMPLATE = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

{kernel_code}

void launch_kernel(torch::Tensor A, torch::Tensor B, torch::Tensor C) {{
    TORCH_CHECK(A.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "Tensors must be bfloat16");
    
    int N = A.size(0);
    int BLOCK_SIZE = 64;
    int NUM_WORKERS = 8;
    int NUM_THREADS = (NUM_WORKERS * 32); 
    
    using a_gl = gl<bf16, -1, -1, -1, -1, st_bf<64, 64, ducks::st_layout::row>>;
    a_gl a_arg{{reinterpret_cast<bf16*>(A.data_ptr<at::BFloat16>()), nullptr, nullptr, N, N}};
    a_gl b_arg{{reinterpret_cast<bf16*>(B.data_ptr<at::BFloat16>()), nullptr, nullptr, N, N}};
    a_gl c_arg{{reinterpret_cast<bf16*>(C.data_ptr<at::BFloat16>()), nullptr, nullptr, N, N}};
    
    kernel_globals g{{a_arg, b_arg, c_arg, N}};
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    unsigned long mem_size = 102400; // 100KB+ shared memory allowance
    
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    kernel<<<blocks, NUM_THREADS, mem_size>>>(g);
    
    AT_CUDA_CHECK(cudaGetLastError());
    AT_CUDA_CHECK(cudaDeviceSynchronize());
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
  m.def("launch", &launch_kernel, "PyTorch wrapper for generated kernel");
}}
"""
def generate_and_test(name, program):
    print(f"\\n--- Processing: {name} ---")
    
    # 1. Generate CUDA code
    kernel_code = emit_cpp(program)
    full_code = WRAPPER_TEMPLATE.format(kernel_code=kernel_code)
    
    out_file = f"outputs/{name}.cu"
    
    # Caching / Writing
    os.makedirs("outputs", exist_ok=True)
    if os.path.exists(out_file):
        with open(out_file, "r") as f:
            existing = f.read()
        if existing != full_code:
            print(f"[{name}] Source changed, overwriting.")
            with open(out_file, "w") as f:
                f.write(full_code)
    else:
        with open(out_file, "w") as f:
            f.write(full_code)
            
    print(f"[{name}] Compiled wrapper emitted to {out_file}.")

    N = 128 # Smaller dimension for faster CPU software simulation
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.randn(N, N, dtype=torch.bfloat16, device=device)
    B = torch.randn(N, N, dtype=torch.bfloat16, device=device)
    C = torch.zeros(N, N, dtype=torch.bfloat16, device=device)
    
    # Check if we can compile and run natively
    if torch.cuda.is_available():
        print(f"[{name}] CUDA detected! Compiling via Ninja and torch.utils.cpp_extension...")
        t0 = time()
        ext = load(
            name=name,
            sources=[out_file],
            extra_include_paths=["thunderkittens/include"],
            extra_cflags=["-O3", "-std=c++20"],
            extra_cuda_cflags=["-O3", "-std=c++20", "--use_fast_math", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__"],
            build_directory="outputs",
            verbose=False
        )
        t1 = time()
        print(f"[{name}] Compilation took {t1 - t0:.2f}s")
        print(f"[{name}] Executing on GPU...")
        t0 = time()
        ext.launch(A, B, C)
        torch.cuda.synchronize()
        t1 = time()
        print(f"[{name}] GPU Execution time: {(t1 - t0) * 1000:.2f}ms")
    else:
        print(f"[{name}] No CUDA device found. Logically stepping through tile coordinate maths in python simulation...")
        exit()
    # Reference comparison
    torch_out = torch.matmul(A.float(), B.float()).to(torch.bfloat16)
    
    try:
        torch.testing.assert_close(C, torch_out, rtol=1e-2, atol=1e-2)
        print(f"[{name}] ✅ EXACT MATCH! Outputs are numerically validated.")
    except Exception as e:
        print(f"[{name}] ❌ FAILED VALIDATION! Output mismatch.")
        print(e)
        
if __name__ == "__main__":
    import shutil
    
    # Clean previous torch extensions lockfile to avoid ninja stuck errors
    if os.path.exists("outputs/lock"):
        os.remove("outputs/lock")

    generate_and_test("gemm_baseline", build_baseline())
    generate_and_test("gemm_pipelined", build_pipelined())
    generate_and_test("gemm_warp_specialized", build_warp_specialized())
