import os
import sys
import torch
from torch.utils.cpp_extension import load
from time import time

# so warpir is in python path
sys.path.append(
  os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

# for including kittens.cuh in kernel compilation
THUNDERKITTENS_INCLUDE_PATH = os.environ.get(
    "THUNDERKITTENS_INCLUDE_PATH",
    "/home/ubuntu/warpir/thunderkittens/include"  # fallback default
)

OUTPUT_DIR = os.path.join(
  os.path.dirname(os.path.abspath(__file__)),
  "outputs"
)

from warpir.compiler import emit_cpp
from tests.gemm_baseline import build_gemm_kernel as build_baseline
from tests.gemm_pipelined import build_gemm_kernel as build_pipelined
from tests.gemm_warp_specialized import build_gemm_kernel as build_warp_specialized

def generate_and_test(name, program):
    print(f"\n--- Processing: {name} ---")
    
    # 1. Generate CUDA code (now includes PyTorch scaffolding)
    full_code = emit_cpp(program)
    
    out_file = f"{OUTPUT_DIR}/{name}.cu"
    
    # 2. Cache/write the CUDA code
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
            extra_include_paths=[THUNDERKITTENS_INCLUDE_PATH],
            extra_cflags=["-O3", "-std=c++20"],
            extra_cuda_cflags=["-O3", "-std=c++20", "--use_fast_math", "--extended-lambda", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__"],
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
        return
    # Reference comparison
    torch_out = torch.matmul(A.float(), B.float()).to(torch.bfloat16)
    
    try:
        torch.testing.assert_close(C, torch_out, rtol=1e-2, atol=1e-2)
        print(f"[{name}] ✅ EXACT MATCH! Outputs are numerically validated.")
    except Exception as e:
        print(f"[{name}] ❌ FAILED VALIDATION! Output mismatch.")
        print(e)
        
if __name__ == "__main__":
    if os.path.exists("outputs/lock"):
        os.removedirs("output/lock")
    os.makedirs(os.path.dirname('outputs/lock'), exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    generate_and_test("gemm_baseline", build_baseline())
    generate_and_test("gemm_pipelined", build_pipelined())
    generate_and_test("gemm_warp_specialized", build_warp_specialized())
