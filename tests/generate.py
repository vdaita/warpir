#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from warpir.compiler import emit_cpp
from tests.gemm_baseline         import build_gemm_kernel as build_baseline
from tests.gemm_pipelined        import build_gemm_kernel as build_pipelined
from tests.gemm_warp_specialized import build_gemm_kernel as build_warp_specialized

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
LAUNCH_CU   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "launch.cu")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

KERNELS = [
    ("gemm_baseline",         build_baseline),
    ("gemm_pipelined",        build_pipelined),
    ("gemm_warp_specialized", build_warp_specialized),
]

for name, builder in KERNELS:
    src  = emit_cpp(builder())
    path = os.path.join(OUTPUTS_DIR, f"{name}.cu")
    with open(path, "w") as f:
        f.write(src)
        f.write(f'\n#include "{LAUNCH_CU}"\n')
    print(f"[generate] wrote {path}")