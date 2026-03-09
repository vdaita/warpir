#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUTS_DIR="$SCRIPT_DIR/outputs"
KERNELS=(gemm_baseline gemm_pipelined gemm_warp_specialized)
FAILED=()

# ── codegen ───────────────────────────────────────────────────────────────────
echo "=== Generating CUDA sources ==="
uv run "$SCRIPT_DIR/generate.py"

# ── compile & test ────────────────────────────────────────────────────────────
for KERNEL in "${KERNELS[@]}"; do
    echo ""
    echo "=== Building $KERNEL ==="
    make -C "$SCRIPT_DIR" "outputs/$KERNEL.out"

    echo "=== Testing $KERNEL ==="
    if "$OUTPUTS_DIR/$KERNEL.out"; then
        echo "  ✓ $KERNEL passed"
    else
        echo "  ✗ $KERNEL FAILED"
        FAILED+=("$KERNEL")
    fi
done

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
echo "================================="
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All kernels passed ✓"
    exit 0
else
    echo "FAILED: ${FAILED[*]}"
    exit 1
fi