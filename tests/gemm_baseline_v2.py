from warpir_v2.ir import Kernel
from warpir_v2.lowering import LoweringPipeline, ThunderKittensLowerer

def build_gemm_kernel() -> Kernel:
    BLOCK_SIZE = 32

    shared_tile_type = SharedTileType(GPUType.bf16, BLOCK_SIZE, BLOCK_SIZE, SharedTileLayout.row_major)
    As = Tile("As", shared_tile_type)
    Bs = Tile("Bs", shared_tile_type)

    global_type = GlobalType(GPUType.bf16, shared_tile_type, 1, 1, -1, -1)
    A = Var("A", global_type)
    B = Var("B", global_type)
    C = Var("C", global_type)
    N = Var("N", ScalarType("int"))