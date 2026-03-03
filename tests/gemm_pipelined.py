from warpir import *
from warpir.ops import BuiltinExpr

def build_gemm_kernel() -> Program:
    block_size_val = 64
    BLOCK_SIZE = BuiltinExpr("BLOCK_SIZE")
    
    constants = f"""
static constexpr int BLOCK_SIZE = {block_size_val};
"""

    # Define types with programmatic aliases
    sub_tile = SharedTileType(GPUType.bf16, BLOCK_SIZE, BLOCK_SIZE, SharedTileLayout.row_major, alias_name="sub_tile")
    tile_gl = GlobalType(GPUType.bf16, sub_tile, w=1, x=1, y=-1, z=-1, alias_name="tile_gl")

    g = KernelGlobals(
        name="matmul_globals",
        A=tile_gl,
        B=tile_gl,
        C=tile_gl, 
        N=ScalarType("int")
    )
    
    # Shared memory tiles
    As0 = Tile("As0", sub_tile)
    As1 = Tile("As1", sub_tile)
    Bs0 = Tile("Bs0", sub_tile)
    Bs1 = Tile("Bs1", sub_tile)

    # Register accumulator tiles
    C_accum = Var("C_accum", RegTileType(GPUType.fp32, 16, BLOCK_SIZE, RegTileLayout.row_major))
    C_accum_cpy = Var("C_accum_cpy", RegTileType(GPUType.fp32, 16, BLOCK_SIZE, RegTileLayout.row_major))

    row, col = Var("row", ScalarType("int")), Var("col", ScalarType("int"))
    num_tiles = Var("num_tiles", ScalarType("int"))
    
    tile = Var("tile", ScalarType("int"))
    bar = Symbol("bar")

    # Constants for the kernel
    constants = """
static constexpr int BLOCK_SIZE = 64;
static constexpr int NUM_WORKERS = 4;
static constexpr int NUM_THREADS = NUM_WORKERS * 32;
static constexpr int NUM_WARPS = NUM_WORKERS;
"""

    # Kernel body statements
    body = [
        As0.declare(),
        Bs0.declare(),
        As1.declare(),
        Bs1.declare(),
        DeclStmt(row, BuiltinExpr("blockIdx.y")),
        DeclStmt(col, BuiltinExpr("blockIdx.x")),
        As0.load(g.A, Coord(0, 0, row, 0)),
        Bs0.load(g.B, Coord(0, 0, 0, col)),
        OpCall("kittens::warp::zero", C_accum_cpy),
        DeclStmt(num_tiles, (g.N + BLOCK_SIZE - 1) / BLOCK_SIZE),
        DeclStmt(tile),
    ]

    # Straight-line loop body for pipelining
    loop_body = SeqStmt([
        As1.await_empty(),
        Bs1.await_empty(),
        As1.load(g.A, Coord(0, 0, row, tile + 1)),
        Bs1.load(g.B, Coord(0, 0, tile + 1, col)),
        As0.await_loaded(),
        Bs0.await_loaded(),
        MMAOp(C_accum, As0.ref(), Bs0.ref()),
    ])

    for_stmt = ForStmt(
        AssignStmt(tile, 0),
        BinaryOp("<", tile, num_tiles),
        AssignStmt("tile", BinaryOp("+", tile, 2))    
        loop_body
    )
    body.append(for_stmt)
    body.append(OpCall("warpgroup::store", g.C, C_accum_cpy, Coord(0, 0, row, col)))

    return Program(
        kernel_vars=g, 
        kernel_stmt=SeqStmt(body),
        constants=constants,
        grid_dims=Coord("(N + BLOCK_SIZE - 1) / BLOCK_SIZE", "(N + BLOCK_SIZE - 1) / BLOCK_SIZE"),
        block_dims=Symbol("NUM_THREADS"),
        launch_name="matmul"
    )

if __name__ == "__main__":
    print(emit_cpp(build_gemm_kernel()))
