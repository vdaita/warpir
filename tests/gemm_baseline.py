from warpir import *

def build_gemm_kernel() -> Program:
    block_size = 64
    # Shared tiles: 64x64 bf16 tiles for A and B, and a 64x64 bf16 tile for the C output
    ab_type   = SharedTileType(GPUType.bf16, block_size, block_size, SharedTileLayout.row_major)
    c_sh_type = SharedTileType(GPUType.bf16, block_size, block_size, SharedTileLayout.row_major)

    g = KernelGlobals(
        A=GlobalType(GPUType.bf16, ab_type),
        B=GlobalType(GPUType.bf16, ab_type),
        C=GlobalType(GPUType.bf16, c_sh_type),
        N=ScalarType("int"),
    )

    A_sh = Tile("A_sh", ab_type)
    B_sh = Tile("B_sh", ab_type)
    C_sh = Tile("C_sh", c_sh_type)

    row       = Var("row",       ScalarType("int"))
    col       = Var("col",       ScalarType("int"))
    num_tiles = Var("num_tiles", ScalarType("int"))
    k_index   = Var("k_index",   ScalarType("int"))

    # Inner loop body – use RawStmt so we can use exact TK register-tile types
    # and warp-level MMA that compile on both Ampere and Hopper.
    #
    # Each warpgroup (4 warps, 128 threads) cooperatively loads a 64x64 tile into
    # shared memory, then each warp independently handles a 16-row slice via
    # warp::load (shared→register) and warp::mma_AB (fp32 accumulation).
    loop_body = SeqStmt([
        # Global → Shared (warpgroup collaborative, no TMA needed)
        OpCall("warpgroup::load", A_sh, g.A, Coord(0, 0, row, k_index)),
        OpCall("warpgroup::load", B_sh, g.B, Coord(0, 0, k_index, col)),
        RawStmt("__syncthreads();"),
        # Shared → Register + MMA (warp-level; each warp owns 16 rows of A)
        # We accumulate into a local fp32 register tile and add into C_accum.
        RawStmt("""\
{
    rt_fl<16, 64, ducks::rt_layout::row> local_c;
    rt_bf<16, 64, ducks::rt_layout::row> local_a;
    rt_bf<64, 16, ducks::rt_layout::col> local_b;
    warp::load(local_a, A_sh);
    warp::load(local_b, B_sh);
    warp::mma_AB(C_accum, local_a, local_b, C_accum);
}"""),
        RawStmt("__syncthreads();"),
    ])

    for_stmt = ForStmt(
        AssignStmt(k_index, 0),
        k_index < num_tiles,
        AssignStmt(k_index, k_index + 1),
        loop_body,
    )

    # After accumulation: convert fp32 → bf16, spill to shared, then global store.
    store_stmts = SeqStmt([
        RawStmt("""\
{
    rt_bf<16, 64, ducks::rt_layout::row> C_bf;
    warp::copy(C_bf, C_accum);
    warp::store(C_sh, C_bf);
}"""),
        RawStmt("__syncthreads();"),
        OpCall("warpgroup::store", g.C, C_sh, Coord(0, 0, row, col)),
    ])

    return Program(
        input_vars=[],
        kernel_vars=g,
        kernel_stmt=SeqStmt([
            DeclStmt(row, getConst("blockIdx.y")),
            DeclStmt(col, getConst("blockIdx.x")),
            DeclStmt(num_tiles, (g.N + block_size - 1) / block_size),
            A_sh.def_(),
            B_sh.def_(),
            C_sh.def_(),
            # Declare the fp32 accumulator register tile (local to each warp)
            RawStmt("rt_fl<16, 64, ducks::rt_layout::row> C_accum;"),
            OpCall("warp::zero", getConst("C_accum")),
            DeclStmt(k_index),
            for_stmt,
            store_stmts,
        ]),
    )


if __name__ == "__main__":
    print(emit_cpp(build_gemm_kernel()))
