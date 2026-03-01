// Forward-only, simplified MHA kernel for H100.
// Keeps the same tile shapes as mha_h100.cu, but removes
// software pipelining and producer/consumer warp specialization.

#include "kittens.cuh"
#include <cooperative_groups.h>
#include <iostream>

constexpr int SIMPLE_CONSUMER_WARPGROUPS = (3);
constexpr int SIMPLE_NUM_WARPGROUPS = (SIMPLE_CONSUMER_WARPGROUPS);
constexpr int SIMPLE_NUM_WORKERS = (SIMPLE_NUM_WARPGROUPS * kittens::WARPGROUP_WARPS);

using namespace kittens;
namespace cg = cooperative_groups;

template<int D> struct fwd_attend_ker_tile_dims_simple {};
template<> struct fwd_attend_ker_tile_dims_simple<64> {
    constexpr static int tile_width = (64);
    constexpr static int qo_height = (4 * 16);
    constexpr static int kv_height = (8 * 16);
};
template<> struct fwd_attend_ker_tile_dims_simple<128> {
    constexpr static int tile_width = (128);
    constexpr static int qo_height = (4 * 16);
    constexpr static int kv_height = (8 * 16);
};

template<int D> struct fwd_globals_simple {
    using q_tile = st_bf<fwd_attend_ker_tile_dims_simple<D>::qo_height, fwd_attend_ker_tile_dims_simple<D>::tile_width>;
    using k_tile = st_bf<fwd_attend_ker_tile_dims_simple<D>::kv_height, fwd_attend_ker_tile_dims_simple<D>::tile_width>;
    using v_tile = st_bf<fwd_attend_ker_tile_dims_simple<D>::kv_height, fwd_attend_ker_tile_dims_simple<D>::tile_width>;
    using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims_simple<D>::qo_height, fwd_attend_ker_tile_dims_simple<D>::tile_width>>;
    using o_tile = st_bf<fwd_attend_ker_tile_dims_simple<D>::qo_height, fwd_attend_ker_tile_dims_simple<D>::tile_width>;

    using q_gl = gl<bf16, -1, -1, -1, -1, q_tile>;
    using k_gl = gl<bf16, -1, -1, -1, -1, k_tile>;
    using v_gl = gl<bf16, -1, -1, -1, -1, v_tile>;
    using l_gl = gl<float, -1, -1, -1, -1, l_col_vec>;
    using o_gl = gl<bf16, -1, -1, -1, -1, o_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    l_gl l;
    o_gl o;

    const int N;
    const int hr;
};

template<int D, bool is_causal>
__global__ __launch_bounds__((SIMPLE_NUM_WORKERS) * kittens::WARP_THREADS, 1)
void fwd_attend_ker_simple(const __grid_constant__ fwd_globals_simple<D> g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);

    const int warpid = kittens::warpid();
    const int warpgroupid = warpid / kittens::WARPGROUP_WARPS;
    using K = fwd_attend_ker_tile_dims_simple<D>;

    using q_tile = st_bf<K::qo_height, K::tile_width>;
    using k_tile = st_bf<K::kv_height, K::tile_width>;
    using v_tile = st_bf<K::kv_height, K::tile_width>;
    using l_col_vec = col_vec<st_fl<K::qo_height, K::tile_width>>;
    using o_tile = st_bf<K::qo_height, K::tile_width>;

    q_tile (&q_smem)[SIMPLE_CONSUMER_WARPGROUPS] = al.allocate<q_tile, SIMPLE_CONSUMER_WARPGROUPS>();
    k_tile (&k_smem) = al.allocate<k_tile>();
    v_tile (&v_smem) = al.allocate<v_tile>();
    l_col_vec (&l_smem)[SIMPLE_CONSUMER_WARPGROUPS] = al.allocate<l_col_vec, SIMPLE_CONSUMER_WARPGROUPS>();
    auto (*o_smem) = reinterpret_cast<o_tile(*)>(q_smem);

    const int kv_blocks = g.N / (K::kv_height);
    const int kv_head_idx = blockIdx.y / g.hr;
    const int seq_idx = blockIdx.x * SIMPLE_CONSUMER_WARPGROUPS;

    __shared__ kittens::semaphore qsmem_semaphore, kv_smem_semaphore;
    if (threadIdx.x == 0) {
        init_semaphore(qsmem_semaphore, 0, 1);
        init_semaphore(kv_smem_semaphore, 0, 1);

        tma::expect_bytes(qsmem_semaphore, sizeof(q_smem));
        for (int wg = 0; wg < SIMPLE_CONSUMER_WARPGROUPS; wg++) {
            coord<q_tile> q_tile_idx = {blockIdx.z, blockIdx.y, seq_idx + wg, 0};
            tma::load_async(q_smem[wg], g.q, q_tile_idx, qsmem_semaphore);
        }
    }
    __syncthreads();

    wait(qsmem_semaphore, 0);

    rt_fl<16, K::kv_height> att_block;
    rt_bf<16, K::kv_height> att_block_mma;
    rt_fl<16, K::tile_width> o_reg;
    col_vec<rt_fl<16, K::kv_height>> max_vec, norm_vec, max_vec_last_scaled, max_vec_scaled;

    warp::neg_infty(max_vec);
    warp::zero(norm_vec);
    warp::zero(o_reg);

    int kv_iters;
    if constexpr (is_causal) {
        kv_iters = (seq_idx * 4) - 1 + (SIMPLE_CONSUMER_WARPGROUPS * 4);
        kv_iters = (kv_iters / 8);
    } else {
        kv_iters = kv_blocks - 1;
    }

    for (int kv_idx = 0; kv_idx <= kv_iters; kv_idx++) {
        if (threadIdx.x == 0) {
            coord<k_tile> kv_tile_idx = {blockIdx.z, kv_head_idx, kv_idx, 0};
            tma::expect_bytes(kv_smem_semaphore, sizeof(k_tile) + sizeof(v_tile));
            tma::load_async(k_smem, g.k, kv_tile_idx, kv_smem_semaphore);
            tma::load_async(v_smem, g.v, kv_tile_idx, kv_smem_semaphore);
        }
        __syncthreads();
        wait(kv_smem_semaphore, kv_idx % 2);

        warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem);
        warp::copy(max_vec_last_scaled, max_vec);
        if constexpr (D == 64) {
            warp::mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f * 0.125f);
        } else {
            warp::mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f * 0.08838834764f);
        }

        warpgroup::mma_async_wait();

        if constexpr (is_causal) {
            const int q_blk = (seq_idx * (K::qo_height / kittens::TILE_ROW_DIM<bf16>)) + warpid;
            int k_blk = (kv_idx * (K::kv_height / kittens::TILE_ROW_DIM<bf16>));

            #pragma unroll
            for (; k_blk == (kv_iters - 1) * (K::kv_height / kittens::TILE_ROW_DIM<bf16>) ||
                   k_blk == (kv_iters) * (K::kv_height / kittens::TILE_ROW_DIM<bf16>);
                   k_blk += 10000) {
                #pragma unroll
                for (int j = 0; j < (K::kv_height / kittens::TILE_ROW_DIM<bf16>); j++) {
                    const int k_idx = k_blk + j;
                    auto &attn_subtile = reinterpret_cast<rt_fl<16, 16> &>(att_block.tiles[0][j]);
                    if (k_idx > q_blk) {
                        warp::neg_infty(attn_subtile);
                    } else if (k_idx == q_blk) {
                        warp::make_causal(attn_subtile, attn_subtile, kittens::base_types::constants<float>::neg_infty());
                    }
                    __syncwarp();
                }
            }
        }

        warp::row_max(max_vec, att_block, max_vec);

        if constexpr (D == 64) {
            warp::mul(att_block, att_block, 1.44269504089f * 0.125f);
            warp::mul(max_vec_scaled, max_vec, 1.44269504089f * 0.125f);
        } else {
            warp::mul(att_block, att_block, 1.44269504089f * 0.08838834764f);
            warp::mul(max_vec_scaled, max_vec, 1.44269504089f * 0.08838834764f);
        }

        warp::sub_row(att_block, att_block, max_vec_scaled);
        warp::exp2(att_block, att_block);
        warp::sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
        warp::exp2(max_vec_last_scaled, max_vec_last_scaled);
        warp::mul(norm_vec, norm_vec, max_vec_last_scaled);
        warp::row_sum(norm_vec, att_block, norm_vec);
        warp::add(att_block, att_block, 0.f);
        warp::copy(att_block_mma, att_block);
        warp::mul_row(o_reg, o_reg, max_vec_last_scaled);

        warpgroup::mma_AB(o_reg, att_block_mma, v_smem);
        warpgroup::mma_async_wait();

        __syncthreads();
    }

    warp::div_row(o_reg, o_reg, norm_vec);
    warpgroup::store(o_smem[warpgroupid], o_reg);
    warpgroup::sync(warpgroupid + 4);

    if (warpid % 4 == 0) {
        coord<o_tile> o_tile_idx = {blockIdx.z, blockIdx.y, seq_idx + warpgroupid, 0};
        warp::tma::store_async(g.o, o_smem[warpgroupid], o_tile_idx);
    }

    warp::mul(max_vec_scaled, max_vec_scaled, 0.69314718056f);
    warp::log(norm_vec, norm_vec);
    warp::add(norm_vec, norm_vec, max_vec_scaled);

    if constexpr (D == 64) {
        warp::mul(norm_vec, norm_vec, -8.0f);
    } else {
        warp::mul(norm_vec, norm_vec, -11.313708499f);
    }

    warpgroup::store(l_smem[warpgroupid], norm_vec);
    warpgroup::sync(warpgroupid + 4);

    if (warpid % 4 == 0) {
        coord<l_col_vec> tile_idx = {blockIdx.z, blockIdx.y, 0, seq_idx + warpgroupid};
        warp::tma::store_async(g.l, l_smem[warpgroupid], tile_idx);
    }
    warp::tma::store_async_wait();
}
