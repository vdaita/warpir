
#include "kittens.cuh"
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

using namespace kittens;

struct global_vars {
  gl<bf16, 1, 1, -1, -1, st_bf<32, 32>> A;
  gl<bf16, 1, 1, -1, -1, st_bf<32, 32>> B;
  gl<bf16, 1, 1, -1, -1, st_bf<32, 32>> C;
  int N;
};

__global__ void kernel(const __grid_constant__ global_vars globals) {
  __shared__ st_bf<32, 32> As;
  __shared__ st_bf<32, 32> Bs;
  rt_bf<32, 32, ducks::rt_layout::row> A_reg;
  rt_bf<32, 32, ducks::rt_layout::row> B_reg;
  rt_bf<32, 32, ducks::rt_layout::col> B_reg_col;
  rt_fl<32, 32, ducks::rt_layout::row> C_accum;
  int tile;
  kittens::warp::zero(C_accum);
tile = (tile + 1)) {
    kittens::warp::load(As, globals.A, {0, 0, blockIdx.y, tile});
    __syncthreads();
    kittens::warp::load(Bs, globals.B, {0, 0, blockIdx.y, tile});
    __syncthreads();
    kittens::warp::load(A_reg, As);
    __syncthreads();
    kittens::warp::load(B_reg, Bs);
    __syncthreads();
    kittens::warp::swap_layout(B_reg_col, B_reg);
    __syncthreads();
    kittens::warp::mma_AB(C_accum, A_reg, B_reg_col, C_accum);
    __syncthreads();
  }
  kittens::warp::store(C_accum, globals.C, {0, 0, blockIdx.y, blockIdx.x});
}

#include "/Users/sbfisher/Stanford/CS343D/final_project/WarpIR/tests/launch.cu"
