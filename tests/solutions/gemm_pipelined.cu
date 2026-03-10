
#include "kittens.cuh"
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

using namespace kittens;

struct global_vars {
  gl<bf16, 1, 1, -1, -1, st_bf<64, 64>> A;
  gl<bf16, 1, 1, -1, -1, st_bf<64, 64>> B;
  gl<bf16, 1, 1, -1, -1, st_bf<64, 64>> C;
  int N;
};

__global__ void kernel(const __grid_constant__ global_vars globals) {
  __shared__ st_bf<64, 64> As0;
  __shared__ st_bf<64, 64> As1;
  __shared__ st_bf<64, 64> Bs0;
  __shared__ st_bf<64, 64> Bs1;
  rt_fl<16, 64, ducks::rt_layout::row> C_accum;
  rt_fl<16, 64, ducks::rt_layout::row> C_accum_cpy;
  int tile;
  __shared__ semaphore full_lm0;
  __shared__ semaphore empty_lm0;
  int tic_full_lm0;
  int tic_empty_lm0;
  tic_full_lm0 = 0;
  tic_empty_lm0 = 0;
  if ((0 == threadIdx.x)) {
    init_semaphore(full_lm0, 0, 1);
  }
  if ((0 == threadIdx.x)) {
    init_semaphore(empty_lm0, 1, 0);
  }
  __shared__ semaphore full_lm1;
  __shared__ semaphore empty_lm1;
  int tic_full_lm1;
  int tic_empty_lm1;
  tic_full_lm1 = 0;
  tic_empty_lm1 = 0;
  if ((0 == threadIdx.x)) {
    init_semaphore(full_lm1, 0, 1);
  }
  if ((0 == threadIdx.x)) {
    init_semaphore(empty_lm1, 1, 0);
  }
  if ((0 == warpgroup::laneid())) {
    tma::expect_bytes(full_lm0,
                      (size_bytes<typeof(As0)> + size_bytes<typeof(Bs0)>));
    tma::load_async(As0, globals.A, {0, 0, blockIdx.y, 0}, full_lm0);
    tma::load_async(Bs0, globals.B, {0, 0, 0, blockIdx.x}, full_lm0);
  }
  __syncthreads();
  kittens::warp::zero(C_accum);
  kittens::warp::zero(C_accum_cpy);
  for (tile = 0; (tile < ((globals.N + 63) / 64)); tile = (tile + 1)) {
    if (((tile % 2) == 0)) {
      wait(full_lm0, tic_full_lm0);
      tic_full_lm0 = (tic_full_lm0 ^ 1);
      __syncthreads();
      if (((tile + 1) < ((globals.N + 63) / 64))) {
        if ((0 == warpgroup::laneid())) {
          tma::expect_bytes(
              full_lm1, (size_bytes<typeof(As1)> + size_bytes<typeof(Bs1)>));
          tma::load_async(As1, globals.A, {0, 0, blockIdx.y, (tile + 1)},
                          full_lm1);
          tma::load_async(Bs1, globals.B, {0, 0, (tile + 1), blockIdx.x},
                          full_lm1);
        }
      }
      warpgroup::mma_AB(C_accum, As0, Bs0);
      warpgroup::mma_async_wait();
      kittens::warp::add(C_accum_cpy, C_accum_cpy, C_accum);
      kittens::warp::zero(C_accum);
      __syncthreads();
    } else {
      wait(full_lm1, tic_full_lm1);
      tic_full_lm1 = (tic_full_lm1 ^ 1);
      __syncthreads();
      if (((tile + 1) < ((globals.N + 63) / 64))) {
        if ((0 == warpgroup::laneid())) {
          tma::expect_bytes(
              full_lm0, (size_bytes<typeof(As0)> + size_bytes<typeof(Bs0)>));
          tma::load_async(As0, globals.A, {0, 0, blockIdx.y, (tile + 1)},
                          full_lm0);
          tma::load_async(Bs0, globals.B, {0, 0, (tile + 1), blockIdx.x},
                          full_lm0);
        }
      }
      warpgroup::mma_AB(C_accum, As1, Bs1);
      warpgroup::mma_async_wait();
      kittens::warp::add(C_accum_cpy, C_accum_cpy, C_accum);
      kittens::warp::zero(C_accum);
      __syncthreads();
    }
  }
  warpgroup::store(C_accum_cpy, globals.C, {0, 0, blockIdx.y, blockIdx.x});
}

#include "/Users/sbfisher/Stanford/CS343D/final_project/WarpIR/tests/launch.cu"
