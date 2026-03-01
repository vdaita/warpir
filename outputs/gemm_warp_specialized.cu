
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>


#include "kittens.cuh"
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

using namespace kittens;

struct kernel_globals {
  gl<bf16, -1, -1, -1, -1, st_bf<64, 64, ducks::st_layout::row>> A;
  gl<bf16, -1, -1, -1, -1, st_bf<64, 64, ducks::st_layout::row>> B;
  gl<bf16, -1, -1, -1, -1, st_bf<64, 64, ducks::st_layout::row>> C;
  int N;
};

__global__ void kernel(const __grid_constant__ kernel_globals g) {
  int k_index;
  int num_tiles;
  int warpid;
  int warpgroupid;
  int num_consumers = (NUM_THREADS / 128) - 1;
  extern __shared__ alignment_dummy __shm[];
  shared_allocator al((int *)&__shm[0]);
  st_bf<64, 64, ducks::st_layout::row>(&As_0) =
      al.allocate<st_bf<64, 64, ducks::st_layout::row>>();
  st_bf<64, 64, ducks::st_layout::row>(&As_1) =
      al.allocate<st_bf<64, 64, ducks::st_layout::row>>();
  st_bf<64, 64, ducks::st_layout::row>(&Bs_0) =
      al.allocate<st_bf<64, 64, ducks::st_layout::row>>();
  st_bf<64, 64, ducks::st_layout::row>(&Bs_1) =
      al.allocate<st_bf<64, 64, ducks::st_layout::row>>();
  __shared__ semaphore full_As_0, empty_As_0;
  __shared__ semaphore full_As_1, empty_As_1;
  __shared__ semaphore full_Bs_0, empty_Bs_0;
  __shared__ semaphore full_Bs_1, empty_Bs_1;
  if ((threadIdx.x == 0)) {
    init_semaphore(full_As_0, 0, 1);
    init_semaphore(empty_As_0, num_consumers, 0);
    init_semaphore(full_As_1, 0, 1);
    init_semaphore(empty_As_1, num_consumers, 0);
    init_semaphore(full_Bs_0, 0, 1);
    init_semaphore(empty_Bs_0, num_consumers, 0);
    init_semaphore(full_Bs_1, 0, 1);
    init_semaphore(empty_Bs_1, num_consumers, 0);
  }
  __syncthreads();
  if (warpgroupid == 0) {
    warpgroup::decrease_registers<32>();
    for (k_index = 0; (k_index < num_tiles); k_index = (k_index + 1)) {
      wait(empty_As_0);
      if ((warpgroup::laneid() == 0)) {
        tma::load_async(As_0, g.A, {0, 0, blockIdx.x, k_index}, full_As_0);
      }
      if ((warpgroup::laneid() == 0)) {
        tma::expect_bytes(full_As_0, size_bytes<typeof(As_0)>);
      }
      wait(empty_Bs_0);
      if ((warpgroup::laneid() == 0)) {
        tma::load_async(Bs_0, g.B, {0, 0, k_index, blockIdx.y}, full_Bs_0);
      }
      if ((warpgroup::laneid() == 0)) {
        tma::expect_bytes(full_Bs_0, size_bytes<typeof(Bs_0)>);
      }
      wait(empty_As_1);
      if ((warpgroup::laneid() == 0)) {
        tma::load_async(As_1, g.A, {0, 0, blockIdx.x, (k_index + 1)},
                        full_As_1);
      }
      if ((warpgroup::laneid() == 0)) {
        tma::expect_bytes(full_As_1, size_bytes<typeof(As_1)>);
      }
      wait(empty_Bs_1);
      if ((warpgroup::laneid() == 0)) {
        tma::load_async(Bs_1, g.B, {0, 0, (k_index + 1), blockIdx.y},
                        full_Bs_1);
      }
      if ((warpgroup::laneid() == 0)) {
        tma::expect_bytes(full_Bs_1, size_bytes<typeof(Bs_1)>);
      }
    }
  } else if (warpgroupid == 1) {
    warpgroup::increase_registers<256>();
    rt_fl<16, 64, ducks::rt_layout::row> C_accum;
    kittens::warp::zero(C_accum);
    for (k_index = 0; (k_index < num_tiles); k_index = (k_index + 1)) {
      wait(full_As_0);
      wait(full_Bs_0);
      warpgroup::mma_AB(C_accum, As_0, Bs_0);
      warpgroup::mma_async_wait();
      if ((warpgroup::laneid() == 0)) {
        arrive(empty_As_0, 1);
      }
      if ((warpgroup::laneid() == 0)) {
        arrive(empty_Bs_0, 1);
      }
      wait(full_As_1);
      wait(full_Bs_1);
      warpgroup::mma_AB(C_accum, As_1, Bs_1);
      warpgroup::mma_async_wait();
      if ((warpgroup::laneid() == 0)) {
        arrive(empty_As_1, 1);
      }
      if ((warpgroup::laneid() == 0)) {
        arrive(empty_Bs_1, 1);
      }
    }
    tma::store_async(g.C, C_accum, {0, 0, blockIdx.x, blockIdx.y});
  }
}

void launch_kernel(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    TORCH_CHECK(A.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "Tensors must be bfloat16");
    
    int N = A.size(0);
    int BLOCK_SIZE = 64;
    int NUM_WORKERS = 8;
    int NUM_THREADS = (NUM_WORKERS * 32); 
    
    using a_gl = gl<bf16, -1, -1, -1, -1, st_bf<64, 64, ducks::st_layout::row>>;
    a_gl a_arg{reinterpret_cast<bf16*>(A.data_ptr<at::BFloat16>()), nullptr, nullptr, N, N};
    a_gl b_arg{reinterpret_cast<bf16*>(B.data_ptr<at::BFloat16>()), nullptr, nullptr, N, N};
    a_gl c_arg{reinterpret_cast<bf16*>(C.data_ptr<at::BFloat16>()), nullptr, nullptr, N, N};
    
    kernel_globals g{a_arg, b_arg, c_arg, N};
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    unsigned long mem_size = 102400; // 100KB+ shared memory allowance
    
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    kernel<<<blocks, NUM_THREADS, mem_size>>>(g);
    
    AT_CUDA_CHECK(cudaGetLastError());
    AT_CUDA_CHECK(cudaDeviceSynchronize());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch", &launch_kernel, "PyTorch wrapper for generated kernel");
}
