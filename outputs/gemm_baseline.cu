
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
  int row = blockIdx.y;
  int col = blockIdx.x;
  int num_tiles = (((g.N + 64) - 1) / 64);
  st_bf<64, 64, ducks::st_layout::row>(&A) =
      al.allocate<st_bf<64, 64, ducks::st_layout::row>>();
  st_bf<64, 64, ducks::st_layout::row>(&B) =
      al.allocate<st_bf<64, 64, ducks::st_layout::row>>();
  rt_fl<16, 64, ducks::rt_layout::row> C_accum;
  kittens::warp::zero(C_accum);
  int k_index;
  for (k_index = 0; (k_index < num_tiles); k_index = (k_index + 1)) {
    tma::load_async(A, g.A, {0, 0, row, k_index});
    tma::load_async(B, g.B, {0, 0, k_index, col});
    warpgroup::mma_AB(C_accum, A, B);
    warpgroup::mma_async_wait();
  }
  tma::store_async(g.C, C_accum, {0, 0, row, col});
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
