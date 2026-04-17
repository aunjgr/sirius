/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda/tae/tae_decode_kernels.hpp>

#include <cuda_runtime.h>

#include <cstdint>

namespace sirius::cuda::tae {

namespace {

constexpr uint32_t THREADS_PER_BLOCK = 256;

// Generic copy kernel for fixed-width elements (no epoch adjustment)
__global__ void copy_fixed_kernel(const uint8_t* __restrict__ src,
                                  uint8_t* __restrict__ dst,
                                  uint32_t n_rows,
                                  uint32_t elem_size)
{
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  // Process as bytes — each thread handles one byte offset
  uint32_t total_bytes = n_rows * elem_size;
  for (uint32_t i = tid; i < total_bytes; i += gridDim.x * blockDim.x) {
    dst[i] = src[i];
  }
}

// Epoch adjustment kernel for 4-byte types (DATE: int32)
__global__ void adjust_epoch_i32_kernel(int32_t* __restrict__ data,
                                        uint32_t n_rows,
                                        int32_t adjust)
{
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t i = tid; i < n_rows; i += gridDim.x * blockDim.x) {
    data[i] -= adjust;
  }
}

// Epoch adjustment kernel for 8-byte types (TIMESTAMP/DATETIME: int64)
__global__ void adjust_epoch_i64_kernel(int64_t* __restrict__ data,
                                        uint32_t n_rows,
                                        int64_t adjust)
{
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t i = tid; i < n_rows; i += gridDim.x * blockDim.x) {
    data[i] -= adjust;
  }
}

}  // anonymous namespace

void decode_fixed_width(const uint8_t* d_src,
                        uint8_t* d_dst,
                        uint32_t n_rows,
                        uint32_t elem_size,
                        int64_t epoch_adjust,
                        rmm::cuda_stream_view stream)
{
  if (n_rows == 0) return;

  // Caller provides a pointer directly to the data section (past vector header).
  uint32_t total_bytes      = n_rows * elem_size;
  uint32_t blocks           = (total_bytes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  copy_fixed_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    d_src, d_dst, n_rows, elem_size);

  // Apply epoch adjustment if needed
  if (epoch_adjust != 0) {
    uint32_t row_blocks = (n_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (elem_size == 4) {
      adjust_epoch_i32_kernel<<<row_blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
        reinterpret_cast<int32_t*>(d_dst), n_rows, static_cast<int32_t>(epoch_adjust));
    } else if (elem_size == 8) {
      adjust_epoch_i64_kernel<<<row_blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
        reinterpret_cast<int64_t*>(d_dst), n_rows, epoch_adjust);
    }
  }
}

}  // namespace sirius::cuda::tae
