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

#include <cudf/utilities/error.hpp>

#include <cuda_runtime.h>

#include <cstdint>

namespace sirius::cuda::tae {

namespace {

constexpr uint32_t THREADS_PER_BLOCK = 256;

// Fused copy + epoch adjustment kernel for 4-byte types (DATE: int32)
// Source may be unaligned (MO vector header is 29 bytes), destination is aligned.
__global__ void copy_and_adjust_i32_kernel(const uint8_t* __restrict__ src,
                                           int32_t* __restrict__ dst,
                                           uint32_t n_rows,
                                           int32_t adjust)
{
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t i = tid; i < n_rows; i += gridDim.x * blockDim.x) {
    int32_t val;
    memcpy(&val, src + i * sizeof(int32_t), sizeof(int32_t));
    dst[i] = val - adjust;
  }
}

// Fused copy + epoch adjustment kernel for 8-byte types (TIMESTAMP/DATETIME: int64)
// Source may be unaligned (MO vector header is 29 bytes), destination is aligned.
__global__ void copy_and_adjust_i64_kernel(const uint8_t* __restrict__ src,
                                           int64_t* __restrict__ dst,
                                           uint32_t n_rows,
                                           int64_t adjust)
{
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t i = tid; i < n_rows; i += gridDim.x * blockDim.x) {
    int64_t val;
    memcpy(&val, src + i * sizeof(int64_t), sizeof(int64_t));
    dst[i] = val - adjust;
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

  if (epoch_adjust == 0) {
    // No adjustment — caller already uses cudaMemcpyAsync D2D for this case,
    // but handle it here as fallback.
    CUDF_CUDA_TRY(cudaMemcpyAsync(d_dst, d_src, n_rows * elem_size,
                                   cudaMemcpyDeviceToDevice, stream.value()));
  } else {
    // Fused copy + epoch adjustment in a single pass (halves memory bandwidth)
    uint32_t row_blocks = (n_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (elem_size == 4) {
      copy_and_adjust_i32_kernel<<<row_blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
        d_src, reinterpret_cast<int32_t*>(d_dst), n_rows, static_cast<int32_t>(epoch_adjust));
    } else if (elem_size == 8) {
      copy_and_adjust_i64_kernel<<<row_blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
        d_src, reinterpret_cast<int64_t*>(d_dst), n_rows, epoch_adjust);
    } else {
      // Fallback for unusual element sizes with epoch adjustment — shouldn't happen
      CUDF_CUDA_TRY(cudaMemcpyAsync(d_dst, d_src, n_rows * elem_size,
                                     cudaMemcpyDeviceToDevice, stream.value()));
    }
  }
}

}  // namespace sirius::cuda::tae
