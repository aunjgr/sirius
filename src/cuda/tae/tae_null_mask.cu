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

#include <algorithm>
#include <cstdint>

namespace sirius::cuda::tae {

namespace {

constexpr uint32_t THREADS_PER_BLOCK = 256;

// MO null bitmap: bit=1 → NULL.
// cuDF validity mask: bit=1 → VALID.
// Inversion: bitwise NOT each 32-bit word.
// Uses 128-bit vectorized loads/stores for better memory throughput.
__global__ void invert_mask_kernel(const uint32_t* __restrict__ src,
                                   uint32_t* __restrict__ dst,
                                   uint32_t n_words)
{
  // Vectorized path: 4 words per iteration
  uint32_t vec_count = n_words / 4;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_count;
       i += gridDim.x * blockDim.x) {
    uint4 s = reinterpret_cast<const uint4*>(src)[i];
    uint4 d = {~s.x, ~s.y, ~s.z, ~s.w};
    reinterpret_cast<uint4*>(dst)[i] = d;
  }
  // Scalar tail for remaining 0-3 words
  uint32_t base = vec_count * 4;
  for (uint32_t i = base + blockIdx.x * blockDim.x + threadIdx.x; i < n_words;
       i += gridDim.x * blockDim.x) {
    dst[i] = ~src[i];
  }
}

}  // anonymous namespace

void invert_null_mask(const uint8_t* d_src,
                      uint32_t* d_dst,
                      uint32_t n_rows,
                      rmm::cuda_stream_view stream)
{
  if (n_rows == 0) return;

  // Number of 32-bit words needed: ceil(n_rows / 32)
  uint32_t n_words = (n_rows + 31) / 32;
  uint32_t vec_count = n_words / 4;
  uint32_t launch_count = std::max(vec_count, n_words);
  uint32_t blocks  = (launch_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  invert_mask_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    reinterpret_cast<const uint32_t*>(d_src), d_dst, n_words);
}

}  // namespace sirius::cuda::tae
