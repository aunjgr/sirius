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

// MO null bitmap: bit=1 → NULL.
// cuDF validity mask: bit=1 → VALID.
// Inversion: XOR each 32-bit word with 0xFFFFFFFF.
__global__ void invert_mask_kernel(const uint32_t* __restrict__ src,
                                   uint32_t* __restrict__ dst,
                                   uint32_t n_words)
{
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t i = tid; i < n_words; i += gridDim.x * blockDim.x) {
    dst[i] = src[i] ^ 0xFFFFFFFF;
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
  uint32_t blocks  = (n_words + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  invert_mask_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    reinterpret_cast<const uint32_t*>(d_src), d_dst, n_words);
}

}  // namespace sirius::cuda::tae
