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

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <rmm/device_uvector.hpp>

#include <cstdint>

namespace sirius::cuda::tae {

namespace {

constexpr uint32_t THREADS_PER_BLOCK = 256;
constexpr uint32_t VARLENA_SIZE      = 24;
constexpr uint32_t VARLENA_INLINE_MAX = 23;

// Device-side Varlena reader
struct VarlenaReader {
  const uint8_t* base;       // pointer to first Varlena struct
  const uint8_t* area_base;  // pointer to area section (for big string reads)

  __device__ uint32_t get_length(uint32_t row) const
  {
    const uint8_t* v = base + row * VARLENA_SIZE;
    uint8_t first    = v[0];
    if (first <= VARLENA_INLINE_MAX) {
      return first;
    } else {
      // Big format: bytes 8..11 = length
      uint32_t len;
      memcpy(&len, v + 8, 4);
      return len;
    }
  }

  __device__ void copy_data(uint32_t row, uint8_t* dst) const
  {
    const uint8_t* v = base + row * VARLENA_SIZE;
    uint8_t first    = v[0];
    if (first <= VARLENA_INLINE_MAX) {
      for (uint32_t i = 0; i < first; i++) {
        dst[i] = v[1 + i];
      }
    } else {
      // Big format: bytes 4..7 = offset into area data
      uint32_t offset;
      memcpy(&offset, v + 4, 4);
      uint32_t len;
      memcpy(&len, v + 8, 4);
      for (uint32_t i = 0; i < len; i++) {
        dst[i] = area_base[offset + i];
      }
    }
  }
};

// Kernel: compute per-row string lengths (input for prefix-sum)
__global__ void compute_lengths_kernel(const uint8_t* __restrict__ varlena_base,
                                       const uint8_t* __restrict__ area_base,
                                       int32_t* __restrict__ lengths,
                                       uint32_t n_rows)
{
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  VarlenaReader reader{varlena_base, area_base};
  for (uint32_t i = tid; i < n_rows; i += gridDim.x * blockDim.x) {
    lengths[i] = static_cast<int32_t>(reader.get_length(i));
  }
}

// Kernel: scatter string data using precomputed offsets
__global__ void scatter_chars_kernel(const uint8_t* __restrict__ varlena_base,
                                     const uint8_t* __restrict__ area_base,
                                     const int32_t* __restrict__ offsets,
                                     uint8_t* __restrict__ chars,
                                     uint32_t n_rows)
{
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  VarlenaReader reader{varlena_base, area_base};
  for (uint32_t i = tid; i < n_rows; i += gridDim.x * blockDim.x) {
    int32_t dst_off = offsets[i];
    reader.copy_data(i, chars + dst_off);
  }
}

// Kernel: sum reduction for total chars (used by compute_varchar_total_chars)
__global__ void sum_lengths_kernel(const uint8_t* __restrict__ varlena_base,
                                   const uint8_t* __restrict__ area_base,
                                   uint32_t n_rows,
                                   unsigned long long* __restrict__ total)
{
  __shared__ unsigned long long block_sum;
  if (threadIdx.x == 0) block_sum = 0;
  __syncthreads();

  VarlenaReader reader{varlena_base, area_base};
  unsigned long long thread_sum = 0;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_rows;
       i += gridDim.x * blockDim.x) {
    thread_sum += reader.get_length(i);
  }

  atomicAdd(&block_sum, thread_sum);
  __syncthreads();

  if (threadIdx.x == 0) { atomicAdd(total, block_sum); }
}

}  // anonymous namespace

void decode_varchar(const uint8_t* d_varlena_base,
                    const uint8_t* d_area_base,
                    int32_t* d_offsets,
                    uint8_t* d_chars,
                    void* d_temp_storage,
                    std::size_t& temp_bytes,
                    uint32_t n_rows,
                    rmm::cuda_stream_view stream)
{
  if (n_rows == 0) {
    if (d_offsets) { cudaMemsetAsync(d_offsets, 0, sizeof(int32_t), stream.value()); }
    return;
  }

  uint32_t blocks = (n_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  // Step 1: Compute per-row lengths into d_offsets[0..n_rows-1]
  compute_lengths_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    d_varlena_base, d_area_base, d_offsets, n_rows);

  // Step 2: Exclusive prefix sum → d_offsets[0..n_rows] (length n_rows+1)
  //         d_offsets[n_rows] = total chars
  cub::DeviceScan::ExclusiveSum(
    d_temp_storage, temp_bytes, d_offsets, d_offsets, n_rows + 1, stream.value());

  if (!d_temp_storage) {
    // Size query only — caller will allocate and call again
    return;
  }

  // Step 3: Scatter character data
  scatter_chars_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    d_varlena_base, d_area_base, d_offsets, d_chars, n_rows);
}

// Kernel: add a constant base to each offset value (for multi-block global adjustment)
__global__ void adjust_offsets_kernel(int32_t* __restrict__ offsets, int32_t base, uint32_t count)
{
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x) {
    offsets[i] += base;
  }
}

void adjust_offsets(int32_t* d_offsets, int32_t base, uint32_t count, rmm::cuda_stream_view stream)
{
  if (count == 0 || base == 0) return;
  uint32_t blocks = (count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  adjust_offsets_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(d_offsets, base, count);
}

std::size_t compute_varchar_total_chars(const uint8_t* d_varlena_base,
                                        const uint8_t* d_area_base,
                                        uint32_t n_rows,
                                        rmm::cuda_stream_view stream)
{
  if (n_rows == 0) return 0;

  uint32_t blocks = (n_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  rmm::device_uvector<unsigned long long> d_total(1, stream);
  cudaMemsetAsync(d_total.data(), 0, sizeof(unsigned long long), stream.value());

  sum_lengths_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    d_varlena_base, d_area_base, n_rows, d_total.data());

  unsigned long long h_total = 0;
  cudaMemcpyAsync(&h_total, d_total.data(), sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                  stream.value());
  stream.synchronize();

  return static_cast<std::size_t>(h_total);
}

}  // namespace sirius::cuda::tae
