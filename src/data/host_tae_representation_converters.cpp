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

// sirius
#include <cuda/tae/tae_decode_kernels.hpp>
#include <data/host_tae_representation.hpp>
#include <data/host_tae_representation_converters.hpp>
#include <log/logging.hpp>
#include <tae/tae_format.hpp>

// cucascade
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>

// cudf
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

// nvcomp
#include <nvcomp/lz4.h>

// rmm
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

// CUDA
#include <cuda_runtime.h>

// cudf error checking
#include <cudf/utilities/error.hpp>

// standard library
#include <algorithm>
#include <cassert>
#include <cstring>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace sirius {

namespace detail {

// ---------------------------------------------------------------------------
// MO type → cuDF type mapping
// ---------------------------------------------------------------------------
static cudf::data_type mo_oid_to_cudf_type(tae::MOTypeOid oid, int32_t scale = 0)
{
  switch (oid) {
    case tae::MO_T_bool:    return cudf::data_type{cudf::type_id::BOOL8};
    case tae::MO_T_int8:    return cudf::data_type{cudf::type_id::INT8};
    case tae::MO_T_int16:   return cudf::data_type{cudf::type_id::INT16};
    case tae::MO_T_int32:   return cudf::data_type{cudf::type_id::INT32};
    case tae::MO_T_int64:   return cudf::data_type{cudf::type_id::INT64};
    case tae::MO_T_uint8:   return cudf::data_type{cudf::type_id::UINT8};
    case tae::MO_T_uint16:  return cudf::data_type{cudf::type_id::UINT16};
    case tae::MO_T_uint32:  return cudf::data_type{cudf::type_id::UINT32};
    case tae::MO_T_uint64:  return cudf::data_type{cudf::type_id::UINT64};
    case tae::MO_T_float32: return cudf::data_type{cudf::type_id::FLOAT32};
    case tae::MO_T_float64: return cudf::data_type{cudf::type_id::FLOAT64};
    case tae::MO_T_date:    return cudf::data_type{cudf::type_id::TIMESTAMP_DAYS};
    case tae::MO_T_datetime: return cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS};
    case tae::MO_T_char:
    case tae::MO_T_varchar:
    case tae::MO_T_text:
    case tae::MO_T_blob:
    case tae::MO_T_json:    return cudf::data_type{cudf::type_id::STRING};
    case tae::MO_T_decimal64:  return cudf::data_type{cudf::type_id::DECIMAL64, -scale};
    case tae::MO_T_decimal128: return cudf::data_type{cudf::type_id::DECIMAL128, -scale};
    default:                return cudf::data_type{cudf::type_id::INT64};
  }
}

// Fixed-width byte size for a MO type OID
static uint32_t mo_oid_fixed_width(tae::MOTypeOid oid)
{
  switch (oid) {
    case tae::MO_T_bool:
    case tae::MO_T_int8:
    case tae::MO_T_uint8:    return 1;
    case tae::MO_T_int16:
    case tae::MO_T_uint16:   return 2;
    case tae::MO_T_int32:
    case tae::MO_T_uint32:
    case tae::MO_T_float32:
    case tae::MO_T_date:     return 4;
    case tae::MO_T_int64:
    case tae::MO_T_uint64:
    case tae::MO_T_float64:
    case tae::MO_T_datetime:
    case tae::MO_T_decimal64: return 8;
    case tae::MO_T_decimal128: return 16;
    default: return 0;  // variable-width
  }
}

static bool is_varchar_type(tae::MOTypeOid oid)
{
  switch (oid) {
    case tae::MO_T_char:
    case tae::MO_T_varchar:
    case tae::MO_T_text:
    case tae::MO_T_blob:
    case tae::MO_T_json: return true;
    default: return false;
  }
}

static bool is_date_type(tae::MOTypeOid oid) { return oid == tae::MO_T_date; }
static bool is_timestamp_type(tae::MOTypeOid oid) { return oid == tae::MO_T_datetime; }

// ---------------------------------------------------------------------------
// MO serialized vector header offsets.
// After LZ4 decompression, each column chunk is:
//   [IOEntryHeader 4B][class 1B][MOType 16B][row_count 4B][dataLen 4B]
//   [data dataLen B][areaLen 4B][area areaLen B]
//   [nspLen 4B][nsp nspLen B][sorted 1B]
//
// Data section starts at byte 29 from the decompressed start.
// For fixed-width columns: dataLen = row_count * elem_size, areaLen = 0
// For varchar columns:     dataLen = row_count * 24 (varlena structs), area = string payloads
// ---------------------------------------------------------------------------
constexpr uint32_t VEC_HEADER_SIZE = 4 + 1 + 16 + 4 + 4;  // 29 bytes to data start
constexpr uint32_t VARLENA_STRUCT_SIZE = 24;

// ---------------------------------------------------------------------------
// Convert host_tae_representation → gpu_table_representation
// ---------------------------------------------------------------------------
std::unique_ptr<cucascade::idata_representation> convert_host_tae_to_gpu(
  cucascade::idata_representation& source,
  cucascade::memory::memory_space const* target_memory_space,
  rmm::cuda_stream_view stream)
{
  auto& host_src = source.cast<host_tae_representation>();
  auto const& chunks = host_src.get_column_chunks();
  auto const& post_filter_projection_ids = host_src.get_post_filter_projection_ids();

  if (chunks.empty()) {
    auto empty_table = std::make_unique<cudf::table>();
    return std::make_unique<cucascade::gpu_table_representation>(
      std::move(empty_table),
      *const_cast<cucascade::memory::memory_space*>(target_memory_space));
  }

  rmm::device_async_resource_ref mr_ref(target_memory_space->get_default_allocator());
  rmm::cuda_device_id target_device_id(target_memory_space->get_device_id());
  rmm::cuda_set_device_raii target_device_raii(target_device_id);

  // 1. Get contiguous host buffer
  auto const& linear_host = *host_src.get_host_data();

  // 2. Group chunks by column_idx (ordered by block index within each column)
  //    Each column may have multiple blocks that need to be concatenated.
  struct chunk_ref {
    std::size_t chunk_index;
  };
  std::map<uint16_t, std::vector<chunk_ref>> col_chunks;
  for (std::size_t i = 0; i < chunks.size(); i++) {
    col_chunks[chunks[i].column_idx].push_back({i});
  }

  // 3. Separate compressed chunks and uncompressed chunks
  std::vector<std::size_t> compressed_indices;     // indices into chunks[]
  std::vector<std::size_t> uncompressed_indices;
  for (std::size_t i = 0; i < chunks.size(); i++) {
    if (chunks[i].extent.is_compressed()) {
      compressed_indices.push_back(i);
    } else {
      uncompressed_indices.push_back(i);
    }
  }

  // 4. Transfer entire host buffer to GPU in a single contiguous copy.
  //    This replaces per-chunk H→D copies, reducing driver overhead and
  //    enabling full PCIe bandwidth utilization.
  rmm::device_buffer d_mirror(linear_host.size(), stream, mr_ref);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_mirror.data(), linear_host.data(),
                                 linear_host.size(), cudaMemcpyHostToDevice, stream.value()));

  // 5. Allocate decompression output buffers
  std::size_t total_compressed = 0;
  for (auto idx : compressed_indices) {
    total_compressed += chunks[idx].pinned_length;
  }

  std::size_t total_decompressed = 0;
  for (auto idx : compressed_indices) {
    total_decompressed += chunks[idx].extent.origin_size;
  }

  rmm::device_buffer d_decompressed(total_decompressed, stream, mr_ref);

  std::vector<std::size_t> d_decompressed_offsets(compressed_indices.size());
  {
    std::size_t off = 0;
    for (std::size_t ci = 0; ci < compressed_indices.size(); ci++) {
      d_decompressed_offsets[ci] = off;
      off += chunks[compressed_indices[ci]].extent.origin_size;
    }
  }

  // 6. nvCOMP batch LZ4 decompress (source pointers reference d_mirror directly)
  if (!compressed_indices.empty()) {
    std::size_t num_chunks_to_decompress = compressed_indices.size();

    // Build pointer and size arrays on device
    std::vector<void const*> h_comp_ptrs(num_chunks_to_decompress);
    std::vector<std::size_t> h_comp_sizes(num_chunks_to_decompress);
    std::vector<void*>       h_decomp_ptrs(num_chunks_to_decompress);
    std::vector<std::size_t> h_decomp_buf_sizes(num_chunks_to_decompress);

    std::size_t max_uncomp_size = 0;

    for (std::size_t ci = 0; ci < num_chunks_to_decompress; ci++) {
      auto& chunk = chunks[compressed_indices[ci]];
      h_comp_ptrs[ci]  = static_cast<uint8_t*>(d_mirror.data()) + chunk.pinned_offset;
      h_comp_sizes[ci] = chunk.pinned_length;
      h_decomp_ptrs[ci]      = static_cast<uint8_t*>(d_decompressed.data()) + d_decompressed_offsets[ci];
      h_decomp_buf_sizes[ci] = chunk.extent.origin_size;
      max_uncomp_size = std::max(max_uncomp_size, static_cast<std::size_t>(chunk.extent.origin_size));
    }

    // Device arrays
    rmm::device_uvector<void const*> d_comp_ptrs(num_chunks_to_decompress, stream, mr_ref);
    rmm::device_uvector<std::size_t> d_comp_sizes(num_chunks_to_decompress, stream, mr_ref);
    rmm::device_uvector<void*>       d_decomp_ptrs(num_chunks_to_decompress, stream, mr_ref);
    rmm::device_uvector<std::size_t> d_decomp_buf_sizes(num_chunks_to_decompress, stream, mr_ref);
    rmm::device_uvector<std::size_t> d_decomp_actual_sizes(num_chunks_to_decompress, stream, mr_ref);
    rmm::device_uvector<nvcompStatus_t> d_statuses(num_chunks_to_decompress, stream, mr_ref);

    CUDF_CUDA_TRY(cudaMemcpyAsync(d_comp_ptrs.data(), h_comp_ptrs.data(),
                              num_chunks_to_decompress * sizeof(void*),
                              cudaMemcpyHostToDevice, stream.value()));
    CUDF_CUDA_TRY(cudaMemcpyAsync(d_comp_sizes.data(), h_comp_sizes.data(),
                              num_chunks_to_decompress * sizeof(std::size_t),
                              cudaMemcpyHostToDevice, stream.value()));
    CUDF_CUDA_TRY(cudaMemcpyAsync(d_decomp_ptrs.data(), h_decomp_ptrs.data(),
                              num_chunks_to_decompress * sizeof(void*),
                              cudaMemcpyHostToDevice, stream.value()));
    CUDF_CUDA_TRY(cudaMemcpyAsync(d_decomp_buf_sizes.data(), h_decomp_buf_sizes.data(),
                              num_chunks_to_decompress * sizeof(std::size_t),
                              cudaMemcpyHostToDevice, stream.value()));

    // Get temp size
    std::size_t temp_bytes = 0;
    auto opts = nvcompBatchedLZ4DecompressDefaultOpts;
    auto status = nvcompBatchedLZ4DecompressGetTempSizeAsync(
      num_chunks_to_decompress, max_uncomp_size, opts, &temp_bytes, total_decompressed);
    if (status != nvcompSuccess) {
      throw std::runtime_error("nvcompBatchedLZ4DecompressGetTempSizeAsync failed: " +
                               std::to_string(status));
    }

    rmm::device_buffer d_temp(temp_bytes, stream, mr_ref);

    // Decompress
    status = nvcompBatchedLZ4DecompressAsync(
      d_comp_ptrs.data(),
      d_comp_sizes.data(),
      d_decomp_buf_sizes.data(),
      d_decomp_actual_sizes.data(),
      num_chunks_to_decompress,
      d_temp.data(),
      temp_bytes,
      d_decomp_ptrs.data(),
      opts,
      d_statuses.data(),
      stream.value());

    if (status != nvcompSuccess) {
      throw std::runtime_error("nvcompBatchedLZ4DecompressAsync failed: " +
                               std::to_string(status));
    }
  }

  // 7. Pre-built chunk→device_ptr map (O(1) lookup, replaces O(n) linear scan)
  std::vector<uint8_t*> chunk_device_ptrs(chunks.size());
  for (std::size_t ci = 0; ci < compressed_indices.size(); ci++) {
    chunk_device_ptrs[compressed_indices[ci]] =
      static_cast<uint8_t*>(d_decompressed.data()) + d_decompressed_offsets[ci];
  }
  for (auto idx : uncompressed_indices) {
    chunk_device_ptrs[idx] = static_cast<uint8_t*>(d_mirror.data()) + chunks[idx].pinned_offset;
  }

  auto get_chunk_decompressed_size = [&](std::size_t chunk_idx) -> std::size_t {
    auto& chunk = chunks[chunk_idx];
    return chunk.extent.is_compressed() ? chunk.extent.origin_size : chunk.pinned_length;
  };

  // 8. Decode columns: for each column, concatenate all blocks and run CUDA kernel
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(col_chunks.size());

  for (auto& [col_idx, chunk_refs] : col_chunks) {
    auto& first_chunk = chunks[chunk_refs[0].chunk_index];
    auto type_oid = first_chunk.type_oid;
    auto cudf_type = mo_oid_to_cudf_type(type_oid, first_chunk.scale);

    // Total rows for this column across all blocks
    std::size_t col_total_rows = 0;
    for (auto& cr : chunk_refs) {
      col_total_rows += chunks[cr.chunk_index].row_count;
    }

    if (is_varchar_type(type_oid)) {
      // VARCHAR: build cuDF offsets + chars from MO varlena format
      // MO vector layout: [header 29B][varlena_structs row_count*24B][areaLen 4B][area...][nsp...]
      //
      // Performance: batched GPU reads reduce stream.synchronize() calls from
      // 4N (one per block per phase) down to 2 total syncs.

      struct block_info {
        uint8_t* d_data;        // pointer to varlena struct array
        uint8_t* d_area;        // pointer to area section
        std::size_t d_size;
        uint32_t rows;
        uint32_t actual_data_len;  // actual dataLen from vector header
        uint32_t area_len;         // areaLen for nsp offset computation
        std::size_t chunk_index;   // for null mask pass
      };

      // Build block_info using metadata-derived data_len (no GPU read needed).
      // For FLAT varchar vectors, data_len = row_count * VARLENA_STRUCT_SIZE (24).
      std::vector<block_info> blocks;
      blocks.reserve(chunk_refs.size());
      for (std::size_t i = 0; i < chunk_refs.size(); i++) {
        auto& cr = chunk_refs[i];
        auto& chunk = chunks[cr.chunk_index];
        auto* d_ptr = chunk_device_ptrs[cr.chunk_index];
        auto d_size = get_chunk_decompressed_size(cr.chunk_index);
        uint32_t actual_data_len = chunk.row_count * VARLENA_STRUCT_SIZE;

        if (VEC_HEADER_SIZE + actual_data_len > d_size) {
          throw std::runtime_error("varchar varlena section exceeds decompressed buffer");
        }

        auto* d_varlena = d_ptr + VEC_HEADER_SIZE;
        auto* d_area = d_ptr + VEC_HEADER_SIZE + actual_data_len + 4;
        blocks.push_back({d_varlena, d_area, d_size, chunk.row_count, actual_data_len, 0, cr.chunk_index});
      }

      // === Single sync: launch sum_lengths + area_len reads, then batch D→H ===
      std::vector<unsigned long long> h_block_totals(blocks.size());
      {
        rmm::device_uvector<unsigned long long> d_totals(blocks.size(), stream, mr_ref);
        CUDF_CUDA_TRY(cudaMemsetAsync(d_totals.data(), 0,
                                       blocks.size() * sizeof(unsigned long long), stream.value()));
        for (std::size_t i = 0; i < blocks.size(); i++) {
          cuda::tae::compute_varchar_total_chars_async(blocks[i].d_data, blocks[i].d_area, blocks[i].rows,
                                 d_totals.data() + i, stream);
        }
        // Area_len reads for blocks with nulls (overlaps with sum_lengths on same stream)
        for (std::size_t i = 0; i < blocks.size(); i++) {
          auto& chunk = chunks[blocks[i].chunk_index];
          if (chunk.null_cnt > 0) {
            auto* d_ptr = chunk_device_ptrs[blocks[i].chunk_index];
            CUDF_CUDA_TRY(cudaMemcpyAsync(&blocks[i].area_len,
                                           d_ptr + VEC_HEADER_SIZE + blocks[i].actual_data_len,
                                           sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.value()));
          }
        }
        // D→H of totals (async, waits for sum_lengths on same stream)
        CUDF_CUDA_TRY(cudaMemcpyAsync(h_block_totals.data(), d_totals.data(),
                                       blocks.size() * sizeof(unsigned long long),
                                       cudaMemcpyDeviceToHost, stream.value()));
        stream.synchronize();
      }

      // Compute per-block char offsets and grand total on host
      std::size_t grand_total_chars = 0;
      std::vector<std::size_t> block_char_offsets(blocks.size());
      for (std::size_t i = 0; i < blocks.size(); i++) {
        block_char_offsets[i] = grand_total_chars;
        grand_total_chars += h_block_totals[i];
      }

      // Allocate output: offsets (int32) and chars (uint8)
      auto offsets_buf = rmm::device_buffer(
        (col_total_rows + 1) * sizeof(int32_t), stream, mr_ref);
      auto chars_buf = rmm::device_buffer(grand_total_chars, stream, mr_ref);

      // === Phase 3: Decode all blocks ===
      // CUB temp storage is queried once for the largest block, then reused.
      std::size_t max_temp_bytes = 0;
      for (auto& bi : blocks) {
        std::size_t tb = 0;
        cuda::tae::decode_varchar(nullptr, nullptr, nullptr, nullptr, nullptr, tb, bi.rows, stream);
        max_temp_bytes = std::max(max_temp_bytes, tb);
      }
      rmm::device_buffer d_temp(max_temp_bytes, stream, mr_ref);

      std::size_t row_offset = 0;
      for (std::size_t i = 0; i < blocks.size(); i++) {
        auto& bi = blocks[i];
        auto* d_offsets_out = static_cast<int32_t*>(offsets_buf.data()) + row_offset;
        auto* d_chars_out = static_cast<uint8_t*>(chars_buf.data()) + block_char_offsets[i];

        std::size_t temp_bytes = max_temp_bytes;
        cuda::tae::decode_varchar(
          bi.d_data, bi.d_area, d_offsets_out, d_chars_out, d_temp.data(), temp_bytes, bi.rows, stream);

        // Adjust offsets to be globally monotonic by adding cumulative char_offset
        if (block_char_offsets[i] > 0) {
          cuda::tae::adjust_offsets(d_offsets_out, static_cast<int32_t>(block_char_offsets[i]),
                                    bi.rows + 1, stream);
        }

        row_offset += bi.rows;
      }

      // Build cuDF string column from offsets + chars
      auto offsets_col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        static_cast<cudf::size_type>(col_total_rows + 1),
        std::move(offsets_buf),
        rmm::device_buffer{},
        0);

      // Build null mask
      uint32_t total_nulls = 0;
      for (auto& cr : chunk_refs) {
        total_nulls += chunks[cr.chunk_index].null_cnt;
      }

      rmm::device_buffer null_mask;
      if (total_nulls > 0) {
        null_mask = cudf::create_null_mask(col_total_rows, cudf::mask_state::ALL_VALID, stream, mr_ref);
        std::size_t bitmask_row_offset = 0;
        for (auto& bi : blocks) {
          auto& chunk = chunks[bi.chunk_index];
          if (chunk.null_cnt > 0) {
            auto* d_src_blk = chunk_device_ptrs[bi.chunk_index];
            uint32_t nsp_bitmap_offset = VEC_HEADER_SIZE + bi.actual_data_len + 4 + bi.area_len + 4 + 24;
            auto* d_validity = static_cast<uint32_t*>(null_mask.data());
            cuda::tae::invert_null_mask(d_src_blk + nsp_bitmap_offset,
                                        d_validity + (bitmask_row_offset / 32),
                                        bi.rows, stream);
          }
          bitmask_row_offset += bi.rows;
        }
      }

      auto str_col = cudf::make_strings_column(
        static_cast<cudf::size_type>(col_total_rows),
        std::move(offsets_col),
        std::move(chars_buf),
        total_nulls > 0 ? static_cast<cudf::size_type>(total_nulls) : 0,
        total_nulls > 0 ? std::move(null_mask) : rmm::device_buffer{});

      columns.push_back(std::move(str_col));
    } else {
      // Fixed-width type
      uint32_t elem_size = mo_oid_fixed_width(type_oid);
      if (elem_size == 0) elem_size = 8;  // fallback

      // Compute epoch adjustment for temporal types
      int64_t epoch_adjust = 0;
      if (is_date_type(type_oid)) epoch_adjust = tae::MO_UNIX_EPOCH_DAYS;
      else if (is_timestamp_type(type_oid)) epoch_adjust = tae::MO_UNIX_EPOCH_USEC;

      // Allocate output column data
      auto data_buf = rmm::device_buffer(col_total_rows * elem_size, stream, mr_ref);

      // Decode each block: pass pointer to data section (past vector header)
      std::size_t row_offset = 0;
      for (auto& cr : chunk_refs) {
        auto& chunk = chunks[cr.chunk_index];
        auto* d_src = chunk_device_ptrs[cr.chunk_index];
        auto* d_data = d_src + VEC_HEADER_SIZE;  // skip IOEntry + vec header
        auto* d_dst = static_cast<uint8_t*>(data_buf.data()) + row_offset * elem_size;

        if (epoch_adjust == 0) {
          // Direct D2D copy — avoids kernel launch overhead for simple memcpy
          CUDF_CUDA_TRY(cudaMemcpyAsync(d_dst, d_data, chunk.row_count * elem_size,
                                        cudaMemcpyDeviceToDevice, stream.value()));
        } else {
          cuda::tae::decode_fixed_width(
            d_data, d_dst, chunk.row_count, elem_size, epoch_adjust, stream);
        }

        row_offset += chunk.row_count;
      }

      // Build null mask
      uint32_t total_nulls = 0;
      for (auto& cr : chunk_refs) {
        total_nulls += chunks[cr.chunk_index].null_cnt;
      }

      rmm::device_buffer null_mask;
      if (total_nulls > 0) {
        null_mask = cudf::create_null_mask(col_total_rows, cudf::mask_state::ALL_VALID, stream, mr_ref);

        // MO null bitmap (nsp section) is after data and area sections.
        // For fixed-width: areaLen = 0, so nsp starts at
        //   VEC_HEADER_SIZE + data_len + 4(areaLen) + 0(area) + 4(nspLen) + 24(nsp header)
        std::size_t bitmask_row_offset = 0;
        for (auto& cr : chunk_refs) {
          auto& chunk = chunks[cr.chunk_index];
          if (chunk.null_cnt == 0) {
            bitmask_row_offset += chunk.row_count;
            continue;
          }

          auto* d_src = chunk_device_ptrs[cr.chunk_index];
          uint32_t data_len = chunk.row_count * elem_size;
          uint32_t nsp_bitmap_offset = VEC_HEADER_SIZE + data_len + 4 + 0 + 4 + 24;
          auto* d_validity = static_cast<uint32_t*>(null_mask.data());
          cuda::tae::invert_null_mask(
            d_src + nsp_bitmap_offset, d_validity + (bitmask_row_offset / 32),
            chunk.row_count, stream);

          bitmask_row_offset += chunk.row_count;
        }
      }

      auto col = std::make_unique<cudf::column>(
        cudf_type,
        static_cast<cudf::size_type>(col_total_rows),
        std::move(data_buf),
        total_nulls > 0 ? std::move(null_mask) : rmm::device_buffer{},
        total_nulls > 0 ? static_cast<cudf::size_type>(total_nulls) : 0);

      columns.push_back(std::move(col));
    }
  }

  // 9. Apply filter expression on GPU (if present)
  auto const& filter_expr = host_src.get_filter_expression();
  if (filter_expr && filter_expr->size() > 0) {
    // Build a temporary table to evaluate the filter against ALL columns
    auto pre_filter_table = std::make_unique<cudf::table>(std::move(columns));
    auto table_view = pre_filter_table->view();

    SIRIUS_LOG_DEBUG("[tae_converter] applying GPU filter on {} rows, {} columns",
                     table_view.num_rows(), table_view.num_columns());

    // Evaluate filter AST → boolean mask column
    auto mask = cudf::compute_column(table_view, filter_expr->back(), stream, mr_ref);

    // Apply boolean mask to keep only matching rows
    auto filtered_table = cudf::apply_boolean_mask(table_view, mask->view(), stream, mr_ref);

    SIRIUS_LOG_DEBUG("[tae_converter] filter reduced {} → {} rows",
                     table_view.num_rows(), filtered_table->num_rows());

    // Release filtered columns back into the vector
    columns = filtered_table->release();
  }

  // 10. Apply post-filter projection (prune filter-only columns)
  if (!post_filter_projection_ids.empty()) {
    std::vector<std::unique_ptr<cudf::column>> projected;
    projected.reserve(post_filter_projection_ids.size());
    for (auto id : post_filter_projection_ids) {
      if (id < columns.size()) {
        projected.push_back(std::move(columns[id]));
      }
    }
    columns = std::move(projected);
  }

  stream.synchronize();

  auto table = std::make_unique<cudf::table>(std::move(columns));

  SIRIUS_LOG_INFO("[tae_converter] produced GPU table: {} columns, {} rows",
                  table->num_columns(), table->num_rows());

  return std::make_unique<cucascade::gpu_table_representation>(
    std::move(table),
    *const_cast<cucascade::memory::memory_space*>(target_memory_space));
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Public registration
// ---------------------------------------------------------------------------
void register_tae_converters(cucascade::representation_converter_registry& registry)
{
  if (!registry.has_converter<host_tae_representation, cucascade::gpu_table_representation>()) {
    registry.register_converter<host_tae_representation, cucascade::gpu_table_representation>(
      detail::convert_host_tae_to_gpu);
  }
}

}  // namespace sirius
