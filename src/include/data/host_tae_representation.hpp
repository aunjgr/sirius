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

#pragma once

// sirius
#include <expression_executor/gpu_expression_translator.hpp>
#include <tae/tae_format.hpp>

// cucascade
#include <cucascade/data/common.hpp>
#include <cucascade/memory/memory_space.hpp>

// rmm
#include <rmm/cuda_stream_view.hpp>

// standard library
#include <cstddef>
#include <memory>
#include <vector>

namespace sirius {

/**
 * @brief Host representation of compressed TAE column data.
 *
 * Each instance holds one TAE object's worth of projected column data in
 * LZ4-compressed form (CRC already stripped). The converter pipeline transfers
 * this data to GPU, decompresses with nvCOMP batch LZ4, then decodes MO types
 * to cuDF columns via custom CUDA kernels.
 *
 * Layout per column chunk in the buffer:
 *   [IOEntryHeader 4B] [compressed payload ...]
 * The Extent in column_meta tells offset+length for the compressed data,
 * and origin_size for the decompressed size.
 */
class host_tae_representation : public cucascade::idata_representation {
  using translated_expression = gpu_expression_translator::translated_expression;

 public:
  /**
   * @brief Metadata for a single column chunk within this representation.
   */
  struct column_chunk_info {
    uint16_t            column_idx;          ///< Column ordinal in the TAE object
    tae::MOTypeOid      type_oid;            ///< MO type for decode
    int32_t             width;               ///< DECIMAL precision
    int32_t             scale;               ///< DECIMAL scale
    tae::Extent         extent;              ///< Compressed data location
    uint32_t            null_cnt;            ///< Number of nulls in this chunk
    uint32_t            row_count;           ///< Number of rows in the block
    std::size_t         pinned_offset;       ///< Byte offset into host buffer
    std::size_t         pinned_length;       ///< Byte count in host buffer (compressed)
  };

  /**
   * @brief Constructs a host_tae_representation.
   *
   * @param memory_space  The memory space for the target GPU.
   * @param host_data     Contiguous host buffer holding compressed column data.
   * @param chunks        Per-column metadata for the converter to decode.
   * @param total_rows    Total row count across all blocks in this object.
   * @param compressed_bytes  Total compressed bytes in host_data.
   * @param uncompressed_bytes  Sum of all origin_size values.
   * @param filter_expression  Optional pushdown filter.
   * @param post_filter_projection_ids  Column indices surviving filter.
   */
  host_tae_representation(
    cucascade::memory::memory_space* memory_space,
    std::vector<uint8_t> host_data,
    std::vector<column_chunk_info> chunks,
    std::size_t total_rows,
    std::size_t compressed_bytes,
    std::size_t uncompressed_bytes,
    std::shared_ptr<translated_expression> filter_expression = nullptr,
    std::vector<std::size_t> post_filter_projection_ids      = {});

  // idata_representation interface
  std::unique_ptr<idata_representation> clone(rmm::cuda_stream_view stream) override;
  [[nodiscard]] std::size_t get_size_in_bytes() const override;
  [[nodiscard]] std::size_t get_uncompressed_data_size_in_bytes() const override;

  // Accessors
  [[nodiscard]] auto const& get_host_data() const { return _host_data; }
  [[nodiscard]] auto const& get_column_chunks() const { return _chunks; }
  [[nodiscard]] std::size_t get_total_rows() const { return _total_rows; }

  [[nodiscard]] std::shared_ptr<translated_expression> const& get_filter_expression() const
  {
    return _filter_expression;
  }

  [[nodiscard]] std::vector<std::size_t> const& get_post_filter_projection_ids() const
  {
    return _post_filter_projection_ids;
  }

 private:
  std::shared_ptr<std::vector<uint8_t>> _host_data;

  std::vector<column_chunk_info> _chunks;
  std::size_t _total_rows;
  std::size_t _compressed_bytes;
  std::size_t _uncompressed_bytes;
  std::shared_ptr<translated_expression> _filter_expression;
  std::vector<std::size_t> _post_filter_projection_ids;
};

}  // namespace sirius
