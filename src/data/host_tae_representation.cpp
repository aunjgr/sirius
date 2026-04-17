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

#include <data/host_tae_representation.hpp>

#include <stdexcept>

namespace sirius {

host_tae_representation::host_tae_representation(
  cucascade::memory::memory_space* memory_space,
  pinned_host_buffer host_data,
  std::vector<column_chunk_info> chunks,
  std::size_t total_rows,
  std::size_t compressed_bytes,
  std::size_t uncompressed_bytes,
  std::shared_ptr<translated_expression> filter_expression,
  std::vector<std::size_t> post_filter_projection_ids)
  : idata_representation([&]() -> cucascade::memory::memory_space& {
      if (!memory_space) {
        throw std::runtime_error(
          "[host_tae_representation] null memory_space pointer");
      }
      return *memory_space;
    }()),
    _host_data(std::make_shared<pinned_host_buffer>(std::move(host_data))),
    _chunks(std::move(chunks)),
    _total_rows(total_rows),
    _compressed_bytes(compressed_bytes),
    _uncompressed_bytes(uncompressed_bytes),
    _filter_expression(std::move(filter_expression)),
    _post_filter_projection_ids(std::move(post_filter_projection_ids))
{
}

std::unique_ptr<cucascade::idata_representation> host_tae_representation::clone(
  rmm::cuda_stream_view /*stream*/)
{
  // Shallow clone: shares host data buffer
  auto result = std::unique_ptr<host_tae_representation>(new host_tae_representation(*this));
  return result;
}

std::size_t host_tae_representation::get_size_in_bytes() const { return _compressed_bytes; }

std::size_t host_tae_representation::get_uncompressed_data_size_in_bytes() const
{
  return _uncompressed_bytes;
}

}  // namespace sirius
