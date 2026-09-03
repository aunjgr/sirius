/*
 * Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf/cudf_compat.hpp"
#include "offload/mo_native_result_pack.hpp"
#include "tae/tae_format.hpp"

#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace sirius::offload {
namespace {

enum class conversion : std::uint8_t { COPY = 0, I64_TO_U32, DATE_EPOCH };

template <typename T>
void append_scalar(std::string& output, T value)
{
  static_assert(std::is_trivially_copyable_v<T>);
  output.append(reinterpret_cast<const char*>(&value), sizeof(value));
}

std::uint32_t target_size(std::uint32_t oid)
{
  auto size = tae::MOTypeFixedSize(static_cast<tae::MOTypeOid>(oid));
  return size > 0 ? static_cast<std::uint32_t>(size) : 0;
}

bool column_eligible(cudf::column_view column,
                     const mo_native_result_column& schema,
                     conversion* kind = nullptr)
{
  using id          = cudf::type_id;
  conversion result = conversion::COPY;
  bool matches      = false;
  switch (schema.oid) {
    case tae::MO_T_bool: matches = column.type().id() == id::BOOL8; break;
    case tae::MO_T_int8: matches = column.type().id() == id::INT8; break;
    case tae::MO_T_int16: matches = column.type().id() == id::INT16; break;
    case tae::MO_T_int32: matches = column.type().id() == id::INT32; break;
    case tae::MO_T_int64: matches = column.type().id() == id::INT64; break;
    case tae::MO_T_uint32:
      matches = column.type().id() == id::INT64;
      result  = conversion::I64_TO_U32;
      break;
    case tae::MO_T_float32: matches = column.type().id() == id::FLOAT32; break;
    case tae::MO_T_float64: matches = column.type().id() == id::FLOAT64; break;
    case tae::MO_T_decimal64:
      matches = column.type().id() == id::DECIMAL64 && column.type().scale() == -schema.scale;
      break;
    case tae::MO_T_decimal128:
      matches = column.type().id() == id::DECIMAL128 && column.type().scale() == -schema.scale;
      break;
    case tae::MO_T_date:
      matches = column.type().id() == id::TIMESTAMP_DAYS;
      result  = conversion::DATE_EPOCH;
      break;
    default: matches = false; break;
  }
  if (matches && schema.not_nullable && column.null_count() != 0) {
    throw std::invalid_argument("required GPU native result column contains nulls");
  }
  if (kind) *kind = result;
  // The fast packer intentionally handles null-free fixed-width batches. A
  // nullable schema with no actual nulls is eligible; batches containing nulls
  // use the host-representation packer, which also handles split-range counts.
  return matches && target_size(schema.oid) != 0 && column.null_count() == 0;
}

__global__ void pack_fixed(const std::uint8_t* source,
                           std::uint8_t* target,
                           std::size_t rows,
                           std::uint32_t source_size,
                           std::uint32_t target_width,
                           conversion kind,
                           int* invalid)
{
  auto row = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  if (kind == conversion::I64_TO_U32) {
    auto value = reinterpret_cast<const std::int64_t*>(source)[row];
    if (value < 0 || static_cast<std::uint64_t>(value) > 0xffffffffULL) {
      atomicExch(invalid, 1);
      return;
    }
    reinterpret_cast<std::uint32_t*>(target)[row] = static_cast<std::uint32_t>(value);
  } else if (kind == conversion::DATE_EPOCH) {
    auto value = reinterpret_cast<const std::int32_t*>(source)[row];
    if (value > 2147483647 - tae::MO_UNIX_EPOCH_DAYS) {
      atomicExch(invalid, 1);
      return;
    }
    reinterpret_cast<std::int32_t*>(target)[row] = value + tae::MO_UNIX_EPOCH_DAYS;
  } else {
    for (std::uint32_t byte = 0; byte < target_width; ++byte) {
      target[row * target_width + byte] = source[row * source_size + byte];
    }
  }
}

__global__ void pack_null_words(const cudf::bitmask_type* source,
                                std::size_t source_bit_offset,
                                std::uint64_t* target,
                                std::size_t rows)
{
  auto word = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (word >= (rows + 63U) / 64U) return;
  std::uint64_t nulls = 0;
  for (std::size_t bit = 0; bit < 64U; ++bit) {
    auto row = word * 64U + bit;
    if (row >= rows) break;
    auto source_row = source_bit_offset + row;
    auto valid = (source[source_row / 32U] & (cudf::bitmask_type{1} << (source_row % 32U))) != 0;
    if (!valid) nulls |= std::uint64_t{1} << bit;
  }
  target[word] = nulls;
}

}  // namespace

bool can_pack_mo_native_result_on_gpu(cudf::table_view table,
                                      const std::vector<mo_native_result_column>& schema,
                                      std::size_t minimum_bytes)
{
  if (table.num_rows() <= 0 || static_cast<std::size_t>(table.num_columns()) != schema.size()) {
    return false;
  }
  std::size_t bytes = 0;
  for (cudf::size_type i = 0; i < table.num_columns(); ++i) {
    if (!column_eligible(table.column(i), schema[static_cast<std::size_t>(i)])) return false;
    bytes += static_cast<std::size_t>(table.num_rows()) *
             target_size(schema[static_cast<std::size_t>(i)].oid);
  }
  return bytes >= minimum_bytes;
}

std::string pack_mo_native_result_on_gpu(cudf::table_view table,
                                         const std::vector<mo_native_result_column>& schema,
                                         std::size_t row_offset,
                                         std::size_t row_count,
                                         rmm::cuda_stream_view stream)
{
  if (row_count == 0 || row_offset > static_cast<std::size_t>(table.num_rows()) ||
      row_count > static_cast<std::size_t>(table.num_rows()) - row_offset ||
      static_cast<std::size_t>(table.num_columns()) != schema.size()) {
    throw std::invalid_argument("invalid GPU native result range or schema width");
  }
  if (row_count > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error("GPU native result row count exceeds uint32");
  }

  struct column_layout {
    std::size_t data_offset;
    std::size_t null_words_offset;
    std::uint32_t width;
    conversion kind;
  };
  std::vector<column_layout> layouts;
  layouts.reserve(schema.size());
  std::string payload;
  append_scalar(payload, static_cast<std::int64_t>(row_count));
  append_scalar(payload, static_cast<std::int32_t>(schema.size()));
  for (std::size_t i = 0; i < schema.size(); ++i) {
    auto column = table.column(static_cast<cudf::size_type>(i));
    conversion kind;
    if (!column_eligible(column, schema[i], &kind)) {
      throw std::invalid_argument("GPU native result pack received an ineligible column");
    }
    const auto width      = target_size(schema[i].oid);
    const auto null_count = column.null_count();
    const auto words      = null_count == 0 ? 0U : (row_count + 63U) / 64U;
    const auto null_bytes = words == 0 ? 0U : 24U + words * sizeof(std::uint64_t);
    const auto vector_bytes =
      1U + sizeof(tae::MOType) + 4U + 4U + row_count * width + 4U + 4U + null_bytes + 1U;
    append_scalar(payload, static_cast<std::uint32_t>(vector_bytes));
    payload.push_back('\0');
    tae::MOType type{static_cast<std::uint8_t>(schema[i].oid),
                     static_cast<std::uint8_t>(schema[i].charset),
                     static_cast<std::uint8_t>(schema[i].not_nullable ? 1 : 0),
                     0,
                     static_cast<std::int32_t>(width),
                     schema[i].width,
                     schema[i].scale};
    payload.append(reinterpret_cast<const char*>(&type), sizeof(type));
    append_scalar(payload, static_cast<std::uint32_t>(row_count));
    append_scalar(payload, static_cast<std::uint32_t>(row_count * width));
    const auto data_offset = payload.size();
    payload.append(row_count * width, '\0');
    append_scalar(payload, std::uint32_t{0});
    append_scalar(payload, static_cast<std::uint32_t>(null_bytes));
    std::size_t null_words_offset = 0;
    if (words != 0) {
      append_scalar(payload, static_cast<std::int64_t>(null_count));
      append_scalar(payload, static_cast<std::uint64_t>(row_count));
      append_scalar(payload, static_cast<std::uint64_t>(words * sizeof(std::uint64_t)));
      null_words_offset = payload.size();
      payload.append(words * sizeof(std::uint64_t), '\0');
    }
    payload.push_back('\0');
    layouts.push_back({data_offset, null_words_offset, width, kind});
  }
  append_scalar(payload, std::int32_t{0});
  append_scalar(payload, std::int32_t{0});
  append_scalar(payload, std::int32_t{0});
  append_scalar(payload, std::int32_t{0});

  rmm::device_buffer device_payload(payload.size(), stream);
  rmm::device_buffer invalid(sizeof(int), stream);
  CUDF_CUDA_TRY(cudaMemsetAsync(invalid.data(), 0, sizeof(int), stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    device_payload.data(), payload.data(), payload.size(), cudaMemcpyHostToDevice, stream.value()));
  constexpr int threads = 256;
  const auto blocks     = static_cast<int>((row_count + threads - 1U) / threads);
  for (std::size_t i = 0; i < schema.size(); ++i) {
    auto column            = table.column(static_cast<cudf::size_type>(i));
    const auto source_size = static_cast<std::uint32_t>(cudf::size_of(column.type()));
    auto* source           = static_cast<const std::uint8_t*>(column.head()) +
                   (static_cast<std::size_t>(column.offset()) + row_offset) * source_size;
    auto* target = static_cast<std::uint8_t*>(device_payload.data()) + layouts[i].data_offset;
    pack_fixed<<<blocks, threads, 0, stream.value()>>>(source,
                                                       target,
                                                       row_count,
                                                       source_size,
                                                       layouts[i].width,
                                                       layouts[i].kind,
                                                       static_cast<int*>(invalid.data()));
    CUDF_CUDA_TRY(cudaPeekAtLastError());
    if (layouts[i].null_words_offset != 0) {
      const auto words       = (row_count + 63U) / 64U;
      const auto null_blocks = static_cast<int>((words + threads - 1U) / threads);
      pack_null_words<<<null_blocks, threads, 0, stream.value()>>>(
        column.null_mask(),
        static_cast<std::size_t>(column.offset()) + row_offset,
        reinterpret_cast<std::uint64_t*>(static_cast<std::uint8_t*>(device_payload.data()) +
                                         layouts[i].null_words_offset),
        row_count);
      CUDF_CUDA_TRY(cudaPeekAtLastError());
    }
  }
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    payload.data(), device_payload.data(), payload.size(), cudaMemcpyDeviceToHost, stream.value()));
  int host_invalid = 0;
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    &host_invalid, invalid.data(), sizeof(host_invalid), cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();
  if (host_invalid != 0) {
    throw std::overflow_error("GPU native result conversion is outside the MatrixOne type range");
  }
  return payload;
}

}  // namespace sirius::offload
