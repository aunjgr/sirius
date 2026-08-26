/*
 * Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "tae/tae_format.hpp"

#include <duckdb/function/function.hpp>

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace sirius::offload {

struct mo_native_column_view {
  std::uint8_t vector_class = 0;
  tae::MOType type{};
  std::uint32_t logical_rows = 0;
  std::string_view encoded;
  std::string_view data;
  std::string_view area;
  std::string_view null_words;
  std::uint64_t null_count = 0;
  bool sorted              = false;

  [[nodiscard]] bool is_null(std::uint64_t row) const noexcept;
};

class mo_native_batch {
 public:
  virtual ~mo_native_batch()                                         = default;
  [[nodiscard]] virtual std::uint64_t sequence() const noexcept      = 0;
  [[nodiscard]] virtual std::uint64_t rows() const noexcept          = 0;
  [[nodiscard]] virtual std::uint64_t payload_bytes() const noexcept = 0;
  [[nodiscard]] virtual const std::vector<mo_native_column_view>& columns() const noexcept = 0;
};

class mo_native_batch_source {
 public:
  virtual ~mo_native_batch_source()                           = default;
  virtual std::shared_ptr<mo_native_batch> next_batch()       = 0;
  virtual void mark_consumed(std::uint64_t sequence) noexcept = 0;
};

// DuckDB binds this marker to mo_stream_scan. Sirius recognizes the common
// bind-data type and replaces the CPU table-function source with its native
// MO-vector GPU scan. Copying retains the query-owned source lifetime.
class mo_native_scan_bind_data final : public duckdb::FunctionData {
 public:
  explicit mo_native_scan_bind_data(std::shared_ptr<mo_native_batch_source> source_p)
    : source(std::move(source_p))
  {
  }

  duckdb::unique_ptr<duckdb::FunctionData> Copy() const override
  {
    return duckdb::make_uniq<mo_native_scan_bind_data>(source);
  }

  bool Equals(const duckdb::FunctionData& other) const override
  {
    auto const* rhs = dynamic_cast<const mo_native_scan_bind_data*>(&other);
    return rhs != nullptr && rhs->source.get() == source.get();
  }

  std::shared_ptr<mo_native_batch_source> source;
};

}  // namespace sirius::offload
