/*
 * Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "execution/sirius_execution_evidence.hpp"
#include "offload/tae_read_resolver.hpp"

#include <duckdb/common/types.hpp>
#include <duckdb/common/types/data_chunk.hpp>
#include <duckdb/main/client_context.hpp>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace sirius {
class sirius_prepared_statement_data;
}

namespace sirius::offload {

inline constexpr std::uint32_t k_tae_read_protocol_version = 1;
inline constexpr std::uint64_t k_tae_read_feature_bits     = 0;
inline constexpr std::size_t k_max_substrait_plan_bytes    = 16U * 1024U * 1024U;
inline constexpr std::size_t k_max_read_ref_bytes          = 4096U;
inline constexpr std::string_view k_tae_read_type_url =
  "type.googleapis.com/matrixone.sirius.v1.TaeRead";

enum class substrait_error_code : std::uint8_t {
  UNSUPPORTED_PLAN = 0,
  INVALID_PLAN,
  READ_RESOLUTION_FAILED,
  AUTHENTICATION_FAILED,
  EXECUTION_FAILED,
  CANCELLED,
};

class substrait_execution_error final : public std::runtime_error {
 public:
  substrait_execution_error(substrait_error_code code, std::string message);

  [[nodiscard]] substrait_error_code code() const noexcept { return code_; }
  [[nodiscard]] bool fallback_eligible() const noexcept
  { return code_ == substrait_error_code::UNSUPPORTED_PLAN; }

 private:
  substrait_error_code code_;
};

enum class chunk_action : std::uint8_t { CONTINUE = 0, CANCEL };
using chunk_consumer = std::function<chunk_action(const duckdb::DataChunk&)>;

struct execution_schema {
  duckdb::vector<std::string> names;
  duckdb::vector<duckdb::LogicalType> types;
};

enum class execution_state : std::uint8_t {
  PREPARED = 0,
  RUNNING,
  SUCCEEDED,
  FAILED,
  CANCELLED,
};

/// Single-use owner of a validated Sirius physical plan and read lifetimes.
class substrait_execution final {
 public:
  substrait_execution(duckdb::ClientContext& context,
                      duckdb::shared_ptr<sirius_prepared_statement_data> prepared,
                      execution_schema schema,
                      std::shared_ptr<execution_evidence> evidence,
                      std::vector<std::unique_ptr<resolved_tae_read>> resolutions);
  ~substrait_execution() = default;

  substrait_execution(const substrait_execution&)            = delete;
  substrait_execution& operator=(const substrait_execution&) = delete;
  substrait_execution(substrait_execution&&)                 = delete;
  substrait_execution& operator=(substrait_execution&&)      = delete;

  [[nodiscard]] const execution_schema& schema() const noexcept { return schema_; }
  [[nodiscard]] execution_state state() const noexcept { return state_.load(); }

  void cancel() noexcept;
  void run(const chunk_consumer& consumer);

 private:
  bool transition(execution_state expected, execution_state desired) noexcept;
  void release_resolutions() noexcept;

  duckdb::ClientContext& context_;
  duckdb::shared_ptr<sirius_prepared_statement_data> prepared_;
  execution_schema schema_;
  std::shared_ptr<execution_evidence> evidence_;
  std::vector<std::unique_ptr<resolved_tae_read>> resolutions_;
  std::mutex resolutions_mutex_;
  std::atomic<execution_state> state_{execution_state::PREPARED};
  std::atomic<bool> cancel_requested_{false};
};

std::unique_ptr<substrait_execution> prepare_substrait(
  duckdb::ClientContext& context,
  std::string_view binary_plan,
  tae_read_resolver& resolver,
  std::shared_ptr<execution_evidence> evidence);

namespace detail {

struct validated_substrait_plan {
  std::string serialized;
  std::vector<std::unique_ptr<resolved_tae_read>> resolutions;
};

/// Pure validation/resolution seam used by prepare_substrait and contract
/// tests. now_unix_ms=0 selects the system clock.
validated_substrait_plan validate_and_resolve_substrait(std::string_view binary_plan,
                                                        tae_read_resolver& resolver,
                                                        std::uint64_t now_unix_ms = 0);

}  // namespace detail

}  // namespace sirius::offload
