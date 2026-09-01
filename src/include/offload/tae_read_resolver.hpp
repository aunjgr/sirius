/*
 * Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "offload/stream_read.hpp"
#include "offload/tae_read.hpp"
#include "substrait/algebra.pb.h"

#include <cstdint>
#include <memory>
#include <string>

namespace sirius::offload {

/// Move-only lifetime token for one authenticated, query-local relation.
/// Its destructor unregisters the relation and releases its backing handle.
class resolved_read {
 public:
  virtual ~resolved_read() noexcept = default;

  resolved_read(const resolved_read&)            = delete;
  resolved_read& operator=(const resolved_read&) = delete;
  resolved_read(resolved_read&&)                 = delete;
  resolved_read& operator=(resolved_read&&)      = delete;

  [[nodiscard]] virtual const std::string& relation_name() const noexcept                 = 0;
  [[nodiscard]] virtual const ::substrait::NamedStruct& canonical_schema() const noexcept = 0;

 protected:
  resolved_read() = default;
};

class resolved_tae_read : public resolved_read {
 public:
  ~resolved_tae_read() noexcept override = default;

  [[nodiscard]] virtual const std::string& read_ref() const noexcept        = 0;
  [[nodiscard]] virtual const std::string& query_id() const noexcept        = 0;
  [[nodiscard]] virtual std::uint64_t account_id() const noexcept           = 0;
  [[nodiscard]] virtual std::uint64_t database_id() const noexcept          = 0;
  [[nodiscard]] virtual std::uint64_t table_id() const noexcept             = 0;
  [[nodiscard]] virtual const std::string& snapshot_ts() const noexcept     = 0;
  [[nodiscard]] virtual const std::string& schema_digest() const noexcept   = 0;
  [[nodiscard]] virtual const std::string& manifest_sha256() const noexcept = 0;
  [[nodiscard]] virtual const std::string& capability_hash() const noexcept = 0;
  [[nodiscard]] virtual std::uint64_t expires_at_unix_ms() const noexcept   = 0;

 protected:
  resolved_tae_read() = default;
};

class resolved_stream_read : public resolved_read {
 public:
  ~resolved_stream_read() noexcept override = default;

  [[nodiscard]] virtual const std::string& stream_ref() const noexcept      = 0;
  [[nodiscard]] virtual const std::string& query_id() const noexcept        = 0;
  [[nodiscard]] virtual std::uint64_t account_id() const noexcept           = 0;
  [[nodiscard]] virtual const std::string& snapshot_ts() const noexcept     = 0;
  [[nodiscard]] virtual const std::string& schema_digest() const noexcept   = 0;
  [[nodiscard]] virtual const std::string& capability_hash() const noexcept = 0;
  [[nodiscard]] virtual std::uint64_t expires_at_unix_ms() const noexcept   = 0;

 protected:
  resolved_stream_read() = default;
};

/// Authenticated boundary between retained plan bytes and MatrixOne storage.
class tae_read_resolver {
 public:
  virtual ~tae_read_resolver() noexcept = default;

  virtual std::unique_ptr<resolved_tae_read> resolve(
    const tae_read& request, const ::substrait::NamedStruct& requested_schema) = 0;

  virtual std::unique_ptr<resolved_stream_read> resolve(
    const stream_read& request, const ::substrait::NamedStruct& requested_schema) = 0;
};

}  // namespace sirius::offload
