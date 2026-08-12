/*
 * Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>

namespace sirius::offload {

/// Parsed value of matrixone.sirius.v1.TaeRead. The wire contract lives in
/// proto/matrixone/sirius/v1/tae_read.proto; Sirius parses it strictly because
/// DuckDB's bundled protobuf runtime intentionally differs from host protoc.
struct tae_read {
  std::uint32_t protocol_version = 0;
  std::uint64_t feature_bits     = 0;
  std::string read_ref;
  std::string query_id;
  std::uint64_t account_id  = 0;
  std::uint64_t database_id = 0;
  std::uint64_t table_id    = 0;
  std::string snapshot_ts;
  std::string schema_digest;
  std::string manifest_sha256;
  std::string capability_hash;
  std::uint64_t expires_at_unix_ms = 0;
};

}  // namespace sirius::offload
