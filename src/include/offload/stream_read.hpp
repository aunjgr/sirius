/*
 * Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>

namespace sirius::offload {

/// Parsed value of matrixone.sirius.v1.StreamRead. MatrixOne supplies
/// MO-native batches over an authenticated Flight input stream; the retained
/// Substrait plan carries only this query-scoped identity and schema digest.
struct stream_read {
  std::uint32_t protocol_version = 0;
  std::uint64_t feature_bits     = 0;
  std::string stream_ref;
  std::string query_id;
  std::uint64_t account_id = 0;
  std::string snapshot_ts;
  std::string schema_digest;
  std::string capability_hash;
  std::uint64_t expires_at_unix_ms = 0;
};

}  // namespace sirius::offload
