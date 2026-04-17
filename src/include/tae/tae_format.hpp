// Copyright 2025 Matrix Origin
// SPDX-License-Identifier: Apache-2.0
//
// TAE format facade for Sirius GPU pipeline.
// Re-exports types from tae-scanner headers and adds CRC/metadata helpers.

#pragma once

// All TAE types, constants, and structs come from tae-scanner:
#include "tae_types.hpp"          // MOTypeOid, MOType, Varlena, epoch constants, etc.
#include "tae_object_reader.hpp"  // Extent, IOEntryHeader, ObjectMeta, BlockInfo, ColumnMetaInfo, etc.
#include "tae_zonemap.hpp"        // ZM_SIZE, zone map offsets

namespace tae {

// CRC block format constants (MO LocalFS wraps data in [4B CRC32][2044B content] blocks)
constexpr uint32_t CRC_SIZE         = 4;
constexpr uint32_t CRC_CONTENT_SIZE = 2044;
constexpr uint32_t CRC_BLOCK_SIZE   = 2048;

// ParseMetadata — parse ObjectMeta from decompressed metadata buffer
// (skip IOEntryHeader before calling). Standalone reimplementation for
// Sirius GPU pipeline (avoids TAEObjectReader DuckDB FileSystem dependency).
void ParseMetadata(const uint8_t *buf, uint32_t len, ObjectMeta &meta);

}  // namespace tae
