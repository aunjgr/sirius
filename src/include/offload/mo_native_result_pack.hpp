/*
 * Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sirius::offload {

struct mo_native_result_column {
  std::uint32_t oid     = 0;
  std::int32_t width    = 0;
  std::int32_t scale    = 0;
  std::uint32_t charset = 0;
  bool not_nullable     = false;
};

bool can_pack_mo_native_result_on_gpu(cudf::table_view table,
                                      const std::vector<mo_native_result_column>& schema,
                                      std::size_t minimum_bytes);

std::string pack_mo_native_result_on_gpu(cudf::table_view table,
                                         const std::vector<mo_native_result_column>& schema,
                                         std::size_t row_offset,
                                         std::size_t row_count,
                                         rmm::cuda_stream_view stream);

}  // namespace sirius::offload
