/*
 * Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "op/sirius_physical_gpu_mo_scan.hpp"

#include "sirius/exception.hpp"

#include <algorithm>

namespace sirius::op {

sirius_physical_gpu_mo_scan::sirius_physical_gpu_mo_scan(sirius_physical_table_scan* table_scan)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::GPU_MO_SCAN, table_scan->types, table_scan->estimated_cardinality)
{
  if (!table_scan->bind_data) {
    throw internal_exception("GPU MO scan requires native stream bind data");
  }
  auto& bind = table_scan->bind_data->Cast<offload::mo_native_scan_bind_data>();
  source     = bind.source;
  if (!source) { throw internal_exception("GPU MO scan requires a native batch source"); }

  auto append_column = [&](std::size_t position) {
    if (position >= table_scan->column_ids.size() ||
        table_scan->column_ids[position].IsRowIdColumn()) {
      throw invalid_input_exception("GPU MO scan does not support synthetic row-id columns");
    }
    source_column_ids.push_back(table_scan->column_ids[position].GetPrimaryIndex());
  };
  if (!table_scan->projection_ids.empty()) {
    auto projection_ids = table_scan->projection_ids;
    std::sort(projection_ids.begin(), projection_ids.end());
    for (auto position : projection_ids) {
      append_column(position);
    }
  } else {
    for (std::size_t position = 0; position < table_scan->column_ids.size(); ++position) {
      append_column(position);
    }
  }
  if (source_column_ids.empty()) {
    throw invalid_input_exception("GPU MO scan requires at least one projected column");
  }
}

}  // namespace sirius::op
