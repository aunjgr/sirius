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

#include "op/sirius_physical_gpu_tae_scan.hpp"

#include "expression_executor/gpu_expression_translator_internal.hpp"
#include "log/logging.hpp"
#include "op/scan/scan_utils.hpp"
#include "op/scan/tae_scan_plan.hpp"
#include "op/sirius_physical_parquet_scan.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "sirius/exception.hpp"
#include "tae_scanner.hpp"

namespace sirius {
namespace op {

sirius_physical_gpu_tae_scan::sirius_physical_gpu_tae_scan(sirius_physical_table_scan* table_scan)
  : sirius_physical_gpu_tae_scan(
      table_scan->types,
      table_scan->function,
      table_scan->bind_data ? table_scan->bind_data->Copy() : nullptr,
      table_scan->returned_types,
      table_scan->column_ids,
      table_scan->projection_ids,
      table_scan->names,
      table_scan->table_filters ? table_scan->table_filters->Copy() : nullptr,
      table_scan->estimated_cardinality,
      copy_extra_info_parquet_scan(table_scan->extra_info),
      table_scan->parameters,
      table_scan->virtual_columns,
      table_scan)
{
}

sirius_physical_gpu_tae_scan::sirius_physical_gpu_tae_scan(
  duckdb::vector<sirius::logical_type> types,
  duckdb::TableFunction function_p,
  duckdb::unique_ptr<duckdb::FunctionData> bind_data_p,
  duckdb::vector<sirius::logical_type> returned_types_p,
  duckdb::vector<duckdb::ColumnIndex> column_ids_p,
  duckdb::vector<std::size_t> projection_ids_p,
  duckdb::vector<std::string> names_p,
  duckdb::unique_ptr<duckdb::TableFilterSet> table_filters_p,
  std::size_t estimated_cardinality,
  duckdb::ExtraOperatorInfo extra_info,
  duckdb::vector<duckdb::Value> parameters_p,
  duckdb::virtual_column_map_t virtual_columns_p,
  sirius_physical_table_scan* table_scan)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::GPU_TAE_SCAN, std::move(types), estimated_cardinality),
    function(std::move(function_p)),
    bind_data(std::move(bind_data_p)),
    returned_types(std::move(returned_types_p)),
    column_ids(std::move(column_ids_p)),
    projection_ids(std::move(projection_ids_p)),
    names(std::move(names_p)),
    table_filters(std::move(table_filters_p)),
    extra_info(std::move(extra_info)),
    parameters(std::move(parameters_p)),
    virtual_columns(std::move(virtual_columns_p))
{
  // Build the canonical scan plan once. All downstream consumers — the AST
  // filter translation below, the tae_scan_task_global_state ctor, and the
  // per-task compute_task — read from this struct instead of re-deriving
  // the same data from the operator's parallel vectors.
  if (!bind_data) {
    throw sirius::internal_exception(
      "[sirius_physical_gpu_tae_scan] missing bind_data; cannot build scan plan");
  }
  auto& tae_bind = bind_data->Cast<tae::TAEScanBindData>();
  plan           = scan::build_tae_scan_plan(
    tae_bind, column_ids, projection_ids, returned_types, this->types.size(), table_filters.get());

  if (table_filters && !table_filters->filters.empty()) {
    auto duckdb_expression = convert_table_filters_to_expression(
      *table_filters, column_ids, returned_types, plan.batch_column_map);
    if (duckdb_expression) {
      gpu_expression_translator translator(rmm::cuda_stream_default,
                                           cudf::get_current_device_resource_ref());
      // Use index-based column references (not name-based) since cudf::compute_column
      // in the TAE converter only supports cudf::ast::column_reference.
      translated_filter = translator.translate_expression(*duckdb_expression);
      if (!translated_filter) {
        SIRIUS_LOG_INFO(
          "[sirius_physical_gpu_tae_scan] Failed to translate filter expression for pushdown. "
          "Filter will be applied in the table scan operator.");
      } else {
        if (table_scan) { table_scan->passthrough = true; }
      }
      if (table_scan) { table_scan->filter_expr = sirius::wrap(std::move(duckdb_expression)); }
    }
  } else {
    translated_filter = std::nullopt;
  }
}

}  // namespace op
}  // namespace sirius
