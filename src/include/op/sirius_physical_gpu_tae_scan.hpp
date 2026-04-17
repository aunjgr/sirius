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

#pragma once

#include "duckdb/common/extra_operator_info.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/storage/data_table.hpp"
#include "expression_executor/gpu_expression_translator.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_table_scan.hpp"

namespace sirius {
namespace op {

class sirius_physical_gpu_tae_scan : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::GPU_TAE_SCAN;

 public:
  sirius_physical_gpu_tae_scan(sirius_physical_table_scan* table_scan);

  sirius_physical_gpu_tae_scan(duckdb::vector<duckdb::LogicalType> types,
                               duckdb::TableFunction function,
                               duckdb::unique_ptr<duckdb::FunctionData> bind_data,
                               duckdb::vector<duckdb::LogicalType> returned_types,
                               duckdb::vector<duckdb::ColumnIndex> column_ids,
                               duckdb::vector<std::size_t> projection_ids,
                               duckdb::vector<std::string> names,
                               duckdb::unique_ptr<duckdb::TableFilterSet> table_filters,
                               std::size_t estimated_cardinality,
                               duckdb::ExtraOperatorInfo extra_info,
                               duckdb::vector<duckdb::Value> parameters,
                               duckdb::virtual_column_map_t virtual_columns,
                               sirius_physical_table_scan* physical_table_scan);

  std::optional<task_creation_hint> get_next_task_hint() override
  {
    if (exhausted.load()) { return std::nullopt; }
    return task_creation_hint{TaskCreationHint::READY, this};
  }

  //! The table function
  duckdb::TableFunction function;
  //! Bind data of the function
  duckdb::unique_ptr<duckdb::FunctionData> bind_data;
  //! The types of ALL columns that can be returned by the table function
  duckdb::vector<duckdb::LogicalType> returned_types;
  //! The column ids used within the table function
  duckdb::vector<duckdb::ColumnIndex> column_ids;
  //! The projected-out column ids
  duckdb::vector<std::size_t> projection_ids;
  //! The names of the columns
  duckdb::vector<std::string> names;
  //! The table filters
  duckdb::unique_ptr<duckdb::TableFilterSet> table_filters;
  //! Extra operator info
  duckdb::ExtraOperatorInfo extra_info;
  //! Parameters
  duckdb::vector<duckdb::Value> parameters;
  //! Dynamic filters
  duckdb::shared_ptr<duckdb::DynamicTableFilterSet> dynamic_filters;
  //! Virtual columns
  duckdb::virtual_column_map_t virtual_columns;

  duckdb::PhysicalTableScan* physical_table_scan;

  duckdb::unique_ptr<duckdb::ColumnDataCollection> collection;

  duckdb::vector<duckdb::LogicalType> scanned_types;
  duckdb::vector<std::size_t> scanned_ids;

  std::atomic<bool> exhausted{false};
  std::atomic<bool> has_more_partitions{true};

  //! Translated filter for GPU pushdown
  std::optional<gpu_expression_translator::translated_expression> translated_filter;

 public:
  bool is_source() const override { return true; }
};

}  // namespace op
}  // namespace sirius
