/*
 * Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <duckdb/common/shared_ptr.hpp>
#include <duckdb/common/unique_ptr.hpp>

namespace duckdb {
class ClientContext;
class LogicalOperator;
class PreparedStatementData;
}  // namespace duckdb

namespace sirius {
class sirius_prepared_statement_data;

/// Single logical-plan-to-Sirius preparation boundary shared by SQL and
/// Substrait entry points. This function plans only; it does not allocate GPU
/// execution state or start a backend.
duckdb::shared_ptr<sirius_prepared_statement_data> prepare_sirius_statement(
  duckdb::ClientContext& context,
  duckdb::shared_ptr<duckdb::PreparedStatementData> prepared,
  duckdb::unique_ptr<duckdb::LogicalOperator> logical_plan);

}  // namespace sirius
