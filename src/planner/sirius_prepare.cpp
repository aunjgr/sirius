/*
 * Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "planner/sirius_prepare.hpp"

#include "planner/sirius_physical_plan_generator.hpp"
#include "sirius_interface.hpp"

#include <duckdb/main/client_context.hpp>
#include <duckdb/main/prepared_statement_data.hpp>
#include <duckdb/planner/logical_operator.hpp>

namespace sirius {

duckdb::shared_ptr<sirius_prepared_statement_data> prepare_sirius_statement(
  duckdb::ClientContext& context,
  duckdb::shared_ptr<duckdb::PreparedStatementData> prepared,
  duckdb::unique_ptr<duckdb::LogicalOperator> logical_plan)
{
  if (!prepared || !logical_plan) {
    throw duckdb::InvalidInputException("cannot prepare an empty Sirius logical plan");
  }
  planner::sirius_physical_plan_generator physical_planner(context);
  auto physical_plan = physical_planner.create_plan(std::move(logical_plan));
  return duckdb::make_shared_ptr<sirius_prepared_statement_data>(std::move(prepared),
                                                                 std::move(physical_plan));
}

}  // namespace sirius
