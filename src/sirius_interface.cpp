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

#include "sirius_interface.hpp"

#include "duckdb/execution/operator/helper/physical_result_collector.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/pending_query_result.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/main/query_result.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/planner/planner.hpp"
#include "helper/type_conversions.hpp"
#include "log/logging.hpp"

#include <chrono>

namespace sirius {

void bind_prepared_statement_parameters(duckdb::PreparedStatementData& statement,
                                        const duckdb::PendingQueryParameters& parameters)
{
  duckdb::case_insensitive_map_t<duckdb::BoundParameterData> owned_values;
  statement.Bind(std::move(owned_values));
}

sirius_interface::sirius_interface(duckdb::ClientContext& client_context,
                                   std::shared_ptr<execution_evidence> evidence)
  : client_context(client_context), evidence(std::move(evidence)) {};

void sirius_interface::sirius_process_error(duckdb::ErrorData& error,
                                            const duckdb::string& query) const
{
  error.FinalizeError();
  if (duckdb::Settings::Get<duckdb::ErrorsAsJSONSetting>(client_context)) {
    error.ConvertErrorToJSON();
  } else {
    error.AddErrorLocation(query);
  }
}

template <class T>
duckdb::unique_ptr<T> sirius_interface::sirius_error_result(duckdb::ErrorData error,
                                                            const duckdb::string& query)
{
  sirius_process_error(error, query);
  return duckdb::make_uniq<T>(std::move(error));
}

void sirius_interface::check_executable_internal(duckdb::PendingQueryResult& pending)
{
  // bool invalidated = HasError() || !(client_context);
  D_ASSERT(sirius_active_query->is_open_result(pending));
  bool invalidated = pending.HasError();
  if (!invalidated) {
    D_ASSERT(sirius_active_query);
    invalidated = !sirius_active_query->is_open_result(pending);
  }
  if (invalidated) {
    if (pending.HasError()) {
      throw duckdb::InvalidInputException(
        "Attempting to execute an unsuccessful pending query result\n");
    }
    throw duckdb::InvalidInputException("Attempting to execute a closed pending query result");
  }
}

void sirius_interface::begin_query_internal(const duckdb::string& query)
{
  // check if we are on AutoCommit. In this case we should start a transaction
  D_ASSERT(!sirius_active_query);
  sirius_active_query        = duckdb::make_uniq<sirius_active_query_context>();
  sirius_active_query->query = query;
}

duckdb::unique_ptr<duckdb::QueryResult> sirius_interface::fetch_result_internal(
  duckdb::PendingQueryResult& pending)
{
  D_ASSERT(sirius_active_query);
  D_ASSERT(sirius_active_query->is_open_result(pending));
  D_ASSERT(sirius_active_query->sirius_prepared->prepared);
  auto& engine   = get_sirius_engine();
  auto& prepared = *sirius_active_query->sirius_prepared->prepared;
  duckdb::unique_ptr<duckdb::QueryResult> result;
  D_ASSERT(engine.has_result_collector());
  SIRIUS_LOG_DEBUG("Fetching result from GPU executor");
  result = engine.get_result();
  cleanup_internal(result.get(), false);
  return result;
}

void sirius_interface::cleanup_internal(duckdb::BaseQueryResult* result,
                                        bool invalidate_transaction)
{
  if (!sirius_active_query) {
    // no query currently active
    return;
  }
  sirius_active_query->progress_bar.reset();

  auto error = end_query_internal(result ? !result->HasError() : false, invalidate_transaction);
  if (result && !result->HasError()) { result->SetError(error); }
  D_ASSERT(!sirius_active_query);
}

duckdb::ErrorData sirius_interface::end_query_internal(bool success, bool invalidate_transaction)
{
  sirius_active_query->progress_bar.reset();
  D_ASSERT(sirius_active_query.get());
  sirius_active_query.reset();
  duckdb::ErrorData error;
  return error;
}

// This function is based on ClientContext::PendingStatementOrPreparedStatement
duckdb::unique_ptr<duckdb::PendingQueryResult>
sirius_interface::sirius_pending_statement_or_prepared_statement(
  duckdb::ClientContext& context,
  const duckdb::string& query,
  duckdb::shared_ptr<sirius_prepared_statement_data>& statement_p,
  const duckdb::PendingQueryParameters& parameters)
{
  begin_query_internal(query);

  bool invalidate_query = true;
  duckdb::unique_ptr<duckdb::PendingQueryResult> pending =
    sirius_pending_statement_internal(context, statement_p, parameters);

  if (pending->HasError()) { return pending; }
  D_ASSERT(sirius_active_query->is_open_result(*pending));
  return pending;
};

// This function is based on ClientContext::PendingPreparedStatementInternal
duckdb::unique_ptr<duckdb::PendingQueryResult> sirius_interface::sirius_pending_statement_internal(
  duckdb::ClientContext& context,
  duckdb::shared_ptr<sirius_prepared_statement_data>& statement_p,
  const duckdb::PendingQueryParameters& parameters)
{
  D_ASSERT(sirius_active_query);
  auto& statement = *(statement_p->prepared);

  bind_prepared_statement_parameters(statement, parameters);

  duckdb::unique_ptr<sirius_engine> temp = duckdb::make_uniq<sirius_engine>(context, *this);
  auto prop                              = temp->context.GetClientProperties();
  sirius_active_query->engine            = std::move(temp);
  auto& engine                           = get_sirius_engine();
  bool stream_result                     = false;

  duckdb::unique_ptr<op::sirius_physical_result_collector> sirius_collector =
    duckdb::make_uniq_base<op::sirius_physical_result_collector,
                           op::sirius_physical_materialized_collector>(*statement_p,
                                                                       client_context);
  if (sirius_collector->type != op::SiriusPhysicalOperatorType::RESULT_COLLECTOR) {
    return sirius_error_result<duckdb::PendingQueryResult>(
      duckdb::ErrorData("Error in sirius_pending_statement_internal"));
  }
  D_ASSERT(sirius_collector->type == op::SiriusPhysicalOperatorType::RESULT_COLLECTOR);
  auto types = sirius::to_duckdb_vec(sirius_collector->get_types());
  D_ASSERT(types == statement.types);
  engine.initialize(std::move(sirius_collector));

  D_ASSERT(!sirius_active_query->has_open_result());

  auto pending_result = duckdb::make_uniq<duckdb::PendingQueryResult>(
    context.shared_from_this(), *(statement_p->prepared), std::move(types), stream_result);
  sirius_active_query->sirius_prepared = std::move(statement_p);
  sirius_active_query->set_open_result(*pending_result);
  return pending_result;
};

// This function is based on PendingQueryResult::ExecuteInternal
duckdb::unique_ptr<duckdb::QueryResult> sirius_interface::sirius_execute_pending_query_result(
  duckdb::PendingQueryResult& pending)
{
  D_ASSERT(sirius_active_query->is_open_result(pending));
  check_executable_internal(pending);
  auto& engine = get_sirius_engine();
  try {
    SIRIUS_LOG_DEBUG("Executing sirius_engine");
    if (evidence) { (void)evidence->mark_backend_started(execution_backend::SIRIUS_GPU); }
    auto exec_t0 = std::chrono::high_resolution_clock::now();
    engine.execute();
    auto exec_t1 = std::chrono::high_resolution_clock::now();
    auto exec_ms = std::chrono::duration<double, std::milli>(exec_t1 - exec_t0).count();
    SIRIUS_LOG_INFO("[sirius_interface] engine.execute(): {:.2f} ms", exec_ms);
  } catch (std::exception& e) {
    duckdb::ErrorData error(e);
    SIRIUS_LOG_ERROR("Error in sirius_execute_pending_query_result: {}", error.RawMessage());
    return sirius_error_result<duckdb::MaterializedQueryResult>(error);
  }
  if (pending.HasError()) {
    duckdb::ErrorData error = pending.GetErrorObject();
    return duckdb::make_uniq<duckdb::MaterializedQueryResult>(error);
  }
  SIRIUS_LOG_DEBUG("Done sirius_execute_pending_query_result");
  auto result = fetch_result_internal(pending);
  return result;
}

// This function is based on ClientContext::Query
duckdb::unique_ptr<duckdb::QueryResult> sirius_interface::sirius_execute_query(
  duckdb::ClientContext& context,
  const duckdb::string& query,
  duckdb::shared_ptr<sirius_prepared_statement_data>& statement_p,
  const duckdb::PendingQueryParameters& parameters)
{
  auto query_t0 = std::chrono::high_resolution_clock::now();
  auto pending_query =
    sirius_pending_statement_or_prepared_statement(context, query, statement_p, parameters);
  auto plan_t1 = std::chrono::high_resolution_clock::now();
  auto plan_ms = std::chrono::duration<double, std::milli>(plan_t1 - query_t0).count();
  SIRIUS_LOG_INFO("[sirius_interface] planning: {:.2f} ms", plan_ms);
  D_ASSERT(sirius_active_query->is_open_result(*pending_query));
  duckdb::unique_ptr<duckdb::QueryResult> current_result;
  if (pending_query->HasError()) {
    current_result =
      sirius_error_result<duckdb::MaterializedQueryResult>(pending_query->GetErrorObject());
  } else {
    current_result = sirius_execute_pending_query_result(*pending_query);
  }
  SIRIUS_LOG_DEBUG("Done sirius_execute_query");
  auto query_t1 = std::chrono::high_resolution_clock::now();
  auto total_ms = std::chrono::duration<double, std::milli>(query_t1 - query_t0).count();
  SIRIUS_LOG_INFO("[sirius_interface] total query: {:.2f} ms", total_ms);
  return current_result;
};

void sirius_interface::sirius_execute_streaming(
  duckdb::ClientContext& context,
  const duckdb::string& query,
  duckdb::shared_ptr<sirius_prepared_statement_data> statement,
  std::function<bool(const duckdb::DataChunk&)> callback)
{
  if (!statement || !callback) {
    throw duckdb::InvalidInputException("streaming Sirius execution requires a plan and callback");
  }

  begin_query_internal(query);
  try {
    auto engine                 = duckdb::make_uniq<sirius_engine>(context, *this);
    sirius_active_query->engine = std::move(engine);
    auto collector              = duckdb::make_uniq<op::sirius_physical_streaming_collector>(
      *statement, client_context, std::move(callback));
    get_sirius_engine().initialize(std::move(collector));
    sirius_active_query->sirius_prepared = std::move(statement);

    if (evidence) { (void)evidence->mark_backend_started(execution_backend::SIRIUS_GPU); }
    get_sirius_engine().execute();
    (void)end_query_internal(true, false);
  } catch (...) {
    if (sirius_active_query) { (void)end_query_internal(false, false); }
    throw;
  }
}

void sirius_interface::sirius_execute_batch_streaming(
  duckdb::ClientContext& context,
  const duckdb::string& query,
  duckdb::shared_ptr<sirius_prepared_statement_data> statement,
  op::result_batch_callback callback)
{
  if (!statement || !callback) {
    throw duckdb::InvalidInputException(
      "streaming Sirius execution requires a plan and batch callback");
  }

  begin_query_internal(query);
  try {
    auto engine                 = duckdb::make_uniq<sirius_engine>(context, *this);
    sirius_active_query->engine = std::move(engine);
    auto collector              = duckdb::make_uniq<op::sirius_physical_streaming_collector>(
      *statement, client_context, std::move(callback));
    get_sirius_engine().initialize(std::move(collector));
    sirius_active_query->sirius_prepared = std::move(statement);
    if (evidence) { (void)evidence->mark_backend_started(execution_backend::SIRIUS_GPU); }
    get_sirius_engine().execute();
    (void)end_query_internal(true, false);
  } catch (...) {
    if (sirius_active_query) { (void)end_query_internal(false, false); }
    throw;
  }
}

sirius::sirius_engine& sirius_interface::get_sirius_engine()
{
  D_ASSERT(sirius_active_query);
  D_ASSERT(sirius_active_query->engine);
  return *sirius_active_query->engine;
}

};  // namespace sirius
