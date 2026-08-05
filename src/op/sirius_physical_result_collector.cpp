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

// sirius

#include <nvtx3/nvtx3.hpp>

#include <data/sirius_converter_registry.hpp>
#include <helper/type_conversions.hpp>
#include <op/result/host_table_chunk_reader.hpp>
#include <op/sirius_physical_result_collector.hpp>
#include <pipeline/sirius_meta_pipeline.hpp>
#include <pipeline/sirius_pipeline.hpp>
#include <sirius_interface.hpp>

// cucascade
#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>

// sirius exceptions
#include "sirius/exception.hpp"

// duckdb
#include <duckdb/main/materialized_query_result.hpp>
#include <duckdb/main/prepared_statement_data.hpp>

// standard library
#include <algorithm>
#include <cassert>

namespace sirius {
namespace op {

namespace {

template <typename VISITOR>
void visit_result_chunks(duckdb::ClientContext& client_ctx,
                         const duckdb::vector<sirius::logical_type>& types,
                         const operator_data& input_data,
                         rmm::cuda_stream_view stream,
                         VISITOR&& visitor)
{
  auto& pipelineable_input      = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches     = pipelineable_input.get_data_batches();
  using host_table_chunk_reader = ::sirius::op::result::host_table_chunk_reader;

  for (const auto& input_batch : input_batches) {
    auto* data = input_batch->get_data();
    std::shared_ptr<cucascade::data_batch> clone_batch;
    if (!data) { throw invalid_input_exception("result collector received a batch without data"); }
    if (data->get_size_in_bytes() == 0) { continue; }

    if (data->get_current_tier() == cucascade::memory::Tier::GPU) {
      auto sirius_ctx = client_ctx.registered_state->Get<duckdb::SiriusContext>("sirius_state");
      if (!sirius_ctx) {
        throw internal_exception("result collector requires an initialized Sirius context");
      }
      auto& memory_mgr = sirius_ctx->get_memory_manager();
      auto reservation = memory_mgr.request_reservation(
        cucascade::memory::any_memory_space_in_tier{cucascade::memory::Tier::HOST},
        data->get_size_in_bytes());
      if (!reservation) {
        throw internal_exception("result collector failed to reserve host memory");
      }

      auto& registry      = sirius::converter_registry::get();
      auto& mem_space     = reservation->get_memory_space();
      auto& data_repo_mgr = sirius_ctx->get_data_repository_manager();
      clone_batch         = input_batch->clone(data_repo_mgr.get_next_data_batch_id(), stream);
      clone_batch->convert_to<cucascade::host_data_representation>(registry, &mem_space, stream);
      data = clone_batch->get_data();
    } else if (data->get_current_tier() != cucascade::memory::Tier::HOST) {
      throw invalid_input_exception("result collector only accepts HOST or GPU data");
    }

    auto* host_data = dynamic_cast<cucascade::host_data_representation*>(data);
    if (!host_data) {
      throw invalid_input_exception("result collector expected host_data_representation");
    }
    auto const* host_table = host_data->get_host_table().get();
    if (!host_table || !host_table->allocation) {
      throw invalid_input_exception("result collector received invalid host table storage");
    }

    host_table_chunk_reader chunk_reader(client_ctx, *host_data, types);
    while (true) {
      duckdb::DataChunk chunk;
      if (!chunk_reader.get_next_chunk(chunk)) { break; }
      visitor(chunk);
    }
  }
}

}  // namespace

sirius_physical_result_collector::sirius_physical_result_collector(
  ::sirius::sirius_prepared_statement_data& data)
  : sirius_physical_operator(SiriusPhysicalOperatorType::RESULT_COLLECTOR,
                             {sirius::logical_type::make(sirius::type_id::BOOLEAN)},
                             0),
    statement_type(data.prepared->statement_type),
    properties(data.prepared->properties),
    plan(*data.sirius_physical_plan),
    names(data.prepared->names)
{
  this->types = sirius::from_duckdb_vec(data.prepared->types);
}

std::unique_ptr<operator_data> sirius_physical_result_collector::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_result_collector::execute"};
  return std::make_unique<pipelineable_operator_data>(
    dynamic_cast<const pipelineable_operator_data&>(input_data).get_data_batches());
}

duckdb::vector<duckdb::const_reference<sirius_physical_operator>>
sirius_physical_result_collector::get_children() const
{
  return {plan};
}

void sirius_physical_result_collector::build_pipelines(
  pipeline::sirius_pipeline& current, pipeline::sirius_meta_pipeline& meta_pipeline)
{
  // operator is a sink, build a pipeline
  D_ASSERT(children.empty());

  // single operator: the operator becomes the data source of the current pipeline
  auto& state = meta_pipeline.get_state();
  state.set_pipeline_source(current, *this);

  // we create a new pipeline starting from the child
  auto& child_meta_pipeline = meta_pipeline.create_child_meta_pipeline(current, *this);
  child_meta_pipeline.build(plan);
}

sirius_physical_materialized_collector::sirius_physical_materialized_collector(
  ::sirius::sirius_prepared_statement_data& data, duckdb::ClientContext& client_ctx)
  : sirius_physical_result_collector(data),
    _client_ctx(client_ctx),
    result_collection(
      duckdb::make_uniq<duckdb::ColumnDataCollection>(client_ctx, sirius::to_duckdb_vec(types)))
{
}

duckdb::unique_ptr<duckdb::QueryResult> sirius_physical_materialized_collector::get_result()
{
  auto props = _client_ctx.GetClientProperties();

  std::lock_guard<std::mutex> guard(lock);
  // Return an empty result collection if the result_collection is null (from a move)
  if (!result_collection) {
    result_collection =
      duckdb::make_uniq<duckdb::ColumnDataCollection>(_client_ctx, sirius::to_duckdb_vec(types));
  }

  return duckdb::make_uniq<duckdb::MaterializedQueryResult>(
    statement_type, properties, names, std::move(result_collection), props);
}

void sirius_physical_materialized_collector::sink(const operator_data& input_data,
                                                  rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_materialized_collector::sink"};
  visit_result_chunks(_client_ctx, types, input_data, stream, [this](duckdb::DataChunk& chunk) {
    std::lock_guard<std::mutex> guard(lock);
    if (!result_collection) {
      result_collection =
        duckdb::make_uniq<duckdb::ColumnDataCollection>(_client_ctx, sirius::to_duckdb_vec(types));
    }
    result_collection->Append(chunk);
  });
}

sirius_physical_streaming_collector::sirius_physical_streaming_collector(
  ::sirius::sirius_prepared_statement_data& data,
  duckdb::ClientContext& client_ctx,
  result_chunk_callback callback)
  : sirius_physical_result_collector(data), client_ctx_(client_ctx), callback_(std::move(callback))
{
  if (!callback_) { throw invalid_input_exception("streaming result callback is required"); }
}

duckdb::unique_ptr<duckdb::QueryResult> sirius_physical_streaming_collector::get_result()
{
  throw internal_exception("streaming result collector cannot materialize a QueryResult");
}

void sirius_physical_streaming_collector::sink(const operator_data& input_data,
                                               rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_streaming_collector::sink"};
  std::lock_guard<std::mutex> guard(lock);
  visit_result_chunks(client_ctx_, types, input_data, stream, [this](duckdb::DataChunk& chunk) {
    if (!callback_(chunk)) {
      throw invalid_input_exception("streaming result consumer cancelled execution");
    }
  });
}

}  // namespace op
}  // namespace sirius
