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

// sirius
#include <config.hpp>
#include <data/host_tae_representation.hpp>
#include <op/sirius_physical_gpu_tae_scan.hpp>
#include <pipeline/sirius_pipeline_itask.hpp>
#include <pipeline/sirius_pipeline_task_states.hpp>
#include <sirius_config.hpp>
#include <sirius_context.hpp>
#include <tae/tae_format.hpp>

// tae-scanner (zone map filtering)
#include "tae_scanner.hpp"

// cucascade
#include <cucascade/data/data_repository.hpp>
#include <cucascade/memory/memory_space.hpp>

// duckdb
#include <duckdb/common/file_system.hpp>
#include <duckdb/main/client_context.hpp>

// rmm
#include <rmm/cuda_stream_view.hpp>

// standard library
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// TAE Object Partition — work unit for one TAE object file
//===----------------------------------------------------------------------===//
struct tae_object_partition {
  std::string file_path;             ///< Full URL (data_dir + relative path)
  uint32_t rows;                     ///< Total rows in the object
  uint32_t size_bytes;               ///< File size on disk
  std::vector<uint8_t> sort_key_zm;  ///< Object-level sort-key zone map (64 bytes, or empty)
};

//===----------------------------------------------------------------------===//
// TAE Scan Task Global State
//===----------------------------------------------------------------------===//
class tae_scan_task_global_state : public pipeline::sirius_pipeline_task_global_state {
 public:
  tae_scan_task_global_state(duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
                             sirius_physical_gpu_tae_scan* scan_op,
                             duckdb::ClientContext& client_ctx,
                             cucascade::memory::memory_space* host_memory_space);

  [[nodiscard]] sirius_physical_gpu_tae_scan& get_operator() { return *_scan_op; }

  /**
   * @brief Atomically claim the next object partition.
   * @return The next partition, or nullopt if exhausted.
   */
  [[nodiscard]] std::optional<tae_object_partition> claim_next_partition()
  {
    auto total          = _partitions.size();
    std::size_t current = _next_partition.load(std::memory_order_relaxed);
    while (true) {
      if (current >= total) return std::nullopt;
      if (_next_partition.compare_exchange_weak(
            current, current + 1, std::memory_order_relaxed, std::memory_order_relaxed)) {
        return _partitions[current];
      }
    }
  }

  [[nodiscard]] bool has_more_partitions() const
  {
    return _next_partition.load(std::memory_order_relaxed) < _partitions.size();
  }

  [[nodiscard]] std::size_t get_num_partitions() const { return _partitions.size(); }

  [[nodiscard]] duckdb::ClientContext& get_client_context() { return _client_ctx; }

  [[nodiscard]] cucascade::memory::memory_space* get_host_memory_space() const
  {
    return _host_memory_space;
  }

  // Column mapping info extracted from TAEScanBindData
  [[nodiscard]] std::vector<uint8_t> const& get_column_mo_oids() const { return _all_col_mo_oids; }
  [[nodiscard]] std::vector<std::string> const& get_column_names() const { return _all_col_names; }

  // Filter expression for GPU pushdown
  [[nodiscard]] std::shared_ptr<gpu_expression_translator::translated_expression>
  get_filter_expression() const
  {
    return _translated_filter;
  }

  [[nodiscard]] std::vector<std::size_t> const& get_post_filter_projection_ids() const
  {
    return _post_filter_projection_ids;
  }

  [[nodiscard]] std::vector<tae::PushedFilter> const& get_pushed_filters() const
  {
    return _pushed_filters;
  }

  [[nodiscard]] int32_t get_sort_column_idx() const { return _sort_column_idx; }

  void rebind(duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
              sirius_physical_gpu_tae_scan* scan_op)
  {
    set_pipeline(std::move(pipeline));
    _scan_op = scan_op;
    _next_partition.store(0, std::memory_order_relaxed);
    scan_op->has_more_partitions.store(true, std::memory_order_relaxed);
    scan_op->exhausted.store(false, std::memory_order_relaxed);
  }

 private:
  sirius_physical_gpu_tae_scan* _scan_op;
  duckdb::ClientContext& _client_ctx;
  cucascade::memory::memory_space* _host_memory_space;

  std::vector<tae_object_partition> _partitions;
  std::atomic<std::size_t> _next_partition{0};

  // Schema info (from TAEScanBindData)
  std::vector<std::string> _all_col_names;
  std::vector<uint8_t> _all_col_mo_oids;
  int32_t _sort_column_idx = -1;

  // Filter state
  std::shared_ptr<gpu_expression_translator::translated_expression> _translated_filter;
  std::vector<std::size_t> _post_filter_projection_ids;

  // Zone-map pushed filters (extracted from DuckDB TableFilterSet)
  std::vector<tae::PushedFilter> _pushed_filters;
};

//===----------------------------------------------------------------------===//
// TAE Scan Task Local State
//===----------------------------------------------------------------------===//
class tae_scan_task_local_state : public pipeline::sirius_pipeline_task_local_state {
 public:
  tae_scan_task_local_state(tae_scan_task_global_state& g_state, tae_object_partition partition);

  [[nodiscard]] tae_object_partition const& get_partition() const { return _partition; }
  [[nodiscard]] std::size_t get_task_consumption_basis() const override
  {
    return _partition.size_bytes;
  }

 private:
  tae_object_partition _partition;
};

//===----------------------------------------------------------------------===//
// TAE Scan Task
//===----------------------------------------------------------------------===//
class tae_scan_task : public pipeline::sirius_pipeline_itask {
  using shared_data_repository = cucascade::shared_data_repository;

 public:
  tae_scan_task(uint64_t task_id,
                shared_data_repository* data_repo,
                std::unique_ptr<tae_scan_task_local_state> l_state,
                std::shared_ptr<tae_scan_task_global_state> g_state)
    : pipeline::sirius_pipeline_itask(std::move(l_state), g_state),
      _task_id(task_id),
      _data_repo(data_repo)
  {
  }

  ~tae_scan_task() override;

  void execute(rmm::cuda_stream_view stream) override;

  std::unique_ptr<op::operator_data> compute_task(rmm::cuda_stream_view stream) override;

  void publish_output(op::operator_data& output_data, rmm::cuda_stream_view stream) override;

  [[nodiscard]] std::size_t get_estimated_reservation_size() const override;

  std::vector<op::sirius_physical_operator*> get_output_consumers() override
  {
    auto& g_state = this->_global_state->cast<tae_scan_task_global_state>();
    std::vector<sirius_physical_operator*> output_consumers;
    auto ports = g_state.get_operator().get_next_port_after_sink();
    for (auto& next_port : ports) {
      output_consumers.push_back(next_port.next_operator);
    }
    return output_consumers;
  }

  [[nodiscard]] uint64_t get_task_id() const { return _task_id; }

 private:
  // CRC detection and stripping helpers
  static bool detect_crc_format(duckdb::FileSystem& fs, duckdb::FileHandle& handle);
  static std::vector<uint8_t> read_bytes(duckdb::FileSystem& fs,
                                         duckdb::FileHandle& handle,
                                         uint64_t logical_offset,
                                         uint64_t length,
                                         bool crc_format);

  uint64_t _task_id;
  shared_data_repository* _data_repo;
};

}  // namespace sirius::op::scan
