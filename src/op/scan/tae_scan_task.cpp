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

#include "op/scan/tae_scan_task.hpp"

#include "cucascade/data/data_batch.hpp"
#include "data/data_batch_utils.hpp"
#include "log/logging.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/sirius_pipeline_task_states.hpp"
#include "tae/tae_format.hpp"

// tae-scanner (for TAEScanBindData access)
#include "tae_scanner.hpp"

// tae-scanner (zone map filtering)
#include "tae_filter.hpp"

// lz4 (for metadata decompression only — column data is decompressed on GPU)
#include <lz4.h>

// duckdb
#include <duckdb/common/file_system.hpp>

// standard library
#include <algorithm>
#include <chrono>
#include <cstring>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// CPU LZ4 decompression (metadata only)
//===----------------------------------------------------------------------===//
static std::vector<uint8_t> decompress_lz4_cpu(const uint8_t* src,
                                               uint32_t src_len,
                                               uint32_t origin_size)
{
  std::vector<uint8_t> dst(origin_size);
  int decoded = LZ4_decompress_safe(reinterpret_cast<const char*>(src),
                                    reinterpret_cast<char*>(dst.data()),
                                    static_cast<int>(src_len),
                                    static_cast<int>(origin_size));
  if (decoded < 0 || static_cast<uint32_t>(decoded) != origin_size) {
    throw std::runtime_error("LZ4 metadata decompression failed: expected " +
                             std::to_string(origin_size) + " got " + std::to_string(decoded));
  }
  return dst;
}

//===----------------------------------------------------------------------===//
// CRC detection + stripping
//===----------------------------------------------------------------------===//

bool tae_scan_task::detect_crc_format(duckdb::FileSystem& fs, duckdb::FileHandle& handle)
{
  uint8_t probe[12];
  fs.Read(handle, probe, 12, 0);

  uint64_t magic_at_0, magic_at_4;
  memcpy(&magic_at_0, probe, 8);
  memcpy(&magic_at_4, probe + 4, 8);

  if (magic_at_0 == tae::OBJECT_MAGIC) return false;
  if (magic_at_4 == tae::OBJECT_MAGIC) return true;

  throw std::runtime_error("invalid TAE magic: neither raw nor CRC-wrapped");
}

std::vector<uint8_t> tae_scan_task::read_bytes(duckdb::FileSystem& fs,
                                               duckdb::FileHandle& handle,
                                               uint64_t logical_offset,
                                               uint64_t length,
                                               bool crc_format)
{
  if (!crc_format) {
    std::vector<uint8_t> buf(length);
    fs.Read(handle, buf.data(), length, logical_offset);
    return buf;
  }

  // CRC mode: translate logical offset to physical file offset
  uint64_t first_block = logical_offset / tae::CRC_CONTENT_SIZE;
  uint64_t last_block  = (logical_offset + length - 1) / tae::CRC_CONTENT_SIZE;
  uint64_t phys_start  = first_block * tae::CRC_BLOCK_SIZE;
  uint64_t phys_end    = (last_block + 1) * tae::CRC_BLOCK_SIZE;

  auto file_size = handle.GetFileSize();
  if (phys_end > static_cast<uint64_t>(file_size)) { phys_end = static_cast<uint64_t>(file_size); }

  uint64_t raw_size = phys_end - phys_start;
  std::vector<uint8_t> raw(raw_size);
  fs.Read(handle, raw.data(), raw_size, phys_start);

  // Strip CRC headers
  uint64_t num_blocks = (raw_size + tae::CRC_BLOCK_SIZE - 1) / tae::CRC_BLOCK_SIZE;
  std::vector<uint8_t> stripped;
  stripped.reserve(num_blocks * tae::CRC_CONTENT_SIZE);

  for (uint64_t off = 0; off < raw_size; off += tae::CRC_BLOCK_SIZE) {
    uint64_t remaining = raw_size - off;
    if (remaining <= tae::CRC_SIZE) break;
    uint32_t content_len = static_cast<uint32_t>(
      std::min(static_cast<uint64_t>(tae::CRC_CONTENT_SIZE), remaining - tae::CRC_SIZE));
    stripped.insert(stripped.end(),
                    raw.data() + off + tae::CRC_SIZE,
                    raw.data() + off + tae::CRC_SIZE + content_len);
  }

  // Slice the requested logical range
  uint64_t content_start = first_block * tae::CRC_CONTENT_SIZE;
  uint64_t local_offset  = logical_offset - content_start;

  if (local_offset + length > stripped.size()) {
    throw std::runtime_error("CRC read: offset+length exceeds stripped range");
  }
  return std::vector<uint8_t>(stripped.begin() + local_offset,
                              stripped.begin() + local_offset + length);
}

//===----------------------------------------------------------------------===//
// tae_scan_task_global_state
//===----------------------------------------------------------------------===//

tae_scan_task_global_state::tae_scan_task_global_state(
  duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
  sirius_physical_gpu_tae_scan* scan_op,
  duckdb::ClientContext& client_ctx,
  cucascade::memory::memory_space* host_memory_space,
  uint64_t scan_task_batch_size)
  : sirius_pipeline_task_global_state(std::move(pipeline)),
    _scan_op(scan_op),
    _client_ctx(client_ctx),
    _host_memory_space(host_memory_space),
    _scan_task_batch_size(scan_task_batch_size)
{
  // Extract object list and schema from TAEScanBindData
  auto& bind_data = scan_op->bind_data->Cast<tae::TAEScanBindData>();

  _all_col_names   = bind_data.all_col_names;
  _all_col_mo_oids = bind_data.all_col_mo_oids;
  _sort_column_idx = bind_data.sort_column_idx;

  // Build object partitions
  _partitions.reserve(bind_data.objects.size());
  for (auto& obj : bind_data.objects) {
    tae_object_partition p;
    p.file_path   = bind_data.data_dir + "/" + obj.file_path;
    p.rows        = obj.rows;
    p.size_bytes  = obj.size_bytes;
    p.sort_key_zm = obj.sort_key_zm;
    _partitions.push_back(std::move(p));
  }

  SIRIUS_LOG_INFO("[tae_scan_task_global_state] {} object partitions, {} total rows",
                  _partitions.size(),
                  bind_data.total_rows);

  // Propagate filter expression from the scan operator
  if (scan_op->translated_filter.has_value()) {
    _translated_filter = std::make_shared<gpu_expression_translator::translated_expression>(
      std::move(*scan_op->translated_filter));
  }

  // Build post-filter projection IDs (same logic as parquet).
  // Only the first types.size() entries are output columns;
  // any extra entries in projection_ids are pure filter columns.
  auto n_output = std::min(scan_op->projection_ids.size(), scan_op->types.size());
  for (std::size_t i = 0; i < n_output; i++) {
    _post_filter_projection_ids.push_back(scan_op->projection_ids[i]);
  }

  // Extract pushed filters from DuckDB TableFilterSet for zone-map pruning.
  // Each table_filters entry is keyed by column_ids index; we resolve the TAE
  // seqnum and MO type to build PushedFilter structs that zone map evaluation needs.
  if (scan_op->table_filters) {
    auto& col_ids = scan_op->column_ids;
    for (auto& [col_idx, filter] : scan_op->table_filters->filters) {
      if (col_idx >= col_ids.size()) continue;
      auto seqnum    = static_cast<uint16_t>(col_ids[col_idx].GetPrimaryIndex());
      uint8_t mo_oid = (seqnum < _all_col_mo_oids.size()) ? _all_col_mo_oids[seqnum] : 0;
      tae::ExtractFilter(*filter, static_cast<uint16_t>(col_idx), seqnum, mo_oid, _pushed_filters);
    }
    if (!_pushed_filters.empty()) {
      SIRIUS_LOG_INFO("[tae_scan_task_global_state] extracted {} zone-map filters",
                      _pushed_filters.size());
    }
  }

  if (_partitions.empty()) {
    scan_op->exhausted.store(true, std::memory_order_relaxed);
    scan_op->has_more_partitions.store(false, std::memory_order_relaxed);
  }
}

//===----------------------------------------------------------------------===//
// tae_scan_task_local_state
//===----------------------------------------------------------------------===//

tae_scan_task_local_state::tae_scan_task_local_state(tae_scan_task_global_state& /*g_state*/,
                                                     std::vector<tae_object_partition> partitions)
  : _partitions(std::move(partitions))
{
}

//===----------------------------------------------------------------------===//
// tae_scan_task
//===----------------------------------------------------------------------===//

tae_scan_task::~tae_scan_task()
{
  if (_global_state != nullptr) {
    auto& g_state = _global_state->cast<tae_scan_task_global_state>();
    if (auto pipeline = g_state.get_pipeline()) { pipeline->mark_task_completed(); }
  }
}

void tae_scan_task::execute(rmm::cuda_stream_view stream)
{
  auto output = compute_task(stream);
  if (output) { publish_output(*output, stream); }
}

std::size_t tae_scan_task::get_estimated_reservation_size() const
{
  return _local_state->cast<tae_scan_task_local_state>().get_task_consumption_basis();
}

std::unique_ptr<op::operator_data> tae_scan_task::compute_task(rmm::cuda_stream_view /*stream*/)
{
  auto& g_state = _global_state->cast<tae_scan_task_global_state>();
  auto& l_state = _local_state->cast<tae_scan_task_local_state>();

  auto scan_t0 = std::chrono::high_resolution_clock::now();

  // --- Shared state (computed once, reused across all objects in batch) ---

  auto& pushed_filters = g_state.get_pushed_filters();
  auto sort_column_idx = g_state.get_sort_column_idx();
  auto& col_ids        = g_state.get_operator().column_ids;
  auto& proj_ids       = g_state.get_operator().projection_ids;
  auto& returned_types = g_state.get_operator().returned_types;
  auto& mo_oids        = g_state.get_column_mo_oids();
  auto& fs             = duckdb::FileSystem::GetFileSystem(g_state.get_client_context());

  // Resolve projected column seqnums (shared across all objects)
  std::vector<uint16_t> projected_seqnums;
  std::vector<uint16_t> projected_col_ids_positions;
  if (!proj_ids.empty()) {
    for (auto pid : proj_ids) {
      if (pid < col_ids.size()) {
        auto primary_idx = col_ids[pid].GetPrimaryIndex();
        projected_seqnums.push_back(static_cast<uint16_t>(primary_idx));
        projected_col_ids_positions.push_back(static_cast<uint16_t>(pid));
      }
    }
  } else {
    for (std::size_t i = 0; i < col_ids.size(); i++) {
      auto primary_idx = col_ids[i].GetPrimaryIndex();
      projected_seqnums.push_back(static_cast<uint16_t>(primary_idx));
      projected_col_ids_positions.push_back(static_cast<uint16_t>(i));
    }
  }

  // Zone-map filter seqnums (shared across all objects)
  std::vector<uint16_t> filter_seqnums;
  {
    std::set<uint16_t> seen;
    for (auto& pf : pushed_filters) {
      if (seen.insert(pf.seqnum).second) filter_seqnums.push_back(pf.seqnum);
    }
  }

  // --- Per-object helper structs ---

  struct ReadChunk {
    uint64_t offset;
    uint32_t compressed_length;
    uint32_t origin_size;
    uint8_t alg;
    uint16_t seqnum;
    uint16_t col_ids_position;
    uint32_t block_rows;
    uint32_t null_cnt;
  };

  struct ObjectResult {
    std::vector<ReadChunk> reads;
    uint32_t total_rows         = 0;
    std::size_t total_compressed   = 0;
    std::size_t total_uncompressed = 0;
    std::unique_ptr<duckdb::FileHandle> handle;
    bool crc = false;
  };

  // --- Lambda: process one TAE object (zone-map → metadata → read plan) ---

  auto process_object = [&](const tae_object_partition& partition) -> std::optional<ObjectResult> {
    SIRIUS_LOG_INFO("[tae_scan_task] scanning object: {} ({} rows, {} bytes)",
                    partition.file_path, partition.rows, partition.size_bytes);

    // Object-level sort-key zone map pruning
    if (!partition.sort_key_zm.empty() && !pushed_filters.empty() && sort_column_idx >= 0) {
      auto sort_seqnum = static_cast<uint16_t>(sort_column_idx);
      if (!tae::ZoneMapPassesFilters(pushed_filters, partition.sort_key_zm.data(), sort_seqnum)) {
        SIRIUS_LOG_INFO("[tae_scan_task] object pruned by sort-key zone map: {}",
                        partition.file_path);
        return std::nullopt;
      }
    }

    ObjectResult result;
    result.handle = fs.OpenFile(partition.file_path, duckdb::FileOpenFlags::FILE_FLAGS_READ);
    result.crc    = detect_crc_format(fs, *result.handle);

    // Read + parse metadata
    auto header_buf = read_bytes(fs, *result.handle, 0, tae::HEADER_SIZE, result.crc);
    tae::Extent meta_ext;
    memcpy(&meta_ext, header_buf.data() + tae::HEADER_META_EXTENT_OFF, sizeof(tae::Extent));

    auto meta_raw = read_bytes(fs, *result.handle, meta_ext.offset, meta_ext.length, result.crc);
    std::vector<uint8_t> meta_buf;
    if (meta_ext.is_compressed()) {
      meta_buf = decompress_lz4_cpu(meta_raw.data(), meta_ext.length, meta_ext.origin_size);
    } else {
      meta_buf = std::move(meta_raw);
    }

    tae::ObjectMeta obj_meta;
    if (meta_buf.size() <= tae::IO_ENTRY_HEADER_LEN) {
      throw std::runtime_error("metadata too small after IOEntryHeader skip");
    }
    tae::ParseMetadata(meta_buf.data() + tae::IO_ENTRY_HEADER_LEN,
                       static_cast<uint32_t>(meta_buf.size() - tae::IO_ENTRY_HEADER_LEN),
                       obj_meta);

    // Build read plan with per-block zone-map filtering
    uint32_t blocks_pruned = 0;
    for (uint32_t b = 0; b < obj_meta.block_count; b++) {
      auto& blk = obj_meta.blocks[b];
      if (!filter_seqnums.empty()) {
        bool passes = true;
        for (auto seq : filter_seqnums) {
          if (seq < blk.columns.size() &&
              !tae::ZoneMapPassesFilters(pushed_filters, blk.columns[seq].zone_map, seq)) {
            passes = false;
            break;
          }
        }
        if (!passes) { blocks_pruned++; continue; }
      }
      result.total_rows += blk.rows;
      for (std::size_t si = 0; si < projected_seqnums.size(); si++) {
        auto seq = projected_seqnums[si];
        if (seq >= blk.columns.size()) continue;
        auto& col = blk.columns[seq];
        auto& ext = col.location;
        result.reads.push_back({ext.offset,
                                ext.length,
                                ext.origin_size,
                                ext.alg,
                                seq,
                                projected_col_ids_positions[si],
                                blk.rows,
                                col.null_cnt});
      }
    }

    if (blocks_pruned > 0) {
      SIRIUS_LOG_INFO("[tae_scan_task] zone-map pruned {}/{} blocks for {}",
                      blocks_pruned, obj_meta.block_count, partition.file_path);
    }

    if (result.reads.empty() || result.total_rows == 0) {
      SIRIUS_LOG_INFO("[tae_scan_task] no data after zone-map filtering: {}", partition.file_path);
      return std::nullopt;
    }

    for (auto& r : result.reads) {
      result.total_compressed += r.compressed_length;
      result.total_uncompressed += r.origin_size;
    }
    return result;
  };

  // --- Phase 1: Process all partitions from local_state ---

  std::vector<ObjectResult> objects;
  std::size_t cumulative_compressed = 0;
  uint32_t objects_scanned          = 0;

  for (auto& partition : l_state.get_partitions()) {
    auto result = process_object(partition);
    objects_scanned++;
    if (result) {
      cumulative_compressed += result->total_compressed;
      objects.push_back(std::move(*result));
    }
  }

  if (objects.empty()) {
    SIRIUS_LOG_INFO("[tae_scan_task] all {} objects pruned, no data to produce", objects_scanned);
    return nullptr;
  }

  SIRIUS_LOG_INFO("[tae_scan_task] batch: {} objects ({} scanned), {} cumulative compressed bytes",
                  objects.size(), objects_scanned, cumulative_compressed);

  // --- Phase 2: Compute combined totals ---

  std::size_t total_compressed = 0, total_uncompressed = 0;
  uint32_t total_rows = 0;
  for (auto& obj : objects) {
    total_compressed += obj.total_compressed;
    total_uncompressed += obj.total_uncompressed;
    total_rows += obj.total_rows;
  }

  // --- Phase 3: Combined pinned buffer + coalesced I/O + chunk building ---

  pinned_host_buffer host_data(total_compressed);
  std::vector<host_tae_representation::column_chunk_info> chunks;
  std::size_t global_offset = 0;

  static constexpr std::size_t MAX_COALESCE_GROUP = 32;

  for (auto& obj : objects) {
    // Sort reads by file offset for sequential I/O
    std::sort(obj.reads.begin(), obj.reads.end(),
              [](const ReadChunk& a, const ReadChunk& b) { return a.offset < b.offset; });

    // Coalesced I/O for this object
    {
      std::size_t group_start = 0;
      std::size_t pinned_dest = global_offset;
      uint32_t io_calls       = 0;

      auto flush_group = [&](std::size_t group_end) {
        auto& first               = obj.reads[group_start];
        auto& last                = obj.reads[group_end - 1];
        std::size_t group_log_end = last.offset + last.compressed_length;
        std::size_t group_log_len = group_log_end - first.offset;

        if (!obj.crc) {
          fs.Read(*obj.handle, host_data.data() + pinned_dest, group_log_len, first.offset);
        } else {
          uint64_t first_blk  = first.offset / tae::CRC_CONTENT_SIZE;
          uint64_t last_blk   = (group_log_end - 1) / tae::CRC_CONTENT_SIZE;
          uint64_t phys_start = first_blk * tae::CRC_BLOCK_SIZE;
          uint64_t phys_end   = (last_blk + 1) * tae::CRC_BLOCK_SIZE;
          auto file_size      = obj.handle->GetFileSize();
          if (phys_end > static_cast<uint64_t>(file_size))
            phys_end = static_cast<uint64_t>(file_size);

          uint64_t raw_size = phys_end - phys_start;
          std::vector<uint8_t> raw(raw_size);
          fs.Read(*obj.handle, raw.data(), raw_size, phys_start);

          std::vector<uint8_t> stripped;
          stripped.reserve((raw_size / tae::CRC_BLOCK_SIZE + 1) * tae::CRC_CONTENT_SIZE);
          for (uint64_t off = 0; off < raw_size; off += tae::CRC_BLOCK_SIZE) {
            uint64_t remaining = raw_size - off;
            if (remaining <= tae::CRC_SIZE) break;
            uint32_t content_len = static_cast<uint32_t>(
              std::min(static_cast<uint64_t>(tae::CRC_CONTENT_SIZE), remaining - tae::CRC_SIZE));
            stripped.insert(stripped.end(),
                            raw.data() + off + tae::CRC_SIZE,
                            raw.data() + off + tae::CRC_SIZE + content_len);
          }

          uint64_t content_start = first_blk * tae::CRC_CONTENT_SIZE;
          std::size_t dest       = pinned_dest;
          for (std::size_t j = group_start; j < group_end; j++) {
            uint64_t local_off = obj.reads[j].offset - content_start;
            if (local_off + obj.reads[j].compressed_length > stripped.size()) {
              throw std::runtime_error("CRC coalesced read: offset exceeds stripped range");
            }
            std::memcpy(
              host_data.data() + dest, stripped.data() + local_off, obj.reads[j].compressed_length);
            dest += obj.reads[j].compressed_length;
          }
        }

        io_calls++;
        for (std::size_t j = group_start; j < group_end; j++)
          pinned_dest += obj.reads[j].compressed_length;
        group_start = group_end;
      };

      for (std::size_t i = 0; i <= obj.reads.size(); i++) {
        std::size_t group_len = i - group_start;
        bool adjacent =
          (i < obj.reads.size()) &&
          (i == 0 || obj.reads[i].offset == obj.reads[i - 1].offset + obj.reads[i - 1].compressed_length);
        bool end_of_group =
          (i == obj.reads.size()) || (!adjacent && i > group_start) || (group_len >= MAX_COALESCE_GROUP);
        if (end_of_group && i > group_start) { flush_group(i); }
      }
      SIRIUS_LOG_INFO(
        "[tae_scan_task] coalesced {} reads into {} I/O calls (crc={})", obj.reads.size(), io_calls, obj.crc);
    }

    // Build chunk metadata for this object
    for (auto& r : obj.reads) {
      tae::MOTypeOid type_oid = tae::MO_T_any;
      int32_t width = 0, scale = 0;
      if (r.seqnum < mo_oids.size()) { type_oid = static_cast<tae::MOTypeOid>(mo_oids[r.seqnum]); }
      if (r.seqnum < returned_types.size() && returned_types[r.seqnum].is_decimal()) {
        width = static_cast<int32_t>(returned_types[r.seqnum].decimal_precision());
        scale = static_cast<int32_t>(returned_types[r.seqnum].decimal_scale());
      }

      host_tae_representation::column_chunk_info chunk;
      chunk.column_idx = r.col_ids_position;
      chunk.type_oid   = type_oid;
      chunk.width      = width;
      chunk.scale      = scale;
      chunk.extent =
        tae::Extent{r.alg, static_cast<uint32_t>(r.offset), r.compressed_length, r.origin_size};
      chunk.null_cnt      = r.null_cnt;
      chunk.row_count     = r.block_rows;
      chunk.pinned_offset = global_offset;
      chunk.pinned_length = r.compressed_length;
      chunks.push_back(chunk);

      global_offset += r.compressed_length;
    }
  }

  // --- Phase 4: Build host_tae_representation ---

  auto* host_space = g_state.get_host_memory_space();

  auto repr = std::make_unique<host_tae_representation>(host_space,
                                                        std::move(host_data),
                                                        std::move(chunks),
                                                        total_rows,
                                                        total_compressed,
                                                        total_uncompressed,
                                                        g_state.get_filter_expression(),
                                                        g_state.get_post_filter_projection_ids());

  auto batch = std::make_shared<cucascade::data_batch>(get_next_batch_id(), std::move(repr));

  SIRIUS_LOG_INFO("[tae_scan_task] produced batch: {} objects, {} rows, {} compressed, {} uncompressed",
                  objects.size(), total_rows, total_compressed, total_uncompressed);

  auto scan_t1 = std::chrono::high_resolution_clock::now();
  auto scan_ms = std::chrono::duration<double, std::milli>(scan_t1 - scan_t0).count();
  SIRIUS_LOG_INFO("[tae_scan_task] scan I/O total: {:.2f} ms ({} bytes pinned)",
                  scan_ms, global_offset);

  return std::make_unique<op::pipelineable_operator_data>(
    std::vector<std::shared_ptr<cucascade::data_batch>>{std::move(batch)});
}

void tae_scan_task::publish_output(op::operator_data& output_data, rmm::cuda_stream_view /*stream*/)
{
  auto& pipelineable_output = dynamic_cast<op::pipelineable_operator_data&>(output_data);
  for (auto& batch : pipelineable_output.release_data_batches()) {
    _data_repo->add_data_batch(std::move(batch));
  }
}

}  // namespace sirius::op::scan
