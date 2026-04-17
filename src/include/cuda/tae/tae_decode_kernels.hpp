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

#include <cstddef>
#include <cstdint>

#include <rmm/cuda_stream_view.hpp>

namespace sirius::cuda::tae {

/**
 * @brief Decode a fixed-width TAE column from decompressed MO format to cuDF format.
 *
 * Caller must provide a pointer to the data section of the MO vector
 * (past IOEntryHeader + vector header). This kernel copies the data portion
 * directly (possibly with epoch adjustment for DATE/TIMESTAMP types).
 * The null bitmap is handled separately by invert_null_mask.
 *
 * @param d_src        Pointer to the data section of the MO vector
 * @param d_dst        Output buffer for cuDF column values
 * @param n_rows       Number of rows
 * @param elem_size    Element size in bytes (1,2,4,8,16)
 * @param epoch_adjust Value to subtract for epoch conversion (0 for non-temporal types,
 *                     MO_UNIX_EPOCH_DAYS for DATE, MO_UNIX_EPOCH_USEC for TIMESTAMP)
 * @param stream       CUDA stream
 */
void decode_fixed_width(const uint8_t* d_src,
                        uint8_t* d_dst,
                        uint32_t n_rows,
                        uint32_t elem_size,
                        int64_t epoch_adjust,
                        rmm::cuda_stream_view stream);

/**
 * @brief Decode a varlena TAE column to cuDF string column format.
 *
 * Caller provides separate pointers to the varlena struct array (data section)
 * and the area section (for big string payloads). These are obtained by parsing
 * the MO vector header offsets.
 *
 * Each Varlena is 24 bytes. If data[0] <= 23, data is inline (data[1..data[0]]).
 * Otherwise, it's a big marker with offset into area + length.
 *
 * This kernel produces cuDF string column format:
 *   offsets: int32[n_rows+1] — exclusive cumulative byte offsets
 *   chars:   packed UTF-8 character data
 *
 * Uses CUB device-wide exclusive sum for the prefix-sum.
 *
 * @param d_varlena_base  Pointer to the varlena struct array (data section)
 * @param d_area_base     Pointer to the area section (for big string reads)
 * @param d_offsets       Output: int32[n_rows+1] offsets array
 * @param d_chars         Output: character data buffer
 * @param d_temp_storage  CUB temporary storage (nullptr on first call for size query)
 * @param temp_bytes      [in/out] Size of temp storage
 * @param n_rows          Number of rows
 * @param stream          CUDA stream
 */
void decode_varchar(const uint8_t* d_varlena_base,
                    const uint8_t* d_area_base,
                    int32_t* d_offsets,
                    uint8_t* d_chars,
                    void* d_temp_storage,
                    std::size_t& temp_bytes,
                    uint32_t n_rows,
                    rmm::cuda_stream_view stream);

/**
 * @brief Adjust offsets array by adding a constant base value.
 *
 * Used to make multi-block offsets globally monotonic for cuDF.
 * Each block's exclusive prefix-sum starts from 0; this adds the cumulative
 * character count from all previous blocks so cuDF sees strictly increasing offsets.
 *
 * @param d_offsets  Offsets array to adjust in-place
 * @param base       Constant to add to each element
 * @param count      Number of elements (typically n_rows + 1)
 * @param stream     CUDA stream
 */
void adjust_offsets(int32_t* d_offsets,
                    int32_t base,
                    uint32_t count,
                    rmm::cuda_stream_view stream);

/**
 * @brief Compute the total character data size for a varlena column.
 *
 * Used to pre-allocate the chars buffer before calling decode_varchar.
 * Runs a parallel reduction over varlena lengths.
 *
 * @param d_varlena_base  Pointer to the varlena struct array (data section)
 * @param d_area_base     Pointer to the area section (for big string reads)
 * @param n_rows          Number of rows
 * @param stream          CUDA stream
 * @return Total bytes needed for the chars buffer
 */
std::size_t compute_varchar_total_chars(const uint8_t* d_varlena_base,
                                        const uint8_t* d_area_base,
                                        uint32_t n_rows,
                                        rmm::cuda_stream_view stream);

/**
 * @brief Invert a MO null bitmap to cuDF validity bitmask.
 *
 * MO: bit=1 means NULL.  cuDF: bit=1 means VALID.
 * Simply XOR each 32-bit word with 0xFFFFFFFF.
 *
 * @param d_src       Pointer to the MO null bitmap words (in the nsp section,
 *                    after the 24-byte nsp header: count+len+dataSize)
 * @param d_dst       Output cuDF validity bitmask
 * @param n_rows      Number of rows
 * @param stream      CUDA stream
 */
void invert_null_mask(const uint8_t* d_src,
                      uint32_t* d_dst,
                      uint32_t n_rows,
                      rmm::cuda_stream_view stream);

}  // namespace sirius::cuda::tae
