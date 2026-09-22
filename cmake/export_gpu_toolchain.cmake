# Capture the provider used by CMake, rather than the environment of a later
# exporter invocation. This opt-in profile does not affect ordinary SDK builds.
set(_sirius_export_gpu_default OFF)
if("$ENV{PIXI_ENVIRONMENT_NAME}" STREQUAL "mo")
  set(_sirius_export_gpu_default ON)
endif()
option(SIRIUS_EXPORT_GPU_TOOLCHAIN
       "Export the MatrixOne-compatible Pixi GPU toolchain"
       ${_sirius_export_gpu_default})
if(SIRIUS_EXPORT_GPU_TOOLCHAIN)
  foreach(_sirius_gpu_path "$ENV{CONDA_PREFIX}" "${CMAKE_CURRENT_SOURCE_DIR}")
    if(NOT "${_sirius_gpu_path}" MATCHES "^/[A-Za-z0-9_./+@-]+$")
      message(
        FATAL_ERROR
          "GPU provider/source paths cannot contain whitespace or metacharacters"
      )
    endif()
  endforeach()
  if(NOT "$ENV{PIXI_ENVIRONMENT_NAME}" STREQUAL "mo"
     OR NOT EXISTS "$ENV{CONDA_PREFIX}/conda-meta")
    message(FATAL_ERROR "GPU toolchain export requires pixi run --frozen -e mo")
  endif()
  # cuVS loads C++ targets implicitly, but its C API is an explicit component.
  find_package(cuvs REQUIRED CONFIG COMPONENTS c_api)
  if(NOT TARGET cuvs::c_api)
    message(FATAL_ERROR "The MO SDK requires the shared cuVS C API target")
  endif()
  set(_sirius_gpu_config "${CMAKE_CURRENT_BINARY_DIR}/gpu-toolchain-build.json")
  file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/pixi.lock" _sirius_pixi_lock_hash)
  file(
    GENERATE
    OUTPUT "${_sirius_gpu_config}"
    CONTENT
      "{\n\
  \"prefix\": \"$ENV{CONDA_PREFIX}\",\n\
  \"environment\": \"mo\",\n\
  \"lockfile\": \"${CMAKE_CURRENT_SOURCE_DIR}/pixi.lock\",\n\
  \"lockfile_sha256\": \"${_sirius_pixi_lock_hash}\",\n\
  \"cc\": \"${CMAKE_C_COMPILER}\",\n\
  \"cxx\": \"${CMAKE_CXX_COMPILER}\",\n\
  \"nvcc\": \"${CMAKE_CUDA_COMPILER}\",\n\
  \"cuda_version\": \"${CUDAToolkit_VERSION}\",\n\
  \"cuda_include_dirs\": \"${CUDAToolkit_INCLUDE_DIRS}\",\n\
  \"cudart\": \"$<TARGET_FILE:CUDA::cudart>\",\n\
  \"libcuvs\": \"$<TARGET_FILE:cuvs::cuvs>\",\n\
  \"libcuvs_c\": \"$<TARGET_FILE:cuvs::c_api>\",\n\
  \"libcudf\": \"$<TARGET_FILE:cudf::cudf>\"\n\
}\n")
  set(_sirius_gpu_export_args --gpu-toolchain-config "${_sirius_gpu_config}")
  set_property(
    TARGET sirius_c_smoke
    APPEND
    PROPERTY LINK_DEPENDS "${_sirius_gpu_config}"
             "${CMAKE_CURRENT_SOURCE_DIR}/pixi.lock"
             "${CMAKE_CURRENT_SOURCE_DIR}/scripts/export_gpu_toolchain.py")
endif()
