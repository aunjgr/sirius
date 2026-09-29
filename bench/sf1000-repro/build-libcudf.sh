#!/usr/bin/env bash
# Clone and build the patched libcudf that this benchmark LD_PRELOADs.
#
# Three cuDF patches live on felipeblazing/cudf @ perf/sirius-sf1000-repro, on top of
# the v26.06.01 release tag (77ced62) that the pixi env ships:
#   4a345cc  strings::like  -- skip non-candidate bytes when backtracking   (q13 -36.5%)
#   9af88b0  cuda_memcpy    -- prefer copy bandwidth over compute overlap   (q9  -5.8%)
#   7375a46  groupby        -- one shared-mem aggregation slot per warp lane (q1 ~-5%)
#
# Run this from inside the Sirius pixi environment:  pixi run bash build-libcudf.sh
set -euo pipefail

CUDF_SRC="${CUDF_SRC:-$HOME/cudf-src}"
SHIM="${SHIM:-$HOME/cudf-shim}"
BRANCH=perf/sirius-sf1000-repro

[ -n "${PIXI_PROJECT_ROOT:-}" ] && [ -n "${PIXI_ENVIRONMENT_NAME:-}" ] &&
  [ -d "${CONDA_PREFIX:-}" ] &&
  [ "$(realpath "$CONDA_PREFIX")" = "$(realpath "$PIXI_PROJECT_ROOT/.pixi/envs/$PIXI_ENVIRONMENT_NAME")" ] || {
  echo "ERROR: run inside the Sirius pixi env (pixi run --frozen bash $0)" >&2
  exit 1
}

# jitify2_preprocess needs the development header, not just libnvJitLink's
# runtime. Both must come from the same Pixi environment as cuDF and NVCC.
NVJITLINK_HEADER=
for include_dir in "$CONDA_PREFIX/include" "$CONDA_PREFIX/targets/$(uname -m)-linux/include"; do
  if [ -f "$include_dir/nvJitLink.h" ]; then
    NVJITLINK_HEADER=$include_dir/nvJitLink.h
    break
  fi
done
[ -n "$NVJITLINK_HEADER" ] || {
  echo "ERROR: Pixi libnvjitlink-dev is missing nvJitLink.h; run pixi install --frozen" >&2
  exit 1
}
[ -f "$CONDA_PREFIX/lib/libnvJitLink.so.13" ] || {
  echo "ERROR: Pixi libnvjitlink is missing libnvJitLink.so.13; run pixi install --frozen" >&2
  exit 1
}
if [ "${SIRIUS_CUDA_PREFLIGHT_ONLY:-0}" = 1 ]; then
  echo "Pixi CUDA development inputs are present in $CONDA_PREFIX"
  exit 0
fi

if [ ! -d "$CUDF_SRC/.git" ]; then
  echo "==> cloning felipeblazing/cudf @ $BRANCH"
  git clone --filter=blob:none --branch "$BRANCH" \
      https://github.com/felipeblazing/cudf.git "$CUDF_SRC"
else
  echo "==> updating existing clone"
  git -C "$CUDF_SRC" fetch https://github.com/felipeblazing/cudf.git "$BRANCH"
  git -C "$CUDF_SRC" checkout -q FETCH_HEAD
fi

# Keep the jitify2 shim, but link only inputs from the activated Pixi prefix.
echo "==> nvJitLink shim at $SHIM"
mkdir -p "$SHIM/include" "$SHIM/lib"
ln -sf "$NVJITLINK_HEADER" "$SHIM/include/nvJitLink.h"
ln -sf "$CONDA_PREFIX/lib/libnvJitLink.so.13" "$SHIM/lib/libnvJitLink.so"

# CRITICAL: append to $CXXFLAGS / $LDFLAGS, never replace them. Passing a bare
# -DCMAKE_CXX_FLAGS= clobbers conda's flags, dropping -isystem $CONDA_PREFIX/include; CMake then
# strips rmm's include dir as "redundant" (it is still cached in CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)
# and 17 CXX files fail with "rmm/cuda_stream_view.hpp: No such file".
echo "==> configure"
cmake -GNinja -S "$CUDF_SRC/cpp" -B "$CUDF_SRC/cpp/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF \
  -DCUDF_BUILD_TESTUTIL=OFF -DCUDF_BUILD_STREAMS_TEST_UTIL=OFF \
  -DUSE_NVTX=ON \
  -DCMAKE_CUDA_ARCHITECTURES=NATIVE \
  -DCMAKE_CXX_FLAGS="$CXXFLAGS -isystem $SHIM/include" \
  -DCMAKE_CUDA_FLAGS="-isystem $SHIM/include" \
  -DCMAKE_EXE_LINKER_FLAGS="$LDFLAGS -L$SHIM/lib" \
  -DCMAKE_SHARED_LINKER_FLAGS="$LDFLAGS -L$SHIM/lib"

echo "==> build (557 targets, ~25 min cold; ~25 s incremental)"
nice -n 19 ninja -C "$CUDF_SRC/cpp/build" -j "$(( $(nproc) > 20 ? 20 : $(nproc) ))" cudf

echo
echo "built: $CUDF_SRC/cpp/build/libcudf.so"
echo "export CUDF_SO=$CUDF_SRC/cpp/build/libcudf.so   # run.sh LD_PRELOADs this"
