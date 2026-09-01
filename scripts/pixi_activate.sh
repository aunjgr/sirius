#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  exit 0
fi

clang_cpp="$CONDA_PREFIX/bin/clang-cpp"
clang_pp="$CONDA_PREFIX/bin/clang++"

if [[ -x "$clang_cpp" && ! -e "$clang_pp" ]]; then
  ln -s "$clang_cpp" "$clang_pp"
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Standalone build: symlink the Sirius presets as DuckDB's ignored user presets.
# Skip when duckdb/ doesn't exist (building via mo-sirius-sidecar).
if [[ -d "$project_root/duckdb" ]]; then
  cmake_presets_src="$project_root/cmake/CMakePresets.json"
  cmake_presets_dst="$project_root/duckdb/CMakeUserPresets.json"
  if [[ ! -e "$cmake_presets_dst" ]]; then
    ln -s "$cmake_presets_src" "$cmake_presets_dst"
  fi
fi

mkdir -p build
pixi shell-hook -s bash > build/sirius_pixi_env_for_clion.sh
