# Optional shared GPU toolchain

Refs [MatrixOne #28966](https://github.com/matrixorigin/matrixone/issues/28966).

The `mo` Pixi environment provides CUDA 13.3, GCC 14, cuVS/cuDF 26.08.01,
compatible RMM, and packaging tools for the combined MO/Sirius binary. The
default CUDA 13.2 and `cuda12` environments retain their existing selection.
The NVIDIA driver and GPU devices remain host dependencies.

```sh
pixi install --frozen -e mo
pixi run --frozen -e mo mo-build-embedding-sdk
```

The profile-specific configure task disables `sccache` only for CUDA. The
ordinary presets retain their existing launchers; the `mo` SDK avoids the NVCC
temporary-PTX failure observed when CUDA 13.3 is launched through sccache 0.15.

The `mo` profile enables `SIRIUS_EXPORT_GPU_TOOLCHAIN`. CMake captures the actual
compiler paths, CUDA include directories and imported CUDA/RAPIDS libraries.
The SDK exporter verifies the captured CUDA version, installed package set,
package archive hashes, and lockfile hash before writing `toolchain.json`
beside `link.json`. Other profiles continue exporting the original SDK.

MO consumes the absolute `GPU_TOOLCHAIN_MANIFEST=/path/to/toolchain.json`.
Provider and source paths must not contain whitespace or shell/Make
metacharacters, matching MO's build contract.
Schema version 1 contains `provider`, `platform`, `prefix`, `compilers` (each
with `path` and `version`), `cuda` (version and separate include, library and
linker-stub directories), `rapids` (directories and package versions), `pixi`
(environment, lockfile and hash), installed `packages`, `runtime_roots`, and
`artifact_sha256`. The SDK's ABI and link schema remain version 1; its added
`gpu_toolchain_manifest` field names the manifest and the SDK fingerprint
includes both that file and its inputs.

Both `libcuvs.so` and `libcuvs_c.so` are explicit runtime roots; their package
identity is the same `libcuvs` package. cuDF and RMM are also roots. Driver stubs
are recorded only as link inputs and must never be shipped as runtime driver
libraries. Packaging must discover the full dependency closure of the combined
MO binary and `libmo`, preserve the single provider, and resolve deployment
libraries relative to the packaged binary.

The manifest is build-tree metadata, not a relocatable SDK. Reconfigure and
re-export after changing the Pixi lock or replacing the environment. A manifest
does not establish GPU execution: the combined cuVS/Sirius test with two streams
still requires a working host NVIDIA driver.
