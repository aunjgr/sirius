# Copyright 2026 Sirius Contributors.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import yaml

spec = importlib.util.spec_from_file_location(
    "export_gpu_toolchain",
    Path(__file__).resolve().parents[1] / "export_gpu_toolchain.py",
)
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)


class ToolchainTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.prefix = self.root / "pixi"
        self.config = {
            "prefix": str(self.prefix),
            "environment": "mo",
            "lockfile": str(self.root / "pixi.lock"),
            "cuda_version": "13.3.73",
            "cuda_include_dirs": str(self.prefix / "targets/x86_64-linux/include"),
        }
        for name in ("cc", "cxx", "nvcc"):
            self.config[name] = str(self.file("bin/" + name))
            Path(self.config[name]).chmod(0o755)
        for name in ("libcuvs", "libcuvs_c", "libcudf"):
            self.config[name] = str(self.file("lib/" + name + ".so"))
        self.config["cudart"] = str(self.file("lib/libcudart.so"))
        self.file("lib/librmm.so")
        self.file("targets/x86_64-linux/lib/stubs/libcuda.so")
        self.file("targets/x86_64-linux/include/cuda.h", "#define CUDA_VERSION 13030\n")
        self.file("include/cuvs/core/c_api.h")
        self.file("include/cuvs/core/c_config.h")
        self.file("include/rmm/cuda_stream_view.hpp")
        entries = []
        for name, version in (
            ("cuda-version", "13.3"),
            ("libcuvs", "26.08.01"),
            ("libcudf", "26.08.01"),
            ("librmm", "26.08.00"),
            ("gcc_linux-64", "14.4.0"),
            ("gxx_linux-64", "14.4.0"),
        ):
            record = {
                "name": name,
                "version": version,
                "build": "fixture",
                "url": "https://packages.invalid/" + name + ".conda",
                "sha256": "a" * 64,
            }
            self.file("conda-meta/" + name + ".json", json.dumps(record))
            entries.append({"conda": record["url"], "sha256": record["sha256"]})
        Path(self.config["lockfile"]).write_text(
            yaml.safe_dump(
                {
                    "environments": {"mo": {"packages": {"linux-64-cuda13": entries}}},
                    "packages": entries,
                }
            )
        )
        self.config["lockfile_sha256"] = gpu.sha256(self.config["lockfile"])
        self.mock = patch.object(
            gpu.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(
                [], 0, "release 13.3, fixture", ""
            ),
        )
        self.mock.start()
        self.addCleanup(self.mock.stop)

    def file(self, name, data="fixture"):
        path = self.prefix / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
        return path

    def export(self, **options):
        config = self.root / "build.json"
        config.write_text(json.dumps(self.config))
        return gpu.export_toolchain(
            config, options.get("cc", self.config["cc"]), self.config["cxx"]
        )

    def test_shared_contract_contains_c_api_and_separates_driver_stub(self):
        exported = self.export()
        self.assertEqual(exported["schema_version"], 1)
        self.assertEqual(exported["rapids"]["versions"]["libcuvs_c"], "26.08.01")
        self.assertIn(self.config["libcuvs_c"], exported["runtime_roots"])
        self.assertIn(self.config["libcuvs_c"], exported["artifact_sha256"])
        self.assertFalse(any("stubs" in path for path in exported["runtime_roots"]))
        self.assertTrue(exported["cuda"]["stub_library_dirs"])
        self.assertIn(self.config["cc"], exported["artifact_sha256"])

    def test_wrong_compiler_or_external_provider_fails(self):
        with self.assertRaisesRegex(RuntimeError, "verified C consumer"):
            self.export(cc="/usr/bin/cc")
        self.config["libcuvs_c"] = "/usr/lib/libcuvs_c.so"
        with self.assertRaisesRegex(RuntimeError, "escapes"):
            self.export()

    def test_changed_lock_and_missing_c_api_fail(self):
        Path(self.config["libcuvs_c"]).unlink()
        with self.assertRaisesRegex(RuntimeError, "missing GPU"):
            self.export()
        Path(self.config["lockfile"]).write_text("changed")
        with self.assertRaisesRegex(RuntimeError, "lock changed"):
            self.export()

    def test_incompatible_headers_and_toolkit_fail(self):
        self.file("targets/x86_64-linux/include/cuda.h", "#define CUDA_VERSION 13020\n")
        with self.assertRaisesRegex(RuntimeError, "CUDA header"):
            self.export()
        self.config["cuda_version"] = "13.2.86"
        with self.assertRaisesRegex(RuntimeError, "incompatible CUDA"):
            self.export()

    def test_stale_or_missing_installed_package_fails(self):
        record = self.prefix / "conda-meta/libcuvs.json"
        data = json.loads(record.read_text())
        data["sha256"] = "b" * 64
        record.write_text(json.dumps(data))
        with self.assertRaisesRegex(RuntimeError, "disagrees with Pixi lock"):
            self.export()
        record.unlink()
        with self.assertRaisesRegex(RuntimeError, "package set disagrees"):
            self.export()

    def test_stub_symlink_and_nonexecutable_compiler_fail(self):
        root = Path(self.config["libcuvs_c"])
        root.unlink()
        root.symlink_to(self.prefix / "targets/x86_64-linux/lib/stubs/libcuda.so")
        with self.assertRaisesRegex(RuntimeError, "driver or stub"):
            self.export()
        Path(self.config["cc"]).chmod(0o644)
        with self.assertRaisesRegex(RuntimeError, "not executable"):
            self.export()


if __name__ == "__main__":
    unittest.main()
