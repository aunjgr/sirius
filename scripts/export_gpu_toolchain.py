#!/usr/bin/env python3
# Copyright 2026 Sirius Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Export a build-bound, optional Pixi GPU provider for MatrixOne.

The schema is shared with MO's standalone exporter. Absolute paths describe a
build installation; runtime packaging must discover the combined ELF closure.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import yaml


def sha256(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def export_toolchain(configuration, cc, cxx):
    config = json.loads(Path(configuration).read_text())
    prefix = Path(config["prefix"])
    if not prefix.is_absolute() or not (prefix / "conda-meta").is_dir():
        raise RuntimeError("GPU provider must be an absolute Pixi prefix")
    prefix = prefix.resolve()
    artifacts = {}

    def inside(value, directory=False):
        path = Path(value)
        if not re.fullmatch(r"/[A-Za-z0-9_./+@-]+", str(path)):
            raise RuntimeError(
                "GPU artifact path contains whitespace or metacharacters"
            )
        if not path.is_absolute() or not path.resolve().is_relative_to(prefix):
            raise RuntimeError(
                "GPU artifact escapes the configured provider: " + str(path)
            )
        path = path.resolve()
        if not (path.is_dir() if directory else path.is_file()):
            raise RuntimeError("missing GPU provider artifact: " + str(path))
        if not directory:
            artifacts[str(path)] = sha256(path)
        return path

    lockfile = Path(config["lockfile"])
    if (
        config["environment"] != "mo"
        or not lockfile.is_absolute()
        or sha256(lockfile) != config["lockfile_sha256"]
    ):
        raise RuntimeError("Pixi lock changed since CMake configuration; reconfigure")
    artifacts[str(lockfile.resolve())] = sha256(lockfile)
    lock = yaml.safe_load(lockfile.read_text())
    selected = {
        entry["conda"]
        for entry in lock["environments"]["mo"]["packages"]["linux-64-cuda13"]
        if "conda" in entry
    }
    locked_hashes = {
        entry["conda"]: entry["sha256"]
        for entry in lock["packages"]
        if entry.get("conda") in selected
    }
    packages = []
    records = {}
    for metadata in sorted((prefix / "conda-meta").glob("*.json")):
        record = json.loads(metadata.read_text())
        if record["name"] in records:
            raise RuntimeError("duplicate installed package: " + record["name"])
        records[record["name"]] = record
        archive_hash = record.get("sha256", "")
        if not re.fullmatch(r"[0-9a-f]{64}", archive_hash):
            raise RuntimeError("missing package SHA256: " + record["name"])
        packages.append(
            {key: record[key] for key in ("name", "version", "build", "url", "sha256")}
        )
        if locked_hashes.get(record["url"]) != archive_hash:
            raise RuntimeError(
                "installed package disagrees with Pixi lock: " + record["name"]
            )
        inside(metadata)
    if {record["url"] for record in records.values()} != selected:
        raise RuntimeError("installed package set disagrees with Pixi lock")

    def version(name, expected):
        record = records.get(name)
        if record is None or not (
            record["version"].startswith(expected)
            if expected.endswith(".")
            else record["version"] == expected
        ):
            raise RuntimeError("incompatible or missing GPU package: " + name)
        return record["version"]

    cuda_version = version("cuda-version", "13.3")
    if not config["cuda_version"].startswith("13.3."):
        raise RuntimeError("CMake selected an incompatible CUDA toolkit")
    versions = {
        "libcuvs": version("libcuvs", "26.08.01"),
        "libcudf": version("libcudf", "26.08.01"),
        "librmm": version("librmm", "26.08."),
    }
    # Both cuVS shared libraries belong to the libcuvs package.
    versions["libcuvs_c"] = versions["libcuvs"]
    version("gcc_linux-64", "14.")
    version("gxx_linux-64", "14.")
    compilers = {}
    for name in ("cc", "cxx", "nvcc"):
        path = inside(config[name])
        if not os.access(path, os.X_OK):
            raise RuntimeError("GPU compiler is not executable: " + str(path))
        if (
            name in ("cc", "cxx")
            and path != Path({"cc": cc, "cxx": cxx}[name]).resolve()
        ):
            raise RuntimeError("GPU compiler disagrees with verified C consumer")
        output = subprocess.run(
            [str(path), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
        if name == "nvcc" and "release 13.3," not in output:
            raise RuntimeError("NVCC disagrees with configured CUDA version")
        compilers[name] = {"path": str(path), "version": output}

    cuda_includes = [
        inside(path, True) for path in config["cuda_include_dirs"].split(";")
    ]
    target = prefix / "targets/x86_64-linux"
    cuda_header = inside(target / "include/cuda.h")
    if cuda_header.parent not in cuda_includes:
        raise RuntimeError("CMake did not use the exported CUDA header root")
    if not re.search(
        r"^#define\s+CUDA_VERSION\s+13030\b", cuda_header.read_text(), re.M
    ):
        raise RuntimeError("CUDA header disagrees with CUDA 13.3 provider")
    inside(prefix / "include/cuvs/core/c_api.h")
    inside(prefix / "include/cuvs/core/c_config.h")
    inside(prefix / "include/rmm/cuda_stream_view.hpp")
    cudart = inside(config["cudart"])
    roots = [inside(config[name]) for name in ("libcuvs", "libcuvs_c", "libcudf")]
    roots.append(inside(prefix / "lib/librmm.so"))
    for root in roots + [cudart]:
        if "stubs" in root.parts or root.name.startswith(("libcuda.so", "libnvidia-")):
            raise RuntimeError("driver or stub cannot be a runtime root")
    library_dirs = list(
        dict.fromkeys([str(cudart.parent), str(inside(target / "lib", True))])
    )
    stub_directory = inside(target / "lib/stubs", True)
    inside(stub_directory / "libcuda.so")
    return {
        "schema_version": 1,
        "provider": "pixi",
        "platform": "linux-64",
        "prefix": str(prefix),
        "compilers": compilers,
        "cuda": {
            "version": cuda_version,
            "include_dirs": [str(path) for path in cuda_includes],
            "library_dirs": library_dirs,
            "stub_library_dirs": [str(stub_directory)],
        },
        "rapids": {
            "include_dirs": [str(inside(prefix / "include", True))],
            "library_dirs": [str(inside(prefix / "lib", True))],
            "versions": versions,
        },
        "pixi": {
            "environment": "mo",
            "lockfile": str(lockfile.resolve()),
            "lockfile_sha256": config["lockfile_sha256"],
        },
        "packages": packages,
        "runtime_roots": [str(path) for path in roots],
        "artifact_sha256": artifacts,
    }
