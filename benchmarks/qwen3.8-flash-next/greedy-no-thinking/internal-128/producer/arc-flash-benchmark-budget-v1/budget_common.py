"""Verify the launcher and the unchanged, previously staged benchmark v2."""
import hashlib
import importlib
import json
import sys
from pathlib import Path

NAME = "arc-flash-benchmark-budget-v1"
BENCHMARK = "arc-flash-dev-benchmark-v2"
WORKSPACE = Path(__file__).resolve().parent.parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_files(root, files):
    for name, sha in files.items():
        if Path(name).name != name or digest(root / name) != sha:
            raise RuntimeError(f"Reviewed source changed: {root / name}")


def verify(run):
    manifest = json.loads((run / "package-manifest.json").read_text())
    if manifest["name"] != NAME or run.name != NAME:
        raise RuntimeError("Unexpected allocation-budget package")
    check_files(run, manifest["files_sha256"])
    benchmark = run.parent / BENCHMARK
    expected = json.loads((run / "benchmark-package-manifest.json").read_text())
    if json.loads((benchmark / "package-manifest.json").read_text()) != expected:
        raise RuntimeError("Benchmark manifest differs from the completed debug run")
    check_files(benchmark, expected["files_sha256"])
    sys.path.insert(0, str(benchmark))
    common = importlib.import_module("common")
    if Path(common.__file__).resolve().parent != benchmark.resolve():
        raise RuntimeError("A different benchmark helper was already imported")
    common.verify_ready(benchmark)
    common.verify_training_package(run.parent, benchmark)
    common.verify_loader_sources(benchmark)
    return benchmark, common
