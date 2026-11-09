"""Verify loader and kernel sources before generation."""
import importlib.util
from pathlib import Path
from run_checks import digest


def verify_kernel_sources(review):
    actual = {}
    for package, files in review.items():
        root = Path(importlib.util.find_spec(package).origin).parent
        for relative, record in files.items():
            path = root / relative
            actual[str(path)] = digest(path)
            if actual[str(path)] != record["sha256"]:
                raise RuntimeError(f"Installed kernel source differs from the GPU-tested pin: {path}")
    return actual
