"""Explicit CCE launch context, verified against completed GPU diagnostic 7849447."""
from __future__ import annotations

import contextlib
import functools
import importlib.util
import math
from pathlib import Path

if __package__:
    from .run_checks import digest
else:
    from run_checks import digest


def tensor_device_scope(torch, tensor):
    return torch.cuda.device(tensor.device) if tensor.device.type == "cuda" else contextlib.nullcontext()


def owner_scoped_backward(torch, original, observer=None):
    """CCE's pinned autograd implementation passes the embedding as keyword e."""
    @functools.wraps(original)
    def call(*args, **kwargs):
        if "e" not in kwargs:
            raise RuntimeError("CCE backward interface changed: expected its embedding operand as keyword e")
        tensor = kwargs["e"]
        with tensor_device_scope(torch, tensor):
            if observer:
                observer(tensor)
            return original(*args, **kwargs)
    return call


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


def validate_device_evidence(report):
    if (report.get("phase") != "debug_completed" or report.get("slurm_job_id") != "7849447"
        or report.get("optimizer_updates") != 0 or report.get("training_readiness_established") is not False
        or [case.get("task_id") for case in report.get("cases", [])] != ["423a55dc", "d492a647"]):
        raise RuntimeError("Missing completed, zero-update GPU loss diagnostic evidence")
    for case in report["cases"]:
        calls = case.get("calls", [])
        if ([item.get("mode") for item in calls] != ["unchanged", "owner_scoped", "unchanged", "owner_scoped"]
            or any(item.get("entry_current_device") != 0 for item in calls)
            or case["e"]["device"] != "cuda:2" or case["c"]["device"] != "cuda:2"
            or not all(item.get("launch_scope") == "cuda:2" and item.get("matches_reference") is True
                       for item in calls if item["mode"] == "owner_scoped")
            or not any(item.get("matches_reference") is False for item in calls if item["mode"] == "unchanged")):
            raise RuntimeError("GPU diagnostic does not demonstrate the current-device mismatch and scoped correction")
    gradient = report["cases"][1].get("gradient", {})
    if (not all(gradient.get(key) is True for key in ("candidate_finite", "candidate_nonzero", "reference_finite"))
        or not math.isfinite(float(gradient.get("relative_l2_error", math.inf)))
        or gradient["relative_l2_error"] > 0.01 or gradient.get("cosine_similarity", 0) < 0.999):
        raise RuntimeError("GPU diagnostic lacks the observed finite, matching isolated gradient")
