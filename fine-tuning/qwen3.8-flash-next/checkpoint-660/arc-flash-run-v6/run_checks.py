#!/usr/bin/env python3
"""Integrity checks and an exclusive output lock; imports no ML or GPU libraries."""
from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import socket
from contextlib import contextmanager
from pathlib import Path


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def snapshot(workspace: Path, paths: list[Path]) -> dict:
    workspace = workspace.resolve()
    result = {}
    for path in paths:
        path = path.resolve()
        name = str(path.relative_to(workspace))
        result[name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    return result


def verify_snapshot(workspace: Path, files: dict) -> None:
    workspace = workspace.resolve()
    if not files:
        raise RuntimeError("Empty integrity manifest")
    for name, expected in files.items():
        path = (workspace / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(workspace):
            raise RuntimeError(f"Integrity path leaves the workspace: {name}")
        if not path.is_file() or path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise RuntimeError(f"File changed or is missing: {name}")


def verify_ready(run: Path) -> dict:
    ready = json.loads((run / "READY.json").read_text())
    if ready.get("schema_version") != 2:
        raise RuntimeError("Run the reviewed-package staging check first")
    workspace = Path(ready["workspace"])
    required = {str((run / name).resolve().relative_to(workspace.resolve())) for name in
                ("guarded_train.py", "run_checks.py", "train_arc.sbatch", "arc-flash.yml", "preparation.json")}
    if not required.issubset(ready["files"]):
        raise RuntimeError("Readiness record is missing runtime files")
    verify_snapshot(workspace, ready["files"])
    return ready


@contextmanager
def output_lock(output: Path):
    """The OS releases this advisory lock even if training crashes or is killed."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".training.lock").open("a+") as stream:
        try:
            # POSIX record locking; flock(2) can be disabled or client-local on Lustre.
            fcntl.lockf(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in (errno.EACCES, errno.EAGAIN):
                raise RuntimeError(f"Another job is using this output directory: {output}") from error
            raise RuntimeError("This filesystem did not provide the required exclusive output lock") from error
        try:
            stream.seek(0)
            stream.truncate()
            json.dump({"pid": os.getpid(), "host": socket.gethostname(),
                       "slurm_job_id": os.environ.get("SLURM_JOB_ID")}, stream)
            stream.flush()
            yield
        finally:
            fcntl.lockf(stream.fileno(), fcntl.LOCK_UN)


def checkpoint_payload(path: Path) -> list[Path]:
    required = ("adapter_config.json", "trainer_state.json", "optimizer.pt", "scheduler.pt",
                "rng_state.pth", "arc_cuda_rng_all.pt", "tokenizer.json", "tokenizer_config.json")
    files = [path / name for name in required]
    adapters = sorted(path.glob("adapter_model*.safetensors"))
    if not adapters or any(not file.is_file() or file.stat().st_size == 0 for file in files + adapters):
        raise RuntimeError(f"Incomplete restart checkpoint: {path}")
    for name in ("adapter_config.json", "trainer_state.json"):
        json.loads((path / name).read_text())
    # Include tokenizer/templates and auxiliary framework state in the same seal.
    return sorted(file for file in path.rglob("*") if file.is_file()
                  and file.name not in {"checkpoint-complete.json", "numerical-validation.json"})


def check_checkpoint_validation(path: Path, validation: dict) -> None:
    step = int(path.name.removeprefix("checkpoint-"))
    state = json.loads((path / "trainer_state.json").read_text())
    if not isinstance(state, dict) or not isinstance(validation, dict) or not isinstance(validation.get("adapter_roundtrip"), dict):
        raise RuntimeError("Malformed checkpoint numerical validation")
    if (step <= 0 or state.get("global_step") != step or validation.get("schema_version") != 1
        or validation.get("global_step") != step
        or validation.get("adapter_roundtrip", {}).get("roundtrip_equal") is not True
        or validation.get("optimizer_roundtrip") is not True
        or validation.get("scheduler_roundtrip") is not True or validation.get("cuda_rng_streams") != 3
        or validation.get("tokenizer_roundtrip") is not True):
        raise RuntimeError("Checkpoint lacks matching decoded-tensor validation")


def seal_checkpoint(path: Path, validation: dict | None = None) -> None:
    files = checkpoint_payload(path)
    if validation is None:
        raise RuntimeError("Checkpoint sealing requires decoded-tensor validation, not just file hashes")
    check_checkpoint_validation(path, validation)
    atomic_json(path / "numerical-validation.json", validation)
    files += [path / "numerical-validation.json"]
    atomic_json(path / "checkpoint-complete.json", {"schema_version": 2, "files": snapshot(path, files)})


def checkpoint_files(path: Path) -> None:
    try:
        files = checkpoint_payload(path)
        receipt = json.loads((path / "checkpoint-complete.json").read_text())
        if receipt.get("schema_version") != 2:
            raise RuntimeError("Checkpoint was not certified with the corrected numerical guards")
        check_checkpoint_validation(path, json.loads((path / "numerical-validation.json").read_text()))
        files += [path / "numerical-validation.json"]
        if set(receipt["files"]) != {str(file.relative_to(path)) for file in files}:
            raise RuntimeError("Checkpoint completion record does not match its files")
        verify_snapshot(path, receipt["files"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as error:
        raise RuntimeError(f"Incomplete or changed restart checkpoint: {path}: {error}") from error


def latest_complete_checkpoint(output: Path) -> Path | None:
    candidates = [path for path in output.glob("checkpoint-*")
                  if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()]
    candidates.sort(key=lambda path: int(path.name.split("-")[-1]), reverse=True)
    for path in candidates:
        try:
            checkpoint_files(path)
        except RuntimeError as error:
            print(f"Skipping unusable checkpoint: {error}", flush=True)
            continue
        return path
    if candidates:
        raise RuntimeError("No complete checkpoint remains; refusing to overwrite this run with a fresh fit")
    return None
