"""Small stdlib-only adapter SDK; see README.md for the resume contract."""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
import time
import tempfile
import shutil
import threading
import errno
from contextlib import contextmanager, nullcontext
from pathlib import Path

_thread_locks = {}
_thread_locks_guard = threading.Lock()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)

@contextmanager
def lock(path, blocking=False):
    """POSIX record lock; process death releases it. Unsupported locks fail closed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # POSIX record locks serialize processes, not threads in the local HTTP server.
    with _thread_locks_guard:
        thread_lock = _thread_locks.setdefault(str(path.resolve()), threading.Lock())
    if not thread_lock.acquire(blocking=blocking):
        raise BlockingIOError(errno.EAGAIN, 'Another thread holds this run lock')
    try:
        with path.open("a+") as stream:
            fcntl.lockf(stream.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            try: yield
            finally: fcntl.lockf(stream.fileno(), fcntl.LOCK_UN)
    finally:
        thread_lock.release()

def read_json(path, default=None):
    try: return json.loads(Path(path).read_text())
    except FileNotFoundError: return default

def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""): hasher.update(block)
    return hasher.hexdigest()

def model_cache_path(root, *, weights, loader, precision, layout, adapter=None):
    """Use immutable source/loader fingerprints. Include an adapter only when baked in."""
    identity = dict(weights=weights, loader=loader, precision=precision, layout=layout, adapter=adapter)
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return Path(root) / hashlib.sha256(encoded.encode()).hexdigest(), json.loads(encoded)


def _check_cache(path, fingerprints, full):
    root = Path(path)
    try:
        receipt = read_json(root / "READY.json", {})
        if receipt.get("fingerprints") != fingerprints or not receipt.get("files"):
            return False
        for name, entry in receipt["files"].items():
            file = root / name
            if Path(name).is_absolute() or file.resolve() != root.resolve() / name or not file.is_file():
                return False
            stat = file.stat()
            if stat.st_size != entry["size"] or stat.st_mtime_ns != entry["mtime_ns"]:
                return False
            if full and file_hash(file) != entry["sha256"]:
                return False
    except (FileNotFoundError, KeyError, ValueError, TypeError, AttributeError):
        return False
    return True


def cache_ready(path, fingerprints):
    return _check_cache(path, fingerprints, False)


def verify_cache(path, fingerprints):
    return _check_cache(path, fingerprints, True)


def _recover_stage(path, stage):
    if stage.is_symlink() or read_json(stage / ".owner.json") != str(path.absolute()):
        raise ValueError("Cache staging path is owned by another writer")
    retired = stage / "retired"
    if retired.exists() and not path.exists():
        retired.rename(path)
    shutil.rmtree(stage)


@contextmanager
def _staging_directory(path, stage):
    # Establish ownership before any large payload is created. A killed writer
    # leaves one known location, recoverable under the sibling lock on the next hit or miss.
    with tempfile.TemporaryDirectory(dir=path.parent) as temp:
        marker = Path(temp) / "stage"
        marker.mkdir()
        atomic_json(marker / ".owner.json", str(path.absolute()))
        marker.rename(stage)
    try:
        yield stage
    finally:
        _recover_stage(path, stage)


def prepare_cache(path, fingerprints, prepare, *, verify=False):
    """Publish a verified directory. Return True on reuse. Set verify on a suspect hit."""
    path = Path(path)
    with lock(path.parent / ("." + path.name + ".prepare.lock"), blocking=True):
        staging = path.parent / ("." + path.name + ".staging")
        if staging.exists() or staging.is_symlink():
            _recover_stage(path, staging)
        legacy_lock = path / ".prepare.lock"
        legacy = legacy_lock.is_file() and not legacy_lock.is_symlink()
        with lock(legacy_lock, blocking=True) if legacy else nullcontext():
            if (verify_cache if verify else cache_ready)(path, fingerprints):
                return True
            prior = read_json(path / "READY.json")
            if prior and prior.get("fingerprints") != fingerprints:
                raise ValueError("Use a distinct cache directory for changed fingerprints")
            with _staging_directory(path, staging) as temp:
                stage = Path(temp) / "new"
                stage.mkdir()
                entries = {}
                for name in prepare(stage):
                    name = str(name)
                    file = stage / name
                    if (Path(name).is_absolute() or file.resolve() != stage.resolve() / name
                            or not file.is_file() or name == "READY.json"):
                        raise ValueError("Invalid cache artifact: " + name)
                    stat = file.stat()
                    entries[name] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=file_hash(file))
                if any(p.is_symlink() or (p.is_file() and str(p.relative_to(stage)) not in entries)
                       for p in stage.rglob("*")):
                    raise ValueError("Cache builder produced an undeclared artifact")
                if not entries:
                    raise ValueError("Preparation must validate at least one artifact")
                atomic_json(stage / "READY.json", dict(fingerprints=fingerprints, files=entries))
                if not verify_cache(stage, fingerprints):
                    raise ValueError("Prepared cache failed verification")
                if path.exists():
                    # Only retire a dedicated cache. Never remove unrelated user files.
                    known = set((prior or {}).get("files", {})) | {"READY.json", ".prepare.lock"}
                    if prior is None and legacy:
                        # A pre-staging builder may have died with these declared payloads half written.
                        known.update(entries)
                    if path.is_symlink() or any(str(f.relative_to(path)) not in known
                            for f in path.rglob("*") if not f.is_dir() or f.is_symlink()):
                        raise ValueError("Destination contains unowned files")
                    path.rename(Path(temp) / "retired")
                try:
                    stage.rename(path)
                except OSError:
                    retired = Path(temp) / "retired"
                    if retired.exists() and not path.exists():
                        retired.rename(path)
                    raise
            return False


class Adapter:
    def __init__(self):
        self.root = Path(os.environ["WM_RUN_DIR"])
        self.worker = int(os.environ["WM_WORKER_INDEX"])
        self.workers = int(os.environ["WM_WORKER_COUNT"])
        self.directory = self.root / "workers" / str(self.worker)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = os.environ["WM_IDENTITY"]
        self.attempt = os.environ["WM_ATTEMPT"]
        self.deadline = float(os.environ["WM_DEADLINE"])
        self._first_result_seconds = None

    def assigned(self, index):
        return index % self.workers == self.worker

    def stopping(self):
        return time.time() >= self.deadline or (self.root / "STOP").exists() or (self.directory / "DRAIN").exists()

    def checkpoint(self, value):
        atomic_json(self.directory / "checkpoint.json", {"identity": self.identity, "value": value})

    def load_checkpoint(self, default=None):
        checkpoint = read_json(self.directory / "checkpoint.json")
        if checkpoint is None: return default
        if checkpoint.get("identity") != self.identity: raise ValueError("Checkpoint identity changed")
        return checkpoint["value"]

    def metric(self, **values):
        now = time.time()
        start = os.getenv("WM_ALLOCATION_START")
        if start and "completed" in values:
            elapsed = now - float(start)
            completed = values["completed"] - int(os.getenv("WM_PRIOR_COMPLETED","0"))
            if self._first_result_seconds is None and completed > 0:
                self._first_result_seconds = elapsed
            allocated = os.getenv("SLURM_GPUS_ON_NODE","")
            if allocated.isdecimal() and int(allocated) > 0 and elapsed > 0 and completed >= 0:
                hours = elapsed * int(allocated) / 3600
                values["allocated_gpu_hours_so_far"] = hours
                values["new_items_per_gpu_hour_so_far"] = completed / hours
        if self._first_result_seconds is not None:
            values["first_saved_result_seconds"] = self._first_result_seconds
        row = {"time": now, "attempt": self.attempt, "worker": self.worker, **values}
        with (self.directory / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()

    @contextmanager
    def phase(self, name):
        if name not in ("preparation", "model_load", "restore", "execution"):
            raise ValueError("Unknown timing phase")
        started = time.monotonic()
        try: yield
        finally: self.metric(**{name + "_seconds": time.monotonic() - started})

    def gpu_metrics(self):
        """Optional: call inside the adapter's ML environment, mapping devices by UUID."""
        import subprocess
        try:
            import torch
            uuids = [str(torch.cuda.get_device_properties(i).uuid) for i in range(torch.cuda.device_count())]
            if not uuids or any(not u.startswith("GPU-") for u in uuids): return []
            result = subprocess.run(["nvidia-smi", "--id=" + ",".join(uuids), "--query-gpu=uuid,utilization.gpu,memory.used,memory.total,power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
            if result.returncode: return []
            rows = []
            for line in result.stdout.splitlines():
                gpu, busy, used, total, power = (x.strip() for x in line.split(","))
                if gpu not in uuids: continue
                rows.append({"uuid": gpu, "utilization": float(busy), "memory_used_mib": float(used), "memory_total_mib": float(total), "power_watts": float(power)})
            self.metric(gpus=rows)
            return rows
        except (ImportError, AttributeError, OSError, ValueError, subprocess.TimeoutExpired):
            return []

    def finish(self, state, progress, completed=0, checkpoint_files=()):
        if state not in ("complete", "partial_checkpointed", "failed"): raise ValueError("Invalid adapter state")
        paths = list(checkpoint_files)
        if not paths and (self.directory / "checkpoint.json").exists():
            paths = [str((self.directory / "checkpoint.json").relative_to(self.root))]
        files = {}
        for name in paths:
            path = (self.root / name).resolve()
            if Path(name).is_absolute() or not path.is_relative_to(self.root.resolve()):
                raise ValueError("Checkpoint leaves run directory")
            files[name] = {"size": path.stat().st_size, "sha256": file_hash(path)}
        atomic_json(self.directory / "result.json", {"identity": self.identity, "attempt": self.attempt, "state": state, "progress": progress, "completed": completed, "checkpoint_files": files, "finished_at": time.time()})
