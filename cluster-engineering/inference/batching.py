"""Length-banded admission and a durable per-run inference journal (POSIX)."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import time
import uuid

WIDTHS = (1, 2, 4, 8, 12, 16)
TERMINAL = {"eos", "token_limit"}


def checksum(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_record(path, record):
    value = {**record, "checksum": checksum(record)}
    with NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(value, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temp, path)
            fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            temp.unlink(missing_ok=True)


def read_record(path, run_id):
    record = json.loads(path.read_text())
    expected = record.pop("checksum")
    if checksum(record) != expected or record["run_id"] != run_id:
        raise ValueError(f"Unexpected journal data in {path.name}")
    return record


def choose_group(tasks, states, attempts, *, context, bands):
    """Tasks contain task_id, prompt_tokens (count) and budget (allowed new tokens)."""
    if tuple(sorted(set(bands))) != tuple(bands) or not bands or bands[-1] != context:
        raise ValueError("Band endpoints must increase strictly and end at the context limit")
    if len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("Duplicate task IDs")
    successful, ceilings, retry = {}, {}, {}
    for record in attempts:
        band, size = record["band_end"], record["actual_size"]
        if record["status"] in {"complete", "segment_complete"}:
            successful[band] = max(successful.get(band, 0), size)
        elif record["status"] == "oom":
            lower = max((n for n in WIDTHS if n < size), default=0)
            ceilings[band] = min(ceilings.get(band, 16), max(1, lower))
            for key in record["task_ids"]:
                retry[key] = min(retry.get(key, 16), lower)

    pending = []
    for task in tasks:
        key = task["task_id"]
        state = states.get(key, {})
        if state.get("reason") in TERMINAL or retry.get(key) == 0:
            continue
        prefix = len(state.get("tokens", []))
        width = task["prompt_tokens"] + prefix
        left = task["budget"] - prefix
        if task["prompt_tokens"] <= 0 or left <= 0 or width + left > context:
            raise ValueError(f"Invalid budget or saved prefix for {key}")
        band = next(b for b in bands if width <= b)
        pending.append(((key not in retry, prefix == 0, width, key), band, task, width))
    if not pending:
        return None
    priority, band, seed, seed_width = min(pending, key=lambda r: r[0])
    safe = successful.get(band, 0)
    target = 8 if safe < 8 else 12 if safe < 12 else 16
    ceiling = min(target, ceilings.get(band, 16), retry.get(seed["task_id"], 16))
    peers = sorted((p for p in pending if p[1] == band),
                   key=lambda p: (p[0] != priority, p[0][:2], abs(p[3]-seed_width), p[0]))
    chosen = []
    low = high = seed_width
    for _, _, task, width in peers:
        candidate_cap = min(ceiling, retry.get(task["task_id"], 16))
        new_low, new_high = min(low, width), max(high, width)
        if len(chosen) + 1 > candidate_cap or new_high * 100 > new_low * 135:
            continue
        chosen.append((task, width))
        low, high, ceiling = new_low, new_high, candidate_cap
        if len(chosen) == ceiling:
            break
    if 8 <= len(chosen) < target:
        chosen = chosen[:max(n for n in WIDTHS if n <= len(chosen))]
    # A task's full logical budget is preserved across physical calls.
    return dict(task_ids=[t["task_id"] for t, _ in chosen], band_end=band,
                actual_size=len(chosen), target=target, previous_success=safe,
                call_token_ceiling=context-max(w for _, w in chosen))


class Journal:
    """Use one exclusive writer for the entire allocation."""
    def __init__(self, directory, identity, hardware):
        self.root = Path(directory)
        self.run_id = checksum(identity)
        self.hardware = hardware
        self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def writer(self):
        with (self.root / "worker.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest = self.root / "manifest.json"
            if manifest.exists():
                read_record(manifest, self.run_id)
            else:
                write_record(manifest, {"run_id": self.run_id})
            for name in ("states", "attempts"):
                (self.root / name).mkdir(exist_ok=True)
            yield self

    def states(self):
        rows = [read_record(p, self.run_id) for p in (self.root/"states").glob("*.json")]
        return {r["task_id"]: r for r in rows}

    def attempts(self):
        rows = [read_record(p, self.run_id) for p in (self.root/"attempts").glob("*.json")]
        return [r for r in rows if r["hardware"] == self.hardware]

    def checkpoint(self, task, tokens, reason):
        if reason not in {*TERMINAL, "checkpoint"} or not tokens or len(tokens) > task["budget"]:
            raise ValueError("Invalid prefix or finish reason")
        if any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("Token IDs must be nonnegative integers")
        if (reason == "checkpoint" and len(tokens) == task["budget"]
                or reason == "token_limit" and len(tokens) != task["budget"]):
            raise ValueError("The stopping reason is inconsistent with the token count")
        path = self.root/"states"/(checksum(task["task_id"])+".json")
        row = dict(run_id=self.run_id, task_id=task["task_id"], tokens=list(tokens), reason=reason)
        if path.exists():
            old = read_record(path, self.run_id)
            if old == row:
                return
            if old["reason"] in TERMINAL or list(tokens[:len(old["tokens"])]) != old["tokens"]:
                raise ValueError("Cannot replace a terminal response or change a committed prefix")
        write_record(path, row)

    def start(self, group):
        path = self.root/"attempts"/(uuid.uuid4().hex+".json")
        write_record(path, dict(group, hardware=self.hardware, run_id=self.run_id,
                                started_at=time.time(), status="running"))
        return path

    def finish(self, path, status, *, elapsed_seconds):
        if status not in {"complete", "segment_complete", "oom", "interrupted", "failed"}:
            raise ValueError("Unknown attempt outcome")
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("Invalid elapsed time")
        row = read_record(path, self.run_id)
        if row["status"] != "running":
            raise ValueError("Attempt already closed")
        write_record(path, dict(row, status=status, elapsed_seconds=elapsed_seconds))
