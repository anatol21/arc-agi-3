"""Copy an immutable model snapshot to worker-local scratch and verify its manifest."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time


@contextmanager
def local_snapshot(source, manifest, scratch, *, workers=32):
    """Yield (path, receipt); cleanup on scope exit, including after a failed copy.

    manifest maps relative filenames to dictionaries containing bytes and sha256.
    Supply independent, trusted hashes of the exact checkpoint revision.
    """
    source = Path(source).resolve(strict=True)
    scratch = Path(scratch).resolve(strict=True)
    if workers < 1 or not manifest:
        raise ValueError("Expected at least one worker and one file")
    for name, info in manifest.items():
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or name != rel.as_posix() or not rel.parts:
            raise ValueError("Expected canonical relative file paths")
        if not (source / rel).resolve(strict=True).is_relative_to(source):
            raise ValueError("Source path escapes the snapshot")
        if type(info["bytes"]) is not int or info["bytes"] < 0:
            raise ValueError("Invalid expected size")
    total = sum(info["bytes"] for info in manifest.values())
    if shutil.disk_usage(scratch).free < total:
        raise OSError("Insufficient local scratch capacity")
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    count = min(workers, available or 1, len(manifest))
    start = time.monotonic()
    with TemporaryDirectory(dir=scratch, prefix="model-") as tmp:
        dest = Path(tmp)

        def copy_verified(name):
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
            checksum = hashlib.sha256()
            with target.open("rb") as stream:
                for chunk in iter(lambda: stream.read(4 << 20), b""):
                    checksum.update(chunk)
            expected = manifest[name]
            if target.stat().st_size != expected["bytes"] or checksum.hexdigest() != expected["sha256"]:
                raise ValueError(f"Snapshot verification failed for {name}")

        with ThreadPoolExecutor(count) as pool:
            list(pool.map(copy_verified, manifest))
        yield dest, dict(files=len(manifest), bytes=total, workers=count,
                         staging_seconds=time.monotonic() - start)
