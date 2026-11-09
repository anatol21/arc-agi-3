"""Short result transactions and committed-line snapshots; no model imports."""
import contextlib
import datetime
import fcntl
import errno
import hashlib
import json
import os
import tempfile
from pathlib import Path


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


@contextlib.contextmanager
def results_lock(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.results.lock').open('a+') as stream:
        fcntl.lockf(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.lockf(stream.fileno(), fcntl.LOCK_UN)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        try:
            os.fsync(fd)
        except OSError as error:
            # Some shared filesystems do not implement directory fsync. File
            # content is still fsynced; don't turn that limitation into data loss.
            if error.errno not in (errno.EINVAL, errno.ENOTSUP): raise
    finally:
        os.close(fd)


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def snapshot_jsonl(path, *, repair_tail=False, allow_partial_tail=False):
    """Caller holds results_lock when repair is requested. Reads never repair by default."""
    path = Path(path)
    data = path.read_bytes() if path.exists() else b''
    boundary = len(data) if not data or data.endswith(b'\n') else data.rfind(b'\n') + 1
    tail = data[boundary:]
    if tail and repair_tail:
        backup = path.with_suffix('.interrupted-tail')
        with backup.open('ab') as stream:
            stream.write(tail + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        with path.open('r+b') as stream:
            stream.truncate(boundary)
            stream.flush()
            os.fsync(stream.fileno())
        sync_directory(path.parent)
    elif tail and not allow_partial_tail:
        raise ValueError('Incomplete prediction tail; resume generation to recover it')
    committed = data[:boundary]
    return committed, {'sha256': hashlib.sha256(committed).hexdigest(), 'bytes': len(committed),
                       'records': len(committed.splitlines()), 'ignored_trailing_bytes': len(tail),
                       'captured_at': utcnow()}
