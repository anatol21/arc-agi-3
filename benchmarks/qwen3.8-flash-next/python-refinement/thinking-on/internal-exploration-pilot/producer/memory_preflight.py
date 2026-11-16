"""Report this worker's effective Linux memory cgroup limit before staging weights."""
import json
from pathlib import Path


def limits():
    memberships = Path('/proc/self/cgroup').read_text().splitlines()
    for entry in memberships:
        _, controllers, member = entry.split(':', 2)
        if not controllers:
            root = Path('/sys/fs/cgroup')
            filename = 'memory.max'
        elif 'memory' in controllers.split(','):
            root = Path('/sys/fs/cgroup/memory')
            filename = 'memory.limit_in_bytes'
        else:
            continue
        directory = root / member.lstrip('/')
        while directory.is_relative_to(root):
            file = directory / filename
            if file.is_file():
                raw = file.read_text().strip()
                yield str(file), None if raw == 'max' else int(raw)
            directory = directory.parent


if __name__ == '__main__':
    rows = dict(limits())
    print('WORKER_MEMORY_LIMITS', json.dumps(rows), flush=True)
    finite = [v for v in rows.values() if v is not None]
    if not rows or (finite and min(finite) < 309788960 * 1024):
        raise SystemExit('Worker memory limit unavailable or below the measured AX peak')
