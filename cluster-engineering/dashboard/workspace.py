"""Bounded, read-only workspace inspection; never follows workspace symlinks."""
import contextlib
import json
import math
import os
import stat
from pathlib import Path

TEXT_SUFFIXES = {'.txt', '.md', '.yaml', '.yml', '.json', '.jsonl', '.py', '.sh', '.out', '.log', '.csv', '.toml', '.cfg', '.ini', '.sbatch', '.err'}

def relative_path(root, value):
    root = Path(root).resolve()
    if not isinstance(value, str) or any(c in value for c in '\x00\n\r'):
        raise ValueError('Invalid workspace path')
    candidate = Path(value or '.')
    if candidate.is_absolute():
        try: candidate = candidate.relative_to(root)
        except ValueError: raise ValueError('Path must stay inside the configured workspace')
    if '..' in candidate.parts: raise ValueError('Path must stay inside the configured workspace')
    return root, candidate

@contextlib.contextmanager
def open_entry(root, value):
    root, relative = relative_path(root, value)
    descriptors = []
    try:
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        descriptors.append(fd)
        for i, part in enumerate(relative.parts):
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            if i < len(relative.parts) - 1: flags |= os.O_DIRECTORY
            fd = os.open(part, flags, dir_fd=fd)
            descriptors.append(fd)
        yield root, relative, fd
    except OSError as error:
        raise ValueError('Cannot open this entry. Links, unavailable paths, and permission-restricted files are not browsable.') from error
    finally:
        for fd in reversed(descriptors): os.close(fd)

def browse(root, value='', search='', offset=0, hidden=False):
    if not isinstance(search, str) or len(search) > 120: raise ValueError('Invalid folder filter')
    if not isinstance(offset, int) or not 0 <= offset <= 2000: raise ValueError('Invalid folder page')
    with open_entry(root, value) as (root, relative, fd):
        info = os.fstat(fd)
        common = {'root': str(root), 'path': str(relative) if relative.parts else '', 'absolute_path': str(root / relative), 'parent': str(relative.parent) if relative.parts else None}
        if stat.S_ISDIR(info.st_mode):
            rows, limited = [], False
            with os.scandir(fd) as entries:
                for number, entry in enumerate(entries):
                    if number >= 2000:
                        limited = True
                        break
                    if (not hidden and entry.name.startswith('.')) or search.casefold() not in entry.name.casefold(): continue
                    try: item = entry.stat(follow_symlinks=False)
                    except OSError: continue
                    kind = 'directory' if stat.S_ISDIR(item.st_mode) else 'link' if stat.S_ISLNK(item.st_mode) else 'file' if stat.S_ISREG(item.st_mode) else 'special'
                    rows.append({'name': entry.name, 'path': str(relative / entry.name), 'kind': kind, 'bytes': item.st_size, 'modified': item.st_mtime})
            rows.sort(key=lambda row: (row['kind'] != 'directory', row['name'].casefold()))
            return {**common, 'kind': 'directory', 'entries': rows[offset:offset+200], 'offset': offset, 'more': len(rows) > offset+200, 'scan_limited': limited}
        if not stat.S_ISREG(info.st_mode): raise ValueError('Only folders and regular text files can be inspected')
        if relative.suffix.lower() not in TEXT_SUFFIXES:
            return {**common, 'kind': 'file', 'bytes': info.st_size, 'preview_available': False, 'reason': 'Preview is available for text, configuration, and log files. Model weights are not loaded.'}
        raw = os.read(fd, 65537)
        if b'\0' in raw:
            return {**common, 'kind': 'file', 'bytes': info.st_size, 'preview_available': False, 'reason': 'Binary content is not previewed.'}
        return {**common, 'kind': 'file', 'bytes': info.st_size, 'preview_available': True, 'text': raw[:65536].decode('utf-8', errors='replace'), 'truncated': len(raw) > 65536}

def review(path, manifest):
    """Recent numeric metrics, without reading full token journals or model files."""
    workers = []
    for index in range(min(manifest['config']['workers'], 16)):
        metrics, last_time, error = {}, None, None
        latest = next((w['token'] for w in reversed(manifest.get('waves',[])) if index in w['indices']),None)
        allocation_metrics = {'first_saved_result_seconds','allocated_gpu_hours_so_far','new_items_per_gpu_hour_so_far'}
        log = path / 'workers' / str(index) / 'metrics.jsonl'
        if log.exists() or log.is_symlink():
            try:
                with open_entry(path, str(log.relative_to(path))) as (_, _, fd):
                    info = os.fstat(fd)
                    if not stat.S_ISREG(info.st_mode): raise ValueError('Metrics journal is not a regular file')
                    os.lseek(fd, max(0, info.st_size - 65536), os.SEEK_SET)
                    # One bounded snapshot, even if the adapter keeps appending.
                    lines = os.read(fd, 65536).splitlines()
                    if info.st_size > 65536: lines = lines[1:]
                for raw in lines:
                    try: row = json.loads(raw)
                    except (ValueError, UnicodeDecodeError): continue
                    if not isinstance(row, dict): continue
                    if row.get('identity') not in (None, manifest['identity']): continue
                    for key, value in row.items():
                        if key in allocation_metrics and latest is not None and row.get('attempt') != latest:
                            continue
                        if key not in ('worker', 'time', 'timestamp') and isinstance(value, (int, float)) and not isinstance(value, bool) and abs(value) <= 1.7976931348623157e308 and math.isfinite(value):
                            if key in metrics or len(metrics) < 40: metrics[key] = value
                    last_time = row.get('time', last_time)
            except (ValueError, OSError):
                error = 'Metrics journal is unavailable or is not a regular file inside this run.'
        workers.append({'index': index, 'metrics': metrics, 'last_time': last_time, 'error': error})
    return {'workers': workers, 'workers_limited': manifest['config']['workers'] > 16, 'journal_window_bytes': 65536}
