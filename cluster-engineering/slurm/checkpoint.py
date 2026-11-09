"""Local durable ledgers on a shared POSIX filesystem."""

import hashlib
import json
import os
from pathlib import Path
import tempfile


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        tmp.unlink(missing_ok=True)


def read_progress(root, identity, item_ids):
    result = {}
    for path in sorted((root / "items").glob("*.json")):
        row = json.loads(path.read_bytes())
        seal = row.pop("sha256")
        if (
            seal != digest(row)
            or row["campaign"] != identity
            or row["item"] not in item_ids
            or row["item"] != path.stem
            or row["status"] not in {"partial", "finished"}
            or type(row["units"]) != int
            or row["units"] < 0
        ):
            raise ValueError(f"Incompatible checkpoint: {path.name}")
        result[row["item"]] = row
    return result


def save_item(root, identity, item, units, status, payload):
    """Commit from the single active worker. Units count durable work such as tokens."""
    if not isinstance(item, str) or not item or Path(item).name != item:
        raise ValueError("Item must be a file-safe identifier")
    if type(units) != int or units < 0 or status not in {"partial", "finished"}:
        raise ValueError("Invalid checkpoint")
    path = root / "items" / (item + ".json")
    row = dict(
        campaign=identity, item=item, units=units, status=status, payload=payload
    )
    row["sha256"] = digest(row)
    if path.exists():
        previous = json.loads(path.read_bytes())
        if previous == row:
            return
        if previous.get("sha256") != digest(
            {k: v for k, v in previous.items() if k != "sha256"}
        ):
            raise ValueError("Corrupt previous checkpoint")
        if (
            previous["campaign"] != identity
            or previous["status"] == "finished"
            or previous["units"] > units
        ):
            raise ValueError("Cannot overwrite completed or newer work")
    atomic_json(path, row)
