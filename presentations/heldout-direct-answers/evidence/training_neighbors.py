"""Inspect the archived fine-tuning rows for task 1e97544e. Uses the standard library."""
import hashlib
import json
from pathlib import Path

workspace = Path(__file__).resolve().parents[3]
archive = workspace / "artifacts/cluster/arc-agi3-data-v1"
manifest = json.loads((archive / "manifest.json").read_text())
target_id = "1e97544e"


def load_verified(name):
    payload = (archive / name).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == manifest["files_sha256"][name]
    return [json.loads(line) for line in payload.splitlines()]


def fills_zeros(pair):
    a, b = pair["input"], pair["output"]
    return (len(a) == len(b) and len(a[0]) == len(b[0]) and a != b
            and all(0 not in row for row in b)
            and all(x == 0 or x == y for ra, rb in zip(a, b) for x, y in zip(ra, rb)))


def consecutive_palettes(pairs):
    palettes = [set(v for row in q["output"] for v in row) for q in pairs]
    sizes = list(map(len, palettes))
    return (all(p == set(range(1, len(p) + 1)) for p in palettes)
            and sizes == list(range(sizes[0], sizes[0] + len(pairs))))


def has_target_geometry(grid):
    # Any assignment of 3–9 distinct nonzero colors, all eight grid orientations.
    if min(len(grid), len(grid[0])) < 2:  # A line alone cannot exhibit the diagonal boundary.
        return False
    palette_size = len({v for row in grid for v in row})
    if not 3 <= palette_size <= 9 or any(0 in row for row in grid):
        return False
    for oriented in (grid, list(zip(*grid))):
        for vertical in (oriented, oriented[::-1]):
            for rows in (vertical, [row[::-1] for row in vertical]):
                cycle = rows[0][:palette_size]
                if len(set(cycle)) != palette_size:
                    continue
                if all(v == cycle[(c + int(c < r)) % palette_size]
                       for r, row in enumerate(rows) for c, v in enumerate(row)):
                    return True
    return False


train = load_verified("data/train.jsonl")
messages = load_verified("sft/train.jsonl")
assert len(train) == len(messages) == 880
assert [r["task_id"] for r in train] == [r["task_id"] for r in messages] == manifest["task_ids"]["train"]
assert target_id in manifest["task_ids"]["evaluation"] and target_id not in manifest["task_ids"]["train"]
target = json.loads((workspace / f"data/ARC-AGI-master/data/evaluation/{target_id}.json").read_text())
target_pairs = target["train"] + target["test"]
assert all(has_target_geometry(q["output"]) and fills_zeros(q) for q in target_pairs)
families, exact_geometry, shortlist, duplicate_pairs = [], [], [], []
for task, message in zip(train, messages):
    task_id = task["task_id"]
    pairs = task["train"] + task["test"]
    assert json.loads(message["messages"][-1]["content"])["outputs"] == [q["output"] for q in task["test"]]
    if any(q in target_pairs for q in pairs):
        duplicate_pairs.append(task_id)
    if any(has_target_geometry(q["output"]) for q in pairs):
        exact_geometry.append(task_id)
    if all(map(fills_zeros, pairs)):
        families.append(task_id)
        if consecutive_palettes(pairs):
            shortlist.append(task_id)
print(json.dumps(dict(broad_family=families, consecutive_palette_family=shortlist,
                      matching_output_geometry=exact_geometry, duplicate_pairs=duplicate_pairs), indent=2))
