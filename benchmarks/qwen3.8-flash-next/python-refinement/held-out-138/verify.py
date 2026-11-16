"""Verify released first-pass outcomes against the frozen task journals and query grids."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def indexed(rows):
    items = {(r["arm"], r["task_id"]): r for r in rows}
    if len(items) != len(rows):
        raise ValueError("Duplicate task and arm in ledger")
    return items


def main():
    audit = ROOT / "python-case-audit"
    snapshot = json.loads((ROOT / "python-first-pass/snapshot.json").read_bytes())
    observed = json.loads((audit / "case-records.json").read_bytes())
    visible = (ROOT / "producer/arc-python-heldout-length-v1/visible.jsonl").read_bytes()
    assert hashlib.sha256(visible).hexdigest() == snapshot["visible_manifest_sha256"] == observed["visible_sha256"]
    assert observed["snapshot"] == snapshot["snapshot"]
    inputs = [json.loads(line) for line in visible.splitlines()]
    task_ids = {r["task_id"] for r in inputs}
    assert len(task_ids) == len(inputs) == 138 and sum(len(r["test"]) for r in inputs) == 185
    assert all(set(q) == {"input"} for r in inputs for q in r["test"])
    expected = {(arm, task) for arm in ("A", "B") for task in task_ids}
    cases = indexed(observed["rows"])
    states = indexed(snapshot["rows"])
    with (audit / "per-task-audit.csv").open(newline="") as f:
        ledger = indexed(list(csv.DictReader(f)))
    assert expected == cases.keys() == states.keys() == ledger.keys()
    summary = {"A": Counter(), "B": Counter()}
    for key, case in cases.items():
        state, row = states[key], ledger[key]
        assert all(state[k] == case[k] for k in state)
        assert case["status"] == row["outcome"]
        queries = case["query"]
        if case["returned_grids"]:
            assert len(queries) == case["expected_grids"]
            assert sum(bool(q["saved"]) for q in queries) == case["returned_grids"]
            exact = all(q["saved"] == q["reference"] for q in queries)
            assert case["status"] == ("exact" if exact else "wrong")
        else:
            assert case["status"] in {"open", "unattempted", "no_grid"}
        summary[key[0]][case["status"]] += 1
    solved = sorted(key for key, r in cases.items() if r["status"] == "exact")
    assert solved == [("B", "e95e3d8e")]
    winner = cases[solved[0]]
    source = winner["attempts"][winner["selected_candidate_number"] - 1]["code"]
    assert source.rstrip() == (audit / "e95e3d8e-B-candidate0.py").read_text().rstrip()
    assert sum(r["probes"] for r in cases.values()) == 0
    print("Held-out Python first pass", snapshot["snapshot"], dict(summary), "exact", solved)


if __name__ == "__main__":
    main()
