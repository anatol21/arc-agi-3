#!/usr/bin/env python3
"""Verify the standalone ARC-only data bundle without ML dependencies."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify(root: Path) -> dict:
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["schema_version"] != 1 or manifest["external_synthetic_tasks"] != 0:
        raise ValueError("Expected an ARC-only v1 manifest")
    for name, expected in manifest["files_sha256"].items():
        path = (root / name).resolve()
        if root not in path.parents or file_hash(path) != expected:
            raise ValueError(f"Checksum/path mismatch: {name}")

    task_sets = {}
    for split in ("train", "dev"):
        raw = read_records(root / f"data/{split}.jsonl")
        messages = read_records(root / f"sft/{split}.jsonl")
        expected_ids = manifest["task_ids"][split]
        if [r["task_id"] for r in raw] != expected_ids:
            raise ValueError(f"Wrong raw task IDs: {split}")
        if [r["task_id"] for r in messages] != expected_ids:
            raise ValueError(f"Wrong conversation task IDs: {split}")
        if len(set(expected_ids)) != len(expected_ids):
            raise ValueError(f"Duplicate task IDs: {split}")
        for record, chat in zip(raw, messages):
            if record["source"] not in {"arc1_train", "arc2_train"}:
                raise ValueError("Unexpected training source")
            turns = chat["messages"]
            if len(turns) != 2 or [t["role"] for t in turns] != ["user", "assistant"]:
                raise ValueError("Expected user/assistant conversation")
            expected_outputs = {"outputs": [pair["output"] for pair in record["test"]]}
            if json.loads(turns[1]["content"]) != expected_outputs:
                raise ValueError("Wrong supervised answer")
            visible = json.loads(turns[0]["content"].split("\n\nTask:\n", 1)[1])
            expected_visible = {"train": record["train"],
                                "test": [{"input": p["input"]} for p in record["test"]]}
            if visible != expected_visible:
                raise ValueError("Wrong prompt or query answer leaked into prompt")
        task_sets[split] = set(expected_ids)

    public = read_records(root / "evaluation/tasks.jsonl")
    answers = read_records(root / "scoring/answers.jsonl")
    expected_ids = manifest["task_ids"]["evaluation"]
    if [r["task_id"] for r in public] != expected_ids:
        raise ValueError("Wrong public evaluation IDs")
    if [r["task_id"] for r in answers] != expected_ids:
        raise ValueError("Wrong scoring answer IDs")
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Duplicate evaluation IDs")
    for record, answer in zip(public, answers):
        if set(record) != {"task_id", "train", "test"}:
            raise ValueError("Unexpected public evaluation fields")
        if any(set(p) != {"input", "output"} for p in record["train"]):
            raise ValueError("Invalid visible demonstration")
        if any(set(p) != {"input"} for p in record["test"]):
            raise ValueError("Query answer leaked into public evaluation")
        if set(answer) != {"task_id", "outputs"} or len(answer["outputs"]) != len(record["test"]):
            raise ValueError("Invalid scorer-owned answer")
    task_sets["evaluation"] = set(expected_ids)
    for first, second in (("train", "dev"), ("train", "evaluation"), ("dev", "evaluation")):
        if task_sets[first] & task_sets[second]:
            raise ValueError(f"Overlapping splits: {first}, {second}")
    actual = {key: len(value) for key, value in task_sets.items()}
    if actual != manifest["counts"]:
        raise ValueError("Split count mismatch")
    return {"verified": True, "counts": actual, "external_synthetic_tasks": 0}


if __name__ == "__main__":
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    print(json.dumps(verify(directory), indent=2))
