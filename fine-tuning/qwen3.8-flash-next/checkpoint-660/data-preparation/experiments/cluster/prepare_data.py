#!/usr/bin/env python3
"""Export only official ARC data for the selected bwUniCluster workspace."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.arc_finetune import Example, _load_official, compact, sha256_file
from experiments.cluster.verify_data import verify
from scripts.build_local_training_archive import pair_hash

SOURCE = ROOT / "artifacts/releases/arc-synthetic-training-package-v1"
DESTINATION = ROOT / "artifacts/cluster/arc-agi3-data-v1"


def write_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(compact(record) + "\n" for record in records))


def select_development(records: list[dict], size: int, seed: int) -> tuple[list[dict], list[dict]]:
    if not 0 < size < len(records):
        raise ValueError("Development split must leave nonempty training data")
    ordered = sorted(records, key=lambda r: hashlib.sha256(
        f"{seed}:{r['task_id']}".encode()).hexdigest())
    dev_ids = {r["task_id"] for r in ordered[:size]}
    return (sorted((r for r in records if r["task_id"] not in dev_ids), key=lambda r: r["task_id"]),
            sorted((r for r in records if r["task_id"] in dev_ids), key=lambda r: r["task_id"]))


def conversations(records: list[dict]) -> list[dict]:
    result = []
    for record in records:
        example = Example.from_record(record)
        result.append({"task_id": record["task_id"], "messages": [
            {"role": "user", "content": example.task.prompt()},
            {"role": "assistant", "content": example.completion},
        ]})
    return result


def build(source: Path, destination: Path, dev_size: int = 128, seed: int = 42) -> dict:
    archive = destination.with_suffix(".zip")
    if destination.exists() or archive.exists():
        raise FileExistsError("Choose a new output directory; existing bundles are immutable")
    expected_hashes = {}
    for line in (source / "manifests/checksums.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        expected_hashes[name] = digest
    source_hashes = {}
    # Deliberately never open the external synthetic datasets.
    for name in ("data/arc_train.jsonl", "data/arc_validate.jsonl"):
        source_hashes[name] = sha256_file(source / name)
        if source_hashes[name] != expected_hashes[name]:
            raise ValueError(f"Source checksum mismatch: {name}")
    all_train = _load_official(source / "data/arc_train.jsonl", 1008)
    evaluation = sorted(_load_official(source / "data/arc_validate.jsonl", 138),
                        key=lambda r: r["task_id"])
    if {r["task_id"] for r in all_train} & {r["task_id"] for r in evaluation}:
        raise ValueError("Training/evaluation task ID overlap")
    if {r["task_fingerprint"] for r in all_train} & {r["task_fingerprint"] for r in evaluation}:
        raise ValueError("Training/evaluation task content overlap")
    train, dev = select_development(all_train, dev_size, seed)
    train_pairs = {pair_hash(p) for r in all_train for split in ("train", "test") for p in r[split]}
    evaluation_pairs = {pair_hash(p) for r in evaluation for split in ("train", "test") for p in r[split]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".arc-data-", dir=destination.parent) as temporary:
        staged = Path(temporary) / destination.name
        staged.mkdir()
        for split, records in (("train", train), ("dev", dev)):
            write_records(staged / f"data/{split}.jsonl", records)
            write_records(staged / f"sft/{split}.jsonl", conversations(records))
        write_records(staged / "evaluation/tasks.jsonl", [
            {"task_id": r["task_id"], "train": r["train"],
             "test": [{"input": p["input"]} for p in r["test"]]} for r in evaluation])
        write_records(staged / "scoring/answers.jsonl", [
            {"task_id": r["task_id"], "outputs": [p["output"] for p in r["test"]]}
            for r in evaluation])
        shutil.copyfile(Path(__file__).with_name("verify_data.py"), staged / "verify.py")
        (staged / "README.md").write_text(
            "# ARC-only cluster data v1\n\n"
            f"{len(train)} training / {len(dev)} development / {len(evaluation)} protected evaluation tasks.\n"
            "The exact split and checksums are recorded in manifest.json. No BARC or other external synthetic data.\n\n"
            "sft/train.jsonl is for global training; sft/dev.jsonl is development-only.\n"
            "Both use the existing ARC JSON prompt and user/assistant messages. The training backend\n"
            "must apply the pinned Flash chat template, disable thinking, mask user/prefix tokens,\n"
            "retain all complete examples, and audit lengths before GPU training.\n\n"
            "data/ contains corresponding labeled ARC records. evaluation/tasks.jsonl contains\n"
            "only visible demonstrations and query inputs. scoring/answers.jsonl belongs solely\n"
            "to the scorer and must never be passed to the model or TTT code. TTT uses visible\n"
            "demonstrations, resets state for each task, and must use settings selected on development tasks.\n\n"
            "Run: python3 verify.py\n\n"
            "This is a data transfer package, not a runnable GPU training environment.\n"
            "Three-GPU Flash loading/training support and throughput remain to be validated.\n"
        )
        files = {str(path.relative_to(staged)): sha256_file(path)
                 for path in sorted(staged.rglob("*")) if path.is_file()}
        task_ids = {"train": [r["task_id"] for r in train],
                    "dev": [r["task_id"] for r in dev],
                    "evaluation": [r["task_id"] for r in evaluation]}
        manifest = {"schema_version": 1, "source_files_sha256": source_hashes,
                    "external_synthetic_tasks": 0, "split_seed": seed,
                    "counts": {name: len(ids) for name, ids in task_ids.items()},
                    "task_ids": task_ids, "files_sha256": files,
                    "official_train_evaluation_shared_pair_hashes": len(train_pairs & evaluation_pairs),
                    "model_token_lengths": "not yet audited with the selected Flash tokenizer",
                    "cluster_workspace": "/path/to/arc-workspace",
                    "selected_resources": {"nodes": 1, "gpus": 3, "wall_hours": 6}}
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        verified = verify(staged)
        temporary_archive = Path(temporary) / archive.name
        with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
            for path in sorted(staged.rglob("*")):
                if path.is_file():
                    output.write(path, str(Path(destination.name) / path.relative_to(staged)))
        with tempfile.TemporaryDirectory(dir=temporary) as extracted:
            with zipfile.ZipFile(temporary_archive) as output:
                output.extractall(extracted)
            verify(Path(extracted) / destination.name)
        staged.rename(destination)
        temporary_archive.rename(archive)
    return {**verified, "directory": str(destination), "archive": str(archive),
            "archive_bytes": archive.stat().st_size, "archive_sha256": sha256_file(archive)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    parser.add_argument("--dev-tasks", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.source, arguments.output, arguments.dev_tasks, arguments.seed), indent=2))
