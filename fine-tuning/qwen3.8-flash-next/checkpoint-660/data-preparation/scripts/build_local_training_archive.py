#!/usr/bin/env python3
"""Build and verify the local ARC synthetic-training transfer archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path


PACKAGE = "arc-synthetic-training-package-v1"
EXPECTED = {
    "arc_train": 1008,
    "arc_validate": 138,
    "google_cleaned": 99532,
    "synthetic_arc_train": 199899,
}


class ValidationError(ValueError):
    pass


def stable_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_grid(value, where):
    if not isinstance(value, list) or not value:
        raise ValidationError(f"{where}: grid must be a non-empty list")
    if not all(isinstance(row, list) and row for row in value):
        raise ValidationError(f"{where}: every grid row must be non-empty")
    width = len(value[0])
    if any(len(row) != width for row in value):
        raise ValidationError(f"{where}: grid is not rectangular")
    for r, row in enumerate(value):
        for c, cell in enumerate(row):
            if type(cell) is not int or not 0 <= cell <= 9:
                raise ValidationError(f"{where}[{r}][{c}]: expected int in 0..9")
    return value


def validate_pair(value, where):
    if not isinstance(value, dict) or set(value) != {"input", "output"}:
        raise ValidationError(f"{where}: pair must contain only input and output")
    return {
        "input": validate_grid(value["input"], where + ".input"),
        "output": validate_grid(value["output"], where + ".output"),
    }


def pair_hash(pair):
    return stable_hash({"input": pair["input"], "output": pair["output"]})


def task_hash(train, test):
    return stable_hash({
        "train": sorted(pair_hash(pair) for pair in train),
        "test": sorted(pair_hash(pair) for pair in test),
    })


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{path}: invalid JSON: {exc}") from exc


def raw_arc_split(directory):
    output = {}
    for path in sorted(directory.glob("*.json")):
        record = read_json(path)
        if not isinstance(record, dict) or not {"train", "test"}.issubset(record):
            raise ValidationError(f"{path}: expected ARC train/test task")
        if not isinstance(record["train"], list) or not record["train"]:
            raise ValidationError(f"{path}: missing train pairs")
        if not isinstance(record["test"], list) or not record["test"]:
            raise ValidationError(f"{path}: missing test pairs")
        output[path.stem] = {
            "train": [validate_pair(pair, f"{path}:train[{i}]") for i, pair in enumerate(record["train"])],
            "test": [validate_pair(pair, f"{path}:test[{i}]") for i, pair in enumerate(record["test"])],
        }
    if not output:
        raise ValidationError(f"{directory}: no JSON tasks")
    return output


def arc_record(task_id, source, task):
    train, test = task["train"], task["test"]
    return {
        "task_id": task_id,
        "source": source,
        "task_fingerprint": task_hash(train, test),
        "train": train,
        "test": test,
    }


def reconstruct_arc(root):
    data = root / "data"
    a1t = raw_arc_split(data / "ARC-AGI-master" / "data" / "training")
    a1v = raw_arc_split(data / "ARC-AGI-master" / "data" / "evaluation")
    a2t = raw_arc_split(data / "ARC-AGI-2-main" / "data" / "training")
    a2v = raw_arc_split(data / "ARC-AGI-2-main" / "data" / "evaluation")

    if a1v.pop("070dd51e", None) is None:
        raise ValidationError("ARC-1 validation task 070dd51e is missing")

    train, seen_hashes, seen_ids = [], set(), set()
    for source, split in (("arc2_train", a2t), ("arc1_train", a1t)):
        for task_id in sorted(split):
            record = arc_record(task_id, source, split[task_id])
            fingerprint = record["task_fingerprint"]
            if fingerprint in seen_hashes:
                continue
            if task_id in seen_ids:
                raise ValidationError(f"ARC train task ID collision: {task_id}")
            train.append(record)
            seen_hashes.add(fingerprint)
            seen_ids.add(task_id)
    train.sort(key=lambda item: item["task_id"])

    # The notebook removes all ARC-1 validation IDs already selected for train.
    a1v = {task_id: task for task_id, task in a1v.items() if task_id not in seen_ids}

    validate, seen_hashes, seen_ids = [], set(), set()
    for source, split in (("arc2_validation", a2v), ("arc1_validation", a1v)):
        for task_id in sorted(split):
            record = arc_record(task_id, source, split[task_id])
            fingerprint = record["task_fingerprint"]
            if fingerprint in seen_hashes:
                continue
            if task_id in seen_ids:
                raise ValidationError(f"ARC validation task ID collision: {task_id}")
            validate.append(record)
            seen_hashes.add(fingerprint)
            seen_ids.add(task_id)
    validate.sort(key=lambda item: item["task_id"])

    if len(train) != EXPECTED["arc_train"] or len(validate) != EXPECTED["arc_validate"]:
        raise ValidationError(
            f"Unexpected ARC split sizes: train={len(train)}, validate={len(validate)}"
        )
    if {item["task_fingerprint"] for item in train} & {item["task_fingerprint"] for item in validate}:
        raise ValidationError("ARC_Train overlaps ARC_Validate by full task content")
    return train, validate


def reconstruct_google(root, arc_train):
    train_ids = {record["task_id"] for record in arc_train}
    train_pairs = {
        pair_hash(pair)
        for task in arc_train
        for split in ("train", "test")
        for pair in task[split]
    }
    output, seen = [], set()
    stats = {
        "source_files": 0,
        "raw_pairs": 0,
        "dropped_unlinked_pairs": 0,
        "removed_exact_arc_train_pair_copies": 0,
        "removed_repeated_googleds_pairs": 0,
    }
    for path in sorted((root / "data" / "GoogleDS").glob("*.json")):
        pairs = read_json(path)
        if not isinstance(pairs, list):
            raise ValidationError(f"{path}: expected a list of pairs")
        stats["source_files"] += 1
        stats["raw_pairs"] += len(pairs)
        task_id = path.stem
        # Policy selected in Manual_JOIN: drop the unlinked 40853293 file.
        if task_id not in train_ids:
            stats["dropped_unlinked_pairs"] += len(pairs)
            continue
        for index, raw_pair in enumerate(pairs):
            pair = validate_pair(raw_pair, f"{path}[{index}]")
            fingerprint = pair_hash(pair)
            if fingerprint in train_pairs:
                stats["removed_exact_arc_train_pair_copies"] += 1
                continue
            if fingerprint in seen:
                stats["removed_repeated_googleds_pairs"] += 1
                continue
            seen.add(fingerprint)
            output.append({
                "record_type": "googleds_generated_pair",
                "source": "googleds",
                "googleds_task_id": task_id,
                "linked_arc_task_id": task_id,
                "pair_index_in_source_file": index,
                "pair_fingerprint": fingerprint,
                "input": pair["input"],
                "output": pair["output"],
            })
    if stats["source_files"] != 400 or stats["raw_pairs"] != 100000:
        raise ValidationError(f"Unexpected GoogleDS input inventory: {stats}")
    if len(output) != EXPECTED["google_cleaned"]:
        raise ValidationError(f"Unexpected Google_cleaned count: {len(output)}")
    if len({record["pair_fingerprint"] for record in output}) != len(output):
        raise ValidationError("Google_cleaned has duplicate pairs")
    return output, stats


def validate_arc_record(record, where, allowed_sources=None, require_metadata=False):
    expected = {"task_id", "source", "task_fingerprint", "train", "test"}
    if require_metadata:
        expected.add("metadata")
    if not isinstance(record, dict) or set(record) != expected:
        raise ValidationError(f"{where}: does not match ARC task schema")
    if not isinstance(record["task_id"], str) or not record["task_id"]:
        raise ValidationError(f"{where}: invalid task_id")
    if not isinstance(record["source"], str):
        raise ValidationError(f"{where}: invalid source")
    if allowed_sources is not None and record["source"] not in allowed_sources:
        raise ValidationError(f"{where}: invalid source {record['source']}")
    if not isinstance(record["train"], list) or not record["train"]:
        raise ValidationError(f"{where}: empty train")
    if not isinstance(record["test"], list) or not record["test"]:
        raise ValidationError(f"{where}: empty test")
    train = [validate_pair(pair, f"{where}.train[{i}]") for i, pair in enumerate(record["train"])]
    test = [validate_pair(pair, f"{where}.test[{i}]") for i, pair in enumerate(record["test"])]
    if record["task_fingerprint"] != task_hash(train, test):
        raise ValidationError(f"{where}: task fingerprint mismatch")
    if require_metadata and not isinstance(record["metadata"], dict):
        raise ValidationError(f"{where}: metadata must be an object")
    return record


def validate_google_record(record, where):
    expected = {
        "record_type", "source", "googleds_task_id", "linked_arc_task_id",
        "pair_index_in_source_file", "pair_fingerprint", "input", "output",
    }
    if not isinstance(record, dict) or set(record) != expected:
        raise ValidationError(f"{where}: does not match Google pair schema")
    if record["record_type"] != "googleds_generated_pair" or record["source"] != "googleds":
        raise ValidationError(f"{where}: incorrect record_type or source")
    if not all(isinstance(record[key], str) and record[key] for key in (
        "googleds_task_id", "linked_arc_task_id", "pair_fingerprint"
    )):
        raise ValidationError(f"{where}: invalid identifier")
    if type(record["pair_index_in_source_file"]) is not int or record["pair_index_in_source_file"] < 0:
        raise ValidationError(f"{where}: invalid pair index")
    pair = validate_pair({"input": record["input"], "output": record["output"]}, where)
    if record["pair_fingerprint"] != pair_hash(pair):
        raise ValidationError(f"{where}: pair fingerprint mismatch")
    return record


def pairs_in_tasks(tasks):
    return {
        pair_hash(pair)
        for task in tasks
        for split in ("train", "test")
        for pair in task[split]
    }


def write_jsonl(path, records):
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def verify_jsonl(path, validator):
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValidationError(f"{path}:{line_number}: blank JSONL line")
            try:
                validator(json.loads(line), f"{path.name}:{line_number}")
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_number}: invalid JSON") from exc
            count += 1
    return count


def copy_validate_synthetic(source, destination, protected_pairs):
    counts = {"google_aux": 0, "barc_aux": 0}
    with source.open("r", encoding="utf-8") as reader, destination.open(
        "w", encoding="utf-8", newline="\n"
    ) as writer:
        for number, line in enumerate(reader, 1):
            if not line.strip():
                raise ValidationError(f"{source}:{number}: blank JSONL line")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{source}:{number}: invalid JSON") from exc
            validate_arc_record(record, f"synthetic:{number}", {"google_aux", "barc_aux"}, True)
            hashes = {
                pair_hash(pair)
                for split in ("train", "test")
                for pair in record[split]
            }
            if hashes & protected_pairs:
                raise ValidationError(f"synthetic:{number}: exact ARC_Validate pair overlap")
            counts[record["source"]] += 1
            writer.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            total = sum(counts.values())
            if total % 50000 == 0:
                print(f"Validated synthetic tasks: {total:,}", flush=True)
    if sum(counts.values()) != EXPECTED["synthetic_arc_train"]:
        raise ValidationError(f"Unexpected synthetic count: {sum(counts.values())}")
    if counts != {"google_aux": 12673, "barc_aux": 187226}:
        raise ValidationError(f"Unexpected synthetic source counts: {counts}")
    return counts


def frontmatter(path, field):
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return ""
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if line.startswith(field + ":"):
            return line.split(":", 1)[1].strip().strip("\"'")
    return ""


def create_docs(root, package):
    (package / "README.md").write_text(
        """# ARC synthetic-training package v1

This local research archive contains the finalized official ARC train/validation
split and two generated-data views for experiments in ARC-style rule inference.
It is for local transfer only. Check source-dataset terms before any publication
or redistribution.

## Extract and verify

    unzip arc-synthetic-training-package-v1.zip
    cd arc-synthetic-training-package-v1
    shasum -a 256 -c manifests/checksums.sha256

## Included datasets

| File | Records | Role |
| --- | ---: | --- |
| data/arc_train.jsonl | 1,008 tasks | Official ARC training tasks: ARC-2 primary plus 8 ARC-1-only tasks. |
| data/arc_validate.jsonl | 138 tasks | Protected official evaluation tasks: 120 ARC-2 plus 18 ARC-1-only tasks. |
| data/google_cleaned.jsonl | 99,532 pairs | Clean generated input/output pairs associated with ARC training tasks. |
| data/synthetic_arc_train.jsonl | 199,899 tasks | ARC-shaped generated episodes: 12,673 Google-derived and 187,226 BARC-derived. |

ARC_Validate is evaluation-only. Do not use it for model fitting, prompt
selection, synthetic-data construction, or hyperparameter selection.

Raw ARC, GoogleDS, and BARC downloads, BARC_cleaned, temporary SQLite indexes,
audit intermediates, and Jupyter checkpoints are intentionally excluded.
Synthetic metadata is provenance-only, not model-prompt content.

manifests/checksums.sha256 covers every archived file except itself.
""",
        encoding="utf-8",
    )
    (package / "docs" / "schemas.md").write_text(
        """# Dataset schemas

All files use UTF-8 JSON Lines. Every grid is a non-empty rectangular list of
integer color values from 0 through 9.

## Official ARC task

Files: data/arc_train.jsonl and data/arc_validate.jsonl.

Required fields: task_id (string), source (string), task_fingerprint (SHA-256
string), train (non-empty input/output pair list), and test (non-empty
input/output pair list). The task fingerprint hashes train and test pair
multisets separately, so it is stable when demonstration ordering changes.

## Google generated pair

File: data/google_cleaned.jsonl.

Required fields: record_type equal to googleds_generated_pair, source equal to
googleds, googleds_task_id, linked_arc_task_id, pair_index_in_source_file,
pair_fingerprint, input, and output. This is pair-level auxiliary data, not an
official ARC task.

## Synthetic ARC-style task

File: data/synthetic_arc_train.jsonl.

Required fields: task_id, source (google_aux or barc_aux), task_fingerprint,
train, test, and metadata. Train/test have the same episode shape as official
ARC tasks. Metadata is provenance-only: it must not enter model prompts, model
targets, or evaluation inputs. No executable BARC source code is included in
these model-facing synthetic task records.
""",
        encoding="utf-8",
    )

    notes = sorted((root / "papers").glob("*.md"))
    if len(notes) != 5:
        raise ValidationError(f"Expected five paper notes, found {len(notes)}")
    rows = []
    for note in notes:
        title = frontmatter(note, "title") or note.stem
        status = frontmatter(note, "status") or "to-read"
        rows.append(f"| {note.name} | {title} | {status} |")
    (package / "docs" / "reading-list.md").write_text(
        "# Reading list\n\n"
        "The existing local paper notes are indexed below. Their status is "
        "preserved; this list does not claim the papers have been read.\n\n"
        "| Note | Title | Status |\n| --- | --- | --- |\n"
        + "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def write_manifest(package, stats):
    manifest = {
        "package": PACKAGE,
        "format": "jsonl",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": {
            "ARC_Train": {
                "path": "data/arc_train.jsonl", "record_count": 1008,
                "record_type": "official_arc_task",
                "source_composition": {"arc2_train": 1000, "arc1_train": 8},
            },
            "ARC_Validate": {
                "path": "data/arc_validate.jsonl", "record_count": 138,
                "record_type": "official_arc_task", "role": "protected_evaluation_only",
                "source_composition": {"arc2_validation": 120, "arc1_validation": 18},
            },
            "Google_cleaned": {
                "path": "data/google_cleaned.jsonl", "record_count": 99532,
                "record_type": "generated_pair", "source_composition": stats,
            },
            "Synthetic_ARC_Train": {
                "path": "data/synthetic_arc_train.jsonl", "record_count": 199899,
                "record_type": "synthetic_arc_style_task",
                "source_composition": {"google_aux": 12673, "barc_aux": 187226},
            },
        },
        "cleaning_and_overlap_decisions": {
            "arc_train": "ARC-2 primary; append only ARC-1 train task content absent from ARC-2.",
            "arc_validate": "Remove 070dd51e; remove ARC-1 validation IDs in ARC_Train; ARC-2 validation primary; deduplicate by full task content.",
            "googleds": "Drop unlinked 40853293; remove exact official ARC train pairs; remove repeated generated pairs globally.",
            "synthetic": "Exact pair overlap with ARC_Validate is prohibited; metadata is provenance-only.",
            "excluded": "Raw data, BARC_cleaned, BARC programs, temporary SQLite indexes, audits, and checkpoints.",
        },
    }
    (package / "manifests" / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_checksums(package):
    target = package / "manifests" / "checksums.sha256"
    paths = sorted(path for path in package.rglob("*") if path.is_file() and path != target)
    target.write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.relative_to(package).as_posix()}"
            for path in paths
        ) + "\n",
        encoding="utf-8",
    )


def assert_exclusions(package):
    fragments = ("barc_clean", "data_100k", "suggestfunction", ".sqlite", ".ipynb_checkpoints")
    offenders = [
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if any(fragment in path.as_posix().lower() for fragment in fragments)
    ]
    if offenders:
        raise ValidationError(f"Excluded files entered staging: {offenders}")


def zip_and_verify(package, archive):
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
    ) as zipped:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                zipped.write(path, path.relative_to(package.parent).as_posix())
    with zipfile.ZipFile(archive) as zipped:
        bad = zipped.testzip()
        if bad:
            raise ValidationError(f"ZIP integrity failed: {bad}")
    with tempfile.TemporaryDirectory(prefix="arc-package-verify-") as temporary:
        destination = Path(temporary)
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(destination)
        extracted = destination / PACKAGE
        for line in (extracted / "manifests" / "checksums.sha256").read_text(
            encoding="utf-8"
        ).splitlines():
            digest, relative = line.split("  ", 1)
            if sha256_file(extracted / relative) != digest:
                raise ValidationError(f"Extracted checksum mismatch: {relative}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="replace an existing archive")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    archive = root / "artifacts" / "releases" / f"{PACKAGE}.zip"
    synthetic = root / "artifacts" / "arc_style_auxiliary" / "20261015_011606" / "auxiliary_arc_train.jsonl"
    synthetic_manifest = root / "artifacts" / "arc_style_auxiliary" / "20261015_011606" / "auxiliary_arc_train_manifest.json"
    notebook = root / "notebooks" / "Manual_JOIN.ipynb"
    for path in (synthetic, synthetic_manifest, notebook):
        if not path.is_file():
            raise FileNotFoundError(f"Required input missing: {path}")
    if archive.exists() and not args.force:
        raise FileExistsError(f"Archive exists: {archive}. Rerun with --force to replace it.")

    # The count composition must agree with the existing synthetic-file manifest.
    manifest = read_json(synthetic_manifest)
    source_counts = manifest.get("counts", {}) if isinstance(manifest, dict) else {}
    if source_counts.get("google_aux_tasks_written") != 12673 or source_counts.get("barc_aux_tasks_written") != 187226:
        raise ValidationError("Existing synthetic manifest does not match expected composition")

    print("Reconstructing ARC_Train, ARC_Validate, and Google_cleaned...", flush=True)
    arc_train, arc_validate = reconstruct_arc(root)
    google, google_stats = reconstruct_google(root, arc_train)
    protected_pairs = pairs_in_tasks(arc_validate)
    if {record["pair_fingerprint"] for record in google} & protected_pairs:
        raise ValidationError("Google_cleaned overlaps ARC_Validate")

    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="arc-package-stage-", dir=archive.parent) as temporary:
        package = Path(temporary) / PACKAGE
        for name in ("data", "docs", "manifests", "notebooks"):
            (package / name).mkdir(parents=True, exist_ok=True)

        counts = {
            "arc_train": write_jsonl(package / "data" / "arc_train.jsonl", arc_train),
            "arc_validate": write_jsonl(package / "data" / "arc_validate.jsonl", arc_validate),
            "google_cleaned": write_jsonl(package / "data" / "google_cleaned.jsonl", google),
        }
        print("Streaming, validating, and copying synthetic ARC-style tasks...", flush=True)
        composition = copy_validate_synthetic(
            synthetic, package / "data" / "synthetic_arc_train.jsonl", protected_pairs
        )
        counts["synthetic_arc_train"] = sum(composition.values())
        if counts != EXPECTED:
            raise ValidationError(f"Staged counts differ from expected: {counts}")

        # Validate every staged JSONL line and recompute protected-pair checks.
        staged_arc_train = verify_jsonl(
            package / "data" / "arc_train.jsonl", validate_arc_record
        )
        staged_arc_validate = verify_jsonl(
            package / "data" / "arc_validate.jsonl", validate_arc_record
        )
        staged_google = verify_jsonl(
            package / "data" / "google_cleaned.jsonl", validate_google_record
        )
        if (staged_arc_train, staged_arc_validate, staged_google) != (1008, 138, 99532):
            raise ValidationError("Staged JSONL verification count mismatch")
        with (package / "data" / "synthetic_arc_train.jsonl").open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                record = json.loads(line)
                validate_arc_record(record, f"staged synthetic:{number}", {"google_aux", "barc_aux"}, True)
                if {
                    pair_hash(pair)
                    for split in ("train", "test")
                    for pair in record[split]
                } & protected_pairs:
                    raise ValidationError(f"Staged synthetic validation overlap at row {number}")

        shutil.copy2(notebook, package / "notebooks" / "Manual_JOIN.ipynb")
        create_docs(root, package)
        write_manifest(package, google_stats)
        assert_exclusions(package)
        write_checksums(package)

        if archive.exists():
            archive.unlink()
        print("Creating ZIP64 archive...", flush=True)
        zip_and_verify(package, archive)

    print(json.dumps({
        "status": "verified",
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "counts": EXPECTED,
    }, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValidationError, OSError, zipfile.BadZipFile) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

