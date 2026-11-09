"""Read the audit's native journals, verify bytes and export all scored attempts as JSONL."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def digest(data):
    return hashlib.sha256(data).hexdigest()


def export(workspace, audit):
    expected = {}
    for task, arms in audit["tasks_by_id"].items():
        for arm, attempts in arms.items():
            for order, row in enumerate(attempts):
                key = row["run"], task, arm
                if key in expected:
                    raise ValueError(f"Duplicate audit key {key}")
                expected[key] = order, row
    records = {}
    for filename, checksum in audit["source_sha256"].items():
        path = Path(filename)
        if path.name == "answers.scorer-only.jsonl":
            continue
        if path.is_absolute() or ".." in path.parts or path.parts[0] not in {"runs", "cluster-ui"}:
            raise ValueError(f"Unexpected source: {path}")
        data = (workspace / path).read_bytes()
        if digest(data) != checksum:
            raise ValueError(f"Native journal changed: {path}")
        if path.name == "predictions.jsonl":
            rows = [json.loads(s) for s in data.splitlines() if s.strip()]
        elif path.parent.name == "batches":
            rows = json.loads(data)["rows"]
        elif path.parent.name == "terminals":
            rows = [json.loads(data)["row"]]
        else:
            raise ValueError(f"Unknown file type: {path}")
        for row in rows:
            if row["finish_reason"] not in {"eos", "token_limit"}:
                continue
            key = path.parts[1], row["task_id"], row["arm"]
            order, score = expected[key]
            checksum = digest(row["text"].encode("utf-8"))
            if checksum != score["response_sha256"] or row["finish_reason"] != score["finish_reason"]:
                raise ValueError(f"Response differs from scoring audit: {key}")
            record = dict(task_id=key[1], arm=key[2], source=score["package"], source_order=order,
                          response_sha256=checksum, text=row["text"], finish_reason=row["finish_reason"],
                          prompt_tokens=row["prompt_tokens"], generated_tokens=row["generated_tokens"],
                          token_budget=row["token_budget"], grid_correct=score["grid_correct"],
                          task_correct=score["task_correct"], format_error=score["format_error"])
            if key in records and record != records[key]:
                raise ValueError(f"Inconsistent duplicate: {key}")
            records[key] = record
    if records.keys() != expected.keys():
        raise ValueError("Incomplete journal coverage")
    return sorted(records.values(), key=lambda r: (r["task_id"], r["arm"], r["source_order"]))


if __name__ == "__main__":
    cmd = argparse.ArgumentParser(description=__doc__)
    cmd.add_argument("workspace", type=Path)
    cmd.add_argument("audit", help="Original final audit JSON, or - to read from stdin")
    args = cmd.parse_args()
    audit = json.load(sys.stdin) if args.audit == "-" else json.loads(Path(args.audit).read_bytes())
    for row in export(args.workspace, audit):
        print(json.dumps(row, allow_nan=False, ensure_ascii=False))
