"""Rebuild the published task table and summary, optionally checking each response against ARC."""
from argparse import ArgumentParser
import csv
import hashlib
import json
from pathlib import Path

from runpy import run_path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "observations.jsonl"
# SHA-256 of the sorted (task_id, outputs) answer rows written with json.dumps and a trailing LF per row.
REFERENCE_DIGEST = "32c9cea985069b19710ae2ed886ee7cd2f9a8f0c9e862fd102ca0bdf897b0031"
SERIAL = "arc-flash-eval138-v1"
ARMS = ("base", "finetuned")


def load_observations():
    pairs = {}
    for line in EVIDENCE.read_text().splitlines():
        row = json.loads(line)
        if hashlib.sha256(row["text"].encode()).hexdigest() != row["response_sha256"]:
            raise ValueError("Response digest mismatch")
        if row["arm"] not in ARMS or row["finish_reason"] not in {"eos", "token_limit"}:
            raise ValueError("Unknown model arm or nonterminal response")
        if not row["grid_correct"] or all(row["grid_correct"]) != row["task_correct"]:
            raise ValueError("Inconsistent task score")
        pairs.setdefault((row["task_id"], row["arm"]), []).append(row)
    ids = sorted({task for task, _ in pairs})
    if len(ids) != 138 or set(pairs) != {(t, a) for t in ids for a in ARMS}:
        raise ValueError("Expected finished responses for both arms on all 138 tasks")
    for rows in pairs.values():
        rows.sort(key=lambda r: r["source_order"])
        if ([r["source_order"] for r in rows] != list(range(len(rows))) or
                len({r["source"] for r in rows}) != len(rows)):
            raise ValueError("Inconsistent source precedence")
    return ids, pairs


def verify_scores(ids, pairs, directories):
    answers = []
    for task in ids:
        choices = []
        for base in directories:
            path = base / (task + ".json")
            if path.is_file():
                choices.append([q["output"] for q in json.loads(path.read_bytes())["test"]])
        if not choices or any(v != choices[0] for v in choices):
            raise ValueError(f"Missing or inconsistent reference for {task}")
        answers.append({"task_id": task, "outputs": choices[0]})
    normalized = "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in answers).encode()
    if hashlib.sha256(normalized).hexdigest() != REFERENCE_DIGEST:
        raise ValueError("Reference grid digest differs from the checked release")
    by_source = {}
    for key, rows in pairs.items():
        for r in rows:
            by_source.setdefault(r["source"], {})[key] = r
    score = run_path(str(ROOT / "producer/arc-flash-eval138-v1/protocol.py"))["score"]
    for records in by_source.values():
        # The original aggregator treats output truncated at the cap as an invalid answer.
        inputs = {k: dict(r, text="" if r["finish_reason"] == "token_limit" else r["text"]) for k, r in records.items()}
        _, scored = score(answers, inputs)
        for s in scored:
            r = records[s["task_id"], s["arm"]]
            if any(s[field] != r[field] for field in ("task_correct", "grid_correct", "format_error")):
                raise ValueError(f"Scoring mismatch: {s['task_id']} / {s['arm']} / {r['source']}")
        if len(scored) != len(inputs):
            raise ValueError("Incomplete scorer output")


def summarize(ids, pairs, output):
    output.mkdir(parents=True, exist_ok=True)
    table = []
    for task in ids:
        attempts = {a: pairs[task, a] for a in ARMS}
        first = {a: rows[0] for a, rows in attempts.items()}
        lengths = {len(r["grid_correct"]) for rs in attempts.values() for r in rs}
        if len(lengths) != 1:
            raise ValueError("Inconsistent number of query grids")
        row = dict(task_id=task, query_grids=lengths.pop())
        for a in ARMS:
            row[a + "_source"] = first[a]["source"]
            row[a + "_correct"] = first[a]["task_correct"]
            preferred = next((r for r in attempts[a] if r["source"] == SERIAL), first[a])
            row[a + "_serial_preferred_correct"] = preferred["task_correct"]
        table.append(row)
    def paired(suffix):
        solved = {a: set(r["task_id"] for r in table if r[a + suffix]) for a in ARMS}
        b, f = solved["base"], solved["finetuned"]
        return dict(base=sorted(b), checkpoint_660=sorted(f), both=sorted(b & f),
                    base_only=sorted(b - f), checkpoint_660_only=sorted(f - b),
                    neither=len(ids) - len(b | f))
    summary = dict(tasks=len(ids), query_grids=sum(t["query_grids"] for t in table),
                   finished_task_arm_pairs=len(pairs), archived_observations=sum(map(len, pairs.values())),
                   repeated_pairs=sum(len(rs) > 1 for rs in pairs.values()),
                   first_completion=paired("_correct"), serial_preferred=paired("_serial_preferred_correct"))
    summary["primary_by_arm"] = {}
    for arm in ARMS:
        primary = [pairs[t, arm][0] for t in ids]
        summary["primary_by_arm"][arm] = {
            "finished_tasks": len(primary),
            "solved_tasks": sum(r["task_correct"] for r in primary),
            "correct_grids": sum(sum(r["grid_correct"]) for r in primary),
            "query_grids": sum(len(r["grid_correct"]) for r in primary),
            "format_errors_including_token_limit": sum(r["format_error"] is not None for r in primary),
            "token_limit": sum(r["finish_reason"] == "token_limit" for r in primary),
        }
    with (output / "paired-results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    cmd = ArgumentParser(description=__doc__)
    cmd.add_argument("--references", type=Path, nargs="+",
                     help="Local ARC-AGI and ARC-AGI-2 evaluation directories with public per-task JSON")
    cmd.add_argument("--out", type=Path, required=True)
    args = cmd.parse_args()
    ids, pairs = load_observations()
    if args.references:
        verify_scores(ids, pairs, args.references)
    summarize(ids, pairs, args.out)
