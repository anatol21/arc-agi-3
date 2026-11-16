"""Usage: python3 token_accounting.py /path/to/arc-agi-3 > token-accounting.json

Count the generated tokens in the primary terminal answer for each task and
condition. The two input-file digests pin the released response archives.
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REVISION = "e37ec8b840155c28c843b175cf1f2116d5c5d8cc"
ARCHIVES = {
    "greedy-no-thinking/held-out-138/evidence/observations.jsonl":
        "cbb3cdad15a4d67feba6c50e0b3a2371ce1ca9ff0e877ff797f3226744a593ca",
    "bf16-sampling/responses.jsonl":
        "91d00b8d791b0bac93546ae5a113ad03c3158fc3f51426f19235fc98c7941b46",
}


def failure_kind(row):
    if row["finish_reason"] == "token_limit":
        return "token_limit"
    if row["finish_reason"] != "eos":
        raise ValueError(f"Unexpected terminal reason for {row['task_id']}")
    error = row["format_error"]
    try:
        json.loads(row["text"])
    except json.JSONDecodeError:
        return "eos_invalid_json"
    schema_errors = {
        "Grid must be rectangular": "eos_ragged_grid",
        "Response has the wrong number of grids": "eos_wrong_grid_count",
        "Grid must have 1–30 rows and columns": "eos_grid_dimension_out_of_bounds",
    }
    return schema_errors[error]


def summarize(checkout):
    root = checkout / "benchmarks/qwen3.8-flash-next"
    groups = defaultdict(dict)
    for path, expected_digest in ARCHIVES.items():
        payload = (root / path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_digest:
            raise ValueError(f"Input archive differs from release {REVISION}: {path}")
        for line in payload.splitlines():
            row = json.loads(line)
            if row.get("source_order", 0) != 0:
                continue
            if hashlib.sha256(row["text"].encode()).hexdigest() != row["response_sha256"]:
                raise ValueError(f"Response hash mismatch for {row['task_id']}")
            variant = row.get("strategy", row.get("arm"))
            task = row["task_id"]
            if task in groups[variant]:
                raise ValueError(f"Duplicate primary record for {variant}, {task}")
            groups[variant][task] = row

    if set(groups) != {"base", "finetuned", "low_nucleus", "instruct_sampling"}:
        raise ValueError("Unexpected set of variants")
    cohort = set(groups["base"])
    if len(cohort) != 138 or any(set(rows) != cohort for rows in groups.values()):
        raise ValueError("Conditions do not cover the same 138 distinct tasks")

    results = {}
    for variant, rows in groups.items():
        total = sum(row["generated_tokens"] for row in rows.values())
        correct = {t: r["generated_tokens"] for t, r in sorted(rows.items()) if r["task_correct"]}
        count = len(correct)
        record = {
            "terminal_answers": len(rows),
            "exact_tasks": count,
            "generated_tokens_total": total,
            "generated_tokens_per_answer": round(total / len(rows), 3),
            "generated_tokens_in_exact_answers": sum(correct.values()),
            "generated_tokens_per_exact_answer": round(sum(correct.values()) / count, 3),
            "all_generated_tokens_divided_by_exact_tasks": round(total / count, 3),
            "generated_tokens_by_exact_task": correct,
        }
        if variant in ("instruct_sampling", "low_nucleus"):
            failures = {}
            for task, row in sorted(rows.items()):
                if row["finish_reason"] == "token_limit" and not row["format_error"]:
                    raise ValueError(f"Token-limited response absent from format-error count: {task}")
                if row["format_error"]:
                    if row["task_correct"]:
                        raise ValueError(f"Unexpected correct score on malformed response: {task}")
                    failures[task] = {"kind": failure_kind(row), "scorer_message": row["format_error"]}
            record["format_error_counts"] = dict(sorted(Counter(
                item["kind"] for item in failures.values()).items()))
            record["format_errors_total"] = len(failures)
            record["format_errors_by_task"] = failures
        results[variant] = record
    return {
        "repository": "https://github.com/anatol21/arc-agi-3",
        "revision": REVISION,
        "files_relative_to_benchmarks_qwen3.8-flash-next_sha256": ARCHIVES,
        "unit": "One retained terminal answer per task and condition; NF4 source_order == 0.",
        "scope": "Output tokens in scored answers. Excludes prompts, duplicate observations, padding, replay and abandoned work. Solved subsets differ between variants.",
        "failure_definition": "Task-level answers flagged by the scorer. Stop at the token limit takes precedence over parse or schema exceptions. All other cases ended at EOS.",
        "variants": results,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(json.dumps(summarize(Path(sys.argv[1])), indent=2, ensure_ascii=False))
