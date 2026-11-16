"""Usage: python3 analyze.py /path/to/arc-agi-3 (pinned revision in report.md)."""
import gzip
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

here = Path(__file__).resolve().parent
workspace = here.parents[2]
bench = Path(sys.argv[1]) / "benchmarks"
model = bench / "qwen3.8-flash-next"
nf_path = model / "greedy-no-thinking/held-out-138/evidence/observations.jsonl"
bf_path = model / "bf16-sampling/responses.jsonl"
spot = {"1e97544e", "af22c60d", "e95e3d8e", "e66aafb8"}
sources = [workspace / "data" / name / "data/evaluation" for name in ("ARC-AGI-master", "ARC-AGI-2-main")]


def shape(grid):
    return [len(grid), len(grid[0])]


def error_kind(row):
    if row["finish_reason"] == "token_limit":
        return "token limit"
    err = row["format_error"] or ""
    if "number of grids" in err:
        return "wrong grid count"
    if "rectangular" in err:
        return "ragged grid"
    return err.split(": line ")[0]


def inspect(row):
    task = row["task_id"]
    taskfile = next(p / (task + ".json") for p in sources if (p / (task + ".json")).exists())
    tests = json.loads(taskfile.read_text())["test"]
    expected = [q["output"] for q in tests]
    raw = row["text"]
    assert hashlib.sha256(raw.encode()).hexdigest() == row["response_sha256"]
    d = {k: row[k] for k in ("task_id", "response_sha256", "generated_tokens", "finish_reason", "format_error", "grid_correct", "task_correct")}
    d.update(condition=row.get("arm", row.get("strategy")), source=row.get("source", row.get("stage")), precedence=row.get("source_order", 0), expected_shapes=list(map(shape, expected)))
    try:
        pred = json.loads(raw)["outputs"]
    except (ValueError, TypeError, KeyError):
        pred = None
    if d["format_error"] or d["finish_reason"] == "token_limit":
        d["outcome"] = "token limit" if d["finish_reason"] == "token_limit" else "invalid at EOS"
        d["diagnosis"] = error_kind(row)
        d["exact_if_wrapped_once"] = len(expected) == 1 and pred == expected[0]
    else:
        assert [g == target for g, target in zip(pred, expected, strict=True)] == d["grid_correct"]
        d["predicted_shapes"] = list(map(shape, pred))
        d["outcome"] = "exact" if d["task_correct"] else ("wrong shape" if d["predicted_shapes"] != d["expected_shapes"] else "wrong cells")
        d["copies_all_inputs"] = pred == [q["input"] for q in tests]
        if d["predicted_shapes"] == d["expected_shapes"]:
            d["cell_errors"] = sum(v != w for g, target in zip(pred, expected) for a, b in zip(g, target) for v, w in zip(a, b))
    return d


rows = [inspect(json.loads(line)) for path in (nf_path, bf_path) for line in path.read_text().splitlines()]
summary = {}
for cond in sorted({r["condition"] for r in rows}):
    primary = [r for r in rows if r["condition"] == cond and r["precedence"] == 0]
    assert len({r["task_id"] for r in primary}) == len(primary) == 138
    summary[cond] = {
        "outcomes": dict(Counter(r["outcome"] for r in primary)),
        "invalid_at_eos": dict(Counter(r["diagnosis"] for r in primary if r["outcome"] == "invalid at EOS")),
        "exact_after_singleton_wrap": [r["task_id"] for r in primary if r.get("exact_if_wrapped_once")],
        "wrong_exact_input_copies": [r["task_id"] for r in primary if r.get("copies_all_inputs") and not r["task_correct"]],
        "one_cell_from_exact": [r["task_id"] for r in primary if r.get("cell_errors") == 1],
    }

detail = defaultdict(dict)
for path in (nf_path, bf_path):
    for row in map(json.loads, path.read_text().splitlines()):
        t = row["task_id"]
        if t not in spot:
            continue
        cond = row.get("arm", row.get("strategy"))
        q = json.loads((sources[0] / (t + ".json")).read_text())["test"][0]
        i, gold = q["input"], q["output"]
        out = {}
        detail[t][cond] = out
        if t == "af22c60d" and row["finish_reason"] == "token_limit":
            raw = row["text"]
            chunks = re.findall(r'\[([0-9, ]+)\]', raw)
            completed = [[int(n) for n in re.findall(r'\d+', s)] for s in chunks]
            nums = list(map(int, re.findall(r'\d+', raw)))
            ref = sum(gold, [])
            out.update(completed_row_lengths=list(map(len, completed)), numeric_values=len(nums), prefix_errors=sum(a != b for a, b in zip(nums, ref)), first_wrong_positions_1based=[i+1 for i,(a,b) in enumerate(zip(nums,ref)) if a!=b][:10], initial_text=raw[:35], final_text=raw[-35:])
            continue
        p = json.loads(row["text"])["outputs"]
        if p == gold:
            out["missing_outer_list"] = True
        else:
            p = p[0]
        if t == "e66aafb8":
            out.update(prediction=p, gold=gold)
        else:
            errors = [(x+1,y+1,i[x][y],a,b) for x,(ar,br) in enumerate(zip(p,gold,strict=True)) for y,(a,b) in enumerate(zip(ar,br,strict=True)) if a!=b]
            out.update(errors=errors, wrong_on_blank=sum(e[2]==0 for e in errors), wrong_on_visible=sum(e[2]!=0 for e in errors))
api = defaultdict(Counter)
with gzip.open(bench / "qwen3.8-flash/openrouter-prompt-study/trials.jsonl.gz", "rt") as stream:
    for r in map(json.loads, stream):
        if r["task_id"] in spot:
            api[r["task_id"]][r["status"]] += 1
# Exploratory task-family label derived from demonstrations only.
family = []
for task in sorted({r["task_id"] for r in rows}):
    taskfile = next(p / (task + ".json") for p in sources if (p / (task + ".json")).exists())
    demos = json.loads(taskfile.read_text())["train"]
    def zero_only(pair):
        before, after = pair["input"], pair["output"]
        return (shape(before) == shape(after) and before != after
                and all(a == 0 or a == b for ra, rb in zip(before, after)
                        for a, b in zip(ra, rb)))
    if all(map(zero_only, demos)):
        family.append(task)
result = dict(demonstrations_change_only_zero_cells=family, counts=summary, focus=detail, historical_api=api, observations=rows)
(here / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(summary, indent=2))
print(json.dumps(api))
for cond in ("low_nucleus","instruct_sampling"):
    print(cond,detail["af22c60d"][cond])
