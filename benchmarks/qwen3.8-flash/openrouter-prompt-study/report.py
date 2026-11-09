"""Export ARC Practice trial records and verify the public OpenRouter result."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
from collections import Counter
from contextlib import ExitStack

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from apps.arc_practice.server import BenchmarkError, DatasetSchemaError, parse_model_outputs, task_prompt

IDENTITY = ("task_id", "experiment_id", "trial_index")
FIELDS = IDENTITY + ("source", "status", "correct", "test_count", "tests_correct",
                     "valid_response", "provider_success")
STATUSES = ("correct", "incorrect", "malformed_response", "provider_error")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def key(row):
    return tuple(row[k] for k in IDENTITY)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows, settings, reference=None):
    require(len(rows) == len({key(r) for r in rows}) == 1380, "Expected 1380 unique logical trials")
    tasks = {r["task_id"] for r in rows}
    config = {e["id"]: e for e in settings["experiments"]}
    methods = set(config)
    require(len(tasks) == 138 and len(methods) == 5, "Wrong cohort or method count")
    require({key(r) for r in rows} == {(t,e,j) for t in tasks for e in methods for j in (0,1)},
            "Missing or extra task/method/trial")
    if reference is not None:
        require(tasks == reference.keys(), "Reference dataset differs from trial task IDs")
    for r in rows:
        require(r["status"] in STATUSES, "Unknown recorded status")
        require((r["status"] == "correct") == r["correct"], "Status and score disagree")
        require(r["valid_response"] == (r["status"] in ("correct","incorrect")), "Validity disagrees with status")
        if reference is not None:
            task = reference[r["task_id"]]
            require(r["test_count"] == len(task["test"]), "Query count differs")
            e = config[r["experiment_id"]]
            message = task_prompt(task, e["prompt_modifier"], e["core_instruction"])
            require(hashlib.sha256(message.encode()).hexdigest() == r["prompt_sha256"], "Prompt mismatch")
        if r["status"] == "provider_error":
            require(not r["correct"] and r["tests_correct"] == 0, "Invalid provider error score")
            continue
        try:
            grids = parse_model_outputs(r["raw_response"], r["test_count"])
        except (BenchmarkError, DatasetSchemaError):
            require(r["status"] == "malformed_response" and r["tests_correct"] == 0, "Parse error disagrees with score")
            continue
        require(r["valid_response"], "Valid JSON reported as malformed")
        if reference is not None:
            expected = reference[r["task_id"]]["test"]
            n = sum(g == pair["output"] for g, pair in zip(grids, expected, strict=True))
            require(n == r["tests_correct"] and (n == len(expected)) == r["correct"], f"Score mismatch: {key(r)}")
    out = {"tasks": len(tasks), "logical_trials":len(rows), "methods":{}}
    for exp in settings["experiments"]:
        chosen = [r for r in rows if r["experiment_id"] == exp["id"]]
        counts = Counter(r["status"] for r in chosen)
        solved = sorted({r["task_id"] for r in chosen if r["correct"]})
        out["methods"][exp["id"]] = {"label":exp["label"], "logical_trials":len(chosen),
            "statuses":{status:counts[status] for status in STATUSES}, "solved_at_least_once":solved}
    out["solved_by_any_method"] = sorted({r["task_id"] for r in rows if r["correct"]})
    return out


def export(combined, checkpoint, config, dest):
    archive, old, original = read(combined), read(checkpoint), read(config)
    prior = {key(r):r for r in old["records"]}
    require(len(prior) == len(old["records"]) == 824, "Expected original pre-resume checkpoint")
    require(old["state"] == "cancelled" and old["config"]["seed"] == 1911, "Wrong pre-resume run")
    require([(r["run_id"], r["record_count"]) for r in archive["source_runs"]] ==
            [("initial_checkpoint", 20), ("continuation", 1360)], "Wrong source runs")
    rows = []
    recovered = 0
    run_settings = []
    for run in archive["source_runs"]:
        run_settings.append({k:run[k] for k in ("run_id","seed","global_parallel_requests","record_count","state")})
        for r in run["records"]:
            source = r
            if "raw_response" not in r:
                require(run["run_id"] == "continuation" and run["seed"] == old["config"]["seed"],
                        "Cannot reconstruct a row from a different run")
                source = prior[key(r)]
                require(all(r[k] == source[k] for k in FIELDS + ("job_index","task_order","line_number",
                            "queue_seconds","latency_seconds","attempts","prediction","usage")),
                        f"Checkpoint mismatch: {key(r)}")
                recovered += 1
            row = {k:r[k] for k in FIELDS}
            request = source["model_request"]
            initial = request["messages"][0]
            require(initial["role"] == "user", "Expected initial user prompt")
            texts = [a["number"] for a in source["attempts"] if a["status"] != "provider_error"]
            retained = texts[-1] if texts else None
            require((retained is None) == (source["raw_response"] is None), "Unexpected saved response")
            row.update(source_run=run["run_id"], raw_response=source["raw_response"], raw_response_attempt=retained,
                       prompt_sha256=hashlib.sha256(initial["content"].encode()).hexdigest(),
                       attempt_statuses=[a["status"] for a in source["attempts"]])
            rows.append(row)
    require(recovered == 804, "Unexpected number of recovered compacted responses")
    settings = {"experiments": original["experiments"], "trials_per_task":original["trials_per_task"],
                "max_format_repairs":original["format_repair_max_attempts"], "source_runs":run_settings}
    summary = summarize(rows, settings)
    dest.mkdir(parents=True, exist_ok=True)
    # The archive and end-of-run checkpoint remain private. These hashes identify exact inputs.
    write(dest/"sources.json",{"producer_snapshot_sha256":digest(HERE.parents[2]/"apps"/"arc_practice"/"server.py"),
         "combined_export_sha256":digest(combined), "pre_resume_checkpoint_sha256":digest(checkpoint),
         "settings_sha256":digest(config), "restored_compact_rows":recovered})
    write(dest/"settings.json", settings)
    rows.sort(key=key)
    with ExitStack() as ctx:
        stream = ctx.enter_context((dest/"trials.jsonl.gz").open("wb"))
        gz = ctx.enter_context(gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0))
        writer = ctx.enter_context(io.TextIOWrapper(gz, encoding="utf-8", newline="\n"))
        for row in rows:
            writer.write(json.dumps(row, ensure_ascii=False, separators=(",",":"), allow_nan=False) + "\n")
    write(dest / "summary.json", summary)


def verify(folder, reference_path):
    with gzip.open(folder/"trials.jsonl.gz", "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    reference = None
    if reference_path:
        with reference_path.open() as f:
            records = [json.loads(line) for line in f if line.strip()]
        reference = {r["task_id"]:r for r in records}
        require(len(reference) == len(records), "Duplicate reference task IDs")
    require(summarize(rows, read(folder/"settings.json"), reference) == read(folder/"summary.json"),
            "Stored summary disagrees with trial records")
    print(f"Verified {len(rows)} distinct trials across 138 tasks" +
          (" including per-query exact scores." if reference is not None else " using archived scores."))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    e = sub.add_parser("export")
    for arg in ("combined","checkpoint","config","dest"):
        e.add_argument(arg, type=Path)
    v = sub.add_parser("verify")
    v.add_argument("--folder", type=Path, default=HERE)
    v.add_argument("--reference", type=Path)
    a = p.parse_args()
    if a.action == "export":
        export(a.combined, a.checkpoint, a.config, a.dest)
    else:
        verify(a.folder, a.reference)
