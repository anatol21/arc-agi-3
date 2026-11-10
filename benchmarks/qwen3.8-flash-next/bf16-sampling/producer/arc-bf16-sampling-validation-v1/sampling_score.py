"""Post-generation CPU scoring of single-draw BF16 sampling arms versus the greedy pilot."""
import json
import hashlib
import os
import sys
import fcntl
from pathlib import Path
from runpy import run_path

FOLDER = Path(__file__).resolve().parent
if __package__:
    from .continuation_score import checked_states, publish_snapshot
    from .generation import TERMINAL, digest
else:
    sys.path.insert(0, str(FOLDER))
    from continuation_score import checked_states, publish_snapshot
    from generation import TERMINAL, digest


GREEDY_JOB = "8487396"
GREEDY_ID = "b0bc62947697ba1fa5cfbc3ba6d66bbf1bed47dea6f115a05eb8ddfb080a5d0a"
GREEDY_STATES = "b9c8033c7d66a1428cea5cd0380b51ef1944905c315720d8dfc09f3576182d6b"
GREEDY_LOADER = "2d6ef31c9491f452e7c8c5b6aea12d51b3a2b072e9ae234b8967da03e1b3f670"


def greedy_states(workspace, plan):
    root = workspace / "runs/arc-bf16-base-debug-v1" / GREEDY_ID
    if plan["pilot_benchmark_id"] != GREEDY_ID:
        raise ValueError("Unexpected greedy reference")
    receipt = root / "allocations" / GREEDY_JOB / "verified-loader.json"
    if hashlib.sha256(receipt.read_bytes()).hexdigest() != GREEDY_LOADER:
        raise ValueError("Original BF16 loader receipt changed")
    pilot = json.loads((workspace / "arc-bf16-base-debug-v1/pilot-plan.json").read_bytes())
    if (pilot["benchmark_id"] != GREEDY_ID
            or digest({k: v for k, v in pilot.items() if k != "benchmark_id"}) != GREEDY_ID
            or pilot["prompts"] != plan["prompts"] or pilot["groups"] != plan["groups"]):
        raise ValueError("Greedy and sampling prompts or cohorts differ")
    states = checked_states(root, plan["prompts"], GREEDY_ID, plan["eos_token_id"])
    if (len(states) != 12 or digest({t: digest(r) for t, r in states.items()}) != GREEDY_STATES
            or any(r["finish_reason"] not in TERMINAL or r["slurm_job_id"] != GREEDY_JOB for r in states.values())):
        raise ValueError("Original greedy terminal journals changed")
    return states


def score_single(answers, states, native_score):
    terminal = {t: r for t, r in states.items() if r["finish_reason"] in TERMINAL}
    summary, details = native_score(answers, {(t, "base"): r for t, r in terminal.items()})
    totals = summary["arms"]["base"]
    totals["valid_responses"] = totals["processed_tasks"] - totals["invalid_json_or_grids"]
    for task in details:
        row = terminal[task["task_id"]]
        task.update(generated_tokens=row["generated_tokens"], batch_size=row["actual_batch_size"],
                    response_sha256=hashlib.sha256(row["text"].encode()).hexdigest(),
                    job_id=row["slurm_job_id"], attempt_id=row["attempt_id"])
    return {"summary": totals, "tasks": details}


def comparison(trial, greedy):
    by_task = {r["task_id"]: r for r in greedy["tasks"]}
    pairs = {"both_exact": [], "sample_only_exact": [], "greedy_only_exact": [], "neither_exact": []}
    for r in trial["tasks"]:
        greedy_exact = by_task[r["task_id"]]["task_correct"]
        if r["task_correct"]:
            key = "both_exact" if greedy_exact else "sample_only_exact"
        else:
            key = "greedy_only_exact" if greedy_exact else "neither_exact"
        pairs[key].append(r["task_id"])
    return {"compared_tasks": sum(map(len, pairs.values())), **pairs}


def arm_result(root, arm_benchmark, policy, plan, answers, greedy, score):
    tasks = set(plan["prompts"])
    states = checked_states(root, plan["prompts"], arm_benchmark, plan["eos_token_id"])
    if any(r.get("sampling") != policy for r in states.values()):
        raise ValueError(f"State uses another sampling recipe: {root}")
    attempts = []
    attempted_ids = set()
    for path in sorted((root / "attempts").glob("*.json")):
        r = json.loads(path.read_bytes())
        if r["benchmark_id"] != arm_benchmark or not set(r["task_ids"]) <= tasks:
            raise ValueError(f"Inconsistent attempt: {path}")
        attempts.append(r)
        attempted_ids.update(r["task_ids"])
    terminal = {t for t, r in states.items() if r["finish_reason"] in TERMINAL}
    completed_batch_times = [a["elapsed_seconds"] for a in attempts if "elapsed_seconds" in a]
    result = score_single(answers, states, score)
    result.update(benchmark_id=arm_benchmark, sampling=policy,
                  coverage={"target": len(tasks), "terminal": len(terminal),
                            "checkpoints": sorted(states.keys() - terminal),
                            "attempted_without_checkpoint": sorted(attempted_ids - states.keys()),
                            "unattempted": sorted(tasks - (attempted_ids | states.keys()))},
                  attempt_timing={"seconds_recorded": sum(completed_batch_times),
                                  "timed_attempts": len(completed_batch_times), "all_attempts": len(attempts),
                                  "failed_attempt_seconds": sum(a.get("elapsed_seconds", 0) for a in attempts
                                                                if a["status"] in {"oom", "failed"}),
                                  "first_save_seconds_from_allocation": min(
                                      (s["seconds_from_allocation"] for a in attempts for s in a.get("terminal_saves", [])),
                                      default=None)})
    result["paired_to_greedy"] = comparison(result, greedy)
    return result


def main():
    from sampling_stage import read_plan
    from sampling_run import arm_id

    plan = read_plan(FOLDER)
    job = os.environ["SLURM_JOB_ID"]
    if not job.isdecimal():
        raise ValueError("Expected a Slurm allocation ID")
    root = FOLDER.parent / "runs" / FOLDER.name / plan["benchmark_id"]
    allocation = root / "allocations" / job
    native = FOLDER.parent / "arc-flash-eval138-v1"
    protocol = run_path(str(native / "protocol.py"))
    answer_file = native / "answers.scorer-only.jsonl"
    checksum = json.loads((native / "benchmark-recipe.json").read_bytes())["answers_sha256"]
    if hashlib.sha256(answer_file.read_bytes()).hexdigest() != checksum:
        raise ValueError("Native scoring answers changed")
    selected = {task for group in plan["groups"] for task in group}
    answers = [a for a in protocol["rows"](answer_file) if a["task_id"] in selected]
    if len(answers) != 12 or {a["task_id"] for a in answers} != selected:
        raise ValueError("The task scoring cohort differs from the verified pilot")
    greedy = score_single(answers, greedy_states(FOLDER.parent, plan), protocol["score"])
    with (root / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        arms = {slug: arm_result(root / "arms" / slug, arm_id(plan, slug), policy,
                                 plan, answers, greedy, protocol["score"])
                for slug, policy in plan["sampling"].items()}
        common = set.intersection(*[{r["task_id"] for r in arm["tasks"]} for arm in arms.values()])
        common_counts = {slug: sum(r["task_correct"] for r in arm["tasks"] if r["task_id"] in common)
                         for slug, arm in {"greedy": greedy, **arms}.items()}
        result = {"benchmark_id": plan["benchmark_id"], "job": job, "greedy_reference_job": GREEDY_JOB,
                  "greedy": greedy, "arms": arms, "common_completed_task_ids": sorted(common),
                  "solves_on_common_completed_tasks": common_counts}
        publish_snapshot(allocation / "scores.json", result)
    print("BF16_SAMPLING_SCORE", json.dumps({slug: a["summary"] for slug, a in arms.items()}), flush=True)


if __name__ == "__main__":
    main()
