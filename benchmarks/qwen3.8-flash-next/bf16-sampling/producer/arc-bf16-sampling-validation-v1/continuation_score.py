"""Post-inference, CPU-only scoring of the disjoint BF16 pilot and continuation."""
import fcntl
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from runpy import run_path

FOLDER = Path(__file__).resolve().parent
if __package__:
    from .generation import CONTEXT, MAX_TOKENS, TERMINAL, digest, load_state
else:
    sys.path.insert(0, str(FOLDER))
    from generation import CONTEXT, MAX_TOKENS, TERMINAL, digest, load_state


def checked_states(root, prompts, benchmark, eos):
    unexpected = {p.stem for p in (root / "states").glob("*.json")} - prompts.keys()
    if unexpected:
        raise ValueError(f"Unexpected task files under {root}: {sorted(unexpected)}")
    states = {}
    for task, prompt in prompts.items():
        ids = prompt["token_ids"]
        prepared = dict(task_id=task, ids=ids, budget=min(MAX_TOKENS, CONTEXT - len(ids)),
                        messages_sha256=prompt["messages_sha256"], prompt_sha256=prompt["prompt_sha256"])
        row = load_state(root, prepared, benchmark, eos)
        if row is not None:
            states[task] = row
    return states


def score_records(answers, native_score, reference, pilot_states, continuation_states, attempted, job):
    if set(pilot_states) & set(continuation_states):
        raise ValueError("A pilot task was repeated in the continuation")
    states = {**pilot_states, **continuation_states}
    all_ids = {a["task_id"] for a in answers}
    if (len(all_ids) != 138 or len(answers) != 138 or sum(len(a["outputs"]) for a in answers) != 185
            or set(reference) != all_ids or not states.keys() <= all_ids):
        raise ValueError("The inputs do not match the held-out 138-task, 185-grid benchmark")
    terminal = {t: row for t, row in states.items() if row["finish_reason"] in TERMINAL}
    summary, details = native_score(answers, {(t, "base"): row for t, row in terminal.items()})
    categories = {name: [] for name in ("both_exact", "bf16_only_exact", "nf4_only_exact", "neither_exact")}
    allocation_counts = Counter()
    allocation_solves = {}
    for result in details:
        task = result["task_id"]
        state, baseline = terminal[task], reference[task]
        exact, prior_exact = result["task_correct"], baseline["task_correct"]
        category = ("both_exact" if prior_exact else "bf16_only_exact") if exact else (
            "nf4_only_exact" if prior_exact else "neither_exact")
        categories[category].append(task)
        source_job = str(state["slurm_job_id"])
        allocation_counts[source_job] += 1
        allocation_solves.setdefault(source_job, [])
        if exact:
            allocation_solves[source_job].append(task)
        response_sha = hashlib.sha256(state["text"].encode()).hexdigest()
        result.update(experiment="pilot" if task in pilot_states else "continuation",
                      job_id=source_job, benchmark_id=state["benchmark_id"], batch_size=state["actual_batch_size"],
                      generated_tokens=state["generated_tokens"], bf16_response_sha256=response_sha,
                      nf4_exact=prior_exact, nf4_generated_tokens=baseline["generated_tokens"],
                      nf4_package=baseline["package"], nf4_response_sha256=baseline["response_sha256"],
                      same_response_text=response_sha == baseline["response_sha256"])
    pending = sorted(all_ids - terminal.keys())
    checkpointed = sorted(states.keys() - terminal.keys())
    attempted = set(attempted) | states.keys()
    unsaved = sorted(attempted - states.keys())
    unattempted = sorted(all_ids - attempted)
    return dict(
        model="unquantized base only", base=summary["arms"]["base"],
        coverage=dict(total=138, completed=len(terminal), pilot_completed=len(pilot_states),
                      continuation_completed=len(terminal) - len(pilot_states), remaining=len(pending),
                      checkpointed=len(checkpointed), attempted_without_saved_tokens=len(unsaved), unattempted=len(unattempted)),
        pending_task_ids=pending, checkpointed_task_ids=checkpointed,
        attempted_without_saved_tokens_task_ids=unsaved, unattempted_task_ids=unattempted,
        allocation=dict(job=job, completed=allocation_counts[job], solved_task_ids=allocation_solves.get(job, [])),
        terminal_responses_by_job=dict(sorted(allocation_counts.items())),
        nf4_reference="Historical first completion per task. Batch size and GPU placement may differ.",
        paired_completed_tasks=categories, tasks=details)


def publish_snapshot(target, result):
    # Publish an entire immutable snapshot. An interrupted write cannot leave a partial scores.json.
    payload = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=target.parent) as temp:
        temp.write(payload)
        temp.flush()
        os.fsync(temp.fileno())
        try:
            os.link(temp.name, target)
        except FileExistsError:
            if target.read_bytes() != payload:
                raise ValueError(f"Existing allocation score differs; preserved {target}")


def main():
    from continuation_stage import read_plan

    plan = read_plan(FOLDER)
    job = os.environ["SLURM_JOB_ID"]
    if not job.isdecimal():
        raise ValueError("Invalid Slurm allocation ID")
    root = FOLDER.parent / "runs" / FOLDER.name / plan["benchmark_id"]
    allocation = root / "allocations" / job
    native = FOLDER.parent / "arc-flash-eval138-v1"
    native_recipe = json.loads((native / "benchmark-recipe.json").read_text())
    native_protocol = run_path(str(native / "protocol.py"))
    answer_path = native / "answers.scorer-only.jsonl"
    if hashlib.sha256(answer_path.read_bytes()).hexdigest() != native_recipe["answers_sha256"]:
        raise ValueError("Native answer file digest mismatch")
    answers = native_protocol["rows"](answer_path)
    reference = json.loads((FOLDER / "nf4-reference.json").read_text())
    pilot_folder = FOLDER.parent / "arc-bf16-base-debug-v1"
    pilot_plan = run_path(str(pilot_folder / "stage.py"))["read_plan"](pilot_folder)
    if pilot_plan["benchmark_id"] != plan["pilot_benchmark_id"] or plan["pilot_job"] != "8487396":
        raise ValueError("The pilot source differs from this continuation's evidence")
    pilot_root = FOLDER.parent / "runs" / pilot_folder.name / pilot_plan["benchmark_id"]
    pilot = checked_states(pilot_root, pilot_plan["prompts"], pilot_plan["benchmark_id"], pilot_plan["eos_token_id"])
    if (set(pilot) != set(plan["excluded"]) or len(pilot) != 12
            or any(row["finish_reason"] not in TERMINAL or str(row["slurm_job_id"]) != "8487396"
                   for row in pilot.values())):
        raise ValueError("Pilot completion evidence changed")
    for task, row in pilot.items():
        if digest(row) != plan["excluded"][task]["state_sha256"]:
            raise ValueError(f"Pilot terminal record differs from staged digest for {task}")
    with (root / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        continuation = checked_states(root, plan["prompts"], plan["benchmark_id"], plan["eos_token_id"])
        attempted = set()
        for path in (root / "attempts").glob("*.json"):
            a = json.loads(path.read_bytes())
            if a["benchmark_id"] != plan["benchmark_id"] or not set(a["task_ids"]) <= plan["prompts"].keys():
                raise ValueError(f"Unexpected generation attempt: {path.name}")
            attempted.update(a["task_ids"])
        result = score_records(answers, native_protocol["score"], reference, pilot, continuation, attempted, job)
        result.update(benchmark_id=plan["benchmark_id"], pilot_benchmark_id=pilot_plan["benchmark_id"])
        publish_snapshot(allocation / "scores.json", result)
    print("BF16_CONTINUATION_SCORE", json.dumps(dict(base=result["base"], coverage=result["coverage"],
                                                       allocation=result["allocation"])), flush=True)


if __name__ == "__main__":
    main()
