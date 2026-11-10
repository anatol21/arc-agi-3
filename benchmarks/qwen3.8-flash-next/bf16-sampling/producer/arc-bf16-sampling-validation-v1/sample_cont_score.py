"""Score each strategy on its own 138 tasks and compare only paired terminal responses."""
import fcntl
import json
import os
from pathlib import Path
from runpy import run_path
import sys

FOLDER = Path(__file__).resolve().parent
if __package__:
    from .sample_cont_stage import read_plan, arm_id, arm_prompts, verified_pilot
    from .continuation_stage import file_hash, verified_plan
    from .continuation_score import checked_states, publish_snapshot
    from .sampling_score import greedy_states, score_single, comparison
    from .generation import TERMINAL, digest
else:
    sys.path.insert(0, str(FOLDER))
    from sample_cont_stage import read_plan, arm_id, arm_prompts, verified_pilot
    from continuation_stage import file_hash, verified_plan
    from continuation_score import checked_states, publish_snapshot
    from sampling_score import greedy_states, score_single, comparison
    from generation import TERMINAL, digest

GREEDY_NAME = "arc-bf16-base-validation-v1"
GREEDY_CONT_ID = "0289db8e1a28334da3dfb3747ebd541481dd7b749656f52a9df4a8a426de7b37"


def greedy_comparator(workspace, pilot, public):
    # Terminal states are atomically published. A concurrent greedy job can add
    # results; this read captures only the completed records visible at scoring time.
    rows = greedy_states(workspace, pilot)
    plan_file = workspace / GREEDY_NAME / "continuation-plan.json"
    if not plan_file.exists():
        return rows
    plan = verified_plan(plan_file)
    if (plan["benchmark_id"] != GREEDY_CONT_ID or set(plan["prompts"]) & rows.keys()
        or any(public[t] != p for t, p in plan["prompts"].items())):
        raise ValueError("Greedy continuation identity or task cohort differs")
    root = workspace / "runs" / GREEDY_NAME / GREEDY_CONT_ID
    new = checked_states(root, plan["prompts"], GREEDY_CONT_ID, plan["eos_token_id"])
    rows.update({t: r for t,r in new.items() if r["finish_reason"] in TERMINAL})
    return rows


def sampled_summary(answers, prior, new, attempted, score_fn):
    if prior.keys() & new.keys():
        raise ValueError("Previously completed sampled task was repeated")
    all_ids = {r["task_id"] for r in answers}
    rows = {**prior, **new}
    if not set(rows) | attempted <= all_ids:
        raise ValueError("Unknown task in sampling continuation")
    terminal = {t for t,r in rows.items() if r["finish_reason"] in TERMINAL}
    result = score_single(answers, rows, score_fn)
    result.update(coverage=dict(total=len(all_ids), pilot_completed=len(prior),
                                newly_completed=len(terminal) - len(prior), completed=len(terminal),
                                remaining=len(all_ids)-len(terminal),
                                checkpointed=sorted(rows.keys()-terminal),
                                attempted_without_checkpoint=sorted(attempted - rows.keys()),
                                unattempted=sorted(all_ids - (rows.keys() | attempted))),
                  solved_task_ids=sorted(r["task_id"] for r in result["tasks"] if r["task_correct"]))
    return result


def paired_coverage(result, other):
    other_ids = {r["task_id"] for r in other["tasks"]}
    return comparison({"tasks":[r for r in result["tasks"] if r["task_id"] in other_ids]}, other)


def main():
    plan = read_plan(FOLDER)
    job = os.environ["SLURM_JOB_ID"]
    if not job.isdecimal():
        raise ValueError("Expected a numeric Slurm job ID")
    workspace = FOLDER.parent
    root = workspace / "runs" / FOLDER.name / plan["benchmark_id"]
    allocation = root / "allocations" / job
    native = workspace / "arc-flash-eval138-v1"
    oracle = native / "answers.scorer-only.jsonl"
    if file_hash(oracle) != json.loads((native / "benchmark-recipe.json").read_bytes())["answers_sha256"]:
        raise ValueError("Native scorer file changed")
    proto = run_path(str(native / "protocol.py"))
    answers = proto["rows"](oracle)
    pilot, prior = verified_pilot(workspace)
    public = {**plan["prompts"], **pilot["prompts"]}
    if (len(answers) != 138 or {r["task_id"] for r in answers} != public.keys()
        or sum(len(r["outputs"]) for r in answers) != 185):
        raise ValueError("Expected the held-out 138 tasks and 185 query grids")
    if pilot["benchmark_id"] != plan["pilot_benchmark_id"] or plan["sampling"] != pilot["sampling"]:
        raise ValueError("Pilot and continuation use different strategies")
    greedy = score_single(answers, greedy_comparator(workspace, pilot, public), proto["score"])
    with (root / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        arms = {}
        for slug in plan["arm_order"]:
            folder, identity = root / "arms" / slug, arm_id(plan, slug)
            prompts = arm_prompts(plan, slug)
            if (plan["excluded"][slug] != {t:digest(r) for t,r in prior[slug].items()}
                or prior[slug].keys() & prompts.keys() or prior[slug].keys() | prompts.keys() != public.keys()):
                raise ValueError(f"Eligible or excluded cohort changed for {slug}")
            rows = checked_states(folder, prompts, identity, plan["eos_token_id"])
            if any(r.get("sampling") != plan["sampling"][slug] for r in rows.values()):
                raise ValueError(f"Wrong sampling policy in {slug} journal")
            attempted = set()
            for path in (folder / "attempts").glob("*.json"):
                r = json.loads(path.read_bytes())
                if r["benchmark_id"] != identity or not set(r["task_ids"]) <= prompts.keys():
                    raise ValueError(f"Attempt identity mismatch: {path}")
                attempted.update(r["task_ids"])
            arm = sampled_summary(answers, prior[slug], rows, attempted, proto["score"])
            arm.update(sampling=plan["sampling"][slug], benchmark_id=identity,
                       paired_to_greedy=paired_coverage(arm, greedy))
            arms[slug] = arm
        a,b = plan["arm_order"]
        paired = paired_coverage(arms[a],arms[b])
        paired = dict(compared_tasks=paired["compared_tasks"], both_exact=paired["both_exact"],
                      neither_exact=paired["neither_exact"], **{a+"_only_exact": paired["sample_only_exact"],
                      b+"_only_exact": paired["greedy_only_exact"]})
        result = dict(benchmark_id=plan["benchmark_id"], job=job, pilot_job=plan["pilot_job"],
                      greedy=greedy, arms=arms, paired_sampled_strategies=paired)
        target = allocation / "scores.json"
        if target.exists():
            # The parallel greedy run can acquire new states between repeat score calls.
            if json.loads(target.read_bytes())["benchmark_id"] != plan["benchmark_id"]:
                raise ValueError("Existing scores belong to another benchmark")
        else:
            publish_snapshot(target, result)
    print("SAMPLED_CONTINUATION_SCORE", json.dumps({a:s["summary"] for a,s in arms.items()}), flush=True)


if __name__ == "__main__":
    main()
