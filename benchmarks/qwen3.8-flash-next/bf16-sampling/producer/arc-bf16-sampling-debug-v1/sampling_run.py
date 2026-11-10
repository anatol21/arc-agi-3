"""Exercise two sampled decoders on the BF16 pilot cohorts, sharing one live model."""
import gc
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

FOLDER = Path(__file__).resolve().parent
if __package__:
    from . import generation as native
else:
    sys.path.insert(0, str(FOLDER))
    import generation as native


def arm_id(plan, slug):
    return native.digest({"pilot": plan["benchmark_id"], "arm": slug, "sampling": plan["sampling"][slug]})


def measure_batch(torch, model, tokenizer, group, states, folder, identity, start, stop,
                  segment_end, slug, settings, cohort_size):
    device = model.get_input_embeddings().weight.device
    attempt = dict(attempt_id=uuid.uuid4().hex, benchmark_id=identity,
                   slurm_job_id=os.environ.get("SLURM_JOB_ID"), started_at=time.time(),
                   strategy=slug, sampling=settings, requested_cohort=cohort_size,
                   actual_batch_size=len(group), task_ids=[t["task_id"] for t in group],
                   terminal_saves=[], status="running", time_slice_end=segment_end)
    path = folder / "attempts" / (attempt["attempt_id"] + ".json")
    native.write_json(path, attempt)
    print("SAMPLE_BATCH_START", json.dumps(attempt), flush=True)

    def end_call():
        return stop() or time.time() >= segment_end

    if device.type == "cuda":
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
    try:
        native.generate_group(torch, model, tokenizer, group, states, folder, identity,
                              start, end_call, attempt, sampling=settings)
        complete = all(states.get(t["task_id"], {}).get("finish_reason") in native.TERMINAL for t in group)
        attempt["status"] = "complete" if complete else "checkpointed"
    except torch.cuda.OutOfMemoryError as err:
        attempt.update(status="oom", error=str(err), retry_cap=len(group) // 2)
    except Exception as err:
        attempt.update(status="failed", error=repr(err))
        raise
    finally:
        attempt["elapsed_seconds"] = time.time() - attempt["started_at"]
        attempt["memory"] = native.memory(torch, device)
        native.write_json(path, attempt)
        print("SAMPLE_BATCH_END", json.dumps(attempt), flush=True)
    if attempt["status"] == "oom":
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return attempt


def run_strategies(torch, model, tokenizer, plan, output, start, end, stop):
    groups = [[native.prepare(tokenizer, plan["prompts"][task]) for task in cohort]
              for cohort in plan["groups"]]
    tasks = [task for cohort in groups for task in cohort]
    if list(map(len, groups)) != [4, 8] or len({t["task_id"] for t in tasks}) != 12:
        raise ValueError("Expected the original disjoint BF16 cohorts of four and eight")
    settings = plan["sampling"]
    slugs = plan["arm_order"]
    if (len(slugs) != 2 or len(set(slugs)) != 2 or set(slugs) != set(settings)
            or any(not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]*", s) for s in slugs)):
        raise ValueError("Expected two distinct named sampled policies")
    cells = [(s, g) for g in range(len(groups)) for s in (slugs if g == 0 else reversed(slugs))]
    states, roots, caps, already_finished, attempts, blocked = {}, {}, {}, {}, {}, {}
    for slug in slugs:
        root = roots[slug] = Path(output) / "arms" / slug
        for child in ("states", "attempts"):
            (root / child).mkdir(parents=True, exist_ok=True)
        identity = arm_id(plan, slug)
        states[slug] = {}
        for task in tasks:
            row = native.load_state(root, task, identity, tokenizer.eos_token_id)
            if row is not None:
                if row.get("sampling") != settings[slug]:
                    raise ValueError(f"Saved sampling policy differs for {slug}/{task['task_id']}")
                states[slug][task["task_id"]] = row
        already_finished[slug] = sum(row["finish_reason"] in native.TERMINAL for row in states[slug].values())
        attempts[slug] = []
        caps[slug] = 8
        blocked[slug] = set()
        for receipt in (root / "attempts").glob("*.json"):
            previous = json.loads(receipt.read_text())
            if previous["benchmark_id"] != identity:
                raise ValueError(f"Unexpected sampling attempt in {root}")
            if previous["status"] == "oom":
                caps[slug] = min(caps[slug], max(1, previous["retry_cap"]))
                if previous["retry_cap"] == 0:
                    blocked[slug].update(previous["task_ids"])

    def pending(slug, cohort):
        return [t for t in groups[cohort]
                if t["task_id"] not in blocked[slug]
                and states[slug].get(t["task_id"], {}).get("finish_reason") not in native.TERMINAL]

    # Allocate a first opportunity to every cell. A long-running cell preserves its
    # prefixes at the slice boundary while short calls release their time immediately.
    # Cells with completed rows have no claim on later passes.
    queue = list(cells)
    while queue and not stop() and time.time() < end:
        runnable = [(slug, cohort) for slug, cohort in queue if caps[slug] and pending(slug, cohort)]
        if not runnable:
            break
        for index, (slug, cohort) in enumerate(runnable):
            now = time.time()
            if stop() or now >= end:
                break
            boundary = now + (end - now) / (len(runnable) - index)
            while caps[slug] and time.time() < boundary and not stop():
                group = pending(slug, cohort)[:caps[slug]]
                if not group:
                    break
                attempt = measure_batch(torch, model, tokenizer, group, states[slug], roots[slug],
                                        arm_id(plan, slug), start, stop, boundary, slug, settings[slug], len(groups[cohort]))
                attempts[slug].append(attempt)
                if attempt["status"] == "oom":
                    caps[slug] = max(1, attempt["retry_cap"])
                    if attempt["retry_cap"] == 0:
                        blocked[slug].update(attempt["task_ids"])
                if attempt["status"] == "checkpointed":
                    break
        queue = [(slug, cohort) for slug, cohort in runnable if caps[slug] and pending(slug, cohort)]

    by_arm = {}
    for slug in slugs:
        rows = states[slug]
        completed = sorted(t for t, row in rows.items() if row["finish_reason"] in native.TERMINAL)
        by_arm[slug] = dict(benchmark_id=arm_id(plan, slug), sampling=settings[slug],
                            completed_task_ids=completed, completed=len(completed),
                            new_completed=len(completed) - already_finished[slug],
                            checkpointed_task_ids=sorted(set(rows) - set(completed)),
                            blocked_singleton_oom_task_ids=sorted(blocked[slug]),
                            generated_attempt_seconds=sum(a["elapsed_seconds"] for a in attempts[slug]),
                            actual_batch_sizes=[a["actual_batch_size"] for a in attempts[slug]],
                            oom_batch_sizes=[a["actual_batch_size"] for a in attempts[slug] if a["status"] == "oom"])
    return dict(arms=by_arm, new_completed=sum(a["new_completed"] for a in by_arm.values()),
                first_saved_seconds=min((s["seconds_from_allocation"] for arm in attempts.values()
                                        for a in arm for s in a["terminal_saves"]), default=None))


def main():
    from run import run_allocation
    from sampling_stage import read_plan

    plan = read_plan(FOLDER)
    pilot_loader = json.loads((FOLDER / "pilot-loader.json").read_text())

    def drive(torch, model, tokenizer, output, start, stop):
        allocation = output / "allocations" / os.environ["SLURM_JOB_ID"]
        verified = json.loads((allocation / "verified-loader.json").read_text())
        if verified != pilot_loader:
            raise ValueError("The live loader differs from the archived greedy BF16 pilot")
        end = json.loads((allocation / "status.json").read_text())["allocation_end"] - 60
        return run_strategies(torch, model, tokenizer, plan, output, start, end, stop)

    run_allocation(plan, drive, scope={"groups": plan["groups"], "sampling": plan["sampling"]})


if __name__ == "__main__":
    main()
