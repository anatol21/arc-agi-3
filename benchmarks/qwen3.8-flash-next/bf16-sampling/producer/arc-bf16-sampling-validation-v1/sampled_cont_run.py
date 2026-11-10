"""Generate the unfinished task/strategy pairs with a single resident BF16 model."""
from dataclasses import dataclass
import fcntl
import json
import os
import sys
import time
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
if __package__:
    from . import continuation_generation as batching, generation as native
else:
    sys.path.insert(0, str(FOLDER))
    import continuation_generation as batching
    import generation as native


@dataclass
class Arm:
    slug: str
    identity: str
    policy: dict
    directory: Path
    tasks: list
    states: dict
    history: list
    prior: set
    attempts: list

    def last_start(self):
        return max((a["started_at"] for a in self.history), default=0)

    def summary(self):
        done = {t for t, row in self.states.items() if row["finish_reason"] in native.TERMINAL}
        blocked = {t for a in self.history if a["status"] == "oom" and a["actual_batch_size"] == 1
                   for t in a["task_ids"]} - done
        return dict(benchmark_id=self.identity, sampling=self.policy, completed=len(done),
                    skipped_completed=len(self.prior), new_completed=len(done - self.prior),
                    pending=len(self.tasks) - len(done), blocked_task_ids=sorted(blocked),
                    successful_batch_sizes=[a["actual_batch_size"] for a in self.attempts if a["status"] == "complete"],
                    oom_batch_sizes=[a["actual_batch_size"] for a in self.attempts if a["status"] == "oom"],
                    context_segment_batch_sizes=[a["actual_batch_size"] for a in self.attempts if a["status"] == "segment_complete"],
                    generation_seconds=sum(a["elapsed_seconds"] for a in self.attempts))


def restore_arm(slug, output, identity, policy, prompts, tokenizer, hardware):
    directory = output / "arms" / slug
    for child in ("states", "attempts"):
        (directory / child).mkdir(parents=True, exist_ok=True)
    tasks = [native.prepare(tokenizer, prompt) for prompt in prompts.values()]
    if len(tasks) != 126 or {t["task_id"] for t in tasks} != set(prompts):
        raise ValueError(f"Unexpected continuation tasks for strategy {slug}")
    if {p.stem for p in (directory / "states").glob("*.json")} - set(prompts):
        raise ValueError(f"Found a task outside the sealed manifest for {slug}")
    states = {}
    for task in tasks:
        row = native.load_state(directory, task, identity, tokenizer.eos_token_id)
        if row is not None:
            if row.get("sampling") != policy:
                raise ValueError(f"Saved response has a different sampling policy: {slug}/{task['task_id']}")
            states[task["task_id"]] = row
    prior = {t for t, r in states.items() if r["finish_reason"] in native.TERMINAL}
    history = batching._history(directory, identity, hardware)
    if any(a.get("strategy") != slug or a.get("sampling") != policy for a in history):
        raise ValueError(f"Attempt receipt strategy mismatch under {directory}")
    return Arm(slug, identity, policy, directory, tasks, states, history, prior, [])


def run_strategies(torch, model, tokenizer, output, start, stop, arms):
    """Alternate successful batches and retries across arms, including after a restart."""
    device = model.get_input_embeddings().weight.device
    hardware = batching.hardware_identity(torch, device)
    output = Path(output)
    with (output / "generation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        workers = [restore_arm(slug, output, identity, policy, prompts, tokenizer, hardware)
                   for slug, identity, policy, prompts in arms]
        while not stop():
            for arm in sorted(workers, key=Arm.last_start):
                group, decision = batching.choose_group(arm.tasks, arm.states, arm.history)
                if group:
                    attempt = batching.execute_attempt(torch, model, tokenizer, group, arm.states, arm.directory,
                        arm.identity, start, stop, hardware, decision, sampling=arm.policy, strategy=arm.slug)
                    arm.history.append(attempt)
                    arm.attempts.append(attempt)
                    break
            else:
                break
        details = {arm.slug: arm.summary() for arm in workers}
        saved_times = [r["seconds_from_allocation"] for arm in workers for a in arm.attempts for r in a["terminal_saves"]]
        elapsed = time.time() - start
        gpu_hours = (torch.cuda.device_count() if device.type == "cuda" else 0) * elapsed / 3600
        new = sum(r["new_completed"] for r in details.values())
        return dict(arms=details, new_completed=new, completed=sum(r["completed"] for r in details.values()),
                    skipped_completed=sum(r["skipped_completed"] for r in details.values()),
                    pending=sum(r["pending"] for r in details.values()), first_saved_seconds=min(saved_times, default=None),
                    blocked_task_ids=[f"{slug}/{tid}" for slug, r in details.items() for tid in r["blocked_task_ids"]],
                    allocation_elapsed_seconds=elapsed, gpu_hours_until_report=gpu_hours,
                    new_responses_per_gpu_hour_until_report=new/gpu_hours if gpu_hours else None)


def main():
    from run import run_allocation
    from sample_cont_stage import arm_id, arm_prompts, read_plan
    plan = read_plan(FOLDER)
    slugs = plan["arm_order"]
    if len(slugs) != 2 or set(slugs) != set(plan["sampling"]):
        raise ValueError("Expected the two original sampling strategies")
    arms = []
    for slug in slugs:
        prompts = arm_prompts(plan, slug)
        if (len(prompts) != 126 or len(plan["excluded"][slug]) != 12 or
                set(prompts) & set(plan["excluded"][slug])):
            raise ValueError("Invalid continuation-only task set")
        arms.append((slug, arm_id(plan, slug), plan["sampling"][slug], prompts))
    expected_loader = json.loads((FOLDER / "pilot-loader.json").read_text())

    def drive(torch, model, tokenizer, output, start, stop):
        path = output / "allocations" / os.environ["SLURM_JOB_ID"] / "verified-loader.json"
        if json.loads(path.read_text()) != expected_loader:
            raise ValueError("Live BF16 loader differs from the sampling pilot")
        return run_strategies(torch, model, tokenizer, output, start, stop, arms)

    run_allocation(plan, drive, scope=dict(sampling=plan["sampling"], eligible={slug: len(prompts) for slug, _, _, prompts in arms}))


if __name__ == "__main__":
    main()
