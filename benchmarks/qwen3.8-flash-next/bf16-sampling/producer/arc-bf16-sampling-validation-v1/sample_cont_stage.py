"""Build the two sampled continuation manifests from verified, answer-free source records."""
import fcntl
from pathlib import Path
import json
import sys

if __package__:
    from .continuation_stage import archived_prompts, verified_plan, file_hash, PILOT_LOADER_SHA
    from .continuation_score import checked_states
    from .generation import digest, TERMINAL, write_json
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from continuation_stage import archived_prompts, verified_plan, file_hash, PILOT_LOADER_SHA
    from continuation_score import checked_states
    from generation import digest, TERMINAL, write_json

PILOT = "arc-bf16-sampling-debug-v1"
JOB = "8488924"
ID = "1798db4eabb88036a1f78c6c14102151638a3e607479600d431eafa57b237d9c"
# Seals of task -> terminal-state digests exported from the completed debug allocation.
STATE_SEALS = {
    "low_nucleus": "05ecc5597fe258a51dbbf6e312e02ab48f837d7b4c063f2d1cf319ba1643674f",
    "instruct_sampling": "3d5ddd79a367db6be64fbcf145f72185324f5cd6e94e5fb9c282324141fca6e4",
}
PLAN = "sample-cont-plan.json"


def arm_id(plan, slug):
    return digest({"pilot": plan["benchmark_id"], "arm": slug, "sampling": plan["sampling"][slug]})


def arm_prompts(plan, slug):
    return {t: plan["prompts"][t] for t in plan["eligible_ids"][slug]}


def read_plan(folder):
    folder = Path(folder)
    plan = verified_plan(folder / PLAN)
    if plan["package_sha256"] != file_hash(folder / "package.sha256"):
        raise ValueError("Continuation package changed after staging")
    return plan


def verified_pilot(workspace):
    pilot = verified_plan(workspace / PILOT / "sampling-plan.json")
    if pilot["benchmark_id"] != ID or set(pilot["arm_order"]) != STATE_SEALS.keys():
        raise ValueError("Unexpected sampled pilot identity")
    root = workspace / "runs" / PILOT / ID
    if file_hash(root / "allocations" / JOB / "verified-loader.json") != PILOT_LOADER_SHA:
        raise ValueError("Pilot precision or placement changed")
    states = {}
    for slug in pilot["arm_order"]:
        rows = checked_states(root / "arms" / slug, pilot["prompts"], arm_id(pilot, slug), pilot["eos_token_id"])
        if (len(rows) != 12 or digest({t: digest(r) for t, r in rows.items()}) != STATE_SEALS[slug]
            or any(r["finish_reason"] not in TERMINAL or r["slurm_job_id"] != JOB
                   or r["sampling"] != pilot["sampling"][slug] for r in rows.values())):
            raise ValueError(f"Pilot terminal state differs for {slug}")
        states[slug] = rows
    return pilot, states


def build(folder):
    from transformers import AutoTokenizer
    folder = Path(folder).resolve()
    workspace = folder.parent
    pilot, states = verified_pilot(workspace)
    source = workspace / "arc-flash-eval138-v1"
    recipe = json.loads((source / "benchmark-recipe.json").read_bytes())
    if (pilot["generation"] != {**recipe["generation"], "do_sample": True}
        or recipe["checkpoint"] != pilot["checkpoint_tokenizer"]
        or file_hash(workspace / "arc-flash-run-v6/model-verified.json") != pilot["model_manifest_sha256"]
        or file_hash(folder / "pilot-loader.json") != PILOT_LOADER_SHA):
        raise ValueError("Source model, loader, generation or tokenizer identity changed")
    tokenizer = AutoTokenizer.from_pretrained(pilot["checkpoint_tokenizer"], local_files_only=True)
    if (tokenizer.eos_token_id, tokenizer.pad_token_id) != (pilot["eos_token_id"], pilot["pad_token_id"]):
        raise ValueError("EOS/PAD identity changed")
    public = archived_prompts(source, workspace / "arc-flash-eval138-util-v1", tokenizer)
    if any(public[t] != p for t, p in pilot["prompts"].items()):
        raise ValueError("Sampled pilot prompts differ from public validation prompts")
    ordered = sorted(public, key=lambda t: (public[t]["prompt_tokens"], t))
    excluded, eligible = {}, {}
    for slug, rows in states.items():
        excluded[slug] = {task: digest(r) for task, r in rows.items()}
        eligible[slug] = [t for t in ordered if t not in rows]
        if len(eligible[slug]) != 126:
            raise ValueError("Unexpected number of unfinished task/strategy pairs")
    union = set().union(*eligible.values())
    plan = {k: pilot[k] for k in ("model_path", "model_manifest_sha256", "checkpoint_tokenizer", "eos_token_id",
                                   "pad_token_id", "generation", "sampling", "arm_order")}
    plan.update(pilot_job=JOB, pilot_benchmark_id=ID, eligible_ids=eligible, excluded=excluded,
                prompts={t: public[t] for t in ordered if t in union}, package_sha256=file_hash(folder / "package.sha256"))
    plan["benchmark_id"] = digest(plan)
    with (folder / ".stage.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder / PLAN).exists():
            if read_plan(folder) != plan:
                raise ValueError("An incompatible continuation has already been staged")
        else:
            write_json(folder / PLAN, plan)
    return plan


if __name__ == "__main__":
    plan = build(Path(__file__).resolve().parent)
    print("SAMPLED_CONTINUATION_STAGED", plan["benchmark_id"],
          {slug: len(plan["eligible_ids"][slug]) for slug in plan["arm_order"]}, flush=True)
