"""Stage the two sampled decoding variants against the original BF16 debug controls."""
import json
import sys
from pathlib import Path

if __package__:
    from .generation import digest, prepare, write_json
    from .continuation_stage import PILOT_NAME, PILOT_JOB, PILOT_ID, PILOT_LOADER_SHA, file_hash, verified_plan
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generation import digest, prepare, write_json
    from continuation_stage import PILOT_NAME, PILOT_JOB, PILOT_ID, PILOT_LOADER_SHA, file_hash, verified_plan

PRESETS = {
    "low_nucleus": dict(temperature=0.2, top_p=0.95, top_k=0, seed=42),
    "instruct_sampling": dict(temperature=0.7, top_p=0.8, top_k=20, seed=42),
}
PLAN = "sampling-plan.json"


def read_plan(folder):
    plan = verified_plan(Path(folder) / PLAN)
    if plan["package_sha256"] != file_hash(Path(folder) / "package.sha256"):
        raise ValueError("Sampling pilot package differs from its plan")
    return plan


def stage(folder):
    workspace = folder.parent
    pilot = verified_plan(workspace / PILOT_NAME / "pilot-plan.json")
    if pilot["benchmark_id"] != PILOT_ID or [len(g) for g in pilot["groups"]] != [4, 8]:
        raise ValueError("Different greedy pilot identity or cohorts")
    if file_hash(folder / "pilot-loader.json") != PILOT_LOADER_SHA:
        raise ValueError("Expected original BF16 loader receipt")
    if file_hash(workspace / "arc-flash-run-v6/model-verified.json") != pilot["model_manifest_sha256"]:
        raise ValueError("The pinned source weight manifest changed")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(pilot["checkpoint_tokenizer"], local_files_only=True)
    if (tokenizer.eos_token_id, tokenizer.pad_token_id) != (pilot["eos_token_id"], pilot["pad_token_id"]):
        raise ValueError("Tokenizer stopping and padding IDs changed")
    for prompt in pilot["prompts"].values():
        prepare(tokenizer, prompt)
    plan = {k: pilot[k] for k in ("model_path", "model_manifest_sha256", "checkpoint_tokenizer", "groups",
                                   "prompts", "eos_token_id", "pad_token_id")}
    plan.update(generation={**pilot["generation"], "do_sample": True}, pilot_benchmark_id=PILOT_ID,
                pilot_job=PILOT_JOB, sampling=PRESETS, arm_order=list(PRESETS),
                package_sha256=file_hash(folder / "package.sha256"))
    plan["benchmark_id"] = digest(plan)
    path = folder / PLAN
    if path.exists():
        if read_plan(folder) != plan:
            raise ValueError("Different sampling plan already staged")
    else:
        write_json(path, plan)
    return plan


if __name__ == "__main__":
    plan = stage(Path(__file__).resolve().parent)
    print("BF16_SAMPLING_STAGED", plan["benchmark_id"], plan["groups"], plan["sampling"], flush=True)
