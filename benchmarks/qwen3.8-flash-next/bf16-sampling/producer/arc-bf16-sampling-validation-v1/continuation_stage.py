"""Stage the remaining held-out prompts against the verified BF16 pilot, on CPU."""
import fcntl
import hashlib
import json
from pathlib import Path
import sys

if __package__:
    from .generation import CONTEXT, MAX_TOKENS, TERMINAL, digest, load_state, prepare, write_json
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generation import CONTEXT, MAX_TOKENS, TERMINAL, digest, load_state, prepare, write_json


PILOT_NAME = "arc-bf16-base-debug-v1"
PILOT_JOB = "8487396"
PILOT_ID = "b0bc62947697ba1fa5cfbc3ba6d66bbf1bed47dea6f115a05eb8ddfb080a5d0a"
# Digests from the locally collected pilot records and the archived public input manifest.
PILOT_STATES_SHA = "b9c8033c7d66a1428cea5cd0380b51ef1944905c315720d8dfc09f3576182d6b"
PILOT_LOADER_SHA = "2d6ef31c9491f452e7c8c5b6aea12d51b3a2b072e9ae234b8967da03e1b3f670"
PUBLIC_PROMPTS_SHA = "114c0416d36df673e9105458525410bc811911497eff520818879ba2f8794e8d"
PUBLIC_LENGTHS_SHA = "d099d009368166f49545a2edb0a06229be6a280f67047369bfc72d444a1a4cad"
PLAN = "continuation-plan.json"


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_plan(path):
    plan = json.loads(path.read_bytes())
    if plan["benchmark_id"] != digest({k: v for k, v in plan.items() if k != "benchmark_id"}):
        raise ValueError(f"Plan checksum failed: {path}")
    return plan


def read_plan(folder):
    folder = Path(folder)
    plan = verified_plan(folder / PLAN)
    if plan["package_sha256"] != file_hash(folder / "package.sha256"):
        raise ValueError("Continuation package differs from its staged plan")
    return plan


def archived_prompts(native, util, tokenizer):
    # These files contain the public user message and tokenizer outputs only.
    recipe = json.loads((native / "benchmark-recipe.json").read_bytes())
    prompt_file = native / "prompts.jsonl"
    if file_hash(prompt_file) != recipe["prompts_sha256"]:
        raise ValueError("Archived native prompt file changed")
    rows = [json.loads(line) for line in prompt_file.read_text().splitlines() if line.strip()]
    if len(rows) != 138 or digest(rows) != PUBLIC_PROMPTS_SHA:
        raise ValueError("The held-out task IDs or their exact user prompts changed")
    token_file = util / "tokenized-prompts.json"
    token_receipt = json.loads((util / "READY.json").read_bytes())["files_sha256"][token_file.name]
    if file_hash(token_file) != token_receipt:
        raise ValueError("Archived NF4 token file changed")
    tokens = json.loads(token_file.read_bytes())
    if set(tokens) != {row["task_id"] for row in rows} or len(tokens) != 138:
        raise ValueError("Archived prompt and token task IDs differ")
    if digest({t: len(ids) for t, ids in tokens.items()}) != PUBLIC_LENGTHS_SHA:
        raise ValueError("Archived lengths differ from the measured NF4 profiles")

    prompts = {}
    for row in rows:
        key = row["task_id"]
        ids = tokens[key]
        prompt = dict(row, prompt_tokens=len(ids), token_ids=ids,
                      prompt_sha256=digest(ids), messages_sha256=digest(row["messages"]))
        # Exercise the live checkpoint chat template with the original decoding flags.
        prepare(tokenizer, prompt)
        prompts[key] = prompt
    return prompts


def pilot_exclusions(pilot, output, prompts, tokenizer):
    groups = pilot["groups"]
    ids = [task for group in groups for task in group]
    if (pilot["benchmark_id"] != PILOT_ID or list(map(len, groups)) != [4, 8] or len(set(ids)) != 12
            or {p.stem for p in (output / "states").glob("*.json")} != set(ids)):
        raise ValueError("Expected the twelve terminal states from the original pilot")
    if file_hash(output / "allocations" / PILOT_JOB / "verified-loader.json") != PILOT_LOADER_SHA:
        raise ValueError("Pilot's unquantized base-loader receipt differs from the verified archive")
    excluded = {}
    for task in ids:
        prompt = prompts[task]
        if pilot["prompts"][task] != prompt:
            raise ValueError(f"Pilot prompt differs from archived NF4 prompt: {task}")
        expected = dict(task_id=task, ids=prompt["token_ids"], prompt_sha256=prompt["prompt_sha256"],
                        messages_sha256=prompt["messages_sha256"], budget=min(MAX_TOKENS, CONTEXT - prompt["prompt_tokens"]))
        row = load_state(output, expected, PILOT_ID, tokenizer.eos_token_id)
        if row is None or row["finish_reason"] not in TERMINAL or row["slurm_job_id"] != PILOT_JOB:
            raise ValueError(f"Missing completed pilot response from job {PILOT_JOB}: {task}")
        text = tokenizer.decode(row["completion_token_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if row["text"] != text:
            raise ValueError(f"Pilot saved text does not match its completion tokens: {task}")
        excluded[task] = dict(job=PILOT_JOB, state_sha256=digest(row), generated_tokens=row["generated_tokens"],
                              finish_reason=row["finish_reason"])
    if digest({t: item["state_sha256"] for t, item in excluded.items()}) != PILOT_STATES_SHA:
        raise ValueError("Pilot states differ from those collected after job 8487396")
    return excluded


def build(folder):
    folder = Path(folder).resolve()
    workspace = folder.parent
    pilot = verified_plan(workspace / PILOT_NAME / "pilot-plan.json")
    if pilot["benchmark_id"] != PILOT_ID:
        raise ValueError("Unexpected BF16 pilot identity")
    native = workspace / "arc-flash-eval138-v1"
    recipe = json.loads((native / "benchmark-recipe.json").read_bytes())
    if (recipe["generation"] != pilot["generation"] or recipe["checkpoint"] != pilot["checkpoint_tokenizer"]
            or pilot["model_path"] != str(workspace / "models/qwen3.8-flash-next")
            or pilot["model_manifest_sha256"] != file_hash(workspace / "arc-flash-run-v6/model-verified.json")):
        raise ValueError("Source weights, generation configuration or tokenizer path changed since the pilot")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(pilot["checkpoint_tokenizer"], local_files_only=True)
    if any(getattr(tokenizer, key) != pilot[key] or recipe[key] != pilot[key] for key in ("eos_token_id", "pad_token_id")):
        raise ValueError("EOS or PAD token differs from the pilot")
    prompts = archived_prompts(native, workspace / "arc-flash-eval138-util-v1", tokenizer)
    excluded = pilot_exclusions(pilot, workspace / "runs" / PILOT_NAME / PILOT_ID, prompts, tokenizer)
    ids = sorted(prompts.keys() - excluded.keys(), key=lambda t: (prompts[t]["prompt_tokens"], t))
    if len(ids) != 126:
        raise ValueError("Expected exactly 126 unfinished task/model pairs")
    settings = {key: pilot[key] for key in ("model_path", "model_manifest_sha256", "checkpoint_tokenizer",
                                             "eos_token_id", "pad_token_id", "generation")}
    plan = dict(settings, pilot_job=PILOT_JOB, pilot_benchmark_id=PILOT_ID, eligible_ids=ids,
                excluded=excluded, prompts={t: prompts[t] for t in ids},
                package_sha256=file_hash(folder / "package.sha256"))
    plan["benchmark_id"] = digest(plan)
    with (folder / ".stage.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder / PLAN).exists():
            if read_plan(folder) != plan:
                raise ValueError("A different immutable continuation is already staged")
        else:
            write_json(folder / PLAN, plan)
    return plan


if __name__ == "__main__":
    plan = build(Path(__file__).resolve().parent)
    print("BF16_CONTINUATION_STAGED", json.dumps(dict(benchmark_id=plan["benchmark_id"],
          verified_pilot_responses=len(plan["excluded"]), remaining=len(plan["eligible_ids"]))), flush=True)
