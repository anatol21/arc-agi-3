"""Checkpointed ARC inference adapter for the repository's Slurm campaign scheduler."""
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

CLUSTER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLUSTER_ROOT))
from inference.batching import Journal, TERMINAL, choose_group
from slurm.checkpoint import digest, read_progress, save_item
from arc_worker.decode import decode_batch, prepare_tasks, workload_identity


def settings_from_config(config, base):
    settings = dict(config["arc"])
    settings.setdefault("backend", "causal_lm")
    if settings["backend"] not in {"causal_lm", "qwen_4gpu_bf16"}:
        raise ValueError("Unknown ARC model backend")
    settings.setdefault("tokenizer", settings["model"])
    settings.setdefault("tokenizer_revision", settings["model_revision"])
    for key in ("model", "tokenizer"):
        if not re.fullmatch(r"[0-9a-f]{40}", settings[key + "_revision"]):
            raise ValueError(f"Pin {key}_revision to the exact 40-digit source commit")
        # Hub IDs remain IDs. Explicit local paths are resolved relative to the campaign.
        value = settings[key]
        if value.startswith((".", "/", "~")):
            path = Path(value).expanduser()
            path = (base / path).resolve()
            if not path.is_dir():
                raise ValueError(f"Missing local {key} directory: {path}")
            settings[key] = str(path)
    if settings["backend"] == "qwen_4gpu_bf16":
        path = (base / settings["axolotl_config"]).resolve()
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if checksum != settings["axolotl_config_sha256"]:
            raise ValueError("Axolotl configuration differs from its campaign fingerprint")
        settings["axolotl_config"] = str(path)
    for key in ("context", "max_new_tokens", "checkpoint_tokens"):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f"Expected a positive integer for {key}")
    margin = settings.get("deadline_margin_seconds", 60)
    if type(margin) is not int or margin < 0:
        raise ValueError("deadline_margin_seconds must be a nonnegative integer")
    bands = settings["bands"]
    if (not bands or any(type(n) is not int or n <= 0 for n in bands)
            or sorted(set(bands)) != bands or bands[-1] != settings["context"]):
        raise ValueError("Use increasing positive prompt bands ending at context")
    sampling = settings.get("sampling")
    if sampling is not None:
        if set(sampling) != {"seed", "temperature", "top_p", "top_k"}:
            raise ValueError("Expected seed, temperature, top_p and top_k for sampling")
    return settings


def read_tasks(settings, base):
    raw = (base / settings["tasks_file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != settings["tasks_sha256"]:
        raise ValueError("Task file differs from its campaign fingerprint")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def native_tokenizer(settings):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(settings["tokenizer"], revision=settings["tokenizer_revision"],
                                         local_files_only=True, trust_remote_code=False)


def native_model(settings):
    if settings["backend"] == "qwen_4gpu_bf16":
        from arc_worker.qwen_bf16 import load_model
        return load_model(settings)
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    kwargs = dict(revision=settings["model_revision"], local_files_only=True, trust_remote_code=False)
    config = AutoConfig.from_pretrained(settings["model"], **kwargs)
    if getattr(config, "quantization_config", None):
        raise ValueError("This base-only example expects original model weights")
    placement = settings.get("device_map", "auto" if torch.cuda.is_available() else None)
    model = AutoModelForCausalLM.from_pretrained(settings["model"], config=config, dtype=torch.bfloat16,
                                                 device_map=placement, **kwargs)
    if (getattr(model, "peft_config", None) or getattr(model, "hf_quantizer", None)
            or any("lora_" in name or getattr(p, "quant_state", None) is not None
                   for name, p in model.named_parameters())):
        raise ValueError("Expected an unquantized base model without adapters")
    return model.eval().requires_grad_(False)


def stop_requested(root, margin):
    interrupted = False

    def interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, interrupt)
    end = os.environ.get("SLURM_JOB_END_TIME")
    if os.environ.get("SLURM_JOB_ID") and not end:
        # Slurm versions that omit the env var still publish EndTime on the controller.
        import subprocess
        from datetime import datetime
        record = subprocess.check_output(["scontrol", "-o", "show", "job", os.environ["SLURM_JOB_ID"]],
                                         text=True, timeout=15)
        stamp = next(field[8:] for field in record.split() if field.startswith("EndTime="))
        end = datetime.fromisoformat(stamp).timestamp()
    deadline = float(end) - margin if end else float("inf")
    return lambda: interrupted or (root / "STOP").exists() or time.time() >= deadline


def hardware_identity(model, torch):
    place = getattr(model, "hf_device_map", None)
    devices = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
    return {
        "gpus": [{"name": p.name, "total_bytes": p.total_memory, "capability": [p.major, p.minor]} for p in devices],
        "placement": {k: str(v) for k, v in place.items()} if place else str(model.get_input_embeddings().weight.device),
    }


def mirror(root, campaign_id, run_id, tokenizer, task_id, tokens, reason):
    """Publish scheduling progress only after the authoritative token journal is durable."""
    final = reason in TERMINAL
    payload = {"run_id": run_id, "reason": reason, "tokens": tokens}
    if final:
        payload["text"] = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    save_item(root, campaign_id, task_id, len(tokens), "finished" if final else "partial", payload)


def reconcile(root, campaign_id, journal, tasks, tokenizer):
    states = journal.states()
    progress = read_progress(root, campaign_id, set(tasks))
    if set(states) - set(tasks) or set(progress) - set(states):
        raise ValueError("Campaign and token journal cannot be reconciled")
    for key, state in states.items():
        tokens, reason = state["tokens"], state["reason"]
        if (not tokens or len(tokens) > tasks[key]["budget"] or reason not in TERMINAL | {"checkpoint"}
                or any(type(t) is not int or t < 0 for t in tokens) or tokenizer.eos_token_id in tokens[:-1]):
            raise ValueError(f"Invalid saved tokens or stopping reason for {key}")
        expected = ("eos" if tokens[-1] == tokenizer.eos_token_id else
                    "token_limit" if len(tokens) == tasks[key]["budget"] else "checkpoint")
        if reason != expected:
            raise ValueError(f"Inconsistent stopping reason for {key}")
        previous = progress.get(key)
        if previous:
            old = previous["payload"]
            if (old["run_id"] != journal.run_id or previous["units"] != len(old["tokens"])
                    or tokens[:len(old["tokens"])] != old["tokens"]):
                raise ValueError(f"Campaign progress is ahead of or incompatible with the journal for {key}")
        mirror(root, campaign_id, journal.run_id, tokenizer, key, tokens, reason)
    return states


def run(campaign_path, root, campaign_id, *, load_tokenizer=None, load_model=None, stop=None, hardware=None):
    import torch
    campaign_path, root = Path(campaign_path).resolve(), Path(root).resolve()
    config = json.loads(campaign_path.read_bytes())
    if digest(config) != campaign_id:
        raise ValueError("Campaign config changed")
    settings = settings_from_config(config, campaign_path.parent)
    records = read_tasks(settings, campaign_path.parent)
    tokenizer = (load_tokenizer or native_tokenizer)(settings)
    tasks = prepare_tasks(tokenizer, records, context=settings["context"], max_new_tokens=settings["max_new_tokens"])
    by_id = {task["task_id"]: task for task in tasks}
    if len(by_id) != len(config["items"]) or set(by_id) != set(config["items"]):
        raise ValueError("The campaign items must exactly match the distinct ARC task IDs")
    identity = workload_identity(tasks, model_revision=settings["model_revision"],
                                 tokenizer_revision=settings["tokenizer_revision"], sampling=settings.get("sampling"),
                                 context=settings["context"], max_new_tokens=settings["max_new_tokens"])
    identity["campaign_id"] = campaign_id
    journal = Journal(root / "inference", identity, hardware)
    stop = stop or stop_requested(root, settings.get("deadline_margin_seconds", 60))

    with journal.writer():
        states = reconcile(root, campaign_id, journal, by_id, tokenizer)
        if stop() or all(states.get(key, {}).get("reason") in TERMINAL for key in by_id):
            return
        model = (load_model or native_model)(settings)
        journal.hardware = hardware if hardware is not None else hardware_identity(model, torch)

        def commit(key, tokens, reason):
            mirror(root, campaign_id, journal.run_id, tokenizer, key, tokens, reason)

        while not stop():
            states = journal.states()
            group = choose_group(tasks, states, journal.attempts(), context=settings["context"],
                                 bands=settings["bands"])
            if group is None:
                pending = [key for key in by_id if states.get(key, {}).get("reason") not in TERMINAL]
                if pending:
                    raise RuntimeError(f"Remaining tasks cannot fit on this hardware: {pending}")
                break
            attempt = journal.start(group)
            start = time.monotonic()
            print("ARC_BATCH", group["actual_size"], *group["task_ids"], flush=True)
            try:
                status = decode_batch(model, tokenizer, by_id, group, journal, stop=stop,
                                      checkpoint_tokens=settings["checkpoint_tokens"], sampling=settings.get("sampling"),
                                      on_checkpoint=commit)
            except torch.cuda.OutOfMemoryError:
                status = "oom"
            except Exception:
                journal.finish(attempt, "failed", elapsed_seconds=time.monotonic() - start)
                raise
            journal.finish(attempt, status, elapsed_seconds=time.monotonic() - start)
            if status == "oom":
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    run(Path(os.environ["CAMPAIGN_CONFIG"]), Path(os.environ["CAMPAIGN_DIR"]), os.environ["CAMPAIGN_ID"])
