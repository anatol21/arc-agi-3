"""Remote RPC and Slurm worker. This module and its remote dependencies use only stdlib."""
from __future__ import annotations
import argparse
import copy
import getpass
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from core import ConfigError, digest, duration_seconds, estimate_plans, experiment_identity, normalize_config, normalize_profile, utc_now
from sdk import atomic_json, file_hash, lock, read_json

FILES = ("runtime.py", "core.py", "sdk.py", "demo_adapter.py", "training_example.py", "workspace.py", "live_generation.py")
TERMINAL = {"complete", "failed", "cancelled", "needs_attention", "budget_reached", "partial_checkpointed"}

def run_path(root, run_id):
    if not re.fullmatch(r"[a-f0-9]{32}", run_id): raise ValueError("Invalid run identifier")
    return Path(root) / "cluster-ui" / run_id

def load_run(path):
    m = read_json(path / "manifest.json")
    if not m: raise ValueError("Run not found")
    if m.get('id') != path.name or not re.fullmatch(r'[a-f0-9]{32}', path.name):
        raise ValueError('Run identifier does not match its folder')
    normalize_config(m['config'])
    normalize_profile(m['profile'])
    for wave in m.get('waves', []):
        for key in ('job_id', 'dependency'):
            if wave.get(key) is not None and not re.fullmatch(r'[0-9]+', str(wave[key])):
                raise ValueError('Invalid recorded scheduler job identifier')
    if experiment_identity(m["config"]) != m["identity"]: raise ValueError("Immutable experiment identity changed")
    for name, expected in m["runtime_files"].items():
        if name not in FILES: raise ValueError('Unknown runtime file in manifest')
        if file_hash(path / name) != expected: raise ValueError("Runtime source changed: " + name)
    return m

def clock_seconds(text):
    if text in ("UNLIMITED", "INFINITE"): return math.inf
    if "-" in text:
        days, text = text.split("-", 1)
    else: days = 0
    parts = [int(x) for x in text.split(":")]
    seconds = parts[0] * 60 if len(parts) == 1 else parts[0] * 60 + parts[1] if len(parts) == 2 else parts[0] * 3600 + parts[1] * 60 + parts[2]
    return int(days) * 86400 + seconds

def command(argv, timeout=15):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SBATCH_", "SLURM_"))}
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
        return {"code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"code": -1, "stdout": "", "stderr": str(error)}

def sbatch_args(m, part, indices, concurrency, token, script, dependency=None):
    cfg, profile = m["config"], m["profile"]
    args = ["sbatch", "--parsable", "--job-name=wm-" + token, "--comment=wm:" + m["id"] + ":" + token,
            "--partition=" + part["name"], "--nodes=1", "--ntasks=1", "--time=" + part["time_limit"],
            "--array=" + ",".join(map(str, indices)) + "%" + str(concurrency),
            "--signal=B:USR1@" + str(int(profile["checkpoint_margin_seconds"])),
            "--chdir=" + profile["remote_root"], "--output=" + str(script.parent / "slurm-%A_%a.log"),
            "--error=" + str(script.parent / "slurm-%A_%a.log")]
    if cfg["gpus"]: args.append("--gres=gpu:" + str(cfg["gpus"]))
    if cfg["cpus"]: args.append("--cpus-per-task=" + str(cfg["cpus"]))
    if "wall_hours" in cfg["budget"]:
        deadline = m["created_epoch"] + cfg["budget"]["wall_hours"] * 3600
        args.append("--deadline=" + datetime.fromtimestamp(deadline).strftime("%Y-%m-%dT%H:%M:%S"))
    for key in ("account", "qos"):
        if profile[key]: args.append("--" + key + "=" + profile[key])
    if dependency: args.append("--dependency=afterany:" + str(dependency))
    return args + [str(script)]

class Scheduler:
    def __init__(self, root, profile):
        self.root, self.profile = Path(root), normalize_profile(profile)

    def discover(self, cfg):
        if self.profile["simulation"]:
            parts = [{**p, "available": True, "queue_seconds": 0.0 if p["role"] != "normal" else 14400.0, "usable_seconds": 2.2, "reason": "Simulator"} for p in self.profile["partitions"]]
            return {"partitions": parts, "availability": "Simulated GPU allocations; no cluster connection", "associations": "", "qos": "", "checked_at": utc_now()}
        idle = command(["sinfo_t_idle"])
        assoc = command(["sacctmgr", "-nP", "show", "assoc", "where", "user=" + getpass.getuser(), "format=Account,QOS,MaxJobs,MaxSubmitJobs"])
        qos = command(["sacctmgr", "-nP", "show", "qos", self.profile["qos"], "format=Name,MaxJobsPU,MaxSubmitJobsPU"]) if self.profile["qos"] else {"stdout": ""}
        caps = []
        for output, columns in ((assoc["stdout"], (2, 3)), (qos["stdout"], (1, 2))):
            for line in output.splitlines():
                fields = line.split("|")
                if fields and fields[0] not in (self.profile["account"], self.profile["qos"]): continue
                for i in columns:
                    if len(fields) > i and fields[i].isdigit() and int(fields[i]) > 0: caps.append(int(fields[i]))
        parts = []
        for original in self.profile["partitions"]:
            part = dict(original)
            info = command(["scontrol", "-o", "show", "partition", part["name"]])
            values = dict(re.findall(r"(\w+)=([^\s]+)", info["stdout"]))
            part.update(available=info["code"] == 0 and values.get("State") == "UP", queue_seconds=None, reason=info["stderr"].strip(), limits=values)
            if caps: part["concurrency_limit"] = min(part["concurrency_limit"], *caps)
            if part["role"] == "development": part["concurrency_limit"] = min(1, part["concurrency_limit"])
            try:
                if clock_seconds(values.get("MaxTime", "0")) < duration_seconds(part["time_limit"]):
                    part.update(available=False, reason="Requested wall time exceeds live partition limit")
            except (ValueError, ConfigError):
                part.update(available=False, reason="Cannot establish live partition wall-time limit")
            if part["available"]:
                args = ["sbatch", "--test-only", "--partition=" + part["name"], "--nodes=1", "--ntasks=1", "--time=" + part["time_limit"], "--wrap=true"]
                if cfg["gpus"]: args.append("--gres=gpu:" + str(cfg["gpus"]))
                if cfg["cpus"]: args.append("--cpus-per-task=" + str(cfg["cpus"]))
                for key in ("account", "qos"):
                    if self.profile[key]: args.append("--" + key + "=" + self.profile[key])
                test = command(args)
                output = test["stdout"] + test["stderr"]
                part["available"] = test["code"] == 0
                part["reason"] = output.strip()
                match = re.search(r"(?:start at|StartTime[=:])\s*([0-9T:+-]+)", output)
                if match:
                    try: part["queue_seconds"] = max(0, datetime.fromisoformat(match[1]).timestamp() - time.time())
                    except ValueError: pass
            part["usable_seconds"] = max(0, duration_seconds(part["time_limit"]) - self.profile["checkpoint_margin_seconds"])
            parts.append(part)
        return {"partitions": parts, "availability": idle["stdout"] or idle["stderr"], "associations": assoc["stdout"] or assoc.get("stderr", ""), "qos": qos["stdout"], "checked_at": utc_now()}

    def reconcile(self, token):
        # Never retry an ambiguous submission when neither accounting source can establish its identity.
        active = command(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%j"])
        history = command(["sacct", "-nP", "-X", "--starttime", "now-7days", "--name", "wm-" + token, "--format=JobIDRaw,JobName%100"])
        ids = set()
        for output in (active["stdout"], history["stdout"]):
            for line in output.splitlines():
                fields = line.strip().split("|")
                if len(fields) > 1 and fields[1] == "wm-" + token and re.fullmatch(r"[0-9]+(?:_[0-9]+)?", fields[0]):
                    ids.add(fields[0].split("_")[0])
        return sorted(ids), active["code"] == 0 and history["code"] == 0

    def submit(self, m, wave, script):
        if self.profile["simulation"]:
            log = (script.parent / "controller.log").open("ab")
            p = subprocess.Popen([sys.executable, str(script.parent.parent.parent / "runtime.py"), str(self.root), "simulate", m["id"], str(wave["number"])], stdout=log, stderr=log, start_new_session=True)
            log.close()
            return str(p.pid), ""
        result = command(sbatch_args(m, wave["partition"], wave["indices"], wave["concurrency"], wave["token"], script, wave.get("dependency")), timeout=30)
        match = re.fullmatch(r"(\d+)(?:;[^\n]+)?\s*", result["stdout"])
        if result["code"] == 0 and match: return match[1], ""
        ids, _ = self.reconcile(wave["token"])
        if len(ids) == 1: return ids[0], ""
        return None, "Submission not confirmed; reconcile before resuming. " + result["stderr"]

    def status(self, job_id):
        if self.profile["simulation"]:
            return {"state": "SIMULATED", "reason": "Inspect worker progress", "start": None}
        result = command(["squeue", "-h", "-j", str(job_id), "-o", "%i|%T|%R|%S"])
        rows = [line.split("|") for line in result["stdout"].splitlines() if line]
        if rows: return {"state": ", ".join(sorted({r[1] for r in rows})), "reason": "; ".join(sorted({r[2] for r in rows})), "rows": rows}
        accounting = command(["sacct", "-nP", "-X", "-j", str(job_id), "--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES"])
        states = [r.split("|")[1].split()[0].rstrip("+") for r in accounting["stdout"].splitlines() if len(r.split("|")) > 1]
        terminal = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
        confirmed = accounting["code"] == 0 and bool(states) and all(s in terminal for s in states)
        return {"state": "TERMINAL" if confirmed else "UNKNOWN", "reason": result["stderr"], "accounting": accounting["stdout"]}

class BudgetReached(ValueError):
    pass


def choose_plan(m, discovery, remaining, *, worker_count=None):
    profile, cfg = m["profile"], m["config"]
    dev_allowed = bool(profile["dev_permission"]) and cfg["kind"] in profile["dev_run_kinds"]
    parts = [p for p in discovery["partitions"] if p["available"]]
    normal = [p for p in parts if p["role"] == "normal" and duration_seconds(p["time_limit"]) >= profile["normal_min_seconds"]]
    fast_normal = [p for p in normal if p["queue_seconds"] is not None and p["queue_seconds"] <= profile["queue_threshold_seconds"]]
    if fast_normal: allowed = fast_normal
    elif normal and all(p["queue_seconds"] is not None and p["queue_seconds"] > profile["queue_threshold_seconds"] for p in normal):
        allowed = [p for p in parts if p["role"] == "short" or (p["role"] == "development" and dev_allowed)] or normal
    else:
        allowed = normal or [p for p in parts if p["role"] == "short" or (p["role"] == "development" and dev_allowed)]
    if not allowed:
        raise ValueError("No eligible partition; inspect discovery results and profile")
    budget = cfg.get("budget", {})
    count = cfg.get("workers",1) if worker_count is None else worker_count
    now = time.time()
    used = sum(w["reserved_gpu_hours"] for w in m.get("waves", []) if w["status"] == "submitted")
    def affordable(part):
        dev = part["role"] == "development"
        slots = 1 if dev else count
        parallel = 1 if dev else min(cfg["concurrency"], part.get("concurrency_limit",1))
        seconds = duration_seconds(part["time_limit"])
        cost = used + seconds * cfg.get("gpus",0) * slots / 3600
        end = now + seconds * math.ceil(slots / parallel)
        return (("gpu_hours" not in budget or cost <= budget["gpu_hours"])
                and ("wall_hours" not in budget or end <= m.get("created_epoch",now) + budget["wall_hours"] * 3600))
    candidates = [p for p in allowed if affordable(p)]
    if not candidates and allowed != normal:
        candidates = [p for p in normal if affordable(p)]
    plans = estimate_plans(cfg, candidates, m.get("history", []), remaining)
    if not plans:
        raise BudgetReached("No eligible partition fits the remaining budget")
    return plans[0]

def remaining_indices(path, m):
    result = []
    for index in range(m["config"]["workers"]):
        state = read_json(path / "workers" / str(index) / "result.json", {})
        if state.get("state") == "complete" and m["config"]["resume_capable"]:
            prior_wave = next((w for w in m["waves"] if w["token"] == state.get("attempt")), None)
            if prior_wave is None: raise ValueError("Complete receipt belongs to an unknown allocation")
            validate_result(path, m, prior_wave, index, state)
        elif state.get("state") != "complete": result.append(index)
    return result

def new_wave(path, m, dependency=None):
    if (path / "STOP").exists(): raise ValueError("Run is stopped")
    if any(w["status"] in ("intent", "uncertain") for w in m["waves"]):
        raise ValueError("An uncertain submission must be reconciled before another wave")
    control = read_json(path / "control.json")
    cfg = {**m["config"], "concurrency": control["concurrency"]}
    scheduler = Scheduler(path.parent.parent, m["profile"])
    discovery = scheduler.discover(cfg)
    m["discovery"] = discovery
    done = sum(read_json(path / "workers" / str(i) / "result.json", {}).get("completed", 0) for i in range(cfg["workers"]))
    indices = remaining_indices(path, m)
    if not indices:
        m["state"] = "complete"
        atomic_json(path / "manifest.json", m)
        return m
    cfg["budget"] = control["budget"]
    try:
        selection = choose_plan({**m,"config":cfg}, discovery, max(0,cfg["task_count"]-done),
                                worker_count=len(indices))
    except BudgetReached:
        m["state"] = "budget_reached"
        atomic_json(path / "manifest.json", m)
        return m
    concurrency = min(cfg["concurrency"], selection["concurrency_limit"])
    if selection["role"] == "development":
        concurrency = 1
        indices = indices[:1]
    seconds = duration_seconds(selection["time_limit"])
    if not m["profile"]["simulation"] and seconds <= m["profile"]["checkpoint_margin_seconds"]:
        raise ValueError("Wall time must exceed checkpoint margin")
    reserved = seconds * cfg["gpus"] * len(indices) / 3600
    number = len(m["waves"])
    directory = path / "waves" / str(number)
    directory.mkdir(parents=True, exist_ok=True)
    token = m["id"] + "-" + str(number)
    before = {str(i): read_json(path / "workers" / str(i) / "result.json", {}).get("progress", 0) for i in indices}
    before_completed = {str(i): read_json(path / "workers" / str(i) / "result.json", {}).get("completed", 0) for i in indices}
    if (path / "STOP").exists(): raise ValueError("Run was stopped during scheduling checks")
    wave = {"number": number, "token": token, "indices": indices, "concurrency": concurrency, "partition": selection, "status": "intent", "dependency": dependency, "reserved_gpu_hours": reserved, "before": before, "before_completed": before_completed, "created_at": utc_now()}
    m["waves"].append(wave)
    script = directory / "job.sh"
    script.write_text("#!/usr/bin/env bash\nset -eu\nexec " + shlex.join([m["profile"]["python"], str(path / "runtime.py"), str(path.parent.parent), "worker", m["id"], str(number)]) + "\n")
    atomic_json(directory / "wave.json", wave)
    m["state"] = "submitting"
    atomic_json(path / "manifest.json", m)
    job_id, error = scheduler.submit(m, wave, script)
    wave.update(job_id=job_id, status="submitted" if job_id else "uncertain", error=error)
    m["state"] = "queued" if job_id else "needs_attention"
    m["error"] = error
    atomic_json(directory / "wave.json", wave)
    atomic_json(path / "manifest.json", m)
    return m

def create_run(root, profile, config, request_id=None):
    profile, cfg = normalize_profile(profile), normalize_config(config)
    run_id = request_id or uuid.uuid4().hex
    path = run_path(root, run_id)
    # Client-generated request identity survives a lost create response.
    path.parent.mkdir(parents=True, exist_ok=True)
    try: path.mkdir()
    except FileExistsError:
        with lock(path / ".control.lock", blocking=True):
            existing = load_run(path)
            if existing["identity"] != experiment_identity(cfg): raise ValueError("Request identifier belongs to a different experiment")
            return existing
    for name in FILES: shutil.copyfile(Path(__file__).parent / name, path / name)
    m = {"schema_version": 1, "id": run_id, "owner": getpass.getuser(), "owner_uid": os.getuid(), "identity": experiment_identity(cfg), "config": cfg, "profile": profile, "created_at": utc_now(), "created_epoch": time.time(), "state": "draft", "waves": [], "history": [], "runtime_files": {name: file_hash(path / name) for name in FILES}}
    atomic_json(path / "manifest.json", m)
    atomic_json(path / "control.json", {"concurrency": cfg["concurrency"], "auto_continue": cfg["auto_continue"], "budget": cfg["budget"]})
    (path / "notes.md").write_text(cfg["notes"])
    return m

def validate_result(path, m, wave, index, result):
    if result.get("identity") != m["identity"] or result.get("attempt") != wave["token"]: raise ValueError("Missing or stale adapter result")
    if result.get("state") not in ("complete", "partial_checkpointed"): raise ValueError("Adapter failed")
    for key in ("progress", "completed"):
        if isinstance(result.get(key), bool) or not isinstance(result.get(key), (int, float)) or not math.isfinite(result[key]) or result[key] < 0:
            raise ValueError("Invalid progress counter")
    if result["progress"] < wave["before"][str(index)]: raise ValueError("Progress moved backwards")
    files = result.get("checkpoint_files")
    if (result["state"] == "partial_checkpointed" or m["config"]["kind"] == "train") and not files:
        raise ValueError("Resumable work has no sealed checkpoint")
    for name, entry in (files or {}).items():
        candidate = (path / name).resolve()
        if Path(name).is_absolute() or not candidate.is_relative_to(path.resolve()) or candidate.stat().st_size != entry["size"] or file_hash(candidate) != entry["sha256"]:
            raise ValueError("Incomplete or changed checkpoint: " + name)
    assigned = len(range(index, m["config"]["task_count"], m["config"]["workers"]))
    if result["completed"] < wave.get("before_completed", {}).get(str(index), 0) or result["completed"] > assigned:
        raise ValueError("Completed item count moved backwards or exceeds shard inventory")
    if result["state"] == "complete" and result["completed"] != assigned:
        raise ValueError("Complete result does not cover its declared shard inventory")

def handoff(path, wave_number):
    # Blocking record lock avoids a lost-last-worker race. Only the handoff runs scheduler queries.
    with lock(path / ".control.lock", blocking=True):
        m = load_run(path)
        wave = m["waves"][wave_number]
        if wave.get("handled"): return m
        receipts = [read_json(path / "waves" / str(wave_number) / f"receipt-{i}.json") for i in wave["indices"]]
        if not all(receipts): return m
        wave["handled"] = True
        for receipt in receipts:
            if receipt["state"] in ("complete", "partial_checkpointed") and m["config"]["resume_capable"]:
                try: validate_result(path, m, wave, receipt["worker"], receipt)
                except Exception as error: receipt.update(state="failed", error=str(error))
        for receipt in receipts:
            if receipt.get("observed"):
                m["history"].append({"identity": m["identity"], "partition": wave["partition"]["name"], **receipt["observed"]})
        if (path / "STOP").exists() or any(r["state"] == "cancelled" for r in receipts):
            m["state"] = "cancelled"
        elif any(r["state"] == "failed" for r in receipts):
            m["state"] = "failed"
            m["error"] = "; ".join(r.get("error", "Worker failed; inspect its receipt") for r in receipts if r["state"] == "failed")
        elif not remaining_indices(path, m):
            m["state"] = "complete"
        elif any(r["state"] == "partial_checkpointed" and r["progress"] <= wave["before"][str(r["worker"])] for r in receipts):
            m.update(state="needs_attention", error="An unfinished worker made no durable progress")
        else:
            m["state"] = "partial_checkpointed"
        atomic_json(path / "manifest.json", m)
        if m["state"] == "partial_checkpointed" and read_json(path / "control.json")["auto_continue"]:
            try: m = new_wave(path, m, wave.get("job_id"))
            except Exception as error:
                m.update(state="needs_attention", error=str(error))
                atomic_json(path / "manifest.json", m)
        return m

def allocation_deadline(m, wave):
    if m["profile"]["simulation"]: return time.time(), time.time() + 2.2, time.time() + 4
    info = command(["scontrol", "-o", "show", "job", os.environ["SLURM_JOB_ID"]])
    match = re.search(r"\bEndTime=(\S+)", info["stdout"])
    if not match: raise ValueError("Cannot establish actual Slurm end time")
    end = datetime.fromisoformat(match[1]).timestamp()
    if "wall_hours" in m["config"]["budget"]:
        end = min(end, m["created_epoch"] + m["config"]["budget"]["wall_hours"] * 3600)
    start_match = re.search(r"\bStartTime=(\S+)", info["stdout"])
    start = datetime.fromisoformat(start_match[1]).timestamp() if start_match else None
    return start, end - m["profile"]["checkpoint_margin_seconds"], end

def worker(root, run_id, number, index=None):
    path = run_path(root, run_id)
    m = load_run(path)
    wave = read_json(path / "waves" / str(number) / "wave.json")
    index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) if index is None else index
    if index not in wave["indices"]: raise ValueError("Worker is not assigned to this wave")
    directory = path / "workers" / str(index)
    directory.mkdir(parents=True, exist_ok=True)
    receipt = {"worker": index, "state": "failed", "progress": wave["before"][str(index)], "completed": 0, "started_at": time.time()}
    cancelled, drained = [], []
    def caught(signum, frame):
        (cancelled if signum in (signal.SIGTERM, signal.SIGINT) else drained).append(signum)
        (directory / "DRAIN").touch()
    old = {sig: signal.signal(sig, caught) for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)}
    try:
        with lock(directory / ".worker.lock"):
            atomic_json(directory / "active.json", {"attempt": wave["token"], "started_at": time.time()})
            if (path / "STOP").exists():
                receipt["state"] = "cancelled"
            else:
                start, deadline, end = allocation_deadline(m, wave)
                if time.time() >= deadline: raise ValueError("Insufficient allocation time remains for useful work and checkpointing")
                if (directory / "DRAIN").exists(): (directory / "DRAIN").unlink()
                prior = read_json(directory / "result.json", {})
                if prior:
                    # Validate restart files again; no fresh work may overwrite a broken checkpoint.
                    previous = {**wave, "token": prior.get("attempt")}
                    validate_result(path, m, previous, index, prior)
                if number == 0 and m["config"]["auto_continue"] and not m["profile"]["simulation"]:
                    probe = command(["sbatch", "--test-only", "--partition=" + wave["partition"]["name"], "--time=" + wave["partition"]["time_limit"], "--nodes=1", "--ntasks=1", "--wrap=true"] + (["--account=" + m["profile"]["account"]] if m["profile"]["account"] else []) + (["--qos=" + m["profile"]["qos"]] if m["profile"]["qos"] else []) + (["--gres=gpu:" + str(m["config"]["gpus"])] if m["config"]["gpus"] else []))
                    atomic_json(directory / "submission-probe.json", probe)
                    if probe["code"]: raise ValueError("Cluster-side submission check failed: " + probe["stderr"])
                env = {k: v for k, v in os.environ.items() if not k.startswith("SBATCH_")}
                env.update({k: str(v) for k, v in m["config"]["environment"].items()})
                env.update(WM_RUN_DIR=str(path), WM_WORKER_INDEX=str(index), WM_WORKER_COUNT=str(m["config"]["workers"]), WM_TASK_COUNT=str(m["config"]["task_count"]), WM_IDENTITY=m["identity"], WM_ATTEMPT=wave["token"], WM_DEADLINE=str(deadline), WM_CONFIG=str(path / "manifest.json"), PYTHONUNBUFFERED="1")
                if start is not None:
                    env["WM_ALLOCATION_START"] = str(start)
                env["WM_PRIOR_COMPLETED"] = str(wave["before_completed"][str(index)])
                argv = [a.replace("{run_dir}", str(path)).replace("{python}", m["profile"]["python"]) for a in m["config"]["command"]]
                if not m["profile"]["simulation"]: argv = ["srun", "--ntasks=1", "--mem=0", "--kill-on-bad-exit=1"] + argv
                log_path = path / "waves" / str(number) / f"worker-{index}.log"
                metric_path = directory / "metrics.jsonl"
                metric_offset = metric_path.stat().st_size if metric_path.exists() else 0
                with log_path.open("ab") as log:
                    child = subprocess.Popen(argv, env=env, cwd=m["profile"]["remote_root"], stdout=log, stderr=log, start_new_session=True)
                    while child.poll() is None:
                        if time.time() >= deadline or cancelled or (path / "STOP").exists():
                            (directory / "DRAIN").touch()
                        if cancelled or (path / "STOP").exists() or time.time() >= end - (0.3 if m["profile"]["simulation"] else 10):
                            os.killpg(child.pid, signal.SIGTERM)
                            try: child.wait(timeout=max(0.1, min(5, end - time.time() - 2)))
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid, signal.SIGKILL)
                                child.wait(timeout=2)
                            break
                        time.sleep(0.1 if m["profile"]["simulation"] else 0.5)
                if cancelled or (path / "STOP").exists(): receipt["state"] = "cancelled"
                elif child.returncode: raise ValueError(f"Adapter exited with status {child.returncode}; see allocation log")
                elif not m["config"]["resume_capable"]:
                    receipt.update(state="complete", progress=1, completed=1)
                    atomic_json(directory / "result.json", {**receipt, "identity": m["identity"], "attempt": wave["token"], "checkpoint_files": {}})
                else:
                    result = read_json(directory / "result.json", {})
                    validate_result(path, m, wave, index, result)
                    receipt.update(result)
                observed = {}
                if metric_path.exists():
                    with metric_path.open("rb") as metrics:
                        metrics.seek(metric_offset)
                        for raw in metrics:
                            try: row = json.loads(raw)
                            except (ValueError, UnicodeDecodeError): continue
                            if row.get("attempt") != wave["token"]: continue
                            for key in ("seconds_per_task", "model_load_seconds", "preparation_seconds", "restore_seconds", "first_saved_result_seconds"):
                                value = row.get(key)
                                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
                                    observed[key] = value
                receipt["observed"] = observed
    except BaseException as error:
        receipt.update(state="cancelled" if cancelled else "failed", error=str(error))
    finally:
        receipt.update(worker=index, finished_at=time.time())
        atomic_json(path / "waves" / str(number) / f"receipt-{index}.json", receipt)
        for sig, previous in old.items(): signal.signal(sig, previous)
    handoff(path, number)

def simulate(root, run_id, number):
    path = run_path(root, run_id)
    # Wait for the submitter to publish the job id before a tiny simulated worker can finish.
    for _ in range(100):
        wave = read_json(path / "waves" / str(number) / "wave.json")
        if wave["status"] == "submitted": break
        time.sleep(0.02)
    def invoke(index):
        return subprocess.run([sys.executable, str(path / "runtime.py"), str(root), "worker", run_id, str(number), str(index)], capture_output=True, text=True).returncode
    with ThreadPoolExecutor(max_workers=wave["concurrency"]) as pool:
        list(pool.map(invoke, wave["indices"]))

def snapshot(path):
    m = load_run(path)
    workers = []
    for i in range(m["config"]["workers"]):
        directory = path / "workers" / str(i)
        checkpoint = directory / "checkpoint.json"
        workers.append({"index": i, "result": read_json(directory / "result.json"), "checkpoint": {"path": str(checkpoint), "bytes": checkpoint.stat().st_size} if checkpoint.exists() else None})
    m["workers"] = workers
    m["control"] = read_json(path / "control.json")
    m["notes"] = (path / "notes.md").read_text()
    if m["waves"]:
        last = m["waves"][-1]
        m["scheduler"] = Scheduler(path.parent.parent, m["profile"]).status(last.get("job_id")) if last.get("job_id") and m["state"] not in TERMINAL else {"state": m["state"].upper(), "reason": last.get("error", "")}
        if m["state"] in ("queued", "submitting") and not m["profile"]["simulation"] and m["scheduler"]["state"] == "TERMINAL":
            m["attention"] = "Allocation ended; missing worker receipts or interrupted handoff require reconciliation."
        if m["state"] == "queued" and any(read_json(path / "workers" / str(i) / "active.json", {}).get("attempt") == last["token"] and not (path / "waves" / str(last["number"]) / f"receipt-{i}.json").exists() for i in last["indices"]):
            m["state"] = "running"
    return m

def dispatch(root, profile, action, data):
    if action == 'browse':
        from workspace import browse
        return browse(root, data.get('path', ''), data.get('search', ''), data.get('offset', 0), data.get('hidden', False))
    if action == "discover":
        cfg = normalize_config(data["config"])
        discovery = Scheduler(root, profile).discover(cfg)
        plans = estimate_plans(cfg, discovery["partitions"])
        try: recommendation = choose_plan({"profile": profile, "config": cfg}, discovery, cfg["task_count"])
        except ValueError: recommendation = None
        return {**discovery, "plans": plans, "recommendation": recommendation}
    if action == "create": return create_run(root, profile, data["config"], data.get("request_id"))
    if action == "list":
        return [snapshot(p.parent) for p in sorted((Path(root) / "cluster-ui").glob("*/manifest.json"), key=lambda p:p.stat().st_mtime, reverse=True)]
    path = run_path(root, data["id"])
    if action == "get": return snapshot(path)
    if action == 'review':
        from workspace import review
        return review(path, load_run(path))
    if action == "tail":
        candidate = (path / data["file"]).resolve()
        if not candidate.is_relative_to(path.resolve()) or candidate.suffix not in (".log", ".jsonl"): raise ValueError("Invalid log path")
        offset = max(0, int(data.get("offset", 0)))
        if not candidate.exists(): return {"text": "", "offset": 0}
        with candidate.open("rb") as stream:
            if offset > candidate.stat().st_size: offset = 0
            stream.seek(offset)
            block = stream.read(65536)
            return {"text": block.decode("utf-8", errors="replace"), "offset": stream.tell()}
    if action == "stop":
        # Publish the stop immediately even if a handoff currently holds the control lock.
        m = load_run(path)
        if m.get("owner_uid", os.getuid()) != os.getuid(): raise ValueError("Only the run owner may change its jobs")
        (path / "STOP").touch()
    with lock(path / ".control.lock", blocking=True):
        m = load_run(path)
        if m.get("owner_uid", os.getuid()) != os.getuid(): raise ValueError("Only the run owner may change its jobs")
        scheduler = Scheduler(root, m["profile"])
        if action == "launch" or action == "resume":
            if m["state"] not in TERMINAL | {"draft"}:
                if not m["waves"] or profile["simulation"] or scheduler.status(m["waves"][-1].get("job_id"))["state"] != "TERMINAL":
                    raise ValueError("Run already has an active or unconfirmed allocation")
            if m["state"] == "complete": raise ValueError("Run is already complete")
            for wave in m["waves"]:
                if wave["status"] in ("intent", "uncertain"):
                    ids, _ = scheduler.reconcile(wave["token"]) if not profile["simulation"] else ([], False)
                    if len(ids) == 1:
                        wave.update(status="submitted", job_id=ids[0])
                        m.update(state="queued", error="")
                        atomic_json(path / "manifest.json", m)
                        return m
                    raise ValueError("Uncertain submission: inspect Slurm before resetting the run; no duplicate submitted")
            if m["waves"] and not profile["simulation"]:
                state = scheduler.status(m["waves"][-1].get("job_id"))
                if state["state"] != "TERMINAL": raise ValueError("Previous allocation is active or unconfirmed; inspect Slurm before resuming")
            if (path / "STOP").exists():
                (path / "STOP").unlink()
                control = read_json(path / "control.json")
                control["auto_continue"] = m["config"]["auto_continue"]
                atomic_json(path / "control.json", control)
            return new_wave(path, m)
        if action == "stop":
            (path / "STOP").touch()
            control = read_json(path / "control.json")
            control["auto_continue"] = False
            atomic_json(path / "control.json", control)
            errors = []
            for wave in m["waves"][-1:]:
                if wave.get("job_id") and not profile["simulation"]:
                    result = command(["scancel", wave["job_id"]])
                    if result["code"]: errors.append(result["stderr"])
            m.update(state="cancelled", error="; ".join(errors))
        elif action == "concurrency":
            target = data["concurrency"]
            if not isinstance(target, int) or isinstance(target, bool) or not 1 <= target <= m["config"]["workers"]: raise ValueError("Invalid concurrency")
            active = m["waves"][-1] if m["waves"] else None
            cap = active["partition"]["concurrency_limit"] if active else max(p["concurrency_limit"] for p in profile["partitions"])
            if target > cap: raise ValueError("Concurrency exceeds profile/live limit")
            if active and active["status"] == "submitted" and not profile["simulation"]:
                result = command(["scontrol", "update", "JobId=" + active["job_id"], "ArrayTaskThrottle=" + str(target)])
                if result["code"]: raise ValueError(result["stderr"])
            control = read_json(path / "control.json")
            control["concurrency"] = target
            atomic_json(path / "control.json", control)
        elif action == "notes":
            if not isinstance(data["notes"], str) or len(data["notes"]) > 131072: raise ValueError("Invalid notes")
            (path / "notes.md").write_text(data["notes"])
        else: raise ValueError("Unknown action")
        atomic_json(path / "manifest.json", m)
        return m

def main():
    root, action = sys.argv[1:3]
    if action in ("worker", "simulate"):
        run_id, number = sys.argv[3], int(sys.argv[4])
        if action == "worker": worker(root, run_id, number, int(sys.argv[5]) if len(sys.argv) > 5 else None)
        else: simulate(root, run_id, number)
        return
    try:
        payload = json.loads(sys.stdin.read(2097152))
        result = dispatch(root, normalize_profile(payload["profile"]), action, payload.get("data", {}))
        print(json.dumps({"ok": True, "result": result}, allow_nan=False))
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        sys.exit(1)

if __name__ == "__main__": main()
