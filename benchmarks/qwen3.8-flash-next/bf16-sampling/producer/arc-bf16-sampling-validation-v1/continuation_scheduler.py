"""Submit and settle a bounded series of four-H100 validation allocations."""
import datetime
from dataclasses import dataclass
from typing import Callable
import fcntl
import json
import getpass
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

FOLDER = Path(__file__).resolve().parent
if __package__:
    from .continuation_stage import read_plan
    from .continuation_score import checked_states
    from .generation import TERMINAL, write_json
else:
    sys.path.insert(0, str(FOLDER))
    from continuation_stage import read_plan
    from continuation_score import checked_states
    from generation import TERMINAL, write_json

CAP_SECONDS = 6 * 3600  # Aggregate elapsed GPU-node wall time, with four GPUs on every node.
GPUS = 4
INFRASTRUCTURE = {"PREEMPTED", "NODE_FAIL", "BOOT_FAIL", "TIMEOUT"}
TERMINAL_STATES = INFRASTRUCTURE | {"COMPLETED", "CANCELLED", "FAILED", "OUT_OF_MEMORY", "DEADLINE", "REVOKED"}


@dataclass(frozen=True)
class Workload:
    plan_reader: Callable
    measure_progress: Callable
    target_count: Callable
    choose_slot: Callable
    gpu_script: str
    watch_script: str
    name_prefix: str


def greedy_workload():
    return Workload(read_plan, progress, lambda plan: len(plan["eligible_ids"]),
                    lambda folder, minutes, command: {"partition": "gpu_h100", "minutes": minutes},
                    "continuation.sbatch", "continuation_settle.sbatch", "bf16")


def slurm(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=30, check=True)
    return result.stdout + result.stderr if "--test-only" in args else result.stdout


def accounting(job, command=slurm):
    raw = command(["sacct", "-X", "-nP", "-j", job, "--format=JobIDRaw,State%32,ExitCode,ElapsedRaw,AllocTRES%512"])
    rows = [line.split("|") for line in raw.splitlines() if line.split("|")[0] == job]
    if not rows:
        return None
    if len(rows) != 1:
        raise ValueError(f"Ambiguous accounting for job {job}")
    _, state, code, elapsed, tres = rows[0][:5]
    state = state.split()[0].rstrip("+")
    if state not in TERMINAL_STATES:
        return None
    seconds = int(elapsed)
    resources = dict(pair.split("=", 1) for pair in tres.split(",") if "=" in pair)
    gpus = int(resources.get("gres/gpu", "0"))
    if seconds < 0 or (gpus != GPUS and (seconds or gpus)):
        raise ValueError(f"Unexpected allocated GPU accounting on job {job}: {raw}")
    return dict(job=job, state=state, exit_code=code, elapsed_seconds=seconds, gpus=gpus,
                gpu_seconds=gpus * seconds)


def reconcile(intent, command):
    name = intent["name"]
    active = command(["squeue", "-h", "-u", getpass.getuser(), "--name", name, "-o", "%A|%j"])
    historic = command(["sacct", "-X", "-nP", "-S", intent["since"], "--name", name,
                        "--format=JobIDRaw,JobName%64"])
    ids = {fields[0] for line in (active + "\n" + historic).splitlines()
           if len(fields := line.strip().split("|")) >= 2 and fields[1] == name and fields[0].isdecimal()}
    if len(ids) != 1:
        raise RuntimeError(f"Uncertain sbatch intent {name} matches {sorted(ids)}. No duplicate was submitted.")
    return ids.pop()


def submit_once(ledger, wave, kind, args, save, command=slurm, *, prefix="bf16"):
    if kind in wave:
        intent = wave[kind]
        if not intent.get("job"):
            intent["job"] = reconcile(intent, command)
            save(ledger)
        return intent["job"]
    name = prefix + "-" + kind + "-" + uuid.uuid4().hex[:16]
    intent = wave[kind] = dict(name=name, since=datetime.date.today().isoformat(), job=None)
    save(ledger)  # If this process dies after sbatch accepts, recover exclusively by unique job name.
    try:
        text = command(["sbatch", "--parsable", "--job-name=" + name, *args]).strip()
        match = re.fullmatch(r"(\d+)(?:;[^;\s]+)?", text)
        job = match[1] if match else reconcile(intent, command)
    except (OSError, subprocess.SubprocessError) as exc:
        intent["error"] = str(exc)
        save(ledger)
        job = reconcile(intent, command)
    intent["job"] = job
    save(ledger)
    return job


def progress(root, plan):
    states = checked_states(root, plan["prompts"], plan["benchmark_id"], plan["eos_token_id"])
    return dict(terminal=sum(r["finish_reason"] in TERMINAL for r in states.values()),
                committed_tokens=sum(r["generated_tokens"] for r in states.values()))


def retry_decision(account, wave, progress_now, pending, summary, worker_failed):
    state = account["state"]
    if state not in INFRASTRUCTURE and (state != "COMPLETED" or account["exit_code"] != "0:0"):
        return "needs_attention"
    if worker_failed:
        return "needs_attention"
    if not pending:
        return "complete"
    if state == "COMPLETED":
        if summary is None or summary.get("blocked_task_ids") or progress_now["committed_tokens"] <= wave["before"]["committed_tokens"]:
            return "needs_attention"
    return "resumable"


def ensure_wave(folder, root, ledger, plan, save, command, workload):
    if ledger["phase"] in {"complete", "needs_attention", "budget_reached", "stopped"}:
        return
    wave = ledger["waves"][-1] if ledger["waves"] else None
    if wave is None or wave.get("accounting"):
        before = workload.measure_progress(root, plan)
        left = (ledger["cap_gpu_seconds"] - ledger["charged_gpu_seconds"]) // GPUS
        if before["terminal"] == workload.target_count(plan) or left < 60:
            ledger["phase"] = "complete" if before["terminal"] == workload.target_count(plan) else "budget_reached"
            save(ledger)
            return
        # Slurm rounds time limits up to whole minutes. Always round the remaining budget down.
        choice = workload.choose_slot(folder, left // 60, command)
        if not 1 <= choice["minutes"] <= left // 60:
            raise ValueError("Selected allocation exceeds the remaining GPU budget")
        wave = dict(choice, before=before)
        ledger["waves"].append(wave)
        save(ledger)
    if (root / "STOP").exists():
        ledger["phase"] = "stopped"
        save(ledger)
        return
    job = submit_once(ledger, wave, "gpu",
                      ["--partition=" + wave.get("partition", "gpu_h100"), "--gres=gpu:4", "--nodes=1", "--time=" + str(wave["minutes"]),
                       "--no-requeue", "--chdir=" + str(folder.parent), str(folder / workload.gpu_script)],
                      save, command, prefix=workload.name_prefix)
    submit_once(ledger, wave, "watch", ["--partition=cpu", "--dependency=afterany:" + job,
               "--chdir=" + str(folder.parent), str(folder / workload.watch_script), job], save, command, prefix=workload.name_prefix)
    ledger["phase"] = "submitted"
    save(ledger)


def settle(folder, root, ledger, plan, job, save, command, workload, *, wait):
    if not ledger["waves"]:
        raise ValueError("Settlement without a tracked allocation")
    wave = ledger["waves"][-1]
    if not any(w.get("gpu", {}).get("job") == job for w in ledger["waves"]):
        raise ValueError("Untracked GPU job")
    if wave["gpu"]["job"] != job:
        return  # Delayed watcher for an earlier wave.
    if wave.get("accounting"):
        # Resume an interrupted transition between durable settlement and successor submission.
        ensure_wave(folder, root, ledger, plan, save, command, workload)
        return
    deadline = time.monotonic() + (180 if wait else 0)
    while True:
        account = accounting(job, command)
        if account is not None:
            break
        if time.monotonic() >= deadline:
            if wait:
                raise RuntimeError(f"Final accounting unavailable for {job}; resume with 'launch' after sacct settles")
            return
        time.sleep(5)
    before = wave["before"]
    current = workload.measure_progress(root, plan)
    if any(current[k] < before[k] for k in before):
        raise ValueError("Progress counters decreased during allocation")
    alloc = root / "allocations" / job
    summary = json.loads((alloc / "generation-summary.json").read_bytes()) if (alloc / "generation-summary.json").is_file() else None
    status = json.loads((alloc / "status.json").read_bytes()) if (alloc / "status.json").is_file() else {}
    pending = workload.target_count(plan) - current["terminal"]
    choice = retry_decision(account, wave, current, pending, summary, status.get("phase") == "failed")
    wave.update(accounting=account, after=current)
    if (choice == "resumable" and account["state"] in INFRASTRUCTURE and current == before
            and len(ledger["waves"]) >= 2 and ledger["waves"][-2]["after"] == ledger["waves"][-2]["before"]):
        choice = "needs_attention"  # Two successive infrastructure losses with no durable progress.
    ledger["charged_gpu_seconds"] = sum(w["accounting"]["gpu_seconds"] for w in ledger["waves"])
    ledger["phase"] = choice
    save(ledger)
    ensure_wave(folder, root, ledger, plan, save, command, workload)


def run(folder, action, job=None, command=slurm, *, workload=None, cap_gpu_seconds=None):
    if workload is None:
        workload = greedy_workload()
        cap_gpu_seconds = CAP_SECONDS * GPUS
    folder = Path(folder).resolve()
    plan = workload.plan_reader(folder)
    root = folder.parent / "runs" / folder.name / plan["benchmark_id"]
    root.mkdir(parents=True, exist_ok=True)
    path = root / "scheduler.json"

    def save(value):
        write_json(path, value)

    with (root / ".scheduler.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            ledger = json.loads(path.read_bytes())
            if ledger["benchmark_id"] != plan["benchmark_id"] or (cap_gpu_seconds is not None and ledger["cap_gpu_seconds"] != cap_gpu_seconds):
                raise ValueError("Incompatible scheduler ledger")
        else:
            if action != "launch" or any((root / "allocations").glob("*")):
                raise ValueError("Missing scheduler ledger for an existing allocation")
            if type(cap_gpu_seconds) is not int or cap_gpu_seconds < GPUS * 60:
                raise ValueError("A new campaign needs an explicit positive GPU-hour cap")
            ledger = dict(benchmark_id=plan["benchmark_id"], cap_gpu_seconds=cap_gpu_seconds,
                          charged_gpu_seconds=0, phase="ready", waves=[])
            save(ledger)
        if action == "launch":
            ensure_wave(folder, root, ledger, plan, save, command, workload)
            if ledger["phase"] == "submitted":
                settle(folder, root, ledger, plan, ledger["waves"][-1]["gpu"]["job"], save, command, workload, wait=False)
        elif action == "settle" and job and job.isdecimal():
            settle(folder, root, ledger, plan, job, save, command, workload, wait=True)
        else:
            raise ValueError("Expected launch or settle <gpu-job-id>")
        return ledger


if __name__ == "__main__":
    ledger = run(FOLDER, *sys.argv[1:])
    print("BF16_SCHEDULER", json.dumps(ledger, allow_nan=False), flush=True)
