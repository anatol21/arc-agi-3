"""Select an eligible Slurm partition and resume checkpointed workloads within a GPU-hour cap."""

import argparse
from datetime import datetime, timedelta
from decimal import Decimal
import fcntl
import getpass
import json
import math
from pathlib import Path
import re
import subprocess
import time
import uuid

from checkpoint import atomic_json, digest, read_progress

HERE = Path(__file__).resolve().parent
INFRA = {"PREEMPTED", "NODE_FAIL", "BOOT_FAIL", "TIMEOUT"}
TERMINAL = INFRA | {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "OUT_OF_MEMORY",
    "DEADLINE",
    "REVOKED",
}


def slurm(args):
    result = subprocess.run(
        args, capture_output=True, text=True, timeout=30, check=True
    )
    return result.stdout + result.stderr if "--test-only" in args else result.stdout


def load_config(path):
    c = json.loads(path.read_bytes())
    if (
        not isinstance(c["workload_revision"], str)
        or not c["workload_revision"].strip()
        or type(c["gpus"]) != int
        or c["gpus"] < 1
        or type(c["max_minutes"]) != int
        or c["max_minutes"] < 1
        or type(c["reload_seconds"]) != int
        or c["reload_seconds"] < 0
        or not isinstance(c["partitions"], list)
        or not c["partitions"]
        or not c["cpu_partition"]
        or not isinstance(c["items"], list)
        or not c["items"]
        or len(c["items"]) != len(set(c["items"]))
        or any(
            not isinstance(s, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", s)
            for s in c["items"]
        )
        or not isinstance(c["worker_command"], list)
        or not c["worker_command"]
        or any(not isinstance(arg, str) for arg in c["worker_command"])
    ):
        raise ValueError("Invalid resource, item, or worker configuration")
    seconds = Decimal(str(c["gpu_hours"])) * 3600
    if not seconds.is_finite() or seconds != int(seconds) or seconds < c["gpus"] * 60:
        raise ValueError("GPU budget must contain at least one whole allocation minute")
    return c


def wall_minutes(value):
    if value in {"UNLIMITED", "INFINITE"}:
        return math.inf
    m = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d\d):(\d\d)", value)
    if not m:
        raise ValueError(f"Unknown Slurm time: {value}")
    days, hours, minutes, _ = m.groups()
    return int(days or 0) * 1440 + int(hours) * 60 + int(minutes)


def common_args(c):
    return [f"--{key}={c[key]}" for key in ("account", "qos") if c.get(key)]


def gpu_args(c, path, part, minutes):
    return [
        *common_args(c),
        "--nodes=1",
        "--ntasks=1",
        f"--gres=gpu:{c['gpus']}",
        f"--partition={part}",
        f"--time={minutes}",
        "--no-requeue",
        f"--chdir={path.parent}",
        *([f"--cpus-per-task={c['cpu_threads']}"] if c.get("cpu_threads") else []),
        *([f"--mem={c['memory']}"] if c.get("memory") else []),
        str(HERE / "worker.sbatch"),
        str(path),
        str(HERE / "worker.py"),
    ]


def choose_partition(c, path, remaining_minutes, command):
    probes, rejected = [], {}
    for part in sorted(set(c["partitions"])):
        try:
            raw = command(["scontrol", "-o", "show", "partition", part])
            data = dict(t.split("=", 1) for t in raw.split() if "=" in t)
            if data["PartitionName"] != part or data["State"] != "UP":
                rejected[part] = "inactive"
                continue
            n = min(remaining_minutes, c["max_minutes"], wall_minutes(data["MaxTime"]))
            if n < 1:
                rejected[part] = "insufficient wall time"
                continue
            try:
                estimate = command(
                    ["sbatch", "--test-only", *gpu_args(c, path, part, n)]
                )
            except subprocess.TimeoutExpired:
                estimate = ""
        except subprocess.CalledProcessError as e:
            rejected[part] = str(e.stderr or e)
            continue
        match = re.search(r"\bto start at (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)\b", estimate)
        eta = match[1] if match else None
        useful = n * 60 - c["reload_seconds"]
        if useful <= 0:
            rejected[part] = "allocation cannot cover the configured reload cost"
            continue
        work = max(0, remaining_minutes * 60 - c["reload_seconds"])
        extra = max(0, math.ceil(work / useful) - 1)
        adjusted = (
            datetime.fromisoformat(eta).timestamp() + extra * c["reload_seconds"]
            if eta
            else None
        )
        probes.append(
            dict(
                partition=part,
                minutes=n,
                eta=eta,
                extra_reloads=extra,
                adjusted_start=adjusted,
            )
        )
    if not probes:
        raise RuntimeError(f"No eligible partition: {rejected}")
    chosen = min(
        probes,
        key=lambda p: (
            p["adjusted_start"] is None,
            p["adjusted_start"] or 0,
            -p["minutes"],
            p["partition"],
        ),
    )
    return dict(**chosen, probes=probes, rejected=rejected)


def reconcile(intent, command):
    q = command(
        [
            "squeue",
            "-h",
            "-u",
            getpass.getuser(),
            "--name",
            intent["name"],
            "-o",
            "%A|%j",
        ]
    )
    h = command(
        [
            "sacct",
            "-X",
            "-nP",
            "-S",
            intent["since"],
            "--name",
            intent["name"],
            "--format=JobIDRaw,JobName%64",
        ]
    )
    ids = {
        p[0]
        for line in (q + "\n" + h).splitlines()
        if len(p := line.strip().split("|")) >= 2
        and p[1] == intent["name"]
        and p[0].isdecimal()
    }
    if len(ids) != 1:
        raise RuntimeError(
            f"Ambiguous prior submission {intent['name']}: {sorted(ids)}. Inspect squeue and sacct."
        )
    return ids.pop()


def submit_once(wave, kind, args, save, command):
    if kind not in wave:
        intent = wave[kind] = dict(
            name=f"campaign-{kind}-{uuid.uuid4().hex[:16]}",
            since=(datetime.now().date() - timedelta(days=1)).isoformat(),
            job=None,
        )
        save()  # Durable before calling sbatch: a lost reply must not duplicate a job.
        try:
            raw = command(
                ["sbatch", "--parsable", f"--job-name={intent['name']}", *args]
            ).strip()
            match = re.fullmatch(r"(\d+)(?:;[^;\s]+)?", raw)
            if match:
                intent["job"] = match[1]
        except (OSError, subprocess.SubprocessError) as exc:
            intent["error"] = str(exc)
            save()
    intent = wave[kind]
    if intent["job"] is None:
        intent["job"] = reconcile(intent, command)
    save()
    return intent["job"]


def accounting(job, gpus, command):
    raw = command(
        [
            "sacct",
            "-X",
            "-nP",
            "-j",
            job,
            "--format=JobIDRaw,State%32,ExitCode,ElapsedRaw,AllocTRES%512",
        ]
    )
    rows = [line.split("|") for line in raw.splitlines() if line.split("|")[0] == job]
    if not rows:
        return None
    if len(rows) != 1:
        raise ValueError("Ambiguous allocation accounting")
    _, state, code, elapsed, resources = rows[0][:5]
    state = state.split()[0].rstrip("+")
    if state not in TERMINAL:
        return None
    elapsed = int(elapsed)
    tres = dict(s.split("=", 1) for s in resources.split(",") if "=" in s)
    count = int(tres.get("gres/gpu", 0))
    if elapsed < 0 or (count != gpus and (elapsed or count)):
        raise ValueError("GPU count differs from submitted shape")
    return dict(
        state=state,
        exit_code=code,
        elapsed_seconds=elapsed,
        gpu_seconds=elapsed * count,
    )


def run(config_path, action, job=None, command=slurm):
    path = Path(config_path).resolve()
    c = load_config(path)
    identity = digest(c)
    root = path.parent / ".campaigns" / identity
    if action == "status" and not (root / "scheduler.json").exists():
        rows = read_progress(root, identity, set(c["items"]))
        return {
            "phase": "not_launched",
            "items": len(c["items"]),
            "finished": sum(row["status"] == "finished" for row in rows.values()),
        }
    root.mkdir(parents=True, exist_ok=True)
    ledger_path = root / "scheduler.json"
    with (root / "scheduler.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if ledger_path.exists():
            ledger = json.loads(ledger_path.read_bytes())
            if ledger["campaign"] != identity:
                raise ValueError("Ledger identity changed")
        else:
            if action != "launch":
                raise ValueError("Unknown campaign")
            ledger = dict(
                campaign=identity, phase="ready", charged_gpu_seconds=0, waves=[]
            )
        save = lambda: atomic_json(ledger_path, ledger)
        waves = ledger["waves"]
        cap = int(Decimal(str(c["gpu_hours"])) * 3600)

        def progress():
            rows = read_progress(root, identity, set(c["items"]))
            return dict(
                finished=sum(r["status"] == "finished" for r in rows.values()),
                units=sum(r["units"] for r in rows.values()),
            )

        if action == "status":
            return dict(ledger=ledger, progress=progress())
        if action not in {"launch", "settle"}:
            raise ValueError("Unknown action")
        if action == "settle":
            if not any(w.get("gpu", {}).get("job") == job for w in waves):
                raise ValueError("Unknown settlement allocation")
            if waves[-1]["gpu"]["job"] != job:
                return ledger
        if ledger["phase"] in {
            "complete",
            "needs_attention",
            "budget_reached",
            "stopped",
        }:
            return ledger
        wave = waves[-1] if waves else None
        if wave and "accounting" not in wave:
            # Recover GPU and watcher receipts before reconciling terminal accounting.
            gpu = submit_once(
                wave,
                "gpu",
                gpu_args(c, path, wave["partition"], wave["minutes"]),
                save,
                command,
            )
            watch_args = [
                *common_args(c),
                f"--partition={c['cpu_partition']}",
                f"--dependency=afterany:{gpu}",
                f"--chdir={path.parent}",
                str(HERE / "handoff.sbatch"),
                str(path),
                gpu,
                str(HERE / "scheduler.py"),
            ]
            submit_once(wave, "watch", watch_args, save, command)
            until = time.monotonic() + (180 if action == "settle" else 0)
            while True:
                account = accounting(gpu, c["gpus"], command)
                if account or time.monotonic() >= until:
                    break
                time.sleep(5)
            if account is None:
                if action == "settle":
                    raise RuntimeError(
                        "Accounting delayed. Re-run launch after sacct settles."
                    )
                return ledger
            after = progress()
            before = wave["before"]
            if any(after[k] < before[k] for k in after):
                raise ValueError("Checkpoint progress regressed")
            wave.update(accounting=account, after=after)
            ledger["charged_gpu_seconds"] = sum(
                w["accounting"]["gpu_seconds"] for w in waves
            )
            state = account["state"]
            if state == "COMPLETED" and account["exit_code"] == "0:0":
                advanced = after != before or after["finished"] == len(c["items"])
                ledger["phase"] = "resumable" if advanced else "needs_attention"
            elif state in INFRA:
                again = len(waves) >= 2 and waves[-2]["after"] == waves[-2]["before"]
                ledger["phase"] = (
                    "needs_attention" if again and before == after else "resumable"
                )
            else:
                ledger["phase"] = "needs_attention"
            save()

        if ledger["phase"] == "needs_attention":
            return ledger
        current = progress()
        left = (cap - ledger["charged_gpu_seconds"]) // (60 * c["gpus"])
        if current["finished"] == len(c["items"]):
            ledger["phase"] = "complete"
        elif (root / "STOP").exists():
            ledger["phase"] = "stopped"
        elif left < 1 or left * 60 <= c["reload_seconds"]:
            ledger["phase"] = "budget_reached"
        else:
            choice = choose_partition(c, path, left, command)
            wave = dict(**choice, before=current)
            waves.append(wave)
            ledger["phase"] = "submitting"
            save()
            gpu = submit_once(
                wave,
                "gpu",
                gpu_args(c, path, wave["partition"], wave["minutes"]),
                save,
                command,
            )
            args = [
                *common_args(c),
                f"--partition={c['cpu_partition']}",
                f"--dependency=afterany:{gpu}",
                f"--chdir={path.parent}",
                str(HERE / "handoff.sbatch"),
                str(path),
                gpu,
                str(HERE / "scheduler.py"),
            ]
            submit_once(wave, "watch", args, save, command)
            ledger["phase"] = "submitted"
        save()
        return ledger


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=["launch", "settle", "status", "probe"])
    cli.add_argument("config", type=Path)
    cli.add_argument("job", nargs="?")
    opt = cli.parse_args()
    if opt.action == "probe":
        c = load_config(opt.config)
        limit = int(Decimal(str(c["gpu_hours"])) * 3600) // (60 * c["gpus"])
        data = choose_partition(c, opt.config.resolve(), limit, slurm)
    else:
        data = run(opt.config, opt.action, opt.job)
    print(json.dumps(data, indent=2))
