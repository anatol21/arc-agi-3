"""Compute the stop time from the real Slurm allocation, not a fixed duration."""
import datetime
import math
import os
import re
import subprocess


def allocation_budget(environment=None, now=None, query=None):
    import time
    environment = os.environ if environment is None else environment
    now = time.time() if now is None else now
    raw = environment.get("SLURM_JOB_END_TIME")
    source = "SLURM_JOB_END_TIME"
    if raw:
        end = float(raw)
    else:
        job = environment.get("SLURM_JOB_ID", "")
        if not job.isdecimal():
            raise RuntimeError("Run inside a Slurm allocation with a known job end time")
        if query is None:
            query = lambda command: subprocess.run(command, check=True, capture_output=True, text=True, timeout=15).stdout
        record = query(["scontrol", "--oneliner", "show", "job", job])
        match = re.search(r"\bEndTime=([^\s]+)", record)
        if not match or match[1] in ("Unknown", "N/A", "None"):
            raise RuntimeError("Slurm did not supply a usable allocation end time")
        # scontrol emits the node's local wall clock; timestamp() uses the same timezone.
        end = datetime.datetime.fromisoformat(match[1]).timestamp()
        source = "scontrol EndTime"
    if not math.isfinite(end) or end <= 0:
        raise RuntimeError("Invalid allocation end time")
    stop = end - 300  # Leave five minutes to flush predictions and score on CPU.
    if environment.get("ARC_BENCH_STOP_AT"):
        explicit = float(environment["ARC_BENCH_STOP_AT"])
        if not math.isfinite(explicit):
            raise RuntimeError("Invalid explicit benchmark stop time")
        stop = min(stop, explicit)
    return {"allocation_end": end, "stop_at": stop, "reserve_seconds": 300,
            "remaining_generation_seconds": max(0, stop - now), "source": source}
