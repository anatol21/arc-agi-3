"""Configuration and estimates. YAML editing uses local PyYAML; remote code is stdlib-only."""
from __future__ import annotations
import hashlib
import json
import math
import re
from datetime import datetime, timezone

class ConfigError(ValueError):
    pass

def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def parse_yaml(source):
    import yaml
    if len(source.encode()) > 131072:
        raise ConfigError("YAML is larger than 128 KiB")
    class UniqueLoader(yaml.SafeLoader):
        pass
    def mapping(loader, node):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=True)
            if not isinstance(key, str) or key in result:
                raise ConfigError("Mapping keys must be unique strings")
            result[key] = loader.construct_object(value_node, deep=True)
        return result
    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    try:
        result = yaml.load(source, Loader=UniqueLoader)
        json.dumps(result, allow_nan=False)
    except (ValueError, TypeError, RecursionError, yaml.YAMLError) as error:
        raise ConfigError(str(error)) from error
    if not isinstance(result, dict):
        raise ConfigError("The document must be a mapping")
    return result

parse_yaml_subset = parse_yaml

def format_yaml(value):
    import yaml
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True)

def duration_seconds(value):
    value = str(value)
    if not re.fullmatch(r"(?:\d+-)?\d{1,4}:\d{2}:\d{2}", value):
        raise ConfigError("Time must use HH:MM:SS or D-HH:MM:SS")
    days, clock = value.split("-", 1) if "-" in value else (0, value)
    h, m, s = map(int, clock.split(":"))
    total = int(days) * 86400 + h * 3600 + m * 60 + s
    if m > 59 or s > 59 or total <= 0:
        raise ConfigError("Invalid or zero time limit")
    return total

def integer(value, field, low=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise ConfigError(f"{field} must be an integer of at least {low}")
    return value

def number(value, field, low=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < low:
        raise ConfigError(f"{field} must be a finite number of at least {low}")
    return value

def name(value, field):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ConfigError(f"Invalid {field}")
    return value

def normalize_profile(value):
    p = dict(value)
    allowed = {"name", "ssh_host", "ssh_options", "remote_root", "python", "account", "qos", "partitions", "queue_threshold_seconds", "normal_min_seconds", "checkpoint_margin_seconds", "dev_permission", "dev_run_kinds", "poll_seconds", "simulation"}
    if set(p) - allowed:
        raise ConfigError("Unknown profile fields: " + ", ".join(sorted(set(p) - allowed)))
    defaults = {"name": "Cluster", "ssh_options": [], "python": "python3", "account": "", "qos": "", "queue_threshold_seconds": 10800, "normal_min_seconds": 7200, "checkpoint_margin_seconds": 180, "dev_permission": "", "dev_run_kinds": ["evaluation"], "poll_seconds": 30, "simulation": False}
    p = {**defaults, **p}
    if not isinstance(p["simulation"], bool):
        raise ConfigError("simulation must be a boolean")
    if not isinstance(p.get("remote_root"), str) or not p["remote_root"].startswith("/") or any(c in p["remote_root"] for c in "\n\r\x00"):
        raise ConfigError("remote_root must be an absolute directory")
    if not p["simulation"]:
        host = p.get("ssh_host", "")
        if not re.fullmatch(r"[A-Za-z0-9_.@:-]+", host) or host.startswith("-"):
            raise ConfigError("Set ssh_host to an SSH alias or user@host")
    if not isinstance(p["python"], str) or not p["python"] or any(c in p["python"] for c in "\n\r\x00"):
        raise ConfigError("python must be a Python executable path")
    if not isinstance(p["ssh_options"], list) or any(not isinstance(x, str) or "\x00" in x for x in p["ssh_options"]):
        raise ConfigError("ssh_options must be a list of SSH arguments")
    for key in ("account", "qos"):
        if p[key]: name(p[key], key)
    for key in ("queue_threshold_seconds", "normal_min_seconds", "checkpoint_margin_seconds", "poll_seconds"):
        number(p[key], key, 0 if key != "poll_seconds" else 10)
    if not isinstance(p["dev_permission"], str): raise ConfigError("dev_permission must be a note")
    if not isinstance(p["dev_run_kinds"], list) or any(k not in ("train", "evaluation") for k in p["dev_run_kinds"]):
        raise ConfigError("dev_run_kinds must list the run kinds covered by the permission")
    if not isinstance(p.get("partitions"), list) or not p["partitions"]:
        raise ConfigError("Add at least one partition")
    seen = set()
    for part in p["partitions"]:
        if set(part) - {"name", "role", "time_limit", "concurrency_limit"}:
            raise ConfigError("Unknown partition field")
        name(part.get("name"), "partition")
        if part["name"] in seen: raise ConfigError("Duplicate partition")
        seen.add(part["name"])
        if part.get("role") not in ("normal", "short", "development"):
            raise ConfigError("Partition role must be normal, short, or development")
        duration_seconds(part.get("time_limit", ""))
        integer(part.setdefault("concurrency_limit", 1), "concurrency_limit")
    return p

def normalize_config(value):
    defaults = {"name": "Experiment", "kind": "evaluation", "command": [], "workers": 1, "concurrency": 1, "gpus": 1, "cpus": None, "task_count": 1, "environment": {}, "fingerprints": {}, "recipe": {}, "auto_continue": False, "resume_capable": False, "completion": "", "notes": "", "budget": {}, "estimates": {}}
    if set(value) - set(defaults):
        raise ConfigError("Unknown run fields: " + ", ".join(sorted(set(value) - set(defaults))))
    cfg = {**defaults, **value}
    if not isinstance(cfg["name"], str) or not 1 <= len(cfg["name"].strip()) <= 120:
        raise ConfigError("Run name must contain 1–120 characters")
    if cfg["kind"] not in ("train", "evaluation"): raise ConfigError("kind must be train or evaluation")
    if not isinstance(cfg["notes"], str) or not isinstance(cfg["completion"], str):
        raise ConfigError("notes and completion must be text")
    for key in ("workers", "concurrency", "gpus", "task_count"):
        integer(cfg[key], key, 0 if key == "gpus" else 1)
    if cfg["cpus"] is not None: integer(cfg["cpus"], "cpus")
    if cfg["concurrency"] > cfg["workers"]: raise ConfigError("Concurrency cannot exceed fixed worker count")
    if cfg["kind"] == "train" and cfg["workers"] != 1:
        raise ConfigError("Use a separate run for each independent training experiment")
    if not isinstance(cfg["command"], list) or not cfg["command"] or any(not isinstance(x, str) or "\x00" in x for x in cfg["command"]):
        raise ConfigError("command must be a nonempty list of string arguments")
    for flag in ("auto_continue", "resume_capable"):
        if not isinstance(cfg[flag], bool): raise ConfigError(f"{flag} must be a boolean")
    if cfg["auto_continue"] and (not cfg["resume_capable"] or not isinstance(cfg["completion"], str) or not cfg["completion"].strip()):
        raise ConfigError("Automatic continuation requires a resume-capable adapter and completion condition")
    for key in ("environment", "fingerprints", "recipe", "budget", "estimates"):
        if not isinstance(cfg[key], dict): raise ConfigError(f"{key} must be a mapping")
    for key, item in cfg["environment"].items():
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key) or key.startswith(("WM_", "SLURM_", "SBATCH_")) or not isinstance(item, (str, int, float, bool)):
            raise ConfigError("Environment names must be uppercase; WM_, SLURM_, and SBATCH_ are reserved")
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in cfg["fingerprints"].items()):
        raise ConfigError("Fingerprints must map artifact names to hashes or immutable revisions")
    if set(cfg["budget"]) - {"gpu_hours", "wall_hours"}: raise ConfigError("Budget supports gpu_hours and wall_hours")
    for key, item in cfg["budget"].items(): number(item, key, 0.000001)
    if set(cfg["estimates"]) - {"seconds_per_task", "model_load_seconds", "preparation_seconds", "restore_seconds"}:
        raise ConfigError("Unknown estimate field")
    for key, item in cfg["estimates"].items(): number(item, key)
    json.dumps(cfg, allow_nan=False)
    return cfg

def experiment_identity(cfg):
    return digest({k: cfg[k] for k in ("kind", "command", "workers", "gpus", "task_count", "environment", "fingerprints", "recipe", "completion")})

def estimate_plans(cfg, partitions, history=(), remaining=None):
    remaining = cfg["task_count"] if remaining is None else remaining
    result = []
    for part in partitions:
        estimates = dict(cfg["estimates"])
        matches = [h for h in history if h.get("partition") == part["name"] and h.get("identity") == experiment_identity(cfg)]
        for key in ("seconds_per_task", "model_load_seconds", "preparation_seconds", "restore_seconds"):
            values = sorted(h[key] for h in matches if isinstance(h.get(key), (int, float)))
            if values: estimates[key] = values[len(values) // 2]
        concurrency = min(cfg["concurrency"], part.get("concurrency_limit", 1))
        queue = part.get("queue_seconds")
        overheads = [estimates.get(k) for k in ("model_load_seconds", "preparation_seconds", "restore_seconds")]
        # Missing measurements are not evidence of free staging/loading/restoration.
        # Adapters with no such work can explicitly declare a zero estimate.
        startup = None if any(value is None for value in overheads) else sum(overheads)
        rate = estimates.get("seconds_per_task")
        execution = None if rate is None else math.ceil(remaining / concurrency) * rate
        useful = None if startup is None else part.get("usable_seconds", duration_seconds(part["time_limit"])) - startup
        slices = None if execution is None or useful is None or useful <= 0 else max(1, math.ceil(execution / useful))
        finish = None if queue is None or execution is None or slices is None else execution + slices * (startup + queue)
        result.append({**part, "concurrency": concurrency, "queue_seconds": queue, "startup_seconds": startup, "execution_seconds": execution, "estimated_slices": slices, "first_result_seconds": None if queue is None or rate is None or startup is None else queue + startup + rate, "finish_seconds": finish, "estimate_source": "observed matching run" if matches else "configured runtime estimates; live scheduler wait"})
    return sorted(result, key=lambda r: (
        r["finish_seconds"] is None, r["finish_seconds"] or 0,
        r["queue_seconds"] is None, r["queue_seconds"] or 0, r["name"]))
