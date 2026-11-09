"""Run the configured workload with a single-writer lock and a durable output directory."""

import fcntl
import os
from pathlib import Path
import subprocess
import sys

from checkpoint import digest, read_progress
from scheduler import load_config


def main(config_file):
    path = Path(config_file).resolve()
    c = load_config(path)
    identity = digest(c)
    root = path.parent / ".campaigns" / identity
    root.mkdir(parents=True, exist_ok=True)
    with (root / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "STOP").exists():
            return
        states = read_progress(root, identity, set(c["items"]))
        if all(states.get(i, {}).get("status") == "finished" for i in c["items"]):
            return
        env = dict(
            os.environ,
            CAMPAIGN_DIR=str(root),
            CAMPAIGN_ID=identity,
            CAMPAIGN_CONFIG=str(path),
        )
        # Expose the same checkpoint helpers to a user-provided Python workload.
        env["PYTHONPATH"] = os.pathsep.join(
            v
            for v in (str(Path(__file__).resolve().parent), env.get("PYTHONPATH"))
            if v
        )
        subprocess.run(c["worker_command"], cwd=path.parent, env=env, check=True)


if __name__ == "__main__":
    main(sys.argv[1])
