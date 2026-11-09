"""Tiny CPU SGD example demonstrating complete restart state, not an LLM training recipe."""
import json
import math
import os
import random
import time
from sdk import Adapter

def tuples(value):
    return tuple(tuples(x) for x in value) if isinstance(value, list) else value

def main():
    adapter = Adapter()
    cfg = json.loads((adapter.root / "manifest.json").read_text())["config"]
    target = cfg["task_count"]
    rng = random.Random(cfg["recipe"].get("seed", 42))
    with adapter.phase("restore"):
        state = adapter.load_checkpoint()
        if state is None:
            state = {"global_step": 0, "weights": [0.0, 0.0], "optimizer": {"velocity": [0.0, 0.0]}, "scheduler": {"step": 0}, "sampler": {"position": 0}, "rng": rng.getstate()}
        if not all(k in state for k in ("weights", "optimizer", "scheduler", "sampler", "rng", "global_step")):
            raise ValueError("Training checkpoint lacks full restart state")
        if state["scheduler"]["step"] != state["global_step"] or state["sampler"]["position"] != state["global_step"]:
            raise ValueError("Training step/sampler/scheduler state disagree")
        rng.setstate(tuples(state["rng"]))
    delay = float(cfg["recipe"].get("step_seconds", 0))
    while state["global_step"] < target and not adapter.stopping():
        x = rng.uniform(-1, 1)
        prediction = state["weights"][0] * x + state["weights"][1]
        error = prediction - (2 * x - .4)
        lr = .05 / math.sqrt(1 + state["global_step"] / 100)
        gradients = [2 * error * x, 2 * error]
        for j in range(2):
            state["optimizer"]["velocity"][j] = .9 * state["optimizer"]["velocity"][j] + gradients[j]
            state["weights"][j] -= lr * state["optimizer"]["velocity"][j]
        state["global_step"] += 1
        state["scheduler"]["step"] = state["global_step"]
        state["sampler"]["position"] = state["global_step"]
        state["rng"] = rng.getstate()
        adapter.checkpoint(state)
        adapter.metric(step=state["global_step"], loss=error**2, completed=state["global_step"], seconds_per_task=max(.00001, delay))
        if delay: time.sleep(delay)
    adapter.finish("complete" if state["global_step"] == target else "partial_checkpointed", state["global_step"], state["global_step"])

if __name__ == "__main__": main()
