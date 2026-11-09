"""Finite, checkpointed CPU work for testing orchestration without requesting GPUs."""
import json
import os
import time
from pathlib import Path
from sdk import Adapter, prepare_cache
from core import digest

def main():
    adapter = Adapter()
    manifest = json.loads((adapter.root / "manifest.json").read_text())
    cfg = manifest["config"]
    if not manifest["profile"]["simulation"] and not cfg["recipe"].get("allow_cluster_smoke", False):
        raise ValueError("Set recipe.allow_cluster_smoke explicitly before running this CPU smoke adapter on a cluster")
    state = adapter.load_checkpoint({"items": {}, "progress": 0})
    chunks = int(cfg["recipe"].get("chunks_per_item", 3))
    delay = float(cfg["recipe"].get("chunk_seconds", 0.08))
    if chunks < 1 or delay <= 0: raise ValueError("Invalid demo workload")
    cache = adapter.root.parent.parent / "cluster-cache" / digest(cfg["fingerprints"])
    def prepare(directory):
        (directory / "demo.txt").write_text("prepared once\n")
        return ["demo.txt"]
    start = time.time()
    hit = prepare_cache(cache, cfg["fingerprints"], prepare)
    adapter.metric(preparation_seconds=time.time() - start, cache_hit=hit)
    indices = [i for i in range(cfg["task_count"]) if adapter.assigned(i)]
    began = time.time()
    for i in indices:
        key = str(i)
        while state["items"].get(key, 0) < chunks:
            if adapter.stopping():
                adapter.checkpoint(state)
                adapter.finish("partial_checkpointed", state["progress"], sum(v == chunks for v in state["items"].values()))
                return
            time.sleep(delay)
            state["items"][key] = state["items"].get(key, 0) + 1
            state["progress"] += 1
            adapter.checkpoint(state)
            completed = sum(v == chunks for v in state["items"].values())
            adapter.metric(progress=state["progress"], completed=completed, loss=1 / (state["progress"] + 1), seconds_per_task=chunks * delay, tokens_per_second=1 / delay)
            print(f"worker={adapter.worker} item={i} chunk={state['items'][key]}/{chunks}", flush=True)
    adapter.finish("complete", state["progress"], len(indices))
    adapter.metric(execution_seconds=time.time() - began, completed=len(indices))

if __name__ == "__main__": main()
