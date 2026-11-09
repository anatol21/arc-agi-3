"""Checkpoint one deterministic synthetic item per invocation; safe to exercise locally."""

import json
import os
from pathlib import Path
from checkpoint import read_progress, save_item

root = Path(os.environ["CAMPAIGN_DIR"])
identity = os.environ["CAMPAIGN_ID"]
config = json.loads(Path(os.environ["CAMPAIGN_CONFIG"]).read_bytes())
progress = read_progress(root, identity, set(config["items"]))
for item in config["items"]:
    if progress.get(item, {}).get("status") == "finished":
        continue
    save_item(
        root, identity, item, units=1, status="finished", payload={"length": len(item)}
    )
    print("Committed", item)
    break
