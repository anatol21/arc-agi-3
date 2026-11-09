#!/usr/bin/env python3
"""Inspect the final certified checkpoint/export on CPU; no allocation is submitted."""
import argparse
import json
from pathlib import Path

if __package__:
    from .common import WORKSPACE, REVISION, checkpoint_files, verify_snapshot
    from .protocol import digest
else:
    from common import WORKSPACE, REVISION, checkpoint_files, verify_snapshot
    from protocol import digest


def inspect_complete_run(workspace):
    import torch
    from safetensors import safe_open
    output = Path(workspace) / "runs/arc-flash-v6"
    status = json.loads((output / "run-status.json").read_text())
    checkpoint = output / "checkpoint-660"
    if (status.get("phase") != "complete" or status.get("global_step") != 660
        or status.get("export_validated") is not True
        or Path(status["last_valid_checkpoint"]).resolve() != checkpoint.resolve()):
        raise RuntimeError("v6 has no confirmed complete 660-step export; inspect run-status.json")
    checkpoint_files(checkpoint)
    export = json.loads((output / "export-validation.json").read_text())
    if (export.get("global_step") != 660 or export.get("base_revision") != REVISION
        or Path(export["source_checkpoint"]).resolve() != checkpoint.resolve()):
        raise RuntimeError("Export receipt differs from the final checkpoint or base revision")
    verify_snapshot(output, export["files"])
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if digest(output / name) != digest(checkpoint / name):
            raise RuntimeError("Exported adapter differs from checkpoint-660")
    count = nonzero = tensors = 0
    dtypes, groups = set(), {}
    with safe_open(str(checkpoint / "adapter_model.safetensors"), framework="pt", device="cpu") as saved:
        for name in saved.keys():
            value = saved.get_tensor(name)
            if "lora_" not in name or not value.is_floating_point() or not torch.isfinite(value).all().item():
                raise RuntimeError(f"Invalid learned adapter tensor: {name}")
            n = value.numel()
            count += n
            nonzero += torch.count_nonzero(value).item()
            tensors += 1
            dtypes.add(str(value.dtype))
            family = "gdn" if ".linear_attn." in name else "shared" if ".shared_expert." in name else "qsa"
            groups[family] = groups.get(family, 0) + n
    if count != 33478656 or not nonzero:
        raise RuntimeError("Learned adapter coverage differs from the completed training recipe")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if state["global_step"] != 660 or state.get("max_steps") != 660:
        raise RuntimeError("Trainer state does not show completion of all planned updates")
    evaluations = [{key: row[key] for key in ("step", "epoch", "eval_loss", "eval_runtime") if key in row}
                   for row in state.get("log_history", []) if "eval_loss" in row]
    train_losses = [{key: row[key] for key in ("step", "epoch", "loss", "grad_norm", "learning_rate") if key in row}
                    for row in state.get("log_history", []) if "loss" in row]
    return {"training_complete": True, "global_step": 660, "epoch": state.get("epoch"),
            "checkpoint": str(checkpoint), "export": str(output), "export_validated": True,
            "checkpoint_receipt_sha256": digest(checkpoint / "checkpoint-complete.json"),
            "export_receipt_sha256": digest(output / "export-validation.json"),
            "adapter_sha256": digest(checkpoint / "adapter_model.safetensors"),
            "adapter": {"parameters": count, "tensors": tensors, "nonzero_values": nonzero,
                        "dtypes": sorted(dtypes), "families": groups},
            "evaluations": evaluations, "first_training_metrics": train_losses[:1],
            "last_training_metrics": train_losses[-1:], "accuracy_measured": False,
            "limits": "checkpoint/export integrity and finite learned tensors; validation loss is not ARC solve rate"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    args = parser.parse_args()
    print(json.dumps(inspect_complete_run(args.workspace), indent=2))
