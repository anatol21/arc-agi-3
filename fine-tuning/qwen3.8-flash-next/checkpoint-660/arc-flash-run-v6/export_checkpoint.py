"""Finish a verified adapter export without executing another training update."""
import json
import shutil
from pathlib import Path

if __package__:
    from .run_checks import atomic_json, checkpoint_files, snapshot
else:
    from run_checks import atomic_json, checkpoint_files, snapshot


def finish_export(output, checkpoint, run, config):
    checkpoint_files(checkpoint)
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if state["global_step"] != 660:
        raise RuntimeError("Export recovery is only for the completed, unchanged 660-update recipe")
    names = ["adapter_model.safetensors", "adapter_config.json"]
    # Adapter weights are certified against the actual live state at checkpoint time.
    for name in names:
        temporary = output / (name + ".export.tmp")
        shutil.copyfile(checkpoint / name, temporary)
        temporary.replace(output / name)
    identity = json.loads((run / "model-verified.json").read_text())
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "config.json", "generation_config.json"):
        # Preserve the actual training pad token and serialized tokenizer settings.
        path = checkpoint / name
        if not path.is_file():
            path = Path(config["base_model"]) / name
        if path.is_file():
            shutil.copyfile(path, output / name)
            names.append(name)
    if (checkpoint / "chat_templates").is_dir():
        shutil.copytree(checkpoint / "chat_templates", output / "chat_templates", dirs_exist_ok=True)
        names.extend(str(path.relative_to(output)) for path in (output / "chat_templates").rglob("*") if path.is_file())
    atomic_json(output / "base-model-identity.json", identity)
    names.append("base-model-identity.json")
    receipt = {"global_step": state["global_step"], "base_revision": identity["revision"],
               "source_checkpoint": str(checkpoint), "scope": "verified adapter export; full inference/restarted continuation not demonstrated",
               "files": snapshot(output, [output / name for name in names])}
    atomic_json(output / "export-validation.json", receipt)
    atomic_json(output / "run-status.json", {"phase": "complete", "global_step": state["global_step"],
                                             "last_valid_checkpoint": str(checkpoint), "export_validated": True})
