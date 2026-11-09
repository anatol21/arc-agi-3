"""Normalize the loader's path argument and validate the installed schema on CPU."""
import json
from pathlib import Path


def load_inference_config(path, loader=None):
    if loader is None:
        from axolotl.cli.config import load_cfg
        loader = load_cfg
    # Axolotl accepts Path in its API annotation, but stores the argument in a
    # Pydantic field declared str | None without converting it. Pass str explicitly.
    return loader(str(Path(path)))


def schema_preflight(path):
    from axolotl.cli.config import prepare_plugins
    from axolotl.utils.config import validate_config
    from axolotl.utils.dict import DictDefault
    path = Path(path)
    original = json.loads(path.read_text())
    config = DictDefault(original)
    config.axolotl_config_path = str(path)
    prepare_plugins(config)
    validated = validate_config(config, capabilities=None, env_capabilities=None)
    if validated.axolotl_config_path != str(path):
        raise RuntimeError("Installed schema changed the configuration path")
    for key in ("base_model", "lora_model_dir", "tokenizer_config", "device_map", "output_dir"):
        if validated[key] != original[key]:
            raise RuntimeError(f"Installed schema changed the reviewed inference setting: {key}")
    return {"status": "passed", "axolotl_config_path_type": "str",
            "validation": "installed Axolotl input/plugin schema; capabilities omitted on CPU",
            "model_loaded": False, "gpu_capabilities_validated": False}
