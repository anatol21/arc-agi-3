"""Validate ordinary adapter/optimizer tensors; never interpret packed base weights."""
from pathlib import Path


def assert_finite_tree(torch, value, name="state"):
    if torch.is_tensor(value):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all().item():
            raise RuntimeError(f"Nonfinite saved tensor: {name}")
    elif isinstance(value, dict):
        for key, item in value.items():
            assert_finite_tree(torch, item, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite_tree(torch, item, f"{name}.{index}")
    elif isinstance(value, float):
        import math
        if not math.isfinite(value):
            raise RuntimeError(f"Nonfinite saved scalar: {name}")


def assert_same_tree(torch, saved, live, name="state"):
    if torch.is_tensor(live):
        if not torch.is_tensor(saved) or saved.dtype != live.dtype or saved.shape != live.shape:
            raise RuntimeError(f"Saved tensor metadata differs: {name}")
        if not torch.equal(saved.cpu(), live.detach().cpu()):
            raise RuntimeError(f"Saved tensor differs from live training state: {name}")
    elif isinstance(live, dict):
        if not isinstance(saved, dict) or saved.keys() != live.keys():
            raise RuntimeError(f"Saved state keys differ: {name}")
        for key in live:
            assert_same_tree(torch, saved[key], live[key], f"{name}.{key}")
    elif isinstance(live, (list, tuple)):
        if not isinstance(saved, type(live)) or len(saved) != len(live):
            raise RuntimeError(f"Saved state structure differs: {name}")
        for index, item in enumerate(live):
            assert_same_tree(torch, saved[index], item, f"{name}.{index}")
    elif saved != live:
        raise RuntimeError(f"Saved value differs: {name}")


def validate_adapters(torch, path, model):
    import json
    from safetensors.torch import load_file
    from peft import get_peft_model_state_dict
    config = json.loads((Path(path) / "adapter_config.json").read_text())
    live_config = model.peft_config["default"].to_dict()
    for key in ("peft_type", "r", "lora_alpha", "lora_dropout", "bias", "target_modules", "use_dora",
                "modules_to_save", "fan_in_fan_out", "base_model_name_or_path"):
        left, right = config.get(key), live_config.get(key)
        if key == "target_modules":
            left = sorted(left) if isinstance(left, (list, set)) else left
            right = sorted(right) if isinstance(right, (list, set)) else right
        if left != right:
            raise RuntimeError(f"Saved adapter configuration differs: {key}")
    # These are adapter-only state dictionaries, not the giant quantized base.
    # Avoid traversing/materializing the frozen 180B base just to validate LoRA.
    adapter_state = {name: value.detach() for name, value in model.named_parameters() if value.requires_grad}
    live = get_peft_model_state_dict(model, state_dict=adapter_state, save_embedding_layers=False)
    saved = load_file(str(Path(path) / "adapter_model.safetensors"), device="cpu")
    assert_finite_tree(torch, saved, "adapter")
    assert_same_tree(torch, saved, live, "adapter")
    return {"tensors": len(saved), "parameters": sum(t.numel() for t in saved.values()),
            "dtypes": sorted({str(t.dtype) for t in saved.values()}), "roundtrip_equal": True}


def validate_checkpoint(torch, path, trainer):
    path = Path(path)
    step = trainer.state.global_step
    if step <= 0 or path.name != f"checkpoint-{step}":
        raise RuntimeError("Checkpoint step does not match the actual training update")
    adapters = validate_adapters(torch, path, trainer.model)
    state = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True)
    assert_finite_tree(torch, state, "optimizer")
    if not state.get("state"):
        raise RuntimeError("Optimizer state is empty after a real update")
    assert_same_tree(torch, state, trainer.optimizer.state_dict(), "optimizer")
    scheduler = torch.load(path / "scheduler.pt", map_location="cpu", weights_only=True)
    assert_finite_tree(torch, scheduler, "scheduler")
    assert_same_tree(torch, scheduler, trainer.lr_scheduler.state_dict(), "scheduler")
    rng = torch.load(path / "arc_cuda_rng_all.pt", map_location="cpu", weights_only=True)
    current = torch.cuda.get_rng_state_all()
    if len(rng) != 3:
        raise RuntimeError("Checkpoint must contain all three CUDA RNG streams")
    assert_same_tree(torch, rng, current, "cuda_rng")
    # Decode HF's ordinary RNG payload too. Use its numpy allowlist, without
    # restoring/changing any global generator while validating a save.
    from transformers.trainer import safe_globals
    import numpy as np
    import random
    with safe_globals():
        ordinary_rng = torch.load(path / "rng_state.pth", map_location="cpu", weights_only=True)
    random.Random().setstate(ordinary_rng["python"])
    np.random.RandomState().set_state(ordinary_rng["numpy"])
    torch.Generator(device="cpu").set_state(ordinary_rng["cpu"])
    assert_same_tree(torch, ordinary_rng["cpu"], torch.random.get_rng_state(), "cpu_rng")
    assert_same_tree(torch, ordinary_rng["cuda"], current[torch.cuda.current_device()], "current_cuda_rng")
    import json
    tokenizer = trainer.processing_class
    if json.loads((path / "tokenizer.json").read_text()) != json.loads(tokenizer.backend_tokenizer.to_str()):
        raise RuntimeError("Saved tokenizer backend differs from the live training tokenizer")
    tokenizer_config = json.loads((path / "tokenizer_config.json").read_text())
    for key in ("pad_token", "eos_token"):
        actual = tokenizer_config.get(key)
        if isinstance(actual, dict):
            actual = actual.get("content")
        if actual != getattr(tokenizer, key):
            raise RuntimeError(f"Saved tokenizer configuration differs: {key}")
    return {"schema_version": 1, "global_step": step, "adapter_roundtrip": adapters,
            "optimizer_roundtrip": True, "scheduler_roundtrip": True, "cuda_rng_streams": 3,
            "tokenizer_roundtrip": True,
            "scope": "saved tensors decoded and equal to live state; resumed continuation not yet demonstrated"}
