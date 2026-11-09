"""Keep Qwen's large PLE lookup outside both loading and dispatch placement."""
from __future__ import annotations

import re

PLE_SUFFIX = "ple.ple_embedding.ngram_embedding.weight"
BUFFER_NAMES = {"layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"}


def contains(module: str, name: str) -> bool:
    return module == "" or name == module or name.startswith(module + ".")


def mapped_device(placement: dict, name: str):
    matches = [key for key in placement if contains(key, name)]
    return placement[max(matches, key=len)] if matches else "cpu"


def exclude_ple(model, original: dict) -> tuple[dict, dict]:
    """Split ancestor blocks; leaving only a CPU leaf out of a root map is insufficient."""
    if original.get("") != 0 or set(original.values()) != {0, 1, 2}:
        raise RuntimeError("Expected the original explicit three-GPU map")
    if PLE_SUFFIX not in (getattr(model, "_no_placement_params", None) or []):
        raise RuntimeError("Installed model does not support deliberately unplaced PLE weights")
    parameters = dict(model.named_parameters())
    tables = [name for name in parameters if name.endswith(PLE_SUFFIX)]
    if len(tables) != 1:
        raise RuntimeError(f"Expected exactly one large PLE table, found {tables}")
    table = tables[0]
    excluded = table.rsplit(".", 2)[0]
    excluded_module = model.get_submodule(excluded)
    if type(excluded_module).__name__ != "Qwen4ExpTextNGramEmbedding":
        raise RuntimeError("Unexpected PLE lookup module")
    if [name for name in parameters if contains(excluded, name)] != [table]:
        raise RuntimeError("Excluding the PLE lookup would also leave other parameters on CPU")
    if set(dict(excluded_module.named_buffers(recurse=False))) != BUFFER_NAMES:
        raise RuntimeError("The PLE lookup's small-buffer contract changed")
    placement = {}

    def visit(name, module):
        if name == excluded:
            return
        device = mapped_device(original, name)
        different_descendant = any(contains(name, key) and value != device
                                   for key, value in original.items())
        if not contains(name, excluded) and not different_descendant:
            placement[name] = device
            return
        if list(module.named_parameters(recurse=False)) or list(module.named_buffers(recurse=False)):
            raise RuntimeError(f"Cannot safely split an ancestor with direct tensors: {name}")
        for child_name, child in module.named_children():
            visit(f"{name}.{child_name}" if name else child_name, child)

    visit("", model)
    for name in parameters:
        expected = "cpu" if name == table else mapped_device(original, name)
        if mapped_device(placement, name) != expected:
            raise RuntimeError(f"Corrected device map changed another parameter: {name}")
    if any(contains(key, table) for key in placement):
        raise RuntimeError("A placement ancestor still covers the PLE table")
    report = {"cpu_ple_parameter": table, "cpu_ple_module": excluded,
              "ple_buffer_gpu": mapped_device(original, table),
              "ple_buffer_names": sorted(BUFFER_NAMES), "placement_blocks": len(placement),
              "table_bf16_GiB": round(parameters[table].numel() * 2 / 2**30, 3)}
    return placement, report


def probe_loading_and_hooks(model, placement: dict, report: dict) -> dict:
    """Exercise installed Transformers resolution and Accelerate hook setup on meta tensors.

    Intercept only tensor movement: no real weights, GPU allocation, or forward pass.
    A broad ancestor hook would otherwise undo the corrected loading placement.
    """
    from unittest.mock import patch
    import accelerate.hooks as hooks
    import transformers.core_model_loading as loading

    table = report["cpu_ple_parameter"]
    parameters = dict(model.named_parameters())
    for name in parameters:
        if loading.get_device(placement, name) != mapped_device(placement, name):
            raise RuntimeError(f"Installed Transformers resolves placement differently: {name}")
    if str(loading.get_device(placement, table)) != "cpu":
        raise RuntimeError("Installed loader would still put the PLE table on a GPU")
    modules = {id(module): name for name, module in model.named_modules()}
    moves = {}

    def record(module, name, device, **kwargs):
        prefix = modules[id(module)]
        full = f"{prefix}.{name}" if prefix else name
        if full == table:
            raise RuntimeError("An Accelerate placement hook would move the PLE table")
        if full in parameters and device != mapped_device(placement, full):
            raise RuntimeError(f"An Accelerate hook would change placement: {full}")
        moves[full] = device

    # Match dispatch_model's multi-GPU, no-CPU-entry execution/offload dictionaries.
    execution = {**placement, "": 0}
    offload = {name: False for name in placement}
    with patch.object(hooks, "set_module_tensor_to_device", side_effect=record):
        hooks.attach_align_device_hook_on_blocks(model, execution_device=execution,
                                                offload=offload, tied_params_map={})
    missing = set(parameters) - {table} - set(moves)
    if missing:
        raise RuntimeError(f"GPU placement hooks missed parameters: {sorted(missing)[:5]}")
    return {"loader_ple_device": "cpu", "hook_ple_moves": 0,
            "hook_parameter_coverage": len(parameters) - 1}


def prepare_placement(model_path, original: dict) -> tuple[dict, dict]:
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    if config.model_type != "qwen4_exp":
        raise RuntimeError("Unexpected model architecture")
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(config, torch_dtype=torch.bfloat16)
    if any(parameter.device.type != "meta" for parameter in model.parameters()):
        raise RuntimeError("Placement preflight must use meta tensors only")
    placement, report = exclude_ple(model, original)
    report.update(probe_loading_and_hooks(model, placement, report))
    del model
    return placement, report


def place_small_ple_buffers(model) -> list[str]:
    """Hashing buffers follow their layer; the frozen 95-GiB lookup stays on CPU."""
    moved = []
    for name, module in model.named_modules():
        if type(module).__name__ != "Qwen4ExpTextNGramEmbedding":
            continue
        match = re.search(r"(?:^|\.)layers\.(\d+)\.ple\.ple_embedding$", name)
        if match is None or int(match[1]) >= 48:
            raise RuntimeError("Unexpected PLE lookup location")
        if module.ngram_embedding.weight.device.type != "cpu":
            raise RuntimeError("PLE lookup left CPU before the first forward")
        buffers = dict(module.named_buffers(recurse=False))
        if set(buffers) != BUFFER_NAMES or any(buffer.numel() > 1000000 for buffer in buffers.values()):
            raise RuntimeError("Unexpected PLE hashing buffers")
        gpu = int(match[1]) // 16
        for key, buffer in buffers.items():
            module._buffers[key] = buffer.to(f"cuda:{gpu}")
            moved.append(f"{name}.{key}")
    if len(moved) != 3:
        raise RuntimeError("Expected exactly three small PLE hashing buffers")
    return moved
