"""Four-GPU Axolotl BF16 base loader for the audited Qwen flash checkpoint."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile

LAYOUT = Path(__file__).with_name("qwen_layout.json")
PLE_SUFFIX = "ple.ple_embedding.ngram_embedding.weight"
PLE_BUFFERS = {"layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"}


def expected_device(name, layout):
    matches = [key for key in layout if name == key or name.startswith(key + ".")]
    return f"cuda:{layout[max(matches, key=len)]}" if matches else "cpu"


def cached_snapshot(source, revision):
    """Return one exact cached Hub snapshot. Flat local mirrors keep Hub download metadata."""
    path = Path(source)
    if not path.is_dir():
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(repo_id=source, revision=revision, local_files_only=True))
    path = path.resolve()
    if path.parent.name == "snapshots":
        if path.name != revision:
            raise ValueError("Local Hub snapshot is from a different model revision")
        return path
    index = path / "model.safetensors.index.json"
    files = {"config.json", index.name, *json.loads(index.read_bytes())["weight_map"].values()}
    for name in sorted(files):
        entry = Path(name)
        if entry.is_absolute() or ".." in entry.parts or not (path / entry).is_file():
            raise ValueError(f"Missing or invalid checkpoint entry: {name}")
        metadata = path / ".cache/huggingface/download" / (name + ".metadata")
        if not metadata.is_file() or metadata.read_text().splitlines()[:1] != [revision]:
            raise ValueError(f"Cannot confirm the pinned revision of local checkpoint file {name}")
    return path


def verify_geometry(model, reference):
    shapes = {n: list(p.shape) for n, p in model.named_parameters()}
    digest = hashlib.sha256(json.dumps(shapes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if len(shapes) != reference["parameter_count"] or digest != reference["shape_sha256"]:
        raise ValueError("Model parameter names or shapes differ from the audited checkpoint")
    on_host = [name for name in shapes if expected_device(name, reference["device_map"]) == "cpu"]
    if len(on_host) != 1 or not on_host[0].endswith(PLE_SUFFIX):
        raise ValueError("The explicit device map must leave exactly the PLE lookup weight on CPU")


def place_ple_buffers(model, layout):
    moved = []
    for name, module in model.named_modules():
        if type(module).__name__ != "Qwen4ExpTextNGramEmbedding":
            continue
        if module.ngram_embedding.weight.device.type != "cpu":
            raise RuntimeError("The PLE embedding must stay in host RAM")
        buffers = dict(module.named_buffers(recurse=False))
        if set(buffers) != PLE_BUFFERS or any(buf.numel() > 1_000_000 for buf in buffers.values()):
            raise RuntimeError("Unexpected PLE hashing buffers")
        device = expected_device(name.rsplit(".ple_embedding", 1)[0] + ".key_proj.weight", layout)
        if device == "cpu":
            raise RuntimeError("Missing PLE projection placement")
        for key, buf in buffers.items():
            module._buffers[key] = buf.to(device)
            moved.append(f"{name}.{key}")
    if len(moved) != 3:
        raise RuntimeError("Expected three PLE hashing buffers")


def verify_live(model, reference):
    import torch
    verify_geometry(model, reference)
    for flag in ("peft_config", "is_loaded_in_4bit", "is_loaded_in_8bit", "hf_quantizer", "_moe_experts_quantized"):
        if getattr(model, flag, None):
            raise ValueError(f"Unexpected quantizer or adapter: {flag}")
    for name, module in model.named_modules():
        if ("lora" in name.lower() or type(module).__module__.startswith(("peft.", "bitsandbytes."))
                or getattr(module, "parametrizations", None)):
            raise ValueError(f"Unexpected adapter or packed module: {name}")
    dtypes = Counter()
    for name, p in model.named_parameters():
        target = expected_device(name, reference["device_map"])
        if (str(p.device) != target or p.dtype not in (torch.bfloat16, torch.float32)
                or getattr(p, "quant_state", None) is not None or p.requires_grad):
            raise ValueError(f"Unexpected runtime weight: {name}, {p.dtype}, {p.device}")
        dtypes[str(p.dtype)] += 1
    if dict(dtypes) != reference["dtypes"]:
        raise ValueError(f"Runtime precision differs from the audited model: {dict(dtypes)}")
    return dict(quantized_parameters=0, adapter_tensors=0, base_tensors=sum(dtypes.values()), dtypes=dict(dtypes))


def load_model(spec):
    import torch
    from axolotl.cli.config import load_cfg
    from axolotl.cli.utils import load_model_and_tokenizer
    from transformers import AutoConfig

    reference = json.loads(LAYOUT.read_text())
    if torch.cuda.device_count() != 4 or spec["model_revision"] != reference["model_revision"]:
        raise ValueError("This loader requires four visible GPUs and the documented model revision")
    snapshot = cached_snapshot(spec["model"], reference["model_revision"])
    source = AutoConfig.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    if source.model_type != "qwen4_exp" or getattr(source, "quantization_config", None):
        raise ValueError("Expected the original unquantized qwen4_exp checkpoint")
    config = json.loads(Path(spec["axolotl_config"]).read_text())
    config = {k: v for k, v in config.items() if not k.startswith(("lora_", "peft_"))}
    tokenizer = spec["tokenizer"]
    if not Path(tokenizer).is_dir():
        from huggingface_hub import snapshot_download
        tokenizer = snapshot_download(repo_id=tokenizer, revision=spec["tokenizer_revision"], local_files_only=True)
    config.update(base_model=str(snapshot), tokenizer_config=tokenizer,
                  revision_of_model=reference["model_revision"], device_map=reference["device_map"],
                  bf16=True, adapter=None, lora_model_dir=None, load_in_4bit=False, load_in_8bit=False,
                  quantize_moe_experts=False, ple_cpu_offload=False, gradient_checkpointing=False,
                  experimental_skip_move_to_device=True, attn_needs_dtype_cast=True)
    with tempfile.TemporaryDirectory(prefix="arc-bf16-") as tmp:
        config["output_dir"] = tmp
        path = Path(tmp) / "inference.json"
        path.write_text(json.dumps(config))
        live = load_cfg(str(path))
        if any(live[k] for k in ("adapter", "lora_model_dir", "load_in_4bit", "load_in_8bit", "quantize_moe_experts")):
            raise ValueError("Loader settings enable quantization or an adapter")
        model, _, _ = load_model_and_tokenizer(cfg=live, inference=False)
    model.eval()
    model.requires_grad_(False)
    place_ple_buffers(model, reference["device_map"])
    print("ARC_LOADER", json.dumps(verify_live(model, reference)), flush=True)
    return model
