"""Pin the observed stack and verify local model bytes against Hub download metadata."""
from __future__ import annotations
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path

if __package__:
    from .run_checks import digest
else:
    from run_checks import digest

REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
CCE_COMMIT = "f2d026e6961cb4325a6e3d316461235aeeab0ac2"
CCE_UTILS_SHA256 = "a5d601658e46300709c4ec816b54f4f35f56accc35851c59c5d5005e0d938dfd"
VERSIONS = {"torch": "2.13.0+cu126", "transformers": "5.17.0", "axolotl": "0.20.0",
            "peft": "0.21.0", "accelerate": "1.15.0", "datasets": "4.8.4",
            "bitsandbytes": "0.50.2", "cut-cross-entropy": "25.5.2", "triton": "3.7.1",
            "flash-linear-attention": "0.5.2", "fla-core": "0.5.2"}
SOURCES = {"transformers": ["trainer.py", "models/qwen4_exp/modeling_qwen4_exp.py"],
           "axolotl": ["core/trainers/base.py", "core/trainers/mixins/rng_state_loader.py", "monkeypatch/moe_quant.py",
                       "core/trainers/mixins/layer_offloading.py", "core/trainers/mixins/activation_checkpointing.py"],
           "accelerate": ["hooks.py"], "cut_cross_entropy": ["transformers/utils.py", "transformers/qwen4_exp.py", "cce.py"]}


def environment_identity():
    versions = {name: metadata.version(name) for name in VERSIONS}
    if versions != VERSIONS:
        raise RuntimeError(f"Installed training stack changed: {versions}; expected {VERSIONS}")
    dist = metadata.distribution("cut-cross-entropy")
    source = json.loads(dist.read_text("direct_url.json") or "{}")
    if source.get("vcs_info", {}).get("commit_id") != CCE_COMMIT:
        raise RuntimeError("Installed CCE source commit does not match the reviewed fork")
    files = {}
    for package, relative in SOURCES.items():
        spec = importlib.util.find_spec(package)
        root = Path(spec.origin).parent
        for name in relative:
            path = root / name
            files[str(path)] = digest(path)
            if package == "cut_cross_entropy" and name == "transformers/utils.py" and files[str(path)] != CCE_UTILS_SHA256:
                raise RuntimeError("Installed CCE normalization code differs from the reviewed source")
    return {"versions": versions, "cce_source": source, "source_sha256": files}


def verify_environment(expected):
    if environment_identity() != expected:
        raise RuntimeError("Installed code changed since CPU staging")


def extend_environment_identity(previous):
    """Keep all previous pins while recording the newly exercised mixin sources."""
    current = environment_identity()
    if (current["versions"] != previous["versions"] or current["cce_source"] != previous["cce_source"]
        or any(current["source_sha256"].get(path) != sha for path, sha in previous["source_sha256"].items())):
        raise RuntimeError("Installed code changed since the successful v4 staging")
    return current


def model_paths(model):
    return sorted(path for path in model.rglob("*") if path.is_file()
                  and ".cache" not in path.relative_to(model).parts and ".git" not in path.relative_to(model).parts)


def verify_model_files(model: Path):
    """One full CPU-side read of each file; no network and no GPU allocation."""
    model = model.resolve()
    files = {}
    for path in model_paths(model):
        name = str(path.relative_to(model))
        if not path.resolve().is_relative_to(model):
            raise RuntimeError(f"Model file escapes its directory: {name}")
        meta = model / ".cache/huggingface/download" / (name + ".metadata")
        lines = meta.read_text().splitlines()
        if len(lines) < 3 or lines[0] != REVISION:
            raise RuntimeError(f"Model download metadata has the wrong revision: {name}")
        etag = lines[1].strip('"')
        if len(etag) == 64 and all(c in "0123456789abcdef" for c in etag):
            actual = digest(path)
            sha = actual
        elif len(etag) == 40 and all(c in "0123456789abcdef" for c in etag):
            sha = digest(path)
            checksum = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum.update(chunk)
            actual = checksum.hexdigest()
        else:
            raise RuntimeError(f"Unrecognized Hub content digest: {name}")
        if actual != etag:
            raise RuntimeError(f"Model content checksum failed: {name}")
        files[name] = {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
                       "sha256": sha, "hub_content_digest": etag}
        print("MODEL_FILE_VERIFIED " + name, flush=True)
    if "config.json" not in files or "tokenizer_config.json" not in files or not any(name.endswith(".safetensors") for name in files):
        raise RuntimeError("Incomplete local model/tokenizer inventory")
    config = json.loads((model / "config.json").read_text())
    if config.get("model_type") != "qwen4_exp":
        raise RuntimeError("Unexpected Flash architecture")
    index_path = model / "model.safetensors.index.json"
    if index_path.exists():
        shards = set(json.loads(index_path.read_text())["weight_map"].values())
        if not shards or not shards.issubset(files):
            raise RuntimeError("Missing indexed model shards")
    return {"model_id": "Qwen/Qwen3.8-Flash-Next", "revision": REVISION, "files": files,
            "verification": "all bytes matched local Hub download digests during CPU staging; runtime checks size/mtime plus small-file hashes"}


def verify_model_inventory(model, expected):
    paths = {str(path.relative_to(model)): path for path in model_paths(model)}
    if set(paths) != set(expected["files"]):
        raise RuntimeError("Model inventory changed since CPU staging")
    for name, path in paths.items():
        info = expected["files"][name]
        stat = path.stat()
        if stat.st_size != info["bytes"] or stat.st_mtime_ns != info["mtime_ns"]:
            raise RuntimeError(f"Model file changed since content verification: {name}")
        if not name.endswith((".safetensors", ".bin")) and digest(path) != info["sha256"]:
            raise RuntimeError(f"Model/tokenizer metadata checksum changed: {name}")
