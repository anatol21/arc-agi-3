"""Offline checks for the published four-GPU Qwen placement and snapshot selection."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

if __package__:
    from .qwen_bf16 import LAYOUT, cached_snapshot, expected_device, verify_geometry
else:
    from qwen_bf16 import LAYOUT, cached_snapshot, expected_device, verify_geometry


class QwenLayoutTests(unittest.TestCase):
    def test_stage_boundaries_and_only_lookup_on_cpu(self):
        reference = json.loads(LAYOUT.read_text())
        layout = reference["device_map"]
        prefix = "model.language_model.layers."
        for idx in range(48):
            self.assertEqual(expected_device(f"{prefix}{idx}.mlp.weight", layout), f"cuda:{idx // 12}")
        table = prefix + "1.ple.ple_embedding.ngram_embedding.weight"
        projection = prefix + "1.ple.key_proj.weight"
        self.assertEqual(expected_device(table, layout), "cpu")
        self.assertEqual(expected_device(projection, layout), "cuda:0")

        shapes = {table: [12, 2], projection: [2, 2]}
        model = SimpleNamespace(named_parameters=lambda: [(n, SimpleNamespace(shape=s)) for n, s in shapes.items()])
        digest = hashlib.sha256(json.dumps(shapes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        reference.update(parameter_count=2, shape_sha256=digest)
        verify_geometry(model, reference)
        del layout[projection.removesuffix(".weight")]
        with self.assertRaisesRegex(ValueError, "PLE lookup weight"):
            verify_geometry(model, reference)

    def test_reject_local_revision_mismatch(self):
        revision = json.loads(LAYOUT.read_text())["model_revision"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap = root / "snapshots" / revision
            snap.mkdir(parents=True)
            self.assertEqual(cached_snapshot(str(snap), revision), snap.resolve())
            with self.assertRaisesRegex(ValueError, "different model revision"):
                cached_snapshot(str(snap), "0" * 40)
            mirror = root / "mirror"
            metadata = mirror / ".cache/huggingface/download"
            metadata.mkdir(parents=True)
            (mirror / "config.json").write_text("{}")
            (mirror / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"p": "shard.safetensors"}}))
            (mirror / "shard.safetensors").write_bytes(b"synthetic")
            for name in ("config.json", "model.safetensors.index.json", "shard.safetensors"):
                (metadata / (name + ".metadata")).write_text(revision + "\n")
            self.assertEqual(cached_snapshot(str(mirror), revision), mirror.resolve())
            (metadata / "shard.safetensors.metadata").write_text("0" * 40 + "\n")
            with self.assertRaisesRegex(ValueError, "revision"):
                cached_snapshot(str(mirror), revision)
