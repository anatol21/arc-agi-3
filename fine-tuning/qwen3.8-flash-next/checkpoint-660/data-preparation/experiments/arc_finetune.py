"""Local ARC transduction: staged MLX QLoRA and isolated test-time adaptation.

The model backend receives PublicTask objects, which cannot contain test answers.
Only AnswerScorer reads the protected answers. Importing this module needs only
the standard library; MLX is imported when a model is actually loaded.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import heapq
import importlib.metadata
import inspect
import json
import math
import os
import platform
import random
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
from apps.arc_practice.server import (  # Reuse the benchmark's exact contract.
    BenchmarkError, DatasetSchemaError, parse_model_outputs, require_task,
    task_max_grid_side, task_prompt,
)
from scripts.build_local_training_archive import pair_hash, task_hash

PACKAGE = ROOT / "artifacts/releases/arc-synthetic-training-package-v1"
SCHEMA_VERSION = 1


def compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(compact(value).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def append_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(compact(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def restore_rng(mx, state) -> None:
    """Restore MLX's two-word global key through its public 64-bit seed API."""
    words = state.tolist()
    mx.random.seed((int(words[0]) << 32) | int(words[1]))


@dataclass(frozen=True)
class Config:
    run_name: str = "qwen3-8b-pilot-v1"
    model_id: str = "Qwen/Qwen3-8B-MLX-4bit"
    seed: int = 42
    synthetic_tasks: int | None = 2048
    synthetic_epochs: int = 1
    arc_epochs: int = 3
    synthetic_lr: float = 1e-4
    arc_lr: float = 5e-5
    rank: int = 16
    scale: float = 2.0
    dropout: float = 0.0
    num_layers: int = 16
    accumulation: int = 4
    max_seq_length: int = 32768
    max_output_tokens: int = 4096
    adaptation_steps: int = 16
    adaptation_lr: float = 1e-5
    checkpoint_every: int = 100
    memory_limit_gib: float = 36.0

    @classmethod
    def full(cls) -> Config:
        return cls(run_name="qwen3-8b-full-v1", synthetic_tasks=None)

    @property
    def run_dir(self) -> Path:
        return ROOT / "artifacts/arc_finetune" / self.run_name

    def validate(self) -> None:
        if Path(self.run_name).name != self.run_name or self.run_name in {".", ".."}:
            raise ValueError("run_name must be one directory name")
        for value in (self.accumulation, self.rank, self.num_layers, self.max_seq_length,
                      self.max_output_tokens, self.synthetic_epochs, self.arc_epochs,
                      self.checkpoint_every):
            if value <= 0:
                raise ValueError("Training sizes and budgets must be positive")
        if self.synthetic_tasks is not None and self.synthetic_tasks <= 0:
            raise ValueError("synthetic_tasks must be positive or None")
        if self.adaptation_steps < 0 or self.max_seq_length > 32768:
            raise ValueError("Invalid adaptation steps or native context budget")


@dataclass(frozen=True)
class PublicTask:
    task_id: str
    demonstrations: list[dict]
    test_inputs: list[list[list[int]]]

    @classmethod
    def from_record(cls, record: dict) -> PublicTask:
        return cls(record["task_id"], copy.deepcopy(record["train"]),
                   [copy.deepcopy(pair["input"]) for pair in record["test"]])

    def prompt(self) -> str:
        return task_prompt({"train": self.demonstrations,
                            "test": [{"input": grid} for grid in self.test_inputs]}, "")


@dataclass(frozen=True)
class Example:
    task: PublicTask
    outputs: list[list[list[int]]]

    @classmethod
    def from_record(cls, record: dict) -> Example:
        return cls(PublicTask.from_record(record), [p["output"] for p in record["test"]])

    @property
    def completion(self) -> str:
        return json.dumps({"outputs": self.outputs}, separators=(",", ":"))


class AnswerScorer:
    """Evaluator-owned answer storage. Never passed to model or adaptation code."""

    def __init__(self, records: Sequence[dict]):
        self.__answers = {r["task_id"]: [p["output"] for p in r["test"]] for r in records}
        self.__pair_hashes = {pair_hash(p) for r in records for split in ("train", "test")
                              for p in r[split]}

    def protected_pair_hashes(self) -> set[str]:
        return set(self.__pair_hashes)

    def score(self, task_id: str, raw_response: str) -> dict:
        answers = self.__answers[task_id]
        try:
            outputs = parse_model_outputs(raw_response, len(answers))
        except (BenchmarkError, DatasetSchemaError) as error:
            return {"correct": False, "tests_correct": 0, "test_count": len(answers),
                    "malformed": True, "error": str(error), "prediction": None}
        correct = [prediction == expected for prediction, expected in zip(outputs, answers, strict=True)]
        return {"correct": all(correct), "tests_correct": sum(correct), "test_count": len(answers),
                "malformed": False, "error": None, "prediction": {"outputs": outputs}}


@dataclass
class EvaluationSet:
    tasks: list[PublicTask]
    scorer: AnswerScorer
    identity: str

    @classmethod
    def from_records(cls, records: Sequence[dict]) -> EvaluationSet:
        return cls([PublicTask.from_record(r) for r in records], AnswerScorer(records),
                   fingerprint([(r["task_id"], task_hash(r["train"], r["test"])) for r in records]))


class TaskStore(Sequence):
    """Byte-indexed JSONL; full-data runs do not load every grid into RAM."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.offsets = []
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                if not handle.readline():
                    break
                self.offsets.append(offset)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            return json.loads(handle.readline())


@dataclass
class Prepared:
    synthetic: TaskStore
    arc: TaskStore
    validation: EvaluationSet
    audit: dict


def _load_official(path: Path, expected: int) -> list[dict]:
    records = [json.loads(line) for line in path.open()]
    if len(records) != expected:
        raise ValueError(f"{path.name}: expected {expected}, found {len(records)}")
    for record in records:
        require_task(record, record["task_id"])
        if task_max_grid_side(record) > 30:
            raise ValueError("Official ARC task exceeds 30×30")
        if task_hash(record["train"], record["test"]) != record["task_fingerprint"]:
            raise ValueError("Task fingerprint mismatch")
    if len({r["task_id"] for r in records}) != expected:
        raise ValueError("Duplicate task IDs")
    if len({r["task_fingerprint"] for r in records}) != expected:
        raise ValueError("Duplicate official task contents")
    return records


def prepare(config: Config, package: Path = PACKAGE) -> Prepared:
    """Audit all sources once, sample reproducibly, and isolate evaluation answers."""
    config.validate()
    package = Path(package)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    config_file = config.run_dir / "config.json"
    specification = {"schema_version": SCHEMA_VERSION, **asdict(config)}
    if config_file.exists() and json.loads(config_file.read_text()) != specification:
        raise ValueError("Settings changed: choose a new run_name to preserve comparability")
    write_json(config_file, specification)
    print("Verifying source checksums…", flush=True)
    expected_hashes = {}
    for line in (package / "manifests/checksums.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        expected_hashes[name.strip()] = digest
    source_hashes = {}
    for name in ("arc_train", "arc_validate", "synthetic_arc_train"):
        relative = f"data/{name}.jsonl"
        actual = sha256_file(package / relative)
        if actual != expected_hashes[relative]:
            raise ValueError(f"Source checksum mismatch: {relative}")
        source_hashes[name] = actual
    train_records = _load_official(package / "data/arc_train.jsonl", 1008)
    validation_records = _load_official(package / "data/arc_validate.jsonl", 138)
    validation = EvaluationSet.from_records(validation_records)
    train_ids = {r["task_id"] for r in train_records}
    validation_ids = {r["task_id"] for r in validation_records}
    if train_ids & validation_ids:
        raise ValueError("Official train/validation task ID overlap")
    if {r["task_fingerprint"] for r in train_records} & {r["task_fingerprint"] for r in validation_records}:
        raise ValueError("Official train/validation task content overlap")
    protected_pairs = validation.scorer.protected_pair_hashes()
    cache = config.run_dir / "data"
    cache.mkdir(exist_ok=True)
    audit_file = cache / "audit.json"
    if audit_file.exists():
        audit = json.loads(audit_file.read_text())
        if audit["source_hashes"] != source_hashes:
            raise ValueError("Source datasets changed: choose a new run_name")
        for name, digest in audit["prepared_hashes"].items():
            if sha256_file(cache / name) != digest:
                raise ValueError(f"Prepared data was modified: {name}")
        return Prepared(TaskStore(cache / "synthetic.jsonl"), TaskStore(cache / "arc.jsonl"), validation, audit)
    print("Auditing 199,899 synthetic tasks and selecting the training subset…", flush=True)
    counts, seen_ids, seen_fingerprints = Counter(), set(), set()
    pools: dict[str, list] = {"google_aux": [], "barc_aux": []}
    full_path = cache / "synthetic.jsonl.tmp"
    total = excluded = 0
    with full_path.open("w") as full:
        for line in (package / "data/synthetic_arc_train.jsonl").open():
            record = json.loads(line)
            total += 1
            require_task(record, f"synthetic row {total}")
            source = record["source"]
            if source not in pools:
                raise ValueError(f"Unexpected synthetic source: {source}")
            if record["task_id"] in seen_ids or record["task_fingerprint"] in seen_fingerprints:
                raise ValueError("Duplicate synthetic task")
            seen_ids.add(record["task_id"])
            seen_fingerprints.add(record["task_fingerprint"])
            pairs = [pair_hash(p) for split in ("train", "test") for p in record[split]]
            if protected_pairs.intersection(pairs):
                raise ValueError("Synthetic pair overlaps protected validation data")
            if record["task_fingerprint"] != task_hash(record["train"], record["test"]):
                raise ValueError("Synthetic fingerprint mismatch")
            linked = record.get("metadata", {}).get("linked_arc_task_id")
            if linked and linked not in train_ids:
                raise ValueError("Synthetic Google episode links outside official training")
            if task_max_grid_side(record) > 30:
                excluded += 1
                continue
            counts[source] += 1
            if config.synthetic_tasks is None:
                full.write(compact(record) + "\n")
            else:
                priority = int(fingerprint([config.seed, record["task_id"]]), 16)
                item = (-priority, total, record)
                heap = pools[source]
                if len(heap) < config.synthetic_tasks:
                    heapq.heappush(heap, item)
                elif priority < -heap[0][0]:
                    heapq.heapreplace(heap, item)
            if total % 25000 == 0:
                print(f"  audited {total:,} tasks", flush=True)
    if total != 199899:
        raise ValueError(f"Expected 199,899 synthetic tasks; found {total}")
    selected_counts = dict(counts)
    if config.synthetic_tasks is not None:
        budget = config.synthetic_tasks
        if budget > sum(counts.values()):
            raise ValueError("Requested more synthetic tasks than eligible")
        proportions = {s: budget * count / sum(counts.values()) for s, count in counts.items()}
        selected_counts = {s: math.floor(n) for s, n in proportions.items()}
        remainder = budget - sum(selected_counts.values())
        for source in sorted(counts, key=lambda s: (-(proportions[s] % 1), s))[:remainder]:
            selected_counts[source] += 1
        selected = []
        for source, count in selected_counts.items():
            selected.extend(item[2] for item in sorted(pools[source], key=lambda item: -item[0])[:count])
        with full_path.open("w") as handle:
            for record in sorted(selected, key=lambda r: r["task_id"]):
                handle.write(compact(record) + "\n")
    full_path.replace(cache / "synthetic.jsonl")
    with (cache / "arc.jsonl").open("w") as handle:
        for record in train_records:
            handle.write(compact(record) + "\n")
    with (cache / "selection.jsonl").open("w") as handle:
        for record in TaskStore(cache / "synthetic.jsonl"):
            handle.write(compact({k: record[k] for k in ("task_id", "source", "task_fingerprint")}) + "\n")
    audit = {"created_at": utc_now(), "source_hashes": source_hashes,
             "official_train": len(train_records), "official_validation": len(validation.tasks),
             "synthetic_total": total, "excluded_large_grids": excluded,
             "eligible_synthetic_sources": dict(counts), "selected_sources": selected_counts,
             "synthetic_protected_pair_overlap": 0,
             "official_pair_overlap_count": len(protected_pairs & {pair_hash(p) for r in train_records
                                                for s in ("train", "test") for p in r[s]}),
             "prepared_hashes": {name: sha256_file(cache / name) for name in
                                 ("synthetic.jsonl", "arc.jsonl", "selection.jsonl")}}
    write_json(audit_file, audit)
    print(f"Ready: {sum(selected_counts.values()):,} synthetic, 1,008 ARC train, 138 validation tasks.", flush=True)
    return Prepared(TaskStore(cache / "synthetic.jsonl"), TaskStore(cache / "arc.jsonl"), validation, audit)


def transform_grid(grid: list, rotation: int, reflect: bool, colors: list[int]) -> list:
    result = [[colors[cell] for cell in row] for row in grid]
    for _ in range(rotation % 4):
        result = [list(row) for row in zip(*result[::-1])]
    return [row[::-1] for row in result] if reflect else result


def adaptation_examples(task: PublicTask, steps: int, seed: int) -> list[Example]:
    """Leave one visible demonstration out; actual test inputs/answers are unused."""
    if len(task.demonstrations) < 2:
        raise ValueError("Per-task adaptation needs at least two visible demonstrations")
    rng = random.Random(seed)
    order = list(range(len(task.demonstrations)))
    rng.shuffle(order)
    result = []
    for step in range(steps):
        held_out = order[step % len(order)]
        rotation, reflect = rng.randrange(8) % 4, bool(rng.randrange(2))
        colors = list(range(10))
        rng.shuffle(colors)
        pairs = [{key: transform_grid(pair[key], rotation, reflect, colors)
                  for key in ("input", "output")} for pair in task.demonstrations]
        demonstrations = [pair for index, pair in enumerate(pairs) if index != held_out]
        rng.shuffle(demonstrations)
        result.append(Example(PublicTask(task.task_id, demonstrations, [pairs[held_out]["input"]]),
                              [pairs[held_out]["output"]]))
    return result


def runtime_identity(config: Config) -> dict:
    metadata_file = ROOT / "models/qwen3-8b-mlx-4bit/model_revision.json"
    if not metadata_file.exists():
        raise FileNotFoundError("Download the pinned model with scripts/setup_arc_finetune.py first")
    metadata = json.loads(metadata_file.read_text())
    if metadata["model_id"] != config.model_id:
        raise ValueError("Model ID differs from the downloaded checkpoint")
    for name, digest in metadata["files_sha256"].items():
        if sha256_file(metadata_file.parent / name) != digest:
            raise ValueError(f"Downloaded model file was modified: {name}")
    versions = {name: importlib.metadata.version(name) for name in
                ("mlx", "mlx-lm", "transformers", "numpy", "huggingface-hub")}
    return {"model": metadata, "packages": versions, "python": platform.python_version(),
            "implementation": sha256_file(Path(__file__)),
            "prompt_implementation": fingerprint(inspect.getsource(task_prompt))}


class MLXBackend:
    """One frozen 4-bit base, with explicit complete adapter restoration."""

    def __init__(self, config: Config):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load
        from mlx_lm.tuner.utils import linear_to_lora_layers
        from mlx_lm.tuner.trainer import grad_checkpoint
        self.mx, self.nn, self.config = mx, nn, config
        self.identity = runtime_identity(config)
        self.model, self.tokenizer = load(str(ROOT / "models/qwen3-8b-mlx-4bit"))
        self.model.freeze()
        mx.random.seed(config.seed)
        self.lora_config = {"rank": config.rank, "scale": config.scale, "dropout": config.dropout,
                            "keys": ["self_attn.q_proj", "self_attn.v_proj"]}
        linear_to_lora_layers(self.model, config.num_layers, self.lora_config)
        if not self.weights():
            raise RuntimeError("No adapter parameters were attached")
        layer_type = type(self.model.layers[0])
        if not getattr(layer_type, "_arc_gradient_checkpointed", False):
            grad_checkpoint(self.model.layers[0])
            layer_type._arc_gradient_checkpointed = True
        mx.eval(self.model.parameters())
        self.initial = {key: mx.array(value) for key, value in self.weights().items()}
        self.value_and_grad = nn.value_and_grad(self.model, self._loss)
        self.model.eval()
        print(f"Loaded Qwen3-8B; {sum(v.size for v in self.initial.values()):,} adapter parameters.", flush=True)

    def weights(self) -> dict:
        from mlx.utils import tree_flatten
        return dict(tree_flatten(self.model.trainable_parameters()))

    def restore(self, weights: dict) -> None:
        actual = self.weights()
        if set(weights) != set(actual) or any(weights[key].shape != actual[key].shape for key in actual):
            raise ValueError("Incomplete or incompatible adapter snapshot")
        self.model.load_weights(list(weights.items()), strict=False)
        self.mx.eval(self.model.trainable_parameters())
        self.model.eval()
        self.mx.clear_cache()

    def render(self, task: PublicTask) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": task.prompt()}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)

    def encode(self, example: Example) -> tuple[list[int], int]:
        prefix = self.tokenizer.encode(self.render(example.task), add_special_tokens=False)
        answer = self.tokenizer.encode(example.completion, add_special_tokens=False)
        tokens = prefix + answer + [self.tokenizer.eos_token_id]
        if len(tokens) > self.config.max_seq_length:
            raise ValueError(f"{example.task.task_id}: {len(tokens)} tokens exceed context; no truncation allowed")
        return tokens, len(prefix)

    def _loss(self, model, tokens, offset):
        # Causal alignment: logits at prefix_length-1 predict the FIRST answer token.
        # Qwen3's frozen vocabulary projection is needed only at answer positions.
        # Slice hidden states BEFORE that projection to avoid prompt×vocabulary logits.
        hidden = model.model(tokens[:, :-1])[:, offset - 1:, :]
        logits = (model.model.embed_tokens.as_linear(hidden) if model.args.tie_word_embeddings
                  else model.lm_head(hidden))
        targets = tokens[:, offset:]
        return self.nn.losses.cross_entropy(logits.astype(self.mx.float32), targets, reduction="mean")

    def optimizer(self, learning_rate: float):
        import mlx.optimizers as optim
        return optim.Adam(learning_rate=learning_rate)

    def microstep(self, example: Example):
        tokens, offset = self.encode(example)
        self.model.train()
        loss, gradients = self.value_and_grad(self.model, self.mx.array([tokens], dtype=self.mx.int32), offset)
        self.mx.eval(loss, gradients)
        if not math.isfinite(float(loss.item())):
            raise FloatingPointError("Non-finite training loss; adapter update cancelled")
        return loss, gradients, len(tokens), len(tokens) - offset

    def update(self, optimizer, gradients, count: int):
        from mlx.utils import tree_map
        optimizer.update(self.model, tree_map(lambda g: g / count, gradients))
        self.mx.eval(self.model.parameters(), optimizer.state)
        self.check_memory()
        self.mx.clear_cache()

    def check_memory(self) -> float:
        peak = self.mx.get_peak_memory() / 2**30
        if peak > self.config.memory_limit_gib:
            raise MemoryError(f"Peak MLX memory {peak:.2f} GiB exceeded {self.config.memory_limit_gib:.1f} GiB; "
                              "stopped before starting a long run. See the saved diagnostic.")
        return peak

    def adapt(self, task: PublicTask, seed: int) -> dict:
        self.mx.random.seed(seed)
        optimizer = self.optimizer(self.config.adaptation_lr)
        started, losses = time.perf_counter(), []
        for example in adaptation_examples(task, self.config.adaptation_steps, seed):
            loss, gradients, _, _ = self.microstep(example)
            self.update(optimizer, gradients, 1)
            losses.append(float(loss.item()))
        self.model.eval()
        return {"adaptation_seconds": time.perf_counter() - started,
                "adaptation_steps": len(losses), "adaptation_loss": losses[-1] if losses else None}

    def generate(self, task: PublicTask) -> dict:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        self.model.eval()
        prompt = self.tokenizer.encode(self.render(task), add_special_tokens=False)
        if len(prompt) + self.config.max_output_tokens > self.config.max_seq_length:
            raise ValueError(f"{task.task_id}: prompt plus fixed output budget exceeds native context")
        started = time.perf_counter()
        text, generated, finish_reason = [], 0, None
        for response in stream_generate(self.model, self.tokenizer, prompt=prompt,
                                        max_tokens=self.config.max_output_tokens,
                                        sampler=make_sampler(temp=0.0), prefill_step_size=512):
            text.append(response.text)
            generated = response.generation_tokens
            finish_reason = response.finish_reason
        self.mx.clear_cache()
        return {"raw_response": "".join(text), "generation_seconds": time.perf_counter() - started,
                "prompt_tokens": len(prompt), "generated_tokens": generated,
                "finish_reason": finish_reason, "output_budget_reached": finish_reason == "length"}

    def save_adapter(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "adapters.tmp.safetensors"
        self.mx.save_safetensors(str(temporary), self.weights())
        temporary.replace(directory / "adapters.safetensors")
        write_json(directory / "adapter_config.json", {"fine_tune_type": "lora",
                   "num_layers": self.config.num_layers, "lora_parameters": self.lora_config})

    def load_stage(self, stage: str) -> dict:
        if stage == "base":
            self.restore(self.initial)
            return self.initial
        directory = self.config.run_dir / "checkpoints" / stage
        metadata = json.loads((directory / "complete.json").read_text())
        if metadata["identity"] != self.identity or metadata["config"] != asdict(self.config):
            raise ValueError("Checkpoint runtime/configuration changed")
        if sha256_file(directory / "adapters.safetensors") != metadata["adapter_sha256"]:
            raise ValueError("Stage adapter checksum mismatch")
        weights = self.mx.load(str(directory / "adapters.safetensors"))
        self.restore(weights)
        return weights


def _runtime_file(config: Config, backend) -> None:
    path = config.run_dir / "runtime.json"
    if path.exists() and json.loads(path.read_text()) != backend.identity:
        raise ValueError("Runtime changed: use a new run_name rather than mix results")
    write_json(path, backend.identity)


def preflight(config: Config, data: Prepared, backend: MLXBackend, *, measure: bool = True) -> dict:
    """Tokenize every selected training example, and measure the longest one."""
    _runtime_file(config, backend)
    maximum, longest, lengths = 0, None, {}
    for name, records in (("synthetic", data.synthetic), ("arc", data.arc)):
        values = []
        print(f"Checking {len(records):,} complete {name} sequences…", flush=True)
        for record in records:
            example = Example.from_record(record)
            tokens, _ = backend.encode(example)
            values.append(len(tokens))
            if len(tokens) > maximum:
                maximum, longest = len(tokens), example
        ordered = sorted(values)
        lengths[name] = {"tasks": len(values), "max_tokens": max(values),
                         "mean_tokens": sum(values) / len(values), "p95_tokens": ordered[int(.95 * (len(values) - 1))]}
    for task in data.validation.tasks:
        # The preflight sees only public validation data, never validation targets.
        prompt_tokens = len(backend.tokenizer.encode(backend.render(task), add_special_tokens=False))
        if prompt_tokens + config.max_output_tokens > config.max_seq_length:
            raise ValueError(f"{task.task_id}: validation context budget exceeded")
        for example in adaptation_examples(task, config.adaptation_steps,
                                           int(fingerprint([config.seed, task.task_id])[:8], 16)):
            backend.encode(example)
    report = {"created_at": utc_now(), "identity": backend.identity, "lengths": lengths,
              "longest_task": longest.task.task_id, "longest_tokens": maximum,
              "measured": measure, "config": asdict(config)}
    if measure:
        print(f"Measuring one training step on the longest sequence ({maximum:,} tokens)…", flush=True)
        snapshot = {key: backend.mx.array(value) for key, value in backend.weights().items()}
        random_state = backend.mx.array(backend.mx.random.state[0])
        backend.mx.reset_peak_memory()
        started = time.perf_counter()
        try:
            optimizer = backend.optimizer(config.synthetic_lr)
            loss, gradients, tokens, _ = backend.microstep(longest)
            backend.update(optimizer, gradients, 1)
            report.update({"loss": float(loss.item()), "step_seconds": time.perf_counter() - started,
                           "peak_memory_gib": backend.check_memory()})
            throughput = tokens / report["step_seconds"]
            report["conservative_training_hours"] = {
                name: stats["mean_tokens"] * stats["tasks"] * epochs / throughput / 3600
                for (name, stats), epochs in zip(lengths.items(), (config.synthetic_epochs, config.arc_epochs), strict=True)}
        except Exception as error:
            write_json(config.run_dir / "preflight_failure.json", {**report, "error": str(error)})
            raise
        finally:
            backend.restore(snapshot)
            restore_rng(backend.mx, random_state)
            gc.collect()
    write_json(config.run_dir / "preflight.json", report)
    return report


def _require_preflight(config: Config, backend: MLXBackend):
    report = json.loads((config.run_dir / "preflight.json").read_text())
    if not report["measured"] or report["identity"] != backend.identity or report["config"] != asdict(config):
        raise ValueError("Run the measured preflight with this configuration/runtime first")


def train_stage(config: Config, data: Prepared, backend: MLXBackend, stage: str) -> dict:
    """Fixed epochs; checkpoint adapters + optimizer + RNG for exact step recovery."""
    from mlx.utils import tree_flatten, tree_map, tree_unflatten
    _require_preflight(config, backend)
    if stage not in {"synthetic", "arc"}:
        raise ValueError("stage must be synthetic or arc")
    previous_stage = "base" if stage == "synthetic" else "synthetic"
    for adapted in (False, True):
        condition = previous_stage + ("_adapted" if adapted else "_ordinary")
        evaluation_dir = config.run_dir / "evaluation" / condition
        summary_file = evaluation_dir / "summary.json"
        metadata_file = evaluation_dir / "metadata.json"
        if (not summary_file.exists() or not json.loads(summary_file.read_text())["complete"]
                or json.loads(metadata_file.read_text())["dataset"] != data.validation.identity):
            raise ValueError(f"Complete both {previous_stage} validation conditions before {stage} training")
    directory = config.run_dir / "checkpoints" / stage
    completion = directory / "complete.json"
    if completion.exists():
        backend.load_stage(stage)
        return json.loads(completion.read_text())
    backend.load_stage("base" if stage == "synthetic" else "synthetic")
    records = data.synthetic if stage == "synthetic" else data.arc
    epochs = config.synthetic_epochs if stage == "synthetic" else config.arc_epochs
    lr = config.synthetic_lr if stage == "synthetic" else config.arc_lr
    optimizer = backend.optimizer(lr)
    total_microsteps, cursor, updates = len(records) * epochs, 0, 0
    directory.mkdir(parents=True, exist_ok=True)
    recovery = directory / "recovery"
    state_file = recovery / "state.json"
    if state_file.exists():
        state = json.loads(state_file.read_text())
        if state["identity"] != backend.identity or state["config"] != asdict(config):
            raise ValueError("Recovery checkpoint runtime/configuration changed")
        backend.restore(backend.mx.load(str(recovery / "adapters.safetensors")))
        optimizer.state = tree_unflatten(list(backend.mx.load(str(recovery / "optimizer.safetensors")).items()))
        restore_rng(backend.mx, backend.mx.load(str(recovery / "rng.safetensors"))["state"])
        cursor, updates = state["cursor"], state["updates"]
    else:
        backend.mx.random.seed(config.seed)
    log_file = config.run_dir / f"training_{stage}.jsonl"
    if log_file.exists():
        # Updates after the last durable checkpoint will be replayed, so discard
        # their provisional metrics instead of counting them twice on recovery.
        rows = [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()]
        log_file.write_text("".join(compact(row) + "\n" for row in rows if row["optimizer_updates"] <= updates))
    started = time.perf_counter()
    print(f"{stage}: {total_microsteps:,} examples, {math.ceil(total_microsteps / config.accumulation):,} optimizer updates", flush=True)
    accumulated, count, recent_losses, token_count = None, 0, [], 0
    try:
        for epoch in range(epochs):
            order = list(range(len(records)))
            random.Random(config.seed + epoch).shuffle(order)
            for position, index in enumerate(order):
                global_step = epoch * len(records) + position
                if global_step < cursor:
                    continue
                loss, gradient, tokens, _ = backend.microstep(Example.from_record(records[index]))
                accumulated = gradient if accumulated is None else tree_map(lambda a, b: a + b, accumulated, gradient)
                count += 1
                recent_losses.append(float(loss.item()))
                token_count += tokens
                cursor = global_step + 1
                if count == config.accumulation or cursor == total_microsteps:
                    backend.update(optimizer, accumulated, count)
                    accumulated, count = None, 0
                    updates += 1
                    elapsed = time.perf_counter() - started
                    metric = {"stage": stage, "microstep": cursor, "optimizer_updates": updates,
                              "loss": sum(recent_losses) / len(recent_losses), "elapsed_seconds": elapsed,
                              "tokens_per_second": token_count / max(elapsed, .001),
                              "peak_memory_gib": backend.check_memory()}
                    append_json(log_file, metric)
                    recent_losses = []
                    if updates == 1 or updates % 10 == 0:
                        print(f"  {cursor:,}/{total_microsteps:,} examples; loss {metric['loss']:.4f}; "
                              f"{metric['tokens_per_second']:.0f} tokens/s; {metric['peak_memory_gib']:.2f} GiB", flush=True)
                    if updates % config.checkpoint_every == 0 or cursor == total_microsteps:
                        # A generation directory is committed by an atomic pointer replacement.
                        checkpoint = directory / f"step-{updates:06d}"
                        backend.save_adapter(checkpoint)
                        backend.mx.save_safetensors(str(checkpoint / "optimizer.safetensors"), dict(tree_flatten(optimizer.state)))
                        backend.mx.save_safetensors(str(checkpoint / "rng.safetensors"), {"state": backend.mx.random.state[0]})
                        write_json(checkpoint / "state.json", {"cursor": cursor, "updates": updates,
                                   "identity": backend.identity, "config": asdict(config)})
                        # Recovery files are read via this symlink, switched only after all files exist.
                        link = directory / "recovery.new"
                        link.unlink(missing_ok=True)
                        link.symlink_to(checkpoint.name, target_is_directory=True)
                        link.replace(recovery)
        backend.save_adapter(directory)
        report = {"stage": stage, "completed_at": utc_now(), "examples": total_microsteps,
                  "optimizer_updates": updates, "identity": backend.identity, "config": asdict(config),
                  "adapter_sha256": sha256_file(directory / "adapters.safetensors")}
        write_json(completion, report)
        return report
    except BaseException as error:
        write_json(directory / "interrupted.json", {"cursor": cursor, "error": str(error),
                   "note": "Resume from the last committed optimizer checkpoint; incomplete accumulation is replayed."})
        raise
    finally:
        backend.model.eval()


def _read_results(path: Path) -> list[dict]:
    """Discard an incomplete final append left by a killed process, never a full bad row."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b"\n"):
        path.write_bytes(b"".join(lines[:-1]))
        lines = lines[:-1]
    records = [json.loads(line) for line in lines]
    if len({r["task_id"] for r in records}) != len(records):
        raise ValueError("Duplicate completed validation tasks")
    return records


def summarize(results: list[dict], expected: int, stage: str, adapted: bool) -> dict:
    grids = sum(r["test_count"] for r in results)
    return {"stage": stage, "adapted": adapted, "completed": len(results), "expected": expected,
            "complete": len(results) == expected,
            "solved_tasks": sum(r["correct"] for r in results),
            "task_accuracy": sum(r["correct"] for r in results) / len(results) if results else None,
            "grid_accuracy": sum(r["tests_correct"] for r in results) / grids if grids else None,
            "malformed_rate": sum(r["malformed"] for r in results) / len(results) if results else None,
            "mean_latency_seconds": sum(r["latency_seconds"] for r in results) / len(results) if results else None,
            "peak_memory_gib": max((r["peak_memory_gib"] for r in results), default=0)}


def evaluate(config: Config, data: EvaluationSet, backend, stage: str, adapted: bool,
             *, stop_after: int | None = None) -> dict:
    """Append one scored record per task. Cancellation/errors remain resumable."""
    _runtime_file(config, backend)
    if isinstance(backend, MLXBackend):
        _require_preflight(config, backend)
    name = stage + ("_adapted" if adapted else "_ordinary")
    directory = config.run_dir / "evaluation" / name
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {"identity": backend.identity, "dataset": data.identity,
                "config": asdict(config), "stage": stage, "adapted": adapted}
    metadata_file = directory / "metadata.json"
    if metadata_file.exists() and json.loads(metadata_file.read_text()) != metadata:
        raise ValueError("Evaluation configuration, dataset or implementation changed")
    write_json(metadata_file, metadata)
    baseline = backend.load_stage(stage)
    # The identity includes the actual adapter bytes, preventing unnoticed replacement.
    stage_file = config.run_dir / "checkpoints" / stage / "adapters.safetensors"
    adapter_hash = sha256_file(stage_file) if stage != "base" and stage_file.exists() else "zero-adapter"
    hash_file = directory / "checkpoint.json"
    if hash_file.exists() and json.loads(hash_file.read_text())["sha256"] != adapter_hash:
        raise ValueError("Stage adapter changed since evaluation started")
    write_json(hash_file, {"sha256": adapter_hash})
    path = directory / "predictions.jsonl"
    results = _read_results(path)
    done = {row["task_id"] for row in results}
    if done - {t.task_id for t in data.tasks}:
        raise ValueError("Saved results contain tasks outside this cohort")
    new = 0
    try:
        for task in data.tasks:
            if task.task_id in done:
                continue
            started = time.perf_counter()
            backend.restore(baseline)
            backend.mx.reset_peak_memory()
            seed = int(fingerprint([config.seed, task.task_id])[:8], 16)
            backend.mx.random.seed(seed)
            adaptation = backend.adapt(task, seed) if adapted else {"adaptation_seconds": 0., "adaptation_steps": 0}
            generated = backend.generate(task)
            row = {"task_id": task.task_id, "stage": stage, "adapted": adapted, **generated,
                   **adaptation, **data.scorer.score(task.task_id, generated["raw_response"]),
                   "latency_seconds": time.perf_counter() - started,
                   "peak_memory_gib": backend.check_memory()}
            append_json(path, row)
            results.append(row)
            new += 1
            summary = summarize(results, len(data.tasks), stage, adapted)
            write_json(directory / "summary.json", summary)
            print(f"{name}: {len(results)}/{len(data.tasks)}; solved {summary['solved_tasks']}; "
                  f"last task {row['latency_seconds']:.1f}s", flush=True)
            if stop_after is not None and new >= stop_after:
                break
    except BaseException as error:
        write_json(directory / "interrupted.json", {"completed": len(results), "error": str(error)})
        raise
    finally:
        backend.restore(baseline)
    summary = summarize(results, len(data.tasks), stage, adapted)
    write_json(directory / "summary.json", summary)
    return summary


def comparison(config: Config):
    """Return a DataFrame and export paired gains/regressions for all conditions."""
    import pandas as pd
    directory = config.run_dir / "evaluation"
    summaries, outcomes = [], {}
    for stage in ("base", "synthetic", "arc"):
        for adapted in (False, True):
            name = stage + ("_adapted" if adapted else "_ordinary")
            path = directory / name
            if (path / "summary.json").exists():
                summaries.append(json.loads((path / "summary.json").read_text()))
                outcomes[name] = {r["task_id"]: r["correct"] for r in _read_results(path / "predictions.jsonl")}
    frame = pd.DataFrame(summaries)
    frame.to_csv(config.run_dir / "comparison.csv", index=False)
    pairs = []
    for left, left_results in outcomes.items():
        for right, right_results in outcomes.items():
            if left >= right:
                continue
            shared = sorted(set(left_results) & set(right_results))
            gains = [task for task in shared if not left_results[task] and right_results[task]]
            regressions = [task for task in shared if left_results[task] and not right_results[task]]
            pairs.append({"from": left, "to": right, "paired_tasks": len(shared),
                          "gains": gains, "regressions": regressions, "net_solved": len(gains) - len(regressions)})
    write_json(config.run_dir / "paired_comparisons.json", pairs)
    return frame


def smoke(config: Config | None = None) -> dict:
    """Small real-model run, using ONLY official training tasks for every role."""
    config = config or replace(Config(), run_name="qwen3-8b-smoke-v1", synthetic_tasks=4,
                               synthetic_epochs=1, arc_epochs=1, accumulation=1,
                               adaptation_steps=2, max_output_tokens=128, checkpoint_every=1)
    config.validate()
    records = _load_official(PACKAGE / "data/arc_train.jsonl", 1008)
    # Short tasks keep setup verification quick; all have >=2 demonstrations.
    records = sorted(records, key=lambda r: len(compact(r)))[:6]
    config.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.run_dir / "config.json", {"schema_version": SCHEMA_VERSION, **asdict(config)})
    cache = config.run_dir / "data"
    cache.mkdir(exist_ok=True)
    for name, subset in (("synthetic", records[:2]), ("arc", records[2:4])):
        (cache / f"{name}.jsonl").write_text("".join(compact(r) + "\n" for r in subset))
    cohort = EvaluationSet.from_records(records[4:6])
    data = Prepared(TaskStore(cache / "synthetic.jsonl"), TaskStore(cache / "arc.jsonl"), cohort,
                    {"training_only_smoke": True})
    backend = MLXBackend(config)
    report = {"config": asdict(config), "training_only": True,
              "task_ids": [r["task_id"] for r in records], "identity": backend.identity}
    report["preflight"] = preflight(config, data, backend)
    report["conditions"] = []
    for stage in ("base", "synthetic", "arc"):
        if stage != "base":
            report[f"training_{stage}"] = train_stage(config, data, backend, stage)
            before = backend.generate(cohort.tasks[0])["raw_response"]
            backend.restore(backend.initial)
            backend.load_stage(stage)
            after = backend.generate(cohort.tasks[0])["raw_response"]
            if before != after:
                raise AssertionError("Adapter reload changed greedy generation")
        for adapted in (False, True):
            evaluate(config, cohort, backend, stage, adapted, stop_after=1)
            resumed = evaluate(config, cohort, backend, stage, adapted)
            if resumed["completed"] != 2:
                raise AssertionError("Interrupted validation did not resume exactly once")
            report["conditions"].append(resumed)
    report["finished_at"] = utc_now()
    report["passed"] = True
    write_json(config.run_dir / "smoke_report.json", report)
    comparison(config)
    print("Training-only smoke test passed; full validation has not been run.", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "preflight", "smoke", "train", "evaluate", "compare"])
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--stage", choices=["base", "synthetic", "arc"], default="base")
    parser.add_argument("--adapted", action="store_true")
    args = parser.parse_args()
    config = Config.full() if args.full else Config()
    if args.command == "smoke":
        smoke()
    elif args.command == "compare":
        print(comparison(config).to_string(index=False))
    else:
        data = prepare(config)
        if args.command == "prepare":
            print(json.dumps(data.audit, indent=2))
            return
        backend = MLXBackend(config)
        if args.command == "preflight":
            print(json.dumps(preflight(config, data, backend), indent=2))
        elif args.command == "train":
            train_stage(config, data, backend, args.stage)
        else:
            evaluate(config, data.validation, backend, args.stage, args.adapted)


if __name__ == "__main__":
    main()
