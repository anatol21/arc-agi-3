#!/usr/bin/env python3
"""Local ARC practice and Ollama benchmarking server for ARC-style JSONL datasets."""

from __future__ import annotations

import argparse
import csv
import io
import json
import mimetypes
import random
import sys
import threading
import time
import webbrowser
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen


STATIC_DIR = Path(__file__).with_name("static")
MAX_INTERACTIVE_GRID_SIDE = 30
MAX_REQUEST_BYTES = 100_000
DEFAULT_CORE_INSTRUCTION = (
    "Solve this ARC task. Infer the transformation from the training examples "
    "and apply it to every test input."
)
PACKAGE_DATASETS = (
    ("arc_train", "ARC Train", "arc_train.jsonl", False),
    ("arc_validate", "ARC Validate", "arc_validate.jsonl", True),
    ("synthetic_arc_train", "Synthetic ARC Train", "synthetic_arc_train.jsonl", False),
)


class DatasetSchemaError(ValueError):
    """The selected JSONL is not a supported ARC task dataset."""


class BenchmarkError(ValueError):
    """A benchmark configuration or Ollama request is invalid."""


@dataclass(frozen=True)
class TaskEntry:
    task_id: str
    source: str
    offset: int
    line_number: int
    max_grid_side: int


@dataclass(frozen=True)
class DatasetDefinition:
    dataset_id: str
    label: str
    path: Path
    evaluation_only: bool


def require_grid(grid: Any, location: str, *, validate_cells: bool = True) -> list[list[int]]:
    if not isinstance(grid, list) or not grid:
        raise DatasetSchemaError(f"{location}: grid must be a non-empty list of rows")
    if not all(isinstance(row, list) and row for row in grid):
        raise DatasetSchemaError(f"{location}: every row must be non-empty")
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise DatasetSchemaError(f"{location}: grid must be rectangular")
    if validate_cells:
        for row_index, row in enumerate(grid):
            for column_index, value in enumerate(row):
                if type(value) is not int or not 0 <= value <= 9:
                    raise DatasetSchemaError(
                        f"{location}[{row_index}][{column_index}]: expected an integer from 0 to 9"
                    )
    return grid


def require_pair(pair: Any, location: str, *, validate_cells: bool = True) -> dict[str, list[list[int]]]:
    if not isinstance(pair, dict) or not {"input", "output"}.issubset(pair):
        raise DatasetSchemaError(f"{location}: pair must have input and output grids")
    return {
        "input": require_grid(pair["input"], location + ".input", validate_cells=validate_cells),
        "output": require_grid(pair["output"], location + ".output", validate_cells=validate_cells),
    }


def require_task(task: Any, location: str, *, validate_cells: bool = True) -> dict[str, Any]:
    if not isinstance(task, dict):
        raise DatasetSchemaError(f"{location}: record must be a JSON object")
    missing = {"task_id", "source", "train", "test"} - set(task)
    if missing:
        if {"input", "output"}.issubset(task):
            raise DatasetSchemaError(
                f"{location}: this is a pair-level dataset (such as Google_cleaned), not an ARC task dataset"
            )
        raise DatasetSchemaError(f"{location}: missing ARC task fields: {', '.join(sorted(missing))}")
    if not isinstance(task["task_id"], str) or not task["task_id"].strip():
        raise DatasetSchemaError(f"{location}: task_id must be a non-empty string")
    if not isinstance(task["source"], str) or not task["source"].strip():
        raise DatasetSchemaError(f"{location}: source must be a non-empty string")
    if not isinstance(task["train"], list) or not task["train"]:
        raise DatasetSchemaError(f"{location}: train must be a non-empty list")
    if not isinstance(task["test"], list) or not task["test"]:
        raise DatasetSchemaError(f"{location}: test must be a non-empty list")
    for split_name in ("train", "test"):
        for index, pair in enumerate(task[split_name]):
            require_pair(pair, f"{location}.{split_name}[{index}]", validate_cells=validate_cells)
    return task


def task_max_grid_side(task: dict[str, Any]) -> int:
    return max(
        max(len(pair[grid_name]), len(pair[grid_name][0]))
        for split_name in ("train", "test")
        for pair in task[split_name]
        for grid_name in ("input", "output")
    )


class TaskIndex:
    """A byte-offset index; task bodies stay on disk until selected."""

    def __init__(self, definition: DatasetDefinition | Path) -> None:
        if isinstance(definition, Path):
            definition = DatasetDefinition("dataset", definition.stem, definition, False)
        self.definition = definition
        self.dataset_path = definition.path.resolve()
        if not self.dataset_path.is_file():
            raise FileNotFoundError(f"Dataset does not exist: {definition.path}")
        self.entries: list[TaskEntry] = []
        self.by_task_id: dict[str, int] = {}
        self.by_source: dict[str, list[int]] = defaultdict(list)
        self.playable_positions: list[int] = []
        self.playable_by_source: dict[str, list[int]] = defaultdict(list)
        self._build()

    def _build(self) -> None:
        with self.dataset_path.open("rb") as handle:
            line_number = 0
            while True:
                offset = handle.tell()
                raw_line = handle.readline()
                if not raw_line:
                    break
                line_number += 1
                if not raw_line.strip():
                    raise DatasetSchemaError(f"{self.definition.label}, line {line_number}: blank JSONL lines are not allowed")
                try:
                    task = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise DatasetSchemaError(
                        f"{self.definition.label}, line {line_number}: invalid UTF-8 JSON"
                    ) from error
                require_task(task, f"line {line_number}", validate_cells=False)
                task_id, source = task["task_id"], task["source"]
                if task_id in self.by_task_id:
                    raise DatasetSchemaError(f"line {line_number}: duplicate task_id {task_id!r}")
                position = len(self.entries)
                max_grid_side = task_max_grid_side(task)
                self.entries.append(TaskEntry(task_id, source, offset, line_number, max_grid_side))
                self.by_task_id[task_id] = position
                self.by_source[source].append(position)
                if max_grid_side <= MAX_INTERACTIVE_GRID_SIDE:
                    self.playable_positions.append(position)
                    self.playable_by_source[source].append(position)
                if line_number % 10_000 == 0:
                    print(f"{self.definition.label}: indexed {line_number:,} task records…", flush=True)
        if not self.entries:
            raise DatasetSchemaError(f"{self.definition.label}: the selected JSONL contains no tasks")
        if not self.playable_positions:
            raise DatasetSchemaError(
                f"{self.definition.label}: no task is within the {MAX_INTERACTIVE_GRID_SIDE}x{MAX_INTERACTIVE_GRID_SIDE} limit"
            )

    @property
    def source_counts(self) -> dict[str, int]:
        return dict(sorted((source, len(positions)) for source, positions in self.by_source.items()))

    def status(self) -> dict[str, Any]:
        return {
            "id": self.definition.dataset_id,
            "label": self.definition.label,
            "dataset_name": self.dataset_path.name,
            "evaluation_only": self.definition.evaluation_only,
            "task_count": len(self.entries),
            "playable_task_count": len(self.playable_positions),
            "oversize_task_count": len(self.entries) - len(self.playable_positions),
            "grid_limit": MAX_INTERACTIVE_GRID_SIDE,
            "sources": dict(
                sorted((source, len(positions)) for source, positions in self.playable_by_source.items())
            ),
        }

    def positions(self, source: str | None = None) -> list[int]:
        if not source:
            return self.playable_positions
        if source not in self.playable_by_source:
            raise KeyError(f"Unknown source filter: {source}")
        return self.playable_by_source[source]

    def choose(self, mode: str, position: int | None, source: str | None) -> int:
        positions = self.positions(source)
        if mode == "random":
            return random.choice(positions)
        if position is None:
            return positions[0]
        try:
            selected_index = positions.index(position)
        except ValueError:
            return positions[0]
        if mode == "next":
            return positions[(selected_index + 1) % len(positions)]
        if mode == "previous":
            return positions[(selected_index - 1) % len(positions)]
        raise ValueError(f"Unknown task mode: {mode}")

    def lookup_task_id(self, task_id: str) -> int:
        try:
            position = self.by_task_id[task_id]
        except KeyError as error:
            raise KeyError(f"Unknown task ID: {task_id}") from error
        if self.entries[position].max_grid_side > MAX_INTERACTIVE_GRID_SIDE:
            raise KeyError(
                f"Task ID {task_id!r} exceeds the {MAX_INTERACTIVE_GRID_SIDE}x{MAX_INTERACTIVE_GRID_SIDE} grid limit"
            )
        return position

    def load(self, position: int) -> dict[str, Any]:
        if not 0 <= position < len(self.entries):
            raise IndexError("Task position is outside the dataset")
        entry = self.entries[position]
        with self.dataset_path.open("rb") as handle:
            handle.seek(entry.offset)
            try:
                task = json.loads(handle.readline().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise DatasetSchemaError(f"line {entry.line_number}: invalid UTF-8 JSON") from error
        require_task(task, f"line {entry.line_number}")
        if task_max_grid_side(task) > MAX_INTERACTIVE_GRID_SIDE:
            raise DatasetSchemaError(f"line {entry.line_number}: task exceeds the interactive grid limit")
        return task

    def public_task(self, position: int) -> dict[str, Any]:
        task = self.load(position)
        entry = self.entries[position]
        return {
            "dataset_id": self.definition.dataset_id,
            "task_id": task["task_id"],
            "source": task["source"],
            "position": position,
            "train": [{"input": pair["input"], "output": pair["output"]} for pair in task["train"]],
            "test": [{"input": pair["input"]} for pair in task["test"]],
            "test_count": len(task["test"]),
            "line_number": entry.line_number,
        }

    def evaluate(self, task_id: str, test_index: int, submitted_grid: Any) -> bool:
        position = self.lookup_task_id(task_id)
        task = self.load(position)
        if type(test_index) is not int or not 0 <= test_index < len(task["test"]):
            raise ValueError("Test index is outside this task")
        submitted = require_grid(submitted_grid, "submitted grid")
        return submitted == task["test"][test_index]["output"]


class DatasetRegistry:
    def __init__(self, definitions: list[DatasetDefinition]) -> None:
        if not definitions:
            raise ValueError("At least one dataset is required")
        self.indexes: dict[str, TaskIndex] = {}
        for definition in definitions:
            if definition.dataset_id in self.indexes:
                raise ValueError(f"Duplicate dataset ID: {definition.dataset_id}")
            print(f"Indexing {definition.label}: {definition.path}", flush=True)
            self.indexes[definition.dataset_id] = TaskIndex(definition)
        self.default_dataset_id = "arc_validate" if "arc_validate" in self.indexes else next(iter(self.indexes))

    def get(self, dataset_id: str | None) -> TaskIndex:
        key = dataset_id or self.default_dataset_id
        try:
            return self.indexes[key]
        except KeyError as error:
            raise KeyError(f"Unknown dataset: {key}") from error

    def status(self) -> dict[str, Any]:
        return {
            "default_dataset_id": self.default_dataset_id,
            "grid_limit": MAX_INTERACTIVE_GRID_SIDE,
            "datasets": [self.indexes[dataset_id].status() for dataset_id in self.indexes],
        }


class OllamaClient:
    def __init__(self, base_url: str) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("--ollama-url must point to local http://127.0.0.1, localhost, or ::1")
        self.base_url = base_url.rstrip("/")

    def models(self) -> list[str]:
        payload = self._request("/api/tags", None, timeout_seconds=10)
        models = payload.get("models")
        if not isinstance(models, list):
            raise BenchmarkError("Ollama returned an invalid model list")
        return sorted(
            model["name"] for model in models if isinstance(model, dict) and isinstance(model.get("name"), str)
        )

    def generate(self, request_payload: dict[str, Any], timeout_seconds: int) -> dict[str, str | None]:
        payload = self._request("/api/generate", request_payload, timeout_seconds=timeout_seconds)
        response = payload.get("response")
        if not isinstance(response, str):
            raise BenchmarkError("Ollama did not return a text response")
        thinking = payload.get("thinking")
        if thinking is not None and not isinstance(thinking, str):
            thinking = json.dumps(thinking, ensure_ascii=False)
        return {"response": response, "thinking": thinking}

    def _request(self, path: str, payload: dict[str, Any] | None, *, timeout_seconds: int) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            raise BenchmarkError(f"Ollama request failed: {error}") from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BenchmarkError("Ollama returned invalid JSON") from error
        if not isinstance(body, dict):
            raise BenchmarkError("Ollama returned an invalid response object")
        return body


def task_prompt(
    task: dict[str, Any],
    modifier: str,
    core_instruction: str = DEFAULT_CORE_INSTRUCTION,
) -> str:
    public_task = {
        "train": [{"input": pair["input"], "output": pair["output"]} for pair in task["train"]],
        "test": [{"input": pair["input"]} for pair in task["test"]],
    }
    prompt = core_instruction.strip()
    prompt += """\n\nReturn exactly one JSON object with this schema and no prose, markdown, explanation, or code fence:\n{\"outputs\": [[[0]]]}\n\nThe `outputs` array must contain one rectangular 0–9 integer grid for each item in `test`, in the same order.\n\nTask:\n"""
    prompt += json.dumps(public_task, separators=(",", ":"))
    if modifier.strip():
        prompt += "\n\nAdditional instructions:\n" + modifier.strip()
    return prompt


def parse_model_outputs(raw_response: str, expected_count: int) -> list[list[list[int]]]:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as error:
        raise BenchmarkError("Model response is not valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"outputs"}:
        raise BenchmarkError('Model response must be exactly {"outputs": [...]}')
    outputs = payload["outputs"]
    if not isinstance(outputs, list) or len(outputs) != expected_count:
        raise BenchmarkError(f"Model response must provide exactly {expected_count} output grid(s)")
    return [require_grid(grid, f"model outputs[{index}]") for index, grid in enumerate(outputs)]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class BenchmarkRun:
    """One sequential in-memory run; expected outputs never leave this object."""

    def __init__(self, config: dict[str, Any], index: TaskIndex, ollama: OllamaClient) -> None:
        self.config = config
        self.index = index
        self.ollama = ollama
        self.started_at = utc_now()
        self.finished_at: str | None = None
        self.state = "running"
        self.cancel_requested = False
        self.results: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="arc-ollama-benchmark", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def cancel(self) -> None:
        with self.lock:
            if self.state == "running":
                self.cancel_requested = True

    def _cancelled(self) -> bool:
        with self.lock:
            return self.cancel_requested

    def _append(self, result: dict[str, Any]) -> None:
        with self.lock:
            self.results.append(result)

    def _finish(self, state: str) -> None:
        with self.lock:
            self.state = state
            self.finished_at = utc_now()

    def _run(self) -> None:
        for position in self.config["positions"]:
            if self._cancelled():
                self._finish("cancelled")
                return
            entry = self.index.entries[position]
            started = time.monotonic()
            result: dict[str, Any] = {
                "task_id": entry.task_id,
                "source": entry.source,
                "line_number": entry.line_number,
                "status": "error",
                "correct": False,
                "test_count": None,
                "tests_correct": 0,
                "latency_seconds": 0.0,
                "error": None,
                "raw_response": None,
                "reasoning": None,
                "prediction": None,
                "model_request": None,
            }
            try:
                task = self.index.load(position)
                request_payload = {
                    "model": self.config["model"],
                    "prompt": task_prompt(
                        task,
                        self.config["prompt_modifier"],
                        self.config["core_instruction"],
                    ),
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": self.config["temperature"],
                        "top_p": self.config["top_p"],
                        "num_predict": self.config["num_predict"],
                    },
                }
                # It includes demonstrations and test inputs, but never protected test outputs.
                result["model_request"] = request_payload
                model_response = self.ollama.generate(request_payload, self.config["timeout_seconds"])
                if isinstance(model_response, str):  # Supports simple local test doubles.
                    raw_response, reasoning = model_response, None
                elif isinstance(model_response, dict) and isinstance(model_response.get("response"), str):
                    raw_response = model_response["response"]
                    reasoning = model_response.get("thinking")
                    if reasoning is not None and not isinstance(reasoning, str):
                        raise BenchmarkError("Ollama returned invalid reasoning text")
                else:
                    raise BenchmarkError("Ollama did not return a text response")
                result["raw_response"] = raw_response
                result["reasoning"] = reasoning
                outputs = parse_model_outputs(raw_response, len(task["test"]))
                checks = [output == pair["output"] for output, pair in zip(outputs, task["test"], strict=True)]
                result.update(
                    {
                        "status": "correct" if all(checks) else "incorrect",
                        "correct": all(checks),
                        "test_count": len(checks),
                        "tests_correct": sum(checks),
                        "prediction": {"outputs": outputs},
                    }
                )
            except (BenchmarkError, DatasetSchemaError, ValueError) as error:
                result["status"] = "malformed_response" if result["raw_response"] is not None else "ollama_error"
                result["error"] = str(error)
            except Exception as error:
                result["status"] = "internal_error"
                result["error"] = f"Unexpected benchmark error: {error}"
            result["latency_seconds"] = round(time.monotonic() - started, 3)
            self._append(result)
        self._finish("completed")

    def public_config(self) -> dict[str, Any]:
        return {
            key: self.config[key]
            for key in (
                "dataset_id", "dataset_label", "source", "model", "order", "seed", "task_count",
                "temperature", "top_p", "num_predict", "timeout_seconds", "core_instruction", "prompt_modifier",
            )
        }

    def snapshot(self, *, include_results: bool = False) -> dict[str, Any]:
        with self.lock:
            counts = Counter(result["status"] for result in self.results)
            tests_total = sum(result["test_count"] or 0 for result in self.results)
            tests_correct = sum(result["tests_correct"] for result in self.results)
            snapshot: dict[str, Any] = {
                "state": self.state,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "cancel_requested": self.cancel_requested,
                "configured_task_count": len(self.config["positions"]),
                "completed_task_count": len(self.results),
                "remaining_task_count": len(self.config["positions"]) - len(self.results),
                "task_correct_count": counts["correct"],
                "test_correct_count": tests_correct,
                "test_total_count": tests_total,
                "status_counts": dict(sorted(counts.items())),
                "config": self.public_config(),
            }
            if include_results:
                snapshot["results"] = [
                    {
                        "result_index": result_index,
                        **{
                            key: value
                            for key, value in result.items()
                            if key not in {"raw_response", "reasoning", "prediction", "model_request"}
                        },
                    }
                    for result_index, result in enumerate(self.results)
                ]
            return snapshot

    def inspect_result(self, result_index: int) -> dict[str, Any]:
        """Return one model response without loading or exposing target grids."""
        with self.lock:
            if result_index < 0 or result_index >= len(self.results):
                raise BenchmarkError("That benchmark result is not available")
            result = self.results[result_index]
            return {
                "result_index": result_index,
                "task_id": result["task_id"],
                "source": result["source"],
                "status": result["status"],
                "test_count": result["test_count"],
                "tests_correct": result["tests_correct"],
                "latency_seconds": result["latency_seconds"],
                "error": result["error"],
                "raw_response": result["raw_response"],
                "reasoning": result["reasoning"],
                "prediction": result["prediction"],
                "model_request": result["model_request"],
            }

    def detailed_export(self) -> dict[str, Any]:
        report = self.snapshot()
        report["format"] = "arc-practice-benchmark-v1"
        report["records"] = self.results.copy()
        return report

    def csv_export(self) -> str:
        output = io.StringIO()
        fields = (
            "task_id", "source", "line_number", "status", "correct", "test_count", "tests_correct",
            "latency_seconds", "error",
        )
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for result in self.results:
            writer.writerow({key: result[key] for key in fields})
        return output.getvalue()


class BenchmarkManager:
    def __init__(self, registry: DatasetRegistry, ollama: OllamaClient) -> None:
        self.registry = registry
        self.ollama = ollama
        self.run: BenchmarkRun | None = None
        self.lock = threading.Lock()

    def models(self) -> list[str]:
        return self.ollama.models()

    def start(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise BenchmarkError("Benchmark configuration must be a JSON object")
        with self.lock:
            if self.run and self.run.state == "running":
                raise BenchmarkError("A benchmark is already running. Cancel it before starting another.")
            config, index = self._config(payload)
            self.run = BenchmarkRun(config, index, self.ollama)
            self.run.start()
            return self.run.snapshot()

    def _config(self, payload: dict[str, Any]) -> tuple[dict[str, Any], TaskIndex]:
        dataset_id = payload.get("dataset_id")
        if not isinstance(dataset_id, str):
            raise BenchmarkError("Choose a dataset")
        index = self.registry.get(dataset_id)
        model = payload.get("model")
        if not isinstance(model, str) or not model.strip():
            raise BenchmarkError("Choose an Ollama model")
        source = payload.get("source") or None
        if source is not None and not isinstance(source, str):
            raise BenchmarkError("Source must be text")
        positions = index.positions(source)
        task_count = self._integer(payload.get("task_count"), "Task count", minimum=1, maximum=len(positions))
        order = payload.get("order")
        if order not in {"random", "sequential"}:
            raise BenchmarkError("Order must be random or sequential")
        raw_seed = payload.get("seed")
        seed = None if raw_seed in (None, "") else self._integer(raw_seed, "Seed", minimum=0, maximum=2**63 - 1)
        chosen = random.Random(seed).sample(positions, task_count) if order == "random" else positions[:task_count]
        modifier = payload.get("prompt_modifier", "")
        if not isinstance(modifier, str) or len(modifier) > 8_000:
            raise BenchmarkError("Prompt modifier must be text of at most 8,000 characters")
        core_instruction = payload.get("core_instruction", DEFAULT_CORE_INSTRUCTION)
        if not isinstance(core_instruction, str) or not core_instruction.strip() or len(core_instruction) > 4_000:
            raise BenchmarkError("Core instruction must be non-empty text of at most 4,000 characters")
        config = {
            "dataset_id": dataset_id,
            "dataset_label": index.definition.label,
            "source": source,
            "model": model.strip(),
            "order": order,
            "seed": seed,
            "task_count": task_count,
            "temperature": self._number(payload.get("temperature"), "Temperature", minimum=0, maximum=2),
            "top_p": self._number(payload.get("top_p"), "Top-p", minimum=0, maximum=1),
            "num_predict": self._integer(payload.get("num_predict"), "Maximum tokens", minimum=1, maximum=32_768),
            "timeout_seconds": self._integer(payload.get("timeout_seconds"), "Timeout", minimum=1, maximum=600),
            "core_instruction": core_instruction.strip(),
            "prompt_modifier": modifier,
            "positions": chosen,
        }
        return config, index

    @staticmethod
    def _integer(value: Any, name: str, *, minimum: int, maximum: int) -> int:
        if type(value) is bool:
            raise BenchmarkError(f"{name} must be an integer")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as error:
            raise BenchmarkError(f"{name} must be an integer") from error
        if not minimum <= parsed <= maximum:
            raise BenchmarkError(f"{name} must be between {minimum} and {maximum}")
        return parsed

    @staticmethod
    def _number(value: Any, name: str, *, minimum: float, maximum: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as error:
            raise BenchmarkError(f"{name} must be a number") from error
        if not minimum <= parsed <= maximum:
            raise BenchmarkError(f"{name} must be between {minimum} and {maximum}")
        return parsed

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {"state": "idle"} if not self.run else self.run.snapshot(include_results=True)

    def inspect(self, result_index: str | None) -> dict[str, Any]:
        try:
            parsed_index = int(result_index) if result_index is not None else -1
        except ValueError as error:
            raise BenchmarkError("Result index must be an integer") from error
        with self.lock:
            if not self.run:
                raise BenchmarkError("No benchmark has been started")
            return self.run.inspect_result(parsed_index)

    def cancel(self) -> dict[str, Any]:
        with self.lock:
            if not self.run:
                raise BenchmarkError("No benchmark has been started")
            self.run.cancel()
            return self.run.snapshot(include_results=True)

    def export(self, export_format: str) -> tuple[bytes, str, str]:
        with self.lock:
            if not self.run:
                raise BenchmarkError("No benchmark is available for export")
            timestamp = self.run.started_at.replace(":", "-")
            if export_format == "json":
                return json.dumps(self.run.detailed_export(), indent=2).encode("utf-8"), "application/json; charset=utf-8", f"arc-benchmark-{timestamp}.json"
            if export_format == "csv":
                return self.run.csv_export().encode("utf-8"), "text/csv; charset=utf-8", f"arc-benchmark-{timestamp}.csv"
            raise BenchmarkError("Export format must be json or csv")


class PracticeServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], registry: DatasetRegistry, ollama_url: str) -> None:
        super().__init__(address, PracticeHandler)
        self.registry = registry
        self.benchmarks = BenchmarkManager(registry, OllamaClient(ollama_url))


class PracticeHandler(BaseHTTPRequestHandler):
    server: PracticeServer

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/api/datasets":
            self._json(HTTPStatus.OK, self.server.registry.status())
        elif parsed.path == "/api/dataset":
            self._get_dataset(query)
        elif parsed.path == "/api/task":
            self._get_task(query)
        elif parsed.path == "/api/benchmark/models":
            self._get_models()
        elif parsed.path == "/api/benchmark/status":
            self._json(HTTPStatus.OK, self.server.benchmarks.status())
        elif parsed.path == "/api/benchmark/result":
            self._get_benchmark_result(query)
        elif parsed.path == "/api/benchmark/export":
            self._get_export(query)
        else:
            self._serve_static(parsed.path)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/attempt":
            self._post_attempt()
        elif path == "/api/benchmark/start":
            self._post_benchmark_start()
        elif path == "/api/benchmark/cancel":
            self._post_benchmark_cancel()
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            raise ValueError("Request body must be a small JSON object")
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Request must be a JSON object")
        return payload

    def _get_dataset(self, query: dict[str, list[str]]) -> None:
        try:
            self._json(HTTPStatus.OK, self.server.registry.get(query.get("dataset", [None])[0]).status())
        except KeyError as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def _get_task(self, query: dict[str, list[str]]) -> None:
        try:
            index = self.server.registry.get(query.get("dataset", [None])[0])
            source = query.get("source", [""])[0] or None
            task_id = query.get("task_id", [""])[0].strip()
            if task_id:
                position = index.lookup_task_id(task_id)
            else:
                raw_position = query.get("position", [None])[0]
                position = index.choose(query.get("mode", ["random"])[0], int(raw_position) if raw_position is not None else None, source)
            self._json(HTTPStatus.OK, index.public_task(position))
        except (KeyError, ValueError, IndexError, DatasetSchemaError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def _post_attempt(self) -> None:
        try:
            payload = self._read_json()
            correct = self.server.registry.get(payload.get("dataset_id")).evaluate(
                payload.get("task_id"), payload.get("test_index"), payload.get("grid")
            )
        except (ValueError, DatasetSchemaError, UnicodeDecodeError, json.JSONDecodeError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        except KeyError as error:
            self._json(HTTPStatus.NOT_FOUND, {"error": str(error)})
            return
        self._json(HTTPStatus.OK, {"correct": correct, "message": "Correct solution!" if correct else "Not correct. Try again."})

    def _get_models(self) -> None:
        try:
            self._json(HTTPStatus.OK, {"models": self.server.benchmarks.models()})
        except BenchmarkError as error:
            self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(error)})

    def _post_benchmark_start(self) -> None:
        try:
            self._json(HTTPStatus.ACCEPTED, self.server.benchmarks.start(self._read_json()))
        except (ValueError, BenchmarkError, KeyError, json.JSONDecodeError, UnicodeDecodeError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def _post_benchmark_cancel(self) -> None:
        try:
            self._json(HTTPStatus.OK, self.server.benchmarks.cancel())
        except BenchmarkError as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def _get_benchmark_result(self, query: dict[str, list[str]]) -> None:
        try:
            self._json(HTTPStatus.OK, self.server.benchmarks.inspect(query.get("result_index", [None])[0]))
        except BenchmarkError as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def _get_export(self, query: dict[str, list[str]]) -> None:
        try:
            body, content_type, filename = self.server.benchmarks.export(query.get("format", [""])[0])
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except BenchmarkError as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def _serve_static(self, raw_path: str) -> None:
        relative = "index.html" if raw_path in {"", "/"} else raw_path.lstrip("/")
        if relative.startswith("static/"):
            relative = relative.removeprefix("static/")
        requested = (STATIC_DIR / relative).resolve()
        if (STATIC_DIR not in requested.parents and requested != STATIC_DIR) or not requested.is_file():
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        body = requested.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(requested.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def definitions_from_args(args: argparse.Namespace) -> list[DatasetDefinition]:
    if args.package_dir:
        package_dir = args.package_dir.resolve()
        if not package_dir.is_dir():
            raise FileNotFoundError(f"Package directory does not exist: {args.package_dir}")
        return [DatasetDefinition(dataset_id, label, package_dir / filename, evaluation_only) for dataset_id, label, filename, evaluation_only in PACKAGE_DATASETS]
    dataset_path = args.dataset.resolve()
    return [DatasetDefinition("dataset", dataset_path.stem, dataset_path, False)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve local ARC practice and Ollama benchmarks.")
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--dataset", type=Path, help="One ARC task-level JSONL file")
    sources.add_argument("--package-dir", type=Path, help="Directory containing the three packaged ARC JSONL files")
    parser.add_argument("--port", default=8765, type=int, help="Local TCP port (default: 8765)")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Local Ollama HTTP address")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser window")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1024 <= args.port <= 65535:
        print("ERROR: --port must be between 1024 and 65535", file=sys.stderr)
        return 2
    try:
        registry = DatasetRegistry(definitions_from_args(args))
        server = PracticeServer(("127.0.0.1", args.port), registry, args.ollama_url)
    except (FileNotFoundError, ValueError, DatasetSchemaError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    url = f"http://127.0.0.1:{args.port}/"
    for dataset in registry.status()["datasets"]:
        print(f"Ready: {dataset['label']} — {dataset['playable_task_count']:,}/{dataset['task_count']:,} within {MAX_INTERACTIVE_GRID_SIDE}x{MAX_INTERACTIVE_GRID_SIDE}", flush=True)
    print(f"ARC Practice is running at {url}")
    print("Press Ctrl+C to stop the local server.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping ARC Practice.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
