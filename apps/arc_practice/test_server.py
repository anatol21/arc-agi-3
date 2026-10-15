"""Tests for the local ARC practice dataset registry and Ollama benchmark API."""

from __future__ import annotations

import json
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from apps.arc_practice.server import (
    BenchmarkManager,
    DEFAULT_CORE_INSTRUCTION,
    DatasetDefinition,
    DatasetRegistry,
    DatasetSchemaError,
    PracticeServer,
    TaskIndex,
    parse_model_outputs,
    task_prompt,
)


def task(task_id: str, source: str, output: list[list[int]]) -> dict:
    return {
        "task_id": task_id,
        "source": source,
        "task_fingerprint": "not-used-by-practice-app",
        "train": [{"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}],
        "test": [{"input": [[2, 2], [2, 0]], "output": output}],
        "metadata": {"provenance": "not sent to browser"},
    }


def write_records(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def registry_for(path: Path, dataset_id: str = "arc_validate") -> DatasetRegistry:
    return DatasetRegistry([DatasetDefinition(dataset_id, "ARC Validate", path, True)])


class TaskIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.dataset = Path(self.temp.name) / "tasks.jsonl"
        write_records(self.dataset, [task("alpha", "google_aux", [[3, 3], [3, 0]]), task("beta", "barc_aux", [[9]])])
        self.index = TaskIndex(DatasetDefinition("arc_train", "ARC Train", self.dataset, False))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_indexes_without_retaining_task_bodies(self) -> None:
        self.assertEqual(len(self.index.entries), 2)
        self.assertEqual(self.index.source_counts, {"barc_aux": 1, "google_aux": 1})
        self.assertEqual(self.index.choose("next", 0, None), 1)
        self.assertEqual(self.index.choose("previous", 0, None), 1)
        self.assertEqual(self.index.lookup_task_id("beta"), 1)

    def test_public_task_hides_test_outputs_and_metadata(self) -> None:
        public = self.index.public_task(0)
        self.assertEqual(public["test"], [{"input": [[2, 2], [2, 0]]}])
        self.assertNotIn("output", public["test"][0])
        self.assertNotIn("metadata", public)
        self.assertTrue(self.index.evaluate("alpha", 0, [[3, 3], [3, 0]]))
        self.assertFalse(self.index.evaluate("alpha", 0, [[0, 0], [0, 0]]))

    def test_rejects_pair_level_google_dataset(self) -> None:
        google = Path(self.temp.name) / "google.jsonl"
        google.write_text(json.dumps({"input": [[0]], "output": [[1]]}) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(DatasetSchemaError, "pair-level"):
            TaskIndex(google)

    def test_rejects_non_rectangular_grids(self) -> None:
        malformed = task("bad", "google_aux", [[1]])
        malformed["train"][0]["input"] = [[0], [0, 1]]
        path = Path(self.temp.name) / "bad.jsonl"
        write_records(path, [malformed])
        with self.assertRaisesRegex(DatasetSchemaError, "rectangular"):
            TaskIndex(path)

    def test_rejects_invalid_cell_when_the_task_is_selected(self) -> None:
        invalid = task("invalid", "google_aux", [[1]])
        invalid["train"][0]["input"] = [[10]]
        path = Path(self.temp.name) / "invalid.jsonl"
        write_records(path, [invalid])
        index = TaskIndex(path)
        with self.assertRaisesRegex(DatasetSchemaError, "integer from 0 to 9"):
            index.public_task(0)

    def test_filters_task_when_any_example_grid_exceeds_30_by_30(self) -> None:
        oversized = task("large", "barc_aux", [[1]])
        oversized["train"][0]["output"] = [[1] * 31]
        path = Path(self.temp.name) / "mixed.jsonl"
        write_records(path, [task("small", "google_aux", [[1]]), oversized])
        index = TaskIndex(path)
        self.assertEqual(index.status()["task_count"], 2)
        self.assertEqual(index.status()["playable_task_count"], 1)
        self.assertEqual(index.status()["oversize_task_count"], 1)
        self.assertEqual(index.choose("random", None, None), 0)
        with self.assertRaisesRegex(KeyError, "grid limit"):
            index.lookup_task_id("large")

    def test_registry_exposes_multiple_datasets_and_validate_default(self) -> None:
        paths = []
        for dataset_id, label in (("arc_train", "ARC Train"), ("arc_validate", "ARC Validate"), ("synthetic_arc_train", "Synthetic ARC Train")):
            path = Path(self.temp.name) / f"{dataset_id}.jsonl"
            write_records(path, [task(dataset_id, "source", [[1]])])
            paths.append(DatasetDefinition(dataset_id, label, path, dataset_id == "arc_validate"))
        registry = DatasetRegistry(paths)
        self.assertEqual(registry.status()["default_dataset_id"], "arc_validate")
        self.assertEqual([item["id"] for item in registry.status()["datasets"]], ["arc_train", "arc_validate", "synthetic_arc_train"])


class FakeOllama:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.requests: list[dict] = []

    def models(self) -> list[str]:
        return ["fake:latest"]

    def generate(self, payload: dict, timeout_seconds: int) -> object:
        self.requests.append({"payload": payload, "timeout_seconds": timeout_seconds})
        return self.responses.pop(0)


class BenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.dataset = Path(self.temp.name) / "tasks.jsonl"
        write_records(self.dataset, [task("alpha", "arc2_validation", [[3, 3], [3, 0]]), task("beta", "arc2_validation", [[8]])])
        self.registry = registry_for(self.dataset)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def config(self, **changes: object) -> dict:
        config = {
            "dataset_id": "arc_validate",
            "source": "",
            "model": "fake:latest",
            "order": "sequential",
            "task_count": 1,
            "seed": "17",
            "temperature": 0,
            "top_p": 1,
            "num_predict": 1024,
            "timeout_seconds": 3,
            "prompt_modifier": "Stay concise.",
        }
        config.update(changes)
        return config

    def wait_for_run(self, manager: BenchmarkManager) -> None:
        assert manager.run is not None
        manager.run.thread.join(timeout=2)
        self.assertFalse(manager.run.thread.is_alive())

    def test_prompt_exposes_only_train_and_test_inputs(self) -> None:
        indexed_task = self.registry.get("arc_validate").load(0)
        prompt = task_prompt(indexed_task, "Use symmetry.")
        self.assertIn('"train"', prompt)
        self.assertIn('"test"', prompt)
        self.assertIn("Use symmetry.", prompt)
        self.assertIn(DEFAULT_CORE_INSTRUCTION, prompt)
        custom = task_prompt(indexed_task, "", "First identify the invariant before solving.")
        self.assertTrue(custom.startswith("First identify the invariant before solving."))
        self.assertIn("Return exactly one JSON object", custom)
        self.assertNotIn('[[3,3],[3,0]]', prompt)
        self.assertNotIn("task_id", prompt)

    def test_benchmark_scores_server_side_and_export_hides_expected_answers(self) -> None:
        fake = FakeOllama([{"response": '{"outputs":[[[9]]]}', "thinking": "Find the changed cells."}])
        manager = BenchmarkManager(self.registry, fake)  # type: ignore[arg-type]
        manager.start(self.config())
        self.wait_for_run(manager)
        snapshot = manager.status()
        self.assertEqual(snapshot["state"], "completed")
        self.assertEqual(snapshot["task_correct_count"], 0)
        self.assertEqual(snapshot["results"][0]["status"], "incorrect")
        prompt = fake.requests[0]["payload"]["prompt"]
        self.assertNotIn('[[3,3],[3,0]]', prompt)
        json_export, _, _ = manager.export("json")
        self.assertNotIn('[[3,3],[3,0]]', json_export.decode("utf-8"))
        csv_export, _, _ = manager.export("csv")
        self.assertIn("alpha", csv_export.decode("utf-8"))

    def test_result_inspection_includes_model_output_but_not_targets(self) -> None:
        fake = FakeOllama([{"response": '{"outputs":[[[9]]]}', "thinking": "Find the changed cells."}])
        manager = BenchmarkManager(self.registry, fake)  # type: ignore[arg-type]
        manager.start(self.config())
        self.wait_for_run(manager)
        summary = manager.status()["results"][0]
        self.assertEqual(summary["result_index"], 0)
        self.assertNotIn("raw_response", summary)
        self.assertNotIn("model_request", summary)
        inspected = manager.inspect("0")
        self.assertEqual(inspected["raw_response"], '{"outputs":[[[9]]]}')
        self.assertEqual(inspected["reasoning"], "Find the changed cells.")
        self.assertEqual(inspected["prediction"], {"outputs": [[[9]]]})
        self.assertEqual(inspected["model_request"]["model"], "fake:latest")
        self.assertIn(DEFAULT_CORE_INSTRUCTION, inspected["model_request"]["prompt"])
        self.assertIn("Stay concise.", inspected["model_request"]["prompt"])
        self.assertNotIn("[[3,3],[3,0]]", inspected["model_request"]["prompt"])
        self.assertNotIn("expected", json.dumps(inspected))

    def test_reproducible_random_selection_and_strict_response_validation(self) -> None:
        fake = FakeOllama(['not-json', 'not-json'])
        manager = BenchmarkManager(self.registry, fake)  # type: ignore[arg-type]
        first = manager._config(self.config(order="random", task_count=2, seed="42"))[0]["positions"]
        second = manager._config(self.config(order="random", task_count=2, seed="42"))[0]["positions"]
        self.assertEqual(first, second)
        manager.start(self.config(order="random", task_count=2, seed="42"))
        self.wait_for_run(manager)
        self.assertEqual(manager.status()["status_counts"], {"malformed_response": 2})
        with self.assertRaisesRegex(Exception, "exactly"):
            parse_model_outputs('{"outputs":[[[-1]]],"extra":true}', 1)

    def test_ollama_error_is_recorded_without_ending_remaining_run(self) -> None:
        fake = FakeOllama(['{"outputs":[[[3,3],[3,0]]]}', '{"outputs":[[[8]]]}'])
        manager = BenchmarkManager(self.registry, fake)  # type: ignore[arg-type]
        manager.start(self.config(task_count=2))
        self.wait_for_run(manager)
        self.assertEqual(manager.status()["task_correct_count"], 2)
        self.assertEqual(manager.status()["test_correct_count"], 2)


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        dataset = Path(self.temp.name) / "tasks.jsonl"
        write_records(dataset, [task("alpha", "google_aux", [[3, 3], [3, 0]])])
        self.registry = registry_for(dataset)
        self.server = PracticeServer(("127.0.0.1", 0), self.registry, "http://127.0.0.1:11434")
        self.server.benchmarks = BenchmarkManager(self.registry, FakeOllama(['{"outputs":[[[3,3],[3,0]]]}']))  # type: ignore[arg-type]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def request_json(self, path: str, payload: dict | None = None) -> dict:
        request = Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8") if payload is not None else None,
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        with urlopen(request, timeout=3) as response:
            return json.loads(response.read())

    def test_api_hides_answer_and_scores_server_side(self) -> None:
        public = self.request_json("/api/task?dataset=arc_validate&task_id=alpha")
        self.assertEqual(public["test"], [{"input": [[2, 2], [2, 0]]}])
        self.assertNotIn("metadata", public)
        correct = self.request_json("/api/attempt", {"dataset_id": "arc_validate", "task_id": "alpha", "test_index": 0, "grid": [[3, 3], [3, 0]]})
        self.assertEqual(correct, {"correct": True, "message": "Correct solution!"})

    def test_api_rejects_bad_submissions_and_exposes_dataset_status(self) -> None:
        status = self.request_json("/api/datasets")
        self.assertEqual(status["default_dataset_id"], "arc_validate")
        with self.assertRaises(HTTPError) as error:
            self.request_json("/api/attempt", {"dataset_id": "arc_validate", "task_id": "alpha", "test_index": 0, "grid": [[11]]})
        self.assertEqual(error.exception.code, 400)

    def test_api_inspection_reveals_only_model_response(self) -> None:
        self.request_json("/api/benchmark/start", {
            "dataset_id": "arc_validate", "source": "", "model": "fake:latest", "order": "sequential",
            "task_count": 1, "seed": "", "temperature": 0, "top_p": 1, "num_predict": 20,
            "timeout_seconds": 3, "prompt_modifier": "",
        })
        assert self.server.benchmarks.run is not None
        self.server.benchmarks.run.thread.join(timeout=2)
        result = self.request_json("/api/benchmark/result?result_index=0")
        self.assertEqual(result["prediction"], {"outputs": [[[3, 3], [3, 0]]]})
        self.assertIn("prompt", result["model_request"])
        self.assertNotIn("[[3,3],[3,0]]", result["model_request"]["prompt"])
        public_fields = {
            key: value
            for key, value in result.items()
            if key not in {"prediction", "raw_response", "model_request"}
        }
        self.assertNotIn("output", public_fields)


if __name__ == "__main__":
    unittest.main(verbosity=2)
