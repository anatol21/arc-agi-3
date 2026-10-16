"""Tests for the local ARC practice dataset registry and model benchmark API."""

from __future__ import annotations

import json
import os
import ssl
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import Mock, patch

from apps.arc_practice.server import (
    BenchmarkManager,
    BenchmarkError,
    DEFAULT_CORE_INSTRUCTION,
    DatasetDefinition,
    DatasetRegistry,
    DatasetSchemaError,
    ExperimentBatchManager,
    OllamaClient,
    OpenRouterClient,
    PracticeServer,
    ProviderRequestError,
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
        self.lock = threading.Lock()

    def models(self) -> list[str]:
        return ["fake:latest"]

    @staticmethod
    def request_payload(config: dict, prompt: str) -> dict:
        return OllamaClient.request_payload(config, prompt)

    def generate(self, payload: dict, timeout_seconds: int, *, on_progress=None) -> object:
        with self.lock:
            self.requests.append({"payload": payload, "timeout_seconds": timeout_seconds})
            return self.responses.pop(0)


class FakeOpenRouter:
    def __init__(self, responses: list[object], delay_seconds: float = 0) -> None:
        self.responses = responses
        self.delay_seconds = delay_seconds
        self.requests: list[dict] = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    @staticmethod
    def request_payload(config: dict, prompt: str) -> dict:
        return OpenRouterClient.request_payload(config, prompt)

    def generate(self, payload: dict, timeout_seconds: int, *, on_progress=None) -> object:
        with self.lock:
            self.requests.append({"payload": payload, "timeout_seconds": timeout_seconds})
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            response = self.responses.pop(0)
        try:
            if self.delay_seconds:
                time.sleep(self.delay_seconds)
            if isinstance(response, Exception):
                raise response
            return response
        finally:
            with self.lock:
                self.active -= 1


class OpenRouterTransportTests(unittest.TestCase):
    """Exercise actual HTTP reads rather than replacing the provider client."""

    def setUp(self) -> None:
        self.status_code = 200
        self.content_type = "text/event-stream"
        self.chunks: list[bytes] = []
        self.chunk_delay = 0.0
        self.calls: list[dict] = []
        self.release = threading.Event()
        self.pause_after_first = False
        suite = self

        class ProviderHandler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_POST(self) -> None:
                suite.calls.append({
                    "path": self.path,
                    "authorization": self.headers.get("Authorization"),
                    "payload": json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                })
                self.send_response(suite.status_code)
                self.send_header("Content-Type", suite.content_type)
                self.end_headers()
                try:
                    for index, chunk in enumerate(suite.chunks):
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        if suite.pause_after_first and index == 0:
                            suite.release.wait(timeout=3)
                        if suite.chunk_delay:
                            time.sleep(suite.chunk_delay)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), ProviderHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.endpoint_patch = patch(
            "apps.arc_practice.server.OPENROUTER_API_URL",
            f"http://127.0.0.1:{self.server.server_address[1]}/api/v1",
        )
        self.endpoint_patch.start()
        self.client = OpenRouterClient("sk-or-v1-transport-test")
        self.payload = OpenRouterClient.request_payload({
            "model": "qwen/test", "temperature": 0.2, "top_p": 0.95,
            "num_predict": 16_384, "think": True,
        }, "Solve this task.")

    def tearDown(self) -> None:
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.endpoint_patch.stop()

    @staticmethod
    def event(body: dict) -> bytes:
        return ("data: " + json.dumps(body, ensure_ascii=False) + "\n\n").encode("utf-8")

    def test_stream_delivers_reasoning_and_answer_with_usage(self) -> None:
        self.pause_after_first = True
        self.chunks = [
            self.event({"id": "gen-test", "provider": "Alibaba", "choices": [{"index": 0, "delta": {"reasoning": "Find symmetry ▣"}}]}),
            b": OPENROUTER PROCESSING\r\n\r\n",
            self.event({"choices": [{"index": 0, "delta": {"content": '{"outputs":'}}]}),
            self.event({"choices": [{"index": 0, "delta": {"content": '[[[1]]]}'} , "finish_reason": "stop"}]}),
            self.event({"choices": [], "usage": {"total_tokens": 24, "cost": 0.001}}),
            b"data: [DONE]\n\n",
        ]
        updates: list[dict] = []
        received_reasoning = threading.Event()
        result: dict = {}

        def progress(update: dict) -> None:
            updates.append(update)
            if update.get("reasoning_chars"):
                received_reasoning.set()

        def generate() -> None:
            result.update(self.client.generate(self.payload, 3, on_progress=progress))

        worker = threading.Thread(target=generate)
        worker.start()
        try:
            self.assertTrue(received_reasoning.wait(timeout=1), "Progress must arrive before the answer finishes")
            self.assertTrue(worker.is_alive())
        finally:
            self.release.set()
            worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result["response"], '{"outputs":[[[1]]]}')
        self.assertEqual(result["thinking"], "Find symmetry ▣")
        self.assertEqual(result["usage"]["total_tokens"], 24)
        self.assertEqual(result["provider_response"]["finish_reason"], "stop")
        self.assertEqual(result["provider_response"]["id"], "gen-test")
        self.assertIn("receiving answer", [update["stage"] for update in updates])
        self.assertEqual(self.calls[0]["path"], "/api/v1/chat/completions")
        self.assertTrue(self.calls[0]["payload"]["stream"])
        self.assertEqual(self.calls[0]["payload"]["reasoning"]["max_tokens"], 8192)
        self.assertEqual(self.calls[0]["authorization"], "Bearer sk-or-v1-transport-test")

    def test_json_reasoning_only_reply_retains_trace_and_length_reason(self) -> None:
        self.content_type = "application/json"
        self.chunks = [json.dumps({
            "id": "gen-length", "provider": "Alibaba",
            "choices": [{"message": {"content": None, "reasoning": "Still reasoning"}, "finish_reason": "length"}],
            "usage": {"completion_tokens": 16384, "completion_tokens_details": {"reasoning_tokens": 16384}},
        }).encode()]
        result = self.client.generate(self.payload, 2)
        self.assertEqual(result["response"], "")
        self.assertEqual(result["thinking"], "Still reasoning")
        self.assertIn("maximum-token budget", result["response_error"])
        self.assertEqual(result["provider_response"]["finish_reason"], "length")
        self.assertEqual(result["usage"]["completion_tokens"], 16384)

    def test_http_200_error_is_reported_with_provider_details_and_redaction(self) -> None:
        self.content_type = "application/json"
        self.chunks = [json.dumps({
            "id": "gen-error", "error": {"code": 429, "message": "Busy sk-or-v1-transport-test",
                "metadata": {"provider_name": "Alibaba", "error_type": "rate_limit_exceeded"}},
        }).encode()]
        with self.assertRaises(ProviderRequestError) as raised:
            self.client.generate(self.payload, 2)
        self.assertTrue(raised.exception.transient)
        self.assertEqual(raised.exception.details["id"], "gen-error")
        self.assertIn("Alibaba", str(raised.exception))
        self.assertNotIn(self.client.api_key, str(raised.exception))

    def test_midstream_error_is_not_scored_as_a_successful_answer(self) -> None:
        self.chunks = [
            self.event({"choices": [{"delta": {"content": '{"outputs":'}}]}),
            self.event({"error": {"code": 502, "message": "Provider disconnected"},
                        "choices": [{"delta": {}, "finish_reason": "error"}]}),
        ]
        with self.assertRaisesRegex(ProviderRequestError, "Provider disconnected") as raised:
            self.client.generate(self.payload, 2)
        self.assertTrue(raised.exception.transient)

    def test_choice_error_is_not_hidden_by_partial_content(self) -> None:
        self.content_type = "application/json"
        self.chunks = [json.dumps({"choices": [{
            "message": {"content": '{"outputs":[[[1]]]}'}, "finish_reason": "error",
            "error": {"code": 403, "message": "Content blocked"},
        }]}).encode()]
        with self.assertRaisesRegex(ProviderRequestError, "Content blocked") as raised:
            self.client.generate(self.payload, 2)
        self.assertFalse(raised.exception.transient)

    def test_http_failure_retains_its_json_error(self) -> None:
        self.status_code = 401
        self.content_type = "application/json"
        self.chunks = [b'{"error":{"message":"Invalid API key","code":401}}']
        with self.assertRaisesRegex(ProviderRequestError, "Invalid API key") as raised:
            self.client.generate(self.payload, 2)
        self.assertFalse(raised.exception.transient)

    def test_http_failure_with_html_body_still_has_retryable_status(self) -> None:
        self.status_code = 502
        self.content_type = "text/html"
        self.chunks = [b"<html>Bad gateway</html>"]
        with self.assertRaisesRegex(ProviderRequestError, "502") as raised:
            self.client.generate(self.payload, 2)
        self.assertTrue(raised.exception.transient)

    def test_stream_without_completion_marker_is_a_provider_failure(self) -> None:
        self.chunks = [self.event({"choices": [{"delta": {"content": '{"outputs":'}}]})]
        with self.assertRaisesRegex(ProviderRequestError, "ended before completion"):
            self.client.generate(self.payload, 2)

    def test_keepalives_cannot_extend_overall_request_deadline(self) -> None:
        self.chunks = [b": OPENROUTER PROCESSING\n\n"] * 30
        self.chunk_delay = 0.1
        started = time.monotonic()
        with self.assertRaisesRegex(ProviderRequestError, "timed out after 1s"):
            self.client.generate(self.payload, 1)
        self.assertLess(time.monotonic() - started, 1.6)

    def test_stream_parser_handles_split_utf8_and_multiline_events(self) -> None:
        event = 'data: {"choices": [\r\ndata: {"delta":{"reasoning":"▣"}}]}\r\n\r\n'.encode()
        finish = self.event({"choices": [{"delta": {"content": '{"outputs":[[[1]]]}'}, "finish_reason": "stop"}]})
        data = event + finish + b"data: [DONE]"

        class FragmentedResponse:
            def __init__(self) -> None:
                self.fragments = iter(bytes([byte]) for byte in data)

            def read1(self, size) -> bytes:
                return next(self.fragments, b"")

        result = self.client._read_stream(FragmentedResponse(), lambda: None, None)
        self.assertEqual(result["choices"][0]["message"]["reasoning"], "▣")
        self.assertEqual(result["choices"][0]["message"]["content"], '{"outputs":[[[1]]]}')

    def test_batch_scores_real_streamed_reply_and_retains_provider_details(self) -> None:
        self.chunks = [
            self.event({"id": "gen-batch", "provider": "Alibaba", "choices": [{"delta": {"reasoning": "Infer rule"}}]}),
            self.event({"choices": [{"delta": {"content": '{"outputs":[[[1]]]}'}, "finish_reason": "stop"}]}),
            self.event({"usage": {"total_tokens": 20}, "choices": []}),
            b"data: [DONE]\n\n",
        ]
        with TemporaryDirectory() as directory:
            dataset = Path(directory) / "tasks.jsonl"
            write_records(dataset, [task("alpha", "validation", [[1]])])
            manager = ExperimentBatchManager(registry_for(dataset))
            manager.start({
                "dataset_id": "arc_validate", "openrouter_api_key": self.client.api_key,
                "task_count": 1, "seed": 42, "trials_per_task": 1, "global_parallel_requests": 1,
                "experiments": [{
                    "label": "Stream check", "model": "qwen/test", "temperature": 0.2, "top_p": 0.95,
                    "num_predict": 16384, "timeout_seconds": 2, "think": True,
                }],
            })
            manager.batch.thread.join(timeout=3)
            self.assertFalse(manager.batch.thread.is_alive())
            result = manager.inspect("0")
            self.assertEqual(result["status"], "correct")
            self.assertTrue(result["provider_success"])
            self.assertEqual(result["reasoning"], "Infer rule")
            self.assertEqual(result["provider_response"]["id"], "gen-batch")
            self.assertEqual(result["usage"]["total_tokens"], 20)
            self.assertEqual(manager.status()["in_flight"], [])
            exported, _, _ = manager.export("json")
            self.assertNotIn(self.client.api_key, exported.decode())

    def test_benchmark_retains_reasoning_only_reply_as_model_failure(self) -> None:
        self.content_type = "application/json"
        self.chunks = [json.dumps({
            "id": "gen-incomplete", "choices": [{
                "message": {"content": None, "reasoning": "Unfinished thought"}, "finish_reason": "length",
            }], "usage": {"completion_tokens": 16384},
        }).encode()]
        with TemporaryDirectory() as directory:
            dataset = Path(directory) / "tasks.jsonl"
            write_records(dataset, [task("alpha", "validation", [[1]])])
            manager = BenchmarkManager(registry_for(dataset), FakeOllama([]))
            manager.start({
                "dataset_id": "arc_validate", "provider": "openrouter", "openrouter_api_key": self.client.api_key,
                "model": "qwen/test", "order": "sequential", "task_count": 1,
                "temperature": 0.2, "top_p": 0.95, "num_predict": 16384, "timeout_seconds": 2, "think": True,
            })
            manager.run.thread.join(timeout=3)
            self.assertFalse(manager.run.thread.is_alive())
            result = manager.inspect("0")
            self.assertEqual(result["status"], "malformed_response")
            self.assertEqual(result["raw_response"], "")
            self.assertEqual(result["reasoning"], "Unfinished thought")
            self.assertIn("maximum-token budget", result["error"])
            self.assertEqual(result["provider_response"]["finish_reason"], "length")
            self.assertEqual(result["usage"]["completion_tokens"], 16384)
            self.assertEqual(manager.status()["in_flight"], [])

    def test_macos_without_python_roots_loads_system_bundle(self) -> None:
        context = Mock()
        context.get_ca_certs.return_value = []
        with patch("apps.arc_practice.server.ssl.create_default_context", return_value=context), \
             patch("apps.arc_practice.server.sys.platform", "darwin"), \
             patch("apps.arc_practice.server.Path.is_file", return_value=True), \
             patch.dict(os.environ, {}, clear=True):
            self.assertIs(OpenRouterClient._ssl_context(), context)
        context.load_verify_locations.assert_called_once_with(cafile="/etc/ssl/cert.pem")

    def test_existing_python_roots_do_not_add_another_bundle(self) -> None:
        context = Mock()
        context.get_ca_certs.return_value = [{"issuer": "configured root"}]
        with patch("apps.arc_practice.server.ssl.create_default_context", return_value=context), \
             patch("apps.arc_practice.server.sys.platform", "darwin"):
            OpenRouterClient._ssl_context()
        context.load_verify_locations.assert_not_called()

    def test_explicit_ca_configuration_is_respected(self) -> None:
        for setting in ("SSL_CERT_FILE", "SSL_CERT_DIR"):
            with self.subTest(setting=setting):
                context = Mock()
                context.get_ca_certs.return_value = []
                with patch("apps.arc_practice.server.ssl.create_default_context", return_value=context), \
                     patch("apps.arc_practice.server.sys.platform", "darwin"), \
                     patch.dict(os.environ, {setting: "/configured/trust"}, clear=True):
                    OpenRouterClient._ssl_context()
                context.load_verify_locations.assert_not_called()

    def test_certificate_failure_is_not_retried_as_a_transient_provider_error(self) -> None:
        with patch("apps.arc_practice.server.OPENROUTER_API_URL", "https://openrouter.ai/api/v1"), \
             patch("apps.arc_practice.server.HTTPSConnection") as connection:
            connection.return_value.request.side_effect = ssl.SSLCertVerificationError("untrusted certificate")
            with self.assertRaisesRegex(ProviderRequestError, "certificate verification failed") as raised:
                self.client.generate(self.payload, 2)
        self.assertFalse(raised.exception.transient)
        self.assertEqual(raised.exception.details["error_type"], "tls_certificate_verification_failed")


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

    def test_accepts_a_long_core_instruction_without_truncating_it(self) -> None:
        manager = BenchmarkManager(self.registry, FakeOllama([]))  # type: ignore[arg-type]
        core_instruction = "A" * 5_031
        config, _ = manager._config(self.config(core_instruction=core_instruction))
        self.assertEqual(config["core_instruction"], core_instruction)

    def test_rejects_prompt_fields_larger_than_the_shared_limit(self) -> None:
        manager = BenchmarkManager(self.registry, FakeOllama([]))  # type: ignore[arg-type]
        too_long = "A" * 32_769
        with self.assertRaisesRegex(BenchmarkError, "32,768"):
            manager._config(self.config(core_instruction=too_long))
        with self.assertRaisesRegex(BenchmarkError, "32,768"):
            manager._config(self.config(prompt_modifier=too_long))

    def test_benchmark_scores_server_side_and_export_hides_expected_answers(self) -> None:
        fake = FakeOllama([{"response": '{"outputs":[[[9]]]}', "thinking": "Find the changed cells."}])
        manager = BenchmarkManager(self.registry, fake)  # type: ignore[arg-type]
        manager.start(self.config())
        self.wait_for_run(manager)
        snapshot = manager.status()
        self.assertEqual(snapshot["state"], "completed")
        self.assertEqual(snapshot["task_correct_count"], 0)
        self.assertEqual(snapshot["results"][0]["status"], "incorrect")
        prompt = fake.requests[0]["payload"]["messages"][0]["content"]
        self.assertNotIn('[[3,3],[3,0]]', prompt)
        json_export, _, _ = manager.export("json")
        self.assertNotIn('[[3,3],[3,0]]', json_export.decode("utf-8"))
        csv_export, _, _ = manager.export("csv")
        self.assertIn("alpha", csv_export.decode("utf-8"))

    def test_benchmark_requests_reasoning_when_enabled(self) -> None:
        fake = FakeOllama([{"response": '{"outputs":[[[3,3],[3,0]]]}', "thinking": "Find the changed cells."}])
        manager = BenchmarkManager(self.registry, fake)  # type: ignore[arg-type]
        manager.start(self.config(think=True))
        self.wait_for_run(manager)
        self.assertTrue(fake.requests[0]["payload"]["think"])
        self.assertTrue(manager.status()["config"]["think"])

    def test_ollama_client_uses_chat_endpoint_and_reads_thinking(self) -> None:
        client = OllamaClient("http://127.0.0.1:11434")
        calls: list[tuple[str, dict, int]] = []

        def fake_request(path: str, payload: dict, *, timeout_seconds: int) -> dict:
            calls.append((path, payload, timeout_seconds))
            return {"message": {"content": '{"outputs":[[[1]]]}', "thinking": "Infer the rule."}}

        client._request = fake_request  # type: ignore[method-assign]
        result = client.generate({"model": "fake:latest", "messages": []}, 3)
        self.assertEqual(calls, [("/api/chat", {"model": "fake:latest", "messages": []}, 3)])
        self.assertEqual(result, {"response": '{"outputs":[[[1]]]}', "thinking": "Infer the rule."})

    def test_openrouter_client_uses_compatible_payload_and_response(self) -> None:
        client = OpenRouterClient("sk-or-v1-test-key")
        calls: list[tuple[str, dict | None, int]] = []

        def fake_request(path: str, payload: dict | None, *, timeout_seconds: int, on_progress=None) -> dict:
            calls.append((path, payload, timeout_seconds))
            if path == "/models":
                return {"data": [{"id": "openai/gpt-4.1"}, {"id": "meta-llama/llama-3.3"}]}
            return {
                "choices": [{"message": {
                    "content": '{"outputs":[[[1]]]}',
                    "reasoning_details": [{"type": "reasoning.summary", "text": "Infer the rule."}],
                }}]
            }

        client._request = fake_request  # type: ignore[method-assign]
        self.assertEqual(client.models(), ["meta-llama/llama-3.3", "openai/gpt-4.1"])
        payload = client.request_payload(self.config(provider="openrouter", think=True), "Solve the task.")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertEqual(payload["reasoning"], {"enabled": True})
        self.assertNotIn("options", payload)
        self.assertNotIn("think", payload)
        result = client.generate(payload, 3)
        self.assertEqual(result["response"], '{"outputs":[[[1]]]}')
        self.assertIn("Infer the rule.", result["thinking"])
        self.assertEqual(calls[1], ("/chat/completions", payload, 3))

    def test_openrouter_configuration_requires_key_without_retaining_it(self) -> None:
        manager = BenchmarkManager(self.registry, FakeOllama([]))  # type: ignore[arg-type]
        with self.assertRaisesRegex(BenchmarkError, "OpenRouter API key"):
            manager._config(self.config(provider="openrouter"))
        config, _ = manager._config(
            self.config(provider="openrouter", openrouter_api_key="sk-or-v1-test-key", think=True)
        )
        self.assertEqual(config["provider"], "openrouter")
        self.assertTrue(config["think"])
        self.assertNotIn("openrouter_api_key", config)

    def test_openrouter_allows_its_larger_output_limit_and_parallel_requests(self) -> None:
        manager = BenchmarkManager(self.registry, FakeOllama([]))  # type: ignore[arg-type]
        config, _ = manager._config(
            self.config(
                provider="openrouter",
                openrouter_api_key="sk-or-v1-test-key",
                num_predict=131_072,
                parallel_requests=16,
            )
        )
        self.assertEqual(config["num_predict"], 131_072)
        self.assertEqual(config["parallel_requests"], 16)
        with self.assertRaisesRegex(BenchmarkError, "131072"):
            manager._config(
                self.config(
                    provider="openrouter",
                    openrouter_api_key="sk-or-v1-test-key",
                    num_predict=131_073,
                )
            )

    def test_benchmark_can_process_tasks_in_parallel(self) -> None:
        fake = FakeOllama(['{"outputs":[[[3,3],[3,0]]]}', '{"outputs":[[[8]]]}'])
        manager = BenchmarkManager(self.registry, fake)  # type: ignore[arg-type]
        manager.start(self.config(task_count=2, parallel_requests=2))
        self.wait_for_run(manager)
        status = manager.status()
        self.assertEqual(status["state"], "completed")
        self.assertEqual(status["completed_task_count"], 2)
        self.assertEqual(status["config"]["parallel_requests"], 2)
        self.assertEqual([result["task_id"] for result in status["results"]], ["alpha", "beta"])

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
        prompt = inspected["model_request"]["messages"][0]["content"]
        self.assertIn(DEFAULT_CORE_INSTRUCTION, prompt)
        self.assertIn("Stay concise.", prompt)
        self.assertNotIn("[[3,3],[3,0]]", prompt)
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


class ExperimentBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.dataset = Path(self.temp.name) / "tasks.jsonl"
        write_records(self.dataset, [task("alpha", "arc2_validation", [[3, 3], [3, 0]]), task("beta", "arc2_validation", [[8]])])
        self.registry = registry_for(self.dataset)

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def experiment(label: str) -> dict:
        return {
            "label": label,
            "model": "fake/model",
            "temperature": 0,
            "top_p": 1,
            "num_predict": 64,
            "timeout_seconds": 3,
            "think": False,
            "core_instruction": DEFAULT_CORE_INSTRUCTION,
            "prompt_modifier": "",
        }

    def config(self, **changes: object) -> dict:
        config = {
            "dataset_id": "arc_validate",
            "source": "",
            "openrouter_api_key": "sk-or-v1-test-key",
            "task_count": 2,
            "seed": 42,
            "trials_per_task": 3,
            "global_parallel_requests": 2,
            "experiments": [self.experiment("Fast"), self.experiment("Careful")],
        }
        config.update(changes)
        return config

    def wait_for_batch(self, manager: ExperimentBatchManager) -> None:
        assert manager.batch is not None
        manager.batch.thread.join(timeout=4)
        self.assertFalse(manager.batch.thread.is_alive())

    def test_batch_uses_a_shared_cohort_and_global_cap(self) -> None:
        fake = FakeOpenRouter(["not-json"] * 12, delay_seconds=0.01)
        manager = ExperimentBatchManager(self.registry, lambda _: fake)
        manager.start(self.config())
        self.wait_for_batch(manager)
        status = manager.status()
        self.assertEqual(status["state"], "completed")
        self.assertEqual(status["configured_trial_count"], 12)
        self.assertEqual(status["completed_trial_count"], 12)
        self.assertLessEqual(fake.max_active, 2)
        self.assertEqual(len(status["cohort"]), 2)
        self.assertEqual({item["completed_trials"] for item in status["experiments"]}, {6})
        self.assertEqual({item["target_trials"] for item in status["experiments"]}, {6})
        self.assertEqual({item["valid_responses"] for item in status["experiments"]}, {0})
        self.assertEqual({item["exact_trials"] for item in status["experiments"]}, {0})
        first_experiments = {result["experiment_id"] for result in status["results"][:2]}
        self.assertEqual(first_experiments, {"experiment_1", "experiment_2"})

    def test_batch_retries_only_transient_provider_errors_and_hides_secrets(self) -> None:
        fake = FakeOpenRouter([
            ProviderRequestError("temporary", transient=True),
            {"response": "not-json", "thinking": None, "usage": {"total_tokens": 3}},
        ])
        manager = ExperimentBatchManager(self.registry, lambda _: fake)
        manager.start(self.config(task_count=1, trials_per_task=1, experiments=[self.experiment("Retry")]))
        self.wait_for_batch(manager)
        status = manager.status()
        self.assertEqual(len(fake.requests), 2)
        result = manager.inspect("0")
        self.assertEqual(result["status"], "malformed_response")
        self.assertEqual(len(result["attempts"]), 2)
        self.assertTrue(result["provider_success"])
        self.assertFalse(result["valid_response"])
        self.assertEqual(status["experiments"][0]["target_trials"], 1)
        self.assertEqual(status["experiments"][0]["exact_trials"], 0)
        exported, _, _ = manager.export("json")
        report = exported.decode("utf-8")
        self.assertNotIn("sk-or-v1-test-key", report)
        self.assertNotIn("[[3, 3], [3, 0]]", report)


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

    def test_api_preserves_a_long_core_instruction(self) -> None:
        core_instruction = "Long instruction. " * 296
        self.request_json("/api/benchmark/start", {
            "dataset_id": "arc_validate", "source": "", "model": "fake:latest", "order": "sequential",
            "task_count": 1, "seed": "", "temperature": 0, "top_p": 1, "num_predict": 20,
            "timeout_seconds": 3, "prompt_modifier": "", "core_instruction": core_instruction,
        })
        assert self.server.benchmarks.run is not None
        self.server.benchmarks.run.thread.join(timeout=2)
        result = self.request_json("/api/benchmark/result?result_index=0")
        prompt = result["model_request"]["messages"][0]["content"]
        self.assertTrue(prompt.startswith(core_instruction.strip()))

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
        self.assertIn("messages", result["model_request"])
        self.assertNotIn("[[3,3],[3,0]]", result["model_request"]["messages"][0]["content"])
        public_fields = {
            key: value
            for key, value in result.items()
            if key not in {"prediction", "raw_response", "model_request"}
        }
        self.assertNotIn("output", public_fields)

    def test_batch_api_reports_reliability_failures_without_exposing_secrets(self) -> None:
        fake = FakeOpenRouter(["not-json"])
        self.server.experiment_batches = ExperimentBatchManager(self.registry, lambda _: fake)
        self.request_json("/api/benchmark/batch/start", {
            "dataset_id": "arc_validate",
            "source": "",
            "openrouter_api_key": "sk-or-v1-test-key",
            "task_count": 1,
            "seed": 42,
            "trials_per_task": 1,
            "global_parallel_requests": 1,
            "experiments": [{
                "label": "JSON check",
                "model": "fake/model",
                "temperature": 0,
                "top_p": 1,
                "num_predict": 64,
                "timeout_seconds": 3,
                "think": False,
                "core_instruction": DEFAULT_CORE_INSTRUCTION,
                "prompt_modifier": "",
            }],
        })
        assert self.server.experiment_batches.batch is not None
        self.server.experiment_batches.batch.thread.join(timeout=2)
        status = self.request_json("/api/benchmark/batch/status")
        self.assertEqual(status["state"], "completed")
        self.assertEqual(status["experiments"][0]["target_trials"], 1)
        self.assertEqual(status["experiments"][0]["valid_responses"], 0)
        self.assertNotIn("sk-or-v1-test-key", json.dumps(status))
        result = self.request_json("/api/benchmark/batch/result?result_index=0")
        self.assertEqual(result["status"], "malformed_response")
        self.assertNotIn("[[3,3],[3,0]]", json.dumps(result))


if __name__ == "__main__":
    unittest.main(verbosity=2)
