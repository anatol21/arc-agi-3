"""Behavior and fault-injection tests; all scheduler operations are mocked or simulated."""
import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core
import runtime
import sdk
import server
import transport

class ConfigurationTests(unittest.TestCase):
    def test_real_yaml_and_duplicate_keys(self):
        value = core.parse_yaml("name: test\ncommand:\n  - python\n  - script.py\nrecipe:\n  nested: [1, 2]\n")
        self.assertEqual(value["recipe"]["nested"], [1, 2])
        for text in ("name: a\nname: b", "x: &x [*x]", "x: !!python/object:builtins.object {}", "date: 2026-11-06"):
            with self.assertRaises(core.ConfigError): core.parse_yaml(text)

    def test_invalid_fields_fail_before_submission(self):
        for extra in ({"auto_continue": True}, {"workers": 0}, {"gpus": True}, {"concurrency": 2}, {"notes": []}, {"completion": None}, {"budget": {"gpu_hours": float("nan")}}, {"environment": {"WM_RUN_DIR": "/other"}}, {"command": ["echo", "\x00"]}):
            with self.assertRaises((core.ConfigError, ValueError)):
                core.normalize_config({"command": ["python", "adapter.py"], **extra})

    def test_identity_ignores_operational_changes(self):
        cfg = core.normalize_config({"command": ["python", "adapter.py"], "workers": 4})
        for changes in ({"name": "renamed"}, {"concurrency": 3}, {"notes": "notes"}, {"budget": {"gpu_hours": 4}}):
            self.assertEqual(core.experiment_identity(cfg), core.experiment_identity({**cfg, **changes}))
        for changes in ({"workers": 5}, {"recipe": {"batch_size": 2}}, {"fingerprints": {"model": "changed"}}, {"command": ["python", "other.py"]}):
            self.assertNotEqual(core.experiment_identity(cfg), core.experiment_identity({**cfg, **changes}))

    def test_estimates_keep_wait_unknown_and_update_observations(self):
        cfg = core.normalize_config({"command": ["a"], "task_count": 10, "estimates": {"seconds_per_task": 2, "model_load_seconds": 3, "preparation_seconds": 0, "restore_seconds": 0}})
        p = {"name": "gpu", "time_limit": "00:01:00", "concurrency_limit": 1}
        plan = core.estimate_plans(cfg, [p])[0]
        self.assertIsNone(plan["queue_seconds"])
        self.assertIsNone(plan["finish_seconds"])
        plan = core.estimate_plans(cfg, [{**p, "queue_seconds": 5}], [{"partition": "gpu", "identity": core.experiment_identity(cfg), "seconds_per_task": 1, "model_load_seconds": 2}])[0]
        self.assertEqual((plan["queue_seconds"], plan["startup_seconds"], plan["execution_seconds"], plan["finish_seconds"]), (5, 2, 10, 17))

    def test_missing_startup_measurements_are_not_zero(self):
        estimates = {"seconds_per_task": 2, "model_load_seconds": 0, "preparation_seconds": 0, "restore_seconds": 0}
        partition = {"name": "gpu", "time_limit": "00:01:00", "concurrency_limit": 1, "queue_seconds": 5}
        for missing in ("model_load_seconds", "preparation_seconds", "restore_seconds"):
            cfg = core.normalize_config({"command": ["a"], "task_count": 10, "estimates": {k: v for k, v in estimates.items() if k != missing}})
            plan = core.estimate_plans(cfg, [partition])[0]
            self.assertEqual(plan["execution_seconds"], 20)
            for key in ("startup_seconds", "first_result_seconds", "finish_seconds", "estimated_slices"):
                self.assertIsNone(plan[key], (missing, key))
            observed = {"partition": "gpu", "identity": core.experiment_identity(cfg), missing: 3}
            measured = core.estimate_plans(cfg, [partition], [observed])[0]
            self.assertEqual(measured["startup_seconds"], 3)
            self.assertEqual(measured["finish_seconds"], 28)

    def test_shards_cover_uneven_and_empty_tails_once(self):
        for count in (1, 2, 7, 19):
            for workers in (1, 2, 5, 23):
                assignments = [[i for i in range(count) if i % workers == w] for w in range(workers)]
                self.assertEqual(sorted(i for group in assignments for i in group), list(range(count)))
        self.assertEqual([i for i in range(19) if i % 5 == 4], [4, 9, 14])

class LocalHTTPGuardTests(unittest.TestCase):
    def handler(self, host='127.0.0.1:8766', token=None):
        from types import SimpleNamespace
        handler = object.__new__(server.Handler)
        handler.server = SimpleNamespace(server_port=8766, app=SimpleNamespace(token='test-token'))
        handler.headers = {'Host': host}
        if token is not None: handler.headers['X-Dashboard-Token'] = token
        handler.replies = []
        handler.reply = lambda value, status=200: handler.replies.append((status, value))
        return handler

    def test_foreign_host_and_missing_token_are_rejected_before_dispatch(self):
        for host, token in (('attacker.invalid:8766', 'test-token'), ('127.0.0.1:8766', None), ('127.0.0.1:8766', 'wrong')):
            handler = self.handler(host, token)
            handler.do_POST()
            self.assertEqual(handler.replies[0][0], 403)
        handler = self.handler('attacker.invalid:8766')
        handler.do_GET()
        self.assertEqual(handler.replies[0][0], 403)

    def test_static_page_retains_script_and_frame_restrictions(self):
        handler = self.handler()
        handler.path = '/'
        headers = {}
        handler.send_response = lambda status: None
        handler.send_header = lambda key, value: headers.update({key:value})
        handler.end_headers = lambda: None
        handler.wfile = io.BytesIO()
        handler.do_GET()
        self.assertIn("script-src 'self'", headers['Content-Security-Policy'])
        self.assertIn("frame-ancestors 'none'", headers['Content-Security-Policy'])
        self.assertNotIn('unsafe-inline', headers['Content-Security-Policy'])
        self.assertEqual(headers['X-Content-Type-Options'], 'nosniff')

class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="workspace-browser-test-")
        self.root = Path(self.temp.name) / "workspace"
        self.root.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def test_paths_links_and_special_files_cannot_escape_or_block(self):
        from workspace import browse
        outside = Path(self.temp.name) / "outside.txt"
        outside.write_text("private")
        (self.root / "link.txt").symlink_to(outside)
        (self.root / "dirlink").symlink_to(outside.parent, target_is_directory=True)
        os.mkfifo(self.root / "pipe.log")
        for path in ("../outside.txt", str(outside), "link.txt", "dirlink/outside.txt", "pipe.log"):
            with self.assertRaises(ValueError, msg=path): browse(self.root, path)
        self.assertEqual(outside.read_text(), "private")

    def test_preview_is_bounded_and_binary_weights_not_read(self):
        from workspace import browse
        (self.root / "large.log").write_text("a" * 70000)
        result = browse(self.root, "large.log")
        self.assertTrue(result["truncated"])
        self.assertEqual(len(result["text"]), 65536)
        (self.root / "model.safetensors").write_bytes(b"weights")
        (self.root / "binary.txt").write_bytes(b"x\x00y")
        self.assertFalse(browse(self.root, "model.safetensors")["preview_available"])
        self.assertFalse(browse(self.root, "binary.txt")["preview_available"])

    def test_directory_pages_filters_and_hidden_entries(self):
        from workspace import browse
        for index in range(205): (self.root / f"item-{index:03}.txt").write_text("")
        (self.root / ".hidden").write_text("")
        first, last = browse(self.root), browse(self.root, offset=200)
        self.assertEqual((len(first["entries"]), len(last["entries"])), (200, 5))
        self.assertTrue(first["more"])
        self.assertFalse(last["more"])
        self.assertEqual(browse(self.root, search="item-204")["entries"][0]["name"], "item-204.txt")
        self.assertEqual(browse(self.root, search="hidden", hidden=True)["entries"][0]["name"], ".hidden")

    def test_recent_metric_review_ignores_malformed_or_foreign_rows(self):
        from workspace import review
        directory = self.root / "workers" / "0"
        directory.mkdir(parents=True)
        (directory / "metrics.jsonl").write_text('x' * 70000 + '\n[]\n{"loss":0.5,"completed":2}\n{"identity":"other","loss":99}\n{"loss":0.2,"score":0.8}\n')
        report = review(self.root, {"identity":"mine", "config":{"workers":1}})
        self.assertEqual(report["workers"][0]["metrics"], {"loss":0.2,"completed":2,"score":0.8})

    def test_general_templates_do_not_promise_automatic_resume(self):
        for key in ("evaluation", "training"):
            cfg = core.normalize_config(server.templates()[key])
            self.assertFalse(cfg["auto_continue"])
            self.assertFalse(cfg["resume_capable"])
        self.assertTrue(server.templates()["simulator"]["auto_continue"])

    def test_review_rejects_linked_or_special_journals_and_oversized_numbers(self):
        from workspace import review
        directory = self.root / 'workers' / '0'
        directory.mkdir(parents=True)
        journal = directory / 'metrics.jsonl'
        outside = Path(self.temp.name) / 'outside.jsonl'
        outside.write_text('{"private_score":42}\n')
        manifest = {'identity':'mine', 'config':{'workers':1}}
        journal.symlink_to(outside)
        self.assertEqual(review(self.root, manifest)['workers'][0]['metrics'], {})
        self.assertIsNotNone(review(self.root, manifest)['workers'][0]['error'])
        journal.unlink()
        os.mkfifo(journal)
        self.assertIsNotNone(review(self.root, manifest)['workers'][0]['error'])
        journal.unlink()
        journal.write_text('{"oversized":' + '9' * 1000 + ',"loss":0.25}\n')
        self.assertEqual(review(self.root, manifest)['workers'][0]['metrics'], {'loss':0.25})

    def test_record_lock_also_serializes_local_server_threads(self):
        from concurrent.futures import ThreadPoolExecutor
        counter = [0]
        def increment(_):
            with sdk.lock(self.root / '.control.lock', blocking=True):
                previous = counter[0]
                time.sleep(0.002)
                counter[0] = previous + 1
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(increment, range(24)))
        self.assertEqual(counter[0], 24)

    def test_remote_requests_share_one_ssh_session_at_a_time(self):
        from concurrent.futures import ThreadPoolExecutor
        from types import SimpleNamespace
        from transport import Client
        profile = server.demo_profile(self.root)
        profile.update(simulation=False, ssh_host='test-cluster')
        client = Client(profile)
        client.ready = True
        active, maximum = [0], [0]
        def remote(*args, **kwargs):
            active[0] += 1
            maximum[0] = max(maximum[0], active[0])
            time.sleep(0.01)
            active[0] -= 1
            return SimpleNamespace(stdout='{"ok":true,"result":{}}', stderr='', returncode=0)
        with patch('transport.subprocess.run', side_effect=remote):
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda _: client.call('browse'), range(8)))
        self.assertEqual(maximum[0], 1)

class RunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cluster-dashboard-test-")
        self.root = Path(self.temp.name)
        self.profile = core.normalize_profile(server.demo_profile(self.root))
        self.cfg = core.normalize_config(server.templates()["simulator"])
        self.submission = patch.object(runtime.Scheduler, "submit", return_value=("1234", ""))
        self.submit = self.submission.start()

    def tearDown(self):
        self.submission.stop()
        self.temp.cleanup()

    def create(self, **changes):
        m = runtime.create_run(self.root, self.profile, {**self.cfg, **changes})
        return runtime.run_path(self.root, m["id"]), m

    def launched(self, **changes):
        path, m = self.create(**changes)
        runtime.dispatch(self.root, self.profile, "launch", {"id": m["id"]})
        return path, runtime.load_run(path)

    def receipt(self, path, m, index=0, state="partial_checkpointed", progress=2):
        wave = m["waves"][-1]
        directory = path / "workers" / str(index)
        sdk.atomic_json(directory / "checkpoint.json", {"identity": m["identity"], "value": {"progress": progress}})
        result = {"identity": m["identity"], "attempt": wave["token"], "worker": index, "state": state, "progress": progress, "completed": len(range(index, m["config"]["task_count"], m["config"]["workers"])) if state == "complete" else 0, "checkpoint_files": {str((directory / "checkpoint.json").relative_to(path)): {"size": (directory / "checkpoint.json").stat().st_size, "sha256": sdk.file_hash(directory / "checkpoint.json")}}}
        sdk.atomic_json(directory / "result.json", result)
        sdk.atomic_json(path / "waves" / str(wave["number"]) / f"receipt-{index}.json", result)
        return result

    def test_threshold_permission_and_unknown(self):
        path, m = self.create()
        discovery = runtime.Scheduler(self.root, self.profile).discover(self.cfg)
        self.assertEqual(runtime.choose_plan(m, discovery, 24)["role"], "development")
        m["profile"]["dev_permission"] = ""
        self.assertEqual(runtime.choose_plan(m, discovery, 24)["role"], "normal")
        m["profile"]["dev_permission"] = "test"
        discovery["partitions"][0]["queue_seconds"] = None
        self.assertEqual(runtime.choose_plan(m, discovery, 24)["role"], "normal")
        discovery["partitions"][0]["queue_seconds"] = 10800
        self.assertEqual(runtime.choose_plan(m, discovery, 24)["role"], "normal")

    def test_launch_is_not_repeatable(self):
        path, m = self.launched()
        with self.assertRaises(ValueError):
            runtime.dispatch(self.root, self.profile, "launch", {"id": m["id"]})
        self.assertEqual(self.submit.call_count, 1)

    def test_tampered_identifiers_and_runtime_paths_fail_closed(self):
        path, m = self.create()
        for changes in ({'id':'<img src=x>'}, {'runtime_files':{'../outside.py':'x'}}, {'waves':[{'job_id':'--user=someone'}]}):
            sdk.atomic_json(path / 'manifest.json', {**m, **changes})
            with self.assertRaises(ValueError): runtime.load_run(path)

    def test_create_request_replay_returns_same_run(self):
        request_id = "a" * 32
        first = runtime.create_run(self.root, self.profile, self.cfg, request_id)
        second = runtime.create_run(self.root, self.profile, self.cfg, request_id)
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(len(list((self.root / "cluster-ui").iterdir())), 1)
        with self.assertRaisesRegex(ValueError, "different experiment"):
            runtime.create_run(self.root, self.profile, {**self.cfg, "recipe": {"changed": True}}, request_id)

    def test_benchmark_permission_does_not_enable_training_dev_jobs(self):
        path, m = self.create(kind="train")
        discovery = runtime.Scheduler(self.root, self.profile).discover(m["config"])
        self.assertEqual(runtime.choose_plan(m, discovery, 24)["role"], "normal")

    def test_partial_and_training_results_require_checkpoint_evidence(self):
        path, m = self.launched()
        result = self.receipt(path, m)
        result["checkpoint_files"] = {}
        with self.assertRaisesRegex(ValueError, "sealed checkpoint"):
            runtime.validate_result(path, m, m["waves"][0], 0, result)
        m["config"]["kind"] = "train"
        result["state"] = "complete"
        with self.assertRaisesRegex(ValueError, "sealed checkpoint"):
            runtime.validate_result(path, m, m["waves"][0], 0, result)

    def test_concurrent_handoffs_submit_only_once(self):
        path, m = self.launched()
        self.receipt(path, m)
        code = '''import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import runtime
def submit(path, manifest, dependency=None):
    with (path / "handoff-count").open("a") as stream: stream.write("submitted\\n")
    return manifest
runtime.new_wave = submit
runtime.handoff(Path(sys.argv[2]), 0)
'''
        processes = [subprocess.Popen([sys.executable, "-c", code, str(Path(runtime.__file__).parent), str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(4)]
        for process in processes:
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, stderr.decode())
        self.assertEqual((path / "handoff-count").read_text().splitlines(), ["submitted"])

    def test_one_successor_and_progress_history(self):
        path, m = self.launched()
        result = self.receipt(path, m)
        result["observed"] = {"seconds_per_task": 0.2}
        sdk.atomic_json(path / "waves" / "0" / "receipt-0.json", result)
        runtime.handoff(path, 0)
        runtime.handoff(path, 0)
        current = runtime.load_run(path)
        self.assertEqual(len(current["waves"]), 2)
        self.assertEqual(self.submit.call_count, 2)
        self.assertEqual(current["waves"][1]["dependency"], "1234")
        self.assertEqual(current["history"][0]["seconds_per_task"], 0.2)

    def test_no_progress_failure_stop_and_completion_do_not_chain(self):
        for state, progress, expected in (("partial_checkpointed", 0, "needs_attention"), ("failed", 2, "failed"), ("cancelled", 2, "cancelled"), ("complete", 2, "complete")):
            with self.subTest(state=state):
                path, m = self.launched()
                self.receipt(path, m, state=state, progress=progress)
                runtime.handoff(path, 0)
                current = runtime.load_run(path)
                self.assertEqual(current["state"], expected)
                self.assertEqual(len(current["waves"]), 1)
        path, m = self.launched()
        self.receipt(path, m)
        (path / "STOP").touch()
        runtime.handoff(path, 0)
        self.assertEqual(runtime.load_run(path)["state"], "cancelled")

    def test_checkpoint_corruption_and_stale_receipts(self):
        path, m = self.launched()
        result = self.receipt(path, m)
        runtime.validate_result(path, m, m["waves"][0], 0, result)
        (path / "workers" / "0" / "checkpoint.json").write_text("broken")
        with self.assertRaises(ValueError): runtime.validate_result(path, m, m["waves"][0], 0, result)
        result["attempt"] = "old"
        with self.assertRaises(ValueError): runtime.validate_result(path, m, m["waves"][0], 0, result)

    def test_missing_receipt_cannot_trigger_handoff(self):
        self.profile["partitions"] = [self.profile["partitions"][0]]
        path, m = self.launched(workers=3, concurrency=2)
        self.receipt(path, m, index=0)
        runtime.handoff(path, 0)
        self.assertEqual(len(runtime.load_run(path)["waves"]), 1)

    def test_false_completion_and_changed_checkpoint_stop_before_next_allocation(self):
        path, m = self.launched()
        result = self.receipt(path, m, state="complete")
        result["completed"] = 1
        with self.assertRaisesRegex(ValueError, "shard inventory"):
            runtime.validate_result(path, m, m["waves"][0], 0, result)
        self.receipt(path, m)
        (path / "workers" / "0" / "checkpoint.json").write_text("corrupt")
        runtime.handoff(path, 0)
        self.assertEqual(runtime.load_run(path)["state"], "failed")
        self.assertEqual(self.submit.call_count, 1)

    def test_run_owner_and_stop_during_planning(self):
        path, m = self.create()
        m["owner_uid"] = os.getuid() + 1
        sdk.atomic_json(path / "manifest.json", m)
        with self.assertRaisesRegex(ValueError, "owner"):
            runtime.dispatch(self.root, self.profile, "stop", {"id": m["id"]})
        m["owner_uid"] = os.getuid()
        sdk.atomic_json(path / "manifest.json", m)
        original = runtime.Scheduler.discover
        def stopping(scheduler, config):
            (path / "STOP").touch()
            return original(scheduler, config)
        with patch.object(runtime.Scheduler, "discover", stopping):
            with self.assertRaisesRegex(ValueError, "stopped during scheduling"):
                runtime.new_wave(path, m)
        self.assertEqual(self.submit.call_count, 0)

    def test_sparse_resume_keeps_original_worker_count(self):
        self.profile["partitions"] = [self.profile["partitions"][0]]
        path, m = self.launched(workers=5, concurrency=2)
        for index in range(5): self.receipt(path, m, index=index, state="complete" if index in (0, 1, 3) else "partial_checkpointed")
        runtime.handoff(path, 0)
        current = runtime.load_run(path)
        self.assertEqual(current["waves"][-1]["indices"], [2, 4])
        self.assertEqual(current["config"]["workers"], 5)
        runtime.dispatch(self.root, self.profile, "concurrency", {"id": m["id"], "concurrency": 3})
        self.assertEqual(sdk.read_json(path / "control.json")["concurrency"], 3)
        self.assertEqual(current["identity"], runtime.load_run(path)["identity"])

    def test_budget_is_optional_and_accounted_conservatively(self):
        path, m = self.launched(budget={"gpu_hours": 0.1})
        self.assertEqual(m["state"], "budget_reached")
        self.assertEqual(len(m["waves"]), 0)
        path, m = self.launched(budget={})
        self.assertEqual(m["state"], "queued")

    def test_uncertain_submission_stops_blind_retry(self):
        self.submit.return_value = (None, "timeout")
        path, m = self.launched()
        self.assertEqual(m["state"], "needs_attention")
        with self.assertRaisesRegex(ValueError, "Uncertain"):
            runtime.dispatch(self.root, self.profile, "resume", {"id": m["id"]})
        self.assertEqual(self.submit.call_count, 1)

    def test_cancel_does_not_spawn_and_log_reads_are_incremental(self):
        path, m = self.launched()
        runtime.dispatch(self.root, self.profile, "stop", {"id": m["id"]})
        self.receipt(path, m)
        runtime.handoff(path, 0)
        self.assertEqual(self.submit.call_count, 1)
        log = path / "waves" / "0" / "worker-0.log"
        log.write_text("first\n")
        first = runtime.dispatch(self.root, self.profile, "tail", {"id": m["id"], "file": "waves/0/worker-0.log"})
        with log.open("a") as handle: handle.write("second\n")
        second = runtime.dispatch(self.root, self.profile, "tail", {"id": m["id"], "file": "waves/0/worker-0.log", "offset": first["offset"]})
        self.assertEqual(second["text"], "second\n")
        with self.assertRaises(ValueError):
            runtime.dispatch(self.root, self.profile, "tail", {"id": m["id"], "file": "../outside.log"})

    def test_changed_runtime_refuses_to_launch(self):
        path, m = self.create()
        with (path / "sdk.py").open("a") as handle: handle.write("\n# accidental edit\n")
        with self.assertRaisesRegex(ValueError, "Runtime source changed"):
            runtime.dispatch(self.root, self.profile, "launch", {"id": m["id"]})

class SchedulerTests(unittest.TestCase):
    def scheduler(self):
        p = server.demo_profile(Path("/tmp"))
        p["simulation"] = False
        p["ssh_host"] = "cluster"
        return runtime.Scheduler("/tmp", p)

    def test_accounting_pending_is_not_terminal(self):
        scheduler = self.scheduler()
        for state, expected in (("PENDING", "UNKNOWN"), ("RUNNING", "UNKNOWN"), ("COMPLETED", "TERMINAL"), ("CANCELLED by 123", "TERMINAL"), ("TIMEOUT", "TERMINAL")):
            with patch.object(runtime, "command", side_effect=[{"code": 0, "stdout": "", "stderr": ""}, {"code": 0, "stdout": f"123|{state}|0:0|30|gres/gpu=1\n", "stderr": ""}]):
                self.assertEqual(scheduler.status("123")["state"], expected)

    def test_submission_timeout_reconciles_exact_name(self):
        scheduler = self.scheduler()
        m = {"id": "a"*32, "config": core.normalize_config({"command": ["echo"]}), "profile": scheduler.profile}
        wave = {"token": "token", "partition": scheduler.profile["partitions"][0], "indices": [0], "concurrency": 1}
        with patch.object(runtime, "command", side_effect=[{"code": -1, "stdout": "", "stderr": "timeout"}, {"code": 0, "stdout": "123|wm-token\n999|other\n", "stderr": ""}, {"code": 0, "stdout": "123|wm-token\n", "stderr": ""}]):
            self.assertEqual(scheduler.submit(m, wave, Path("/tmp/job.sh"))[0], "123")

    def test_limits_and_permission_errors_disable_partition(self):
        scheduler = self.scheduler()
        def responses(argv, **kwargs):
            if argv[0] == "scontrol": return {"code": 0, "stdout": "PartitionName=gpu State=UP MaxTime=00:10:00", "stderr": ""}
            return {"code": 1, "stdout": "", "stderr": "Access denied"}
        with patch.object(runtime, "command", side_effect=responses):
            result = scheduler.discover(core.normalize_config({"command": ["echo"]}))
        self.assertTrue(all(not p["available"] for p in result["partitions"]))
        self.assertTrue(all(p["queue_seconds"] is None for p in result["partitions"]))

    def test_submission_arguments_preserve_sparse_shards_and_paths(self):
        scheduler = self.scheduler()
        cfg = core.normalize_config({"command": ["echo"], "workers": 5, "concurrency": 2})
        args = runtime.sbatch_args({"id":"a"*32,"config":cfg,"profile":scheduler.profile}, scheduler.profile["partitions"][0], [2,4], 2, "token", Path("/tmp/a directory/job.sh"), "123")
        self.assertIn("--array=2,4%2", args)
        self.assertIn("--dependency=afterany:123", args)
        self.assertEqual(args[-1], "/tmp/a directory/job.sh")

class CacheTests(unittest.TestCase):
    def test_reuse_invalidation_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            calls = []
            def prepare(root):
                calls.append(1)
                (root / "weights").write_text("data")
                return ["weights"]
            self.assertFalse(sdk.prepare_cache(path, {"model":"v1"}, prepare))
            self.assertTrue(sdk.prepare_cache(path, {"model":"v1"}, prepare))
            self.assertEqual(len(calls), 1)
            self.assertTrue(sdk.verify_cache(path, {"model":"v1"}))
            self.assertFalse(sdk.cache_ready(path, {"model":"v2"}))
            with self.assertRaises(ValueError): sdk.prepare_cache(path, {"model":"v2"}, prepare)
            (path / "weights").write_text("bad!")
            self.assertFalse(sdk.verify_cache(path, {"model":"v1"}))
            sdk.prepare_cache(path, {"model":"v1"}, prepare)
            self.assertTrue(sdk.verify_cache(path, {"model":"v1"}))

    def test_partial_checkpoint_write_does_not_replace_committed_state(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "checkpoint.json"
            sdk.atomic_json(path, {"step": 2})
            path.with_suffix(".tmp").write_text('{"step":')
            self.assertEqual(sdk.read_json(path), {"step":2})

class TransportTests(unittest.TestCase):
    def test_vpn_drop_does_not_retry_mutation(self):
        p = server.demo_profile(Path("/tmp"))
        p.update(simulation=False, ssh_host="cluster")
        client = transport.Client(p)
        client.ready = True
        with patch.object(transport.subprocess, "run", side_effect=subprocess.TimeoutExpired("ssh", 1)) as run:
            with self.assertRaises(ConnectionError): client.call("launch", id="a"*32)
            self.assertEqual(run.call_count, 1)

    def test_local_index_rebuilds_after_missing_index(self):
        with tempfile.TemporaryDirectory() as d:
            app = server.App(Path(d), demo=True)
            client = app.client("simulator")
            m = client.call("create", config=server.templates()["simulator"])
            self.assertEqual(app.runs("simulator")[0]["id"], m["id"])
            (Path(d) / "index-simulator.json").unlink()
            self.assertEqual(app.runs("simulator", force=True)[0]["identity"], m["identity"])

class IntegrationTests(unittest.TestCase):
    def test_training_resume_matches_uninterrupted_full_state(self):
        with tempfile.TemporaryDirectory(prefix="cluster-training-") as d:
            root = Path(d)
            profile = server.demo_profile(root)
            cfg = core.normalize_config({**server.templates()["simulator"], "kind": "train", "task_count": 12, "command": ["{python}", "{run_dir}/training_example.py"], "recipe": {"seed": 42, "step_seconds": .02}})
            resumed = runtime.create_run(root, profile, cfg)
            baseline = runtime.create_run(root, profile, cfg)
            def invoke(m, duration, attempt):
                path = runtime.run_path(root, m["id"])
                env = {**os.environ, "WM_RUN_DIR": str(path), "WM_WORKER_INDEX": "0", "WM_WORKER_COUNT": "1", "WM_IDENTITY": m["identity"], "WM_ATTEMPT": attempt, "WM_DEADLINE": str(time.time() + duration)}
                result = subprocess.run([sys.executable, str(path / "training_example.py")], env=env, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                return sdk.read_json(path / "workers" / "0" / "result.json")
            partial = invoke(resumed, .14, "first")
            self.assertEqual(partial["state"], "partial_checkpointed")
            self.assertTrue(0 < partial["progress"] < 12)
            self.assertEqual(invoke(resumed, 5, "second")["state"], "complete")
            self.assertEqual(invoke(baseline, 5, "baseline")["state"], "complete")
            a = sdk.read_json(runtime.run_path(root, resumed["id"]) / "workers" / "0" / "checkpoint.json")["value"]
            b = sdk.read_json(runtime.run_path(root, baseline["id"]) / "workers" / "0" / "checkpoint.json")["value"]
            self.assertEqual(a, b)

    def test_detached_simulator_continues_and_preserves_exact_work(self):
        with tempfile.TemporaryDirectory(prefix="cluster-dashboard-integration-") as d:
            root = Path(d)
            profile = server.demo_profile(root)
            cfg = {**server.templates()["simulator"], "task_count": 14, "recipe": {"chunks_per_item": 3, "chunk_seconds": .08}}
            m = runtime.create_run(root, profile, cfg)
            path = runtime.run_path(root, m["id"])
            # Launch from a separate client process that immediately exits.
            result = subprocess.run([sys.executable, str(Path(runtime.__file__)), str(root), "launch"], input=json.dumps({"profile":profile,"data":{"id":m["id"]}}), capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            deadline = time.time() + 20
            while time.time() < deadline:
                m = runtime.load_run(path)
                if m["state"] in runtime.TERMINAL: break
                time.sleep(.1)
            self.assertEqual(m["state"], "complete", json.dumps(m))
            self.assertGreaterEqual(len(m["waves"]), 2)
            state = sdk.read_json(path / "workers" / "0" / "checkpoint.json")["value"]
            self.assertEqual(state["progress"], 42)
            self.assertEqual(state["items"], {str(i):3 for i in range(14)})
            self.assertTrue(all(w["handled"] for w in m["waves"]))
            self.assertTrue(all(w["status"] == "submitted" for w in m["waves"]))

if __name__ == "__main__": unittest.main(verbosity=2)
