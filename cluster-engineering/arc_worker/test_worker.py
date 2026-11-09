"""Offline integration test of the ARC worker, Transformers generate and Slurm checkpoints."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel, LogitsProcessor

if __package__:
    from . import worker
else:
    import worker  # unittest discovery from the arc_worker directory


class TestTokenizer:
    bos_token_id = 1
    pad_token_id = 0
    eos_token_id = 63
    name_or_path = "synthetic"

    def apply_chat_template(self, messages, *, tokenize, return_dict, add_generation_prompt, enable_thinking):
        if not (tokenize and not return_dict and add_generation_prompt and not enable_thinking):
            raise ValueError("Changed ARC template flags")
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise ValueError("Expected one user message")
        task = json.loads(messages[0]["content"].split("\nTask:\n")[1])
        if set(task["test"][0]) != {"input"}:
            raise ValueError("Query outputs leaked into the prompt")
        marker = 10 + task["test"][0]["input"][0][0]
        # Exercise a left-padded batch of different prompt lengths.
        return [1, 2, 3, marker] if marker == 10 else [1, 2, 3, 4, marker]

    def decode(self, tokens, **kwargs):
        fragments = {30: '{"outputs":[[[0]]]}', 31: '{"outputs":',
                     32: '[[[', 33: '1', 34: ']]]}', 63: ''}
        return "".join(fragments[t] for t in tokens)


class ForceAnswer(LogitsProcessor):
    """Give two known JSON answers of different lengths using the real decoding loop."""
    def __call__(self, input_ids, scores):
        scores.fill_(float("-inf"))
        for index, row in enumerate(input_ids.tolist()):
            marker = 10 if 10 in row else 11
            response = [30, 63] if marker == 10 else [31, 32, 33, 34, 63]
            emitted = len(row) - row.index(marker) - 1
            scores[index, response[min(emitted, len(response) - 1)]] = 0.
        return scores


def local_model(call_log):
    torch.manual_seed(42)
    model = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=1, n_embd=16,
                                      vocab_size=64, n_positions=32, bos_token_id=1,
                                      eos_token_id=63, pad_token_id=0)).eval()
    generate = model.generate

    def controlled_generate(**kwargs):
        call_log.append(kwargs["input_ids"].cpu().tolist())
        kwargs["logits_processor"].append(ForceAnswer())
        return generate(**kwargs)

    model.generate = controlled_generate
    return model


def configuration(folder):
    records = [
        {"task_id": name, "train": [{"input": [[0]], "output": [[0]]}],
         "test": [{"input": [[color]]}]}
        for name, color in (("early", 0), ("later", 1))
    ]
    tasks_file = folder / "tasks.jsonl"
    tasks_file.write_text("".join(json.dumps(row) + "\n" for row in records))
    config = {
        "items": ["early", "later"], "workload_revision": "synthetic-v1",
        "worker_command": ["python3", "worker.py"], "partitions": ["gpu"],
        "cpu_partition": "cpu", "gpus": 1, "gpu_hours": 1, "max_minutes": 30,
        "reload_seconds": 0,
        "arc": {"model": "synthetic", "model_revision": "a" * 40,
                "tasks_file": tasks_file.name,
                "tasks_sha256": hashlib.sha256(tasks_file.read_bytes()).hexdigest(),
                "context": 32, "max_new_tokens": 12, "bands": [32], "checkpoint_tokens": 1},
    }
    campaign_path = folder / "campaign.json"
    campaign_path.write_text(json.dumps(config))
    return campaign_path, worker.digest(config)


def allocation_process(folder, mode):
    """Subprocess fixture. A hard exit abandons the batch and its exclusive writer lock."""
    root = Path(folder) / "state"
    campaign = Path(folder) / "campaign.json"
    calls = []

    def stop():
        if mode == "crash":
            early = root / "items/early.json"
            if early.exists() and json.loads(early.read_bytes())["status"] == "finished":
                os._exit(73)
        return False

    worker.run(campaign, root, worker.digest(json.loads(campaign.read_bytes())),
               load_tokenizer=lambda settings: TestTokenizer(),
               load_model=lambda settings: local_model(calls), stop=stop, hardware="cpu-test")
    (root / "inputs.json").write_text(json.dumps(calls))


class InferenceResumeTest(unittest.TestCase):
    def test_resume_after_hard_process_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            configuration(Path(tmp))
            root = Path(tmp) / "state"
            script = ("from arc_worker.test_worker import allocation_process; "
                      "import sys; allocation_process(*sys.argv[1:])")

            def launch(mode):
                return subprocess.run([sys.executable, "-c", script, tmp, mode],
                                      cwd=Path(__file__).resolve().parents[1], text=True,
                                      capture_output=True, timeout=60)

            crash = launch("crash")
            self.assertEqual(crash.returncode, 73, crash.stderr)
            early = root / "items/early.json"
            later = root / "items/later.json"
            saved = early.read_bytes()
            self.assertEqual(json.loads(saved)["status"], "finished")
            partial = json.loads(later.read_bytes())
            self.assertEqual(partial["status"], "partial")
            prefix = partial["payload"]["tokens"]
            self.assertTrue(0 < len(prefix) < 5)
            recovered = launch("resume")
            self.assertEqual(recovered.returncode, 0, recovered.stderr)
            self.assertEqual(json.loads((root / "inputs.json").read_bytes()), [[[1, 2, 3, 4, 11] + prefix]])
            self.assertEqual(early.read_bytes(), saved)
            self.assertEqual(json.loads(later.read_bytes())["payload"]["tokens"], [31, 32, 33, 34, 63])

    def test_interruption_replay_and_finished_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            campaign, identity = configuration(folder)
            root = folder / "state"
            root.mkdir()

            def start(stop):
                calls = []
                worker.run(campaign, root, identity, load_tokenizer=lambda settings: TestTokenizer(),
                           load_model=lambda settings: local_model(calls), stop=stop, hardware="cpu-test")
                return calls

            early = root / "items/early.json"
            later = root / "items/later.json"

            def early_finished():
                return early.exists() and json.loads(early.read_bytes())["status"] == "finished"

            first = start(early_finished)
            self.assertEqual(len(first), 1)
            self.assertEqual(len(first[0]), 2)
            self.assertTrue(early_finished())
            partial = json.loads(later.read_bytes())
            self.assertEqual(partial["status"], "partial")
            prefix = partial["payload"]["tokens"]
            self.assertTrue(0 < len(prefix) < 5)
            saved = early.read_bytes()

            # Construct a new model and tokenizer. The only progress they can recover
            # is the durable state in the shared directory.
            second = start(lambda: False)
            self.assertEqual(second, [[[1, 2, 3, 4, 11] + prefix]])
            self.assertEqual(early.read_bytes(), saved)
            completed = json.loads(later.read_bytes())
            self.assertEqual(completed["status"], "finished")
            self.assertEqual(completed["payload"]["tokens"], [31, 32, 33, 34, 63])
            self.assertEqual(json.loads(completed["payload"]["text"]), {"outputs": [[[1]]]})
            self.assertEqual(json.loads(json.loads(saved)["payload"]["text"]), {"outputs": [[[0]]]})
            # Simulate loss of the scheduler-facing write after a durable token
            # commit. Reconciliation must rebuild it without loading the model.
            later_saved = later.read_bytes()
            later.unlink()
            worker.run(campaign, root, identity, load_tokenizer=lambda settings: TestTokenizer(),
                       load_model=lambda settings: self.fail("Reloaded an already completed campaign"),
                       stop=lambda: False, hardware="cpu-test")
            self.assertEqual(later.read_bytes(), later_saved)
            self.assertEqual(early.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
