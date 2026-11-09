import tempfile
import unittest
from pathlib import Path
from batching import Journal, choose_group


def tasks():
    return [dict(task_id=f"sample-{i:02d}", prompt_tokens=16+i%2, budget=12) for i in range(36)]


class AdmissionTests(unittest.TestCase):
    def test_backoff_survives_restart(self):
        manifest = tasks()
        with tempfile.TemporaryDirectory() as tmp:
            size = lambda j: choose_group(manifest, j.states(), j.attempts(), context=32, bands=[32])
            with Journal(tmp, manifest, "device-class-A").writer() as j:
                group = size(j)
                self.assertEqual(group["actual_size"], 8)
                for tid in group["task_ids"]:
                    task = next(t for t in manifest if t["task_id"] == tid)
                    j.checkpoint(task, [1], "eos")
                j.finish(j.start(group), "complete", elapsed_seconds=3)
                probe = size(j)
                self.assertEqual(probe["actual_size"], 12)
                j.finish(j.start(probe), "oom", elapsed_seconds=1)
            with Journal(tmp, manifest, "device-class-A").writer() as j:
                self.assertEqual(size(j)["actual_size"], 8)
                self.assertFalse(set(size(j)["task_ids"]) & set(group["task_ids"]))
                with self.assertRaises(ValueError):
                    j.checkpoint(next(t for t in manifest if t["task_id"] in group["task_ids"]), [4], "eos")
            with Journal(tmp, manifest, "device-class-B").writer() as j:
                self.assertEqual(j.attempts(), [])
                self.assertEqual(len(j.states()), 8)
            with self.assertRaises(ValueError):
                with Journal(tmp, "different-manifest", "device-class-A").writer():
                    pass

    def test_context_and_partial_prefix(self):
        t = [dict(task_id="a", prompt_tokens=11, budget=9), dict(task_id="b", prompt_tokens=15, budget=5)]
        with tempfile.TemporaryDirectory() as tmp, Journal(tmp, t, "cpu").writer() as j:
            j.checkpoint(t[0], [1, 2], "checkpoint")
            group = choose_group(t, j.states(), [], context=20, bands=[20])
            self.assertEqual(group["task_ids"], ["a", "b"])
            self.assertEqual(group["call_token_ceiling"], 5)
            j.checkpoint(t[1], [3], "eos")
            j.checkpoint(t[0], [1, 2, 3, 4, 5, 6, 7], "checkpoint")
            self.assertEqual(choose_group(t, j.states(), [], context=20, bands=[20])["call_token_ceiling"], 2)
            state_file = next((Path(tmp)/"states").glob("*.json"))
            state_file.write_text(state_file.read_text().replace('"run_id"', '"altered_id"'))
            with self.assertRaises((KeyError, ValueError)):
                j.states()
