import hashlib
from pathlib import Path
import tempfile
import unittest
from staging import local_snapshot


class StagingTests(unittest.TestCase):
    def test_hash_and_cleanup(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as scratch:
            file = Path(source) / "weights.bin"
            file.write_bytes(b"example")
            entry = dict(bytes=7, sha256=hashlib.sha256(b"example").hexdigest())
            manifest = {file.name: entry}
            with local_snapshot(source, manifest, scratch) as (snapshot, receipt):
                self.assertEqual((snapshot / file.name).read_bytes(), b"example")
                self.assertEqual(receipt["files"], 1)
            self.assertFalse(snapshot.exists())
            file.write_bytes(b"altered")
            with self.assertRaises(ValueError):
                with local_snapshot(source, manifest, scratch):
                    self.fail("Corrupt snapshot accepted")
            self.assertFalse(list(Path(scratch).iterdir()))
