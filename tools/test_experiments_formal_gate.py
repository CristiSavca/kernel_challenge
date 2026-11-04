"""Mock official CLI: a failed proof must prevent any upload or lost receipt."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import experiments as e


class FormalGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "candidate.py"
        self.source.write_text("# frozen test\n")
        self.digest = hashlib.sha256(self.source.read_bytes()).hexdigest()

    def invoke(self, proof, cli):
        argv = ["experiments.py", "check", "muse_moe_w4a16", "--file", str(self.source),
                "--label", "formal-gate-test", "--verify-ptx"]
        with patch.object(e, "ROOT", self.root), patch.object(e, "STORE", self.root / "experiments"), \
             patch("sys.argv", argv), patch.object(e, "prove_frozen_fragment", side_effect=proof), \
             patch.object(e, "cli", side_effect=cli) as remote, contextlib.redirect_stdout(io.StringIO()):
            e.main()
        return remote

    def test_failed_proof_makes_no_remote_call(self):
        calls = []
        def reject(*args):
            raise RuntimeError("counterexample")
        with self.assertRaisesRegex(RuntimeError, "counterexample"):
            self.invoke(reject, lambda *args: calls.append(args))
        self.assertEqual(calls, [])

    def test_proof_receipt_survives_success_and_status_failure(self):
        proof = {"source_sha256": self.digest, "scope": "fragment"}
        for failure in (False, True):
            with self.subTest(failure=failure):
                def remote(*args):
                    if args[0] == "check":
                        frozen = Path(args[2])
                        self.assertEqual(hashlib.sha256(frozen.read_bytes()).hexdigest(), self.digest)
                        return {"job_id": "test-job", "status": "queued"}
                    if failure:
                        raise RuntimeError("status network error")
                    return {"status": "running"}
                if failure:
                    with self.assertRaisesRegex(RuntimeError, "status network error"):
                        self.invoke(lambda *args: proof, remote)
                else:
                    self.invoke(lambda *args: proof, remote)
                receipt = json.loads((self.root / "experiments/runs/test-job.json").read_text())
                self.assertEqual(receipt["formal_proof"], proof)
                self.assertEqual(receipt["source_sha256"], self.digest)

    def test_stale_or_failed_proof_rejected(self):
        for proof in ({"status": "proved_fragment", "source_sha256": "wrong"},
                      {"status": "rejected", "source_sha256": self.digest}):
            with self.subTest(proof=proof), patch("verify_ptx.verify", return_value=proof):
                with self.assertRaisesRegex(RuntimeError, "does not match"):
                    e.prove_frozen_fragment(self.source, self.digest)


if __name__ == "__main__":
    unittest.main()
