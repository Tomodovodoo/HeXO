import unittest
import time
import hashlib
import json
import shutil
import tempfile
import threading
import subprocess
import sys
from unittest.mock import patch
from pathlib import Path

from strix_reference import StrixReference, validate_pv
from tools import strix_corpus


class SequentialReplay(unittest.TestCase):
    def test_first_stone_win_stops_turn(self):
        stones = [[q, 0, "P1"] for q in range(5)]
        self.assertTrue(validate_pv(stones, "P1", 2, [[5, 0]]))
        self.assertFalse(validate_pv(stones, "P1", 2, [[5, 0], [6, 0]]))

    def test_second_stone_and_phase(self):
        stones = [[q, 0, "P1"] for q in range(4)]
        self.assertTrue(validate_pv(stones, "P1", 2, [[4, 0], [5, 0]]))
        self.assertFalse(validate_pv(stones, "P1", 1, [[4, 0], [5, 0]]))

    def test_no_illegal_radius_or_incomplete_pv(self):
        self.assertFalse(validate_pv([[0, 0, "P1"]], "P2", 2, [[9, 0]]))
        self.assertFalse(validate_pv([[0, 0, "P1"]], "P2", 2, [[8, 0], [16, 0]]))


class CorpusProvenance(unittest.TestCase):
    def run_corpus(self, hashes, output):
        def git(command, **kwargs):
            if command[3] == "rev-parse":
                return strix_corpus.REVISION
            if command[3] == "ls-tree":
                return "scripts/fixtures/forcing_puzzles/a.json\nscripts/fixtures/forcing_puzzles/b.json"
            return json.dumps(dict(stones=[[0, 0, "P1"]], attacker="P2", placements_remaining=2)).encode()
        with patch.object(sys, "argv", ["strix_corpus", "unused-source", "--output", str(output)]), \
             patch("tools.strix_corpus.subprocess.check_output", side_effect=git), \
             patch("tools.strix_corpus.StrixReference") as factory:
            factory.return_value.__enter__.return_value.solve.side_effect = [
                dict(status="UNKNOWN", executable_sha256=digest) for digest in hashes]
            strix_corpus.main()

    def test_report_uses_executed_hash_and_rejects_mixed_images(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"report.json"
            self.run_corpus([None, "actual-executed-image"], output)
            self.assertEqual(json.loads(output.read_text())["executable_sha256"], "actual-executed-image")
            with self.assertRaisesRegex(RuntimeError, "different executable"):
                self.run_corpus(["first", "second"], output)


class NativeReference(unittest.TestCase):
    def setUp(self):
        self.reference = StrixReference()
        if not Path(self.reference.executable).exists():
            self.skipTest("build tools/strix first")
        self.addCleanup(self.reference.close)

    def test_persistent_process_phase_and_scoped_negative(self):
        result = self.reference.solve([[q, 0, "P1"] for q in range(4)], "P1", 2)
        self.assertEqual(result["status"], "REFERENCE_WIN_WITHIN_SCOPE")
        self.assertTrue(result["pv_replay_valid"])
        pid = self.reference.process.pid
        result = self.reference.solve([[0, 0, "P1"]], "P2", 2)
        self.assertEqual(self.reference.process.pid, pid)
        self.assertEqual(result["status"], "NO_FORCING_WIN_WITHIN_SCOPE")
        self.assertFalse(result["independently_verified_proof"])
        result = self.reference.solve([[q, 0, "P1"] for q in range(5)], "P1", 1, wide=True)
        self.assertEqual(result["status"], "REFERENCE_WIN_WITHIN_SCOPE")
        self.assertEqual(len(result["pv"]), 1)
        self.assertEqual(result["scope"]["generator"], "wide")

    def test_timeout_unknown_and_process_recovers(self):
        result = self.reference.solve([[q, 0, "P1"] for q in range(3)], "P1", 2,
                                      depth=30, nodes=10**12, timeout_s=1e-12)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual(result["reason"], "wall_timeout")
        self.assertIsNone(self.reference.process)
        result = self.reference.solve([[q, 0, "P1"] for q in range(5)], "P1", 1)
        self.assertEqual(result["status"], "REFERENCE_WIN_WITHIN_SCOPE")

    def test_invalid_snapshots_do_not_reach_solver(self):
        for stones in ([[0, 0, "P1"], [0, 0, "P2"]], [[1 << 31, 0, "P1"]],
                       [[q, 0, "P1"] for q in range(6)]):
            with self.assertRaises(ValueError):
                self.reference.solve(stones, "P1", 2)
        self.assertIsNone(self.reference.process)

    def test_node_and_coordinate_limits_are_unknown(self):
        result = self.reference.solve([[q, 0, "P1"] for q in range(3)], "P1", 2, nodes=0)
        self.assertEqual(result["status"], "UNKNOWN")
        result = self.reference.solve([[0, 0, "P1"], [1000000, 1000000, "P2"]], "P1", 2)
        self.assertEqual(result["status"], "UNKNOWN")

    def test_protocol_object_and_proof_flag(self):
        stones = [[q, 0, "P1"] for q in range(5)]
        reference_result = self.reference.solve(stones, "P1", 1)
        self.assertEqual(reference_result["executable_sha256"], hashlib.sha256(Path(self.reference.executable).read_bytes()).hexdigest())
        with patch.object(self.reference.responses, "get", return_value="[]"):
            result = self.reference.solve(stones, "P1", 1)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertIsNone(self.reference.process)
        self.reference.solve(stones, "P1", 1)
        reference_result["independently_verified_proof"] = True
        with patch.object(self.reference.responses, "get", return_value=json.dumps(reference_result)):
            result = self.reference.solve(stones, "P1", 1)
        self.assertFalse(result["independently_verified_proof"])

    def test_changed_executable_is_not_restarted(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory)/Path(self.reference.executable).name
            shutil.copyfile(self.reference.executable, binary)
            with StrixReference(binary) as reference:
                result = reference.solve([[0, 0, "P1"]], "P2", 2)
                original_hash = result["executable_sha256"]
                reference.close()
                with binary.open("ab") as output:
                    output.write(b"changed")
                result = reference.solve([[0, 0, "P1"]], "P2", 2)
                self.assertEqual(result["status"], "UNKNOWN")
                self.assertIn("changed across", result["reason"])
                self.assertEqual(result["executable_sha256"], original_hash)
                self.assertIsNone(reference.process)

    def test_queue_timeout_leaves_active_query_alive(self):
        entered, release = threading.Event(), threading.Event()
        results = []
        stones = [[q, 0, "P1"] for q in range(5)]
        self.reference.solve(stones, "P1", 1)
        pid = self.reference.process.pid
        def paused_pv(*args):
            entered.set()
            release.wait(1)
            return validate_pv(*args)
        with patch("strix_reference.validate_pv", paused_pv):
            worker = threading.Thread(target=lambda: results.append(self.reference.solve(stones, "P1", 1)))
            worker.start()
            try:
                self.assertTrue(entered.wait(1))
                result = self.reference.solve(stones, "P1", 1, timeout_s=.005)
                self.assertEqual((result["status"], result["reason"]), ("UNKNOWN", "wall_timeout"))
                self.assertEqual(self.reference.process.pid, pid)
                self.assertIsNone(self.reference.process.poll())
            finally:
                release.set()
                worker.join()
        self.assertEqual(results[0]["status"], "REFERENCE_WIN_WITHIN_SCOPE")

    def test_launch_uses_verified_private_image(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory)/Path(self.reference.executable).name
            shutil.copyfile(self.reference.executable, original)
            expected = hashlib.sha256(original.read_bytes()).hexdigest()
            real_popen = subprocess.Popen
            images = []
            def change_source_before_launch(command, **kwargs):
                with original.open("ab") as output:
                    output.write(b"source changed after hash")
                image = Path(command[0])
                self.assertNotEqual(image, original)
                self.assertEqual(hashlib.sha256(image.read_bytes()).hexdigest(), expected)
                images.append(image)
                return real_popen(command, **kwargs)
            with StrixReference(original) as reference:
                with patch("strix_reference.subprocess.Popen", change_source_before_launch):
                    result = reference.solve([[0, 0, "P1"]], "P2", 2)
                self.assertEqual(result["executable_sha256"], expected)
                self.assertTrue(images[0].exists())
                self.assertIsNone(reference.process.poll())
            self.assertFalse(images[0].exists())

    def test_late_transport_and_late_pv_are_unknown(self):
        stones = [[q, 0, "P1"] for q in range(5)]
        self.reference.solve(stones, "P1", 1)
        flush = self.reference.process.stdin.flush
        def delayed_flush():
            flush()
            time.sleep(.03)
        with patch.object(self.reference.process.stdin, "flush", delayed_flush):
            result = self.reference.solve(stones, "P1", 1, timeout_s=.005)
        self.assertEqual((result["status"], result["reason"]), ("UNKNOWN", "wall_timeout"))
        self.assertIsNone(self.reference.process)
        self.reference.solve(stones, "P1", 1)
        def delayed_pv(*args):
            time.sleep(.03)
            return validate_pv(*args)
        with patch("strix_reference.validate_pv", delayed_pv):
            result = self.reference.solve(stones, "P1", 1, timeout_s=.005)
        self.assertEqual((result["status"], result["reason"]), ("UNKNOWN", "wall_timeout"))
        self.assertIsNone(self.reference.process)


if __name__ == "__main__":
    unittest.main()
