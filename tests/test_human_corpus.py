import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from human_corpus import digest, examples, prepare, snapshot_key, validate
from corpus_warmstart import verified_shards
from hexo import Game


def record():
    moves = [[0, 0], [0, 3], [1, 3], [1, 0], [2, 0], [3, 3], [4, 3],
             [3, 0], [4, 0], [5, 3], [6, 3], [5, 0]]
    return {"moves": moves, "winner": 1, "source": "human", "game_hash": "0"*16,
            "content_sha256": digest(moves), "family": 101}


class HumanCorpusTests(unittest.TestCase):
    def test_exact_replay_and_reject_false_terminal_labels(self):
        row = record()
        self.assertEqual(validate(row), 0)
        for mutation in ("winner", "unfinished", "postwin", "far", "duplicate", "boolean"):
            bad = copy.deepcopy(row)
            if mutation == "winner": bad["winner"] = -1
            elif mutation == "unfinished": bad["moves"].pop()
            elif mutation == "postwin": bad["moves"].append([5, 1])
            elif mutation == "far": bad["moves"][1] = [99, 99]
            elif mutation == "duplicate": bad["moves"][1] = [0, 0]
            else: bad["moves"][1][0] = True
            with self.assertRaises(ValueError, msg=mutation): validate(bad)
        second = copy.deepcopy(row)
        second["moves"] = [[0, 0], [0, 3], [1, 3], [2, 0], [4, 0], [2, 3], [3, 3],
                           [6, 0], [8, 0], [4, 3], [5, 3]]
        second["winner"] = -1
        self.assertEqual(validate(second), 1)

    def test_position_family_preserves_colors_and_phase(self):
        moves = record()["moves"][:9]
        key = snapshot_key(moves)
        for reflect in (False, True):
            points = [[r, q] if reflect else [q, r] for q, r in moves]
            for _ in range(6):
                self.assertEqual(key, snapshot_key([[q+200, r-50] for q, r in points]))
                points = [[-r, q+r] for q, r in points]
        reordered = copy.deepcopy(moves)
        reordered[1], reordered[2] = reordered[2], reordered[1]
        self.assertEqual(key, snapshot_key(reordered))
        reordered[2], reordered[3] = reordered[3], reordered[2]
        self.assertNotEqual(key, snapshot_key(reordered))
        self.assertNotEqual(key, snapshot_key(moves[:-1]))

    def test_dedup_component_and_benchmark_exclusion(self):
        a = record()
        b = copy.deepcopy(a)
        b["moves"] = [[-r, q+r] for q, r in a["moves"]]
        b["game_hash"] = "1"*16
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory/"hexo_human_corpus.jsonl").write_text("\n".join(json.dumps(r) for r in [a, a, b]))
            excluded = directory/"benchmark.json"
            excluded.write_text(json.dumps({"games": [{"moves": a["moves"]}]}))
            with patch("human_corpus.provenance", return_value={"metadata": {"n_games": 3}}):
                rows, report = prepare(directory, 7, [excluded])
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(report["duplicates"]), 1)
            self.assertEqual(report["family_components"], 1)
            self.assertEqual({r["split"] for r in rows}, {"excluded"})
            self.assertEqual(rows[0]["family"], rows[1]["family"])

    def test_native_replay_rows_both_players_and_conditional_seconds(self):
        row = record()
        data = examples(row, minimum=3, candidates=8, positions=8)
        self.assertEqual(set(data["player"].tolist()), {0, 1})
        self.assertFalse(data["search_valid"].any())
        self.assertTrue(data["policy_valid"].all())
        np.testing.assert_array_equal(data["outcome"], np.where(data["player"] == 0, 1, -1))
        np.testing.assert_array_equal(data["search"], data["outcome"])
        for index, ply in enumerate(data["ply"]):
            game = Game(row["moves"][:ply])
            try:
                lo, hi = data["center_offsets"][index:index+2]
                np.testing.assert_array_equal(data["centers"][lo:hi], game.nnue_centers())
                np.testing.assert_allclose(data["phase"][index], game.nnue_context())
                lo, hi = data["candidate_offsets"][index:index+2]
                chosen = data["candidate_coords"][lo+data["chosen"][index]]
                np.testing.assert_array_equal(chosen, row["moves"][ply])
                for move in data["candidate_coords"][lo:hi]: self.assertTrue(game.legal(*move))
            finally:
                game.close()

    def test_warmstart_hash_checks_and_never_reads_test_shards(self):
        data = examples(record(), minimum=3, candidates=4, positions=4)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shards = []
            for split, family in (("train", 101), ("validation", 100)):
                (root/split).mkdir()
                path = root/split/"0.npz"
                data["family"][:] = family
                np.savez_compressed(path, **data)
                shards.append({"path": f"{split}/0.npz", "positions": 4,
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            shards.append({"path": "test/does-not-exist.npz", "sha256": "not read"})
            (root/"manifest.json").write_text(json.dumps({"shards": shards}))
            paths = verified_shards(root)
            self.assertEqual(set(paths), {"train", "validation"})
            path = paths["train"][0]
            path.write_bytes(path.read_bytes()+b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verified_shards(root)


if __name__ == "__main__":
    unittest.main()
