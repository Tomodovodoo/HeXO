import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from human_corpus import examples
from klent import Model, SCHEMA
from nnue_model import collate
from q_warmstart import chosen_features, fit_head, main
from tests.test_human_corpus import record


class QWarmStartTest(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, threads)
        torch.manual_seed(73)

    def test_chosen_features_match_shared_candidate_geometry(self):
        replay = examples(record(), minimum=3, candidates=8, positions=8)
        model = Model()
        batch = collate(replay, np.arange(8), "cpu")
        with torch.no_grad():
            _, all_features = model.nnue.features(batch)
        selected = chosen_features(model.nnue, replay, np.arange(8), "cpu")
        torch.testing.assert_close(selected, all_features[torch.arange(8), batch["chosen"]], rtol=0, atol=0)
        self.assertFalse(selected.requires_grad)
        np.testing.assert_array_equal(replay["outcome"], np.where(replay["player"] == 0, 1, -1))

    def test_validation_labels_do_not_update_head(self):
        left = Model().q
        right = copy.deepcopy(left)
        x = torch.randn(12, 104)
        y = torch.tensor([1., -1.]*6)
        args = SimpleNamespace(device="cpu", lr=.002, seed=7, epochs=1, batch=4)
        fit_head(left, (x, y), (x[:3], y[:3]), args)
        fit_head(right, (x, y), (x[:3], -y[:3]), args)
        for a, b in zip(left.parameters(), right.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_frozen_export_hashed_q_and_no_test_or_excluded_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            corpus = root/"corpus"
            corpus.mkdir()
            replay = examples(record(), minimum=3, candidates=8, positions=8)
            shards = []
            for split, family in (("train", 101), ("validation", 100)):
                (corpus/split).mkdir()
                replay["family"][:] = family
                path = corpus/split/"0000.npz"
                np.savez_compressed(path, **replay)
                shards.append({"path": f"{split}/0000.npz", "positions": 8,
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            for split in ("test", "excluded"):
                shards.append({"path": f"{split}/must-not-load.npz", "sha256": "absent"})
            (corpus/"manifest.json").write_text(json.dumps({"shards": shards}))
            model = Model()
            torch.save(model.nnue.state_dict(), root/"source.pt")
            model.nnue.export(root/"source.nnue")
            original = (root/"source.pt").read_bytes()
            args = SimpleNamespace(corpus=corpus, model=root/"source.pt", output=root/"output", device="cpu",
                epochs=2, batch=4, positions=16, replay_centers=100000, batch_centers=100000,
                feature_batch=4, lr=.002, seed=1729)
            report = main(args)
            self.assertEqual((args.output/"model.pt").read_bytes(), original)
            self.assertEqual((args.output/"model.nnue").read_bytes(), (root/"source.nnue").read_bytes())
            self.assertTrue(report["representation_frozen"])
            self.assertFalse(report["held_out_test_loaded"])
            self.assertFalse(report["excluded_loaded"])
            payload = torch.load(args.output/"q.pt", weights_only=True)
            self.assertEqual(payload["schema"], SCHEMA)
            self.assertEqual(payload["model_sha256"], hashlib.sha256(original).hexdigest())
            self.assertFalse(torch.equal(payload["state"]["2.weight"], model.q[2].weight))
            self.assertEqual(set(payload["initialization"]["shards"]), {"train", "validation"})
            with self.assertRaisesRegex(ValueError, "new directory"):
                main(args)


if __name__ == "__main__":
    unittest.main()
