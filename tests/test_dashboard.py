import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from dashboard import bound_evaluation


class EvaluationBinding(unittest.TestCase):
    def test_latest_checkpoint_identity_and_legacy_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root/'checkpoints/0001/model.nnue'
            model.parent.mkdir(parents=True)
            model.write_bytes(b'first')
            digest = hashlib.sha256(model.read_bytes()).hexdigest()
            folder = root/'evaluation'
            folder.mkdir()
            def write(name, value):
                (folder/name).write_text(json.dumps(value), encoding='utf-8')
            write('status.json', {'wins': 12})
            self.assertIsNone(bound_evaluation(root, 1))
            write('report.json', {'config': {'candidate': str(model)}, 'identity': {str(model): digest}})
            self.assertEqual(bound_evaluation(root, 1)['wins'], 12)
            model.write_bytes(b'changed')
            self.assertIsNone(bound_evaluation(root, 1))
            write('status.json', {'wins': 5, 'candidate_sha256': hashlib.sha256(model.read_bytes()).hexdigest()})
            self.assertEqual(bound_evaluation(root, 1)['wins'], 5)
            self.assertIsNone(bound_evaluation(root, 2))


if __name__ == '__main__':
    unittest.main()
