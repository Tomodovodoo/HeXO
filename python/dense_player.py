"""The dense model and solver a timed Bubble engine worker plays with (timed_engine)."""
import json
from pathlib import Path
import sys


class DensePlayer:
    """An exported dense checkpoint on a batched inference evaluator, its tactical solver when built, and the
    hybrid search graph `timed_engine.hybrid_turn` keeps across turns."""

    def __init__(self, run, device, tactical_package=None, model=None, net_kernels='fused'):
        self.run, self.device = run, device
        self.model_path = Path(model).resolve() if model else None
        self.tactical_package = tactical_package
        self.net_kernels = net_kernels
        self.evaluator = self.prover = None
        self.checkpoint = None
        try:
            from tactical_proof import NativeTactics
            self.prover = NativeTactics(**({'package': tactical_package} if tactical_package else {}))
        except (OSError, ValueError, KeyError) as error:
            print(f'solver off: {error}', file=sys.stderr)
        self.options = dict(search=True, simulations=128, solver=self.prover is not None)
        available = self.models()
        if not available:
            raise ValueError('No playable dense exports found')
        self.select(available[0]['id'])

    def models(self):
        if self.model_path:
            path = self.model_path
            name = (f'{path.parent.parent.name}/{path.parent.name}'
                    if path.name == 'ema.pt' and path.parent.parent.parent.name == 'checkpoints' else path.stem)
            if not hasattr(self, 'model_digest'):
                from legacy.train import digest
                self.model_digest = digest(path)
            checkpoint = f'{name}@{self.model_digest[:12]}'  # notes and caches are keyed by the weights, not the file name
            return [dict(id=checkpoint, label=checkpoint)]
        exports = sorted((p for p in (self.run/'checkpoints').glob('*/*/ema.pt') if p.parent.name.isdigit()),
                         key=lambda p: (int(p.parent.name), p.parent.parent.name), reverse=True)
        ids = [p.parent.relative_to(self.run/'checkpoints').as_posix() for p in exports]
        if not ids:
            return []
        champion_file = self.run/'champion.json'
        champion = json.loads(champion_file.read_text(encoding='utf-8'))['checkpoint'] if champion_file.exists() else None
        reference = 'main/065000' if 'main/065000' in ids else ids[-1]
        selected = []
        for checkpoint in (champion, ids[0], reference, *ids):
            if checkpoint in ids and checkpoint not in selected:
                selected.append(checkpoint)
            if len(selected) == 4:
                break
        labels = {champion: 'champion', ids[0]: 'newest', reference: 'reference'}
        if champion == ids[0]:
            labels[champion] = 'champion, newest'
        return [dict(id=k, label=f'{k} · {labels.get(k, "earlier export")}') for k in selected]

    def select(self, checkpoint):
        if checkpoint not in {m['id'] for m in self.models()}:
            raise ValueError('Choose one of the available checkpoints')
        if checkpoint == self.checkpoint:
            return
        import hexnet
        from legacy.train import digest
        path = self.model_path or self.run/'checkpoints'/checkpoint/'ema.pt'
        if self.evaluator is not None:
            self.close()
            self.checkpoint = None
        model = hexnet.load_model(path, net_kernels=self.net_kernels)
        self.evaluator = hexnet.DenseEvaluator(model, self.device, digest(path), max_batch=128,
                                               cuda_graphs=self.net_kernels == 'fused')
        self.evaluator.free = []  # Packed forwards own staging until their completion fence.
        if self.evaluator.graph is not None:
            self.evaluator.graph.max_batch = 128
        self.checkpoint, self.model_sha256 = checkpoint, digest(path)
        self.set_history()

    def configure(self, options):
        updated = self.options | options
        if any(type(updated[k]) is not bool for k in ('search', 'solver')):
            raise ValueError('Search and solver must be on or off')
        if updated['solver'] and self.prover is None:
            raise ValueError('The tactical solver is not built; run python tools/build_tactical.py')
        if type(updated['simulations']) is not int or not 1 <= updated['simulations'] <= 4096:
            raise ValueError('simulations must be 1..4096')
        self.options = updated

    def set_history(self, history=()):
        from neural_search import EvaluationCache
        if getattr(self, '_timed_hybrid', None):
            graph, pool, _ = self._timed_hybrid
            pool.close()
            graph.close()
            self._timed_hybrid = None
        self.cache = EvaluationCache(4096)

    def close(self):
        self.set_history()
        if self.evaluator is not None and self.evaluator.graph is not None:
            self.evaluator.graph.close()
        self.evaluator = None
