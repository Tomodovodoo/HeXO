"""Local browser game. Run python play.py, then open http://127.0.0.1:8765."""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from hexo import Game


class DensePlayer:
    """Play exported dense checkpoints; analysis searches a copy of the browser game."""
    mode = 'dense'

    def __init__(self, run, device, tactical_package=None):
        self.run, self.device = run, device
        self.tactical_package = tactical_package
        self.evaluator = self.prover = None
        self.checkpoint = None
        self.options = dict(search=True, simulations=128, solver=True, solver_nodes=32768)
        self.select('main/065000')

    def models(self):
        labels = {'main/065000': '65k · champion', 'main/075000': '75k',
                  'main/082500': '82.5k', 'main/085000': '85k · newest'}
        return [dict(id=k, label=v) for k, v in labels.items()
                if (self.run/'checkpoints'/k/'ema.pt').is_file()]

    def select(self, checkpoint):
        if checkpoint not in {m['id'] for m in self.models()}:
            raise ValueError('Choose one of the available checkpoints')
        if checkpoint == self.checkpoint:
            return
        import hexnet
        from train import digest
        path = self.run/'checkpoints'/checkpoint/'ema.pt'
        model = hexnet.load_model(path)
        self.evaluator = hexnet.DenseEvaluator(model, self.device, digest(path), max_batch=16)
        self.checkpoint, self.model_sha256 = checkpoint, digest(path)
        self.set_history()

    def configure(self, options):
        updated = self.options | options
        if any(type(updated[k]) is not bool for k in ('search', 'solver')):
            raise ValueError('Search and solver must be on or off')
        for key, maximum in (('simulations', 4096), ('solver_nodes', 1000000)):
            if type(updated[key]) is not int or not 1 <= updated[key] <= maximum:
                raise ValueError(f'{key} must be 1..{maximum}')
        self.options = updated

    def set_history(self, history=()):
        from neural_search import EvaluationCache
        self.cache = EvaluationCache(4096)

    def close(self):
        self.evaluator = None

    def solve(self, history, attacker='mover'):
        from tactical_proof import NativeTactics
        if self.prover is None:
            self.prover = NativeTactics(**({'package': self.tactical_package} if self.tactical_package else {}))
        return self.prover.history(history, attacker=attacker, nodes=self.options['solver_nodes'], ms=10000)

    @staticmethod
    def winning_line(history, result):
        """One legal continuation of a verified strategy, choosing its first covered defender reply."""
        from dense_solver import Proof
        certificate = result.get('certificate') or json.loads(result['certificate_json'])
        proof = Proof(list(map(tuple, history)), certificate)
        local, line = Game(history), []
        try:
            while local.winner < 0:
                current = [cell[:2] for cell in local.cells]
                move = proof.path(current)[1]
                if move is None:
                    break
                actions = move[0] or proof.reply(current) or local.legal_moves()[:local.remaining]
                for action in actions:
                    line.append([*action, local.player])
                    local.play(*action)
                    if local.winner >= 0:
                        break
            return line
        finally:
            local.close()

    def turn(self, game, milliseconds=None, analyze=False):
        import numpy as np
        from neural_search import NeuralSearch
        if game.winner >= 0:
            raise ValueError('This game has finished')
        history = [cell[:2] for cell in game.cells]
        local, moves, suggestions = Game(history), [], []
        player, start, proof, line, threat = local.player, time.perf_counter(), None, [], None
        win_probability = None
        try:
            if self.options['solver']:
                proof = self.solve(history)
                if proof['status'] == 'PROVEN_WIN' and proof.get('native_verified'):
                    moves = proof['moves']
                    line = self.winning_line(history, proof)
                if analyze:
                    danger = self.solve(history, 'opponent')
                    if danger['status'] == 'PROVEN_WIN' and danger.get('native_verified'):
                        threat = dict(moves=danger['moves'], turns=danger['proof_turns'])
            proven = bool(proof and proof['status'] == 'PROVEN_WIN' and proof.get('native_verified'))
            while not proven and local.player == player and local.winner < 0:
                current = [cell[:2] for cell in local.cells]
                if self.options['search']:
                    tree = NeuralSearch(self.evaluator, self.model_sha256, current, seed=1740,
                                        cache=self.cache, tactics=True)
                    try:
                        result = tree.search(self.options['simulations'], root_samples=16, batch_size=16)
                    finally:
                        tree.close()
                    action, policy, actions = result['action'], result['policy'], result['actions']
                    visits = result['visits']
                    value = float(visits @ result['values']/max(1, visits.sum()))
                else:
                    result = self.evaluator.evaluate([current])[0]
                    actions = result['actions']
                    policy = np.exp(result['logits']-result['logits'].max()); policy /= policy.sum()
                    action, value = actions[policy.argmax()].tolist(), float(result['q'][0])
                if not suggestions:
                    suggestions = [dict(move=actions[i].tolist(), probability=float(policy[i]))
                                   for i in np.argsort(-policy)[:5]]
                    win_probability = (value+1)/2
                moves.append(action)
                local.play(*action)
            return dict(moves=moves, backend='dense', checkpoint=self.checkpoint,
                        elapsed_ms=(time.perf_counter()-start)*1000, suggestions=suggestions,
                        player=player, win_probability=1. if proven else win_probability,
                        proof_status='PROVEN_WIN' if proven else 'UNKNOWN', winning_line=line, threat=threat,
                        solver_status=proof['status'] if proof else 'off', settings=dict(self.options))
        finally:
            local.close()


def promoted_checkpoint(run):
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    return next(c for c in summary["checkpoints"] if c["id"] == summary["incumbent"])


class Handler(BaseHTTPRequestHandler):
    game = Game()
    run = None
    model = None
    label = None
    neural = None
    search_run = None
    search_checkpoint = None
    search_label = "Internal champion"
    neural_options = {}

    @classmethod
    def refresh_champion(cls):
        if cls.search_run is None:
            return
        league = json.loads((cls.search_run / "league.json").read_text(encoding="utf-8"))
        number = league["champion"]
        selected = next(c for c in league["checkpoints"] if c["id"] == number)
        if not selected.get("promoted"):
            raise ValueError("Search champion must be a promoted checkpoint")
        if number == cls.search_checkpoint:
            return
        from relational_player import RelationalPlayer
        directory = cls.search_run / "checkpoints" / f"{number:04d}"
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        candidate = RelationalPlayer(directory / "model.pt", **cls.neural_options)
        if candidate.model_sha256 != manifest["files"]["model.pt"]:
            candidate.close()
            raise ValueError("Search champion model digest changed")
        previous = cls.neural
        cls.neural = candidate
        cls.search_checkpoint = number
        cls.label = f"{cls.search_label} {number} | {candidate.mode} | {candidate.model_sha256[:12]} | updates on New game"
        if previous:
            previous.close()

    def state(self):
        promoted = promoted_checkpoint(self.run) if self.run else None
        backend = "native-pvs"
        if self.neural:
            backend = self.neural.mode
        elif self.model or (promoted and promoted.get("kind") == "nnue"):
            backend = "nnue-pvs"
        elif promoted:
            backend = "table-pvs"
        dense = isinstance(self.neural, DensePlayer)
        return {**self.game.state(), "opponent": f'Dense {self.neural.checkpoint}' if dense else self.label,
                "backend": backend,
                "checkpoint": self.neural.checkpoint if dense else self.search_checkpoint if self.search_run else promoted["id"] if promoted else None,
                "default_budget_ms": 10000 if self.neural else 1000,
                "model_sha256": self.neural.model_sha256 if self.neural else None,
                "models": self.neural.models() if dense else [],
                "dense_settings": self.neural.options if dense else None,
                "dense_checkpoint": self.neural.checkpoint if dense else None}

    def respond(self, status, data, content_type="application/json"):
        payload = data.encode() if isinstance(data, str) else json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/":
            return self.respond(200, (Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8"), "text/html; charset=utf-8")
        if self.path == "/state":
            return self.respond(200, self.state())
        self.respond(404, {"error": "Not found"})

    def do_POST(self):
        # Accept only same-origin browser requests to this local service.
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{self.headers.get('Host')}":
            return self.respond(403, {"error": "Origin rejected"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 <= length <= 4096:
                raise ValueError("Request too large")
            args = json.loads(self.rfile.read(length) or "{}")
            analysis = None
            if self.path == "/new":
                self.refresh_champion()
                self.game.close()
                Handler.game = Game()
                if self.neural:
                    self.neural.set_history()
            elif self.path == "/play":
                self.game.play(args["q"], args["r"])
            elif self.path == "/undo":
                self.game.undo()
                if self.neural:
                    self.neural.set_history([cell[:2] for cell in self.game.cells])
            elif self.path == '/settings' and isinstance(self.neural, DensePlayer):
                if 'checkpoint' in args:
                    self.neural.select(args.pop('checkpoint'))
                self.neural.configure(args)
            elif self.path == '/analyze' and isinstance(self.neural, DensePlayer):
                analysis = self.neural.turn(self.game, analyze=True)
            elif self.path == "/bot":
                ms = args.get("ms", 1000)
                if type(ms) is not int or not 1 <= ms <= 30000:
                    raise ValueError("Think time must be 1..30000 ms")
                checkpoint = self.neural.checkpoint if isinstance(self.neural, DensePlayer) else self.search_checkpoint if self.search_run else None
                if self.neural is not None:
                    analysis = self.neural.turn(self.game, milliseconds=ms)
                elif self.model is not None:
                    self.game.load_model(self.model)
                elif self.run is not None:
                    model = promoted_checkpoint(self.run)
                    checkpoint = model["id"]
                    if model.get("kind") == "nnue":
                        self.game.load_model(self.run / model["nnue"])
                    else:
                        import numpy as np
                        self.game.load_table(np.load(self.run / model["table"], allow_pickle=False))
                if self.neural is None:
                    analysis = self.game.search(ms)
                analysis["checkpoint"] = checkpoint
                for q, r in analysis["moves"]:
                    self.game.play(q, r)
            else:
                return self.respond(404, {"error": "Not found"})
            self.respond(200, {**self.state(), "analysis": analysis})
        except (ValueError, KeyError, TypeError) as error:
            self.respond(400, {"error": str(error)})
        except (TimeoutError, RuntimeError, OSError, StopIteration) as error:
            self.respond(503, {"error": str(error)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    opponent = parser.add_mutually_exclusive_group()
    opponent.add_argument("--run", type=Path, help="Play against the latest promoted checkpoint in this run")
    opponent.add_argument("--model", type=Path, help="Play against a specific NNUE export, without claiming promotion")
    opponent.add_argument("--relational", type=Path, help="Play the actual relational policy/Q checkpoint")
    opponent.add_argument("--search-run", type=Path, help="Use the internal search champion; refresh on New game")
    opponent.add_argument('--dense-run', type=Path, help='Play dense exports with model, search and solver controls')
    parser.add_argument('--tactical-package', type=Path, help='Directory containing the verified prebuilt tactical library')
    parser.add_argument("--neural-mode", choices=("pi", "mu", "gumbel", "gumbel-proof"), default="gumbel")
    parser.add_argument("--simulations", type=int, default=16, help="Maximum neural search simulations per placement")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--label", help="Visible opponent name")
    args = parser.parse_args()
    if args.dense_run:
        import torch
        torch.set_num_threads(2)
        Handler.neural = DensePlayer(args.dense_run.resolve(), args.device, args.tactical_package)
    Handler.run = args.run.resolve() if args.run else None
    if Handler.run:
        promoted_checkpoint(Handler.run)
    Handler.model = args.model.resolve() if args.model else None
    if Handler.model:
        Handler.game.load_model(Handler.model)
    if args.relational:
        from relational_player import RelationalPlayer
        Handler.neural = RelationalPlayer(args.relational.resolve(), mode=args.neural_mode,
                                          simulations=args.simulations, device=args.device)
    Handler.label = args.label or (str(Handler.model) if Handler.model else
                                  f"Promoted checkpoint from {Handler.run.name}" if Handler.run else "Native engine")
    if Handler.neural:
        name = args.label or "Experimental relational policy/Q"
        Handler.label = f"{name} | {Handler.neural.mode} | {Handler.neural.model_sha256[:12]}"
    Handler.search_run = args.search_run.resolve() if args.search_run else None
    Handler.search_label = args.label or "Internal champion"
    Handler.neural_options = dict(mode=args.neural_mode, simulations=args.simulations, device=args.device)
    Handler.refresh_champion()
    print(f"HeXO is ready at http://127.0.0.1:{args.port}", flush=True)
    try:
        HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    finally:
        Handler.game.close()
        if Handler.neural:
            Handler.neural.close()
