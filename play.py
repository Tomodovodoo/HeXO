"""Local browser game. Run python play.py, then open http://127.0.0.1:8765."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from hexo import Game


def promoted_checkpoint(run):
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    return next(c for c in summary["checkpoints"] if c["id"] == summary["incumbent"])


class Handler(BaseHTTPRequestHandler):
    game = Game()
    run = None
    model = None
    label = None
    neural = None

    def state(self):
        promoted = promoted_checkpoint(self.run) if self.run else None
        backend = "native-pvs"
        if self.neural:
            backend = self.neural.mode
        elif self.model or (promoted and promoted.get("kind") == "nnue"):
            backend = "nnue-pvs"
        elif promoted:
            backend = "table-pvs"
        return {**self.game.state(), "opponent": self.label,
                "backend": backend,
                "checkpoint": promoted["id"] if promoted else None,
                "default_budget_ms": 10000 if self.neural else 1000,
                "model_sha256": self.neural.model_sha256 if self.neural else None}

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
            elif self.path == "/bot":
                ms = args.get("ms", 1000)
                if type(ms) is not int or not 1 <= ms <= 30000:
                    raise ValueError("Think time must be 1..30000 ms")
                checkpoint = None
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
        except (TimeoutError, RuntimeError) as error:
            self.respond(503, {"error": str(error)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    opponent = parser.add_mutually_exclusive_group()
    opponent.add_argument("--run", type=Path, help="Play against the latest promoted checkpoint in this run")
    opponent.add_argument("--model", type=Path, help="Play against a specific NNUE export, without claiming promotion")
    opponent.add_argument("--relational", type=Path, help="Play the actual relational policy/Q checkpoint")
    parser.add_argument("--neural-mode", choices=("pi", "mu", "gumbel", "gumbel-proof"), default="gumbel")
    parser.add_argument("--simulations", type=int, default=16, help="Maximum neural search simulations per placement")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--label", help="Visible opponent name")
    args = parser.parse_args()
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
    print(f"HeXO is ready at http://127.0.0.1:{args.port}", flush=True)
    try:
        HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    finally:
        Handler.game.close()
        if Handler.neural:
            Handler.neural.close()
