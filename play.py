"""Local browser game. Run python play.py, then open http://127.0.0.1:8765."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from hexo import Game


class Handler(BaseHTTPRequestHandler):
    game = Game()
    run = None

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
            return self.respond(200, self.game.state())
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
            elif self.path == "/play":
                self.game.play(args["q"], args["r"])
            elif self.path == "/undo":
                self.game.undo()
            elif self.path == "/bot":
                ms = args.get("ms", 1000)
                if type(ms) is not int or not 1 <= ms <= 30000:
                    raise ValueError("Think time must be 1..30000 ms")
                checkpoint = None
                if self.run is not None:
                    import numpy as np
                    summary = json.loads((self.run / "summary.json").read_text(encoding="utf-8"))
                    checkpoint = summary["incumbent"]
                    model = next(c for c in summary["checkpoints"] if c["id"] == checkpoint)
                    if model.get("kind") == "nnue":
                        self.game.load_model(self.run / model["nnue"])
                    else:
                        self.game.load_table(np.load(self.run / model["table"], allow_pickle=False))
                analysis = self.game.search(ms)
                analysis["checkpoint"] = checkpoint
                for q, r in analysis["moves"]:
                    self.game.play(q, r)
            else:
                return self.respond(404, {"error": "Not found"})
            self.respond(200, {**self.game.state(), "analysis": analysis})
        except (ValueError, KeyError, TypeError) as error:
            self.respond(400, {"error": str(error)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--run", type=Path, help="Play against the latest promoted checkpoint in this run")
    args = parser.parse_args()
    Handler.run = args.run.resolve() if args.run else None
    print(f"HeXO is ready at http://127.0.0.1:{args.port}", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
