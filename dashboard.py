"""Read-only local training dashboard for one run directory."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path


class Handler(BaseHTTPRequestHandler):
    run = Path("runs/selfplay")

    def do_GET(self):
        if self.path == "/":
            payload = (Path(__file__).parent / "web/training.html").read_bytes()
            content_type = "text/html; charset=utf-8"
        elif self.path == "/api/run":
            summary = self.run / "summary.json"
            events = self.run / "events.jsonl"
            if not summary.exists():
                data = {"summary": None, "events": []}
            else:
                recent = []
                if events.exists():
                    # Read only the tail so a long run does not grow every response.
                    with events.open("rb") as stream:
                        stream.seek(0, 2)
                        size = stream.tell()
                        stream.seek(max(0, size-256000))
                        if size > 256000:
                            stream.readline()
                        for line in stream.readlines():
                            try:
                                recent.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass  # The trainer may be writing the last line.
                data = {"summary": json.loads(summary.read_text(encoding="utf-8")), "events": recent[-500:]}
            payload = json.dumps(data).encode()
            content_type = "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/selfplay")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    Handler.run = Path(args.run).resolve()
    print(f"Training dashboard: http://127.0.0.1:{args.port}", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
