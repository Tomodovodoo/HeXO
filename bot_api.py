"""Local HTTTX stateless v1-alpha adapter. No external registration or deployment.

Spec revision: hex-tic-tac-toe/htttx-bot-api@37d2385f1016abe8b25798238a7d0c4a17a25dda.
The published schema requires exactly two pieces even for a first-placement win,
which contradicts its prohibition on playing after a win. Return an explicit 409
for unrepresentable responses. Time limits are advisory; capability stays false.
"""
import argparse
import hashlib
import json
import math
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from time import perf_counter

from hexo import Game

MAX_CELLS = 1025
MAX_BODY = 1_048_576


class APIError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def _coord(cell):
    if not isinstance(cell, dict):
        raise APIError("Coordinates must be objects")
    point = cell.get('q'), cell.get('r')
    if any(type(x) is not int or abs(x) > 10**12 for x in point):
        raise APIError("Coordinates must be integers within the engine limit +/- 10^12")
    return point


def board_game(board, *, deadline, node_limit=20000):
    """Reconstruct some legal ordering of supplied cells, never adding any cells.

    Board.cells is unordered in the API. Count/side checks establish turn phase;
    exact backtracking handles cases where a greedy ordering gets stranded.
    Budget exhaustion is 503, not a claim that the board is unreachable.
    """
    if not isinstance(board, dict) or board.get('to_move') not in ('x', 'o'):
        raise APIError("board.to_move must be x or o")
    cells = board.get('cells')
    if not isinstance(cells, list) or len(cells) > MAX_CELLS:
        raise APIError(f"board.cells must be an array of at most {MAX_CELLS} cells")
    owners = {}
    for cell in cells:
        point = _coord(cell)
        if cell.get('p') not in ('x', 'o') or point in owners:
            raise APIError("Cells require x/o pieces and unique coordinates")
        owners[point] = 0 if cell['p'] == 'x' else 1
    if not owners:
        raise APIError("The API's two-piece response cannot represent the one-stone origin turn", 409)
    if owners.get((0, 0)) != 0:
        raise APIError("A reachable board requires cross at the origin")
    n = len(owners)
    if n % 2 == 0:
        raise APIError("The API has no partial-turn field and requires two response pieces", 409)
    counts = [sum(p == side for p in owners.values()) for side in (0, 1)]
    expected = [1+2*((n-1)//4), 2*((n+1)//4)]
    side = (1+(n-1)//2) % 2
    if counts != expected or board['to_move'] != ('x', 'o')[side]:
        raise APIError("Piece counts or to_move disagree with standard turn order")
    for (q, r), owner in owners.items():
        for a, b in ((1, 0), (0, 1), (1, -1)):
            if all(owners.get((q+k*a, r+k*b)) == owner for k in range(6)):
                raise APIError("The supplied board is already terminal", 409)
    game = Game()
    pending = [set(p for p, owner in owners.items() if owner == s) for s in (0, 1)]
    visited = 0
    # Iterative DFS avoids Python recursion limits for the documented cell cap.
    stack = []
    chosen = []
    try:
        while True:
            if perf_counter() >= deadline or visited >= node_limit:
                raise APIError("Legal history reconstruction exceeded its local budget", 503)
            if len(chosen) == n:
                return game
            if len(stack) == len(chosen):
                visited += 1
                legal = sorted(p for p in pending[game.player] if game.legal(*p))
                stack.append(iter(legal))
            point = next(stack[-1], None)
            if point is not None:
                side = game.player
                game.play(*point)
                pending[side].remove(point)
                chosen.append((point, side))
            else:
                stack.pop()
                if not chosen:
                    raise APIError("No legal standard-play ordering exists for the supplied cells")
                point, side = chosen.pop()
                pending[side].add(point)
                game.undo()
    except BaseException:
        game.close()
        raise


class Adapter:
    def __init__(self, *, ms=100, width=16, depth=12, model=None):
        if type(ms) is not int or not 1 <= ms <= 30000 or not 2 <= width <= 128 or depth < 1:
            raise ValueError("Invalid local search settings")
        self.ms, self.width, self.depth = ms, width, depth
        self.model = str(Path(model).resolve()) if model else None
        self.model_sha256 = hashlib.sha256(Path(self.model).read_bytes()).hexdigest() if model else None
        if self.model:
            probe = Game()
            try:
                probe.load_model(self.model)
            finally:
                probe.close()

    def capabilities(self):
        return {'meta': {'name': 'HeXO', 'version': '1',
                         'evaluator': 'nnue' if self.model else 'handwritten',
                         'model_sha256': self.model_sha256,
                         'limitations': '409 for origin, partial turns, terminal boards, and first-placement wins; '
                                        f'local board limit {MAX_CELLS}; reconstruction may return 503'},
                'stateless': {'versions': {'v1-alpha': {'request_id': True, 'move_time_limit': False}}}}

    def turn(self, request):
        if not isinstance(request, dict):
            raise APIError("Request must be an object")
        if 'request_id' in request and (type(request['request_id']) is not int or request['request_id'] < 0):
            raise APIError("request_id must be a nonnegative integer")
        ms = self.ms
        if 'time_limit' in request:
            limit = request['time_limit']
            if type(limit) not in (int, float) or (type(limit) is float and not math.isfinite(limit)) or limit < 0:
                raise APIError("time_limit must be a finite nonnegative number of seconds")
            if limit == 0:
                raise APIError("No processing time was provided", 408)
            ms = max(1, min(ms, int(min(limit, 30)*1000)))
        game = board_game(request.get('board'), deadline=perf_counter()+1)
        try:
            if self.model:
                if hashlib.sha256(Path(self.model).read_bytes()).hexdigest() != self.model_sha256:
                    raise APIError("Configured model changed since adapter startup", 503)
                game.load_model(self.model)
            result = game.search(ms=ms, width=self.width, depth=self.depth)
            moves = result['moves']
            if len(moves) != 2:
                raise APIError("The chosen first-placement win cannot satisfy the API's exactly-two-pieces schema", 409)
            for move in moves:
                game.play(*move)
            response = {'move': {'pieces': [{'q': q, 'r': r} for q, r in moves]}}
            if 'request_id' in request:
                response['request_id'] = request['request_id']
            return response
        finally:
            game.close()


def server(adapter, port=8790, *, read_timeout=2):
    """Create a loopback-only HTTP server; caller owns serve_forever/shutdown."""
    if not math.isfinite(read_timeout) or read_timeout <= 0:
        raise ValueError("Read timeout must be finite and positive")

    class Handler(BaseHTTPRequestHandler):
        timeout = read_timeout

        def respond(self, status, body):
            payload = json.dumps(body, allow_nan=False).encode()
            try:
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except OSError:
                # A disconnected/expired client must not interrupt the server.
                self.close_connection = True

        def read_body(self, length):
            # Absolute body deadline also bounds a client that dribbles bytes.
            deadline, body = perf_counter()+read_timeout, bytearray()
            while len(body) < length:
                remaining = deadline-perf_counter()
                if remaining <= 0:
                    raise TimeoutError("Request body read timed out")
                self.connection.settimeout(remaining)
                chunk = self.rfile.read1(min(65536, length-len(body)))
                if not chunk:
                    raise APIError("Request body ended before Content-Length")
                body.extend(chunk)
            self.connection.settimeout(read_timeout)
            return body

        def do_GET(self):
            if self.path == '/capabilities.json':
                return self.respond(200, adapter.capabilities())
            self.respond(404, {'error': 'Not found'})

        def do_POST(self):
            if self.path != '/stateless/v1-alpha/turn':
                return self.respond(404, {'error': 'Not found'})
            try:
                if self.headers.get('Transfer-Encoding'):
                    raise APIError('Chunked requests are not supported')
                length = int(self.headers.get('Content-Length', '-1'))
                if not 0 <= length <= MAX_BODY:
                    raise APIError('Request exceeds the local body limit', 413)
                if self.headers.get_content_type() != 'application/json':
                    raise APIError('Content-Type must be application/json', 415)
                data = json.loads(self.read_body(length))
                self.respond(200, adapter.turn(data))
            except TimeoutError:
                self.close_connection = True
                self.respond(408, {'error': 'Request body read timed out'})
            except APIError as exc:
                self.respond(exc.status, {'error': str(exc)})
            except (ValueError, UnicodeError, RecursionError) as exc:
                self.respond(400, {'error': str(exc)})

    return HTTPServer(('127.0.0.1', port), Handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8790)
    parser.add_argument('--ms', type=int, default=100)
    parser.add_argument('--width', type=int, default=16)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--model', help='Optional native NNUE .bin evaluator')
    args = parser.parse_args()
    httpd = server(Adapter(ms=args.ms, width=args.width, depth=args.depth, model=args.model), args.port)
    print(f'HeXO stateless API: http://127.0.0.1:{httpd.server_port}/capabilities.json', flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
