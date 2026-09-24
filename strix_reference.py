"""Persistent, isolated Strix IDTT reference. No result is a training label."""
import json
import hashlib
import math
from pathlib import Path
import queue
import subprocess
import threading
import time

REVISION = "5a771e572553a8bd8e010112b2ce65f16e5afa1b"
AXES = ((1, 0), (0, 1), (1, -1))


def wins(board, point, player):
    q, r = point
    for dq, dr in AXES:
        count = 1
        for sign in (-1, 1):
            step = 1
            while board.get((q + sign*step*dq, r + sign*step*dr)) == player:
                count += 1
                step += 1
        if count >= 6:
            return True
    return False


def snapshot(stones, attacker, remaining):
    if attacker not in ("P1", "P2") or type(remaining) is not int or remaining not in (1, 2):
        raise ValueError("invalid attacker or turn phase")
    board = {}
    for stone in stones:
        if len(stone) != 3:
            raise ValueError("stones must be [q,r,player]")
        q, r, player = stone
        if type(q) is not int or type(r) is not int or max(abs(q), abs(r)) > 1 << 30:
            raise ValueError("coordinate outside Strix reference range")
        if player not in ("P1", "P2") or (q, r) in board:
            raise ValueError("invalid player or duplicate stone")
        board[q, r] = player
    if any(wins(board, point, player) for point, player in board.items()):
        raise ValueError("snapshot is already terminal")
    return board


def validate_pv(stones, attacker, remaining, pv):
    """Check one sequential line, not all defender branches of a proof."""
    board = snapshot(stones, attacker, remaining)
    player = attacker
    for index, point in enumerate(pv):
        if len(point) != 2 or any(type(x) is not int for x in point):
            return False
        q, r = point
        if (q, r) in board:
            return False
        if board:
            if not any(max(abs(q-a), abs(r-b), abs(q+r-a-b)) <= 8 for a, b in board):
                return False
        elif (q, r) != (0, 0):
            return False
        board[q, r] = player
        if wins(board, (q, r), player):
            return player == attacker and index == len(pv)-1
        remaining -= 1
        if remaining == 0:
            player = "P2" if player == "P1" else "P1"
            remaining = 2
    return False


class StrixReference:
    def __init__(self, executable=None):
        default = Path(__file__).parent / "tools/strix/target/release/hexo-strix-reference"
        if default.with_suffix(".exe").exists():
            default = default.with_suffix(".exe")
        self.executable = str(executable or default)
        self.process = None
        self.reader = None
        self.writer = None
        self.executable_sha256 = None
        self.lock = threading.Lock()

    def _start(self):
        digest = hashlib.sha256(Path(self.executable).read_bytes()).hexdigest()
        if self.executable_sha256 is not None and digest != self.executable_sha256:
            raise RuntimeError("reference executable changed across process restarts")
        self.executable_sha256 = digest
        self.process = subprocess.Popen([self.executable], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.responses = queue.Queue()
        process, responses = self.process, self.responses
        def read():
            try:
                for line in process.stdout:
                    responses.put(line)
            finally:
                responses.put(None)
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait()
            if self.writer is not None:
                self.writer.join()
                self.writer = None
            self.reader.join()
            self.process.stdin.close()
            self.process.stdout.close()
            self.process = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def solve(self, stones, attacker, remaining, *, depth=8, nodes=100000,
              wide=False, timeout_s=1.0):
        stones = [list(stone) for stone in stones]
        if len(stones) > 4096:
            raise ValueError("reference adapter accepts at most 4096 stones")
        snapshot(stones, attacker, remaining)
        if type(depth) is not int or not 1 <= depth <= 255:
            raise ValueError("depth must be in 1..255")
        if type(nodes) is not int or not 0 <= nodes < 1 << 64:
            raise ValueError("nodes must be an unsigned64 integer")
        if type(wide) is not bool or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("invalid generator/timeout")
        request = dict(stones=stones, attacker=attacker, placements_remaining=remaining,
                       depth=depth, nodes=nodes, wide=wide)
        message = json.dumps(request) + "\n"
        if len(message.encode("utf-8")) > 512*1024:
            raise ValueError("reference request exceeds 512 KiB")
        scope = dict(driver="idtt", generator="wide" if wide else "tight", attacker=attacker,
                     placements_remaining=remaining, depth_cap=depth, node_budget=nodes,
                     depth_convention="attacker turns including completing turn",
                     rules=dict(win_length=6, placement_radius=8, match_move_cap=None),
                     domain="fully forcing attacks consuming the defender's whole turn")
        unknown = dict(status="UNKNOWN", revision=REVISION, scope=scope,
                       independently_verified_proof=False, pv=[])
        with self.lock:
            deadline = time.monotonic() + timeout_s
            def remaining_time():
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise queue.Empty
                return remaining
            try:
                if self.process is None:
                    self._start()
                remaining_time()
                process, responses = self.process, self.responses
                def write():
                    try:
                        process.stdin.write(message)
                        process.stdin.flush()
                    except (OSError, ValueError):
                        responses.put(None)
                self.writer = threading.Thread(target=write, daemon=True)
                self.writer.start()
                self.writer.join(timeout=remaining_time())
                remaining_time()
                line = self.responses.get(timeout=remaining_time())
                remaining_time()
                if line is None:
                    raise RuntimeError("reference process exited")
                response = json.loads(line)
                if not isinstance(response, dict):
                    raise RuntimeError("reference response must be an object")
                if response.get("revision") != REVISION or response.get("scope") != scope:
                    raise RuntimeError("reference identity/scope mismatch")
                if response["status"] not in ("UNKNOWN", "REFERENCE_WIN_WITHIN_SCOPE",
                                               "NO_FORCING_WIN_WITHIN_SCOPE"):
                    raise RuntimeError("invalid reference status")
                if response["status"] == "REFERENCE_WIN_WITHIN_SCOPE":
                    pv = response.get("pv", [])
                    valid = pv and response.get("first_move") == pv[0] and validate_pv(stones, attacker, remaining, pv)
                    if not valid:
                        raise RuntimeError("returned principal variation failed sequential replay")
                    response["pv_replay_valid"] = True
                remaining_time()
                response["independently_verified_proof"] = False
                response["executable_sha256"] = self.executable_sha256
                return response
            except queue.Empty:
                self.close()
                return dict(unknown, reason="wall_timeout", executable_sha256=self.executable_sha256)
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                self.close()
                return dict(unknown, reason=str(error), executable_sha256=self.executable_sha256)
