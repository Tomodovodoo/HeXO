"""Strix opponent: the pinned wrapper running one hexo-safetensors-v1 network through direct relational inference."""
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time

from legacy.strix_reference import REVISION, StrixReference

def mirror(q, r):
    """Strix's axial frame from ours and back: HTTTX (q, r) is Strix (q + r, -r), and the map is its own inverse."""
    return q + r, -r


def validate_turn(game, moves):
    """Validate each stone with HeXO's native rules, then restore the position."""
    if not isinstance(moves, list) or not 1 <= len(moves) <= game.remaining:
        raise ValueError("Strix returned an invalid turn length")
    side, played = game.player, 0
    try:
        for index, move in enumerate(moves):
            if not isinstance(move, list) or len(move) != 2 or any(type(x) is not int for x in move):
                raise ValueError("Strix returned invalid coordinates")
            game.play(*move)
            played += 1
            if game.winner >= 0 and index != len(moves)-1:
                raise ValueError("Strix continued after first-stone win")
        if game.winner < 0 and game.player == side:
            raise ValueError("Strix returned an incomplete turn")
        return [tuple(move) for move in moves]
    finally:
        for _ in range(played):
            game.undo()


class StrixLearned(StrixReference):
    def __init__(self, model_path, simulations=8, actions=4, timeout_ms=5000, seed=0, executable=None):
        if (type(simulations) is not int or not 1 <= simulations <= 100000
                or type(actions) is not int or not 1 <= actions <= 1024
                or type(seed) is not int or not 0 <= seed < 1 << 64 or not 0 < timeout_ms <= 600000):
            raise ValueError("invalid Strix learned search budget")
        binary = Path(__file__).parent/"strix_learned/target/release/hexo-strix-learned"
        if os.name == "nt":
            binary = binary.with_suffix(".exe")
        super().__init__(executable or binary)
        self.model_content = Path(model_path).read_bytes()
        self.model_sha256 = hashlib.sha256(self.model_content).hexdigest()
        header_size = int.from_bytes(self.model_content[:8], "little")
        header = json.loads(self.model_content[8:8+header_size])
        self.source_checkpoint = header["__metadata__"]["source_checkpoint"]
        self.simulations, self.actions, self.timeout_ms, self.seed = simulations, actions, timeout_ms, seed
        self.calls = 0
        self.last_result = None
        self.metadata = dict(revision=REVISION, model_sha256=self.model_sha256,
            model_bytes=len(self.model_content), model_metadata=header["__metadata__"],
            checkpoint_license="hosted with the author's permission", source_license="MIT",
            backend="InferModel.eval_states + gumbel_mcts, native CPU",
            simulations_per_placement=simulations, m_actions=actions, c_visit=50, c_scale=1,
            gumbel_noise=False, timeout_ms=timeout_ms, seed=seed, equal_wall_budget=False,
            root_forcing=dict(enabled=True, phases=[1,2], generator="wide", depth=6, nodes=2000),
            leaf_forcing=False, independent_proof=False,
            rules=dict(win_length=6, placement_radius=8, match_move_cap=None),
            adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        report_path = Path(self.executable).with_name("build-provenance.json")
        self.build_provenance = json.loads(report_path.read_text()) if report_path.exists() else None

    def _exchange(self, request, deadline):
        def remaining():
            value = deadline-time.monotonic()
            if value <= 0:
                raise queue.Empty
            return value
        remaining()
        process, responses = self.process, self.responses
        message = json.dumps(request)+"\n"
        def write():
            try:
                process.stdin.write(message)
                process.stdin.flush()
            except (OSError, ValueError):
                responses.put(None)
        self.writer = threading.Thread(target=write, daemon=True)
        self.writer.start()
        self.writer.join(timeout=remaining())
        remaining()
        line = responses.get(timeout=remaining())
        remaining()
        if line is None:
            raise RuntimeError("Strix learned process exited")
        result = json.loads(line)
        if not isinstance(result, dict) or result.get("revision") != REVISION:
            raise RuntimeError("invalid Strix learned response identity")
        return result

    def _load(self, deadline):
        self._start()  # Verified private executable snapshot; no source-path launch race.
        if self.build_provenance is not None and self.build_provenance.get("executable_sha256") != self.executable_sha256:
            raise RuntimeError("Strix learned executable does not match build provenance")
        model_path = Path(self.snapshot_directory.name)/"model.safetensors"
        model_path.write_bytes(self.model_content)
        result = self._exchange(dict(load=str(model_path)), deadline)
        if result.get("status") != "READY" or result.get("source_checkpoint") != self.source_checkpoint:
            raise RuntimeError(f"Strix model load failed: {result}")
        self.metadata.update(executable_sha256=self.executable_sha256, loaded_metadata=result["metadata"],
                             build_provenance=self.build_provenance)

    def warm_up(self):
        start = time.monotonic()
        try:
            self._load(start+30)
        except Exception:
            self.close()
            raise
        self.metadata["setup_ms"] = (time.monotonic()-start)*1000

    def __call__(self, game, _native_ms):
        start = time.monotonic()
        deadline = start+self.timeout_ms/1000
        cells = game.cells
        if len(cells)>800 or any(max(abs(q),abs(r))>1000000 for q,r,_ in cells):
            raise ValueError("Strix learned adapter supports <=800 stones and coordinates within +/-1000000")
        if not self.lock.acquire(timeout=max(0,deadline-time.monotonic())):
            raise RuntimeError("Strix learned timeout waiting for worker")
        try:
            if time.monotonic() >= deadline:
                raise queue.Empty
            if self.process is None:
                self._load(deadline)
            stones = [[*mirror(q, r), p] for q, r, p in cells]
            result = self._exchange(dict(stones=stones,player=game.player,remaining=game.remaining,
                simulations=self.simulations,actions=self.actions,seed=(self.seed+self.calls) % (1 << 64)),deadline)
            self.calls += 1
            if result.get("status") != "OK":
                raise RuntimeError(f"Strix learned did not select a turn: {result}")
            returned = result.get("moves")
            if not isinstance(returned, list) or any(not isinstance(m, list) or len(m) != 2 or
                                                     any(type(v) is not int for v in m) for m in returned):
                raise ValueError("Strix returned invalid coordinates")
            moves = validate_turn(game, [list(mirror(*m)) for m in returned])
            if time.monotonic() >= deadline:
                raise queue.Empty
            result.update(moves=[list(m) for m in moves], wall_ms=(time.monotonic()-start)*1000,
                          executable_sha256=self.executable_sha256,
                          model_sha256=self.model_sha256)
            self.last_result = result
            return moves
        except (queue.Empty, OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            self.last_result = dict(status="UNKNOWN", reason="wall_timeout" if isinstance(error,queue.Empty) else str(error),
                wall_ms=(time.monotonic()-start)*1000, executable_sha256=self.executable_sha256,model_sha256=self.model_sha256)
            self.close()
            self.last_result["wall_ms"] = (time.monotonic()-start)*1000
            raise RuntimeError(f"Strix learned query failed: {self.last_result['reason']}") from error
        finally:
            self.lock.release()
