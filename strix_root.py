"""Optional reference-selected root turns for controlled arena experiments."""
import time
import hashlib
from pathlib import Path
from strix_reference import StrixReference, REVISION, validate_pv


class StrixRoot:
    def __init__(self, milliseconds, turn_ms, client=None, clock=time.perf_counter,
                 nodes=1000, depth=8, wide=False):
        if not 0 < milliseconds < turn_ms:
            raise ValueError("Strix root budget must be positive and smaller than turn budget")
        if type(nodes) is not int or not 1 <= nodes < 1 << 64 or type(depth) is not int or not 1 <= depth <= 255:
            raise ValueError("Strix nodes must be positive unsigned64 and depth in 1..255")
        self.ms = milliseconds
        self.nodes, self.depth, self.wide = nodes, depth, wide
        self.client = client or StrixReference()
        self.clock = clock
        self.setup = None
        self.counts = dict(calls=0, reference_wins=0, selected=0, unknown=0, negatives=0,
                           invalid_pv=0, skipped=0, overruns=0)

    def close(self):
        self.client.close()

    def warm_up(self):
        start = self.clock()
        result = self.client.solve([[0, 0, "P1"]], "P2", 2, depth=self.depth, nodes=self.nodes,
                                   wide=self.wide, timeout_s=2)
        self.setup = dict(elapsed_ms=(self.clock()-start)*1000, status=result["status"])

    def metadata(self):
        return dict(revision=REVISION, executable_sha256=self.client.executable_sha256,
                    root_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    process_adapter_sha256=hashlib.sha256(Path(__file__).with_name("strix_reference.py").read_bytes()).hexdigest(),
                    depth=self.depth, nodes=self.nodes, generator="wide" if self.wide else "tight", budget_ms=self.ms,
                    independent_proof=False, setup=self.setup, counts=dict(self.counts))

    def search(self, game, turn_ms, start=None, **native_options):
        start = self.clock() if start is None else start
        deadline = start + turn_ms/1000
        cells = [[q, r, "P1" if p == 0 else "P2"] for q, r, p in game.cells]
        attacker = "P1" if game.player == 0 else "P2"
        available = (deadline-self.clock())*1000
        budget = min(self.ms, available-1)
        result = dict(status="UNKNOWN", reason="no_probe_budget")
        if budget > 0:
            self.counts["calls"] += 1
            try:
                result = self.client.solve(cells, attacker, game.remaining, depth=self.depth, nodes=self.nodes,
                                           wide=self.wide, timeout_s=budget/1000)
            except (ValueError, RuntimeError, OSError) as error:
                result = dict(status="UNKNOWN", reason=str(error))
        else:
            self.counts["skipped"] += 1
        status = result["status"]
        selected = None
        if status == "REFERENCE_WIN_WITHIN_SCOPE":
            self.counts["reference_wins"] += 1
            played = 0
            try:
                pv = result.get("pv", [])
                if not validate_pv(cells, attacker, game.remaining, pv):
                    raise ValueError("invalid reference principal variation")
                selected = []
                side = game.player
                for move in pv[:game.remaining]:
                    game.play(*move)
                    played += 1
                    selected.append(tuple(move))
                    if game.winner >= 0:
                        break
                if game.winner < 0 and game.player == side:
                    raise ValueError("reference did not complete current turn")
            except (ValueError, TypeError, IndexError):
                selected = None
                self.counts["invalid_pv"] += 1
            finally:
                for _ in range(played):
                    game.undo()
            if self.clock() >= deadline:
                selected = None
        elif status == "NO_FORCING_WIN_WITHIN_SCOPE":
            self.counts["negatives"] += 1
        elif budget > 0:
            self.counts["unknown"] += 1
        if selected:
            self.counts["selected"] += 1
            search = dict(moves=selected, score=None, nodes=0, depth=0,
                          source="strix_reference", independent_proof=False)
            native_ms = 0
        else:
            # Native requires at least 1ms. Cleanup/scheduling can exhaust the
            # budget; such turns are measured and counted as overruns below.
            native_ms = max(1, int((deadline-self.clock())*1000))
            search = game.search(native_ms, **native_options)
            search["source"] = "native"
        elapsed = (self.clock()-start)*1000
        overrun = elapsed > turn_ms
        self.counts["overruns"] += int(overrun)
        search.update(elapsed_ms=elapsed, strix=dict(status=status, selected=bool(selected),
            probe_budget_ms=max(0, budget), native_budget_ms=native_ms,
            overrun=overrun, scope=result.get("scope"), reason=result.get("reason")))
        return search
