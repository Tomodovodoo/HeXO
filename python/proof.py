"""Budgeted continuous double-threat proofs, separate from selective search scores.

solve(game, deadline=perf_counter()+0.1) cooperatively checks an absolute deadline.
Game.turns is a synchronous native proposal call and cannot be interrupted here;
an overrun returns UNKNOWN. Certificates contain no trusted hashes or evaluations.
verify(certificate, history) independently checks the proof against trusted history; it raises ValueError for an
invalid certificate and VerificationTimeout, which says nothing about validity, when its deadline passes.
"""
from itertools import combinations
from math import isfinite
from time import perf_counter

PROVEN_WIN, PROVEN_LOSS, UNKNOWN = "PROVEN_WIN", "PROVEN_LOSS", "UNKNOWN"
AXES = ((1, 0), (0, 1), (1, -1))
LIMIT = 10**12


class _Budget(Exception):
    pass


class VerificationTimeout(TimeoutError):
    """verify() reached its deadline before finishing; the certificate is neither accepted nor rejected."""


def _phase(n):
    return ((1+(n-1)//2) % 2, 2-(n-1) % 2) if n else (0, 1)


def _completions(cells, side, count, check):
    starts, result = set(), set()
    for (q, r), owner in cells.items():
        check()
        if owner == side:
            for dq, dr in AXES:
                for k in range(6):
                    starts.add((q-k*dq, r-k*dr, dq, dr))
    for q, r, dq, dr in starts:
        check()
        empty = []
        for k in range(6):
            p = q+k*dq, r+k*dr
            owner = cells.get(p)
            if owner == 1-side:
                break
            if owner is None:
                empty.append(p)
        else:
            if 1 <= len(empty) <= count and all(max(map(abs, p)) <= LIMIT for p in empty):
                result.add(tuple(sorted(empty)))
    return sorted(result, key=lambda x: (len(x), x))


def _covers(threats, remaining, check):
    result = set()

    def visit(chosen):
        check()
        for threat in threats:
            if not set(threat).intersection(chosen):
                if len(chosen) < remaining:
                    for cell in threat:
                        visit(chosen + (cell,))
                return
        result.add(tuple(sorted(chosen)))

    visit(())
    return sorted(result)


def solve(game, *, deadline, node_limit=10000, attack_turns=3, width=8):
    """Return status, certificate, nodes, elapsed_ms and reason; restore game exactly.

    Only failure of the root player's exact immediate defense proves a loss.
    Quiet and free-filler defender nodes are deliberately left UNKNOWN.
    """
    if not isfinite(deadline) or node_limit < 0 or attack_turns < 0 or not 2 <= width <= 128:
        raise ValueError("Invalid proof budget")
    start = perf_counter()
    nodes = 0
    history = []
    root = game.player
    reason = "incomplete"

    def check():
        if perf_counter() >= deadline:
            raise _Budget("deadline")

    def enter():
        nonlocal nodes
        check()
        if nodes >= node_limit:
            raise _Budget("node_limit")
        nodes += 1

    def position():
        check()
        return {(q, r): p for q, r, p in game.cells}

    def immediate(cells):
        wins = _completions(cells, game.player, game.remaining, check)
        return {"kind": "move", "moves": list(wins[0]), "child": {"kind": "terminal"}} if wins else None

    def descend(moves, callback):
        made = 0
        try:
            for move in moves:
                check()
                game.play(*move)
                made += 1
            return callback()
        finally:
            for _ in range(made):
                game.undo()

    def attack(depth):
        enter()
        if game.winner >= 0:
            return {"kind": "terminal"} if game.winner == root else None
        cells = position()
        win = immediate(cells)
        if win:
            return win
        if depth == 0:
            return None
        proposals = game.turns(width=width)
        check()  # Native generation has no cancellation hook.
        for proposal in proposals:
            check()
            moves = proposal["moves"]
            child = descend(moves, lambda: defend(depth-1))
            if child is not None:
                return {"kind": "move", "moves": moves, "child": child}
        return None

    def defend(depth):
        enter()
        if game.winner >= 0:
            return {"kind": "terminal"} if game.winner == root else None
        cells = position()
        if immediate(cells):
            return None
        threats = _completions(cells, root, 2, check)
        if not threats:
            return None
        covers = _covers(threats, game.remaining, check)
        if not covers:
            return {"kind": "uncovered"}
        if any(len(c) < game.remaining for c in covers):
            return None
        branches = []
        for cover in covers:
            child = descend(cover, lambda: attack(depth))
            if child is None:
                return None
            branches.append({"moves": list(cover), "child": child})
        return {"kind": "defenses", "branches": branches}

    certificate, status = None, UNKNOWN
    try:
        check()
        history = [[q, r] for q, r, _ in game.cells]
        if game.winner >= 0:
            attacker = game.winner
            tree = {"kind": "terminal"}
            status = PROVEN_WIN if attacker == root else PROVEN_LOSS
        else:
            enter()
            cells = position()
            tree = immediate(cells)
            attacker = root
            if tree is not None:
                status = PROVEN_WIN
            else:
                threats = _completions(cells, 1-root, 2, check)
                if threats and not _covers(threats, game.remaining, check):
                    tree, attacker, status = {"kind": "uncovered"}, 1-root, PROVEN_LOSS
                else:
                    tree = attack(attack_turns)
                    if tree is not None:
                        status = PROVEN_WIN
        check()
        if status != UNKNOWN:
            certificate = {"version": 1, "history": history, "attacker": attacker, "tree": tree}
            reason = "proved"
    except _Budget as exc:
        status, certificate, reason = UNKNOWN, None, str(exc)
    return {"status": status, "certificate": certificate, "nodes": nodes,
            "elapsed_ms": (perf_counter()-start)*1000, "reason": reason}


def verify(certificate, history, *, deadline=None, known=()):
    """Independent raw-board checker; return certified root status, raise ValueError for an invalid certificate or
    VerificationTimeout once perf_counter() reaches `deadline`.

    history is supplied by the caller, not accepted solely from the certificate.
    With "flipped": true the root is a flipped-turn position: the history's stones
    with the side NOT to move starting a fresh two-placement turn; the status is
    then relative to that side.
    Reconstructs legal turns and independently enumerates ALL size-1/2 covers.
    Does not call native code or the solver's completion/cover functions.
    """
    def require(condition):
        if deadline is not None and perf_counter() >= deadline:
            raise VerificationTimeout("Certificate verification deadline")
        if not condition:
            raise ValueError("Invalid forcing certificate")

    def phase(n):
        return (0, 1) if n == 0 else ((n+1)//2 % 2, 1 if n % 2 == 0 else 2)

    def play(board, n, winner, raw, abstract=False):
        require(winner == -1 and len(raw) == 2)
        q, r = raw
        require(type(q) is int and type(r) is int and max(abs(q), abs(r)) <= 10**12)
        p = q, r
        require(p not in board)
        require(abstract or (p == (0, 0) if not board else
                any((abs(q-a)+abs(r-b)+abs(q-a+r-b))//2 <= 8 for a, b in board))
                )
        side, _ = phase(n)
        board = dict(board)
        board[p] = side
        winner = -1
        for a, b in ((0, 1), (1, -1), (1, 0)):
            length = 1
            for sign in (-1, 1):
                step = 1
                while board.get((q+sign*step*a, r+sign*step*b)) == side:
                    length += 1
                    step += 1
            if length >= 6:
                winner = side
        return board, n+1, winner

    def threats(board, side, allowance, outside=None):
        # Enumerate complete coordinate segments rather than native window codes.
        segments = set()
        for q, r in board:
            for a, b in ((0, 1), (1, -1), (1, 0)):
                for offset in range(6):
                    segments.add(tuple((q+(j-offset)*a, r+(j-offset)*b) for j in range(6)))
        result = set()
        for segment in segments:
            if any(board.get(p) == 1-side for p in segment):
                continue
            empty = frozenset(p for p in segment if p not in board)
            spare = min(outside[1], len(empty-outside[0])) if outside else 0
            if 0 < len(empty) <= allowance+spare and all(max(map(abs, p)) <= 10**12 for p in empty):
                result.add(empty)
        return result

    def fullturn(state, moves, reused=False, outside=None):
        board, n, winner = state
        side, remaining = phase(n)
        require(0 < len(moves) <= remaining)
        for move in moves:
            if reused and winner == attacker:
                break
            require(outside is None or tuple(move) in outside[0])
            board, n, winner = play(board, n, winner, move)
        require(winner == side or phase(n)[0] != side)
        return board, n, winner

    def walk(node, state, reused=False, outside=None):
        board, n, winner = state
        side, remaining = phase(n)
        kind = node["kind"]
        if reused and winner == attacker:
            return
        if kind == "terminal":
            require(winner == attacker)
            return
        require(winner == -1)
        if kind == 'reuse':
            require(node['winner'] == attacker and (node['player'], node['remaining']) == (side, remaining))
            # Independently replay the strategy on this actual board. The fast
            # native stamp's footprint and threat guards are not trusted here.
            walk(node['child'], state, True, outside)
            return
        if kind == 'zone':
            require(outside is None and side != attacker and not threats(board, side, remaining))
            region = frozenset(tuple(p) for p in node['zone'])
            require(len(region) == len(node['zone']) <= 256 and not region.intersection(board))
            require(all(len(p) == 2 and all(type(x) is int and abs(x) <= 10**6 for x in p) for p in region))
            require(node['fallback']['kind'] == 'reuse')
            # Universally check the actual strategy with up to `remaining`
            # unknown enemy stones outside the supplied region. No native
            # footprint or danger calculation is trusted by this checker.
            attack_ply = 3 if attacker == 0 else 1
            walk(node['fallback'], (board, attack_ply, -1), True, (region, remaining))
            seen = set()
            for branch in node['branches']:
                require(len(branch['moves']) == 1)
                p = tuple(branch['moves'][0])
                require(p in region and p not in seen)
                seen.add(p)
                walk(branch['child'], play(board, n, winner, p, abstract=True), reused)
            require(seen == region)
            return
        if kind == 'exact':
            index = node['fact']
            require(type(index) is int and 0 <= index < len(known))
            fact = known[index]
            prior = {}, 0, -1
            for point in fact['history']:
                prior = play(*prior, point)
            if node.get('after'):
                require(phase(prior[1])[0] != attacker and len(node['after']) == phase(prior[1])[1])
                prior = fullturn(prior, node['after'])
            require(prior[2] == -1 and prior[0] == board and phase(prior[1]) == phase(n))
            require(fact['winner'] == attacker and type(fact['plies']) is int and fact['plies'] > 0)
            return
        if kind == "move":
            require(side == attacker)
            walk(node["child"], fullturn(state, node["moves"], reused, outside), reused, outside)
            return
        require(side != attacker and not threats(board, side, remaining, outside))
        required = threats(board, attacker, 2)
        if outside:
            required = {t for t in required if t <= outside[0]}
        require(bool(required))
        endpoints = sorted(set().union(*required))
        covers = {frozenset(c) for size in range(1, remaining+1)
                  for c in combinations(endpoints, size) if all(set(c) & t for t in required)}
        if kind == "uncovered":
            require(not covers)
            return
        if reused and not covers:
            return
        require(kind in ("defenses", "defenses_all") and bool(covers))
        if kind == "defenses_all":
            full = {c for c in covers if len(c) == remaining}
            for cover in covers:
                if len(cover) == remaining:
                    continue
                require(remaining == 2 and len(cover) == 1)
                blocked = dict(board)
                blocked[next(iter(cover))] = side
                for q, r in blocked:
                    require(True)
                    for dq in range(-8, 9):
                        for dr in range(-8, 9):
                            if (abs(dq)+abs(dr)+abs(dq+dr))//2 > 8:
                                continue
                            filler = q+dq, r+dr
                            if filler not in blocked and max(map(abs, filler)) <= 10**12:
                                full.add(cover | {filler})
                require(len(full) <= 50000)
            covers = full
        else:
            require(all(len(c) == remaining for c in covers))
        seen = set()
        for branch in node["branches"]:
            cover = frozenset(tuple(p) for p in branch["moves"])
            if reused and cover not in covers:
                continue
            require(cover in covers and cover not in seen)
            seen.add(cover)
            walk(branch["child"], fullturn(state, branch["moves"], outside=outside), reused, outside)
        require(seen == covers)

    try:
        require(certificate["version"] == 1)
        require([list(p) for p in history] == certificate["history"])
        attacker = certificate["attacker"]
        require(type(attacker) is int and attacker in (0, 1))
        flipped = certificate.get("flipped", False)
        require(type(flipped) is bool)
        state = {}, 0, -1
        for point in history:
            state = play(*state, point)
        if flipped:
            board, n, winner = state
            state = board, n+1+n % 2, winner
        root = phase(state[1])[0]
        walk(certificate["tree"], state)
        return PROVEN_WIN if root == attacker else PROVEN_LOSS
    except (KeyError, TypeError, IndexError, RecursionError) as exc:
        raise ValueError("Malformed forcing certificate") from exc


if __name__ == "__main__":
    import argparse
    import hashlib
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--benchmark", action="store_true")
    source.add_argument("--history", type=Path, help="JSON list of placement coordinates in play order")
    parser.add_argument("--verify", type=Path, help="Verify a saved proof report against --history")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ms", type=int, default=100)
    parser.add_argument("--nodes", type=int, default=10000)
    parser.add_argument("--attack-turns", type=int, default=3)
    parser.add_argument("--width", type=int, default=8)
    args = parser.parse_args()
    if args.verify and not args.history:
        parser.error("--verify requires --history")
    if args.ms < 1:
        parser.error("--ms must be positive")
    if args.history:
        history = json.loads(args.history.read_text(encoding="utf-8"))
        if args.verify:
            report = json.loads(args.verify.read_text(encoding="utf-8"))
            result = {"status": verify(report["certificate"], history), "verified": True}
        else:
            from hexo import Game
            game = Game(history)
            try:
                result = solve(game, deadline=perf_counter()+args.ms/1000, node_limit=args.nodes,
                               attack_turns=args.attack_turns, width=args.width)
                if result["certificate"]:
                    if verify(result["certificate"], history) != result["status"]:
                        raise ValueError("Proof result disagrees with independent verification")
                    result["verified"] = True
            finally:
                game.close()
        payload = json.dumps(result, indent=2)
        if args.output:
            args.output.write_text(payload, encoding="utf-8")
        print(payload)
        raise SystemExit(0)
    from hexo import Game, library
    histories = {"native_forcing": [(0,0),(0,6),(2,6),(1,0),(2,0),(4,6),(6,6),
                                     (0,2),(1,2),(8,6),(10,6),(2,2),(10,1)]}
    histories.update({f"distant_{n}": [(8*k, 0) for k in range(n)] for n in (101, 301, 1001)})
    rows = []
    for name, history in histories.items():
        game = Game(history)
        try:
            before = game.state(), game.key
            start = perf_counter()
            result = solve(game, deadline=start+.1, width=8)
            wall_ms = (perf_counter()-start)*1000
            assert (game.state(), game.key) == before
            checked = None
            if result["certificate"]:
                checked = verify(result["certificate"], history)
                assert checked == result["status"]
            rows.append({"position": name, "budget_ms": 100, "wall_ms": wall_ms,
                         "status": result["status"], "reason": result["reason"],
                         "nodes": result["nodes"], "verified": checked})
        finally:
            game.close()
    print(json.dumps({"proof_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "native_sha256": hashlib.sha256(Path(library).read_bytes()).hexdigest(),
                      "results": rows}, indent=2))
