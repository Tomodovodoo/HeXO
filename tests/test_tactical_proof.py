import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import unittest
from proof import VerificationTimeout
from tactical_proof import IsolatedTactics, NativeTactics, independent_verify, threat_cells


OPEN_THREE = [[0,0],[0,8],[2,8],[1,0],[2,0],[4,8],[6,8]]
IMMEDIATE = [[0,0],[0,3],[1,3],[1,0],[2,0],[2,3],[3,3],[3,0],[4,0],[4,3],[5,4]]
# Side 0 to move; side 1 holds an open four (one-turn threat), or two open threes on
# separate lines (a two-turn threat), or no two stones on a common line (no threat).
ONE_TURN = [[0,0],[1,2],[2,2],[0,-2],[-2,0],[3,2],[4,2]]
TWO_TURN = [[0,0],[1,2],[2,2],[0,-3],[-3,0],[3,2],[8,0],[-3,3],[3,-3],[8,1],[8,2]]
NO_THREAT = [[0,0],[1,2],[3,-1]]
# Side 0 to move with a forced win in four turns; the first certificate PDS-PN finds takes five.
LATE_WIN = [[0,0],[1,-2],[-1,-1],[2,-1],[0,-2],[0,-3],[1,-4],[1,-3],[2,-5],[-4,0],[-1,0],[-3,0],[-1,1],[-4,-1],
            [-4,-2],[-3,-1],[-4,-3],[2,-2],[-3,1],[-2,3],[-4,3],[-3,3],[-6,1],[-2,4],[-5,0],[-2,5],[-7,1],[-2,1],
            [-8,1],[-5,-1],[-6,0],[-9,3],[-3,-3],[-6,-1],[-6,-2],[-6,3],[-6,-3],[-7,-2],[-7,-1],[-9,-1],[-5,-3],
            [-7,-3],[-1,-3],[-7,0],[-5,-2],[-3,4],[-8,3],[-3,5],[-1,-5],[0,2],[-3,-4],[-7,6],[0,-8],[-6,5],[1,-8],
            [-2,-7],[-10,5],[-9,5],[-2,-6],[-1,-9],[-11,4],[-12,5],[0,-7],[2,-9],[-10,2],[3,-4],[4,-5],[2,-3],
            [-2,-1],[-2,0],[-6,4],[-1,-8],[-4,2],[0,-4],[1,-9],[0,-9],[1,-10],[-3,-6],[3,-12]]
FIXTURE = json.loads((Path(__file__).with_name('fixtures')/'tactical_positions.json').read_text(encoding='utf-8'))
DETERMINISM_NODES = 540


class NativeStrategy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.engine = NativeTactics()
        except FileNotFoundError:
            raise unittest.SkipTest('Build tools/tactical with tools/build_tactical.py first')

    def test_open_three_wide_builder_full_strategy(self):
        result = self.engine.history(OPEN_THREE, nodes=100000, ms=5000, idtt_nodes=1000)
        self.assertEqual(result['status'], 'PROVEN_WIN', result)
        self.assertTrue(result['native_verified'])
        self.assertIsNotNone(result['idtt_verdict'])
        self.assertGreater(len(result['certificate']['nodes']), 1)
        self.assertEqual(independent_verify(result['certificate'], OPEN_THREE), 'PROVEN_WIN')
        cached = self.engine.history(OPEN_THREE, nodes=100000, ms=5000, idtt_nodes=1000)
        self.assertTrue(cached['cache_hit'])
        for field in ('status', 'certificate', 'nodes_used', 'idtt_verdict', 'proof_turns'):
            self.assertEqual(cached[field], result[field], field)
        shallow = self.engine.history(OPEN_THREE, nodes=100000, ms=5000, idtt_nodes=1000, depth=2)
        self.assertFalse(shallow['cache_hit'])
        for mutation in ['missing', 'duplicate', 'cycle', 'coordinate']:
            cert = copy.deepcopy(result['certificate'])
            if mutation in ('missing', 'duplicate'):
                node = next(n for n in cert['nodes'] if n['kind'] == 'defender_replies')
                if mutation == 'missing': node['responses'].pop()
                else: node['responses'].append(copy.deepcopy(node['responses'][0]))
            elif mutation == 'cycle':
                cert['nodes'][cert['root']]['child'] = cert['root']
            else:
                cert['nodes'][cert['root']]['action'][0] = [999999,999999]
            rejected = self.engine.history(OPEN_THREE, ms=2000, certificate=cert)
            self.assertEqual(rejected['status'], 'UNKNOWN', mutation)
            self.assertFalse(rejected['native_verified'])
            with self.assertRaises(ValueError): independent_verify(cert, OPEN_THREE)
            if mutation == 'cycle':
                with self.assertRaises(VerificationTimeout):
                    independent_verify(cert, OPEN_THREE, deadline_seconds=0)
        with self.assertRaises(VerificationTimeout):
            independent_verify(result['certificate'], OPEN_THREE, deadline_seconds=0)

    def test_shortest_tightens_the_certificate_to_the_fewest_turns(self):
        loose = self.engine.history(LATE_WIN, nodes=32768, ms=20000)
        self.assertEqual((loose['status'], loose['proof_turns'], loose['shortest']), ('PROVEN_WIN', 5, False))
        tight = self.engine.history(LATE_WIN, nodes=32768, ms=20000, shortest=True)
        self.assertEqual((tight['status'], tight['proof_turns'], tight['shortest'], tight['moves']),
                         ('PROVEN_WIN', 4, True, [[-1, -11], [-1, -10]]))
        self.assertEqual(independent_verify(tight['certificate'], LATE_WIN), 'PROVEN_WIN')
        again = self.engine.history(LATE_WIN, nodes=32768, ms=20000, shortest=True)
        self.assertEqual((again['cache_hit'], again['shortest'], again['certificate']), (True, True, tight['certificate']))

    def test_certificate_cap_tracks_granted_budget(self):
        cert = dict(version=1, width='wide', root=0,
                    nodes=[dict(kind='immediate_win', action=[[5, 0]])]*50001)
        small = self.engine.history(IMMEDIATE, nodes=6250, ms=10000, certificate=cert)
        self.assertEqual((small['status'], small['reason']), ('UNKNOWN', 'certificate format/size'))
        large = self.engine.history(IMMEDIATE, nodes=8192, ms=10000, certificate=cert)
        self.assertEqual(large['status'], 'PROVEN_WIN', large['reason'])
        self.assertEqual(large['scope']['budget']['check_nodes'], 65536)
        self.assertEqual(independent_verify(large['certificate'], IMMEDIATE), 'PROVEN_WIN')

    def test_partial_phase_and_first_placement_terminal(self):
        cert = dict(version=1, width='wide', root=0, nodes=[dict(kind='immediate_win', action=[[5,0]])])
        for history in [IMMEDIATE, IMMEDIATE+[[8,0]]]:
            result = self.engine.history(history, ms=1000, certificate=cert)
            self.assertEqual(result['status'], 'PROVEN_WIN', result)
            self.assertEqual(independent_verify(cert, history), 'PROVEN_WIN')
        cert['nodes'][0]['action'].append([6,0])
        self.assertEqual(self.engine.history(IMMEDIATE, ms=1000, certificate=cert)['status'], 'UNKNOWN')
        # Removing one history placement changes whose phase is being proved.
        cert['nodes'][0]['action'] = [[5,0]]
        self.assertEqual(self.engine.history(IMMEDIATE[:-1], ms=1000, certificate=cert)['status'], 'UNKNOWN')

    def test_free_second_missing_filler_and_counterwin_rejected(self):
        # A single covered obligation cannot be represented by one arbitrary filler.
        history = [[0,0],[-1,0],[0,8],[1,0],[2,0],[2,8],[4,8]]
        cert = dict(version=1, width='wide', root=0, nodes=[
            dict(kind='attacker_move', action=[[3,0],[5,0]], child=1),
            dict(kind='defender_replies', responses=[dict(action=[[4,0],[6,8]],child=2)]),
            dict(kind='immediate_win', action=[[0,1]])])
        r = self.engine.history(history, ms=1000, certificate=cert)
        self.assertEqual(r['status'], 'UNKNOWN')
        self.assertIn('free-second', r['reason'])
        # Defender already has a completion after an unrelated attacker move.
        history = [[0,0],[0,3],[1,3],[2,0],[4,0],[2,3],[3,3],[6,0],[8,0],[4,3],[6,4]]
        cert['nodes'][0]['action'] = [[10,0],[12,0]]
        cert['nodes'][1] = dict(kind='unstoppable', threats=[])
        r = self.engine.history(history, ms=1000, certificate=cert)
        self.assertEqual(r['status'], 'UNKNOWN')
        self.assertIn('counterwin', r['reason'])

    def test_free_second_every_legal_filler_has_verified_continuation(self):
        from tests.reference import interleave, Reference
        ours = [(q,r) for r in (0,3,6,9) for q in range(3)] + [(12,0)]
        theirs = [(-1,0)] + [(6+(i%3)*3,2+3*(i//3)) for i in range(13)]
        history = [list(p) for p in interleave([ours,theirs])]
        result = self.engine.history(history, ms=20000, nodes=1000000, root_moves=[[3,0],[5,0]])
        self.assertEqual(result['status'], 'PROVEN_WIN', result.get('reason'))
        cert = result['certificate']
        self.assertEqual(independent_verify(cert, history), 'PROVEN_WIN')
        reference = Reference()
        for p in history+[[3,0],[5,0],[4,0]]: reference.play(*p)
        frontier = {(q+dq,r+dr) for q,r in reference.cells for dq in range(-8,9) for dr in range(-8,9)
                    if max(abs(dq),abs(dr),abs(dq+dr))<=8 and (q+dq,r+dr) not in reference.cells}
        responses = cert['nodes'][cert['nodes'][cert['root']]['child']]['responses']
        actual = {frozenset(map(tuple,response['action'])) for response in responses}
        self.assertEqual(actual, {frozenset(((4,0),filler)) for filler in frontier})
        self.assertEqual(len(actual), 745)
        bad = copy.deepcopy(cert)
        bad['nodes'][bad['nodes'][bad['root']]['child']]['responses'].pop()
        rejected = self.engine.history(history, ms=3000, certificate=bad)
        self.assertEqual(rejected['status'], 'UNKNOWN')
        self.assertFalse(rejected['native_verified'])
        with self.assertRaises(ValueError): independent_verify(bad, history)

    def test_native_search_stops_at_its_deadline(self):
        from tests.reference import interleave
        ours = [(q,r) for r in (0,3,6,9) for q in range(3)] + [(12,0)]
        theirs = [(-1,0)] + [(6+(i%3)*3,2+3*(i//3)) for i in range(13)]
        history = [list(p) for p in interleave([ours,theirs])]
        # The complete candidate proof takes seconds; a 300 ms query must stop the native worker too.
        result = self.engine.history(history, ms=300, nodes=1000000, root_moves=[[3,0],[5,0]])
        self.assertEqual(result['status'], 'UNKNOWN')
        time.sleep(0.2)
        after = self.engine.history([[0,0]], ms=1000)
        self.assertNotIn('busy', after['reason'])
        self.assertLess(after['last_worker_completion']['last_after_deadline']['elapsed_ms'], 450)

    def test_deadline_and_unknown_are_not_loss(self):
        start = time.perf_counter()
        result = self.engine.history([[0,0]], ms=1, nodes=1, attacker='opponent')
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertEqual((result['attacker'], result['build_hash']), ('opponent', self.engine.metadata['binary_sha256']))
        self.assertFalse(result['native_verified'])
        self.assertLess(time.perf_counter()-start, 0.25)
        # Let the sole native worker finish before another test needs it.
        time.sleep(0.05)


class NodeBudget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.engine = NativeTactics()
        except FileNotFoundError:
            raise unittest.SkipTest('Build tools/tactical with tools/build_tactical.py first')

    def test_budget_bounds_total_work(self):
        control = FIXTURE['positions'][FIXTURE['control']]
        for nodes in (1, 50, 2000):
            result = self.engine.history(control, nodes=nodes, ms=60000)
            self.assertEqual((result['status'], result['nodes_used']), ('UNKNOWN', nodes))
        split = self.engine.history(control, nodes=2000, idtt_nodes=500, ms=60000)
        self.assertEqual((split['idtt_verdict'], split['nodes_used']), ('BudgetExceeded', 2000))

    def test_same_budget_same_result_in_fresh_processes(self):
        """Verdict, certificate hash and work agree across fresh processes for the control and 20 shard positions."""
        runs = [subprocess.run([sys.executable, '-m', 'tests.test_tactical_proof', 'determinism'], capture_output=True,
                               text=True, check=True, cwd=Path(__file__).parents[1]).stdout for _ in range(2)]
        self.assertEqual(runs[0], runs[1])
        rows = [json.loads(line) for line in runs[0].splitlines()]
        self.assertEqual(len(rows), 1+2*len(FIXTURE['random']))
        self.assertEqual(rows[0][2], 'PROVEN_WIN')
        self.assertTrue(any(row[2] == 'PROVEN_WIN' for row in rows[1:]))
        self.assertTrue(any(row[4] == DETERMINISM_NODES for row in rows[1:]))
        self.assertTrue(all(row[4] <= DETERMINISM_NODES for row in rows[1:]))

    def test_earlier_hits_prove_at_their_budgets(self):
        for key, nodes in FIXTURE['proving_nodes'].items():
            history = FIXTURE['positions'][key]
            result = self.engine.history(history, nodes=nodes, ms=60000)
            self.assertEqual(result['status'], 'PROVEN_WIN', key)
            self.assertTrue(result['native_verified'])
            self.assertLessEqual(result['nodes_used'], nodes)
            self.assertGreaterEqual(result['proof_turns'], 1)
            self.assertEqual(result['build_hash'], self.engine.metadata['binary_sha256'])
            self.assertEqual(independent_verify(result['certificate'], history), 'PROVEN_WIN', key)


class FlippedTurnThreats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.engine = NativeTactics()
        except FileNotFoundError:
            raise unittest.SkipTest('Build tools/tactical with tools/build_tactical.py first')

    def test_opponent_threats_are_proved_and_checked_by_both_checkers(self):
        for history, turns, line in [(ONE_TURN, 1, {(q, 2) for q in range(-1, 7)}),
                                     (TWO_TURN, 2, {(q, 2) for q in range(-1, 7)} | {(8, r) for r in range(-3, 6)}),
                                     (ONE_TURN+[[-5,5]], 1, {(q, 2) for q in range(-1, 7)})]:
            result = self.engine.history(history, nodes=2000, attacker='opponent')
            self.assertEqual((result['status'], result['proof_turns'], result['attacker']), ('PROVEN_WIN', turns, 'opponent'))
            cells = threat_cells(result['certificate'])
            self.assertEqual(cells, [tuple(cell) for cell in result['moves']])
            self.assertLessEqual(set(cells), line)
            self.assertEqual(independent_verify(result['certificate'], history, attacker='opponent'), 'PROVEN_WIN')
            with self.assertRaises(ValueError):
                independent_verify(result['certificate'], history)
            recheck = self.engine.history(history, attacker='opponent', certificate=result['certificate'])
            self.assertEqual(recheck['status'], 'PROVEN_WIN')
            self.assertEqual(self.engine.history(history, certificate=result['certificate'])['status'], 'UNKNOWN')

    def test_isolated_worker_passes_shortest(self):
        tactics = IsolatedTactics()
        try:
            result = tactics.history(LATE_WIN, nodes=32768, ms=20000, shortest=True)
            self.assertEqual((result['status'], result['proof_turns'], result['shortest']), ('PROVEN_WIN', 4, True))
        finally:
            tactics.close()

    def test_no_threat(self):
        result = self.engine.history(NO_THREAT, nodes=100000, attacker='opponent')
        self.assertEqual((result['status'], result['proof_turns']), ('UNKNOWN', None))

    def test_isolated_worker_carries_budget_fields(self):
        tactics = IsolatedTactics()
        try:
            result = tactics.history(TWO_TURN, nodes=2000, attacker='opponent')
            self.assertEqual((result['status'], result['proof_turns']), ('PROVEN_WIN', 2))
            self.assertEqual(result['build_hash'], self.engine.metadata['binary_sha256'])
            self.assertLessEqual(result['nodes_used'], 2000)
            certificate = json.loads(result['certificate_json'])
            self.assertEqual(independent_verify(certificate, TWO_TURN, attacker='opponent'), 'PROVEN_WIN')
        finally:
            tactics.close()


class Gate(unittest.TestCase):
    GATE = dict(weight=3., floor=32, cap_low=512, cap_high=8192)

    @classmethod
    def setUpClass(cls):
        try:
            cls.engine = NativeTactics()
        except FileNotFoundError:
            raise unittest.SkipTest('Build tools/tactical with tools/build_tactical.py first')

    def test_budget_follows_the_attackers_forcing_material(self):
        strong = FIXTURE['positions']['1790600149713752:2:253']   # forcing material 19.5: gate level 1
        result = self.engine.history(strong, nodes=135, gate=self.GATE)
        self.assertEqual((result['status'], result['budget'], result['gate_score']), ('PROVEN_WIN', 540, 19.5))
        self.assertLessEqual(result['nodes_used'], 540)
        self.assertEqual(self.engine.history(strong, nodes=5000, gate=self.GATE)['budget'], 8192)
        quiet = self.engine.history(NO_THREAT, nodes=135, gate=self.GATE)
        self.assertEqual((quiet['budget'], quiet['gate_score']), (32, 0.))
        plain = self.engine.history(strong, nodes=135)
        self.assertEqual((plain['budget'], plain['gate_score']), (135, None))
        # The opponent's material decides a flipped-turn query.
        flipped = self.engine.history(TWO_TURN, nodes=135, attacker='opponent', gate=self.GATE)
        self.assertGreater(flipped['gate_score'], self.engine.history(TWO_TURN, nodes=135, gate=self.GATE)['gate_score'])

    def test_resident_table_keeps_proofs_and_is_bounded(self):
        strong = FIXTURE['positions']['1790600149713752:2:253']
        for table in (4, 4, 0):
            self.assertEqual(self.engine.history(strong, nodes=135, table_mb=table)['status'], 'PROVEN_WIN')
        with self.assertRaises(ValueError):
            self.engine.history(strong, nodes=135, table_mb=257)

    def test_invalid_gates_are_rejected(self):
        for gate in (dict(self.GATE, cap_low=9000), dict(self.GATE, floor=0), dict(self.GATE, weight=-1),
                     dict(weight=1., floor=1)):
            with self.assertRaises(ValueError):
                self.engine.history(NO_THREAT, nodes=135, gate=gate)
        with self.assertRaises(ValueError):
            IsolatedTactics(priority='realtime')

    def test_isolated_worker_gates_and_runs_at_its_priority(self):
        tactics = IsolatedTactics(priority='idle')
        try:
            result = tactics.history(FIXTURE['positions']['1790600149713752:2:253'], nodes=135, gate=self.GATE)
            self.assertEqual((result['status'], result['budget'], result['gate_score']), ('PROVEN_WIN', 540, 19.5))
        finally:
            tactics.close()


class IndependentCheckerBounds(unittest.TestCase):
    def test_negative_index_and_shared_dag_expansion_rejected(self):
        with self.assertRaises(ValueError):
            independent_verify(dict(version=1, width='wide', root=-1, nodes=[
                dict(kind='immediate_win', action=[[0,0]])]), [])
        nodes = [dict(kind='defender_replies', responses=[
            dict(action=[[0,0]], child=i+1), dict(action=[[0,1]], child=i+1)]) for i in range(18)]
        nodes.append(dict(kind='unstoppable', threats=[]))
        with self.assertRaisesRegex(ValueError, 'work limit'):
            independent_verify(dict(version=1, width='wide', root=0, nodes=nodes), [])


def _determinism_rows():
    """One JSON row per query: key, attacker, status, certificate hash, nodes used, proof turns."""
    engine = NativeTactics()
    queries = [(FIXTURE['control'], 'mover', FIXTURE['proving_nodes'][FIXTURE['control']])]
    queries += [(key, attacker, DETERMINISM_NODES) for key in FIXTURE['random'] for attacker in ('mover', 'opponent')]
    for key, attacker, nodes in queries:
        result = engine.history(FIXTURE['positions'][key], nodes=nodes, ms=60000, attacker=attacker)
        certificate = result['certificate']
        digest = certificate and hashlib.sha256(json.dumps(certificate, sort_keys=True).encode()).hexdigest()
        print(json.dumps([key, attacker, result['status'], digest, result['nodes_used'], result['proof_turns']]))


if __name__ == '__main__':
    if sys.argv[1:] == ['determinism']:
        _determinism_rows()
    else:
        unittest.main()
