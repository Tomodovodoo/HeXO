import copy
import time
import unittest
from tactical_proof import NativeTactics, independent_verify


OPEN_THREE = [[0,0],[0,8],[2,8],[1,0],[2,0],[4,8],[6,8]]
IMMEDIATE = [[0,0],[0,3],[1,3],[1,0],[2,0],[2,3],[3,3],[3,0],[4,0],[4,3],[5,4]]


class NativeStrategy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.engine = NativeTactics()
        except FileNotFoundError:
            raise unittest.SkipTest('Build tools/tactical with tools/build_tactical.py first')

    def test_open_three_wide_builder_full_strategy(self):
        result = self.engine.history(OPEN_THREE, ms=5000, idtt_ms=20)
        self.assertEqual(result['status'], 'PROVEN_WIN', result)
        self.assertTrue(result['native_verified'])
        self.assertGreater(len(result['certificate']['nodes']), 1)
        self.assertEqual(independent_verify(result['certificate'], OPEN_THREE), 'PROVEN_WIN')
        cached = self.engine.history(OPEN_THREE, ms=5000)
        self.assertTrue(cached['cache_hit'])
        self.assertEqual(cached['status'], 'PROVEN_WIN')
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
        result = self.engine.history(history, ms=20000, idtt_ms=0, nodes=1000000,
                                     root_moves=[[3,0],[5,0]])
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
        result = self.engine.history(history, ms=300, idtt_ms=0, nodes=1000000, root_moves=[[3,0],[5,0]])
        self.assertEqual(result['status'], 'UNKNOWN')
        time.sleep(0.2)
        after = self.engine.history([[0,0]], ms=1000, idtt_ms=0)
        self.assertNotIn('busy', after['reason'])
        self.assertLess(after['last_worker_completion']['last_after_deadline']['elapsed_ms'], 450)

    def test_deadline_and_unknown_are_not_loss(self):
        start = time.perf_counter()
        result = self.engine.history([[0,0]], ms=1, idtt_ms=0, nodes=1)
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertFalse(result['native_verified'])
        self.assertLess(time.perf_counter()-start, 0.25)
        # Let the sole native worker finish before another test needs it.
        time.sleep(0.05)


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


if __name__ == '__main__':
    unittest.main()
