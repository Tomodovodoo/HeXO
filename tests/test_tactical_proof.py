import copy
import ctypes as C
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from proof import VerificationTimeout
from tactical_proof import IsolatedTactics, NativeTactics, independent_verify, threat_cells
from tests import slow


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

    def test_puzzle_archive_preserves_turn_phase_and_repeated_positions(self):
        from notation import dumps
        from tools.proof_stamps import load_puzzles
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'puzzles.txt'
            histories = [OPEN_THREE, OPEN_THREE+[[8,8]], OPEN_THREE]
            path.write_text('\n\n'.join(dumps(h) for h in histories), encoding='utf-8-sig')
            archive = load_puzzles(path)
            rows = archive['cases']
            self.assertEqual([r['id'] for r in rows], [1,2,3])
            self.assertEqual([[list(p) for p in r['history']] for r in rows], histories)
            self.assertEqual([(r['mover'], r['remaining'], r['expected']) for r in rows],
                             [(0,2,'PROVEN_WIN'), (0,1,'PROVEN_WIN'), (0,2,'PROVEN_WIN')])
            self.assertEqual(rows[0]['position_sha256'], rows[2]['position_sha256'])
            self.assertNotEqual(rows[0]['position_sha256'], rows[1]['position_sha256'])
            self.assertEqual(archive['input_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            path.write_text(dumps(IMMEDIATE+[[5,0]]), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'already terminal'):
                load_puzzles(path)
            path.write_text('version[1]; 1. [0,0][1,0];', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_puzzles(path)

    def test_puzzle_benchmark_keeps_unknowns_and_checks_wins(self):
        from notation import dumps
        from tools.proof_stamps import puzzle_benchmark
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/'puzzles.txt', Path(directory)/'results.json'
            source.write_text(dumps(IMMEDIATE)+'\n'+dumps(NO_THREAT), encoding='utf-8')
            result = puzzle_benchmark(source, target, nodes=1, ms=5000, stamps=False)
            self.assertEqual(result['summary']['completed'], 2)
            self.assertEqual(result['summary']['solved'], 1)
            self.assertEqual([r['status'] for r in result['rows']], ['PROVEN_WIN', 'UNKNOWN'])
            self.assertGreater(result['rows'][0]['verification_ms'], 0)
            self.assertEqual(result['build_hash'], self.engine.metadata['binary_sha256'])
            self.assertEqual(json.loads(target.read_text())['rows'], result['rows'])

    def test_saved_moves_reprove_changed_positions_without_trusting_their_verdict(self):
        with NativeTactics(independent=True) as native:
            original = native.history(OPEN_THREE, nodes=20000, ms=5000)
            evidence = [dict(history=OPEN_THREE, winner=0, pv=[], certificate=original['certificate'])]
            changed = copy.deepcopy(OPEN_THREE)
            changed[-1] = [7,8]
            found = native.history(changed, replay=evidence, stamps=True, library=[], nodes=5000, ms=5000)
            self.assertEqual(found['status'], 'PROVEN_WIN', found['reason'])
            self.assertFalse(found.get('dependencies'))
            self.assertEqual(independent_verify(found['certificate'], changed), 'PROVEN_WIN')
            warm = native.history(changed, stamps=True, library=[], nodes=1, ms=3000)
            self.assertEqual(warm['status'], 'PROVEN_WIN', warm['reason'])
            counterwin = [[0,0],[0,8],[1,8],[1,0],[2,0],[2,8],[3,8]]
            rejected = native.history(counterwin, replay=evidence, stamps=True, library=[], nodes=5000, ms=5000)
            self.assertEqual(rejected['status'], 'UNKNOWN')
            with self.assertRaises(ValueError):
                independent_verify(found['certificate'], counterwin)
        with NativeTactics(independent=True) as native:
            half = OPEN_THREE + original['moves']
            lost = native.history(half, replay=evidence, attacker='defender', stamps=True, library=[], nodes=5000, ms=5000)
            self.assertEqual(lost['status'], 'PROVEN_LOSS', lost['reason'])
            self.assertEqual(independent_verify(lost['certificate'], half, attacker='defender'), 'PROVEN_LOSS')
        with NativeTactics(independent=True) as native:
            incomplete = [dict(history=OPEN_THREE, winner=0, pv=[[*p,0,i+1] for i,p in enumerate(original['moves'])])]
            failed = native.history(changed, replay=incomplete, stamps=True, library=[], nodes=5000, ms=5000)
            self.assertEqual(failed['status'], 'UNKNOWN')

    def test_independent_workers_keep_caches_and_stamps_separate(self):
        with NativeTactics(independent=True) as first, NativeTactics(independent=True) as second:
            options = dict(nodes=20000, ms=5000, table_mb=4, resume=True)
            discovered = first.history(OPEN_THREE, **options)
            reused = first.history(OPEN_THREE, **options)
            separate = second.history(OPEN_THREE, **options)
            self.assertEqual([r['status'] for r in (discovered, reused, separate)], ['PROVEN_WIN']*3)
            self.assertGreater(discovered['nodes_fresh'], 0)
            self.assertEqual(reused['nodes_fresh'], 0)
            self.assertTrue(reused['cache_hit'])
            self.assertFalse(separate['cache_hit'])
            self.assertGreater(separate['nodes_fresh'], 0)
            self.assertFalse(separate['resident_reused'])
            learned = first.history(OPEN_THREE, stamps=True, library=[], nodes=20000, ms=5000)
            changed = OPEN_THREE + [[8,8],[10,8],[8,10],[10,10]]
            warm = first.history(changed, stamps=True, library=[], nodes=1, ms=5000)
            cold = second.history(changed, stamps=True, library=[], nodes=1, ms=5000)
            self.assertEqual((learned['status'], warm['status'], cold['status']),
                             ('PROVEN_WIN', 'PROVEN_WIN', 'UNKNOWN'))
        self.assertEqual(independent_verify(warm['certificate'], changed), 'PROVEN_WIN')

    def test_direct_workers_bind_to_execution_thread_and_isolate_replacements(self):
        lib=self.engine.lib
        signatures={
            'worker_new_direct':(C.c_void_p,[]), 'worker_free_direct':(None,[C.c_void_p]),
            'worker_answer_direct':(C.c_void_p,[C.c_void_p,C.c_char_p]),
            'worker_busy_direct':(C.c_bool,[C.c_void_p]),
            'answer_json':(C.c_void_p,[C.c_void_p]), 'answer_free':(None,[C.c_void_p]),
        }
        for name,(result,args) in signatures.items():
            function=getattr(lib,'hexo_tactical_'+name);function.restype=result;function.argtypes=args
        def query(worker,history=OPEN_THREE,**options):
            request=dict(history=history,nodes=20000,idtt_nodes=0,ms=5000,depth=8,
                         table_mb=4,resume=True,**options)
            answer=lib.hexo_tactical_worker_answer_direct(worker,json.dumps(request).encode())
            self.assertTrue(answer)
            try:
                raw=lib.hexo_tactical_answer_json(answer)
                try:return json.loads(C.string_at(raw))
                finally:lib.hexo_tactical_free(raw)
            finally:lib.hexo_tactical_answer_free(answer)
        first=lib.hexo_tactical_worker_new_direct();second=lib.hexo_tactical_worker_new_direct()
        self.assertTrue(first and second)
        try:
            cold=query(first);warm=query(first)
            self.assertEqual(independent_verify(cold['certificate'],OPEN_THREE),'PROVEN_WIN')
            self.assertGreater(cold['nodes_fresh'],0);self.assertEqual(warm['nodes_fresh'],0)
            self.assertTrue(warm['cache_hit'])
            rejected=query(second)
            self.assertEqual((rejected['status'],rejected['nodes_fresh']),('UNKNOWN',0))
            self.assertIn('already owned',rejected['reason'])
            migrated=[]
            thread=threading.Thread(target=lambda:migrated.append(query(first)))
            thread.start();thread.join(2);self.assertFalse(thread.is_alive())
            self.assertEqual(migrated[0]['status'],'UNKNOWN')
            self.assertIn('thread changed',migrated[0]['reason'])
            query(first,stamps=True,library=[])
            changed=OPEN_THREE+[[8,8],[10,8],[8,10],[10,10]]
            self.assertEqual(query(first,changed,stamps=True,library=[])['status'],'PROVEN_WIN')
            self.assertFalse(lib.hexo_tactical_worker_busy_direct(first))
        finally:lib.hexo_tactical_worker_free_direct(first)
        try:
            # An unbound second handle can take over after the first retires.
            separate=query(second)
            self.assertEqual(separate['status'],'PROVEN_WIN')
            self.assertFalse(separate['cache_hit']);self.assertFalse(separate['resident_reused'])
            self.assertGreater(separate['nodes_fresh'],0)
            request=dict(history=changed,nodes=1,idtt_nodes=0,ms=5000,depth=8,
                         stamps=True,library=[])
            answer=lib.hexo_tactical_worker_answer_direct(second,json.dumps(request).encode())
            raw=lib.hexo_tactical_answer_json(answer)
            try:self.assertEqual(json.loads(C.string_at(raw))['status'],'UNKNOWN')
            finally:lib.hexo_tactical_free(raw);lib.hexo_tactical_answer_free(answer)
        finally:lib.hexo_tactical_worker_free_direct(second)

    def test_direct_cancellation_releases_one_worker_without_cancelling_another(self):
        lib=self.engine.lib
        for name,result,args in (
            ('worker_new_direct',C.c_void_p,[]), ('worker_free_direct',None,[C.c_void_p]),
            ('worker_answer_direct',C.c_void_p,[C.c_void_p,C.c_char_p]),
            ('worker_busy_direct',C.c_bool,[C.c_void_p]),
            ('answer_json',C.c_void_p,[C.c_void_p]), ('answer_free',None,[C.c_void_p]),
        ):
            function=getattr(lib,'hexo_tactical_'+name);function.restype=result;function.argtypes=args
        workers=[lib.hexo_tactical_worker_new_direct() for _ in range(2)]
        self.assertTrue(all(workers));token=lib.hexo_tactical_prepare()
        alternatives=[dict(action=[[0,0],[i,1]],child=0) for i in range(20000)]
        certificate=dict(version=1,width='wide',root=0,nodes=[dict(kind='attacker_move',
                         action=[[0,0],[1,1]],child=0,alternatives=alternatives)])
        request=dict(history=OPEN_THREE,nodes=20000,idtt_nodes=0,ms=5000,depth=8,
                     request_id=token,stamps=True,library=[],
                     replay=[dict(history=OPEN_THREE,winner=0,pv=[],certificate=certificate)])
        results=[[],[]];errors=[]
        def query(worker,request):
            answer=lib.hexo_tactical_worker_answer_direct(worker,json.dumps(request).encode())
            self.assertTrue(answer)
            raw=lib.hexo_tactical_answer_json(answer)
            try:return json.loads(C.string_at(raw))
            finally:lib.hexo_tactical_free(raw);lib.hexo_tactical_answer_free(answer)
        def execute(index):
            try:
                if index==0:results[0].append(query(workers[0],request))
                results[index].append(query(workers[index],dict(history=IMMEDIATE,nodes=100,
                                      idtt_nodes=0,ms=2000,depth=8)))
            except BaseException as error:errors.append(error)
        threads=[threading.Thread(target=execute,args=(i,)) for i in range(2)]
        try:
            threads[0].start();deadline=time.monotonic()+2
            while not lib.hexo_tactical_worker_busy_direct(workers[0]) and threads[0].is_alive() and time.monotonic()<deadline:
                time.sleep(.001)
            self.assertTrue(lib.hexo_tactical_worker_busy_direct(workers[0]))
            threads[1].start();start=time.perf_counter()
            self.assertTrue(lib.hexo_tactical_cancel(token))
            for thread in threads:thread.join(2)
            self.assertFalse(any(t.is_alive() for t in threads));self.assertFalse(errors,errors)
            self.assertLess(time.perf_counter()-start,.5)
            cancelled=results[0][0]
            self.assertEqual(cancelled['status'],'UNKNOWN');self.assertFalse(cancelled['native_verified'])
            self.assertIsInstance(cancelled['nodes_fresh'],int)
            for row in (results[0][1],results[1][0]):
                self.assertEqual(independent_verify(row['certificate'],IMMEDIATE),'PROVEN_WIN')
            self.assertFalse(any(lib.hexo_tactical_worker_busy_direct(w) for w in workers))
        finally:
            lib.hexo_tactical_cancel(token)
            for thread in threads:
                if thread.ident is not None:thread.join()
            lib.hexo_tactical_release(token)
            for worker in workers:lib.hexo_tactical_worker_free_direct(worker)

    def test_replay_indexes_shared_branches_and_keeps_later_evidence(self):
        with NativeTactics(independent=True) as native:
            source = native.history(OPEN_THREE, nodes=20000, ms=5000)['certificate']
        # These are untrusted move suggestions, deliberately not complete proofs.
        # The shared suffix formerly consumed the scan's 200,000-path limit;
        # the deep source must not discard a later, useful strategy either.
        shared = [dict(kind='defender_replies', responses=[dict(action=[[7+2*i,8+i],[8+2*i,8+i]], child=i+1)]*2)
                  for i in range(20)] + [dict(kind='exact', fact=0, after=[])]
        deep = [dict(kind='stamp_link', source=i+1) for i in range(101)] + [dict(kind='exact', fact=0, after=[])]
        changed = OPEN_THREE[:-1] + [[7,8]]
        for nodes in (shared, deep):
            with self.subTest(kind=nodes[0]['kind']), NativeTactics(independent=True) as native:
                evidence = [dict(history=OPEN_THREE, winner=0, pv=[], certificate=dict(version=1, width='wide', root=0, nodes=nodes)),
                            dict(history=OPEN_THREE, winner=0, pv=[], certificate=source)]
                found = native.history(changed, replay=evidence, stamps=True, library=[], nodes=5000, ms=5000)
                self.assertEqual(found['status'], 'PROVEN_WIN', found['reason'])
                self.assertFalse(found.get('dependencies'))
                self.assertEqual(independent_verify(found['certificate'], changed), 'PROVEN_WIN')

    def test_replay_uses_alternate_attacks_and_local_support_as_suggestions(self):
        with NativeTactics(independent=True) as native:
            source = native.history(OPEN_THREE, nodes=20000, ms=5000)['certificate']
        alternate = copy.deepcopy(source)
        root = alternate['nodes'][alternate['root']]
        root['alternatives'] = [dict(action=root['action'], child=root['child'])]
        root['action'] = [[0,0], [1,1]]  # The saved primary move is illegal.
        zone = copy.deepcopy(source)
        zone['nodes'].append(dict(kind='zone_replies', zone=[], responses=[], fallback=zone['root']))
        zone['root'] = len(zone['nodes'])-1
        changed = OPEN_THREE + [[8,8],[10,8],[8,10],[10,10]]
        counterwin = [[0,0],[0,8],[1,8],[1,0],[2,0],[2,8],[3,8]]
        for certificate in (source, alternate, zone):
            evidence = [dict(history=OPEN_THREE, winner=0, pv=[], certificate=certificate)]
            with self.subTest(kind=certificate['nodes'][certificate['root']]['kind']), NativeTactics(independent=True) as native:
                found = native.history(changed, replay=evidence, stamps=True, library=[], nodes=5000, ms=5000)
                self.assertEqual(found['status'], 'PROVEN_WIN', found['reason'])
                self.assertEqual(independent_verify(found['certificate'], changed), 'PROVEN_WIN')
                self.assertEqual(native.history(counterwin, replay=evidence, stamps=True, library=[], nodes=5000, ms=5000)['status'], 'UNKNOWN')

    def test_replay_checks_an_immediate_win_before_reading_saved_strategies(self):
        # A saved suggestion is irrelevant when this board already completes six.
        # Its invalid edge would fail evidence scanning if we did that first.
        evidence = [dict(history=OPEN_THREE, winner=0, pv=[],
                         certificate=dict(version=1, width='wide', root=0, nodes=[dict(kind='stamp_link', source=999)]))]
        with NativeTactics(independent=True) as native:
            found = native.history(IMMEDIATE, replay=evidence, stamps=True, library=[], nodes=1, ms=3000)
        self.assertEqual(found['status'], 'PROVEN_WIN', found['reason'])
        self.assertEqual(found['nodes_used'], 1)
        self.assertEqual(found['proof_turns'], 1)
        self.assertEqual(independent_verify(found['certificate'], IMMEDIATE), 'PROVEN_WIN')

    def test_large_saved_move_collection_yields_at_its_deadline(self):
        alternatives = [dict(action=[[0,0],[i,1]],child=0) for i in range(20000)]
        certificate = dict(version=1,width='wide',root=0,nodes=[dict(kind='attacker_move',
                           action=[[0,0],[1,1]],child=0,alternatives=alternatives)])
        evidence = [dict(history=OPEN_THREE,winner=0,pv=[],certificate=certificate)]
        with NativeTactics(independent=True) as native:
            start = time.perf_counter()
            found = native.history(OPEN_THREE,replay=evidence,stamps=True,library=[],nodes=20000,ms=100)
            self.assertEqual(found['status'],'UNKNOWN')
            self.assertLess(time.perf_counter()-start,.5)
            time.sleep(.2)
            after = native.history(IMMEDIATE,nodes=100,ms=1000)
            self.assertEqual(after['status'],'PROVEN_WIN')
            self.assertNotIn('busy',after['reason'])

    def test_independent_workers_overlap_and_cancel_only_their_query(self):
        history = [[0,0],[4,0],[7,0],[-1,0],[-2,0],[1,0],[5,0],[6,0],[-2,1]]
        with NativeTactics(independent=True) as first, NativeTactics(independent=True) as second:
            results = [None, None]
            start = threading.Barrier(3)
            def query(i, engine):
                start.wait()
                results[i] = engine.history(history, nodes=10000000, ms=20000)
            threads = [threading.Thread(target=query, args=(i, engine))
                       for i, engine in enumerate((first, second))]
            for thread in threads:
                thread.start()
            start.wait()
            try:
                deadline = time.perf_counter()+2
                while not (first.busy and second.busy) and time.perf_counter() < deadline:
                    time.sleep(.001)
                self.assertTrue(first.busy and second.busy, results)
                self.assertTrue(first.cancel())
                threads[0].join(2)
                self.assertFalse(threads[0].is_alive())
                self.assertEqual(results[0]['status'], 'UNKNOWN')
                self.assertIn('cancelled', results[0]['reason'])
                if not second.busy:
                    threads[1].join(1)
                    self.assertNotIn('cancelled', results[1]['reason'])
                replacement = first.history(IMMEDIATE, nodes=1000, ms=1000)
                self.assertEqual(replacement['status'], 'PROVEN_WIN', replacement)
            finally:
                first.cancel()
                second.cancel()
                for thread in threads:
                    thread.join(2)
                    self.assertFalse(thread.is_alive())
            self.assertNotIn('native worker busy', results[1]['reason'])

    def test_independent_close_stops_work_and_rejects_new_queries(self):
        engine = NativeTactics(independent=True)
        history = [[0,0],[4,0],[7,0],[-1,0],[-2,0],[1,0],[5,0],[6,0],[-2,1]]
        results = []
        thread = threading.Thread(target=lambda: results.append(engine.history(history, nodes=10000000, ms=20000)))
        thread.start()
        try:
            deadline = time.perf_counter()+2
            while not engine.busy and thread.is_alive() and time.perf_counter() < deadline:
                time.sleep(.001)
            self.assertTrue(engine.busy)
            engine.close()
            thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertFalse(engine.busy)
            self.assertEqual(results[0]['status'], 'UNKNOWN')
            self.assertIn('cancelled', results[0]['reason'])
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                engine.history(IMMEDIATE)
            engine.close()
        finally:
            engine.cancel()
            thread.join(2)
            engine.close()

    def test_learned_stamp_still_matches_its_original_position(self):
        proof = self.engine.history(LATE_WIN, nodes=50000, ms=3000)
        self.assertEqual(proof['status'], 'PROVEN_WIN')
        learned = self.engine.history(LATE_WIN, certificate=proof['certificate'], stamps=True, library=[], nodes=1, ms=5000)
        self.assertEqual(learned['status'], 'PROVEN_WIN', learned['reason'])
        # Removing enemy stones makes a shorter strategy possible, but that
        # strategy needs (0,-7) and (0,-3) empty on the real board.
        self.assertFalse(set(map(tuple, learned['stamp_learned']['empty'])) & set(map(tuple, LATE_WIN)))
        reused = self.engine.history(LATE_WIN, stamps=True, library=[], nodes=1, ms=3000)
        self.assertEqual(reused['status'], 'PROVEN_WIN', reused['reason'])
        self.assertEqual(reused['stamp_hits'], 1)
        self.assertLess(reused['stamp_bytes'], 4 * 1024 * 1024)
        self.assertEqual(independent_verify(reused['certificate'], LATE_WIN), 'PROVEN_WIN')

    def test_local_stamp_reuses_changed_positions_and_checks_interference(self):
        learned = self.engine.history(OPEN_THREE, stamps=True, nodes=20000, ms=5000)
        self.assertEqual(learned['status'], 'PROVEN_WIN', learned['reason'])
        history = OPEN_THREE + [[8,8],[10,8],[8,10],[10,10]]
        for sym in range(12):
            def rotate(p):
                q, r = p
                if sym >= 6:
                    q, r = r, q
                for _ in range(sym % 6):
                    q, r = -r, q+r
                return [q, r]
            transformed = list(map(rotate, history))
            result = self.engine.history(transformed, stamps=True, nodes=1, ms=3000)
            self.assertEqual(result['status'], 'PROVEN_WIN', (sym, result['reason']))
            self.assertEqual(result['nodes_fresh'], 1)
            self.assertEqual(independent_verify(result['certificate'], transformed), 'PROVEN_WIN')
        warm = self.engine.history(history, stamps=True, nodes=1, ms=3000)
        certificate = warm['certificate']
        # A relevant blocker and a distant immediate counter-threat must both
        # invalidate this particular strategy. Changed tempo also matters.
        cases = [OPEN_THREE + [[8,8],[10,8],warm['moves'][0],[9,10]],
                 OPEN_THREE + [[8,8],[10,8],[1,8],[3,8]], history + [[11,10]]]
        for changed in cases:
            result = self.engine.history(changed, certificate=certificate, stamps=True, nodes=1, ms=3000)
            self.assertEqual(result['status'], 'UNKNOWN', result)
            with self.assertRaises(ValueError):
                independent_verify(certificate, changed)
        # Enabling stamps does not populate the reference query's exact cache.
        reference = self.engine.history(history, nodes=1, ms=3000)
        self.assertEqual(reference['status'], 'UNKNOWN')
        other_colour = [[0,0],[0,8],[1,8],[8,0],[-8,0],[2,8],[10,-2],[0,-8],[-8,8]]
        result = self.engine.history(other_colour, stamps=True, nodes=1, ms=3000)
        self.assertEqual(result['status'], 'PROVEN_WIN', result['reason'])
        self.assertEqual(independent_verify(result['certificate'], other_colour), 'PROVEN_WIN')

    def test_retained_stamp_reuses_remote_stones_without_recompiling(self):
        with NativeTactics(independent=True) as native, NativeTactics(independent=True) as reference:
            learned = native.history(LATE_WIN, stamps=True, library=[], nodes=20000, ms=5000, bounds=True)
            self.assertEqual(learned['status'], 'PROVEN_WIN', learned['reason'])
            self.assertGreater(learned['stamp_timings']['compile verify'][0], 0)
            q, r = max(LATE_WIN)
            # Each stone is legal but at least eight cells from every earlier
            # stone, so none can block the proof or create a counter-threat.
            remote = [[q+8*(i+1), r] for i in range(8)]
            for count in (2, 4, 8):
                with self.subTest(stones=count):
                    history = LATE_WIN + remote[:count]
                    # Two placements change the mover. Query the original
                    # attacker's next turn; four and eight preserve the phase.
                    attacker = 'opponent' if count == 2 else 'mover'
                    reused = native.history(history, attacker=attacker, stamps=True, library=[],
                                            nodes=1, ms=5000, bounds=True)
                    cold = reference.history(history, attacker=attacker, nodes=1, ms=5000)
                    self.assertEqual(reused['status'], 'PROVEN_WIN', reused['reason'])
                    self.assertEqual(cold['status'], 'UNKNOWN', cold)
                    self.assertEqual(reused['nodes_fresh'], 1)
                    self.assertEqual(reused['stamp_hits'], 1)
                    self.assertEqual(reused['stamp_timings'].get('compile verify', [0])[0], 0)
                    self.assertEqual(independent_verify(reused['certificate'], history, attacker=attacker), 'PROVEN_WIN')

    def test_retained_stamp_closes_a_descendant_search_branch(self):
        with NativeTactics(independent=True) as native, NativeTactics(independent=True) as reference:
            found = reference.history(OPEN_THREE, nodes=20000, ms=5000)
            self.assertEqual(found['status'], 'PROVEN_WIN', found['reason'])
            certificate = copy.deepcopy(found['certificate'])
            root = certificate['nodes'][certificate['root']]
            reply = certificate['nodes'][root['child']]['responses'][0]
            child = OPEN_THREE + root['action'] + reply['action']
            certificate['root'] = reply['child']
            learned = native.history(child, certificate=certificate, stamps=True, library=[], nodes=1, ms=5000)
            self.assertEqual(learned['status'], 'PROVEN_WIN', learned['reason'])
            # Only the child proof is retained. It needs stones missing at the
            # parent, so this exercises a stamp reached during search.
            self.assertTrue(set(map(tuple, learned['stamp_learned']['required'])) - set(map(tuple, OPEN_THREE)))
            q, r = max(OPEN_THREE)
            history = OPEN_THREE + [[q+8*(i+1), r] for i in range(4)]
            warm = native.history(history, stamps=True, library=[], nodes=4096, ms=5000)
            cold = reference.history(history, nodes=4096, ms=5000)
            for result in (warm, cold):
                self.assertEqual(result['status'], 'PROVEN_WIN', result['reason'])
                cert = result['certificate']
                self.assertEqual(cert['nodes'][cert['root']]['kind'], 'attacker_move')
                self.assertEqual(independent_verify(cert, history), 'PROVEN_WIN')
            self.assertGreater(warm['stamp_hits'], 0)
            self.assertLess(warm['nodes_fresh'], cold['nodes_fresh'])
            self.assertEqual(warm['proof_turns'], cold['proof_turns'])

    def test_imported_stamp_requires_a_complete_checked_strategy(self):
        learned = self.engine.history(OPEN_THREE, stamps=True, nodes=20000, ms=5000)
        warm = self.engine.history(OPEN_THREE, stamps=True, nodes=1, ms=3000)
        source = warm['certificate']['nodes'][warm['certificate']['root']]['source']
        broken = copy.deepcopy(source)
        response = next(n for n in broken['certificate']['nodes'] if n['kind'] == 'defender_replies')
        response['responses'].pop()
        rejected = self.engine.history(OPEN_THREE, stamps=True, library=[broken], nodes=1, ms=3000)
        self.assertEqual(rejected['status'], 'UNKNOWN')
        self.assertEqual(learned['status'], 'PROVEN_WIN')

    def test_reordered_stamp_import_keeps_one_checked_strategy(self):
        with NativeTactics(independent=True) as engine:
            learned = engine.history(OPEN_THREE, stamps=True, library=[], nodes=20000, ms=5000)
            self.assertEqual(learned['status'], 'PROVEN_WIN')
            warm = engine.history(OPEN_THREE, stamps=True, library=[], nodes=1, ms=5000)
            source = warm['certificate']['nodes'][warm['certificate']['root']]['source']
            reordered = copy.deepcopy(source)
            reordered['stones'].reverse()
            history = OPEN_THREE + [[8,8],[10,8],[8,10],[10,10]]
            reused = engine.history(history, stamps=True, library=[reordered], nodes=1, ms=5000)
            self.assertEqual(reused['status'], 'PROVEN_WIN', reused['reason'])
            self.assertEqual(reused['stamp_entries'], warm['stamp_entries'])
            self.assertEqual(reused['stamp_hits'], 1)
            self.assertEqual(independent_verify(reused['certificate'], history), 'PROVEN_WIN')

    def test_quiet_defender_zone_covers_every_placement(self):
        history = [[0,0],[0,8],[8,0],[1,0],[0,1],[-8,0],[0,-8],[1,1],[12,-8],[-8,8]]
        reference = self.engine.history(history, attacker='defender', nodes=10000, ms=5000)
        self.assertEqual(reference['status'], 'UNKNOWN')
        result = self.engine.history(history, attacker='defender', stamps=True, nodes=20000, ms=10000)
        self.assertEqual(result['status'], 'PROVEN_LOSS', result['reason'])
        self.assertEqual(independent_verify(result['certificate'], history, attacker='defender'), 'PROVEN_LOSS')
        from dense_solver import Proof
        proof = Proof(list(map(tuple, history)), result['certificate'])
        self.assertTrue(proof.action(history + [[14,-8]]))
        self.assertEqual(proof.path(history + [[14,-8]])[0][0][:2], (len(history), -1))
        certificate = copy.deepcopy(result['certificate'])
        root = certificate['nodes'][certificate['root']]
        self.assertEqual(root['kind'], 'zone_replies')
        root['responses'].pop()
        self.assertEqual(self.engine.history(history, attacker='defender', certificate=certificate,
                                            stamps=True, nodes=20000, ms=5000)['status'], 'UNKNOWN')
        with self.assertRaises(ValueError):
            independent_verify(certificate, history, attacker='defender')

        # Matching the response list alone is insufficient: the outside-region
        # fallback must cover the removed cell too. Both checkers reject this.
        certificate = copy.deepcopy(result['certificate'])
        root = certificate['nodes'][certificate['root']]
        p = root['responses'].pop()['action'][0]
        root['zone'].remove(p)
        self.assertEqual(self.engine.history(history, attacker='defender', certificate=certificate,
                                            stamps=True, nodes=20000, ms=5000)['status'], 'UNKNOWN')
        with self.assertRaises(ValueError):
            independent_verify(certificate, history, attacker='defender')

    def test_open_three_wide_builder_full_strategy(self):
        result = self.engine.history(OPEN_THREE, nodes=100000, ms=5000, idtt_nodes=1000)
        self.assertEqual(result['status'], 'PROVEN_WIN', result)
        self.assertTrue(result['native_verified'])
        self.assertIsNotNone(result['idtt_verdict'])
        self.assertGreater(len(result['certificate']['nodes']), 1)
        self.assertEqual(independent_verify(result['certificate'], OPEN_THREE), 'PROVEN_WIN')
        cached = self.engine.history(OPEN_THREE, nodes=100000, ms=5000, idtt_nodes=1000)
        self.assertTrue(cached['cache_hit'])
        self.assertGreater(result['nodes_fresh'], 0)
        self.assertEqual(cached['nodes_fresh'], 0)
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

    def test_defender_root_reuses_the_three_exact_open_four_covers(self):
        history = OPEN_THREE + [[-1, 0], [2, 1]]
        cold = self.engine.history(history, attacker='defender', nodes=10000, ms=5000)
        self.assertEqual(cold['status'], 'PROVEN_LOSS')
        self.assertEqual(independent_verify(cold['certificate'], history, attacker='defender'), 'PROVEN_LOSS')
        replies = cold['certificate']['nodes'][0]['responses']
        self.assertEqual({tuple(map(tuple, r['action'])) for r in replies},
                         {((-3, 0), (3, 0)), ((-2, 0), (3, 0)), ((-2, 0), (4, 0))})
        known = [dict(history=history+r['action'], winner=0, plies=4*cold['proof_turns']) for r in replies]
        # A one-node budget cannot re-solve the covers. All three are graph terminals.
        warm = self.engine.history(history, attacker='defender', known=known, nodes=1, ms=1000, table_mb=4)
        self.assertEqual((warm['status'], warm['exact_hits'], warm['nodes_fresh']), ('PROVEN_LOSS', 3, 0))
        self.assertEqual(len(warm['certificate']['nodes']), 4)
        self.assertEqual(independent_verify(warm['certificate'], history, attacker='defender', known=known), 'PROVEN_LOSS')
        for premises in ([], known[:2], [dict(k, winner=1) for k in known]):
            rejected = self.engine.history(history, attacker='defender', known=premises,
                                           certificate=warm['certificate'], nodes=1, ms=1000)
            self.assertEqual(rejected['status'], 'UNKNOWN')
            with self.assertRaises(ValueError):
                independent_verify(warm['certificate'], history, attacker='defender', known=premises)
        # A subsequent query cannot use the preceding snapshot's resident or result cache.
        fresh = self.engine.history(history, attacker='defender', nodes=1, ms=1000, table_mb=4)
        self.assertEqual(fresh['status'], 'UNKNOWN')
        half = history + [[-2, 0]]
        result = self.engine.history(half, attacker='defender', known=known, nodes=1, ms=1000)
        self.assertEqual((result['status'], result['exact_hits']), ('PROVEN_LOSS', 2))
        self.assertEqual(independent_verify(result['certificate'], half, attacker='defender', known=known), 'PROVEN_LOSS')
        immediate = history + [[0, 1]]
        result = self.engine.history(immediate, attacker='defender', nodes=1, ms=1000)
        self.assertEqual(result['status'], 'PROVEN_LOSS')
        self.assertEqual(independent_verify(result['certificate'], immediate, attacker='defender'), 'PROVEN_LOSS')
        # The graph need not have allocated second-stone children of a refuted first stone.
        firsts = [dict(history=history+[list(p)], winner=0, plies=4*cold['proof_turns']+1)
                  for p in {tuple(p) for r in replies for p in r['action']}]
        result = self.engine.history(history, attacker='defender', known=firsts, nodes=1, ms=1000)
        self.assertEqual((result['status'], result['nodes_fresh']), ('PROVEN_LOSS', 0))
        self.assertEqual(independent_verify(result['certificate'], history, attacker='defender', known=firsts), 'PROVEN_LOSS')
        self.assertTrue(all(len(n['after']) == 1 for n in result['certificate']['nodes'] if n['kind'] == 'exact'))
        self.assertEqual(self.engine.history(NO_THREAT, attacker='defender', nodes=100, ms=1000)['status'], 'UNKNOWN')

    def test_attacker_search_and_verifier_use_interior_graph_terminals(self):
        post = OPEN_THREE + [[-1, 0], [2, 1]]
        proven = self.engine.history(post, attacker='defender', nodes=10000, ms=5000)
        self.assertEqual(proven['status'], 'PROVEN_LOSS')
        known = [dict(history=post, winner=0, plies=4*proven['proof_turns']+2)]
        result = self.engine.history(OPEN_THREE, known=known, nodes=1000, ms=2000)
        self.assertEqual(result['status'], 'PROVEN_WIN')
        self.assertGreater(result['exact_hits'], 0)
        self.assertEqual(independent_verify(result['certificate'], OPEN_THREE, known=known), 'PROVEN_WIN')
        with self.assertRaises(ValueError):
            independent_verify(result['certificate'], OPEN_THREE)

    def test_only_verified_exact_leaves_become_dependencies(self):
        result = self.engine.history(IMMEDIATE, nodes=1000, ms=1000)
        self.assertEqual(result['status'], 'PROVEN_WIN')
        known = [dict(history=IMMEDIATE, winner=0, plies=2)]
        for fact in (0, 999999):
            certificate = copy.deepcopy(result['certificate'])
            certificate['nodes'].append(dict(kind='exact', fact=fact))
            checked = self.engine.history(IMMEDIATE, known=known, certificate=certificate, nodes=1000, ms=1000)
            self.assertEqual(checked['status'], 'PROVEN_WIN')
            self.assertEqual((checked['exact_hits'], checked['dependencies']), (0, []))
            self.assertFalse(any(n['kind'] == 'exact' for n in checked['certificate']['nodes']))
            self.assertEqual(independent_verify(checked['certificate'], IMMEDIATE), 'PROVEN_WIN')
            certificate['root'] = len(certificate['nodes']) - 1
            rooted = self.engine.history(IMMEDIATE, known=known, certificate=certificate, nodes=1000, ms=1000)
            self.assertEqual(rooted['status'], 'PROVEN_WIN' if fact == 0 else 'UNKNOWN')
        certificate = self.engine.history(OPEN_THREE, nodes=10000, ms=2000)['certificate']
        certificate['nodes'][certificate['root']]['alternatives'] = [dict(action=[[-1, 0], [2, 1]], child=len(certificate['nodes']))]
        certificate['nodes'].append(dict(kind='exact', fact=999999))
        checked = self.engine.history(OPEN_THREE, certificate=certificate, nodes=10000, ms=2000)
        self.assertEqual((checked['status'], checked['dependencies']), ('PROVEN_WIN', []))
        self.assertEqual(independent_verify(checked['certificate'], OPEN_THREE), 'PROVEN_WIN')

    def test_cooperative_cancel_keeps_native_worker_available(self):
        engine = NativeTactics()
        self.assertFalse(engine.cancel())
        results = []
        history = [[0, 0], [4, 0], [7, 0], [-1, 0], [-2, 0], [1, 0], [5, 0], [6, 0], [-2, 1]]
        query = threading.Thread(target=lambda: results.append(engine.history(history, nodes=10000000, ms=20000)))
        query.start()
        deadline = time.perf_counter()+2
        while not engine.cancel() and query.is_alive() and time.perf_counter() < deadline:
            time.sleep(.001)
        query.join(2)
        self.assertFalse(query.is_alive())
        self.assertEqual(results[0]['status'], 'UNKNOWN')
        self.assertIn('cancelled', results[0]['reason'])
        self.assertFalse(engine.cancel())
        next_result = engine.history(IMMEDIATE, nodes=1000, ms=1000)
        self.assertEqual(next_result['status'], 'PROVEN_WIN', next_result)

    def test_isolated_cancel_retains_child_for_the_next_query(self):
        from tests.reference import interleave
        ours = [(q,r) for r in (0,3,6,9) for q in range(3)] + [(12,0)]
        theirs = [(-1,0)] + [(6+(i%3)*3,2+3*(i//3)) for i in range(13)]
        history = [list(p) for p in interleave([ours,theirs])]
        tactics = IsolatedTactics()
        try:
            tactics.history(NO_THREAT, ms=10000)
            pid = tactics.process.pid
            results = []
            query = threading.Thread(target=lambda: results.append(tactics.history(
                history, root_moves=[[3,0],[5,0]], nodes=1000000, ms=20000, table_mb=4)))
            query.start()
            time.sleep(.05)
            self.assertTrue(tactics.cancel())
            query.join(2)
            self.assertFalse(query.is_alive())
            self.assertEqual((results[0]['status'], results[0]['reason']), ('UNKNOWN', 'cancelled'))
            self.assertEqual(tactics.stats['kills'], 0)
            self.assertEqual(tactics.history(IMMEDIATE, ms=1000)['status'], 'PROVEN_WIN')
            self.assertEqual(tactics.process.pid, pid)
            reused = tactics.history(OPEN_THREE, stamps=True, nodes=1, ms=3000)
            self.assertEqual(reused['status'], 'PROVEN_WIN')
            self.assertEqual(independent_verify(json.loads(reused['certificate_json']), OPEN_THREE), 'PROVEN_WIN')
        finally:
            tactics.close()

    def test_shortest_tightens_the_certificate_to_the_fewest_turns(self):
        loose = self.engine.history(LATE_WIN, nodes=32768, ms=20000)
        self.assertEqual((loose['status'], loose['proof_turns'], loose['shortest']), ('PROVEN_WIN', 5, False))
        tight = self.engine.history(LATE_WIN, nodes=32768, ms=20000, shortest=True)
        self.assertEqual((tight['status'], tight['proof_turns'], tight['shortest'], tight['moves']),
                         ('PROVEN_WIN', 4, True, [[-1, -11], [-1, -10]]))
        self.assertEqual(independent_verify(tight['certificate'], LATE_WIN), 'PROVEN_WIN')
        again = self.engine.history(LATE_WIN, nodes=32768, ms=20000, shortest=True)
        self.assertEqual((again['cache_hit'], again['shortest'], again['certificate']), (True, True, tight['certificate']))
        for learned in (False, True):
            with self.subTest(learned_stamp=learned), NativeTactics(independent=True) as native:
                if learned:
                    native.history(LATE_WIN, certificate=loose['certificate'], stamps=True, library=[], nodes=1, ms=20000)
                    limited = native.history(LATE_WIN, shortest=True, stamps=True, library=[], nodes=1, ms=20000)
                    self.assertEqual((limited['status'], limited['proof_turns'], limited['shortest']),
                                     ('PROVEN_WIN', 5, False))
                    self.assertEqual(independent_verify(limited['certificate'], LATE_WIN), 'PROVEN_WIN')
                stamped = native.history(LATE_WIN, shortest=True, stamps=True, library=[], nodes=32768, ms=20000)
                self.assertEqual((stamped['status'], stamped['proof_turns'], stamped['shortest'], stamped['moves']),
                                 ('PROVEN_WIN', 4, True, tight['moves']))
                self.assertLessEqual(stamped['nodes_fresh'], 32768)
                self.assertEqual(independent_verify(stamped['certificate'], LATE_WIN), 'PROVEN_WIN')
        refused = self.engine.history(LATE_WIN * 2000, nodes=32768, ms=20000, shortest=True)
        self.assertEqual((refused['status'], refused['shortest']), ('UNKNOWN', False))

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


@slow
class SlowStrategy(unittest.TestCase):
    """Proofs that search for a minute."""
    @classmethod
    def setUpClass(cls):
        try:
            cls.engine = NativeTactics()
        except FileNotFoundError:
            raise unittest.SkipTest('Build tools/tactical with tools/build_tactical.py first')

    def test_unstoppable_shape_covers_a_complete_quiet_turn(self):
        history = [[0,0],[0,8],[8,0],[1,0],[0,1],[-8,0],[0,-8],[1,1],[12,-8]]
        result = self.engine.history(history, attacker='defender', stamps=True, nodes=50000, ms=60000)
        self.assertEqual(result['status'], 'PROVEN_LOSS', result['reason'])
        self.assertEqual(independent_verify(result['certificate'], history, attacker='defender',
                                           deadline_seconds=120), 'PROVEN_LOSS')


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

    def test_scoped_numbers_do_not_adjudicate_unknown_positions(self):
        quiet = self.engine.history(NO_THREAT, nodes=135, bounds=True)
        self.assertEqual(quiet['status'], 'UNKNOWN')
        self.assertFalse(quiet['native_verified'])
        self.assertEqual(quiet['proof_numbers']['scope'], 'wide-forcing')
        self.assertEqual(quiet['proof_numbers']['dn'], 0)
        self.assertFalse(quiet['proof_numbers']['game_exact'])
        control = FIXTURE['positions'][FIXTURE['control']]
        partial = self.engine.history(control, nodes=50, bounds=True, table_mb=4, resume=True)
        self.assertEqual(partial['status'], 'UNKNOWN')
        self.assertIsNotNone(partial['proof_numbers'])
        self.assertLessEqual(partial['nodes_fresh'], 50)
        with self.assertRaises(ValueError):
            self.engine.history(control, resume=True)

    def test_isolated_worker_returns_slice_accounting(self):
        tactics = IsolatedTactics()
        try:
            first = tactics.history(OPEN_THREE, nodes=100000, ms=10000, bounds=True, resume=True, table_mb=4)
            self.assertEqual(first['status'], 'PROVEN_WIN', first)
            self.assertEqual(independent_verify(json.loads(first['certificate_json']), OPEN_THREE), 'PROVEN_WIN')
            again = tactics.history(OPEN_THREE, nodes=100000, ms=10000, bounds=True, resume=True, table_mb=4)
            self.assertTrue(again['cache_hit'])
            self.assertEqual(again['nodes_fresh'], 0)
            self.assertEqual(again['proof_numbers']['pn'], 0)
        finally:
            tactics.close()

    def test_unfinished_proof_continues_after_table_growth(self):
        history = FIXTURE['positions']['1790600287230040:30:248']
        self.engine.history(NO_THREAT, nodes=1, table_mb=0)
        try:
            first = self.engine.history(history, nodes=512, ms=10000, bounds=True, resume=True, table_mb=4)
            self.assertEqual(first['status'], 'UNKNOWN')
            self.assertFalse(first['native_verified'])
            self.assertIsNotNone(first['proof_numbers'])
            continued = self.engine.history(history, nodes=512, ms=10000, bounds=True, resume=True, table_mb=8)
            self.assertEqual(continued['status'], 'PROVEN_WIN', continued)
            self.assertTrue(continued['resident_reused'])
            self.assertFalse(continued['cache_hit'])
            self.assertLess(continued['nodes_fresh'], first['nodes_fresh'])
            self.assertEqual(independent_verify(continued['certificate'], history), 'PROVEN_WIN')
        finally:
            self.engine.history(NO_THREAT, nodes=1, table_mb=0)

    def test_unfinished_frontiers_reuse_nodes_with_a_new_work_allowance(self):
        history = FIXTURE['positions']['1790591506645044:16:209']
        with NativeTactics(independent=True) as worker:
            attempts = [worker.history(history, nodes=2048, ms=10000, bounds=True, resume=True, table_mb=4)
                        for _ in range(6)]
        self.assertTrue(any(r['frontier_reused_nodes'] > 0 for r in attempts[1:]))
        self.assertTrue(any(r['status'] == 'PROVEN_WIN' and r['native_verified'] for r in attempts))
        for result in attempts:
            self.assertLessEqual(result['frontier_bytes'], 4*1024*1024)
            self.assertLessEqual(result['nodes_fresh'], 2048)
            if result['native_verified']:
                self.assertEqual(independent_verify(result['certificate'], history), 'PROVEN_WIN')
            else:
                self.assertEqual(result['status'], 'UNKNOWN')
                self.assertIsNone(result['certificate'])

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
    def test_negative_index_and_malformed_shared_dag_are_rejected(self):
        with self.assertRaises(ValueError):
            independent_verify(dict(version=1, width='wide', root=-1, nodes=[
                dict(kind='immediate_win', action=[[0,0]])]), [])
        nodes = [dict(kind='defender_replies', responses=[
            dict(action=[[0,0]], child=i+1), dict(action=[[0,1]], child=i+1)]) for i in range(18)]
        nodes.append(dict(kind='unstoppable', threats=[]))
        # Shared nodes no longer expand exponentially during conversion; the
        # raw checker rejects this graph's invalid defender root directly.
        with self.assertRaisesRegex(ValueError, 'Invalid forcing certificate'):
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
