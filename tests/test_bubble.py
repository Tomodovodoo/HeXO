import json
import os
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import bubble


class FakeProcesses:
    """Records spawned commands and lets tests decide which pids are alive."""

    def __init__(self):
        self.spawned, self.killed, self.live, self.markers, self.next_pid = [], [], set(), {}, 100
        self.stubborn = set()

    def spawn(self, command, log):
        self.spawned.append((command, log))
        self.next_pid += 1
        self.live.add(self.next_pid)
        self.markers[self.next_pid] = list(command)
        return self.next_pid

    def arguments(self, pid):
        return self.markers[pid] if pid in self.live else []

    def kill(self, pid, force=False):
        self.killed.append(pid)
        if pid not in self.stubborn or force:
            self.live.discard(pid)


class CommandTests(unittest.TestCase):
    def test_match_command_submits_seats_and_shared_clock_to_the_player(self):
        from unittest.mock import patch
        args = ['bubble.py', 'match', 'run@123{simulations=512,solver_nodes=0}', 'Six@quick',
                '--port', '8877', '--tc', '180+2', '--unique-openings', '16', '--a-device', 'cpu']
        with patch.object(sys, 'argv', args), patch.object(bubble, 'ensure_player', return_value={}), \
                patch.object(bubble, 'player_request', return_value={}) as request:
            bubble.main()
        port, path, body = request.call_args.args
        self.assertEqual((port, path), (8877, '/match'))
        self.assertEqual(body['players'][0], dict(engine='run@123{simulations=512,solver_nodes=0}', device='cpu'))
        self.assertEqual(body['players'][1], dict(engine='Six@quick'))
        self.assertEqual(body['clock'], dict(mode='game', tc='180+2'))
        self.assertEqual(body['unique_openings'], 16)

    def test_every_service_targets_the_run(self):
        plan = bubble.commands(Path('runs/x'), actors=3, dashboard_port=9000)
        self.assertEqual(set(plan), set(bubble.SERVICES))
        for name, command in plan.items():
            self.assertIn('--run', command, name)
            self.assertEqual(command[command.index('--run') + 1], str(Path('runs/x')), name)
        self.assertEqual(plan['actors'][plan['actors'].index('--processes') + 1], '3')
        self.assertEqual(plan['dashboard'][plan['dashboard'].index('--port') + 1], '9000')
        self.assertIn('--eval-anchor-games', plan['evaluator'])
        self.assertNotIn('--eval-anchor-games', bubble.commands(Path('runs/x'), seal=True)['evaluator'])

    def test_proof_pass_is_optional(self):
        plan = bubble.commands(Path('runs/x'), proof=False)
        self.assertNotIn('proof', plan)
        self.assertEqual(set(plan) | {'proof'}, set(bubble.SERVICES))
        actors = plan['actors']
        self.assertEqual(actors[actors.index('--hybrid-proof-workers') + 1], '0')
        self.assertNotIn('--hybrid-proof-workers', bubble.commands(Path('runs/x'))['actors'])

    def test_kernels_reach_gpu_services_only_when_given(self):
        plan = bubble.commands(Path('runs/x'), kernels='fused')
        for name in ('learner', 'actors', 'evaluator'):
            self.assertEqual(plan[name][plan[name].index('--net-kernels') + 1], 'fused', name)
        for command in bubble.commands(Path('runs/x')).values():
            self.assertNotIn('--net-kernels', command)
        self.assertIn('--no-cuda-graphs', bubble.commands(Path('runs/x'), kernels='reference')['actors'])
        self.assertNotIn('--no-cuda-graphs', plan['actors'])


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.run = Path(self.directory.name) / 'run'

    def tearDown(self):
        self.directory.cleanup()

    def test_install_places_weights_and_champion(self):
        source = Path(self.directory.name) / 'ema.pt'
        source.write_bytes(b'weights')
        source.with_name('manifest.json').write_text('{}')
        target = bubble.install(self.run, source, 122500)
        self.assertEqual(target, self.run / 'checkpoints' / 'play' / '122500')
        self.assertEqual((target / 'ema.pt').read_bytes(), b'weights')
        self.assertTrue((target / 'manifest.json').exists())
        self.assertEqual(json.loads((self.run / 'champion.json').read_text())['checkpoint'], 'play/122500')
        self.assertEqual(bubble.exports(self.run), [target / 'ema.pt'])

    def test_install_rejects_a_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            bubble.install(self.run, Path(self.directory.name) / 'missing.pt', 1)
        self.assertFalse((self.run / 'champion.json').exists())

    def test_download_uses_the_release_tag_as_step(self):
        def fetch(url):
            if url.endswith('/latest'):
                return b'', 'https://github.com/Tomodovodoo/HeXO/releases/tag/bubble-125000'
            self.assertIn('/download/bubble-125000/', url)
            name = url.rsplit('/', 1)[-1]
            return {'ema.pt': b'weights', 'manifest.json': b'{}'}[name], 'https://release-assets.example/x/y'
        target = bubble.download(self.run, fetch)
        self.assertEqual(target.name, '125000')
        self.assertEqual((target / 'ema.pt').read_bytes(), b'weights')
        self.assertTrue((target / 'manifest.json').exists())
        self.assertEqual([p.name for p in (self.run / 'checkpoints').iterdir()], ['play'])


class MembersTests(unittest.TestCase):
    def test_workers_follow_accepted_parents_only(self):
        worker = lambda parent: ['python', '-c', f'from multiprocessing.spawn import spawn_main; spawn_main(parent_pid={parent}, pipe_handle=1)']
        candidates = {11: ['python', 'dense_learn.py', '--run', '/r'], 12: worker(11), 13: worker(12),
                      21: ['python', 'dense_learn.py', '--run', '/other'], 22: worker(21), 30: worker(99)}
        self.assertEqual(bubble.members(candidates, '/r', 10), {11, 12, 13})
        self.assertEqual(bubble.members({40: worker(10)}, '/r', 10), {40})
        self.assertEqual(bubble.members({30: worker(99)}, '/r', 10), set())

    def test_posix_workers_follow_parent_pids(self):
        spawn = ['python', '-c', 'from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=5, pipe_handle=7)']
        candidates = {11: ['python', 'dense_learn.py', '--run', '/r'], 12: spawn, 13: spawn, 14: spawn}
        parents = {11: 10, 12: 11, 13: 12, 14: 999}
        self.assertEqual(bubble.members(candidates, '/r', 10, parents), {11, 12, 13})
        self.assertEqual(bubble.members({14: spawn}, '/r', 10, {14: 10}), {14})


class MatchTests(unittest.TestCase):
    def test_exact_run_and_script(self):
        parts = ['python', '-u', '/hexo/python/dense_learn.py', '--run', '/runs/my bubble']
        self.assertTrue(bubble.matches(parts, 'dense_learn.py', '/runs/my bubble'))
        self.assertFalse(bubble.matches(parts, 'dense_learn.py', '/runs/my bubble-old'))
        self.assertFalse(bubble.matches(parts, 'dense_eval.py', '/runs/my bubble'))
        self.assertFalse(bubble.matches([], 'dense_learn.py', '/runs/my bubble'))


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.run = Path(self.directory.name) / 'run'
        self.run.mkdir()
        self.fake = FakeProcesses()
        self.launcher = bubble.Launcher(self.run, self.fake.spawn, self.fake.arguments, self.fake.kill)
        self.fake.survivors = {}
        self.launcher.group = lambda entry: self.fake.survivors.get(entry['pid'], [])
        self.plan = bubble.commands(self.run)

    def tearDown(self):
        self.directory.cleanup()

    def test_prepare_creates_config_and_first_checkpoint_once(self):
        calls = []
        self.launcher.prepare('cpu', calls.append)
        self.assertEqual([Path(c[1]).name for c in calls], ['dense_config.py', 'dense_learn.py'])
        self.assertEqual(calls[0][calls[0].index('--device') + 1], 'cpu')
        (self.run / 'config.json').write_text('{}')
        export = self.run / 'checkpoints' / 'main' / '000000'
        export.mkdir(parents=True)
        calls.clear()
        self.launcher.prepare('cpu', calls.append)
        self.assertEqual([Path(c[1]).name for c in calls], ['dense_learn.py'])
        (export / 'ema.pt').write_bytes(b'')
        (export / 'manifest.json').write_text('{}')
        calls.clear()
        self.launcher.prepare('cpu', calls.append)
        self.assertEqual([Path(c[1]).name for c in calls], ['dense_learn.py'])
        for name in ('model.pt', 'optimizer.pt'):
            (export / name).write_bytes(b'')
        calls.clear()
        self.launcher.prepare('cpu', calls.append)
        self.assertEqual(calls, [])

    def test_prepare_checks_device_and_phase_actors_of_an_existing_run(self):
        (self.run / 'config.json').write_text(json.dumps(dict(device='cuda', learner=dict(phase_actors=6))))
        calls = []
        with self.assertRaises(RuntimeError):
            self.launcher.prepare('cpu', calls.append, 4)
        with self.assertRaises(RuntimeError):
            self.launcher.prepare('auto', calls.append, 4)
        with self.assertRaises(RuntimeError):
            self.launcher.prepare('auto', calls.append, 6)
        (self.run / 'config.json').write_text(json.dumps(dict(device='cuda', learner=dict(phase_actors=6),
                                                              actor=dict(phase_follow=True))))
        self.launcher.prepare('auto', calls.append, 6)
        self.assertEqual([Path(c[1]).name for c in calls], ['dense_learn.py'])

    def test_prepare_requires_the_tactical_build_for_solver_budgets(self):
        (self.run / 'config.json').write_text(json.dumps(dict(actor=dict(solver_root_nodes=135))))
        with self.assertRaises(RuntimeError):
            self.launcher.prepare('auto', lambda c: None, 4, tactical=False)
        self.launcher.prepare('auto', lambda c: None, 4, tactical=True)
        (self.run / 'config.json').write_text(json.dumps(dict(actor=dict(solver_root_nodes=0, solver_cap_nodes=512))))
        self.launcher.prepare('auto', lambda c: None, 4, tactical=False)
        (self.run / 'league.json').write_text(json.dumps(dict(variants=[dict(settings=dict(solver_threat_nodes=135))])))
        with self.assertRaises(RuntimeError):
            self.launcher.prepare('auto', lambda c: None, 4, tactical=False)

    def test_prepare_probes_the_configured_variant(self):
        (self.run / 'config.json').write_text(json.dumps(dict(learner=dict(variant='alt'))))
        export = self.run / 'checkpoints' / 'alt' / '000000'
        export.mkdir(parents=True)
        for name in ('ema.pt', 'model.pt', 'optimizer.pt', 'manifest.json'):
            (export / name).write_bytes(b'')
        calls = []
        self.launcher.prepare('cpu', calls.append)
        self.assertEqual(calls, [])

    def test_downloaded_weights_do_not_count_as_a_learner_checkpoint(self):
        source = Path(self.directory.name) / 'ema.pt'
        source.write_bytes(b'weights')
        bubble.install(self.run, source, 125000)
        (self.run / 'config.json').write_text('{}')
        calls = []
        self.launcher.prepare('cpu', calls.append)
        self.assertEqual([Path(c[1]).name for c in calls], ['dense_learn.py'])

    def test_start_creates_a_missing_run_directory(self):
        launcher = bubble.Launcher(self.run / 'fresh', self.fake.spawn, self.fake.arguments, self.fake.kill)
        self.assertEqual(sorted(launcher.start(bubble.commands(self.run / 'fresh'))), sorted(bubble.SERVICES))

    def test_run_path_is_absolute(self):
        self.assertTrue(bubble.Launcher('runs/x', self.fake.spawn, self.fake.arguments, self.fake.kill).run.is_absolute())

    def test_reused_pid_is_not_a_service(self):
        state = self.launcher.start(self.plan)
        self.fake.markers[state['learner']['pid']] = bubble.commands(Path(f'{self.run}-old'))['learner']
        self.assertNotIn('learner', self.launcher.running())
        self.launcher.stop()
        self.assertNotIn(state['learner']['pid'], self.fake.killed)

    def test_start_and_stop_refuse_while_the_run_is_locked(self):
        with self.launcher.locked():
            other = bubble.Launcher(self.run, self.fake.spawn, self.fake.arguments, self.fake.kill)
            with self.assertRaises(RuntimeError):
                other.start(self.plan)
            with self.assertRaises(RuntimeError):
                other.stop()
        self.assertEqual(self.fake.spawned, [])
        self.assertEqual(sorted(self.launcher.start(self.plan)), sorted(bubble.SERVICES))

    def test_prepare_runs_under_the_lock(self):
        def probe():
            other = bubble.Launcher(self.run, self.fake.spawn, self.fake.arguments, self.fake.kill)
            with self.assertRaises(RuntimeError):
                with other.locked():
                    pass
            seen.append(True)
        seen = []
        self.launcher.start(self.plan, probe)
        self.assertEqual(seen, [True])

    def test_start_records_pids_and_refuses_a_second_start(self):
        state = self.launcher.start(self.plan)
        self.assertEqual(sorted(state), sorted(bubble.SERVICES))
        self.assertEqual(json.loads((self.run / 'processes.json').read_text())['learner']['pid'], state['learner']['pid'])
        self.assertEqual([log.name for _, log in self.fake.spawned], list(bubble.SERVICES))
        with self.assertRaises(RuntimeError):
            self.launcher.start(self.plan)

    def test_failed_start_kills_what_it_spawned(self):
        def failing_spawn(command, log):
            if log.name == 'evaluator':
                raise OSError('no more processes')
            return self.fake.spawn(command, log)
        launcher = bubble.Launcher(self.run, failing_spawn, self.fake.arguments, self.fake.kill)
        launcher.group = lambda entry: []
        self.fake.stubborn.add(101)
        with self.assertRaises(OSError):
            launcher.start(self.plan, timeout=0.01)
        self.assertEqual(sorted(self.fake.killed), [101, 101, 102])
        self.assertFalse(self.fake.live)
        self.assertFalse((self.run / 'processes.json').exists())

    def test_failed_start_records_services_that_survive_cleanup(self):
        def failing_spawn(command, log):
            if log.name == 'proof':
                raise OSError('no more processes')
            return self.fake.spawn(command, log)
        self.fake.kill = lambda p, force=False: self.fake.killed.append(p)
        launcher = bubble.Launcher(self.run, failing_spawn, self.fake.arguments, self.fake.kill)
        launcher.group = lambda entry: []
        with self.assertRaises(OSError):
            launcher.start(self.plan, timeout=0.01)
        recorded = json.loads((self.run / 'processes.json').read_text())
        self.assertEqual(sorted(recorded), ['actors', 'evaluator', 'learner'])
        self.assertEqual(sorted(launcher.running()), ['actors', 'evaluator', 'learner'])

    def test_stop_kills_live_processes_and_clears_state(self):
        state = self.launcher.start(self.plan)
        self.fake.live.discard(state['proof']['pid'])
        self.assertEqual(self.launcher.stop(), sorted(bubble.SERVICES))
        self.assertEqual(sorted(self.fake.killed), sorted(e['pid'] for n, e in state.items() if n != 'proof'))
        self.assertFalse((self.run / 'processes.json').exists())
        self.assertEqual(self.launcher.stop(), [])

    @unittest.skipUnless(os.name == 'nt', 'POSIX signals the whole process group instead')
    def test_stop_signals_surviving_workers_directly(self):
        state = self.launcher.start(self.plan)
        pid = state['actors']['pid']
        self.fake.live.discard(pid)
        self.fake.survivors[pid] = [777, 778]
        self.assertIn('actors', self.launcher.running())
        original = self.fake.kill

        def kill(target, force=False):
            original(target, force)
            self.fake.survivors[pid] = [p for p in self.fake.survivors[pid] if p != target]
        self.launcher.kill = kill
        self.launcher.stop(timeout=0.01)
        self.assertTrue({777, 778} <= set(self.fake.killed))
        self.assertNotIn('actors', self.launcher.running())

    def test_stop_forces_a_service_that_ignores_the_first_signal(self):
        state = self.launcher.start(self.plan)
        self.fake.stubborn.add(state['learner']['pid'])
        self.assertEqual(self.launcher.stop(timeout=0.01), sorted(bubble.SERVICES))
        self.assertEqual(self.fake.killed.count(state['learner']['pid']), 2)
        self.assertFalse((self.run / 'processes.json').exists())

    def test_stop_keeps_records_of_a_service_that_will_not_exit(self):
        state = self.launcher.start(self.plan)
        pid = state['evaluator']['pid']
        self.fake.stubborn.add(pid)
        self.fake.kill = lambda p, force=False: self.fake.killed.append(p)
        self.launcher.kill = self.fake.kill
        with self.assertRaises(RuntimeError):
            self.launcher.stop(timeout=0.01)
        self.assertTrue((self.run / 'processes.json').exists())

    def test_status_reports_life_and_stage(self):
        state = self.launcher.start(self.plan)
        self.fake.live.discard(state['actors']['pid'])
        (self.run / 'learner-status.json').write_text(json.dumps(dict(stage='training', step=42, updated_at=0)))
        (self.run / 'champion.json').write_text(json.dumps(dict(checkpoint='main/000500')))
        lines = dict(line.split(maxsplit=1) for line in self.launcher.status())
        self.assertTrue(lines['learner'].startswith('alive') and 'training 42' in lines['learner'])
        self.assertTrue(lines['actors'].startswith('gone'))
        self.assertEqual(lines['champion'], 'main/000500')

    def test_start_without_the_proof_pass(self):
        state = self.launcher.start(bubble.commands(self.run, proof=False))
        self.assertNotIn('proof', state)
        self.assertTrue(any(line.startswith('proof') and line.endswith('not started') for line in self.launcher.status()))
        self.assertEqual(self.launcher.stop(), sorted(set(bubble.SERVICES) - {'proof'}))

    def test_status_before_start(self):
        self.assertTrue(all(line.endswith('not started') for line in self.launcher.status()))


if __name__ == '__main__':
    unittest.main()
