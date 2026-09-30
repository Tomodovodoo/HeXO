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

    def test_kernels_reach_gpu_services(self):
        plan = bubble.commands(Path('runs/x'), kernels='fused')
        for name in ('learner', 'actors', 'evaluator'):
            self.assertEqual(plan[name][plan[name].index('--net-kernels') + 1], 'fused', name)


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
        self.assertFalse((self.run / 'checkpoints' / 'download').exists())


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
        with self.assertRaises(OSError):
            launcher.start(self.plan)
        self.assertEqual(len(self.fake.killed), 2)
        self.assertFalse(self.fake.live)
        self.assertFalse((self.run / 'processes.json').exists())

    def test_stop_kills_live_processes_and_clears_state(self):
        state = self.launcher.start(self.plan)
        self.fake.live.discard(state['proof']['pid'])
        self.assertEqual(self.launcher.stop(), sorted(bubble.SERVICES))
        self.assertEqual(sorted(self.fake.killed), sorted(e['pid'] for n, e in state.items() if n != 'proof'))
        self.assertFalse((self.run / 'processes.json').exists())
        self.assertEqual(self.launcher.stop(), [])

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

    def test_status_before_start(self):
        self.assertTrue(all(line.endswith('not started') for line in self.launcher.status()))


if __name__ == '__main__':
    unittest.main()
