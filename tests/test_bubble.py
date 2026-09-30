import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import bubble


class FakeProcesses:
    """Records spawned commands and lets tests decide which pids are alive."""

    def __init__(self):
        self.spawned, self.killed, self.live, self.next_pid = [], [], set(), 100

    def spawn(self, command, log):
        self.spawned.append((command, log))
        self.next_pid += 1
        self.live.add(self.next_pid)
        return self.next_pid

    def alive(self, pid):
        return pid in self.live

    def kill(self, pid):
        self.killed.append(pid)
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


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.run = Path(self.directory.name) / 'run'
        self.run.mkdir()
        self.fake = FakeProcesses()
        self.launcher = bubble.Launcher(self.run, self.fake.spawn, self.fake.alive, self.fake.kill)
        self.plan = bubble.commands(self.run)

    def tearDown(self):
        self.directory.cleanup()

    def test_prepare_creates_config_and_first_checkpoint_once(self):
        calls = []
        self.launcher.prepare('cpu', calls.append)
        self.assertEqual([Path(c[1]).name for c in calls], ['dense_config.py', 'dense_learn.py'])
        self.assertEqual(calls[0][calls[0].index('--device') + 1], 'cpu')
        (self.run / 'config.json').write_text('{}')
        (self.run / 'checkpoints' / 'main').mkdir(parents=True)
        calls.clear()
        self.launcher.prepare('cpu', calls.append)
        self.assertEqual(calls, [])

    def test_start_records_pids_and_refuses_a_second_start(self):
        state = self.launcher.start(self.plan)
        self.assertEqual(sorted(state), sorted(bubble.SERVICES))
        self.assertEqual(json.loads((self.run / 'processes.json').read_text())['learner']['pid'], state['learner']['pid'])
        self.assertEqual([log.name for _, log in self.fake.spawned], list(bubble.SERVICES))
        with self.assertRaises(RuntimeError):
            self.launcher.start(self.plan)

    def test_stop_kills_live_processes_and_clears_state(self):
        state = self.launcher.start(self.plan)
        self.fake.live.discard(state['proof']['pid'])
        self.assertEqual(self.launcher.stop(), sorted(bubble.SERVICES))
        self.assertEqual(sorted(self.fake.killed), sorted(e['pid'] for n, e in state.items() if n != 'proof'))
        self.assertFalse((self.run / 'processes.json').exists())
        self.assertEqual(self.launcher.stop(), [])

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
