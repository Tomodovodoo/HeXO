"""web/engine/assets.mjs: engine files from this origin, else from the public site, checked and kept in the Cache API."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT/'web'/'engine'
NODE = shutil.which('node')
SITE = 'https://tomodovodoo.github.io/HeXO/engine/'
LOADERS = ('drip-worker.mjs', 'six-worker.mjs', 'strix-worker.mjs', 'shrimp-worker.mjs', 'seal-worker.mjs',
           'network.mjs', 'shrimp/network.mjs', 'bubble.mjs', 'drip.mjs', 'six.mjs', 'strix.mjs', 'shrimp.mjs', 'seal.mjs')


@unittest.skipUnless(NODE, 'needs node')
class Resolver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        run = subprocess.run([NODE, str(ROOT/'tests'/'web'/'assets.mjs')], capture_output=True, text=True, check=True)
        cls.out = json.loads(run.stdout)

    def test_same_origin_file_is_read_here_and_cached_under_its_url(self):
        case = self.out['same_origin']
        self.assertEqual(case['body'], 'local bytes')
        self.assertEqual(len(case['requests']), 1)
        self.assertTrue(case['requests'][0].endswith('/web/engine/a.wasm'))
        self.assertRegex(case['keys'][0], '/web/engine/a[.]wasm[?]v=[0-9a-f]{64}$')

    def test_missing_file_comes_from_the_site_and_is_cached_under_this_origin(self):
        case = self.out['fallback']
        self.assertEqual(case['body'], 'site bytes')
        self.assertEqual(case['requests'][1], f'GET {SITE}model/b.onnx')
        self.assertEqual(len(case['keys']), 1)
        self.assertIn('/web/engine/model/b.onnx?v=', case['keys'][0])
        self.assertEqual(self.out['unreachable']['requests'][1], f'GET {SITE}b.onnx')

    def test_hash_mismatch_is_an_error_and_nothing_is_cached(self):
        self.assertIn('does not match its SHA-256', self.out['mismatch']['error'])
        self.assertEqual(self.out['mismatch']['keys'], [])
        self.assertIn('no SHA-256', self.out['unpinned']['error'])

    def test_build_json_digests_read_crlf_as_lf(self):
        self.assertEqual(self.out['lines'], 'line one\r\nline two')

    def test_cache_is_reused_and_a_new_version_replaces_the_old(self):
        case = self.out['reuse']
        self.assertEqual(case['body'], 'first')
        self.assertEqual(case['requests'], [])
        self.assertEqual(len(case['keys']), 1)

    def test_module_missing_beside_a_local_manifest_comes_from_the_site(self):
        self.assertEqual(self.out['module']['text'], 'export default 7;')
        self.assertEqual(self.out['module']['requests'][-1], f'GET {SITE}ort/x.mjs')

    def test_a_pinned_file_is_keyed_by_its_digest_not_its_version(self):
        self.assertTrue(self.out['digest_key'])

    def test_a_failing_body_stream_falls_back_to_a_whole_download(self):
        self.assertEqual(self.out['stream_fallback'], {'body': 'streamed', 'requests': 3, 'reset': True})

    def test_a_local_manifest_without_pins_borrows_the_sites_for_the_same_build(self):
        self.assertEqual(self.out['legacy_pins'], [['a', True], ['b', True], ['c', True]])
        self.assertEqual(self.out['legacy_pins_kept'], ['a', 'b', 'c'])

    def test_manifest_falls_back_to_the_site(self):
        self.assertEqual(self.out['json'], {'data': {'version': 3}, 'local': False})
        self.assertEqual(self.out['json_offline'], {'data': {'version': 3}, 'local': False})

    def test_status_and_install(self):
        case = self.out['status']
        self.assertEqual(case['before'], {'state': 'missing', 'bytes': 8})
        self.assertEqual(case['after'], {'state': 'cached'})
        self.assertEqual(case['local'], {'state': 'local'})
        self.assertEqual(case['last'], 1)
        self.assertEqual(self.out['uncached'], {'state': 'uncached'})
        self.assertEqual(self.out['uncached_unpublished'], 'not on site')
        self.assertEqual(self.out['partial']['state'], 'missing')
        self.assertEqual(self.out['partial']['bytes'], 4)
        self.assertEqual(self.out['no_head'], {'here': 'local', 'partial': 'missing'})

    def test_every_engine_lists_pinned_files_and_downloads_them_from_the_site(self):
        self.assertEqual(sorted(self.out['engines']), ['bubble', 'drip', 'seal', 'shrimp', 'six', 'strix'])
        for name, engine in self.out['engines'].items():
            with self.subTest(engine=name):
                self.assertTrue(engine['files'])
                self.assertTrue(all(f['pinned'] and not f['local'] for f in engine['files']))
                self.assertEqual(engine['cached'], 'cached')
                fetched = {r.split(' ')[1][len(SITE):] for r in engine['downloads'] if r.startswith(f'GET {SITE}')}
                self.assertLessEqual(fetched, {f['path'] for f in engine['files']})


    def test_strix_reads_its_network_list_again_after_starting_without_one(self):
        self.assertEqual(self.out['strix_retry'], {'files': ['strix/strix.wasm', 'strix/net.safetensors'], 'checkpoints': ['net']})
        self.assertEqual(self.out['strix_chosen'], ['strix/strix.wasm', 'strix/net.safetensors'])


    def test_six_fills_its_checkpoints_when_it_reads_the_manifest(self):
        self.assertEqual(self.out['six_retry'], ['gen-2', 'gen-1'])
        self.assertEqual(self.out['six_chosen'], ['six/networks/gen-1.onnx'])


    def test_an_interrupted_download_resumes_after_its_stored_part(self):
        case = self.out['resume']
        self.assertEqual((case['first'], case['parts'], case['offline'], case['ignored']), ('Failed to fetch', 1, [True, 1], [True, 1]))
        self.assertEqual(case['ranges'], [f'bytes={4 * 2 ** 20}-'] * 2)
        self.assertTrue(case['same'])
        self.assertEqual(case['start'], [[4 * 2 ** 20, 0], [int(4.5 * 2 ** 20), 5 * 2 ** 20]])
        self.assertEqual(case['keys'], 1)

    def test_a_download_that_goes_quiet_stops_with_an_error(self):
        self.assertEqual(self.out['idle'], {'error': 'quiet.onnx: the download stalled', 'requests': 2, 'reported': [[0, 0, 10]]})

    def test_a_request_without_an_answer_stops_with_an_error(self):
        self.assertEqual(self.out['unanswered'], {'error': f'silent.onnx: no answer in 0.03 s ({SITE}silent.onnx)'})
        self.assertEqual(self.out['unfinished'], {'error': 'slow.json: no answer in 0.03 s'})
        self.assertEqual(self.out['whole_stalls'], {'error': 'halted.onnx: no answer in 0.03 s'})

    def test_assets_query_is_honoured_only_on_a_loopback_page(self):
        self.assertEqual(self.out['override'], {'public': SITE, 'loopback': 'https://other.example/engine/'})


    def test_files_the_site_does_not_publish_need_a_local_build(self):
        self.assertEqual(self.out['unpublished'], {'manifest': 'not on site', 'status': 'not on site', 'seal': 'not on site',
                                                   'strix': 'not on site', 'unreachable': 'error'})

    def test_engine_lists_leave_out_engines_whose_files_no_origin_has(self):
        self.assertEqual(self.out['offered'], ['browser:bubble', 'browser:strix', 'browser:six', 'six'])


class Loaders(unittest.TestCase):
    def test_engine_loaders_fetch_through_assets(self):
        for name in LOADERS:
            with self.subTest(module=name):
                text = (ENGINE/name).read_text(encoding='utf-8')
                self.assertRegex(text, r"from '\.\.?/assets\.mjs'")
                self.assertNotIn('fetch(', text)
                self.assertNotIn('caches.', text)
                self.assertIsNone(re.search(r"new Worker\(new URL", text))


if __name__ == '__main__':
    unittest.main()
