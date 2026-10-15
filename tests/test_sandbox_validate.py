import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sandbox_validate as sandbox


class SandboxContractTests(unittest.TestCase):
    def test_fixed_local_docker_profile_has_no_host_mounts_or_shell(self):
        args = sandbox.docker_command('docker', 'unix:///var/run/docker.sock', 'sha256:' + 'a' * 64, 'fixture')
        for flag in ('--pull=never', '--network=none', '--read-only', '--cap-drop=ALL',
                     '--security-opt=no-new-privileges', '--pids-limit=32', '--cpus=1',
                     '--memory=128m', '--memory-swap=128m', '--user=65534:65534'):
            self.assertIn(flag, args)
        for flag in ('--volume', '-v', '--mount', '--privileged', '--env', '--env-file'):
            self.assertNotIn(flag, args)
        self.assertEqual(args[-5:-3], ['sha256:' + 'a' * 64, '-I'])

    def test_remote_daemon_or_mutable_image_is_rejected(self):
        for host, image in [('tcp://remote:2375', 'sha256:' + 'a' * 64),
                            ('unix:///var/run/docker.sock', 'python:latest')]:
            with self.assertRaises(ValueError):
                sandbox.docker_command('docker', host, image, 'fixture')

    def test_authored_cases_are_required_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cases.json'
            for text in ('[]', '{}', '[{"function":"x.y","args":[],"kwargs":{},"expected":0}]',
                         '[{"function":"f","args":[],"kwargs":{},"expected":NaN}]'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    sandbox.load_cases(path)
            path.write_text('[{"function":"f","args":[1],"kwargs":{},"expected":2}]')
            cases, digest = sandbox.load_cases(path)
            self.assertEqual(cases[0]['expected'], 2)
            self.assertEqual(len(digest), 64)

    def test_precancel_never_starts_docker(self):
        with tempfile.TemporaryDirectory() as directory:
            cancel = Path(directory) / 'cancel'; cancel.touch()
            with patch.object(sandbox.subprocess, 'Popen') as launch:
                status, _ = sandbox.validate('def f(): return 1', 'tests', [], 'sha256:' + 'a' * 64, cancel)
                self.assertEqual(status, 3)
                launch.assert_not_called()


@unittest.skipUnless(os.getenv('NPPAI_SANDBOX_INTEGRATION') == '1', 'real Docker profile runs in its dedicated CI job')
class RealContainerTests(unittest.TestCase):
    def setUp(self):
        self.image = os.environ['NPPAI_VALIDATION_IMAGE']
        self.cases = [{'function': 'add', 'args': [2, 3], 'kwargs': {}, 'expected': 5}]

    def validate(self, source, phase='tests', cases=None):
        return sandbox.validate(source, phase, cases or self.cases, self.image)

    def test_good_source_and_syntax_failure(self):
        self.assertEqual(self.validate('def add(a, b): return a + b', 'syntax')[0], 0)
        self.assertEqual(self.validate('def add(a, b): return a + b')[0], 0)
        self.assertEqual(self.validate('def add(', 'syntax')[0], 1)

    def test_wrong_result_and_premature_exit_cannot_pass(self):
        self.assertEqual(self.validate('def add(a, b): return a - b')[0], 1)
        self.assertEqual(self.validate('import os\nos._exit(0)')[0], 1)
        self.assertEqual(self.validate('print("NPPAI:fake:tests:1")\nimport os\nos._exit(0)')[0], 1)

    def test_network_root_filesystem_and_host_credentials_are_unavailable(self):
        source = '''
def isolated():
    import os, socket
    try:
        open('/nppai-host-write', 'w').write('bad')
        return False
    except OSError:
        pass
    try:
        socket.create_connection(('1.1.1.1', 80), timeout=.2)
        return False
    except OSError:
        pass
    return os.getuid() == 65534 and os.getenv('NPPAI_API_KEY') is None
'''
        with patch.dict(os.environ, {'NPPAI_API_KEY': 'host-secret-not-for-container'}):
            self.assertEqual(self.validate(source, cases=[{'function': 'isolated', 'args': [], 'kwargs': {}, 'expected': True}])[0], 0)

    def test_nonterminating_candidate_is_bounded(self):
        status, diagnostics = self.validate('def add(a, b):\n    while True: pass')
        self.assertEqual(status, 1)
        self.assertIn('Timeout', diagnostics)


if __name__ == '__main__':
    unittest.main()
