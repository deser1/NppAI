import json
from pathlib import Path
import shutil
import tempfile
import unittest
from scripts import start_transfer_diagnostic as diagnostic

DIRECTORY = diagnostic.ROOT/'docs/evidence/start-transfer'


class SourceLocationTests(unittest.TestCase):
    def test_inside_word_and_repeated_requested_name(self):
        self.assertEqual(diagnostic.source_location('Define aa(x).', 11, 'aa'),
                         {'category': 'inside_other_identifier', 'token': 'Define', 'token_offset': 3})
        for cursor in (8, 11):
            self.assertEqual(diagnostic.source_location('aa aa', cursor, 'aa')['category'], 'requested_start')
        self.assertEqual(diagnostic.source_location('aa_2(x)', 10, 'aa_2')['category'], 'inside_requested_name')

    def test_utf8_boundary_and_prompt_end(self):
        prompt = 'é føø_2(x).'
        start = len('[USER]: é '.encode())
        self.assertEqual(diagnostic.source_location(prompt, start, 'føø_2')['category'], 'requested_start')
        self.assertEqual(diagnostic.source_location(prompt, start+2, 'føø_2')['category'], 'utf8_continuation')
        end = len(('[USER]: '+prompt).encode())-1
        self.assertEqual(diagnostic.source_location(prompt, end, 'føø_2')['category'], 'punctuation')
        with self.assertRaises(ValueError):
            diagnostic.source_location(prompt, end+1, 'føø_2')

    def test_observed_copy_overrides_correct_colon(self):
        row = {'id': 'fixture', 'function': 'aa', 'prompt': 'Implement aa(x) in Python.',
               'answer': 'def aa(x):\n    return x\n'}
        raw = b'def aa(x) '
        events = [{'response_query': 4+i, 'cursor': 18+i, 'kind': 'copy',
                   'source_byte': byte, 'chosen': byte, 'base_byte': 58 if i == 5 else byte,
                   'exit_probability': .1} for i, byte in enumerate(b'aa(x) ')]
        case = {'output': raw.decode(), 'output_bytes': list(raw), 'events': events,
                'metrics': diagnostic.p.base.score(row, raw.decode())}
        result = diagnostic.classify(case, row)
        self.assertTrue(result['copy_overrides_correct_vocabulary'])
        self.assertEqual(result['divergence_region'], 'signature')
        self.assertEqual(result['expected_divergence_byte'], 58)
        self.assertEqual(result['first_divergence_event']['chosen'], 32)
        self.assertEqual(result['name_status'], 'complete_name_and_paren')


class RetainedTransferTests(unittest.TestCase):
    def test_reconstruction_and_specific_observations(self):
        self.assertTrue(diagnostic.audit(DIRECTORY))
        report = json.loads((DIRECTORY/'report.json').read_bytes())
        self.assertEqual(sum(v['development_aliases']['total'] for v in report['stages'].values()), 264)
        for stage in ('start_ce', 'start_margin'):
            group = report['stages'][stage]['development_aliases']
            self.assertEqual(group['correct_start_total'], 4)
            self.assertEqual(group['correct_start_copy_override'], 3)
            self.assertEqual(group['exact_response'], 1)
            self.assertEqual(sum(group['source_locations'].values()), 88)

    def test_rehashed_invented_observation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/'evidence'
            shutil.copytree(DIRECTORY, directory)
            report = json.loads((directory/'report.json').read_bytes())
            report['stages']['start_ce']['development_aliases']['correct_start_total'] = 88
            raw = diagnostic.p.quality.encoded(report)
            (directory/'report.json').write_bytes(raw)
            manifest = json.loads((directory/'manifest.json').read_bytes())
            manifest['report.json'] = diagnostic.p.quality.digest(raw)
            (directory/'manifest.json').write_bytes(diagnostic.p.quality.encoded(manifest))
            with self.assertRaisesRegex(ValueError, 'transfer evidence differs'):
                diagnostic.audit(directory)
