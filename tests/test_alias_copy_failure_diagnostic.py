import unittest
from pathlib import Path
from scripts import alias_copy_failure_diagnostic as diagnostic


class AliasDiagnosticTests(unittest.TestCase):
    def case(self, name='aa', output=None, start=None, exit_at=None):
        row = {'id': 'sample', 'function': name, 'prompt': f'Implement {name}(x).',
               'answer': f'def {name}(x):\n    return x\n'}
        start = diagnostic.source_starts(row['prompt'], name)[0] if start is None else start
        raw = output if output is not None else row['answer'].encode()
        count = len(name.encode()) if exit_at is None else exit_at
        frame = ('[USER]: '+row['prompt']+'\n[AI]:\n').encode()
        events = [{'response_query': 4+i, 'cursor': start+i, 'kind': 'copy',
                   'source_byte': frame[start+i], 'base_byte': 97,
                   'exit_probability': .1, 'chosen': frame[start+i]} for i in range(count)]
        events.append({'response_query': 4+count, 'cursor': start+count, 'kind': 'exit',
                       'source_byte': frame[start+count], 'base_byte': raw[4+count],
                       'exit_probability': .9, 'chosen': raw[4+count]})
        return {'output': raw.decode(), 'output_bytes': list(raw), 'events': events}, row

    def test_exact_unicode_and_repeated_bytes(self):
        for name in ('aa', 'føø_2'):
            case, row = self.case(name)
            result = diagnostic.diagnose(case, row)
            self.assertEqual(result['divergence_region'], 'exact_response')
            self.assertEqual(result['mechanism_observations'], [])

    def test_span_boundaries_and_utf8_offsets(self):
        prompt = 'é aa aaa aa_2 aa'
        self.assertEqual(diagnostic.source_starts(prompt, 'aa'), [11, 23])

    def test_wrong_start_and_first_output_byte(self):
        case, row = self.case(output=b'def Im(x):\n', start=8)
        result = diagnostic.diagnose(case, row)
        self.assertEqual(result['first_output_divergence'], 4)
        self.assertIn('wrong_source_start', result['mechanism_observations'])

    def test_early_and_late_exit(self):
        case, row = self.case(output=b'def a(x):\n', exit_at=1)
        self.assertIn('premature_exit', diagnostic.diagnose(case, row)['mechanism_observations'])
        case, row = self.case(output=b'def aa(x):\n', exit_at=3)
        # Copying '(' is an exit error even when emitted text is correct so far.
        self.assertIn('late_exit', diagnostic.diagnose(case, row)['mechanism_observations'])

    def test_tamper_rejected(self):
        case, row = self.case()
        case['events'][1]['cursor'] -= 1
        with self.assertRaises(ValueError):
            diagnostic.diagnose(case, row)

    def test_static_parse_does_not_execute(self):
        case = {'source': "raise RuntimeError('never execute')\ndef foo(): pass"}
        self.assertTrue(diagnostic.functional_record(case)['static_parse']['valid'])
        self.assertFalse(diagnostic.functional_record({'source': 'def ('})['static_parse']['valid'])

    def test_retained_reconstruction_is_deterministic(self):
        directory = Path(__file__).resolve().parents[1]/'docs/evidence/protected-sequential'
        first = diagnostic.build(directory)
        self.assertEqual(first, diagnostic.build(directory))
        dev = [v for k, v in first['summary'].items() if k.startswith(('test/', 'validation/'))]
        self.assertEqual(sum(v['total'] for v in dev), 132)
        self.assertEqual(sum(v['divergence_regions'].get('exact_response', 0) for v in dev), 27)
        self.assertEqual(len(first['functional']), 6)
