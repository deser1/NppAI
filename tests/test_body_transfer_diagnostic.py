import ast
import unittest
from scripts.diagnose_body_transfer import probes, probe_sha
from scripts.function_body_training import load_bank, score


class BodyTransferTests(unittest.TestCase):
    def test_full_orthogonal_grid_and_known_training_control(self):
        rows = probes()
        self.assertEqual(len(rows), 96)
        self.assertEqual(len({r['id'] for r in rows}), 96)
        self.assertEqual(len({r['cell'] for r in rows}), 8)
        for cell in {r['cell'] for r in rows}:
            self.assertEqual(sum(r['cell'] == cell for r in rows), 12)
        _, train, _ = load_bank()
        known = {r['request']['prompt']: r['answer'] for r in train}
        for row in rows:
            self.assertTrue(all(score(row['answer'], row).values()))
            self.assertEqual({c['function'] for c in row['cases']}, {row['function']})
            self.assertNotIn(row['answer'], row['request']['prompt'])
            if row['cell'] == 'name0-args0-wording0':
                self.assertEqual(known[row['request']['prompt']], row['answer'])
        self.assertEqual(probe_sha(rows), probe_sha(probes()))

    def test_each_axis_changes_only_its_declared_input(self):
        rows = [r for r in probes() if r['family'] == 'absolute_gap' and r['language'] == 'en']
        cells = {r['cell']: r for r in rows}
        base = cells['name0-args0-wording0']
        name = cells['name1-args0-wording0']
        args = cells['name0-args1-wording0']
        text = cells['name0-args0-wording1']
        self.assertEqual(base['arguments'], name['arguments'])
        self.assertEqual(base['function'], args['function'])
        self.assertEqual(base['function'], text['function'])
        self.assertEqual(base['arguments'], text['arguments'])
        self.assertEqual(base['request']['prompt'].split('. ',1)[1], name['request']['prompt'].split('. ',1)[1])
        self.assertEqual(base['request']['prompt'].split('. ',1)[1], args['request']['prompt'].split('. ',1)[1])
        self.assertNotEqual(base['request']['prompt'].split('. ',1)[1], text['request']['prompt'].split('. ',1)[1])
        for row in rows:
            self.assertEqual(row['cases'][0]['args'], base['cases'][0]['args'])
            self.assertEqual(row['cases'][0]['expected'], base['cases'][0]['expected'])
