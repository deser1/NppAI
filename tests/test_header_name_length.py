"""Name length curriculum isolation, matched controls and audited greedy evidence."""
import copy
from pathlib import Path
import tempfile
import unittest
from scripts.header_name_length import build, read_sources, score, summarize, paired, audit, SOURCE_RUN, STEPS

DATASET = Path(__file__).resolve().parents[1] / 'evaluation/instruction-v3'


class HeaderNameLengthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.control = build(DATASET, 'short_control')
        cls.mixed = build(DATASET, 'mixed_lengths')

    def test_counts_disjoint_names_and_identical_development(self):
        short, development, sm = self.control
        mixed, md, mm = self.mixed
        self.assertEqual(len(short), 18432)
        self.assertEqual(len(mixed), len(short))
        self.assertEqual(development, md)
        self.assertEqual(sm['development_sha256'], mm['development_sha256'])
        self.assertNotEqual(sm['training_sha256'], mm['training_sha256'])
        self.assertEqual({len(r['function']) for r in short}, {9})
        self.assertEqual({len(r['function']) for r in mixed}, {9, 12, 15})
        for training in (short, mixed):
            names = {r['function'] for r in training}
            self.assertEqual(len(names), 1536)
            self.assertEqual(len({r['id'] for r in training}), len(training))
            for records in development.values():
                self.assertFalse(names & {r['function'] for r in records})
        self.assertEqual({length:sum(r['name_length'] == length for r in mixed) for length in (9,12,15)}, {9:6144,12:6144,15:6144})

    def test_reproducible_and_original_argument_lists_preserved(self):
        self.assertEqual(self.mixed, build(DATASET, 'mixed_lengths'))
        original, _ = read_sources(DATASET)
        source = {r['id']:r for r in original['train']}
        for record in self.mixed[0]:
            # source_id after wording augmentation refers to the renamed view.
            origin = source[record['source_id'].split('::name-')[0]]
            self.assertEqual(record['answer'].split('(',1)[1], origin['answer'].split('(',1)[1])
            self.assertTrue(record['function'].startswith('cp_'))
        for split in ('validation', 'test'):
            self.assertEqual([r['prompt'] for r in self.mixed[1][split+'/original']], [r['prompt'] for r in original[split]])

    def test_names_not_arguments_and_headers_not_bodies(self):
        record = {'function':'cp_example', 'answer':'def cp_example(x):\n'}
        metrics = score(record, 'def cp_example(y):\n')
        self.assertTrue(metrics['requested_name'])
        self.assertFalse(metrics['requested_signature'])
        self.assertFalse(score(record, 'def cp_example(x):\n    return x\n')['valid_header'])
        self.assertTrue(score(record, record['answer'])['exact_header'])

    def report(self):
        _, development, metadata = self.mixed
        groups = {}
        for split, records in development.items():
            cases = [{**r, 'output':r['answer'], 'metrics':score(r,r['answer'])} for r in records]
            groups[split] = {'cases':cases,'summary':summarize(cases)}
        report = {'arm':'mixed_lengths', 'dataset':metadata, 'source_run_id':SOURCE_RUN,
                  'additional_steps':STEPS,'evaluation':{'before':copy.deepcopy(groups),'after':groups}}
        report['paired_before_after'] = paired(report['evaluation'])
        return report

    def test_audit_pairing_and_tamper_rejection(self):
        baseline = self.report()
        self.assertTrue(audit(baseline,DATASET)['valid'])
        self.assertEqual(baseline['paired_before_after']['test/renamed']['exact_header']['both'],336)
        for mutate in (
            lambda r:r['evaluation']['after']['test/renamed']['cases'][0]['metrics'].update(exact_header=False),
            lambda r:r['evaluation']['after']['test/renamed']['cases'][0].update(prompt='tampered'),
            lambda r:r['evaluation']['after']['test/renamed']['cases'].pop(),
            lambda r:r['paired_before_after']['test/renamed']['exact_header'].update(both=0),
            lambda r:r.update(additional_steps=1),
        ):
            report=copy.deepcopy(baseline)
            mutate(report)
            with self.assertRaises(ValueError):
                audit(report,DATASET)

    def test_negative_result_is_valid_evidence(self):
        report=self.report()
        for stage in ('before','after'):
            for group in report['evaluation'][stage].values():
                for case in group['cases']:
                    case['output']=''
                    case['metrics']=score(case,'')
                group['summary']=summarize(group['cases'])
        report['paired_before_after']=paired(report['evaluation'])
        self.assertTrue(audit(report,DATASET)['valid'])

    def test_frozen_training_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            for name in ('manifest.json','train.json','validation.json','test.json'):
                (folder/name).write_bytes((DATASET/name).read_bytes())
            (folder/'train.json').write_bytes(b'[]')
            with self.assertRaises(ValueError):
                build(folder,'mixed_lengths')
