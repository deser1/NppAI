"""Reject forged fresh evidence and distinguish names from signatures."""
import copy
from pathlib import Path
import unittest
from scripts.evaluate_fresh_probe import load_dataset, signature, score, aggregate, audit, SOURCE_RUN

DATASET = Path(__file__).resolve().parents[1] / 'evaluation/fresh-probe-v1'


class FreshEvaluationTests(unittest.TestCase):
    def report(self):
        records, manifest = load_dataset(DATASET)
        cases = []
        for record in records:
            header = record['answer'].splitlines(keepends=True)[0]
            case = {k: record[k] for k in ('id', 'task_id', 'family', 'function', 'wording', 'prompt')}
            case.update(reference=record['answer'], name_length=len(record['function']), generated_header=header,
                        header_accepted=True, header_exact_reference=True, header_requested_signature=True,
                        staged=score(record, record['answer']), single_model=score(record, ''))
            cases.append(case)
        return {'dataset_manifest': manifest, 'source_run_id': SOURCE_RUN, 'policy': 'four_forms',
                'cases': cases, 'summary': aggregate(cases)}

    def test_name_match_is_not_signature_match(self):
        record = {'function': 'foo', 'answer': 'def foo(x):\n    return x\n'}
        scored = score(record, 'def foo(y):\n    return y\n')
        self.assertTrue(scored['requested_function'])
        self.assertFalse(scored['requested_signature'])
        self.assertIsNone(signature('def foo(x):'))

    def test_audit_recomputes_metrics_and_paired_groups(self):
        report = self.report()
        self.assertTrue(audit(report, DATASET)['valid'])
        self.assertEqual(report['summary']['groups']['wording']['brief']['total'], 36)
        self.assertEqual(report['summary']['paired_to_known_format']['brief']['staged']['exact_response']['both'], 36)
        self.assertEqual(report['summary']['task_instances'], 36)
        self.assertEqual(report['summary']['groups']['name_length']['15']['total'], 48)

    def test_reject_tampering_duplicates_metadata_and_source(self):
        baseline = self.report()
        for mutate in (
            lambda r: r['cases'][0]['staged'].update(exact_response=False),
            lambda r: r['cases'][0].update(prompt='different'),
            lambda r: r['cases'][0].update(header_accepted=False),
            lambda r: r['cases'].__setitem__(1, r['cases'][0]),
            lambda r: r['summary']['overall'].update(total=0),
            lambda r: r.update(source_run_id=0),
        ):
            report = copy.deepcopy(baseline)
            mutate(report)
            with self.assertRaises(ValueError):
                audit(report, DATASET)

    def test_negative_quality_is_valid_evidence(self):
        report = self.report()
        records, _ = load_dataset(DATASET)
        for record, case in zip(records, report['cases']):
            case.update(generated_header='', header_accepted=False, header_exact_reference=False,
                        header_requested_signature=False, staged=score(record, ''))
        report['summary'] = aggregate(report['cases'])
        result = audit(report, DATASET)
        self.assertTrue(result['valid'])
        self.assertFalse(result['staged_exact_present_in_each_wording'])
