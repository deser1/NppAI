"""Replay only train examples; count retention by identity, not aggregate accuracy."""
import copy
from pathlib import Path
import random
import unittest
from scripts.header_original_replay import pools, metadata, draw_batch, retention, audit
from scripts.header_name_length import score, summarize, paired, SOURCE_RUN, STEPS

DATASET = Path(__file__).resolve().parents[1] / 'evaluation/instruction-v3'


class HeaderOriginalReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original, cls.renamed, cls.development, cls.base = pools(DATASET)

    def test_training_only_pools_and_distinct_ids(self):
        self.assertEqual(len(self.original), 6144)
        self.assertEqual(len(self.renamed), 18432)
        records = self.original + self.renamed
        self.assertEqual(len({r['id'] for r in records}), len(records))
        names = {r['function'] for r in records}
        for rows in self.development.values():
            self.assertFalse(names & {r['function'] for r in rows})
        self.assertTrue(all(r['source_id'].startswith('train-') for r in self.original))

    def test_balanced_batches_budget_and_repeatable_selection(self):
        first, second = random.Random(42), random.Random(42)
        for _ in range(20):
            batch = draw_batch(self.original, self.renamed, 'half_original', first)
            self.assertEqual(batch, draw_batch(self.original, self.renamed, 'half_original', second))
            self.assertEqual([label for label, _ in batch].count('original'), 2)
            self.assertEqual([label for label, _ in batch].count('renamed'), 2)
            self.assertEqual(len({r['id'] for _, r in batch}), 4)
        self.assertTrue(all(label == 'renamed' for label, _ in draw_batch(self.original,self.renamed,'no_replay',first)))
        self.assertEqual(metadata(self.original,self.renamed,'half_original')['original_example_draws'],12000)
        self.assertEqual(metadata(self.original,self.renamed,'no_replay')['renamed_example_draws'],24000)
        with self.assertRaises(ValueError):
            draw_batch(self.original,self.renamed,'unknown',first)

    def report(self, policy='half_original'):
        groups = {}
        for split, records in self.development.items():
            cases=[{**r,'output':r['answer'],'metrics':score(r,r['answer'])} for r in records]
            groups[split]={'cases':cases,'summary':summarize(cases)}
        replay=metadata(self.original,self.renamed,policy)
        report={'arm':'mixed_lengths','dataset':self.base,'source_run_id':SOURCE_RUN,
                'additional_steps':STEPS,'evaluation':{'before':copy.deepcopy(groups),'after':groups},
                'replay':replay,'observed_training_draws':{'original':replay['original_example_draws'],'renamed':replay['renamed_example_draws']}}
        report['paired_before_after']=paired(report['evaluation'])
        report['retention']=retention(report['evaluation'])
        return report

    def test_audit_policy_draw_and_retention_tampering(self):
        baseline=self.report()
        self.assertTrue(audit(baseline,DATASET)['valid'])
        for mutate in (
            lambda r:r['replay'].update(original_pool_sha256='wrong'),
            lambda r:r['observed_training_draws'].update(original=0),
            lambda r:r['retention'].update(all_baseline_exact_retained=False),
            lambda r:r['evaluation']['after']['test/original']['cases'][0]['metrics'].update(exact_header=False),
        ):
            report=copy.deepcopy(baseline)
            mutate(report)
            with self.assertRaises(ValueError):
                audit(report,DATASET)

    def test_same_total_hits_can_hide_a_lost_baseline_case(self):
        report=self.report()
        before=report['evaluation']['before']['test/original']['cases']
        after=report['evaluation']['after']['test/original']['cases']
        before[0]['output']=''
        before[0]['metrics']=score(before[0],'')
        after[1]['output']=''
        after[1]['metrics']=score(after[1],'')
        # Overall totals match, but identities expose one lost and one new hit.
        self.assertEqual(sum(c['metrics']['exact_header'] for c in before),sum(c['metrics']['exact_header'] for c in after))
        result=retention(report['evaluation'])
        self.assertFalse(result['all_baseline_exact_retained'])
        self.assertEqual(result['test/original']['lost_exact'],1)
        self.assertEqual(result['test/original']['new_exact'],1)

    def test_negative_retention_is_valid_evidence(self):
        report=self.report('no_replay')
        for split,group in report['evaluation']['after'].items():
            for case in group['cases']:
                case['output']=''
                case['metrics']=score(case,'')
            group['summary']=summarize(group['cases'])
        report['paired_before_after']=paired(report['evaluation'])
        report['retention']=retention(report['evaluation'])
        evidence=audit(report,DATASET)
        self.assertTrue(evidence['valid'])
        self.assertFalse(evidence['retention']['all_baseline_exact_retained'])
