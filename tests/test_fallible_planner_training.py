import copy
import json
from pathlib import Path
import unittest
import zipfile

from scripts.fallible_planner_training import training_priors
from scripts.stage_failure_diagnostic import MODELS

ROOT=Path(__file__).resolve().parents[1]


class TrainingProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records=json.loads((ROOT/'evaluation/staged-protocol-v1/train.json').read_bytes())
        with zipfile.ZipFile(ROOT/'docs/evidence/staged-protocol/all-stage-records.zip') as z:
            cls.cases=json.loads(z.read('initial/training.json'))['cases']
        cls.identity='sha256:'+MODELS['staged_mix']

    def test_actual_priors_and_authored_targets_are_distinct(self):
        rows=training_priors(self.records,self.cases,self.identity)
        self.assertEqual(sum(r['source']=='observed_prior' for r in rows),83)
        self.assertEqual(sum(r['source']=='authored_fallback_incomplete_planner' for r in rows),13)
        expected={r['id']:r['answer'] for r in self.records}
        for row in rows:
            self.assertEqual(row['target'],expected[row['id']])
            self.assertEqual(row['target_origin'],'authored_training_answer')

    def test_nontraining_or_changed_reference_is_rejected(self):
        records=copy.deepcopy(self.records);records[0]['id']='validation-injected'
        with self.assertRaises(ValueError):training_priors(records,self.cases,self.identity)
        cases=copy.deepcopy(self.cases);cases[0]['record']['answer']='generated target'
        with self.assertRaises(ValueError):training_priors(self.records,cases,self.identity)

    def test_mismatched_planner_and_prompt_are_rejected(self):
        with self.assertRaises(ValueError):training_priors(self.records,self.cases,'sha256:wrong')
        cases=copy.deepcopy(self.cases)
        case=next(c for c in cases if any(s['stage']=='code' for s in c['autonomous']['trace']['stages']))
        code=next(s for s in case['autonomous']['trace']['stages'] if s['stage']=='code')
        code['request']['prompt']='unrelated input'
        with self.assertRaises(ValueError):training_priors(self.records,cases,self.identity)


if __name__=='__main__':unittest.main()
