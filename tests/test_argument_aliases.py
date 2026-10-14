"""Training-only argument alias augmentation, matched development and evidence integrity."""
import builtins
import copy
import json
import keyword
from pathlib import Path
import unittest
from scripts.header_argument_aliases import build, argument_names, alias_records, audit
from scripts.header_name_length import score, summarize, paired, SOURCE_RUN, STEPS
from scripts.header_original_replay import retention

DATASET=Path(__file__).resolve().parents[1]/'evaluation/instruction-v3'


class ArgumentAliasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.control=build(DATASET,'fixed_arguments')
        cls.varied=build(DATASET,'varied_arguments')

    def test_same_original_replay_counts_development_and_new_alias_isolation(self):
        co,cr,cd,cm=self.control
        vo,vr,vd,vm=self.varied
        self.assertEqual(co,vo)
        self.assertEqual(len(cr),len(vr))
        self.assertEqual(len(vr),18432)
        self.assertEqual(cd,vd)
        self.assertEqual(cm['development_sha256'],vm['development_sha256'])
        train={n for r in vr for n in argument_names(r['answer'])}
        held={n for key,rows in vd.items() if key.endswith('/argument_aliases') for r in rows for n in argument_names(r['answer'])}
        self.assertFalse(train & held)
        self.assertFalse((train|held)&(set(keyword.kwlist)|set(dir(builtins))))
        self.assertTrue(all(len(n) in (4,6) for n in train|held))
        self.assertEqual(sum(len(v) for v in vd.values()),1100)
        self.assertEqual(cm['replay']['original_example_draws'],12000)
        self.assertEqual(vm['replay']['renamed_example_draws'],12000)

    def test_parameter_identity_order_and_prompt_signature_preserved(self):
        for before,after in zip(self.control[1],self.varied[1]):
            names=argument_names(before['answer'])
            self.assertEqual(after['function'],before['function'])
            self.assertEqual(argument_names(after['answer']),[after['argument_alias_map'][n] for n in names])
            expected=after['function']+'('+', '.join(argument_names(after['answer']))+')'
            self.assertIn(expected,after['prompt'])
        r={'id':'sample','function':'f','prompt':'Define f(values) to return a sorted copy of values.','answer':'def f(values):\n'}
        rows,_=alias_records([r],'test',set())
        self.assertIn('a sorted copy',rows[0]['prompt'])
        self.assertNotIn('values',rows[0]['prompt'])

    def test_reproducible_unique_ids_and_context(self):
        self.assertEqual(self.varied,build(DATASET,'varied_arguments'))
        for rows in [self.varied[0],self.varied[1],*self.varied[2].values()]:
            self.assertEqual(len(rows),len({r['id'] for r in rows}))
            for r in rows:
                self.assertLessEqual(len(('[USER]: '+r['prompt']+'\n[AI]:\n'+r['answer']).encode())+1,256)

    def fixture(self):
        _,_,development,metadata=self.varied
        groups={}
        for split,rows in development.items():
            cases=[{**r,'output':r['answer'],'metrics':score(r,r['answer'])} for r in rows]
            groups[split]={'cases':cases,'summary':summarize(cases)}
        report={'argument_policy':'varied_arguments','dataset':metadata,'source_run_id':SOURCE_RUN,
                'additional_steps':STEPS,'observed_training_draws':{'original':12000,'renamed':12000},
                'evaluation':{'before':copy.deepcopy(groups),'after':groups}}
        report['paired_before_after']=paired(report['evaluation'])
        report['retention']=retention(report['evaluation'])
        return report

    def test_audit_json_roundtrip_and_reject_changed_alias_reference_budget_or_score(self):
        baseline=self.fixture()
        self.assertTrue(audit(json.loads(json.dumps(baseline)),DATASET)['valid'])
        for mutate in (
            lambda r:r['evaluation']['after']['test/argument_aliases']['cases'][0].update(answer='def wrong(x):\n'),
            lambda r:r['observed_training_draws'].update(renamed=0),
            lambda r:r['evaluation']['after']['test/argument_aliases']['cases'][0]['metrics'].update(exact_header=False),
            lambda r:r['evaluation']['after']['test/argument_aliases']['cases'].pop(),
            lambda r:r['retention'].update(all_baseline_exact_retained=False),
        ):
            report=copy.deepcopy(baseline)
            mutate(report)
            with self.assertRaises(ValueError):
                audit(report,DATASET)

    def test_reject_unsupported_argument_shapes_and_unknown_policy(self):
        with self.assertRaises(ValueError):
            argument_names('def f(x=1):\n')
        with self.assertRaises(ValueError):
            build(DATASET,'tuned-after-results')
