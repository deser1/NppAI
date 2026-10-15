import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import resampled_name_training as experiment

DATASET = Path(__file__).resolve().parents[1] / 'evaluation/instruction-v3'


class ResampledNameTests(unittest.TestCase):
    def record(self):
        return {'id':'train-fixture','function':'foo','prompt':'Implement foo(x) returning x.',
                'answer':'def foo(x):\n    return x + "foo"\n'}

    def test_rename_preserves_strings_arguments_and_body(self):
        r=experiment.rename(self.record(),'nc_1234abcd')
        self.assertEqual(r['prompt'],'Implement nc_1234abcd(x) returning x.')
        self.assertEqual(r['answer'],'def nc_1234abcd(x):\n    return x + "foo"\n')
        with self.assertRaisesRegex(ValueError,'overflowing'):
            experiment.frame({**r,'prompt':'x'*256})

    def test_training_loader_never_opens_development_splits(self):
        real=experiment.read_split
        calls=[]
        def read(root,split):
            calls.append(split)
            return real(root,split)
        with patch.object(experiment,'read_split',side_effect=read):
            self.assertEqual(len(experiment.training_data(DATASET)),6144)
        self.assertEqual(calls,['train'])

    def test_matched_source_order_exposure_and_fresh_aliases(self):
        rows=[{**self.record(),'id':str(i)} for i in range(3)]
        fixed=experiment.schedule_evidence(rows,'fixed',steps=8)
        fresh=experiment.schedule_evidence(rows,'resampled',steps=8)
        self.assertEqual(fixed['order_sha256'],fresh['order_sha256'])
        self.assertEqual(fixed['response_targets'],fresh['response_targets'])
        self.assertEqual(fresh['draws'],{'original':16,'renamed':16})
        self.assertEqual(fixed['unique_aliases'],1)
        self.assertEqual(fresh['unique_aliases'],16)
        self.assertNotEqual(fixed['content_sha256'],fresh['content_sha256'])

    def test_raw_score_does_not_credit_other_names_or_invalid_bodies(self):
        r=self.record()
        self.assertTrue(experiment.score(r,r['answer'])['exact_response'])
        wrong=r['answer'].replace('def foo','def bar')
        self.assertFalse(experiment.score(r,wrong)['requested_signature'])
        self.assertFalse(experiment.score(r,'def foo(x):\n broken !')['requested_name'])
        body='def foo(x):\n    return 7\n'
        self.assertTrue(experiment.score(r,body)['requested_signature'])
        self.assertFalse(experiment.score(r,body)['reference_ast'])

    @unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch required for actual tensors')
    def test_full_response_mask_includes_names_body_and_eos(self):
        r=self.record();prefix,raw=experiment.frame(r)
        x,y=experiment.sample(r)
        self.assertEqual(x.tolist(),list(raw[:-1]))
        self.assertEqual(y[:len(prefix)-1].tolist(),[-100]*(len(prefix)-1))
        self.assertEqual(y[len(prefix)-1:].tolist(),list(r['answer'].encode())+[0])

    def test_complete_draw_schedule_and_disjoint_development(self):
        rows=experiment.training_data(DATASET)
        development=experiment.development_data(DATASET)
        a=experiment.schedule_evidence(rows,'fixed')
        b=experiment.schedule_evidence(rows,'resampled')
        self.assertEqual(a['order_sha256'],b['order_sha256'])
        self.assertEqual(a['response_targets'],b['response_targets'])
        self.assertEqual(b['unique_aliases'],2048)
        self.assertEqual(b['draws'],{'original':2048,'renamed':2048})
        names={r['function'] for r in rows}
        self.assertFalse(names & {r['function'] for rs in development.values() for r in rs})
        self.assertEqual(sum(map(len,development.values())),88)

    def test_audit_recomputes_raw_scores_and_training_order(self):
        rows=experiment.training_data(DATASET)
        groups=experiment.development_data(DATASET)
        cases={g:[{'id':r['id'],'output':r['answer'],'metrics':experiment.score(r,r['answer'])}
                  for r in rs] for g,rs in groups.items()}
        report={'policy':'fixed','steps':experiment.STEPS,'source_run_id':experiment.SOURCE_RUN,
                'source_report_sha256':experiment.REPORT_SHA256,
                'initial_model_id':experiment.SOURCE_MODEL_ID,'optimizer':experiment.OPTIMIZER,
                'training_sha256':experiment.digest(experiment.encoded(rows)),
                'development_sha256':experiment.digest(experiment.encoded(groups)),
                'schedule':experiment.schedule_evidence(rows,'fixed'),
                'before':copy.deepcopy(cases),'after':copy.deepcopy(cases),
                'summary':{s:{g:experiment.summary(c) for g,c in cases.items()} for s in ('before','after')},
                'retention':experiment.retention(cases,cases), 'exported_weights_equal':True,
                'loss_endpoints':[1.0,0.5]}
        self.assertTrue(experiment.audit(report,DATASET)['valid'])
        for mutate in (
                lambda r:r['schedule'].update(order_sha256='changed'),
                lambda r:r['after']['test/original'][0].update(output='invalid'),
                lambda r:r['retention']['test/original']['exact_response'].update(lost=1),
                lambda r:r.update(initial_model_id='different')):
            changed=copy.deepcopy(report);mutate(changed)
            with self.assertRaises(ValueError):experiment.audit(changed,DATASET)


if __name__=='__main__':
    unittest.main()
