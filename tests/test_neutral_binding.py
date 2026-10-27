import copy
import importlib.util
import unittest
from unittest.mock import patch

from model_backends import GenerationRequest, format_prompt
from scripts.neutral_binding_data import banks, digest
from scripts import neutral_binding_training as experiment


class NeutralBindingTests(unittest.TestCase):
    def test_reviewed_bank_identity_and_neutral_names(self):
        manifest, train, dev = experiment.load_bank()
        self.assertEqual(manifest['counts'], dict(training=144, development=120, primary=24))
        self.assertEqual(manifest['training_sha256'], digest(train))
        families = {r['family'] for r in train}
        for name in ('task_aa', 'task_bb', 'task_cc'):
            self.assertEqual({r['family'] for r in train if r['function']==name}, families)
        self.assertEqual({r['function'] for r in train}, {'task_aa','task_bb','task_cc'})
        # Primary identifiers are absent from all optimizer targets, globally.
        self.assertFalse({'task_dd','ee','ff'} & {a for r in train for a in [r['function']]+r['arguments']})
        old_train = __import__('json').loads((experiment.ROOT/'evaluation/function-body-v1/train.json').read_bytes())
        self.assertFalse({r['request']['prompt'] for r in train} & {r['request']['prompt'] for r in old_train})
        self.assertEqual(sum(r['primary'] for r in dev), 24)
        self.assertEqual(len({r['id'] for r in dev}), 120)

    def test_matched_byte_positions_and_explicit_shift_controls(self):
        _, dev = banks()
        for family in {r['family'] for r in dev}:
            for language in ('en','pl'):
                rows = [r for r in dev if r['family']==family and r['language']==language]
                base = next(r for r in rows if r['cell']=='name0-args0-shift0')
                for row in rows:
                    text = format_prompt(GenerationRequest(**row['request']),256).encode()
                    self.assertEqual(len(text),256)
                    self.assertTrue(text.endswith(b'[AI]:\n'))
                    if not row['cell'].startswith('long-'):
                        self.assertEqual(len(row['function']),len(base['function']))
                        self.assertEqual(list(map(len,row['arguments'])),list(map(len,base['arguments'])))
                        original = format_prompt(GenerationRequest(**base['request']),256).encode()
                        self.assertEqual(text.index(row['function'].encode()), original.index(base['function'].encode())+row['leading'])
                        self.assertEqual(text.index(b'. '),original.index(b'. ')+row['leading'])
                    else:
                        self.assertGreater(len(row['request']['prompt'].encode()),len(base['request']['prompt'].encode()))

    def test_shared_draws_change_only_body_positions(self):
        m, train, _ = experiment.load_bank()
        draws = experiment.schedule(m,96,532,len(train))
        self.assertEqual(draws,experiment.schedule(m,96,532,len(train)))
        self.assertEqual(len(draws),1024)
        old, fact = list(range(96)),list(range(532))
        body = {(i,s):(i,s) for i in range(144) for s in (0,8,16)}
        for draw in draws:
            fixed = experiment.selected_samples('fixed_position',draw,old,fact,body)
            varied = experiment.selected_samples('position_jitter',draw,old,fact,body)
            self.assertEqual(fixed[:4],varied[:4])
            self.assertEqual([x[0] for x in fixed[4:]],[x[0] for x in varied[4:]])
            self.assertEqual([x[1] for x in fixed[4:]],[0]*4)
            self.assertEqual([x[1] for x in varied[4:]],draw['shifts'])

    def test_primary_gate_excludes_training_controls_and_retains_each_prior_hit(self):
        m, _, dev = experiment.load_bank()
        def arm(primary):
            return dict(groups=dict(development=[dict(id=r['id'],functional_status=0 if (not r['primary'] or primary) else 1) for r in dev],
                original=[dict(id='old',metrics=dict(exact=True))],factorial=[dict(id='factorial',functional_status=0)]))
        report = dict(validator_image='sha256:'+'a'*64,arms=dict(source=arm(False),fixed_position=arm(False),position_jitter=arm(False)))
        self.assertFalse(experiment.decision(report,m)['eligible_for_fresh_evaluation'])
        report['arms']['position_jitter'] = arm(True)
        self.assertTrue(experiment.decision(report,m)['eligible_for_fresh_evaluation'])
        self.assertFalse(experiment.decision(report,m)['promotion'])
        report['arms']['position_jitter']['groups']['factorial'][0]['functional_status']=1
        self.assertIn('lost prior factorial hit: factorial',experiment.decision(report,m)['reasons'])
        report['arms']['position_jitter']['groups']['original']=[]
        with self.assertRaisesRegex(ValueError,'omitted'): experiment.decision(report,m)
        report['validator_image']=None
        self.assertEqual(experiment.decision(report,m)['status'],'not_measured')

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Real training workflow installs PyTorch')
    def test_position_views_preserve_completion_supervision(self):
        import torch
        _, train, _ = experiment.load_bank()
        row = train[0]
        x,y = experiment.samples(row,1024,256)
        for shift in (8,16):
            other = copy.deepcopy(row)
            other['request']['prompt']=' '*shift+row['request']['prompt']
            xx,yy = experiment.samples(other,1024,256)
            self.assertFalse(torch.equal(x[:255],xx[:255]))
            self.assertTrue(torch.equal(y,yy))
            self.assertTrue(torch.all(yy[:255]==-100))
            self.assertEqual(bytes(yy[255:-1].tolist()).decode(),row['answer'])
            self.assertEqual(int(yy[-1]),0)
