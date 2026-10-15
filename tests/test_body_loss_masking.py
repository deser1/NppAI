"""Body target boundaries, matched draw order and raw evidence integrity."""
import copy
import importlib.util
from pathlib import Path
import unittest
from scripts.body_loss_masking import (sample,mask_spec,boundaries,scheduled_evidence,audit,build,POLICIES)

DATASET=Path(__file__).resolve().parents[1]/'evaluation/instruction-v3'
HAS_TORCH=bool(importlib.util.find_spec('torch'))


class BodyLossTests(unittest.TestCase):
    def record(self):
        return {'id':'fixture','function':'foo','prompt':'Define foo(abcd) to return abcd + "é".',
                'answer':'def foo(abcd):\n    return abcd + "é"\n'}

    def test_boundaries_include_utf8_bytes_and_eos_no_truncation(self):
        r=self.record();start,header,body=boundaries(r)
        self.assertEqual(start,len(('[USER]: '+r['prompt']+'\n[AI]:\n').encode())-1)
        self.assertEqual(header,len(b'def foo(abcd):\n'))
        self.assertEqual(body,len('    return abcd + "é"\n'.encode())+1)
        self.assertEqual(mask_spec(r,'body_only')['header_targets'],0)
        self.assertEqual(mask_spec(r,'full_response')['header_targets'],header)
        for answer in ('','def foo(abcd):\n','def wrong(abcd):\n    return abcd\n'):
            with self.assertRaises(ValueError):boundaries({**r,'answer':answer})
        with self.assertRaises(ValueError):mask_spec(r,'unknown')

    def test_identical_draw_schedule_and_body_target_exposure(self):
        original=[{**self.record(),'id':'original-'+str(i)} for i in range(3)]
        renamed=[{**self.record(),'id':'renamed-'+str(i)} for i in range(3)]
        full=scheduled_evidence(original,renamed,'full_response')
        body=scheduled_evidence(original,renamed,'body_only')
        self.assertEqual(full['sha256'],body['sha256'])
        self.assertEqual(body['draws'],{'original':12000,'renamed':12000})
        for pool in ('original','renamed'):
            self.assertGreater(full['supervised_targets'][pool]['header'],0)
            self.assertEqual(body['supervised_targets'][pool]['header'],0)
            self.assertEqual(full['supervised_targets'][pool]['body_and_eos'],body['supervised_targets'][pool]['body_and_eos'])
        changed=scheduled_evidence(list(reversed(original)),renamed,'body_only')
        self.assertNotEqual(body['sha256'],changed['sha256'])

    @unittest.skipUnless(HAS_TORCH,'PyTorch required for actual loss tensors')
    def test_only_header_targets_removed_inputs_body_eos_preserved(self):
        import torch
        r=self.record();xf,yf=sample(r,256,'full_response');xb,yb=sample(r,256,'body_only')
        start,header,body=boundaries(r)
        self.assertTrue(torch.equal(xf,xb))
        self.assertEqual(yb[:start+header].tolist(),[-100]*(start+header))
        self.assertEqual(yf[start:start+header].tolist(),list(b'def foo(abcd):\n'))
        self.assertEqual(yb[start+header:].tolist(),list('    return abcd + "é"\n'.encode())+[0])
        self.assertTrue(torch.equal(yf[start+header:],yb[start+header:]))
        with self.assertRaises(ValueError):sample(r,20,'body_only')

    @unittest.skipUnless(HAS_TORCH,'PyTorch required for actual gradient boundary check')
    def test_cross_entropy_ignores_header_but_supervises_body_and_eos(self):
        import torch
        import torch.nn.functional as F
        r=self.record();_,y=sample(r,256,'body_only');start,header,_=boundaries(r)
        logits=torch.zeros((len(y),256),requires_grad=True)
        F.cross_entropy(logits,y,ignore_index=-100).backward()
        self.assertEqual(float(logits.grad[:start+header].abs().sum()),0.0)
        self.assertTrue(bool((logits.grad[start+header:].abs().sum(dim=1)>0).all()))
        self.assertGreater(float(logits.grad[-1].abs().sum()),0.0)

    def test_raw_report_audit_rejects_mask_draw_and_target_tampering(self):
        import tests.test_body_argument_aliases as fixtures
        fixtures.BodyArgumentTests.setUpClass()
        report=fixtures.BodyArgumentTests().fixture()
        original,renamed,_,_=build(DATASET,'varied_arguments')
        report['loss_policy']='body_only'
        report['training_schedule']=scheduled_evidence(original,renamed,'body_only')
        report['masking']={'policy':'body_only','prompt':'ignored','header':'ignored','body_and_eos':'supervised',
                          'header_context':'retained teacher forcing','reduction':'mean over active response targets'}
        self.assertTrue(audit(report,DATASET)['valid'])
        for mutate in (
            lambda r:r['masking'].update(header='supervised'),
            lambda r:r['training_schedule'].update(sha256='different-order'),
            lambda r:r['training_schedule']['supervised_targets']['renamed'].update(header=1),
            lambda r:r['training_schedule']['supervised_targets']['original'].update(body_and_eos=1),
        ):
            changed=copy.deepcopy(report);mutate(changed)
            with self.assertRaises(ValueError):audit(changed,DATASET)
