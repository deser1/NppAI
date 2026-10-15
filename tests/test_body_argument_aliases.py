"""Full-response references, split isolation and raw evidence integrity."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
from scripts.body_argument_aliases import (build, complete, frame_size, evaluate, audit, paired,
    summarize, ARMS, OPTIMIZER, SOURCE_RUN, STEPS)
from scripts.evaluate_argument_assistance import full_records
from scripts.evaluate_task_signature_wording import task_header
from scripts.diagnose_header_replay import score

DATASET = Path(__file__).resolve().parents[1] / 'evaluation/instruction-v3'


class BodyArgumentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixed = build(DATASET, 'fixed_arguments')
        cls.varied = build(DATASET, 'varied_arguments')

    def test_matched_pools_split_isolation_complete_body_and_no_truncation(self):
        a, b, development, metadata = self.fixed
        av, bv, dv, mv = self.varied
        self.assertEqual(a, av)
        self.assertEqual(development, dv)
        self.assertEqual(development, full_records(DATASET))
        self.assertEqual((len(a), len(b), len(bv)), (6144, 18432, 18432))
        self.assertEqual(metadata['development_sha256'], mv['development_sha256'])
        training_names = {r['function'] for r in a + bv}
        training_args = {n for r in bv for n in r['argument_alias_map'].values()}
        for split, rows in dv.items():
            self.assertFalse(training_names & {r['function'] for r in rows})
            self.assertFalse(training_args & {n for r in rows for n in r.get('argument_alias_map', {}).values()})
        self.assertEqual(sum(len(v) for v in dv.values()), 1100)
        for rows in (a, b, bv, *dv.values()):
            self.assertTrue(all(frame_size(r) <= 256 for r in rows))
            self.assertTrue(all(len(r['answer'].splitlines()) >= 2 for r in rows))
        for r in bv[:4]:
            self.assertEqual(r['answer'], 'def cp_2qq52a(wzkd, acadxk):\n    return wzkd + acadxk\n')
        self.assertEqual(metadata['frame_sizes']['renamed']['max_frame_bytes'], 196)
        self.assertEqual(mv['frame_sizes']['renamed']['max_frame_bytes'], 200)
        self.assertEqual(mv['replay']['original_example_draws'], 12000)
        self.assertEqual(mv['replay']['renamed_example_draws'], 12000)

    def test_training_lookup_rejects_unavailable_source_and_header_mismatch(self):
        row = {'id':'train-example::name-0', 'source_id':'train-example::name-0',
               'function':'cp_example','answer':'def cp_example(abcd):\n',
               'argument_alias_map':{'x':'abcd'}}
        source = {'id':'train-example','function':'f','answer':'def f(x):\n    return "x" + x\n'}
        self.assertEqual(complete([row], {'train-example':source})[0]['answer'],
                         'def cp_example(abcd):\n    return "x" + abcd\n')
        with self.assertRaises(KeyError):
            complete([row], {})
        row['answer'] = 'def cp_example(wrong):\n'
        with self.assertRaises(ValueError):
            complete([row], {'train-example':source})

    def fixture(self):
        development = self.varied[2]
        groups = {}
        for split, rows in development.items():
            cases = [{**r, 'assisted_header':task_header(r['prompt']),
                      **{arm:score(r,r['answer']) for arm in ARMS}} for r in rows]
            groups[split] = {'cases':cases,'summary':summarize(cases)}
        report = {'source_run_id':SOURCE_RUN, 'source':{'model_sha256':'body-checkpoint',
                  'prior_body_updates':20000,'source_policy':'four_forms'},
                  'argument_policy':'varied_arguments', 'dataset':self.varied[3],
                  'additional_steps':STEPS,'total_body_updates':26000,
                  'optimizer':OPTIMIZER,'context':256,'new_tokens_per_call':160,
                  'observed_training_draws':{'original':12000,'renamed':12000},
                  'initialization':{'checkpoint_sha256':'body-checkpoint','method':'all dense FP32 weights; fresh AdamW optimizer'},
                  'reload_outputs_identical':True,'evaluation':{'before':copy.deepcopy(groups),'after':groups}}
        report['paired_before_after'] = paired(report['evaluation'])
        return report

    def test_roundtrip_and_tampered_reference_output_budget_and_retention(self):
        baseline = self.fixture()
        self.assertTrue(audit(json.loads(json.dumps(baseline)), DATASET)['valid'])
        for mutate in (
            lambda r:r.update(total_body_updates=20000),
            lambda r:r.update(reload_outputs_identical=False),
            lambda r:r['observed_training_draws'].update(renamed=24000),
            lambda r:r['initialization'].update(checkpoint_sha256='header-checkpoint'),
            lambda r:r['evaluation']['after']['test/argument_aliases']['cases'][0].update(answer='def wrong(x):\n    pass\n'),
            lambda r:r['evaluation']['after']['test/argument_aliases']['cases'][0]['task_signature_assisted'].update(output='def wrong(x):\n    pass\n'),
            lambda r:r['evaluation']['after']['test/argument_aliases']['cases'].pop(),
            lambda r:r['paired_before_after']['test/original']['task_signature_assisted']['exact_response'].update(lost=1),
        ):
            report = copy.deepcopy(baseline)
            mutate(report)
            with self.assertRaises(ValueError):
                audit(report, DATASET)

    def test_assisted_generation_receives_only_prompt_prefix_and_no_reference(self):
        record = self.varied[2]['test/argument_aliases'][0]
        calls = []
        def fake(model, prompt, context, assistant_prefix=''):
            calls.append((prompt, context, assistant_prefix))
            return assistant_prefix + '    return values\n' if assistant_prefix else ''
        output = evaluate(None, {'probe':[record]}, fake)['probe']['cases'][0]
        self.assertEqual(calls, [(record['prompt'],256,task_header(record['prompt'])), (record['prompt'],256,'')])
        self.assertFalse(output['task_signature_assisted']['reference_ast'])
        self.assertTrue(output['task_signature_assisted']['requested_signature'])
        self.assertFalse(output['single_model']['exact_response'])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch required for actual supervision tensors')
    def test_full_body_and_eos_supervised_prompt_masked(self):
        from scripts.train_broader_instruction_experiment import sample
        record = self.varied[1][0]
        x, y = sample(record,256)
        prefix = ('[USER]: '+record['prompt']+'\n[AI]:\n').encode()
        target = y.tolist()
        self.assertEqual(target[:len(prefix)-1], [-100]*(len(prefix)-1))
        self.assertEqual(target[len(prefix)-1:], list(record['answer'].encode())+[0])
        self.assertEqual(len(x),frame_size(record)-1)
