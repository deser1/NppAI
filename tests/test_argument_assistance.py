"""Prompt-only signature assistance stays distinct from model/body correctness."""
import copy
import json
from pathlib import Path
import unittest
from scripts.evaluate_argument_assistance import (full_records, rename_code, task_header, score,
    summaries, paired, audit, ARMS, POLICIES, HEADER_RUN, BODY_RUN)
from scripts.diagnose_header_replay import full_records as original_full_records

DATASET=Path(__file__).resolve().parents[1]/'evaluation/instruction-v3'


class ArgumentAssistanceTests(unittest.TestCase):
    def test_prompt_parser_names_order_arity_and_rejection(self):
        self.assertEqual(task_header('Define foo(abcd, efghij) to return their sum.'),'def foo(abcd, efghij):\n')
        for task in ('foo(x=1)','foo(x,x)','foo(x) and bar(y)','foo(*args)','foo(class)'):
            with self.assertRaises(ValueError):
                task_header(task)

    def test_reference_transforms_preserve_literals_and_old_complete_answers(self):
        self.assertEqual(rename_code('def f(x):\n    return "x" + x\n',{'f':'cp_name','x':'abcd'}),
                         'def cp_name(abcd):\n    return "x" + abcd\n')
        old=original_full_records(DATASET)
        current=full_records(DATASET)
        for key in old:
            self.assertEqual(old[key],current[key])
        self.assertEqual(sum(len(v) for v in current.values()),1100)
        for rows in current.values():
            for r in rows:
                self.assertEqual(task_header(r['prompt']),r['answer'].splitlines(keepends=True)[0])
                self.assertLessEqual(len(('[USER]: '+r['prompt']+'\n[AI]:\n'+r['answer']).encode())+1,256)

    def fixture(self):
        groups={}
        sources={policy:{'evaluation':{'after':{}}} for policy in POLICIES}
        for split,rows in full_records(DATASET).items():
            cases=[]
            cache=[]
            for r in rows:
                header=task_header(r['prompt'])
                case={**r,'single_model':score(r,'')}
                for arm in ARMS:
                    case[arm]={'header':header,'header_accepted':True,'header_exact_reference':True,
                               'score':score(r,r['answer'])}
                cases.append(case)
                cache.append({**r,'output':header})
            groups[split]=cases
            for policy in POLICIES:
                sources[policy]['evaluation']['after'][split]={'cases':copy.deepcopy(cache)}
        report={'header_source_run':HEADER_RUN,'body_source_run':BODY_RUN,'groups':groups,
                'summary':summaries(groups),'assisted_paired_to_learned':paired(groups)}
        return report,sources

    def test_json_roundtrip_scores_and_paired_evidence(self):
        report,sources=self.fixture()
        self.assertTrue(audit(json.loads(json.dumps(report)),DATASET,sources)['valid'])
        self.assertEqual(report['summary']['test/argument_aliases']['overall']['task_signature_assisted']['exact_response'],336)

    def test_correct_assisted_header_does_not_imply_correct_body(self):
        r={'function':'foo','answer':'def foo(abcd):\n    return abcd + 1\n'}
        result=score(r,'def foo(abcd):\n    return values + 1\n')
        self.assertTrue(result['requested_signature'])
        self.assertTrue(result['syntax_any_function'])
        self.assertFalse(result['reference_ast'])
        self.assertFalse(result['exact_response'])

    def test_reject_reference_header_cache_output_and_summary_tampering(self):
        baseline,sources=self.fixture()
        for mutate in (
            lambda r:r['groups']['test/argument_aliases'][0]['task_signature_assisted'].update(header='def repaired(x):\n'),
            lambda r:r['groups']['test/argument_aliases'][0]['fixed_arguments']['score'].update(exact_response=False),
            lambda r:r['groups']['test/argument_aliases'][0].update(answer='def wrong(x):\n    return x\n'),
            lambda r:r['groups']['test/argument_aliases'].pop(),
            lambda r:r['assisted_paired_to_learned']['test/original']['fixed_arguments']['exact_response'].update(both=0),
        ):
            report=copy.deepcopy(baseline)
            mutate(report)
            with self.assertRaises(ValueError):
                audit(report,DATASET,sources)
