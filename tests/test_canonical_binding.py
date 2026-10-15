"""Prompt-only transforms must preserve identifiers, scope limits and raw evidence."""
import copy
import json
from pathlib import Path
import unittest
from scripts.evaluate_canonical_binding import (normalize,restore,build,score,summarize,paired,audit,
    SOURCE_RUN,REPORT_SHA,POLICIES)

DATASET=Path(__file__).resolve().parents[1]/'evaluation/instruction-v3'


class CanonicalBindingTests(unittest.TestCase):
    def test_all_schemas_prompt_only_order_simultaneous_substitution(self):
        self.assertEqual(normalize('Define foo(b, a) to return b minus a.','x'),
            {'prompt':'Define foo(a, b) to return a minus b.','header':'def foo(a, b):\n',
             'original_header':'def foo(b, a):\n','restore_map':{'a':'b','b':'a'},'mode':'canonical'})
        for p in POLICIES:
            t=normalize('Define foo(abcd) to return abcd squared.',p)
            self.assertEqual(t['header'],f'def foo({p}):\n')
            self.assertEqual(restore(f'def foo({p}):\n    return {p} * {p}\n',t)['output'],
                             'def foo(abcd):\n    return abcd * abcd\n')
        for task in ('foo()','foo(x,x)','foo(x=1)','foo(len)','foo(foo)'):
            with self.assertRaises(ValueError):normalize(task,'x')
        self.assertEqual(normalize('foo(x,low,high)','x')['mode'],'passthrough_arity')
        self.assertEqual(normalize('foo(x,low,high)','x')['prompt'],'foo(x,low,high)')
        with self.assertRaises(ValueError):normalize('foo(x)','oracle')

    def test_restore_variable_spans_only_preserves_literals_attributes_and_keyword_names(self):
        t=normalize('Define foo(abcd) to return abcd.','x')
        text='def foo(x):\n    return "x é" + str(x.x(x=x))\n'
        self.assertEqual(restore(text,t),{'status':'ok','output':'def foo(abcd):\n    return "x é" + str(abcd.x(x=abcd))\n'})
        t=normalize('Define foo(b,a) to return b minus a.','x')
        self.assertEqual(restore('def foo(a, b):\n    return a - b\n',t)['output'],
                         'def foo(b, a):\n    return b - a\n')

    def test_invalid_outputs_and_nested_binding_rejected_without_repair(self):
        t=normalize('foo(abcd)','x')
        for text in ('def foo(x):\n    return (\n', 'def wrong(x):\n    return x\n',
                     'def foo(x):\n    return [x for x in range(3)]\n',
                     'def foo(x):\n    return lambda x: x\n',
                     'def foo(x):\n    x = 1\n    return x\n',
                     'def foo(x):\n    return (x:=3)\n',
                     'def foo(x):\n    return abcd + x\n',
                     'def foo(x):\n    return x\nprint(1)\n'):
            item=restore(text,t)
            self.assertNotEqual(item['status'],'ok')
            self.assertEqual(item['output'],'')
        item=restore('def foo(x):\n    return values + 1\n',t)
        self.assertEqual(item['output'],'def foo(abcd):\n    return values + 1\n')
        self.assertFalse(score({'function':'foo','answer':'def foo(abcd):\n    return abcd+1\n'},item['output'])['reference_ast'])

    @classmethod
    def setUpClass(cls):cls.development=build(DATASET,'fixed_arguments')[2]

    def fixture(self):
        groups={};source={'model_sha256':'frozen','evaluation':{'after':{}}}
        for split,rows in self.development.items():
            cases=[];cached=[]
            for record in rows:
                # Trusted reference fixture only, never a generated-code execution.
                direct=score(record,record['answer']);case={**record,'direct_assisted':{'score':direct}}
                cached.append({**record,'task_signature_assisted':direct})
                for p in POLICIES:
                    t=normalize(record['prompt'],p)
                    from scripts.evaluate_argument_assistance import rename_code
                    text=rename_code(record['answer'],{v:k for k,v in t['restore_map'].items()})
                    restored=restore(text,t)
                    case[p]={'transform':t,'normalized_output':text,'restoration':restored,'score':score(record,restored['output'])}
                cases.append(case)
            groups[split]=cases;source['evaluation']['after'][split]={'cases':cached}
        report={'source_run_id':SOURCE_RUN,'source_report_sha256':REPORT_SHA,'model_sha256':'frozen','groups':groups,
                'summary':{s:summarize(c) for s,c in groups.items()},'paired_to_direct':paired(groups)}
        return report,source

    def test_frozen_full_view_roundtrip_and_tampering(self):
        report,source=self.fixture()
        self.assertTrue(audit(json.loads(json.dumps(report)),DATASET,source)['valid'])
        for mutate in (
            lambda r:r.update(source_report_sha256='different'),
            lambda r:r['groups']['test/argument_aliases'][0]['x']['transform'].update(prompt='oracle repair'),
            lambda r:r['groups']['test/argument_aliases'][0]['x']['restoration'].update(output='repaired'),
            lambda r:r['groups']['test/argument_aliases'][0]['x']['score'].update(reference_ast=False),
            lambda r:r['groups']['test/argument_aliases'].pop(),
            lambda r:r['paired_to_direct']['test/original']['x']['exact_response'].update(lost=1),
        ):
            changed=copy.deepcopy(report);mutate(changed)
            with self.assertRaises(ValueError):audit(changed,DATASET,source)
