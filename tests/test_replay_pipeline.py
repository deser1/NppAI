"""Diagnose argument drift and audit complete-reference staged evidence."""
import copy
import json
from pathlib import Path
import unittest
from scripts.diagnose_header_replay import (header_error, lost_cases, full_records, score, summaries,
    paired_pipeline, audit, ARMS, REPLAY_RUN, BODY_RUN)

DATASET = Path(__file__).resolve().parents[1] / 'evaluation/instruction-v3'


class ReplayPipelineTests(unittest.TestCase):
    def test_argument_drift_is_distinct_from_name_or_body_success(self):
        error=header_error('def fn_nta64g(values):\n','def fn_nta64g(x):\n')
        self.assertEqual(error['category'],'arguments')
        self.assertEqual(error['actual_signature'],['fn_nta64g','x'])
        self.assertEqual(header_error('def f(x):\n','def g(x):\n')['category'],'name')
        self.assertEqual(header_error('def f(x):\n','garbage')['category'],'malformed')
        record={'function':'f','answer':'def f(values):\n    return sorted(values)\n'}
        result=score(record,'def f(x):\n    return sorted(values)\n')
        self.assertTrue(result['requested_function'])
        self.assertFalse(result['requested_signature'])
        self.assertFalse(result['reference_ast'])

    def fixture(self):
        records=full_records(DATASET)
        groups={}
        sources={policy:{'evaluation':{'before':{},'after':{}}} for policy in ('no_replay','half_original')}
        for split,rows in records.items():
            cases=[]
            cache=[]
            for record in rows:
                header=record['answer'].splitlines(keepends=True)[0]
                item={**record,'single_model':score(record,'')}
                for arm in ARMS:
                    item[arm]={'header':header,'header_accepted':True,'header_exact_reference':True,
                               'matches_cached_header':True,'score':score(record,record['answer'])}
                cases.append(item)
                cache.append({**record,'answer':header,'output':header,'metrics':{'exact_header':True}})
            groups[split]=cases
            for policy in sources:
                for stage in ('before','after'):
                    sources[policy]['evaluation'][stage][split]={'cases':copy.deepcopy(cache)}
        report={'replay_source_run_id':REPLAY_RUN,'body_source_run_id':BODY_RUN,'groups':groups,
                'summary':summaries(groups),'paired_to_pretrained':paired_pipeline(groups),
                'archived_header_regressions':{policy:lost_cases(r) for policy,r in sources.items()}}
        return report,sources

    def test_full_references_are_bodies_within_context(self):
        groups=full_records(DATASET)
        self.assertEqual(sum(len(v) for v in groups.values()),572)
        for rows in groups.values():
            for r in rows:
                self.assertIn('\n    return ',r['answer'])
                self.assertTrue(r['answer'].startswith('def '+r['function']+'('))
                self.assertLessEqual(len(('[USER]: '+r['prompt']+'\n[AI]:\n'+r['answer']).encode())+1,256)

    def test_roundtrip_evidence_and_cached_header_diagnostics(self):
        report,sources=self.fixture()
        self.assertTrue(audit(json.loads(json.dumps(report)),DATASET,sources)['valid'])
        self.assertEqual(report['summary']['test/renamed']['overall']['half_original']['exact_response'],336)
        self.assertEqual(report['paired_to_pretrained']['test/original']['half_original']['exact_response']['both'],28)

    def test_reject_changed_metrics_reference_prefix_and_missing_case(self):
        baseline,sources=self.fixture()
        for mutate in (
            lambda r:r['groups']['test/original'][0]['half_original']['score'].update(exact_response=False),
            lambda r:r['groups']['test/original'][0].update(answer='def wrong():\n    return 0\n'),
            lambda r:r['groups']['test/original'][0]['half_original'].update(header_accepted=False),
            lambda r:r['groups']['test/original'][0]['half_original'].update(matches_cached_header=False),
            lambda r:r['groups']['test/original'].pop(),
            lambda r:r['summary']['test/original']['overall'].update(total=0),
        ):
            report=copy.deepcopy(baseline)
            mutate(report)
            with self.assertRaises(ValueError):
                audit(report,DATASET,sources)

    def test_saved_lost_case_keeps_raw_argument_error(self):
        _,sources=self.fixture()
        row=sources['half_original']['evaluation']['after']['test/original']['cases'][0]
        row['output']='def '+row['function']+'(changed):\n'
        row['metrics']['exact_header']=False
        cases=lost_cases(sources['half_original'])
        self.assertEqual(len(cases),1)
        self.assertEqual(cases[0]['difference']['category'],'arguments')
        self.assertEqual(cases,json.loads(json.dumps(cases)))
