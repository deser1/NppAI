import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from scripts.vendor_pair_training import qualify,verdict,compare,SLOTS,VENDORS
from scripts.first_step_ops import encoded


class VendorPairContracts(unittest.TestCase):
    def test_actual_cpu_and_os_are_required_not_runner_label_or_architecture(self):
        release={'ID':'ubuntu','VERSION_ID':'24.04'}
        self.assertTrue(qualify('GenuineIntel',{'vendor_id':'GenuineIntel'},'Linux','x86_64',release)['qualified'])
        self.assertFalse(qualify('GenuineIntel',{'vendor_id':'AuthenticAMD'},'Linux','x86_64',release)['qualified'])
        self.assertFalse(qualify('GenuineIntel',{'vendor_id':'GenuineIntel'},'Darwin','x86_64',release)['qualified'])
        self.assertFalse(qualify('GenuineIntel',{'vendor_id':'GenuineIntel'},'Linux','aarch64',release)['qualified'])
        self.assertFalse(qualify('GenuineIntel',{'vendor_id':'GenuineIntel'},'Linux','x86_64',{'ID':'ubuntu','VERSION_ID':'22.04'})['qualified'])

    def test_gate_distinguishes_absence_divergence_and_observed_agreement(self):
        data={'pairs':[{}],'summary':{'both64':{'all_training_traces_and_checkpoints_identical':True}}}
        self.assertEqual(verdict({'missing_vendors':['GenuineIntel']},data)['status'],'inconclusive')
        self.assertEqual(verdict({'missing_vendors':[]},data)['status'],'passed')
        data['summary']['both64']['all_training_traces_and_checkpoints_identical']=False
        self.assertEqual(verdict({'missing_vendors':[]},data)['status'],'failed')
        self.assertEqual(verdict({'missing_vendors':[]},None)['status'],'inconclusive')

    def test_missing_hardware_is_retained_inconclusive_and_strict_gate_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for slot in SLOTS:
                expected=VENDORS[slot.split('-')[0]];actual='AuthenticAMD' if expected=='GenuineIntel' else 'GenuineIntel'
                record=qualify(expected,{'vendor_id':actual},'Linux','x86_64',{'ID':'ubuntu','VERSION_ID':'24.04'})
                p=root/('vendor-full-'+slot);p.mkdir();(p/'hardware.json').write_bytes(encoded(record))
            with patch('builtins.print'):result=compare(root)
            self.assertFalse(result['candidate_repeatability_gate']['passed']);self.assertEqual(result['candidate_repeatability_gate']['status'],'inconclusive')
            self.assertIsNone(result['measurement']);self.assertEqual(len(result['coverage']['blocked_slots']),4)
            with patch('builtins.print'),self.assertRaises(SystemExit) as raised:compare(root,True)
            self.assertEqual(raised.exception.code,2);self.assertTrue((root/'vendor-comparison.json').exists())
            p=root/'vendor-full-intel-1/hardware.json';record=json.loads(p.read_bytes());record['qualified']=True;p.write_bytes(encoded(record))
            with self.assertRaisesRegex(ValueError,'qualification'):compare(root)

    def test_no_training_allowed_on_unqualified_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for slot in SLOTS:
                expected=VENDORS[slot.split('-')[0]];record=qualify(expected,{'vendor_id':'unknown'},'Linux','x86_64',{})
                p=root/('vendor-full-'+slot);p.mkdir();(p/'hardware.json').write_bytes(encoded(record))
            (root/'vendor-full-intel-1/report.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'unqualified'):compare(root)

    def test_retained_hardware_and_strict_inconclusive_result(self):
        import hashlib
        root=Path(__file__).resolve().parents[1]/'docs/evidence/vendor-qualified-training'
        manifest=json.loads((root/'manifest.json').read_bytes());result=json.loads((root/'comparison.json').read_bytes())
        self.assertEqual(hashlib.sha256((root/'comparison.json').read_bytes()).hexdigest(),manifest['comparison_sha256'])
        for slot,record in result['hardware'].items():
            self.assertEqual(record,json.loads((root/(slot+'-hardware.json')).read_bytes()))
            self.assertEqual(record,qualify(VENDORS[slot.split('-')[0]],record['cpu'],record['system'],record['machine'],record['os_release']))
        self.assertEqual(result['candidate_repeatability_gate'],verdict(result['coverage'],result['measurement']))
        self.assertEqual(result['candidate_repeatability_gate']['status'],'inconclusive');self.assertEqual(manifest['strict_cli_exit_code'],2)
        self.assertEqual(result['coverage']['qualified_slots'],['amd-2'])
        self.assertEqual(result['measurement']['pairs'],[])
        self.assertEqual(result['measurement']['reports'],[json.loads((root/'amd-2-report.json').read_bytes())])
        self.assertTrue(all(not s['all_training_traces_and_checkpoints_identical'] for s in result['measurement']['summary'].values()))

    def test_retained_full_intel_amd_traces_and_checkpoint_findings(self):
        import hashlib
        from scripts.portable_full_training import trace_difference
        root=Path(__file__).resolve().parents[1]/'docs/evidence/cross-vendor-full-training'
        manifest=json.loads((root/'manifest.json').read_bytes());comparison=json.loads((root/'comparison.json').read_bytes());traces={};reports={}
        self.assertEqual(hashlib.sha256((root/'comparison.json').read_bytes()).hexdigest(),manifest['comparison_sha256'])
        for sha,record in manifest['traces'].items():
            rows=[]
            for file in record['chunks']:rows.extend(json.loads((root/file).read_bytes()))
            raw=b''.join(encoded(r) for r in rows)
            self.assertEqual(hashlib.sha256(raw).hexdigest(),sha);self.assertEqual(len(raw),record['bytes'])
            self.assertEqual([r['step'] for r in rows],list(range(1025)));traces[sha]=rows
        for record in manifest['records']:
            raw=(root/record['report_file']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),record['report_sha256'])
            report=json.loads(raw);reports[record['artifact_name']]=report
            raw=(root/record['preflight_file']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),record['preflight_sha256'])
            preflight=json.loads(raw);self.assertTrue(preflight['matches_frozen_candidate'])
            self.assertTrue(all(preflight['first_step']==g['reference'] for g in report['gate']))
            for arm in report['arms'].values():
                self.assertEqual(arm['repeats'][0],arm['repeats'][1]);self.assertTrue(arm['same_host_identical'])
                r=arm['repeats'][0];self.assertEqual(traces[r['trace_sha256']][0]['parameters'],r['initial_parameters'])
                self.assertEqual(traces[r['trace_sha256']][-1]['parameters'],r['final_parameters'])
        self.assertEqual(comparison['reports'],[reports[f'portable-full-{i}'] for i in range(1,5)])
        cross=0
        for pair in comparison['pairs']:
            a,b=[reports[f'portable-full-{i}'] for i in pair['replicas']]
            expected=a['environment']['cpu']['vendor_id']!=b['environment']['cpu']['vendor_id']
            self.assertEqual(pair['cross_vendor'],expected);cross+=expected
            for variant,result in pair['arms'].items():
                x,y=[r['arms'][variant]['repeats'][0] for r in (a,b)]
                self.assertEqual(result['trace'],trace_difference(traces[x['trace_sha256']],traces[y['trace_sha256']]))
                self.assertEqual(result['checkpoint_identical'],x['checkpoint_sha256']==y['checkpoint_sha256'])
        self.assertEqual(cross,3)
        self.assertTrue(comparison['summary']['both64']['all_training_traces_and_checkpoints_identical'])
        self.assertFalse(comparison['summary']['native']['all_training_traces_and_checkpoints_identical'])

    def test_cli_uses_standard_library_only(self):
        root=Path(__file__).resolve().parents[1]
        r=subprocess.run([sys.executable,'-S',str(root/'scripts/vendor_pair_training.py'),'--help'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)


if __name__=='__main__':unittest.main()
