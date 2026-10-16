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

    def test_cli_uses_standard_library_only(self):
        root=Path(__file__).resolve().parents[1]
        r=subprocess.run([sys.executable,'-S',str(root/'scripts/vendor_pair_training.py'),'--help'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)


if __name__=='__main__':unittest.main()
