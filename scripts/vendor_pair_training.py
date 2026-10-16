"""Qualify actual Ubuntu CPU vendors before fixed full-training comparisons."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.first_step_ops import encoded
from scripts.portable_full_training import compare as compare_training

VENDORS = {'intel': 'GenuineIntel', 'amd': 'AuthenticAMD'}
SLOTS = tuple(f'{vendor}-{replica}' for vendor in VENDORS for replica in (1, 2))


def qualify(expected, cpu, system, machine, release):
    if expected not in VENDORS.values(): raise ValueError('unsupported expected CPU vendor')
    os_match = system == 'Linux' and machine == 'x86_64' and release.get('ID') == 'ubuntu' and release.get('VERSION_ID') == '24.04'
    vendor_match = cpu.get('vendor_id') == expected
    return {'version': 1, 'expected_vendor': expected, 'cpu': cpu, 'system': system, 'machine': machine,
            'os_release': release, 'vendor_matches': vendor_match, 'os_matches': os_match,
            'qualified': vendor_match and os_match, 'promotion': False}


def probe(expected, output):
    cpu = {}
    for line in Path('/proc/cpuinfo').read_text().splitlines():
        if ':' in line:
            key, value = map(str.strip, line.split(':', 1))
            if key in ('vendor_id', 'model name', 'flags', 'microcode'): cpu.setdefault(key, value)
    release = {}
    for line in Path('/etc/os-release').read_text().splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            if key in ('ID', 'VERSION_ID'): release[key] = value.strip('"')
    result = qualify(expected, cpu, platform.system(), platform.machine(), release)
    output.mkdir(parents=True, exist_ok=True); (output / 'hardware.json').write_bytes(encoded(result))
    if os.environ.get('GITHUB_OUTPUT'):
        with Path(os.environ['GITHUB_OUTPUT']).open('a') as stream:
            stream.write('qualified=' + str(result['qualified']).lower() + '\n')
    print(json.dumps(result))
    return result


def verdict(coverage, measurement):
    if coverage['missing_vendors']:
        return {'status': 'inconclusive', 'passed': False, 'reason': 'required actual CPU vendor unavailable'}
    if measurement is None or not measurement['pairs']:
        return {'status': 'inconclusive', 'passed': False, 'reason': 'no validated training pairs'}
    matched = measurement['summary']['both64']['all_training_traces_and_checkpoints_identical']
    return {'status': 'passed' if matched else 'failed', 'passed': matched,
            'reason': 'all observed 1024-update traces/checkpoints equal' if matched else 'observed training trace or checkpoint divergence'}


def compare(root, require_passed=False):
    probes = {}; completed = []
    for slot in SLOTS:
        directory = root / ('vendor-full-' + slot)
        hardware = json.loads((directory / 'hardware.json').read_bytes())
        expected = VENDORS[slot.split('-')[0]]
        if hardware != qualify(expected, hardware['cpu'], hardware['system'], hardware['machine'], hardware['os_release']):
            raise ValueError('hardware qualification record differs')
        probes[slot] = hardware
        report_path = directory / 'report.json'
        if hardware['qualified']:
            report = json.loads(report_path.read_bytes())
            if report['environment']['cpu'].get('vendor_id') != expected or not report['preflight_passed']:
                raise ValueError('training provenance does not match qualified hardware')
            completed.append(directory)
        elif report_path.exists():
            raise ValueError('training performed on unqualified hardware')
    observed = sorted({probes[p.name.removeprefix('vendor-full-')]['cpu']['vendor_id'] for p in completed})
    coverage = {'required_vendors': sorted(VENDORS.values()), 'observed_training_vendors': observed,
                'missing_vendors': sorted(set(VENDORS.values()) - set(observed)),
                'qualified_slots': [p.name.removeprefix('vendor-full-') for p in completed],
                'blocked_slots': [slot for slot, record in probes.items() if not record['qualified']]}
    measurement = compare_training(root, directories=completed, emit=False, allow_single=True) if completed else None
    result = {'version': 1, 'hardware': probes, 'coverage': coverage, 'measurement': measurement,
              'candidate_repeatability_gate': verdict(coverage, measurement), 'promotion': False,
              'limits': 'Only actual qualified Ubuntu 24.04 x64 hosts count. Missing vendors mean inconclusive. Fixed source/1024 updates/seed only; no universal portability, model-quality, default recipe or weight promotion.'}
    (root / 'vendor-comparison.json').write_bytes(encoded(result))
    print(json.dumps({k: v for k, v in result.items() if k not in ('hardware', 'measurement')}))
    if require_passed and not result['candidate_repeatability_gate']['passed']:
        raise SystemExit(2)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-vendor', choices=tuple(VENDORS.values()))
    parser.add_argument('--output', type=Path); parser.add_argument('--compare', type=Path)
    parser.add_argument('--require-passed-gate', action='store_true'); args = parser.parse_args()
    if args.compare: compare(args.compare, args.require_passed_gate)
    elif args.expected_vendor and args.output: probe(args.expected_vendor, args.output)
    else: parser.error('compare or expected-vendor/output required')
