"""Evaluate recorded REAL model candidates using the production Docker validator.

Input JSONL: {"task_id":"add","attempt":1,"backend":"local","source":"def add(a,b): return a+b","cases_file":"examples/python-validation-cases.json"}
Each line is an actual model output; this tool does not generate candidates or
claim that deterministic fixtures measure model quality.
"""
import argparse
import json
from pathlib import Path
import sys
import sandbox_validate


def evaluate(records, image):
    tasks = {}
    for line_number, raw in enumerate(records, 1):
        if not raw.strip():
            continue
        item = json.loads(raw)
        if set(item) != {'task_id', 'attempt', 'backend', 'source', 'cases_file'}:
            raise ValueError(f'line {line_number}: unexpected/missing fields')
        task = item['task_id']
        attempt = item['attempt']
        if not isinstance(task, str) or not task or not isinstance(attempt, int) or isinstance(attempt, bool):
            raise ValueError(f'line {line_number}: invalid task or attempt')
        if item['backend'] not in ('local', 'online'):
            raise ValueError(f'line {line_number}: invalid backend')
        previous = tasks.get(task, [])
        if attempt != len(previous) + 1 or attempt > 3:
            raise ValueError(f'line {line_number}: attempts must be consecutive, maximum 3')
        if previous and previous[-1]['passed']:
            raise ValueError(f'line {line_number}: attempt after successful candidate')
        if not isinstance(item['source'], str) or not item['source']:\n            raise ValueError(f'line {line_number}: source must be nonempty Python text')\n        cases, digest = sandbox_validate.load_cases(Path(item['cases_file']))\n        if previous and previous[0]['suite_sha256'] != digest:\n            raise ValueError(f'line {line_number}: test suite changed between attempts')
        status, detail = sandbox_validate.validate(item['source'], 'syntax', cases, image)
        if status == 0:
            status, detail = sandbox_validate.validate(item['source'], 'tests', cases, image)
        if status in (2, 3):
            raise RuntimeError(f'line {line_number}: infrastructure/cancellation: {detail}')
        entry = {'attempt': attempt, 'backend': item['backend'], 'passed': status == 0,
                 'suite_sha256': digest, 'diagnostics': detail[:512]}
        tasks.setdefault(task, []).append(entry)
    if not tasks:
        raise ValueError('no model candidates provided')
    passed = sum(any(a['passed'] for a in attempts) for attempts in tasks.values())
    first = sum(attempts[0]['passed'] for attempts in tasks.values())
    repaired = sum(len(a) > 1 and a[-1]['passed'] for a in tasks.values())
    by_backend = {}\n    for backend in ('local', 'online'):\n        subset = [a for a in tasks.values() if a[0]['backend'] == backend]\n        if subset:\n            by_backend[backend] = {\n                'tasks': len(subset),\n                'first_attempt_pass': sum(a[0]['passed'] for a in subset),\n                'pass_within_three': sum(any(x['passed'] for x in a) for a in subset),\n                'repaired_after_failure': sum(len(a) > 1 and a[-1]['passed'] for a in subset),\n            }\n    return {'by_initial_backend': by_backend, 'tasks': len(tasks), 'first_attempt_pass': first,
            'pass_within_three': passed, 'repaired_after_failure': repaired,
            'task_results': tasks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jsonl', type=Path)
    parser.add_argument('--image', required=True, help='locally available immutable sha256 Docker image ID')
    parser.add_argument('--output', type=Path, default=Path('e2e-results.json'))
    args = parser.parse_args()
    result = evaluate(args.jsonl.read_text(encoding='utf-8').splitlines(), args.image)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f"{result['pass_within_three']}/{result['tasks']} tasks passed; first attempt {result['first_attempt_pass']}; repaired {result['repaired_after_failure']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
