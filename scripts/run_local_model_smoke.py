"""Measure first-pass local generation; execute candidates only in Docker."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_backends import DenseLocalBackend, GenerationRequest, format_prompt
import sandbox_validate

SOURCE_RUN = 38702282716
REPORT_SHA256 = 'bdac050d83b2efd9b3ad1bd12d6b991142120d4e0e2ffdc2994fb78b4bba4296'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def pinned_backend(root):
    raw = (root / 'loss-report.json').read_bytes()
    if digest(raw) != REPORT_SHA256:
        raise ValueError('pinned training report differs')
    report = json.loads(raw)
    if report['loss_policy'] != 'full_response':
        raise ValueError('wrong training policy')
    backend = DenseLocalBackend(str(root / 'NppAI-body-loss-mask.nppai'))
    if backend.model_id != 'sha256:' + report['model_sha256']:
        raise ValueError('pinned checkpoint differs')
    return backend


def measure(manifest, backend, image, max_new_tokens=256):
    raw = manifest.read_bytes()
    tasks = json.loads(raw)
    if not isinstance(tasks, list) or not tasks:
        raise ValueError('nonempty task list required')
    # Freeze every authored suite before any inference; expected values never enter prompts.
    prepared, seen = [], set()
    for task in tasks:
        if set(task) != {'task_id', 'prompt', 'cases_file'} or not all(
                isinstance(v, str) and v.strip() for v in task.values()):
            raise ValueError('invalid task fields')
        if task['task_id'] in seen:
            raise ValueError('duplicate task ID')
        seen.add(task['task_id'])
        cases, suite_hash = sandbox_validate.load_cases(Path(task['cases_file']))
        request = GenerationRequest(prompt=task['prompt'], max_new_tokens=max_new_tokens)
        if len(format_prompt(request).encode('utf-8')) > backend.weights['context']:
            raise ValueError('smoke prompt exceeds model context')
        prepared.append((task, request, cases, suite_hash))
    results = []
    for task, request, cases, suite_hash in prepared:
        start = time.perf_counter()
        response = backend.generate(request)
        latency = time.perf_counter() - start
        if response.model_id != backend.model_id:
            raise ValueError('model identity changed during inference')
        source = response.output
        category, detail = 'empty_output', 'empty or whitespace-only model output'
        if source.strip():
            status, detail = sandbox_validate.validate(source, 'syntax', cases, image)
            category = 'syntax_failure'
            if status == 0:
                status, detail = sandbox_validate.validate(source, 'tests', cases, image)
                category = 'passed' if status == 0 else 'functional_failure'
            if status not in (0, 1):
                raise RuntimeError('validation infrastructure/cancellation: ' + detail[:512])
        results.append({**task, 'suite_sha256': suite_hash, 'source': source,
                        'generated_tokens': response.generated_tokens,
                        'finish_reason': response.finish_reason,
                        'generation_seconds': latency, 'category': category,
                        'diagnostics': detail[:4096]})
    counts = {key: sum(r['category'] == key for r in results) for key in
              ('passed', 'empty_output', 'syntax_failure', 'functional_failure')}
    return {'schema_version': 1, 'backend': 'local', 'model_id': backend.model_id,
            'runtime': 'DenseLocalBackend NumPy FP32 greedy',
            'context_bytes': backend.weights['context'], 'max_new_tokens': max_new_tokens,
            'attempts_per_task': 1, 'docker_image': image,
            'manifest_sha256': digest(raw), 'tasks': len(results),
            'counts': counts, 'task_results': results,
            'limits': 'Six newly authored smoke probes; not an unbiased broad holdout. '
                      'No repair, online comparison, UI test, training or checkpoint promotion.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('evaluation/local-smoke-v1/tasks.json'))
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = measure(args.manifest, pinned_backend(args.source), args.image)
    import numpy
    import os
    result.update(source_run_id=SOURCE_RUN, source_report_sha256=REPORT_SHA256,
                  python_version=platform.python_version(), numpy_version=numpy.__version__,
                  evaluator_commit=os.getenv('GITHUB_SHA', ''),
                  workflow_run_id=os.getenv('GITHUB_RUN_ID', ''))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('model_id', 'tasks', 'counts', 'manifest_sha256')}))


if __name__ == '__main__':
    main()
