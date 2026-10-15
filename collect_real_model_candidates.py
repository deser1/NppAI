"""Collect actual backend generations for the Docker replay evaluator.

Input tasks JSON: [{"task_id":"add-01","prompt":"Write a complete Python
buffer implementing add(a,b)","cases_file":"examples/python-validation-cases.json"}].
No tests are shown to the model. This collector does not assert code quality.
"""
import argparse
import json
import os
from pathlib import Path

from model_backends import BackendRouter, DenseLocalBackend, GenerationRequest, RemoteBackend


def collect(tasks, router, max_new_tokens=512):
    if router.mode == "hybrid":
        raise ValueError("hybrid collection requires backend provenance; use local or online")
    seen = set()
    records = []
    for task in tasks:
        if set(task) != {"task_id", "prompt", "cases_file"}:
            raise ValueError("each task needs task_id, prompt, cases_file")
        task_id, prompt, cases_file = task["task_id"], task["prompt"], task["cases_file"]
        if not all(isinstance(v, str) and v.strip() for v in (task_id, prompt, cases_file)):
            raise ValueError("task fields must be nonempty strings")
        if task_id in seen:
            raise ValueError("duplicate task_id")
        seen.add(task_id)
        response = router.generate(GenerationRequest(prompt=prompt, max_new_tokens=max_new_tokens))
        backend = "online" if router.mode == "online" else "local"
        records.append({"task_id": task_id, "attempt": 1, "backend": backend,
                        "source": response.output, "cases_file": cases_file})
    if not records:
        raise ValueError("no tasks")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("local", "online"), required=True)
    parser.add_argument("--model", help="dense local checkpoint")
    parser.add_argument("--endpoint", help="HTTPS /api/v1/generate endpoint")
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.mode == "local":
        if not args.model:
            parser.error("--model required for local mode")
        router = BackendRouter("local", local=DenseLocalBackend(args.model))
    else:
        if not args.allow_remote:
            parser.error("online mode requires --allow-remote")
        remote = RemoteBackend(args.endpoint or "", os.getenv("NPPAI_API_KEY", ""),
                               allow_remote=True)
        router = BackendRouter("online", remote=remote, allow_remote=True)
    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        parser.error("tasks must be a JSON array")
    records = collect(tasks, router, args.max_new_tokens)
    # Write only after all tasks succeed; do not publish private prompts or outputs.
    args.output.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records),
                           encoding="utf-8")
    print(f"Recorded {len(records)} actual first-attempt model outputs to {args.output}")


if __name__ == "__main__":
    main()
