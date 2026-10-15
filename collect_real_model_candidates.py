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
import sandbox_validate


def collect(tasks, router, max_new_tokens=512, *, image=None, max_attempts=1):
    if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or not 1 <= max_attempts <= 3:
        raise ValueError('max_attempts must be 1..3')
    if max_attempts > 1 and not image:
        raise ValueError('Docker image required for repair attempts')
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
        cases = None
        if image:
            cases, _ = sandbox_validate.load_cases(Path(cases_file))
        feedback = ""
        for attempt in range(1, max_attempts + 1):
            repair_prompt = prompt if attempt == 1 else (
                prompt + "\\n\\nYour previous complete Python module failed validation. "
                "Return a corrected complete Python module only.\\n"
                "Previous candidate:\\n" + source[:8192] +
                "\\nValidation feedback:\\n" + feedback[:2048])
            response = router.generate(GenerationRequest(prompt=repair_prompt, max_new_tokens=max_new_tokens))
            source = response.output
            backend = "online" if router.mode == "online" else "local"
            records.append({"task_id": task_id, "attempt": attempt, "backend": backend,
                            "source": source, "cases_file": cases_file})
            if image:
                status, feedback = sandbox_validate.validate(source, "syntax", cases, image)
                if status == 0:
                    status, feedback = sandbox_validate.validate(source, "tests", cases, image)
                if status in (2, 3):
                    raise RuntimeError("validation unavailable or cancelled: " + feedback[:512])
                if status == 0:
                    break
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
    parser.add_argument("--max-attempts", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--image", help="immutable local Docker image ID for validation")
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
    records = collect(tasks, router, args.max_new_tokens, image=args.image, max_attempts=args.max_attempts)
    # Write only after all tasks succeed; do not publish private prompts or outputs.
    args.output.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records),
                           encoding="utf-8")
    print(f"Recorded {len(records)} actual model attempts to {args.output}")


if __name__ == "__main__":
    main()
