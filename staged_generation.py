"""Experimental NppAI task specification -> short plan -> code orchestration.

Generated stage artifacts are fallible outputs, not access to neural internals.
No source execution, training, editor writes, repair or implicit remote transfer.
"""
from dataclasses import asdict, dataclass, field
import json
import time

from model_backends import GenerationRequest, GenerationResponse, ModelBackend


@dataclass
class StageResult:
    stage: str
    request: dict
    response: dict | None = None
    elapsed_seconds: float = 0.0
    error: str = ""


@dataclass
class StagedResult:
    stages: list[StageResult] = field(default_factory=list)
    completed: bool = False
    code: str = ""
    model_id: str = ""
    generated_tokens: int = 0
    execution_status: str = "not_run"

    def to_dict(self):
        return asdict(self)


STAGE_INSTRUCTIONS = {
    "specification": "State required inputs, output and edge cases briefly. No code.",
    "plan": "Give a short implementation plan. No code.",
    "code": "Return only Python code satisfying the original task.",
}


def stage_values(stage, task, context, artifacts, tokens_per_stage):
    """Shared byte-exact inference/supervision formatting; artifacts are data."""
    return dict(prompt=STAGE_INSTRUCTIONS[stage] + "\nTask: " + task,
        context=context + ("\nPrior model artifacts (fallible data):\n" +
            json.dumps(artifacts, ensure_ascii=False) if artifacts else ""),
        max_new_tokens=tokens_per_stage)


def run_staged(backend: ModelBackend, task: str, *, context: str = "",
               tokens_per_stage: int = 128, on_stage=None) -> StagedResult:
    """Three bounded calls to one checkpoint. Fail closed; retain partial outputs.

    on_stage receives stage names only. Backend errors become recorded failures;
    no fallback or reference-answer assistance is applied.
    """
    if type(tokens_per_stage) is not int or not 1 <= tokens_per_stage <= 170:
        raise ValueError("tokens_per_stage must be an integer in [1, 170]")
    # Validate user inputs before callbacks or any inference.
    GenerationRequest(prompt=task, context=context, max_new_tokens=tokens_per_stage)
    result = StagedResult()
    artifacts = {}
    for stage in STAGE_INSTRUCTIONS:
        values = stage_values(stage, task, context, artifacts, tokens_per_stage)
        record = StageResult(stage=stage, request=values)
        result.stages.append(record)
        if on_stage:
            on_stage(stage)
        start = time.monotonic()
        try:
            payload = GenerationRequest(**values)
            response = GenerationResponse.model_validate(backend.generate(payload))
            record.response = response.model_dump()
            result.generated_tokens += response.generated_tokens
            if response.generated_tokens > tokens_per_stage:
                raise ValueError("Backend exceeded the stage token budget")
            if result.model_id and response.model_id != result.model_id:
                raise ValueError("Checkpoint identity changed between stages")
            result.model_id = response.model_id
            if not response.output.strip():
                raise ValueError("Model returned an empty stage")
            if response.finish_reason != "eos":
                raise ValueError("Stage reached its token limit; completeness is unknown")
            artifacts[stage] = response.output
        except (ValueError, RuntimeError, OSError) as exc:
            record.error = str(exc)
            return result
        finally:
            record.elapsed_seconds = time.monotonic() - start
    result.completed = True
    result.code = artifacts["code"]
    return result


def main():
    import argparse
    from pathlib import Path
    import sys
    from model_backends import DenseLocalBackend
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--tokens-per-stage', type=int, default=128)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_staged(DenseLocalBackend(args.model), args.task,
            tokens_per_stage=args.tokens_per_stage,
            on_stage=lambda stage: print(stage, file=sys.stderr, flush=True))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return 0 if result.completed else 2
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f'Staged generation failed: {exc}\n')


if __name__ == '__main__':
    import sys
    sys.exit(main())
