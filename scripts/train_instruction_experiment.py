"""Bounded synthetic instruction-training experiment; no generated code execution."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn.functional as F
from train_nppai import NppAIModel, export_to_bin

# Entire tasks are held out. The test prompt and reference are never optimized.
# Deliberately small, disclosed synthetic dataset: this is not a competence claim.
TRAIN = [
    ("Implement add(a, b) returning their sum.", "def add(a, b):\n    return a + b\n"),
    ("Implement subtract(a, b) returning a minus b.", "def subtract(a, b):\n    return a - b\n"),
    ("Implement multiply(a, b) returning their product.", "def multiply(a, b):\n    return a * b\n"),
    ("Implement square(x) returning x squared.", "def square(x):\n    return x * x\n"),
    ("Implement double(x) returning twice x.", "def double(x):\n    return x * 2\n"),
    ("Implement is_even(x) returning whether x is even.", "def is_even(x):\n    return x % 2 == 0\n"),
    ("Implement total(numbers) returning their sum.", "def total(numbers):\n    return sum(numbers)\n"),
    ("Implement largest(numbers) returning their maximum.", "def largest(numbers):\n    return max(numbers)\n"),
]
HELD_OUT = [
    ("sum_positive", "Implement sum_positive(numbers) returning the sum of positive integers; ignore zero and negative values."),
    ("triple", "Implement triple(x) returning three times x."),
    ("is_odd", "Implement is_odd(x) returning whether x is odd."),
]


def prompt(task):
    return "[USER]: " + task + "\n[AI]:\n"


def generate(model, task, context, assistant_prefix=""):
    tokens = list((prompt(task)+assistant_prefix).encode("utf-8"))
    output = list(assistant_prefix.encode("utf-8"))
    with torch.no_grad():
        for _ in range(160):
            logits = model(torch.tensor([tokens[-context:]], dtype=torch.long))[0, -1]
            token = int(logits.argmax())
            if token == 0:
                break
            tokens.append(token)
            output.append(token)
    return bytes(output).decode("utf-8", errors="replace")


def assess(model, tasks, context):
    cases = []
    for case_id, task in tasks:
        text = generate(model, task, context)
        syntax = False
        try:
            tree = ast.parse(text)
            syntax = any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body)
        except SyntaxError:
            pass
        cases.append({"id": case_id, "prompt": task, "output": text,
                      "nonempty": bool(text.strip()), "syntax_function": syntax})
    return cases


def run(output, steps=2000, seed=42, baseline=None):
    if not 1 <= steps <= 10000:
        raise ValueError("steps must be 1..10000")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    context = 192
    model = NppAIModel(256, 64, 128, 2, context)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    samples = []
    for task, answer in TRAIN:
        prefix = list(prompt(task).encode("utf-8"))
        sequence = prefix + list(answer.encode("utf-8")) + [0]
        if len(sequence) > context:
            raise ValueError("training example exceeds context")
        x = torch.tensor([sequence[:-1]], dtype=torch.long)
        y = torch.tensor([sequence[1:]], dtype=torch.long)
        y[:, :len(prefix)-1] = -100  # response-only loss; includes EOS
        samples.append((x, y))
    model.train()
    losses = []
    for step in range(steps):
        x, y = samples[random.randrange(len(samples))]
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 100 == 0 or step == steps-1:
            print(f"step={step+1} loss={losses[-1]:.5f}", flush=True)
    model.eval()
    cases = assess(model, HELD_OUT, context)
    training_cases = assess(model, [(str(i), task) for i, (task, _) in enumerate(TRAIN)], context)
    reconstruction = sum(case["output"] == TRAIN[i][1] for i, case in enumerate(training_cases))
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / "NppAI-instruction-experiment.nppai"
    export_to_bin(model, str(checkpoint))
    report = {"seed": seed, "steps": steps, "context": context,
        "training_tasks": len(TRAIN), "held_out_tasks": len(HELD_OUT),
        "dataset_sha256": hashlib.sha256(json.dumps(TRAIN, sort_keys=True).encode()).hexdigest(),
        "model_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "initial_loss": losses[0], "final_loss": losses[-1],
        "held_out_nonempty": sum(x["nonempty"] for x in cases),
        "held_out_syntax_functions": sum(x["syntax_function"] for x in cases),
        "training_exact_reconstructions": reconstruction,
        "cases": cases, "limitations": "Eight synthetic training tasks; held-out syntax only, no execution or functional correctness claim. Separate new model, not continued training of the 300-step checkpoint."}
    if baseline:
        from scripts.check_dense_runtime_parity import load_weights, torch_model
        w = load_weights(baseline)
        baseline_cases = assess(torch_model(w), HELD_OUT, w["context"])
        report["baseline"] = {
            "model_sha256": hashlib.sha256(Path(baseline).read_bytes()).hexdigest(),
            "cases": baseline_cases,
            "nonempty": sum(case["nonempty"] for case in baseline_cases),
            "syntax_functions": sum(case["syntax_function"] for case in baseline_cases),
            "comparison_limit": "Same tasks and greedy FP32 decoding; training data, steps and context size differ, so this is not a single-variable ablation."}
    (destination / "instruction-report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--baseline")
    args = parser.parse_args()
    run(args.output, args.steps, baseline=args.baseline)
