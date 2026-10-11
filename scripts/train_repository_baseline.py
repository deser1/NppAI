"""Train a small experimental NppAI checkpoint from repository source files.

This is a smoke-training baseline, not a general-purpose coding assistant.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from train_nppai import BPETokenizer, NppAIModel, export_to_bin


EXTENSIONS = {".py", ".cpp", ".h", ".hpp", ".c", ".cmake", ".md", ".json", ".yml"}
EXCLUDED = {".git", "build", "models", ".venv", "node_modules", "__pycache__"}


def collect_corpus(repo, max_file_bytes=200_000):
    repo = Path(repo)
    chunks, files = [], []
    for path in sorted(repo.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        if any(part in EXCLUDED or part.startswith(".git") for part in path.relative_to(repo).parts):
            continue
        if path.stat().st_size > max_file_bytes:
            continue
        try:
            body = path.read_text(encoding="utf-8")
        except UnicodeError:
            continue
        if body.strip():
            files.append(path.relative_to(repo).as_posix())
            chunks.append("
# FILE: " + files[-1] + "
" + body + "
")
    if not chunks:
        raise ValueError("no supported repository source files found")
    return "".join(chunks), files


def train(repo, output, steps=30, seed=42, block_size=64, batch_size=4):
    if steps < 1 or block_size < 8 or batch_size < 1:
        raise ValueError("invalid training parameters")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    corpus, files = collect_corpus(repo)
    tokenizer = BPETokenizer(vocab_size=256)
    data = torch.tensor(tokenizer.encode(corpus), dtype=torch.long)
    if len(data) <= block_size + 1:
        raise ValueError("corpus is too short for context window")
    model = NppAIModel(vocab_size=256, dim=64, hidden_dim=128,
                       n_layers=2, max_seq_len=block_size)
    # Reserve the final 10% of tokens for held-out next-token validation.
    split = max(block_size + 2, int(len(data) * 0.9))
    if len(data) - split <= block_size + 1:
        raise ValueError("corpus too short for a held-out validation split")
    train_data, val_data = data[:split], data[split:]
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    losses = []
    model.train()
    for step in range(steps):
        offsets = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in offsets])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in offsets])
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(round(float(loss.detach()), 5))
        if step % 10 == 0 or step == steps - 1:
            print(f"step {step + 1}/{steps} loss={losses[-1]:.5f}", flush=True)
    model.eval()
    with torch.no_grad():
        validation_x = val_data[:block_size].unsqueeze(0)
        validation_y = val_data[1:block_size + 1].unsqueeze(0)
        validation_logits = model(validation_x)
        validation_loss = float(F.cross_entropy(
            validation_logits.reshape(-1, 256), validation_y.reshape(-1)))
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    export_to_bin(model, str(target / "NppAI-repository-baseline.nppai"))
    (target / "bpe_merges.txt").write_text("", encoding="utf-8")
    metadata = {"steps": steps, "seed": seed, "block_size": block_size,
                "files": len(files), "corpus_bytes": len(corpus.encode("utf-8")),
                "corpus_sha256": hashlib.sha256(corpus.encode("utf-8")).hexdigest(),
                "initial_loss": losses[0], "final_loss": losses[-1],
                "validation_loss": round(validation_loss, 5),
                "validation_tokens": len(val_data),
                "limitations": "Tiny repo-only baseline; no verified coding competence"}
    (target / "training-report.json").write_text(json.dumps(metadata, indent=2) + "
",
                                                 encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(train(args.repo, args.output, steps=args.steps), indent=2))


if __name__ == "__main__":
    main()
