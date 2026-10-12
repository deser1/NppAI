"""Compare a v2 byte-token dense checkpoint in NumPy, PyTorch and native C++."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile

import numpy as np


def load_weights(path):
    raw = Path(path).read_bytes()
    if raw[:8] != b"NPPAI\0\0\0" or struct.unpack_from("<I", raw, 8)[0] != 2:
        raise ValueError("only dense v2 checkpoints are supported")
    dim, hidden, layers, context, vocab = struct.unpack_from("<5i", raw, 12)
    size = struct.unpack_from("<Q", raw, 32)[0]
    if size != len(raw) - 72 or hashlib.sha256(raw[72:]).digest() != raw[40:72]:
        raise ValueError("invalid payload size or hash")
    if vocab != 256 or min(dim, hidden, layers, context) <= 0:
        raise ValueError("requires byte-token dense model")
    data = np.frombuffer(raw, dtype="<f4", offset=72)
    offset = 0

    def take(*shape):
        nonlocal offset
        count = int(np.prod(shape))
        value = data[offset:offset + count].reshape(shape).copy()
        offset += count
        return value

    weights = {"embedding": take(vocab, dim), "position": take(context, dim), "layers": []}
    for _ in range(layers):
        weights["layers"].append(dict(norm1=take(dim), q=take(dim, dim),
            k=take(dim, dim), v=take(dim, dim), o=take(dim, dim),
            norm2=take(dim), gate=take(dim, hidden), up=take(dim, hidden),
            down=take(hidden, dim)))
    weights.update(norm=take(dim), output=take(dim, vocab), context=context,
                   dim=dim, hidden=hidden)
    if offset != len(data):
        raise ValueError("unexpected tensor payload")
    return weights


def numpy_logits(w, tokens):
    tokens = tokens[-w["context"]:]
    x = w["embedding"][tokens] + w["position"][:len(tokens)]

    def norm(x, weight):
        return x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + 1e-5) * weight

    for layer in w["layers"]:
        h = norm(x, layer["norm1"])
        q, k, v = (h @ layer[key] for key in ("q", "k", "v"))
        scores = q @ k.T / np.sqrt(np.float32(w["dim"]))
        scores[np.triu_indices(len(tokens), 1)] = -np.inf
        attention = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attention /= attention.sum(axis=-1, keepdims=True)
        x = x + attention @ v @ layer["o"]
        h = norm(x, layer["norm2"])
        gate = h @ layer["gate"]
        x = x + (gate / (1 + np.exp(-gate)) * (h @ layer["up"])) @ layer["down"]
    return norm(x[-1:], w["norm"]) @ w["output"]


def torch_model(w):
    import torch
    from train_nppai import NppAIModel
    model = NppAIModel(256, w["dim"], w["hidden"], len(w["layers"]), w["context"])
    with torch.no_grad():
        def assign(param, value):
            param.copy_(torch.from_numpy(value))
        assign(model.tok_embeddings.weight, w["embedding"])
        assign(model.pos_embeddings.weight, w["position"])
        for target, source in zip(model.layers, w["layers"]):
            assign(target.rms_attn.weight, source["norm1"])
            assign(target.rms_ffn.weight, source["norm2"])
            for name, key in (("wq", "q"), ("wk", "k"), ("wv", "v"), ("wo", "o"),
                              ("w1", "gate"), ("w2", "up"), ("w3", "down")):
                assign(getattr(target, name).weight, source[key].T.copy())
        assign(model.norm.weight, w["norm"])
        assign(model.output.weight, w["output"].T.copy())
    return model.eval()


def quantized_weights(w):
    result = copy.deepcopy(w)
    def quantize(value):
        scale = np.max(np.abs(value)) / np.float32(127)
        if scale == 0:
            scale = np.float32(1e-9)
        scaled = value / scale
        rounded = np.sign(scaled) * np.floor(np.abs(scaled) + np.float32(0.5))
        return np.clip(rounded, -127, 127).astype(np.float32) * scale
    for layer in result["layers"]:
        for key in ("q", "k", "v", "o", "gate", "up", "down"):
            layer[key] = quantize(layer[key])
    result["output"] = quantize(result["output"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", default="evaluation/code-quality/manifest.json")
    parser.add_argument("--native")
    parser.add_argument("--torch", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    w = load_weights(args.model)
    quantized = quantized_weights(w)
    model = torch_model(w) if args.torch else None
    cases = []
    for case in json.loads(Path(args.manifest).read_text(encoding="utf-8")):
        prompt = "[USER]: " + case["prompt"] + "\n[AI]:\n"
        tokens = list(prompt.encode("utf-8"))
        results = []
        for step in range(8):
            expected = numpy_logits(w, tokens).reshape(-1)
            expected_native = numpy_logits(quantized, tokens).reshape(-1)
            item = {"step": step, "numpy_token": int(expected.argmax()),
                    "numpy_logit": float(expected.max()),
                    "int8_reference_token": int(expected_native.argmax()),
                    "int8_reference_logit": float(expected_native.max())}
            if model is not None:
                import torch
                with torch.no_grad():
                    actual = model(torch.tensor([tokens[-w["context"]:]], dtype=torch.long))[0, -1].numpy()
                item.update(torch_token=int(actual.argmax()),
                            torch_max_error=float(np.max(np.abs(actual - expected))))
            if args.native:
                with tempfile.TemporaryDirectory() as directory:
                    tokenfile, logfile = Path(directory)/"tokens.txt", Path(directory)/"logits.txt"
                    tokenfile.write_text(" ".join(map(str, tokens)), encoding="ascii")
                    subprocess.run([args.native, args.model, str(tokenfile), str(logfile)], check=True)
                    native = np.loadtxt(logfile, dtype=np.float32).reshape(-1)
                if native.shape != expected.shape:
                    raise ValueError("native logit shape mismatch")
                item.update(native_token=int(native.argmax()),
                            native_max_error=float(np.max(np.abs(native - expected_native))))
            results.append(item)
            tokens.append(int(expected.argmax()))
        cases.append({"id": case["id"], "steps": results})
    report = {"model_sha256": hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
              "note": "Teacher-forced greedy FP32 prefix; full logits compared at eight steps. Native INT8 is compared against dequantized INT8 reference, PyTorch against FP32. RMSNorm epsilon differs: native 1e-5 vs PyTorch default.",
              "cases": cases}
    Path(args.output).write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    for case in cases:
        for item in case["steps"]:
            for backend in ("torch", "native"):
                token_key = "int8_reference_token" if backend == "native" else "numpy_token"
                if backend+"_token" in item and (item[backend+"_token"] != item[token_key] or item[backend+"_max_error"] > 0.001):
                    raise SystemExit("Runtime parity failed: " + backend)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
