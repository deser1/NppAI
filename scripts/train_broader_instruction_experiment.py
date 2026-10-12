"""Train on a frozen synthetic split; evaluate held-out tasks without execution."""
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
from scripts.train_instruction_experiment import generate, prompt


def read_split(root, split, manifest):
    raw = (root/f"{split}.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["splits"][split]["sha256"]:
        raise ValueError("frozen dataset hash mismatch: " + split)
    return json.loads(raw)


def sample(record, context):
    prefix = list(prompt(record["prompt"]).encode())
    sequence = prefix + list(record["answer"].encode()) + [0]
    if len(sequence) > context:
        raise ValueError("context is too short for full training example")
    x = torch.tensor(sequence[:-1], dtype=torch.long)
    y = torch.tensor(sequence[1:], dtype=torch.long)
    y[:len(prefix)-1] = -100
    return x, y


def evaluate(model, records, context):
    cases = []
    with torch.no_grad():
        for record in records:
            text = generate(model, record["prompt"], context)
            syntax = False
            try:
                tree = ast.parse(text)
                syntax = any(isinstance(n, ast.FunctionDef) and n.name == record["function"] for n in tree.body)
            except SyntaxError:
                pass
            x, y = sample(record, context)
            loss = F.cross_entropy(model(x.unsqueeze(0)).reshape(-1,256), y, ignore_index=-100)
            cases.append({"id":record["id"], "family":record["family"], "output":text,
                "nonempty":bool(text.strip()), "syntax_requested_function":syntax,
                "exact_match":text == record["answer"], "response_loss":float(loss)})
    return {"total":len(cases), "nonempty":sum(c["nonempty"] for c in cases),
        "syntax_requested_function":sum(c["syntax_requested_function"] for c in cases),
        "exact_match":sum(c["exact_match"] for c in cases),
        "mean_response_loss":sum(c["response_loss"] for c in cases)/len(cases), "cases":cases}


def run(dataset, output, steps=3000):
    if not 1 <= steps <= 10000:
        raise ValueError("steps must be 1..10000")
    torch.manual_seed(42)
    random.seed(42)
    torch.set_num_threads(2)
    root = Path(dataset)
    manifest = json.loads((root/"manifest.json").read_text())
    training = read_split(root, "train", manifest)
    context = 256
    samples = [sample(record, context) for record in training]
    model = NppAIModel(256,64,128,2,context)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses=[]
    model.train()
    for step in range(steps):
        batch = random.sample(samples,4)
        x = torch.nn.utils.rnn.pad_sequence([b[0] for b in batch], batch_first=True)
        y = torch.nn.utils.rnn.pad_sequence([b[1] for b in batch], batch_first=True, padding_value=-100)
        loss = F.cross_entropy(model(x).reshape(-1,256), y.reshape(-1), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step%100==0 or step==steps-1:
            print(f"step={step+1} loss={losses[-1]:.5f}", flush=True)
    model.eval()
    # Validation/test records are first read after optimization has finished.
    validation = evaluate(model, read_split(root,"validation",manifest),context)
    test = evaluate(model, read_split(root,"test",manifest),context)
    unseen = set(manifest["unseen_test_families"])
    test["by_family_exposure"] = {}
    for label, select in (("seen_family_new_task",False),("unseen_family",True)):
        cases=[case for case in test["cases"] if (case["family"] in unseen)==select]
        test["by_family_exposure"][label]={"total":len(cases),
            "syntax_requested_function":sum(case["syntax_requested_function"] for case in cases),
            "exact_match":sum(case["exact_match"] for case in cases)}
    destination=Path(output)
    destination.mkdir(parents=True,exist_ok=True)
    checkpoint=destination/"NppAI-broader-instruction.nppai"
    export_to_bin(model,str(checkpoint))
    report={"seed":42,"steps":steps,"batch_size":4,"context":context,
        "dataset_manifest":manifest,"model_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "initial_training_loss":losses[0],"final_training_loss":losses[-1],
        "validation":validation,"test":test,
        "limitations":"Synthetic templates only, 16 training families, 4 unseen families. No generated code execution or correctness claim. One fixed-budget experiment; test results are not used to tune training."}
    (destination/"broader-report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"validation":{k:v for k,v in validation.items() if k!="cases"},
                      "test":{k:v for k,v in test.items() if k!="cases"}},indent=2))
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--steps",type=int,default=3000)
    args=parser.parse_args()
    run(args.dataset,args.output,args.steps)
