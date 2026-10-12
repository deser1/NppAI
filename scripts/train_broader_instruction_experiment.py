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
from scripts.identifier_subset import select_training_records, selection_metadata


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


def identifier_weights(record, context, weight):
    """Weight only response name bytes; prompt and padding stay ignored."""
    x, y = sample(record, context)
    weights = torch.ones_like(y, dtype=torch.float32)
    prefix_length = len(prompt(record["prompt"]).encode("utf-8"))
    expected_header = "def " + record["function"] + "("
    if not record["answer"].startswith(expected_header):
        raise ValueError("reference must start with its declared function")
    start = prefix_length - 1 + len("def ")
    weights[start:start + len(record["function"].encode("utf-8"))] = weight
    weights[y == -100] = 0
    return x, y, weights


def weighted_response_loss(logits, targets, weights):
    losses = F.cross_entropy(logits.reshape(-1,256), targets.reshape(-1),
                             ignore_index=-100, reduction="none")
    flat_weights = weights.reshape(-1)
    return (losses * flat_weights).sum() / flat_weights.sum()


def evaluate(model, records, context):
    cases = []
    with torch.no_grad():
        for record in records:
            text = generate(model, record["prompt"], context)
            syntax = False
            any_function = False
            body_match = False
            reference_match = False
            try:
                tree = ast.parse(text)
                any_function = any(isinstance(n,ast.FunctionDef) for n in tree.body)
                syntax = any(isinstance(n, ast.FunctionDef) and n.name == record["function"] for n in tree.body)
                reference = ast.parse(record["answer"])
                reference_match = ast.dump(tree) == ast.dump(reference)
                if len(tree.body)==1 and isinstance(tree.body[0],ast.FunctionDef):
                    # Diagnostic only: ignore the function name in both ASTs.
                    # This does not count as passing the requested task.
                    tree.body[0].name = "__function__"
                    reference.body[0].name = "__function__"
                    body_match = ast.dump(tree)==ast.dump(reference)
            except SyntaxError:
                pass
            x, y = sample(record, context)
            loss = F.cross_entropy(model(x.unsqueeze(0)).reshape(-1,256), y, ignore_index=-100)
            cases.append({"id":record["id"], "family":record["family"], "output":text,
                "nonempty":bool(text.strip()), "syntax_requested_function":syntax,
                "syntax_any_function":any_function,
                "ast_match_ignoring_function_name":body_match,
                "reference_ast":reference_match,
                "exact_match":text == record["answer"], "response_loss":float(loss)})
    return {"total":len(cases), "nonempty":sum(c["nonempty"] for c in cases),
        "syntax_requested_function":sum(c["syntax_requested_function"] for c in cases),
        "syntax_any_function":sum(c["syntax_any_function"] for c in cases),
        "ast_match_ignoring_function_name":sum(c["ast_match_ignoring_function_name"] for c in cases),
        "reference_ast":sum(c["reference_ast"] for c in cases),
        "exact_match":sum(c["exact_match"] for c in cases),
        "mean_response_loss":sum(c["response_loss"] for c in cases)/len(cases), "cases":cases}


def run(dataset, output, steps=3000, identifier_weight=1.0, training_names=None, initial_model=None):
    if not 1 <= steps <= 10000:
        raise ValueError("steps must be 1..10000")
    if not 1.0 <= identifier_weight <= 8.0:
        raise ValueError("identifier weight must be 1..8")
    torch.manual_seed(42)
    random.seed(42)
    torch.set_num_threads(2)
    root = Path(dataset)
    manifest = json.loads((root/"manifest.json").read_text())
    training = select_training_records(read_split(root, "train", manifest), training_names)
    context = 256
    samples = [identifier_weights(record, context, identifier_weight) for record in training]
    model = NppAIModel(256,64,128,2,context)
    if initial_model is not None:
        from scripts.check_dense_runtime_parity import load_weights, torch_model
        from scripts.identifier_curriculum import validate_initial_weights
        initial_weights = load_weights(initial_model)
        validate_initial_weights(initial_weights)
        model.load_state_dict(torch_model(initial_weights).state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses=[]
    model.train()
    for step in range(steps):
        batch = random.sample(samples,4)
        x = torch.nn.utils.rnn.pad_sequence([b[0] for b in batch], batch_first=True)
        y = torch.nn.utils.rnn.pad_sequence([b[1] for b in batch], batch_first=True, padding_value=-100)
        weights = torch.nn.utils.rnn.pad_sequence([b[2] for b in batch], batch_first=True, padding_value=0)
        loss = weighted_response_loss(model(x), y, weights)
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
    if training_names is not None:
        actual_families = {record["family"] for record in training}
        unseen = {case["family"] for case in test["cases"] if case["family"] not in actual_families}
    test["by_family_exposure"] = {}
    for label, select in (("seen_family_new_task",False),("unseen_family",True)):
        cases=[case for case in test["cases"] if (case["family"] in unseen)==select]
        test["by_family_exposure"][label]={"total":len(cases),
            "syntax_requested_function":sum(case["syntax_requested_function"] for case in cases),
            "reference_ast":sum(case["reference_ast"] for case in cases),
            "exact_match":sum(case["exact_match"] for case in cases)}
    destination=Path(output)
    destination.mkdir(parents=True,exist_ok=True)
    checkpoint=destination/"NppAI-broader-instruction.nppai"
    export_to_bin(model,str(checkpoint))
    report={"seed":42,"steps":steps,"batch_size":4,"context":context,
        "identifier_weight":identifier_weight,
        "architecture":{"vocabulary":256,"dimension":64,"hidden_dimension":128,"layers":2},
        "runtime":{"python":sys.version.split()[0],"torch":torch.__version__,"device":"cpu","threads":2},
        "optimizer":{"name":"AdamW","learning_rate":0.001,"gradient_clip_norm":1.0},
        "training_loss_kind":"weighted response cross-entropy; reported evaluation loss is unweighted",
        "dataset_manifest":manifest,"model_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "initial_training_loss":losses[0],"final_training_loss":losses[-1],
        "validation":validation,"test":test,
        "limitations":"Synthetic templates only, 16 training families, 4 unseen families. No generated code execution or correctness claim. Fixed budget and weight chosen before evaluation; test results are not used to tune training. Operation families were exposed by earlier v1/v2 experiments; v3 holds out new identifiers, not a pristine task benchmark."}
    if initial_model is not None:
        report["initialization"] = {"checkpoint_sha256": hashlib.sha256(Path(initial_model).read_bytes()).hexdigest(),
                                    "method": "all dense FP32 weights; fresh AdamW optimizer"}
    if training_names is not None:
        report["training_selection"] = selection_metadata(training, training_names)
        report["training_selection"]["families"] = sorted(actual_families)
        report["limitations"] = "Synthetic memorization probe with selected training names. Training-set size, family coverage, exposure per example and batch sampling may differ between arms. Family exposure metrics refer to the actual selected optimization set. Previously exposed tasks, one seed, no generated code execution, no native INT8 measurement or general correctness claim. Held-out results do not tune training."
        report["training_probe_first24"] = evaluate(model, training[:24], context)
        print(json.dumps({"training_selection_names": report["training_selection"]["names"],
                          "training_selection_records": len(training),
                          "training_probe_first24": {k: v for k, v in report["training_probe_first24"].items() if k != "cases"}}, indent=2), flush=True)
    (destination/"broader-report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"validation":{k:v for k,v in validation.items() if k!="cases"},
                      "test":{k:v for k,v in test.items() if k!="cases"}},indent=2))
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--steps",type=int,default=3000)
    parser.add_argument("--identifier-weight",type=float,default=1.0)
    parser.add_argument("--training-names", type=int, help="use first N distinct training names; all styles retained")
    parser.add_argument("--initial-model", help="initialize from a compatible dense FP32 checkpoint; optimizer resets")
    args=parser.parse_args()
    run(args.dataset,args.output,args.steps,args.identifier_weight,args.training_names,args.initial_model)
