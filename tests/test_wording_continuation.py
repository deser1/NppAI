"""Continuation must preserve the selected sources, budgets and all weights."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prompt_robustness import training_variants
from scripts.run_wording_continuation import verify_source, check_continuation, continuation_metadata, TASKS

ROOT = Path(__file__).resolve().parents[1] / "evaluation/instruction-v3"


class WordingContinuationTests(unittest.TestCase):
    def metadata(self, policy="eight_forms"):
        return {"steps": 10000, "reload_outputs_identical": True,
            "continuation": continuation_metadata("source-hash"),
            "initialization": {"checkpoint_sha256": "source-hash", "method": "all dense FP32 weights; fresh AdamW optimizer"},
            "prompt_augmentation": {"policy": policy, "source_records": 1536,
                "derived_records": 1536 * len(training_variants({"prompt": ""}, policy)),
                "variants": list(training_variants({"prompt": ""}, policy))}}

    def test_budget_initialization_policy_and_reload_must_match(self):
        baseline = self.metadata()
        check_continuation(baseline, "source-hash", "eight_forms")
        for field, value in (("source_checkpoint_sha256", "wrong"), ("total_optimizer_updates", 10000), ("optimizer_reset", False)):
            report = copy.deepcopy(baseline)
            report["continuation"][field] = value
            with self.assertRaises(ValueError):
                check_continuation(report, "source-hash", "eight_forms")
        with self.assertRaises(ValueError):
            check_continuation(baseline, "source-hash", "four_forms")
        baseline["reload_outputs_identical"] = False
        with self.assertRaises(ValueError):
            check_continuation(baseline, "source-hash", "eight_forms")

    def test_frozen_source_hashes_and_preserved_source_copies(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            manifest = json.loads((ROOT / "manifest.json").read_bytes())
            for prefix in ("", "initial-"):
                for task, (role, model_name, report_name, manifest_key) in TASKS.items():
                    target = source / (prefix + role)
                    target.mkdir()
                    model_bytes = task.encode()
                    (target / model_name).write_bytes(model_bytes)
                    report = {"model_sha256": hashlib.sha256(model_bytes).hexdigest(), manifest_key: manifest,
                              "seed": 42, "steps": 10000, "context": 256,
                              "prompt_augmentation": self.metadata("four_forms")["prompt_augmentation"]}
                    (target / report_name).write_bytes(json.dumps(report).encode())
            actual, copies = verify_source(source, ROOT), verify_source(source, ROOT, "initial-")
            self.assertEqual([actual[k]["sha256"] for k in TASKS], [copies[k]["sha256"] for k in TASKS])
            (source / "initial-header-source" / TASKS["headers"][1]).write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify_source(source, ROOT, "initial-")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch required for real initialization check")
    def test_header_initialization_loads_every_exported_weight(self):
        import torch
        from train_nppai import NppAIModel, export_to_bin
        from scripts.train_identifier_headers import initialize_header_weights
        torch.manual_seed(42)
        source = NppAIModel(256, 64, 128, 2, 256)
        torch.manual_seed(43)
        target = NppAIModel(256, 64, 128, 2, 256)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "initial.nppai"
            export_to_bin(source, str(checkpoint))
            metadata = initialize_header_weights(target, checkpoint)
            self.assertEqual(metadata["checkpoint_sha256"], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            for key, weight in source.state_dict().items():
                self.assertTrue(torch.equal(weight, target.state_dict()[key]), key)
