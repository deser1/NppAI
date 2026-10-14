"""Training-only eight-form policy, frozen targets and replay policy checks."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

from scripts.prompt_robustness import (augment_training_records, training_variants,
    evaluation_variants, validate_training_policy, check_augmentation_source)

ROOT = Path(__file__).resolve().parents[1] / "evaluation/instruction-v3"


class DiverseWordingTests(unittest.TestCase):
    def test_header_cli_imports_from_script_path_without_pythonpath(self):
        script = ROOT.parents[1] / "scripts/train_identifier_headers.py"
        for extra, expected in ((["--steps", "0"], "steps must be 1..10000"),
                                (["--augmentation-policy", "eight_forms"], "requires --augment-prompts")):
            result = subprocess.run([sys.executable, "-I", str(script),
                                     "--dataset", str(ROOT), "--output", ".", *extra],
                                    cwd=ROOT, capture_output=True, text=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(expected, result.stderr)
            self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_eight_forms_preserve_targets_and_exclude_both_probe_forms(self):
        records = json.loads((ROOT / "train.json").read_text())
        expanded = augment_training_records(records, "eight_forms")
        self.assertEqual(len(expanded), 12288)
        self.assertEqual(len({r["id"] for r in expanded}), 12288)
        by_id = {r["id"]: r for r in records}
        for derived in expanded:
            original = by_id[derived["source_id"]]
            self.assertEqual(derived["answer"], original["answer"])
            self.assertEqual(derived["function"], original["function"])
            probes = evaluation_variants(original, True)
            self.assertNotIn(derived["prompt"], [probes[v]["prompt"] for v in ("novel_polite_python", "novel_implementation")])
            sequence = "[USER]: " + derived["prompt"] + "\n[AI]:\n" + derived["answer"]
            self.assertLessEqual(len(sequence.encode()) + 1, 256)
        self.assertEqual(augment_training_records(records), augment_training_records(records, "four_forms"))

    def test_invalid_or_disabled_policy_rejected(self):
        with self.assertRaises(ValueError):
            validate_training_policy(False, "eight_forms")
        with self.assertRaises(ValueError):
            training_variants({"prompt": "task"}, "unknown")

    def test_replay_rejects_other_policy_and_wrong_counts(self):
        metadata = {"policy": "eight_forms", "source_records": 1536,
                    "derived_records": 12288,
                    "variants": list(training_variants({"prompt": "task"}, "eight_forms"))}
        check_augmentation_source({"prompt_augmentation": metadata}, "eight_forms")
        with self.assertRaises(ValueError):
            check_augmentation_source({"prompt_augmentation": metadata}, "four_forms")
        metadata["derived_records"] = 6144
        with self.assertRaises(ValueError):
            check_augmentation_source({"prompt_augmentation": metadata}, "eight_forms")
