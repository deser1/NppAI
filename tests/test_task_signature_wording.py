"""Explicit signatures cannot use references or claim learned name accuracy."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_task_signature_wording import task_header, verify_inputs
from scripts.prompt_robustness import evaluation_variants, training_variants
from tests.test_wording_report_audit import fixture, ROOT


class TaskSignatureWordingTests(unittest.TestCase):
    def test_all_frozen_forms_derive_only_task_header_and_fit(self):
        for split in ("validation", "test"):
            for record in json.loads((ROOT / (split + ".json")).read_bytes()):
                for derived in evaluation_variants(record, True).values():
                    self.assertEqual(task_header(derived["prompt"]), record["answer"].splitlines(keepends=True)[0])

    def test_ambiguous_or_missing_signature_is_rejected(self):
        for task in ("write code", "Define f(a) and g(b)", "Define f(a=1)"):
            with self.assertRaises(ValueError):
                task_header(task)

    def test_cached_replay_and_body_hashes_are_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            body = source / "fp32/body-source"
            body.mkdir(parents=True)
            model = body / "NppAI-broader-instruction.nppai"
            model.write_bytes(b"fixture model bytes")
            manifest = json.loads((ROOT / "manifest.json").read_bytes())
            metadata = {"model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                        "dataset_manifest": manifest, "seed": 42, "steps": 10000, "context": 256,
                        "prompt_augmentation": {"policy": "four_forms", "source_records": 1536,
                            "derived_records": 6144, "variants": list(training_variants({"prompt": ""}))}}
            (body / "broader-report.json").write_text(json.dumps(metadata), encoding="utf-8")
            report = fixture()
            report["source_body_report"] = metadata
            comparison = {"source_run_id": 38582420228, "source_policy": "four_forms", "infrastructure_errors": 0}
            for runtime, relative in (("fp32", "fp32/staged-report.json"), ("native", "native-report.json")):
                raw = json.dumps(report).encode()
                (source / relative).write_bytes(raw)
                comparison[runtime + "_report_sha256"] = hashlib.sha256(raw).hexdigest()
            (source / "comparison.json").write_text(json.dumps(comparison), encoding="utf-8")
            verify_inputs(source, ROOT)
            model.write_bytes(b"changed checkpoint")
            with self.assertRaises(ValueError):
                verify_inputs(source, ROOT)
            model.write_bytes(b"fixture model bytes")
            (source / "native-report.json").write_bytes(b"changed cached report")
            with self.assertRaises(ValueError):
                verify_inputs(source, ROOT)
