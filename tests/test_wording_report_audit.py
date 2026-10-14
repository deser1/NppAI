"""Raw replay evidence audits, including zero-score and tampering cases."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_wording_report import audit, main, METRICS
from scripts.evaluate_staged_generation import score_output
from scripts.prompt_robustness import evaluation_variants

ROOT = Path(__file__).resolve().parents[1] / "evaluation/instruction-v3"


def fixture(success=False):
    report = {"dataset_manifest": json.loads((ROOT / "manifest.json").read_text()), "splits": {}}
    for split in ("validation", "test"):
        records = json.loads((ROOT / (split + ".json")).read_text())
        for variant in evaluation_variants(records[0], True):
            cases = []
            for source in records:
                record = evaluation_variants(source, True)[variant]
                output = record["answer"] if success else "def wrong_name():\n    pass\n"
                cases.append({"id": record["id"], "prompt": record["prompt"],
                              "reference": record["answer"], "staged": score_output(record, output),
                              "single_model": score_output(record, output)})
            report["splits"][split + "/" + variant] = {
                "total": len(cases), "cases": cases,
                **{arm: {m: sum(c[arm][m] for c in cases) for m in METRICS}
                   for arm in ("staged", "single_model")}}
    return report


class WordingReportAuditTests(unittest.TestCase):
    def test_valid_syntax_zero_correct_names_fails_smoke(self):
        result = audit(fixture(), ROOT)
        self.assertFalse(result["wording_smoke_passed"])
        self.assertEqual(result["groups"]["test/novel_implementation"]["staged"]["exact_response"], 0)

    def test_reference_fixture_passes_both_report_schemas(self):
        report = fixture(True)
        self.assertTrue(audit(report, ROOT)["wording_smoke_passed"])
        manifest = report.pop("dataset_manifest")
        report.update(header_source={"dataset_manifest": manifest}, body_source={"dataset_manifest": manifest})
        self.assertTrue(audit(report, ROOT)["wording_smoke_passed"])

    def test_rejects_missing_group_duplicate_ids_and_altered_reference(self):
        for mutate in (
            lambda r: r["splits"].pop("test/novel_implementation"),
            lambda r: r["splits"]["test/original"]["cases"].append(copy.deepcopy(r["splits"]["test/original"]["cases"][0])),
            lambda r: r["splits"]["test/original"]["cases"][0].update(reference="fake"),
            lambda r: r["dataset_manifest"].update(version=999),
        ):
            report = fixture()
            mutate(report)
            with self.assertRaises(ValueError):
                audit(report, ROOT)

    def test_rejects_forged_case_and_aggregate_metrics(self):
        for aggregate in (False, True):
            report = fixture()
            group = report["splits"]["test/novel_implementation"]
            if aggregate:
                group["staged"]["exact_response"] = 28
            else:
                group["cases"][0]["staged"]["exact_response"] = True
            with self.assertRaises(ValueError):
                audit(report, ROOT)

    def test_cli_preserves_negative_evidence_and_strict_mode_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "source.json", Path(folder) / "audit.json"
            source.write_text(json.dumps(fixture()))
            args = ["--report", str(source), "--dataset", str(ROOT), "--output", str(output)]
            self.assertEqual(main(args), 0)
            self.assertEqual(main(args + ["--require-wording-success"]), 2)
            saved = json.loads(output.read_text())
            self.assertFalse(saved["wording_smoke_passed"])
            self.assertEqual(len(saved["source_report_sha256"]), 64)
