"""Fresh evaluation views, source isolation and trusted reference correctness."""
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_fresh_probe import build, save, SPECS, FORMS, legacy_context, ROOT


class FreshProbeTests(unittest.TestCase):
    def test_frozen_reproduction_counts_isolation_and_context(self):
        raw, metadata = build()
        self.assertEqual((raw, metadata), build())
        records, manifest = json.loads(raw), json.loads(metadata)
        self.assertEqual(len(records), 144)
        self.assertEqual(len({r['id'] for r in records}), 144)
        self.assertEqual(len({r['task_id'] for r in records}), 36)
        self.assertEqual(len({r['function'] for r in records}), 36)
        names, families, prompts, _ = legacy_context()
        self.assertFalse(names & {r['function'] for r in records})
        self.assertFalse(families & {r['family'] for r in records})
        self.assertFalse(prompts & {r['prompt'] for r in records})
        self.assertEqual(manifest['tasks_sha256'], hashlib.sha256(raw).hexdigest())
        self.assertTrue(manifest['evaluation_only'])
        self.assertLessEqual(manifest['max_full_sequence_bytes'], 256)
        self.assertEqual({len(r['function']) for r in records}, {9, 12, 15})
        for r in records:
            self.assertLessEqual(len(r['answer'].encode()), 160)
            self.assertEqual(r['prompt'].count('('), 1)
            self.assertEqual(r['prompt'].count(')'), 1)

    def test_views_preserve_signature_and_reference(self):
        groups = defaultdict(list)
        for record in json.loads(build()[0]):
            groups[record['task_id']].append(record)
        for records in groups.values():
            self.assertEqual({r['wording'] for r in records}, set(FORMS))
            self.assertEqual(len({r['prompt'] for r in records}), 4)
            self.assertEqual(len({r['answer'] for r in records}), 1)
            self.assertEqual(len({r['function'] for r in records}), 1)

    def test_only_trusted_authored_references_match_edge_case_examples(self):
        # Execute only the fixed project-authored references, never model outputs.
        seen = set()
        for record in json.loads(build()[0]):
            if record['task_id'] in seen:
                continue
            seen.add(record['task_id'])
            tree = ast.parse(record['answer'])
            self.assertEqual(len(tree.body), 1)
            self.assertIsInstance(tree.body[0], ast.FunctionDef)
            self.assertEqual(tree.body[0].name, record['function'])
            namespace = {'__builtins__': {'sum': sum, 'list': list, 'zip': zip}}
            exec(compile(tree, '<authored fresh reference>', 'exec'), namespace)
            for arguments, expected in SPECS[record['family']][3]:
                with self.subTest(task=record['task_id'], arguments=arguments):
                    self.assertEqual(namespace[record['function']](*arguments), expected)

    def test_saved_fixture_is_eval_only_and_tampering_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            save(folder)
            save(folder, check=True)
            self.assertFalse((Path(folder) / 'train.json').exists())
            (Path(folder) / 'tasks.json').write_bytes(b'[]\n')
            with self.assertRaises(ValueError):
                save(folder, check=True)

    def test_committed_files_match_the_generator(self):
        directory = ROOT / 'evaluation/fresh-probe-v1'
        raw, manifest = build()
        self.assertEqual((directory / 'tasks.json').read_bytes(), raw)
        self.assertEqual((directory / 'manifest.json').read_bytes(), manifest)
