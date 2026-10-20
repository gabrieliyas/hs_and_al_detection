"""End-to-end Stage 2 preprocessing tests using a generated synthetic dataset.

These tests deliberately use synthetic records with the confirmed schema/counts;
they test workflow and safeguards without depending on the private annotated CSV.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_ROOT = Path(__file__).resolve().parents[1]


def _synthetic_context_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(20261019)
    needed_counts = {
        "N": {1: 180, 0: 420},
        "AL": {1: 300, 0: 300},
        "HS": {1: 365, 0: 235},
    }
    raw_label = {"N": "neither", "AL": "abusive_language", "HS": "hate_speech"}
    rows: list[dict] = []
    index = 0
    for label, cell_counts in needed_counts.items():
        needs = [need for need, count in cell_counts.items() for _ in range(count)]
        rng.shuffle(needs)
        # Per class: 300 C4 records; remaining 300 are shallow (100 per C1-C3).
        availability = [4] * 300 + [1] * 100 + [2] * 100 + [3] * 100
        rng.shuffle(availability)
        for need, available in zip(needs, availability):
            post_id = f"synthetic-{index:04d}"
            index += 1
            row = {
                "post_id": post_id,
                "parent_4": None,
                "parent_3": None,
                "parent_2": None,
                "parent_1": None,
                "target_text": f"synthetic target text {post_id}",
                "context_size_available": available,
                "label": raw_label[label],
                "context_needed": need,
                "HS": int(label == "HS"),
                "AL": int(label == "AL"),
                "target_group": f"group-{index % 4}",
                "target_group_attribute": f"attribute-{index % 5}",
                "context_type": f"type-{index % 3}",
                "collection_method": "synthetic test fixture",
                "keyword_used": "synthetic",
                "post_date": "2025-01-01",
            }
            for parent_number in range(1, available + 1):
                row[f"parent_{parent_number}"] = f"parent {parent_number} for {post_id}"
            rows.append(row)
    return pd.DataFrame(rows)


class ContextualPreprocessIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="context-preprocess-test-")
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir(parents=True)
        (self.root / "data/raw").mkdir(parents=True)
        script_source = SOURCE_ROOT / "scripts/2_preprocess_contextual.py"
        if not script_source.is_file():  # lets this test run before the repo migration is complete
            script_source = SOURCE_ROOT / "2_preprocess_contextual.py"
        (self.root / "scripts/2_preprocess_contextual.py").write_bytes(script_source.read_bytes())
        (self.root / "experiment_utils.py").write_bytes(
            (SOURCE_ROOT / "experiment_utils.py").read_bytes()
        )
        (self.root / "experiment_protocol.md").write_bytes(
            (SOURCE_ROOT / "experiment_protocol.md").read_bytes()
        )
        (self.root / "pipeline_revised.md").write_bytes(
            (SOURCE_ROOT / "pipeline_revised.md").read_bytes()
        )
        _synthetic_context_dataset().to_csv(
            self.root / "data/raw/contextual_dataset.csv", index=False
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _run_stage2(self) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(self.root / "scripts/2_preprocess_contextual.py")],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    def test_stage2_builds_valid_partitions_and_manifest(self) -> None:
        proc = self._run_stage2()
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)

        cleaned = pd.read_csv(self.root / "data/processed/ctx_cleaned.csv")
        matched = pd.read_csv(self.root / "data/processed/ctx_matched_subset.csv")
        shallow = pd.read_csv(self.root / "data/processed/ctx_shallow_pool.csv")
        self.assertEqual((len(cleaned), len(matched), len(shallow)), (1800, 900, 900))
        self.assertTrue(matched["post_id"].is_unique)
        self.assertTrue(shallow["post_id"].is_unique)
        self.assertFalse(set(matched.post_id) & set(shallow.post_id))
        self.assertEqual(matched["fold"].value_counts().sort_index().to_dict(), {i: 180 for i in range(5)})

        manifest_path = self.root / "results/runs/left_truncation_v2/stage_02/_run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["stage"], 2)
        self.assertIn("output_sha256", manifest["metadata"])
        self.assertEqual(len(manifest["metadata"]["output_sha256"]), 4)

        # Any post-run alteration to a declared output must be detected.
        with (self.root / "data/processed/ctx_matched_subset.csv").open("a", encoding="utf-8") as f:
            f.write("\n")
        verify = subprocess.run(
            [sys.executable, "-c", "import experiment_utils as eu; eu.assert_current_stage_manifest(2, '.')"],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertNotEqual(verify.returncode, 0)
        self.assertIn("output has changed", verify.stderr)

    def test_second_run_refuses_to_overwrite_historical_outputs(self) -> None:
        first = self._run_stage2()
        self.assertEqual(first.returncode, 0, msg=first.stdout + "\n" + first.stderr)
        second = self._run_stage2()
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("will not overwrite existing files", second.stderr)


if __name__ == "__main__":
    unittest.main()
