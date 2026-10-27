"""Focused regression tests for experiment_utils.py protocol safeguards."""
from pathlib import Path
import json
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiment_utils as eu


class FakeTokenizer:
    truncation_side = "right"


class ExperimentUtilsProtocolTests(unittest.TestCase):
    def _make_audited_grid(self, project: Path, backbone: str, *, wrong_seed: bool = False):
        """Create real test manifests/artifacts for one 50-run contextual grid."""
        project.mkdir(parents=True, exist_ok=True)
        script_name = eu.PIPELINE_SCRIPTS[9]
        script_path = project / script_name
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("# test contextual train script\n", encoding="utf-8")
        (project / "experiment_protocol.md").write_text("protocol test\n", encoding="utf-8")
        (project / "pipeline.md").write_text("pipeline test\n", encoding="utf-8")
        stage = 9 if backbone == "indobertweet" else 10
        dataset_hash = eu.sha256_json({"dataset": "matched-900"})
        fold_hash = eu.sha256_json({"folds": "shared-5-fold"})
        records = []
        for context in range(5):
            for fold in range(5):
                for seed in (42, 123):
                    actual_seed = 17 if wrong_seed and context == 0 and fold == 0 and seed == 42 else seed
                    config = {
                        "schema_version": eu.CONFIG_SCHEMA_VERSION,
                        "run_id": eu.RUN_ID,
                        "stage": stage,
                        "script_name": script_name,
                        "backbone": backbone,
                        "context_size": context,
                        "fold": fold,
                        "seed": actual_seed,
                        "dataset_sha256": dataset_hash,
                        "fold_assignment_sha256": fold_hash,
                        "max_length_context": 256,
                        "truncation_side": "left",
                    }
                    run_dir = (project / "results" / "runs" / eu.RUN_ID / f"stage_{stage:02d}" /
                               f"{backbone}_C{context}_f{fold}_s{actual_seed}")
                    manifest = eu.make_run_manifest(
                        stage=stage, script_name=script_name, config=config, project_root=project,
                    )
                    eu.prepare_run_directory(run_dir, manifest)
                    artifact = run_dir / "predictions.npy"
                    artifact.write_bytes(b"test predictions")
                    records.append({
                        "artifact_path": str(artifact),
                        "manifest_fingerprint": manifest["fingerprint"],
                    })
        return records

    def test_build_context_text_supports_c0_through_c4(self):
        row = {
            "parent_4": "oldest",
            "parent_3": "older",
            "parent_2": "middle",
            "parent_1": "newest",
            "target_text": "target",
            "context_size_available": 4,
        }
        expected = {
            0: "target",
            1: "newest [SEP] target",
            2: "middle [SEP] newest [SEP] target",
            3: "older [SEP] middle [SEP] newest [SEP] target",
            4: "oldest [SEP] older [SEP] middle [SEP] newest [SEP] target",
        }
        for context_size, expected_text in expected.items():
            with self.subTest(context_size=context_size):
                self.assertEqual(eu.build_context_text(row, context_size, sep=" [SEP] "), expected_text)

    def test_context_tokenizer_is_left_truncated(self):
        tok = FakeTokenizer()
        eu.configure_tokenizer_for_input(tok, "contextual")
        self.assertEqual(tok.truncation_side, "left")
        eu.validate_truncation_policy({"truncation_side": "left"}, input_mode="contextual")
        with self.assertRaises(ValueError):
            eu.validate_truncation_policy({"truncation_side": "right"}, input_mode="contextual")

    def test_benchmark_tokenizer_is_explicitly_right_truncated(self):
        tok = FakeTokenizer()
        eu.configure_tokenizer_for_input(tok, "target_only")
        self.assertEqual(tok.truncation_side, "right")

    def test_context_selector_uses_paired_parsimony_not_highest_mean_only(self):
        rng = np.random.default_rng(12)
        y = np.tile(np.array([0, 1, 2]), 100)  # 300 rows, balanced
        ids = [f"post-{i:04d}" for i in range(len(y))]
        n = len(y)
        probs = {}
        # C0/C1 are weaker, C2 is close to C4, C3 is slightly between.
        for c, error_n in {0: 65, 1: 45, 2: 12, 3: 10, 4: 8}.items():
            pred = y.copy()
            error_indices = rng.choice(n, size=error_n, replace=False)
            pred[error_indices] = (pred[error_indices] + 1) % 3
            mat = np.full((n, 3), 0.01, dtype=float)
            mat[np.arange(n), pred] = 0.98
            mat /= mat.sum(axis=1, keepdims=True)
            probs[c] = [mat, mat.copy()]
        decision = eu.select_context_size_from_oof(
            y, probs, ids,
            post_ids_by_context={c: ids for c in range(5)},
            backbone="indobertweet", n_bootstrap=300, expected_n=None,
        )
        self.assertIn(decision["selected_context"], range(5))
        self.assertTrue(decision["pooled_oof_macro_f1"]["C4"] >= decision["pooled_oof_macro_f1"]["C2"])
        self.assertIn("run_id", decision)
        self.assertEqual(decision["run_id"], eu.RUN_ID)
        # All comparisons against the top condition are retained for reporting.
        self.assertEqual(
            len(decision["paired_comparisons_top_minus_shallower"]),
            decision["provisional_top_context"],
        )
        top = decision["provisional_top_context"]
        if top == 4:
            comparisons = decision["paired_comparisons_top_minus_shallower"]
            self.assertIn("C4_minus_C2", comparisons)
            self.assertIn("C4_minus_C3", comparisons)
            if comparisons["C4_minus_C2"]["includes_zero"]:
                self.assertEqual(decision["selected_context"], 2)
            elif comparisons["C4_minus_C3"]["includes_zero"]:
                self.assertEqual(decision["selected_context"], 3)
            else:
                self.assertEqual(decision["selected_context"], 4)

    def test_context_selector_rejects_misaligned_post_ids(self):
        y = np.tile([0, 1, 2], 5)
        ids = [f"p{i}" for i in range(len(y))]
        mat = np.full((len(y), 3), 1 / 3)
        with self.assertRaisesRegex(ValueError, "post-ID order"):
            eu.select_context_size_from_oof(
                y, {c: [mat, mat.copy()] for c in range(5)}, ids,
                post_ids_by_context={c: ids[::-1] if c == 3 else ids for c in range(5)},
                expected_n=None, n_bootstrap=10,
            )

    def test_backbone_gate_requires_latency_and_context_decisions(self):
        y = np.tile([0, 1, 2], 10)
        preds = {b: y.copy() for b in eu.BACKBONES}
        with self.assertRaisesRegex(ValueError, "requires measured median inference latency"):
            eu.select_backbone(y, preds, preds)

    def test_versioned_run_directory_rejects_unmanifested_existing_outputs(self):
        temp = Path(tempfile.mkdtemp())
        try:
            project = temp / "project"
            project.mkdir()
            script_name = eu.PIPELINE_SCRIPTS[9]
            script_path = project / script_name
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("# test stage script\n", encoding="utf-8")
            (project / "experiment_protocol.md").write_text("protocol test\n", encoding="utf-8")
            (project / "pipeline.md").write_text("pipeline test\n", encoding="utf-8")
            run_dir = project / "results" / "runs" / eu.RUN_ID / "stage_09" / "indobertweet_C0_f0_s17"
            run_dir.mkdir(parents=True)
            (run_dir / "old_predictions.csv").write_text("old", encoding="utf-8")
            config = {"schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID,
                      "stage": 9, "script_name": script_name, "backbone": "indobertweet", "seed": 17}
            manifest = eu.make_run_manifest(
                stage=9, script_name=script_name, config=config, project_root=project,
            )
            with self.assertRaisesRegex(RuntimeError, "no _run_manifest.json"):
                eu.prepare_run_directory(run_dir, manifest)
        finally:
            shutil.rmtree(temp)

    def test_manifest_allows_only_identical_config_resume(self):
        temp = Path(tempfile.mkdtemp())
        try:
            project = temp / "project"
            project.mkdir()
            script_name = eu.PIPELINE_SCRIPTS[9]
            script_path = project / script_name
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("# test stage script\n", encoding="utf-8")
            (project / "experiment_protocol.md").write_text("protocol test\n", encoding="utf-8")
            (project / "pipeline.md").write_text("pipeline test\n", encoding="utf-8")
            run_dir = project / "results" / "runs" / eu.RUN_ID / "stage_09" / "indobertweet_C0_f0_s17"
            cfg = {"schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID,
                   "stage": 9, "script_name": script_name, "backbone": "indobertweet", "seed": 17}
            manifest = eu.make_run_manifest(stage=9, script_name=script_name, config=cfg, project_root=project)
            eu.prepare_run_directory(run_dir, manifest)
            artifact = run_dir / "prediction.npy"
            artifact.write_bytes(b"prediction")
            eu.prepare_run_directory(run_dir, manifest, allow_resume=True)
            # Artifact consumption requires exact stage, script and known fingerprint.
            verified = eu.assert_current_run_artifact(
                artifact, expected_stage=9, expected_script=script_name,
                expected_fingerprint=manifest["fingerprint"],
            )
            self.assertEqual(verified["fingerprint"], manifest["fingerprint"])
            with self.assertRaisesRegex(RuntimeError, "not 10"):
                eu.assert_current_run_artifact(
                    artifact, expected_stage=10, expected_script=script_name,
                    expected_fingerprint=manifest["fingerprint"],
                )
            changed = eu.make_run_manifest(
                stage=9, script_name=script_name,
                config=dict(cfg, seed=99), project_root=project,
            )
            with self.assertRaisesRegex(RuntimeError, "does not match the current configuration"):
                eu.prepare_run_directory(run_dir, changed)
        finally:
            shutil.rmtree(temp)

    def test_old_backbone_decision_is_rejected(self):
        doc = {
            "decision": "indobertweet",
            "rule_version": "v1",
        }
        with self.assertRaisesRegex(ValueError, "missing"):
            eu.validate_backbone_decision_document(doc)

    def test_mean_only_context_selector_fails_closed(self):
        import pandas as pd
        grid = pd.DataFrame({"condition": [0, 1, 2, 3, 4], "macro_f1": [.5, .6, .7, .8, .9]})
        with self.assertRaisesRegex(RuntimeError, "Mean-only context selection is disabled"):
            eu.select_best_context_from_cv(grid)

    def test_max_length_requires_declared_safe_partition(self):
        lengths = {"indobertweet": [100, 120, 128], "xlmr": [96, 130, 135]}
        result = eu.choose_shared_max_length(lengths, partition="benchmark_train")
        self.assertEqual(result["partition"], "benchmark_train")
        self.assertEqual(result["max_length"] % 8, 0)
        with self.assertRaisesRegex(ValueError, "must explicitly declare partition"):
            eu.choose_shared_max_length(lengths, partition="benchmark_test")

    def test_script_audit_does_not_accept_old_name_as_stage_alias(self):
        temp = Path(tempfile.mkdtemp())
        try:
            old_path = temp / "scripts" / "4_train_e1_indobertweet.py"
            old_path.parent.mkdir(parents=True, exist_ok=True)
            old_path.write_text("# legacy", encoding="utf-8")
            report = eu.audit_pipeline_script_names(temp, require_all=False)
            self.assertIn("scripts/3_train_e1_indobertweet.py", report["missing"])
            self.assertEqual(
                report["rename_leftovers"]["scripts/4_train_e1_indobertweet.py"],
                "scripts/3_train_e1_indobertweet.py",
            )
            with self.assertRaisesRegex(FileNotFoundError, "Do not start the rerun"):
                eu.audit_pipeline_script_names(temp, require_all=True)
        finally:
            shutil.rmtree(temp)

    def test_shared_config_requires_current_schema_and_both_truncation_modes(self):
        import json
        temp = Path(tempfile.mkdtemp())
        try:
            length_path = temp / "sequence_length.json"
            rate_path = temp / "learning_rates.json"
            benchmark_sel = eu.choose_shared_max_length(
                {"indobertweet": [100, 120], "xlmr": [100, 128]},
                partition="benchmark_train",
            )
            eu.save_length_selection_config(benchmark_sel, length_path)
            context_sel = eu.choose_shared_max_length(
                {"indobertweet": [200, 220], "xlmr": [240, 250]},
                partition="contextual_c4",
            )
            eu.save_length_selection_config(context_sel, length_path)
            trial_scores = {
                b: {"1e-05": 0.70, "2e-05": 0.82, "3e-05": 0.78, "5e-05": 0.71}
                for b in eu.BACKBONES
            }
            rates = {
                "schema_version": eu.CONFIG_SCHEMA_VERSION,
                "run_id": eu.RUN_ID,
                "lr_search_grid": list(eu.LR_SEARCH_GRID),
                "lr_tuning_seed": 2026,
                "learning_rate_benchmark": {b: 2e-5 for b in eu.BACKBONES},
                "benchmark_trial_macro_f1": trial_scores,
                "learning_rate_context": {b: 2e-5 for b in eu.BACKBONES},
                "context_trial_macro_f1": trial_scores,
            }
            rate_path.write_text(json.dumps(rates), encoding="utf-8")
            loaded = eu.load_shared_run_config(
                length_path, rate_path, require_context=True, require_context_lr=True,
            )
            self.assertEqual(loaded["sequence_length"]["max_length_context"], 256)
            stale = json.loads(length_path.read_text(encoding="utf-8"))
            stale["schema_version"] = "thesis-run-config-v2"
            length_path.write_text(json.dumps(stale), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema is"):
                eu.load_shared_run_config(length_path, rate_path)
            # Current schema does not excuse carrying forward an untuned LR.
            length_path.write_text(json.dumps({**stale, "schema_version": eu.CONFIG_SCHEMA_VERSION}), encoding="utf-8")
            rates["learning_rate_context"]["xlmr"] = 1e-5
            rate_path.write_text(json.dumps(rates), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not the best rate supported"):
                eu.load_shared_run_config(length_path, rate_path, require_context=True, require_context_lr=True)
        finally:
            shutil.rmtree(temp)

    def test_config_parity_enforces_schema_hashes_and_left_context_truncation(self):
        def cfg(backbone, model_id):
            return {
                "schema_version": eu.CONFIG_SCHEMA_VERSION,
                "run_id": eu.RUN_ID,
                "input_mode": "contextual",
                "dataset_sha256": "dataset-hash",
                "split_seed": 42,
                "eval_seeds_exp1": [42, 123, 2024, 7, 99],
                "optimizer": "AdamW", "weight_decay": 0.01, "warmup_ratio": 0.1,
                "lr_scheduler": "linear", "max_epochs": 5, "early_stopping_patience": 2,
                "loss": "weighted_cross_entropy", "class_weighting_benchmark": "train_only",
                "checkpoint_metric": "macro_f1", "lr_search_grid": list(eu.LR_SEARCH_GRID),
                "lr_tuning_seed": 2026, "mixed_precision": "fp16",
                "input_order": "oldest_to_newest_parents_then_target",
                "max_length_benchmark": 128, "max_length_context": 256,
                "fold_column": "fold", "fold_assignment_sha256": "fold-hash", "n_folds": 5,
                "cv_seeds": [42, 123], "context_order": "parent_4_to_parent_1_then_target",
                "truncation_side": "left", "hf_model_id": model_id,
                "learning_rate_benchmark": 2e-5 if backbone == "indobertweet" else 1e-5,
                "learning_rate_context": 3e-5 if backbone == "indobertweet" else 2e-5,
                "per_device_batch_size": 8, "gradient_accumulation_steps": 2,
            }
        a = cfg("indobertweet", eu.get_backbone("indobertweet").hf_model_id)
        b = cfg("xlmr", eu.get_backbone("xlmr").hf_model_id)
        eu.check_config_parity(a, b, input_mode="contextual")
        b["truncation_side"] = "right"
        with self.assertRaisesRegex(ValueError, "require truncation_side='left'"):
            eu.check_config_parity(a, b, input_mode="contextual")

    def test_context_cv_grid_requires_the_protocol_seed_grid(self):
        temp = Path(tempfile.mkdtemp())
        try:
            project = temp / "project"
            records = self._make_audited_grid(project, "indobertweet")
            report = eu.audit_context_cv_grid(records, backbone="indobertweet")
            self.assertEqual(report["n_runs"], 50)
            self.assertEqual(report["seeds"], [42, 123])
            self.assertEqual(len(report["grid_fingerprint"]), 64)
            bad_records = self._make_audited_grid(temp / "bad_project", "indobertweet", wrong_seed=True)
            with self.assertRaisesRegex(ValueError, "grid audit failed"):
                eu.audit_context_cv_grid(bad_records, backbone="indobertweet")
        finally:
            shutil.rmtree(temp)

    def test_stage11_full_decision_passes_schema_and_rule_validation(self):
        n = 900
        y = np.tile(np.array([0, 1, 2]), 300)
        ids = [f"post-{i:04d}" for i in range(n)]
        p = np.full((n, 3), 0.01, dtype=float)
        p[np.arange(n), y] = 0.98
        p /= p.sum(axis=1, keepdims=True)
        ctx = {
            backbone: eu.select_context_size_from_oof(
                y,
                {c: [p.copy(), p.copy()] for c in range(5)},
                ids,
                post_ids_by_context={c: ids for c in range(5)},
                backbone=backbone,
                n_bootstrap=10_000,
            )
            for backbone in eu.BACKBONES
        }
        grid_audits = {}
        temp = Path(tempfile.mkdtemp())
        try:
            project = temp / "project"
            for backbone in eu.BACKBONES:
                records = self._make_audited_grid(project, backbone)
                grid_audits[backbone] = eu.audit_context_cv_grid(records, backbone=backbone)
        finally:
            # Grid fingerprints and audit reports remain in memory for Stage 11.
            # The temporary artifact files are deleted only after validation.
            shutil.rmtree(temp)
        decision = eu.select_backbone(
            y,
            preds_c_best={b: y.copy() for b in eu.BACKBONES},
            preds_c0={b: y.copy() for b in eu.BACKBONES},
            latency_ms={"indobertweet": 10.0, "xlmr": 8.5},
            latency_machine="test-cpu",
            context_decisions=ctx,
            context_sizes_by_backbone={b: ctx[b]["selected_context"] for b in eu.BACKBONES},
            post_ids=ids,
            post_ids_by_prediction={
                "C_best": {b: ids for b in eu.BACKBONES},
                "C0": {b: ids for b in eu.BACKBONES},
            },
            grid_audits=grid_audits,
            n_bootstrap=10_000,
        )
        eu.validate_backbone_decision_document(decision)
        self.assertEqual(decision["schema_version"], eu.CONFIG_SCHEMA_VERSION)
        self.assertEqual(decision["rule_version"], eu.BACKBONE_RULE_VERSION)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            stale = dict(decision)
            stale["rule_version"] = "old-rule"
            eu.validate_backbone_decision_document(stale)


    def test_training_strategy_decision_accepts_fully_verified_stage15_bundle(self):
        temp = Path(tempfile.mkdtemp())
        try:
            source_root = Path(eu.__file__).resolve().parent
            project = temp / "project"
            for relative in (
                "scripts/11_select_backbone.py", "scripts/15_evaluate_all.py",
                "experiment_protocol.md", "pipeline.md",
            ):
                destination = project / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_root / relative, destination)

            input_file = project / "stage11_input.txt"
            input_file.write_text("stage 11 input\n", encoding="utf-8")
            stage11_dir = project / "results/runs" / eu.RUN_ID / "stage_11/decision"
            stage11_config = {"schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID,
                              "stage": 11, "script_name": eu.PIPELINE_SCRIPTS[11]}
            stage11_manifest = eu.make_run_manifest(
                stage=11, script_name=eu.PIPELINE_SCRIPTS[11], config=stage11_config,
                input_paths=[input_file], project_root=project,
            )
            eu.prepare_run_directory(stage11_dir, stage11_manifest, allow_resume=False)

            ids_hash = eu.sha256_json([f"post-{i:04d}" for i in range(900)])
            mode_doc = {
                "schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID,
                "rule_version": eu.BACKBONE_RULE_VERSION, "post_id_sha256": ids_hash,
                "n_oof_examples": 900, "difference": "itft minus direct", "strategy": "direct",
                "reason": "no_credible_difference_default",
                "bootstrap": {"n_bootstrap": 10_000, "ci_level": 95.0},
            }
            by_mode = {"target_only": mode_doc, "context_aware": mode_doc}
            stage15_dir = project / "results/runs" / eu.RUN_ID / "stage_15"
            stage15_config = {"schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID,
                              "stage": 15, "script_name": eu.PIPELINE_SCRIPTS[15],
                              "strategy_decisions": by_mode}
            stage15_manifest = eu.make_run_manifest(
                stage=15, script_name=eu.PIPELINE_SCRIPTS[15], config=stage15_config,
                input_paths=[stage11_dir / eu.RUN_MANIFEST_NAME],
                metadata={"stage11_manifest_fingerprint": stage11_manifest["fingerprint"]},
                project_root=project,
            )
            eu.prepare_run_directory(stage15_dir, stage15_manifest, allow_resume=False)
            provenance = {
                "stage15_manifest_fingerprint": stage15_manifest["fingerprint"],
                "stage15_output_dir": (Path("results/runs") / eu.RUN_ID / "stage_15").as_posix(),
                "stage11_manifest_fingerprint": stage11_manifest["fingerprint"],
                "strategy_input_post_id_sha256": ids_hash,
            }
            decision = {"schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID,
                        "rule_version": eu.BACKBONE_RULE_VERSION, "by_mode": by_mode,
                        "provenance": provenance}
            bundled_path = stage15_dir / "training_strategy_decision.json"
            bundled_path.write_text(__import__("json").dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            (stage15_dir / "stage15_summary.json").write_text("{}\n", encoding="utf-8")
            index_path = stage15_dir / "artifacts_sha256.json"
            index = {
                "schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID, "stage": 15,
                "script_name": eu.PIPELINE_SCRIPTS[15], "manifest_fingerprint": stage15_manifest["fingerprint"],
                "sha256": {"training_strategy_decision.json": eu.sha256_file(bundled_path),
                           "stage15_summary.json": eu.sha256_file(stage15_dir / "stage15_summary.json")},
            }
            index_path.write_text(__import__("json").dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            canonical = project / eu.TRAINING_STRATEGY_DECISION_PATH
            canonical.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bundled_path, canonical)
            marker = {
                "schema_version": eu.CONFIG_SCHEMA_VERSION, "run_id": eu.RUN_ID, "stage": 15,
                "script_name": eu.PIPELINE_SCRIPTS[15], "manifest_fingerprint": stage15_manifest["fingerprint"],
                "artifact_index_sha256": eu.sha256_file(index_path),
                "strategy_decision_sha256": eu.sha256_file(bundled_path),
                "canonical_strategy_decision_sha256": eu.sha256_file(canonical),
            }
            (stage15_dir / "stage15_complete.json").write_text(__import__("json").dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            loaded = eu.load_training_strategy_decision(canonical, project_root=project)
            self.assertEqual(loaded["by_mode"]["target_only"]["strategy"], "direct")
            # Any post-publication alteration must be detected by the strict loader.
            canonical.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex((RuntimeError, ValueError), "hash|stale|decision"):
                eu.load_training_strategy_decision(canonical, project_root=project)
        finally:
            shutil.rmtree(temp)

    def test_training_strategy_decision_requires_stage15_provenance_by_default(self):
        temp = Path(tempfile.mkdtemp())
        try:
            ids_hash = eu.sha256_json([f"post-{i:04d}" for i in range(900)])
            one = {
                "schema_version": eu.CONFIG_SCHEMA_VERSION,
                "run_id": eu.RUN_ID,
                "rule_version": eu.BACKBONE_RULE_VERSION,
                "post_id_sha256": ids_hash,
                "n_oof_examples": 900,
                "difference": "itft minus direct",
                "strategy": "direct",
                "reason": "no_credible_difference_default",
                "bootstrap": {"n_bootstrap": 10_000, "ci_level": 95.0},
            }
            doc = {
                "schema_version": eu.CONFIG_SCHEMA_VERSION,
                "run_id": eu.RUN_ID,
                "rule_version": eu.BACKBONE_RULE_VERSION,
                "by_mode": {"target_only": one, "context_aware": one},
            }
            path = temp / "strategy.json"
            path.write_text(json.dumps(doc), encoding="utf-8")
            self.assertEqual(eu.load_training_strategy_decision(path, verify_provenance=False)["by_mode"].keys(), doc["by_mode"].keys())
            with self.assertRaisesRegex(ValueError, "provenance"):
                eu.load_training_strategy_decision(path)
        finally:
            shutil.rmtree(temp)



if __name__ == "__main__":
    unittest.main(verbosity=2)
