from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BootstrapTests(unittest.TestCase):
    def test_schemas_are_valid_json_objects(self) -> None:
        for name in (
            "episode_result.schema.json",
            "query_record.schema.json",
            "run_manifest.schema.json",
        ):
            data = json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))
            self.assertEqual(data["type"], "object")
            self.assertTrue(data["required"])

    def test_manuscript_source_is_recorded(self) -> None:
        text = (ROOT / "manuscript" / "README.md").read_text(encoding="utf-8")
        manuscript = ROOT / "manuscript" / (
            "State-Aware Visual Refresh for Efficient VLA Inference.tex"
        )
        self.assertTrue(manuscript.is_file())
        digest = hashlib.sha256(manuscript.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "6fffdbf42a19ba81f1644582a03e685cb0bf3aa9a3dd0ad3cab2cf22785bfb20",
        )
        self.assertIn("6fffdbf42a19ba81f1644582a03e685cb0bf3aa9a3dd0ad3cab2cf22785bfb20", text)
        self.assertIn("Do not edit the manuscript", text)

    def test_project_status_does_not_claim_results(self) -> None:
        text = (ROOT / "PROJECT_STATUS.md").read_text(encoding="utf-8")
        self.assertIn("no positive SAVR performance claim is supported", text)

    def test_exactly_one_phase_is_in_progress(self) -> None:
        text = (ROOT / "docs" / "MILESTONES.md").read_text(encoding="utf-8")
        milestone_rows = [line for line in text.splitlines() if line.startswith("| ") and ". " in line]
        active = sum(" IN_PROGRESS " in line for line in milestone_rows)
        self.assertLessEqual(active, 1)
        if active == 0:
            self.assertTrue(
                any(" STOPPED_NEGATIVE " in line for line in milestone_rows)
            )

    def test_phase1_setup_is_project_scoped(self) -> None:
        text = (ROOT / "scripts" / "setup_phase1_environment.sh").read_text(encoding="utf-8")
        self.assertIn('readonly EXPECTED_ROOT="/home/ved/SAVR"', text)
        self.assertNotIn("sudo", text)
        self.assertNotIn("shell init", text)
        self.assertIn("MICROMAMBA_SHA256", text)
        self.assertIn('export LIBERO_CONFIG_PATH="${CACHE_ROOT}/libero"', text)

    def test_phase1_reproducibility_evidence_is_tracked(self) -> None:
        conda_lock = ROOT / "environment" / "locks" / "conda-linux-64-explicit.txt"
        pip_lock = ROOT / "environment" / "locks" / "pip-freeze.txt"
        report = ROOT / "reports" / "PHASE1_REPORT.md"
        self.assertTrue(conda_lock.is_file())
        self.assertTrue(pip_lock.is_file())
        self.assertTrue(report.is_file())
        self.assertIn("@EXPLICIT", conda_lock.read_text(encoding="utf-8"))
        pip_text = pip_lock.read_text(encoding="utf-8")
        self.assertIn("torch==2.2.0+cu118", pip_text)
        self.assertIn(
            "openvla-oft.git@e4287e94541f459edc4feabc4e181f537cd569a8",
            pip_text,
        )

    def test_phase2_checkpoint_download_is_pinned_and_scoped(self) -> None:
        text = (ROOT / "scripts" / "download_phase2_checkpoint.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('EXPECTED_ROOT = Path("/home/ved/SAVR")', text)
        self.assertIn(
            'REVISION = "638918f3d1c2e43a39a8a20772bdb8b91835e4b7"',
            text,
        )
        self.assertIn("EXPECTED_REMOTE_BYTES = 15_939_168_050", text)
        self.assertIn('os.environ["CUDA_VISIBLE_DEVICES"] = ""', text)
        self.assertIn("ADDITIONAL_PROJECT_CAP_BYTES = 20 * 1024**3", text)

    def test_phase2a_smoke_is_single_gpu_and_single_episode(self) -> None:
        text = (ROOT / "scripts" / "run_phase2a_fr_smoke.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('EXPECTED_ROOT = Path("/home/ved/SAVR")', text)
        self.assertIn("TASK_ID = 0", text)
        self.assertIn("INITIAL_STATE_ID = 0", text)
        self.assertIn("num_trials_per_task=1", text)
        self.assertIn("torch.cuda.device_count() != 1", text)
        self.assertIn("upstream_eval.run_task(", text)
        self.assertIn('"policy": "FR"', text)
        self.assertIn('"checkpoint_restored"', text)

    def test_phase2b_pilot_matches_approved_matrix_and_caps(self) -> None:
        text = (ROOT / "scripts" / "run_phase2b_fr_pilot.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("TASK_IDS = tuple(range(10))", text)
        self.assertIn("INITIAL_STATE_IDS = tuple(range(5))", text)
        self.assertIn("EXPECTED_EPISODES = 50", text)
        self.assertIn("ARTIFACT_CAP_BYTES = 2 * 1024**3", text)
        self.assertIn("global_query_index < 3", text)
        self.assertIn("register_forward_pre_hook", text)
        self.assertIn("CUDA_VISIBLE_DEVICES", text)
        self.assertIn('"checkpoint_restored"', text)

    def test_phase2b_analysis_enforces_matrix_and_threshold(self) -> None:
        text = (ROOT / "scripts" / "analyze_phase2b_pilot.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('EXPECTED_ROOT = Path("/home/ved/SAVR")', text)
        self.assertIn("EXPECTED_PAIRS", text)
        self.assertIn("SUCCESS_THRESHOLD = 45", text)
        self.assertIn("checkpoint_restored", text)
        self.assertIn("visual_share_of_total_cuda", text)
        self.assertIn("return 0 if threshold_passed else 2", text)

    def test_phase3_adapter_targets_projected_features_without_upstream_edits(self) -> None:
        design = (ROOT / "docs" / "PHASE3_IMPLEMENTATION_DESIGN.md").read_text(
            encoding="utf-8"
        )
        adapter = (
            ROOT / "src" / "savr" / "integration" / "openvla_oft.py"
        ).read_text(encoding="utf-8")
        self.assertIn("safe cache target is the output", design)
        self.assertIn("proprioception token", design)
        self.assertIn('METHOD_NAME = "_process_vision_features"', adapter)
        self.assertIn("types.MethodType", adapter)
        self.assertIn("finally:", adapter)
        self.assertNotIn("third_party", adapter)

    def test_phase4_proposal_is_bounded_and_still_gated(self) -> None:
        text = (ROOT / "docs" / "PHASE4_CORRECTNESS_PROPOSAL.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("APPROVED AND EXECUTED", text)
        self.assertIn("at most `6`", text)
        self.assertIn("`45 minutes`", text)
        self.assertIn("`256 MiB`", text)
        self.assertIn("numpy.array_equal", text)
        self.assertIn("zero rollout episodes", text)
        self.assertIn("Do not silently retry", text)

    def test_phase4_runner_is_fail_closed_and_bounded(self) -> None:
        text = (ROOT / "scripts" / "run_phase4_correctness.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('EXPECTED_ROOT = Path("/home/ved/SAVR")', text)
        self.assertIn('RUN_ID = "phase4-correctness-v1"', text)
        self.assertIn("QUERY_COUNT = 6", text)
        self.assertIn("ARTIFACT_CAP_BYTES = 256 * 1024**2", text)
        self.assertIn("HF_HUB_OFFLINE", text)
        self.assertIn("TRANSFORMERS_OFFLINE", text)
        self.assertIn("torch.cuda.device_count() != 1", text)
        self.assertIn("np.array_equal", text)
        self.assertNotIn("env.step(", text)

    def test_phase5_runner_matches_the_approved_diagnostic_matrix(self) -> None:
        protocol = (ROOT / "docs" / "PHASE5_SMOKE_PROTOCOL.md").read_text(
            encoding="utf-8"
        )
        runner = (ROOT / "scripts" / "run_phase5_core_smoke.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("Status: **APPROVED FOR EXECUTION**", protocol)
        self.assertIn("Episodes: exactly `12`", protocol)
        self.assertIn("Phase 6 remains", protocol)
        self.assertIn("unauthorized.", protocol)
        self.assertIn('EXPECTED_ROOT = Path("/home/ved/SAVR")', runner)
        self.assertIn('RUN_ID = "phase5-core-smoke-v1"', runner)
        self.assertIn("ARTIFACT_CAP_BYTES = 1024**3", runner)
        self.assertIn("WALL_CAP_SECONDS = 2 * 60 * 60", runner)
        self.assertIn("torch.cuda.device_count() != 1", runner)
        self.assertIn("HF_HUB_OFFLINE", runner)
        self.assertIn("TRANSFORMERS_OFFLINE", runner)
        self.assertIn("validate_complete_matrix", runner)
        self.assertEqual(runner.count('f"wrapped_{policy.lower()}"'), 1)

    def test_vla_cache_audit_is_pinned_isolated_and_cpu_only(self) -> None:
        setup = (
            ROOT / "scripts" / "setup_vla_cache_compatibility.sh"
        ).read_text(encoding="utf-8")
        audit = (
            ROOT / "scripts" / "audit_vla_cache_compatibility.py"
        ).read_text(encoding="utf-8")
        for revision in (
            "a4909880573868dee2769343d52e793c0341678b",
            "9a90a37acacf453433168db8d7769b7ea3c40c06",
        ):
            self.assertIn(revision, setup)
            self.assertIn(revision, audit)
        self.assertIn("envs/vla-cache-compat", setup)
        self.assertIn("--system-site-packages", setup)
        self.assertIn("--no-deps", setup)
        self.assertNotIn("sudo", setup)
        self.assertIn('os.environ["CUDA_VISIBLE_DEVICES"] = ""', audit)
        self.assertIn("TECHNICAL_EXCLUSION", audit)
        self.assertIn("prev_img = replay_images[-1]", audit)

    def test_phase5_analysis_reconciles_core_and_external_evidence(self) -> None:
        text = (ROOT / "scripts" / "analyze_phase5_smoke.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('CORE_RUN_ID = "phase5-core-smoke-v1"', text)
        self.assertIn(
            'VLA_CACHE_RUN_ID = "phase5-vla-cache-compatibility-v1"',
            text,
        )
        self.assertIn("validate_episode_matrix", text)
        self.assertIn("validate_queries", text)
        self.assertIn("checkpoint_restored", text)
        self.assertIn("TECHNICAL_EXCLUSION", text)
        self.assertIn("Phase 6 calibration remains required", text)

    def test_phase5_report_preserves_the_smoke_claim_boundary(self) -> None:
        text = (ROOT / "reports" / "PHASE5_SMOKE_REPORT.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("TECHNICALLY COMPLETE; AWAITING CHECKPOINT REVIEW", text)
        self.assertIn("All 12 fixed episodes", text)
        self.assertIn("283 policy queries", text)
        self.assertIn("deliberately aggressive, uncalibrated diagnostic reuse", text)
        self.assertIn("technically excluded", text)
        self.assertIn("Phase 6 calibration remains unauthorized", text)
        self.assertIn("setup deviation occurred and was remediated", text)


if __name__ == "__main__":
    unittest.main()
