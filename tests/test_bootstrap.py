from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BootstrapTests(unittest.TestCase):
    def test_schemas_are_valid_json_objects(self) -> None:
        for name in ("episode_result.schema.json", "run_manifest.schema.json"):
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
            "4a0fe130f1cbc5557f77a518dcb65a703a647b1c4b8091499d8bfd8e10ab6e4f",
        )
        self.assertIn("4a0fe130f1cbc5557f77a518dcb65a703a647b1c4b8091499d8bfd8e10ab6e4f", text)
        self.assertIn("Do not edit the manuscript", text)

    def test_project_status_does_not_claim_results(self) -> None:
        text = (ROOT / "PROJECT_STATUS.md").read_text(encoding="utf-8")
        self.assertIn("no empirical SAVR performance claim is supported yet", text)

    def test_exactly_one_phase_is_in_progress(self) -> None:
        text = (ROOT / "docs" / "MILESTONES.md").read_text(encoding="utf-8")
        milestone_rows = [line for line in text.splitlines() if line.startswith("| ") and ". " in line]
        self.assertEqual(sum(" IN_PROGRESS " in line for line in milestone_rows), 1)

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


if __name__ == "__main__":
    unittest.main()
