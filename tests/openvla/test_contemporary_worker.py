import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_contemporary_reference as worker


def configuration():
    reference = json.loads((ROOT / "configs/openvla/original_baseline_40task_v1.json").read_text())
    manifest = json.loads((ROOT / "configs/pair/p3_inputs_v1.json").read_text())
    old = json.loads((ROOT / worker.REQUIRED_FILES[-1]).read_text())
    config = dict(schema_version="contemporary-reference-executable-v1", launch_ready=True,
        mode="hook_qualification", automatic_retry=False, tolerance=1e-6, caps=worker.HOOK_CAPS.copy(),
        output_root="results/contemporary-hook-qualification-v01",
        authenticated_files={p: worker.base.sha(ROOT / p) for p in worker.REQUIRED_FILES},
        worker_sha256=worker.base.sha(ROOT / "scripts/run_contemporary_reference.py"),
        analyzer_sha256=worker.base.sha(ROOT / "scripts/analyze_contemporary_reference.py"),
        initial_state_sha256={r["condition_id"]: r["initial_state_sha256"] for r in old["records"]},
        observation_ids=[r["trajectory_id"] for r in manifest["inputs"]], gpu=reference["gpu"])
    return config, reference, manifest


class WorkerTests(unittest.TestCase):
    def test_preflight_reads_actual_reference_schema(self):
        c, ref, manifest = configuration()
        with patch.object(worker, "ROOT", ROOT), patch.object(worker.prior, "verify", return_value=(ref, manifest)):
            design, actual, _ = worker.verify(c)
        self.assertEqual(actual, ref)
        self.assertEqual(len(design["episode_slots"]), 88)

    def test_unsafe_paths_rejected(self):
        with patch.object(worker, "ROOT", ROOT):
            for name in ("../unrelated", "/tmp/unrelated", str(ROOT / "test")):
                with self.assertRaises(ValueError): worker.scoped(name)

    def test_unfrozen_or_changed_config_rejected(self):
        for mutation in (lambda c: c.update(launch_ready=False),
                         lambda c: c.update(automatic_retry=True),
                         lambda c: c["authenticated_files"].pop("src/savr/openvla/specprune.py"),
                         lambda c: c.update(worker_sha256="f" * 64),
                         lambda c: c["initial_state_sha256"].pop(next(iter(c["initial_state_sha256"]))),
                         lambda c: c["observation_ids"].reverse()):
            c, ref, manifest = configuration(); mutation(c)
            with patch.object(worker, "ROOT", ROOT), patch.object(worker.prior, "verify", return_value=(ref, manifest)):
                with self.assertRaises(ValueError): worker.verify(c)


if __name__ == "__main__": unittest.main()
