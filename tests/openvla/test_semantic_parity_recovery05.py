from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "openvla_semantic_parity_recovery05",
    ROOT / "scripts/run_openvla_semantic_parity_recovery05.py",
)
assert SPEC is not None and SPEC.loader is not None
RECOVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECOVERY)


def test_dense_helper_source_accepts_and_forwards_explicit_false() -> None:
    source = (ROOT / "src/savr/openvla/official_semantics.py").read_text(encoding="utf-8")
    assert "if use_cache not in (None, False, True):" in source
    assert '"use_cache": use_cache,' in source


def test_full_path_qualification_exercises_actual_helper_and_all_modes() -> None:
    source = (
        ROOT / "scripts/run_openvla_dense_helper_qualification.py"
    ).read_text(encoding="utf-8")
    assert source.count("official_semantics.structurally_aligned_dense_forward(") == 1
    assert source.count("official_semantics.structurally_aligned_dense_forward,") == 3
    assert "expected_modes = [None, False, True, True]" in source
    assert '"full_helper_calls": 4' in source
    assert '"openvla_model_calls": 0' in source
    assert "actions_identical" in source


def test_launcher_qualifies_complete_helper_before_one_model_attempt() -> None:
    launcher = (
        ROOT / "scripts/launch_openvla_semantic_parity_recovery05.sh"
    ).read_text(encoding="utf-8")
    qualification = launcher.index("run_openvla_dense_helper_qualification.py")
    model = launcher.index("run_openvla_semantic_parity_recovery05.py")
    assert qualification < model
    assert "CUDA_VISIBLE_DEVICES" in launcher
    assert "while " not in launcher and "retry" not in launcher.lower()
    assert "rm " not in launcher and "sudo" not in launcher
