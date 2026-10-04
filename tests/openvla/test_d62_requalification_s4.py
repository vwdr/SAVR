from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "openvla_d62_requalification_s4",
    ROOT / "scripts/run_openvla_d62_requalification_s4.py",
)
assert SPEC is not None and SPEC.loader is not None
S4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S4)


def test_active_paths_have_no_shifted_slice_and_share_official_selector() -> None:
    for relative in (
        "src/savr/brace/b3_openvla.py",
        "src/savr/pair/p3_openvla.py",
        "src/savr/pair/p4_openvla.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "-57:-1" not in source
        assert "select_official_action_hidden" in source
    p3 = (ROOT / "src/savr/pair/p3_openvla.py").read_text(encoding="utf-8")
    assert "derive_semantic_runtime_positions" in p3


def test_selection_digest_covers_order_and_onset_assignments() -> None:
    class Ordered:
        def __init__(self, values):
            self.values = values

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.values

    class Camera:
        value = "primary"

    camera = Camera()
    first = S4.selection_digest(Ordered([1, 2]), {(camera, 0): 2})
    assert first == S4.selection_digest(Ordered([1, 2]), {(camera, 0): 2})
    assert first != S4.selection_digest(Ordered([2, 1]), {(camera, 0): 2})
    assert first != S4.selection_digest(Ordered([1, 2]), {(camera, 0): 6})


def test_s4_worker_freezes_complete_bounded_controls() -> None:
    source = (ROOT / "scripts/run_openvla_d62_requalification_s4.py").read_text(
        encoding="utf-8"
    )
    for required in (
        "OfficialBoundaryCapture",
        "capture_reused_cache",
        "verify_reused_cache",
        "clone_runtime_cache",
        "recursive_cycle",
        "selection_materially_changed",
        '"planned_model_calls": 37',
        '"model_call_hard_cap": 48',
        '"automatic_retry": False',
    ):
        assert required in source
    assert "terminal_success" not in source and "reward" not in source


def test_s4_launcher_has_one_attempt_and_no_retry_or_cleanup() -> None:
    source = (ROOT / "scripts/launch_openvla_d62_requalification_s4.sh").read_text(
        encoding="utf-8"
    )
    assert source.count("run_openvla_d62_requalification_s4.py") == 1
    assert "while " not in source and "retry" not in source.lower()
    assert "rm " not in source and "sudo" not in source
