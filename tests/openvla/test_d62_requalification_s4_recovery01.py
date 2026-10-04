from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_recovery_uses_explicit_compact_position_map_everywhere() -> None:
    selector = (ROOT / "src/savr/openvla/official_semantics.py").read_text(encoding="utf-8")
    assert "active_positions" in selector
    assert "cache compaction removed an official action state" in selector
    for relative in (
        "src/savr/brace/b3_openvla.py",
        "src/savr/pair/p3_openvla.py",
        "src/savr/pair/p4_openvla.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "cache_fork_active_positions" in source
        assert "-57:-1" not in source


def test_tiny_qualification_exercises_real_fork_and_adversarial_contracts() -> None:
    source = (
        ROOT / "scripts/run_openvla_compact_position_qualification.py"
    ).read_text(encoding="utf-8")
    for required in (
        "LlamaForCausalLM",
        "past_key_values=anchor.past_key_values",
        "cache_fork_active_positions",
        "select_official_action_hidden",
        '"missing": "removed an official action state"',
        '"duplicate": "sorted unique"',
        '"length": "does not match hidden states"',
        '"openvla_checkpoint_loads": 0',
        '"simulator_calls": 0',
    ):
        assert required in source
    assert "from_pretrained" not in source


def test_recovery_launcher_is_single_conditional_attempt_without_retry() -> None:
    source = (
        ROOT / "scripts/launch_openvla_d62_requalification_s4_recovery01.sh"
    ).read_text(encoding="utf-8")
    assert source.count("run_openvla_compact_position_qualification.py") == 1
    assert source.count("run_openvla_d62_requalification_s4_recovery01.py") == 1
    assert "while " not in source and "retry" not in source.lower()
    assert "rm " not in source and "sudo" not in source
