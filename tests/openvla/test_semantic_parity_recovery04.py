from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "openvla_semantic_parity_recovery04",
    ROOT / "scripts/run_openvla_semantic_parity_recovery04.py",
)
assert SPEC is not None and SPEC.loader is not None
RECOVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECOVERY)


def test_none_is_translated_only_to_explicit_false() -> None:
    calls = []

    def probe(*values, **kwargs):
        calls.append(kwargs["use_cache"])
        return kwargs["use_cache"]

    assert RECOVERY.explicit_no_cache_forward(probe, use_cache=None) is False
    assert RECOVERY.explicit_no_cache_forward(probe, use_cache=True) is True
    assert calls == [False, True]
    with pytest.raises(RuntimeError, match="invalid cache mode"):
        RECOVERY.explicit_no_cache_forward(probe, use_cache=False)
    with pytest.raises(RuntimeError, match="explicit use_cache"):
        RECOVERY.explicit_no_cache_forward(probe)


def test_pinned_runtime_source_resolves_none_to_config_default() -> None:
    source_path = Path(
        "/home/ved/SAVR/envs/vla-cache-compat/lib/python3.10/site-packages/"
        "transformers/models/llama/modeling_llama.py"
    )
    if not source_path.is_file():
        pytest.skip("pinned remote Transformers source is unavailable")
    source = source_path.read_text(encoding="utf-8")
    assert "use_cache = use_cache if use_cache is not None else self.config.use_cache" in source


def test_launcher_qualifies_cache_semantics_before_one_model_attempt() -> None:
    launcher = (
        ROOT / "scripts/launch_openvla_semantic_parity_recovery04.sh"
    ).read_text(encoding="utf-8")
    qualification = launcher.index("run_openvla_cache_mode_qualification.py")
    model = launcher.index("run_openvla_semantic_parity_recovery04.py")
    assert qualification < model
    assert "CUDA_VISIBLE_DEVICES" in launcher
    assert "while " not in launcher and "retry" not in launcher.lower()
    assert "rm " not in launcher and "sudo" not in launcher
