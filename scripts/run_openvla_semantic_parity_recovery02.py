#!/usr/bin/env python3
"""Versioned S3 recovery for canonical proprio comparison and sealed failures."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Mapping


ROOT = Path("/home/ved/SAVR")
DEFAULT_CONFIG = Path("configs/openvla/semantic_parity_s3_v03_recovery02.json")
RECOVERY01_WORKER = ROOT / "scripts/run_openvla_semantic_parity_recovery01.py"
QUALIFIED_LOADER_GUARD_SHA256 = (
    "c4dfb61c9b1d36ebc2cacd0f80ec9349ec304304eac37f609c77f5c140eaa643"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_recovery01() -> Any:
    return load_module(RECOVERY01_WORKER, "openvla_semantic_parity_recovery01_for_r02")


def function_source_sha256(function: Callable[..., Any]) -> str:
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def canonicalize_prepared_proprio(prepared: Any) -> Any:
    normalized = prepared.normalized_proprio
    if getattr(normalized, "numel", lambda: -1)() != 8:
        raise RuntimeError("Recovery 02 expected exactly eight normalized proprio values")
    flattened = normalized.reshape(-1)
    if tuple(flattened.shape) != (8,):
        raise RuntimeError("Recovery 02 failed to canonicalize normalized proprio")
    return dataclasses.replace(prepared, normalized_proprio=flattened)


def json_safe_comparison(record: Mapping[str, Any]) -> dict[str, Any]:
    safe = dict(record)
    maximum = safe.get("max_abs")
    if isinstance(maximum, float) and not math.isfinite(maximum):
        safe["max_abs"] = None
        safe["max_abs_nonfinite"] = "nan" if math.isnan(maximum) else (
            "positive_infinity" if maximum > 0 else "negative_infinity"
        )
    json.dumps(safe, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return safe


def validate_recovery02(
    config: Mapping[str, Any], recovery01: Any, v1: Any, *, require_authorized: bool
) -> None:
    if config.get("schema_version") != "openvla-semantic-parity-s3-recovery02-v1":
        raise v1.ParityTechnicalStop("S3 Recovery 02 config schema changed")
    if config.get("semantic_sha256") != v1.semantic_sha256(config):
        raise v1.ParityTechnicalStop("S3 Recovery 02 config hash mismatch")
    expected_authorized = bool(require_authorized)
    if config.get("authorization") != {
        "s3_recovery02_authorized": expected_authorized,
        "source": "User approved S3 Recovery 02 on 2026-09-01"
        if expected_authorized
        else None,
        "simulator": False,
        "terminal_outcomes": False,
        "training": False,
        "automatic_retry": False,
    }:
        raise v1.ParityTechnicalStop("S3 Recovery 02 authorization changed")
    technical = config.get("technical_recovery")
    expected_technical = {
        "source_run_id": "openvla-semantic-parity-s3-v02-recovery01",
        "source_record": (
            "results/openvla-semantic-parity-s3-v02-recovery01/"
            "technical_stop_reconstruction.json"
        ),
        "source_record_sha256": technical.get("source_record_sha256"),
        "source_model_calls": 4,
        "source_observations_completed": 0,
        "source_worker_terminal_record_written": False,
        "scientific_population_changed": False,
        "scientific_gate_changed": False,
        "automatic_retry": False,
    }
    if technical != expected_technical:
        raise v1.ParityTechnicalStop("S3 Recovery 02 recovery boundary changed")
    source_record = ROOT / technical["source_record"]
    if not source_record.is_file() or file_sha256(source_record) != technical[
        "source_record_sha256"
    ]:
        raise v1.ParityTechnicalStop("S3 Recovery 01 stop record changed")
    source = json.loads(source_record.read_text(encoding="utf-8"))
    if (
        source.get("semantic_sha256") != v1.semantic_sha256(source)
        or source.get("classification")
        != "TECHNICAL_HARNESS_STOP_NO_SEMANTIC_DECISION"
        or source.get("model_calls") != 4
        or source.get("observations_completed") != 0
        or source.get("worker_terminal_record_written") is not False
        or source.get("simulator_outcomes_accessed") is not False
        or source.get("automatic_retry") is not False
    ):
        raise v1.ParityTechnicalStop("S3 Recovery 01 stop contents changed")
    if config.get("corrections") != {
        "canonicalize_normalized_proprio_before_comparison": True,
        "official_shape": [8],
        "custom_previous_shape": [1, 8],
        "model_canonical_shape": [1, 8],
        "model_computation_changed": False,
        "strict_json_nonfinite_comparison_encoding": True,
        "failure_record_required": True,
    }:
        raise v1.ParityTechnicalStop("S3 Recovery 02 correction scope changed")
    if config.get("resource_caps") != {
        "gpu_count": 1,
        "model_processes": 1,
        "observations": 8,
        "calls_per_observation": 4,
        "model_call_hard_cap": 32,
        "wall_seconds": 1800,
        "artifact_bytes": 268435456,
        "peak_gpu_memory_mib_strict_max": 23552,
        "downloads": 0,
        "simulator_outcomes": 0,
        "raw_actions_persisted": False,
        "automatic_retry": False,
    }:
        raise v1.ParityTechnicalStop("S3 Recovery 02 resource boundary changed")
    if config.get("advance") != {
        "next_stage": "S4_D62_REQUALIFICATION",
        "authorized": False,
        "stop_before_next_stage": True,
    }:
        raise v1.ParityTechnicalStop("S3 Recovery 02 advance boundary changed")
    if config.get("output_root") != "results/openvla-semantic-parity-s3-v03-recovery02":
        raise v1.ParityTechnicalStop("S3 Recovery 02 output root changed")
    if config.get("loader_recovery") != {
        "official_source_tree": "third_party/openvla-oft",
        "official_source_revision": "e4287e94541f459edc4feabc4e181f537cd569a8",
        "disable_model_logic_sync": True,
        "qualified_loader_guard_sha256": QUALIFIED_LOADER_GUARD_SHA256,
        "require_checkpoint_prompt_derived_action_readout": True,
        "reject_cache_tail_action_readout": True,
        "adapt_official_return_without_changing_actions": True,
        "discard_vla_cache_only_config_argument": True,
        "inject_absent_dense_cache_controls_as_none": True,
    }:
        raise v1.ParityTechnicalStop("S3 Recovery 02 loader boundary changed")
    if function_source_sha256(recovery01.install_official_loader_guard) != (
        QUALIFIED_LOADER_GUARD_SHA256
    ):
        raise v1.ParityTechnicalStop("qualified loader guard changed")
    if (
        config.get("tolerance") != 0.000001
        or config.get("population")
        != {
            "observation_step": 0,
            "selection": "all eight frozen P3 inputs; exactly two per LIBERO suite",
            "expert_actions_accessed": False,
            "terminal_outcomes_accessed": False,
            "locked_state_ids_accessed": False,
        }
        or config.get("call_schedule")
        != {
            "per_observation": [
                "released_official_evaluator",
                "corrected_dense_use_cache_none",
                "corrected_dense_use_cache_true",
                "corrected_dense_use_cache_true_sidecar",
            ],
            "alternate_official_custom_order": True,
            "total_calls": 32,
        }
    ):
        raise v1.ParityTechnicalStop("S3 Recovery 02 scientific contract changed")
    for relative, expected in config["authenticated_files"].items():
        path = ROOT / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise v1.ParityTechnicalStop(f"authenticated Recovery 02 input changed: {relative}")
    manifest = json.loads((ROOT / config["input_manifest"]).read_text(encoding="utf-8"))
    if (
        manifest.get("semantic_sha256") != v1.semantic_sha256(manifest)
        or manifest.get("terminal_outcome_fields_accessed") is not False
        or manifest.get("expert_action_fields_accessed") is not False
        or len(manifest.get("inputs", [])) != 8
        or {row["suite"] for row in manifest["inputs"]}
        != {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
    ):
        raise v1.ParityTechnicalStop("S3 Recovery 02 population changed")
    data_root = ROOT / manifest["data_root_relative"]
    for row in manifest["inputs"]:
        source_path = data_root / row["source_path"]
        if not source_path.is_file() or v1.file_sha256(source_path) != row["source_sha256"]:
            raise v1.ParityTechnicalStop("S3 Recovery 02 observation source changed")
    checkpoint = ROOT / config["model"]["checkpoint_relative"]
    for name, expected in config["model"]["checkpoint_metadata_sha256"].items():
        if not (checkpoint / name).is_file() or v1.file_sha256(checkpoint / name) != expected:
            raise v1.ParityTechnicalStop(f"S3 Recovery 02 checkpoint changed: {name}")
    if (ROOT / config["output_root"]).exists():
        raise v1.ParityTechnicalStop("immutable S3 Recovery 02 output already exists")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f"S3 Recovery 02 must start in {ROOT}")
    recovery01 = load_recovery01()
    v1 = recovery01.load_v1()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    validate_recovery02(config, recovery01, v1, require_authorized=True)

    evaluation, loader_originals = recovery01.install_official_loader_guard()
    from savr.openvla import official_semantics

    original_prepare = official_semantics.prepare_semantic_query
    original_compare = v1.compare_values
    original_validate = v1.validate_static

    def prepared_with_canonical_proprio(*values: Any, **kwargs: Any) -> Any:
        return canonicalize_prepared_proprio(original_prepare(*values, **kwargs))

    def strict_json_compare(*values: Any, **kwargs: Any) -> dict[str, Any]:
        return json_safe_comparison(original_compare(*values, **kwargs))

    official_semantics.prepare_semantic_query = prepared_with_canonical_proprio
    v1.compare_values = strict_json_compare
    v1.validate_static = lambda supplied: validate_recovery02(
        supplied, recovery01, v1, require_authorized=True
    )
    original_argv = sys.argv
    try:
        sys.argv = [str(RECOVERY01_WORKER), "--config", str(args.config)]
        return int(v1.main())
    finally:
        sys.argv = original_argv
        v1.validate_static = original_validate
        v1.compare_values = original_compare
        official_semantics.prepare_semantic_query = original_prepare
        for name, value in loader_originals.items():
            setattr(evaluation, name, value)


if __name__ == "__main__":
    raise SystemExit(main())
