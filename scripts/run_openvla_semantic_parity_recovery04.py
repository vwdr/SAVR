#!/usr/bin/env python3
"""S3 Recovery 04: explicit no-cache control after full harness audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping


ROOT = Path("/home/ved/SAVR")
DEFAULT_CONFIG = Path("configs/openvla/semantic_parity_s3_v05_recovery04.json")
RECOVERY03_WORKER = ROOT / "scripts/run_openvla_semantic_parity_recovery03.py"


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


def load_recovery03() -> Any:
    return load_module(RECOVERY03_WORKER, "openvla_semantic_parity_recovery03_for_r04")


def explicit_no_cache_forward(
    original_forward: Callable[..., dict[str, Any]], *values: Any, **kwargs: Any
) -> dict[str, Any]:
    if "use_cache" not in kwargs:
        raise RuntimeError("Recovery 04 requires an explicit use_cache argument")
    supplied = kwargs["use_cache"]
    if supplied is None:
        kwargs["use_cache"] = False
    elif supplied is not True:
        raise RuntimeError(f"Recovery 04 received invalid cache mode: {supplied!r}")
    return original_forward(*values, **kwargs)


def validate_recovery04(
    config: Mapping[str, Any], recovery03: Any, recovery02: Any, recovery01: Any, v1: Any
) -> None:
    if config.get("schema_version") != "openvla-semantic-parity-s3-recovery04-v1":
        raise v1.ParityTechnicalStop("S3 Recovery 04 schema changed")
    if config.get("semantic_sha256") != v1.semantic_sha256(config):
        raise v1.ParityTechnicalStop("S3 Recovery 04 config hash mismatch")
    if config.get("authorization") != {
        "s3_recovery04_authorized": True,
        "source": "User approved S3 Recovery 04 on 2026-09-01",
        "cache_mode_gpu_qualification": True,
        "semantic_model_attempt": True,
        "simulator": False,
        "terminal_outcomes": False,
        "training": False,
        "automatic_retry": False,
    }:
        raise v1.ParityTechnicalStop("S3 Recovery 04 authorization changed")
    technical = config.get("technical_recovery", {})
    if technical != {
        "source_run_id": "openvla-semantic-parity-s3-v04-recovery03",
        "source_stop": "results/openvla-semantic-parity-s3-v04-recovery03/technical_stop.json",
        "source_stop_sha256": technical.get("source_stop_sha256"),
        "source_traceback": "results/openvla-semantic-parity-s3-v04-recovery03/technical_traceback.log",
        "source_traceback_sha256": technical.get("source_traceback_sha256"),
        "source_model_calls": 4,
        "source_observations_completed": 0,
        "source_ordered_comparisons_passed": 43,
        "scientific_population_changed": False,
        "scientific_gate_changed": False,
        "automatic_retry": False,
    }:
        raise v1.ParityTechnicalStop("S3 Recovery 04 recovery boundary changed")
    stop_path = ROOT / technical["source_stop"]
    traceback_path = ROOT / technical["source_traceback"]
    if (
        not stop_path.is_file()
        or file_sha256(stop_path) != technical["source_stop_sha256"]
        or not traceback_path.is_file()
        or file_sha256(traceback_path) != technical["source_traceback_sha256"]
    ):
        raise v1.ParityTechnicalStop("S3 Recovery 03 evidence changed")
    stop = json.loads(stop_path.read_text(encoding="utf-8"))
    if (
        stop.get("semantic_sha256") != v1.semantic_sha256(stop)
        or stop.get("error_type") != "ParityTechnicalStop"
        or stop.get("error") != "use_cache=None unexpectedly returned a cache"
        or stop.get("model_calls") != 4
        or stop.get("observations_completed") != 0
        or stop.get("checkpoint_restoration_error") is not None
        or stop.get("simulator_outcomes_accessed") is not False
        or stop.get("automatic_retry") is not False
    ):
        raise v1.ParityTechnicalStop("S3 Recovery 03 stop contents changed")
    if config.get("cache_contract") != {
        "runtime_none_resolves_to_config_default": True,
        "runtime_config_use_cache": True,
        "no_cache_control": False,
        "cache_control": True,
        "sidecar_cache_control": True,
        "translate_legacy_none_to_false": True,
        "model_computation_changed_beyond_cache_production": False,
        "remaining_post_comparison_invariants_audited": True,
    }:
        raise v1.ParityTechnicalStop("S3 Recovery 04 cache contract changed")
    if config.get("cache_mode_qualification") != {
        "output_root": "results/openvla-cache-mode-qualification-s3q-v01",
        "gpu_count": 1,
        "synthetic_llama_calls": 3,
        "openvla_model_loads": 0,
        "openvla_model_calls": 0,
        "wall_seconds": 120,
        "artifact_bytes": 1048576,
        "automatic_retry": False,
        "must_pass_before_model_attempt": True,
    }:
        raise v1.ParityTechnicalStop("S3 Recovery 04 cache qualification changed")
    if config.get("prior_comparator_qualification") != {
        "summary": "results/openvla-comparator-qualification-s3q-v01/worker_summary.json",
        "summary_sha256": config["prior_comparator_qualification"]["summary_sha256"],
        "semantic_sha256": "f0f97238fc5e5fd2665db5ecac80f439e46e4f85909ef0128563e856b10e8290",
        "passed": True,
        "boundary_contract_count": 22,
        "model_calls": 0,
    }:
        raise v1.ParityTechnicalStop("prior comparator qualification changed")
    prior_path = ROOT / config["prior_comparator_qualification"]["summary"]
    if not prior_path.is_file() or file_sha256(prior_path) != config[
        "prior_comparator_qualification"
    ]["summary_sha256"]:
        raise v1.ParityTechnicalStop("prior comparator evidence changed")
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if (
        prior.get("semantic_sha256")
        != config["prior_comparator_qualification"]["semantic_sha256"]
        or prior.get("passed") is not True
        or prior.get("boundary_contracts_exercised") != 22
        or prior.get("model_calls") != 0
    ):
        raise v1.ParityTechnicalStop("prior comparator qualification did not pass")
    expected_boundaries = {
        "tensor_cuda_to_tensor_cuda": sorted(recovery03.TENSOR_BOUNDARIES),
        "tensor_cuda_to_ndarray_cpu_float32": sorted(recovery03.MIXED_BOUNDARIES),
        "ndarray_cpu_to_ndarray_cpu": sorted(recovery03.ARRAY_BOUNDARIES),
        "all_boundary_count": 22,
        "reject_uncontracted_boundaries": True,
        "reject_representation_drift": True,
        "strict_json_nonfinite_encoding": True,
        "model_computation_changed": False,
    }
    if config.get("boundary_contract") != expected_boundaries:
        raise v1.ParityTechnicalStop("S3 Recovery 04 boundary contract changed")
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
        raise v1.ParityTechnicalStop("S3 Recovery 04 resource boundary changed")
    if config.get("advance") != {
        "next_stage": "S4_D62_REQUALIFICATION",
        "authorized": False,
        "stop_before_next_stage": True,
    }:
        raise v1.ParityTechnicalStop("S3 Recovery 04 advance boundary changed")
    if config.get("output_root") != "results/openvla-semantic-parity-s3-v05-recovery04":
        raise v1.ParityTechnicalStop("S3 Recovery 04 output root changed")
    if (
        config.get("population")
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
                "corrected_dense_use_cache_false",
                "corrected_dense_use_cache_true",
                "corrected_dense_use_cache_true_sidecar",
            ],
            "alternate_official_custom_order": True,
            "total_calls": 32,
        }
        or config.get("tolerance") != 0.000001
    ):
        raise v1.ParityTechnicalStop("S3 Recovery 04 scientific contract changed")
    if config.get("loader_recovery") != {
        "official_source_tree": "third_party/openvla-oft",
        "official_source_revision": "e4287e94541f459edc4feabc4e181f537cd569a8",
        "disable_model_logic_sync": True,
        "qualified_loader_guard_sha256": recovery02.QUALIFIED_LOADER_GUARD_SHA256,
        "canonical_proprio_shape": [8],
        "model_computation_changed": False,
    }:
        raise v1.ParityTechnicalStop("S3 Recovery 04 loader boundary changed")
    if recovery02.function_source_sha256(recovery01.install_official_loader_guard) != (
        recovery02.QUALIFIED_LOADER_GUARD_SHA256
    ):
        raise v1.ParityTechnicalStop("qualified loader guard changed")
    for relative, expected in config["authenticated_files"].items():
        path = ROOT / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise v1.ParityTechnicalStop(f"authenticated Recovery 04 input changed: {relative}")
    manifest = json.loads((ROOT / config["input_manifest"]).read_text(encoding="utf-8"))
    if (
        manifest.get("semantic_sha256") != v1.semantic_sha256(manifest)
        or manifest.get("terminal_outcome_fields_accessed") is not False
        or manifest.get("expert_action_fields_accessed") is not False
        or len(manifest.get("inputs", [])) != 8
        or {row["suite"] for row in manifest["inputs"]}
        != {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
    ):
        raise v1.ParityTechnicalStop("S3 Recovery 04 population changed")
    data_root = ROOT / manifest["data_root_relative"]
    for row in manifest["inputs"]:
        source = data_root / row["source_path"]
        if not source.is_file() or v1.file_sha256(source) != row["source_sha256"]:
            raise v1.ParityTechnicalStop("S3 Recovery 04 observation source changed")
    checkpoint = ROOT / config["model"]["checkpoint_relative"]
    for name, expected in config["model"]["checkpoint_metadata_sha256"].items():
        if not (checkpoint / name).is_file() or v1.file_sha256(checkpoint / name) != expected:
            raise v1.ParityTechnicalStop(f"S3 Recovery 04 checkpoint changed: {name}")
    if (ROOT / config["output_root"]).exists():
        raise v1.ParityTechnicalStop("immutable S3 Recovery 04 output already exists")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f"S3 Recovery 04 must start in {ROOT}")
    recovery03 = load_recovery03()
    recovery02 = recovery03.load_recovery02()
    recovery01 = recovery02.load_recovery01()
    v1 = recovery01.load_v1()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    validate_recovery04(config, recovery03, recovery02, recovery01, v1)
    qualification_path = (
        ROOT / config["cache_mode_qualification"]["output_root"] / "worker_summary.json"
    )
    if not qualification_path.is_file():
        raise v1.ParityTechnicalStop("cache-mode GPU qualification is missing")
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    if (
        qualification.get("semantic_sha256") != v1.semantic_sha256(qualification)
        or qualification.get("complete") is not True
        or qualification.get("passed") is not True
        or qualification.get("openvla_model_calls") != 0
        or qualification.get("none_returns_cache") is not True
        or qualification.get("false_returns_cache") is not False
        or qualification.get("true_returns_cache") is not True
    ):
        raise v1.ParityTechnicalStop("cache-mode GPU qualification did not pass")

    evaluation, loader_originals = recovery01.install_official_loader_guard()
    from savr.openvla import official_semantics

    original_prepare = official_semantics.prepare_semantic_query
    original_forward = official_semantics.structurally_aligned_dense_forward
    original_compare = v1.compare_values
    original_validate = v1.validate_static

    def prepared_with_canonical_proprio(*values: Any, **kwargs: Any) -> Any:
        return recovery02.canonicalize_prepared_proprio(original_prepare(*values, **kwargs))

    def forward_with_explicit_no_cache(*values: Any, **kwargs: Any) -> dict[str, Any]:
        return explicit_no_cache_forward(original_forward, *values, **kwargs)

    def contracted_compare(*values: Any, **kwargs: Any) -> dict[str, Any]:
        return recovery03.compare_with_contract(original_compare, *values, **kwargs)

    official_semantics.prepare_semantic_query = prepared_with_canonical_proprio
    official_semantics.structurally_aligned_dense_forward = forward_with_explicit_no_cache
    v1.compare_values = contracted_compare
    v1.validate_static = lambda supplied: validate_recovery04(
        supplied, recovery03, recovery02, recovery01, v1
    )
    original_argv = sys.argv
    try:
        sys.argv = [str(RECOVERY03_WORKER), "--config", str(args.config)]
        return int(v1.main())
    finally:
        sys.argv = original_argv
        v1.validate_static = original_validate
        v1.compare_values = original_compare
        official_semantics.structurally_aligned_dense_forward = original_forward
        official_semantics.prepare_semantic_query = original_prepare
        for name, value in loader_originals.items():
            setattr(evaluation, name, value)


if __name__ == "__main__":
    raise SystemExit(main())
