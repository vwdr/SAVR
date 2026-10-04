#!/usr/bin/env python3
"""S3 Recovery 03: fully contracted tensor/array semantic comparison."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Mapping


ROOT = Path("/home/ved/SAVR")
DEFAULT_CONFIG = Path("configs/openvla/semantic_parity_s3_v04_recovery03.json")
RECOVERY02_WORKER = ROOT / "scripts/run_openvla_semantic_parity_recovery02.py"

TENSOR_BOUNDARIES = frozenset(
    {
        "input_ids",
        "input_embeddings",
        "action_mask",
        "pixel_values",
        "language_embeddings",
        "vision_output",
        "pre_proprio_projected",
        "normalized_proprio",
        "projected_with_proprio",
        "input_attention_mask",
        "multimodal_projected",
        "masked_input_embeddings",
        "multimodal_embeddings",
        "multimodal_attention_mask",
        "action_hidden",
        "determinism_multimodal_embeddings",
        "determinism_multimodal_attention_mask",
        "determinism_action_hidden",
    }
)
MIXED_BOUNDARIES = frozenset({"normalized_actions"})
ARRAY_BOUNDARIES = frozenset(
    {
        "unnormalized_actions",
        "determinism_normalized_actions",
        "determinism_actions",
    }
)
ALL_BOUNDARIES = TENSOR_BOUNDARIES | MIXED_BOUNDARIES | ARRAY_BOUNDARIES


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


def load_recovery02() -> Any:
    return load_module(RECOVERY02_WORKER, "openvla_semantic_parity_recovery02_for_r03")


def representation_kind(value: Any, torch_module: Any, np_module: Any) -> str:
    if torch_module.is_tensor(value):
        return "tensor_cuda" if value.is_cuda else "tensor_cpu"
    if isinstance(value, np_module.ndarray):
        return "ndarray_cpu"
    raise RuntimeError(f"unsupported comparison representation: {type(value).__name__}")


def expected_representation_pair(boundary: str) -> tuple[str, str]:
    if boundary in TENSOR_BOUNDARIES:
        return ("tensor_cuda", "tensor_cuda")
    if boundary in MIXED_BOUNDARIES:
        return ("tensor_cuda", "ndarray_cpu")
    if boundary in ARRAY_BOUNDARIES:
        return ("ndarray_cpu", "ndarray_cpu")
    raise RuntimeError(f"uncontracted semantic boundary: {boundary}")


def mixed_float32_array(value: Any, torch_module: Any, np_module: Any) -> Any:
    if torch_module.is_tensor(value):
        return value.detach().float().cpu().contiguous().numpy()
    array = np_module.asarray(value)
    if not np_module.issubdtype(array.dtype, np_module.number):
        raise RuntimeError("heterogeneous comparison requires numeric values")
    return np_module.ascontiguousarray(array, dtype=np_module.float32)


def compare_with_contract(
    original_compare: Callable[..., dict[str, Any]],
    *,
    boundary: str,
    reference: Any,
    candidate: Any,
    tolerance: float,
    torch_module: Any,
    np_module: Any,
) -> dict[str, Any]:
    observed_pair = (
        representation_kind(reference, torch_module, np_module),
        representation_kind(candidate, torch_module, np_module),
    )
    expected_pair = expected_representation_pair(boundary)
    if observed_pair != expected_pair:
        raise RuntimeError(
            f"semantic boundary representation changed: {boundary}: "
            f"expected {expected_pair}, observed {observed_pair}"
        )
    if boundary in MIXED_BOUNDARIES:
        reference = mixed_float32_array(reference, torch_module, np_module)
        candidate = mixed_float32_array(candidate, torch_module, np_module)
    record = original_compare(
        boundary=boundary,
        reference=reference,
        candidate=candidate,
        tolerance=tolerance,
        torch_module=torch_module,
        np_module=np_module,
    )
    maximum = record.get("max_abs")
    if isinstance(maximum, float) and not math.isfinite(maximum):
        record = dict(record)
        record["max_abs"] = None
        record["max_abs_nonfinite"] = "nan" if math.isnan(maximum) else (
            "positive_infinity" if maximum > 0 else "negative_infinity"
        )
    record = {
        **record,
        "reference_representation": observed_pair[0],
        "candidate_representation": observed_pair[1],
        "comparison_domain": "float32_cpu"
        if boundary in MIXED_BOUNDARIES
        else "native",
    }
    json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return record


def validate_recovery03(
    config: Mapping[str, Any], recovery02: Any, recovery01: Any, v1: Any
) -> None:
    if config.get("schema_version") != "openvla-semantic-parity-s3-recovery03-v1":
        raise v1.ParityTechnicalStop("S3 Recovery 03 schema changed")
    if config.get("semantic_sha256") != v1.semantic_sha256(config):
        raise v1.ParityTechnicalStop("S3 Recovery 03 config hash mismatch")
    if config.get("authorization") != {
        "s3_recovery03_authorized": True,
        "source": "User approved S3 Recovery 03 on 2026-09-01",
        "comparator_gpu_qualification": True,
        "semantic_model_attempt": True,
        "simulator": False,
        "terminal_outcomes": False,
        "training": False,
        "automatic_retry": False,
    }:
        raise v1.ParityTechnicalStop("S3 Recovery 03 authorization changed")
    technical = config.get("technical_recovery", {})
    if technical != {
        "source_run_id": "openvla-semantic-parity-s3-v03-recovery02",
        "source_stop": "results/openvla-semantic-parity-s3-v03-recovery02/technical_stop.json",
        "source_stop_sha256": technical.get("source_stop_sha256"),
        "source_traceback": "results/openvla-semantic-parity-s3-v03-recovery02/technical_traceback.log",
        "source_traceback_sha256": technical.get("source_traceback_sha256"),
        "source_model_calls": 4,
        "source_observations_completed": 0,
        "scientific_population_changed": False,
        "scientific_gate_changed": False,
        "automatic_retry": False,
    }:
        raise v1.ParityTechnicalStop("S3 Recovery 03 recovery boundary changed")
    stop_path = ROOT / technical["source_stop"]
    traceback_path = ROOT / technical["source_traceback"]
    if (
        not stop_path.is_file()
        or file_sha256(stop_path) != technical["source_stop_sha256"]
        or not traceback_path.is_file()
        or file_sha256(traceback_path) != technical["source_traceback_sha256"]
    ):
        raise v1.ParityTechnicalStop("S3 Recovery 02 evidence changed")
    stop = json.loads(stop_path.read_text(encoding="utf-8"))
    if (
        stop.get("semantic_sha256") != v1.semantic_sha256(stop)
        or stop.get("error_type") != "TypeError"
        or "cuda:0 device type tensor to numpy" not in stop.get("error", "")
        or stop.get("model_calls") != 4
        or stop.get("observations_completed") != 0
        or stop.get("simulator_outcomes_accessed") is not False
        or stop.get("automatic_retry") is not False
        or stop.get("checkpoint_restoration_error") is not None
    ):
        raise v1.ParityTechnicalStop("S3 Recovery 02 stop contents changed")
    expected_contract = {
        "tensor_cuda_to_tensor_cuda": sorted(TENSOR_BOUNDARIES),
        "tensor_cuda_to_ndarray_cpu_float32": sorted(MIXED_BOUNDARIES),
        "ndarray_cpu_to_ndarray_cpu": sorted(ARRAY_BOUNDARIES),
        "all_boundary_count": 22,
        "reject_uncontracted_boundaries": True,
        "reject_representation_drift": True,
        "strict_json_nonfinite_encoding": True,
        "model_computation_changed": False,
    }
    if config.get("boundary_contract") != expected_contract or len(ALL_BOUNDARIES) != 22:
        raise v1.ParityTechnicalStop("S3 Recovery 03 boundary contract changed")
    if config.get("comparator_qualification") != {
        "output_root": "results/openvla-comparator-qualification-s3q-v01",
        "gpu_count": 1,
        "model_loads": 0,
        "model_calls": 0,
        "wall_seconds": 120,
        "artifact_bytes": 1048576,
        "automatic_retry": False,
        "must_pass_before_model_attempt": True,
    }:
        raise v1.ParityTechnicalStop("S3 Recovery 03 comparator qualification changed")
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
        raise v1.ParityTechnicalStop("S3 Recovery 03 resource boundary changed")
    if config.get("advance") != {
        "next_stage": "S4_D62_REQUALIFICATION",
        "authorized": False,
        "stop_before_next_stage": True,
    }:
        raise v1.ParityTechnicalStop("S3 Recovery 03 advance boundary changed")
    if config.get("output_root") != "results/openvla-semantic-parity-s3-v04-recovery03":
        raise v1.ParityTechnicalStop("S3 Recovery 03 output root changed")
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
                "corrected_dense_use_cache_none",
                "corrected_dense_use_cache_true",
                "corrected_dense_use_cache_true_sidecar",
            ],
            "alternate_official_custom_order": True,
            "total_calls": 32,
        }
        or config.get("tolerance") != 0.000001
    ):
        raise v1.ParityTechnicalStop("S3 Recovery 03 scientific contract changed")
    if config.get("loader_recovery") != {
        "official_source_tree": "third_party/openvla-oft",
        "official_source_revision": "e4287e94541f459edc4feabc4e181f537cd569a8",
        "disable_model_logic_sync": True,
        "qualified_loader_guard_sha256": recovery02.QUALIFIED_LOADER_GUARD_SHA256,
        "canonical_proprio_shape": [8],
        "model_computation_changed": False,
    }:
        raise v1.ParityTechnicalStop("S3 Recovery 03 loader boundary changed")
    if recovery02.function_source_sha256(recovery01.install_official_loader_guard) != (
        recovery02.QUALIFIED_LOADER_GUARD_SHA256
    ):
        raise v1.ParityTechnicalStop("qualified loader guard changed")
    for relative, expected in config["authenticated_files"].items():
        path = ROOT / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise v1.ParityTechnicalStop(f"authenticated Recovery 03 input changed: {relative}")
    manifest = json.loads((ROOT / config["input_manifest"]).read_text(encoding="utf-8"))
    if (
        manifest.get("semantic_sha256") != v1.semantic_sha256(manifest)
        or manifest.get("terminal_outcome_fields_accessed") is not False
        or manifest.get("expert_action_fields_accessed") is not False
        or len(manifest.get("inputs", [])) != 8
        or {row["suite"] for row in manifest["inputs"]}
        != {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
    ):
        raise v1.ParityTechnicalStop("S3 Recovery 03 population changed")
    data_root = ROOT / manifest["data_root_relative"]
    for row in manifest["inputs"]:
        source = data_root / row["source_path"]
        if not source.is_file() or v1.file_sha256(source) != row["source_sha256"]:
            raise v1.ParityTechnicalStop("S3 Recovery 03 observation source changed")
    checkpoint = ROOT / config["model"]["checkpoint_relative"]
    for name, expected in config["model"]["checkpoint_metadata_sha256"].items():
        if not (checkpoint / name).is_file() or v1.file_sha256(checkpoint / name) != expected:
            raise v1.ParityTechnicalStop(f"S3 Recovery 03 checkpoint changed: {name}")
    if (ROOT / config["output_root"]).exists():
        raise v1.ParityTechnicalStop("immutable S3 Recovery 03 output already exists")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f"S3 Recovery 03 must start in {ROOT}")
    recovery02 = load_recovery02()
    recovery01 = recovery02.load_recovery01()
    v1 = recovery01.load_v1()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    validate_recovery03(config, recovery02, recovery01, v1)
    qualification = ROOT / config["comparator_qualification"]["output_root"] / "worker_summary.json"
    if not qualification.is_file():
        raise v1.ParityTechnicalStop("comparator GPU qualification is missing")
    qualification_record = json.loads(qualification.read_text(encoding="utf-8"))
    if (
        qualification_record.get("complete") is not True
        or qualification_record.get("passed") is not True
        or qualification_record.get("model_calls") != 0
        or qualification_record.get("semantic_sha256") != v1.semantic_sha256(qualification_record)
    ):
        raise v1.ParityTechnicalStop("comparator GPU qualification did not pass")

    evaluation, loader_originals = recovery01.install_official_loader_guard()
    from savr.openvla import official_semantics

    original_prepare = official_semantics.prepare_semantic_query
    original_compare = v1.compare_values
    original_validate = v1.validate_static

    def prepared_with_canonical_proprio(*values: Any, **kwargs: Any) -> Any:
        return recovery02.canonicalize_prepared_proprio(original_prepare(*values, **kwargs))

    def contracted_compare(*values: Any, **kwargs: Any) -> dict[str, Any]:
        return compare_with_contract(original_compare, *values, **kwargs)

    official_semantics.prepare_semantic_query = prepared_with_canonical_proprio
    v1.compare_values = contracted_compare
    v1.validate_static = lambda supplied: validate_recovery03(
        supplied, recovery02, recovery01, v1
    )
    original_argv = sys.argv
    try:
        sys.argv = [str(RECOVERY02_WORKER), "--config", str(args.config)]
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
