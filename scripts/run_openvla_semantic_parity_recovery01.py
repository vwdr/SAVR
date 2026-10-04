#!/usr/bin/env python3
"""Versioned S3 recovery: preserve the official checkpoint model logic."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path("/home/ved/SAVR")
DEFAULT_CONFIG = Path("configs/openvla/semantic_parity_s3_v02_recovery01.json")
V1_WORKER = ROOT / "scripts/run_openvla_semantic_parity.py"


def load_v1() -> Any:
    spec = importlib.util.spec_from_file_location("openvla_semantic_parity_v1", V1_WORKER)
    if spec is None or spec.loader is None:
        raise RuntimeError("S3 v01 worker cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_recovery(
    config: Mapping[str, Any], v1: Any, *, require_authorized: bool = False
) -> None:
    if config.get("schema_version") != "openvla-semantic-parity-s3-recovery01-v1":
        raise v1.ParityTechnicalStop("S3 Recovery 01 config schema changed")
    if config.get("semantic_sha256") != v1.semantic_sha256(config):
        raise v1.ParityTechnicalStop("S3 Recovery 01 config hash mismatch")
    expected_authorized = bool(require_authorized)
    if config.get("authorization") != {
        "s3_recovery01_authorized": expected_authorized,
        "source": "User approved S3 Recovery 01 on 2026-09-01"
        if expected_authorized
        else None,
        "simulator": False,
        "terminal_outcomes": False,
        "training": False,
        "automatic_retry": False,
    }:
        raise v1.ParityTechnicalStop(
            "S3 Recovery 01 authorization does not match execution mode"
        )
    if config.get("technical_recovery") != {
        "source_run_id": "openvla-semantic-parity-s3-v01",
        "source_stop": "results/openvla-semantic-parity-s3-v01/technical_stop.json",
        "source_stop_sha256": config["technical_recovery"]["source_stop_sha256"],
        "source_model_calls": 0,
        "source_observations_completed": 0,
        "scientific_population_changed": False,
        "scientific_gate_changed": False,
        "automatic_retry": False,
    }:
        raise v1.ParityTechnicalStop("S3 Recovery 01 boundary changed")
    stop_path = ROOT / config["technical_recovery"]["source_stop"]
    if (
        not stop_path.is_file()
        or v1.file_sha256(stop_path) != config["technical_recovery"]["source_stop_sha256"]
    ):
        raise v1.ParityTechnicalStop("S3 source technical stop changed")
    stop = json.loads(stop_path.read_text(encoding="utf-8"))
    if (
        stop.get("run_id") != "openvla-semantic-parity-s3-v01"
        or stop.get("model_calls") != 0
        or stop.get("observations_completed") != 0
        or stop.get("simulator_outcomes_accessed") is not False
        or stop.get("automatic_retry") is not False
        or stop.get("checkpoint_restoration_error") is not None
    ):
        raise v1.ParityTechnicalStop("S3 source stop contents changed")
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
        raise v1.ParityTechnicalStop("S3 Recovery 01 resource boundary changed")
    if config.get("advance") != {
        "next_stage": "S4_D62_REQUALIFICATION",
        "authorized": False,
        "stop_before_next_stage": True,
    }:
        raise v1.ParityTechnicalStop("S3 Recovery 01 advance boundary changed")
    if config.get("output_root") != "results/openvla-semantic-parity-s3-v02-recovery01":
        raise v1.ParityTechnicalStop("S3 Recovery 01 output root changed")
    if config.get("loader_recovery") != {
        "official_source_tree": "third_party/openvla-oft",
        "official_source_revision": "e4287e94541f459edc4feabc4e181f537cd569a8",
        "disable_model_logic_sync": True,
        "require_checkpoint_prompt_derived_action_readout": True,
        "reject_cache_tail_action_readout": True,
        "adapt_official_return_without_changing_actions": True,
        "discard_vla_cache_only_config_argument": True,
        "inject_absent_dense_cache_controls_as_none": True,
    }:
        raise v1.ParityTechnicalStop("S3 Recovery 01 loader repair changed")
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
        raise v1.ParityTechnicalStop("S3 Recovery 01 scientific contract changed")
    for relative, expected in config["authenticated_files"].items():
        path = ROOT / relative
        if not path.is_file() or v1.file_sha256(path) != expected:
            raise v1.ParityTechnicalStop(f"authenticated recovery input changed: {relative}")
    manifest = json.loads((ROOT / config["input_manifest"]).read_text(encoding="utf-8"))
    if (
        manifest.get("semantic_sha256") != v1.semantic_sha256(manifest)
        or manifest.get("terminal_outcome_fields_accessed") is not False
        or manifest.get("expert_action_fields_accessed") is not False
        or len(manifest.get("inputs", [])) != 8
        or {row["suite"] for row in manifest["inputs"]}
        != {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
        or any(
            sum(row["suite"] == suite for row in manifest["inputs"]) != 2
            for suite in {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
        )
    ):
        raise v1.ParityTechnicalStop("S3 Recovery 01 population changed")
    data_root = ROOT / manifest["data_root_relative"]
    for row in manifest["inputs"]:
        source = data_root / row["source_path"]
        if not source.is_file() or v1.file_sha256(source) != row["source_sha256"]:
            raise v1.ParityTechnicalStop("S3 Recovery 01 observation source changed")
    checkpoint = ROOT / config["model"]["checkpoint_relative"]
    for name, expected in config["model"]["checkpoint_metadata_sha256"].items():
        if not (checkpoint / name).is_file() or v1.file_sha256(checkpoint / name) != expected:
            raise v1.ParityTechnicalStop(f"S3 Recovery 01 checkpoint changed: {name}")
    if (ROOT / config["output_root"]).exists():
        raise v1.ParityTechnicalStop("immutable S3 Recovery 01 output already exists")


def install_official_loader_guard() -> tuple[Any, dict[str, Any]]:
    source = ROOT / "third_party/openvla-oft"
    libero_config = ROOT / "configs/pair/libero_runtime"
    if not (libero_config / "config.yaml").is_file():
        raise RuntimeError("project-local LIBERO configuration is missing")
    os.environ["LIBERO_CONFIG_PATH"] = str(libero_config)
    sys.path.insert(0, str(source))
    from experiments.robot import openvla_utils
    from experiments.robot.libero import run_libero_eval as evaluation

    original_initialize = evaluation.initialize_model
    original_config = evaluation.GenerateConfig
    original_get_action = evaluation.get_action

    def compatible_config(*args: Any, **kwargs: Any):
        if kwargs.pop("use_vla_cache", None) is not False:
            raise RuntimeError("S3 Recovery 01 expected disabled VLA-Cache config")
        return original_config(*args, **kwargs)

    def compatible_get_action(*args: Any, **kwargs: Any):
        actions = original_get_action(*args, **kwargs)
        return actions, None, None, None

    def initialize_official(cfg: Any):
        original_sync = openvla_utils.check_model_logic_mismatch
        openvla_utils.check_model_logic_mismatch = lambda _checkpoint: None
        try:
            values = original_initialize(cfg)
        finally:
            openvla_utils.check_model_logic_mismatch = original_sync
        model = values[0]
        source_text = inspect.getsource(model._regression_or_discrete_prediction)
        if (
            "NUM_PATCHES + NUM_PROMPT_TOKENS" not in source_text
            or "- ACTION_DIM * NUM_ACTIONS_CHUNK - 1" in source_text
        ):
            raise RuntimeError("S3 Recovery 01 did not load official checkpoint action semantics")
        language_config = model.language_model.config
        if hasattr(language_config, "proportion_attn_var") or hasattr(
            language_config, "reusable_patches"
        ):
            raise RuntimeError("official checkpoint unexpectedly contains cache-control fields")
        language_config.proportion_attn_var = None
        language_config.reusable_patches = None
        return values

    evaluation.initialize_model = initialize_official
    evaluation.GenerateConfig = compatible_config
    evaluation.get_action = compatible_get_action
    return evaluation, {
        "initialize_model": original_initialize,
        "GenerateConfig": original_config,
        "get_action": original_get_action,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f"S3 Recovery 01 must start in {ROOT}")
    v1 = load_v1()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    validate_recovery(config, v1, require_authorized=True)
    evaluation, originals = install_official_loader_guard()
    original_validate = v1.validate_static
    v1.validate_static = lambda supplied: validate_recovery(
        supplied, v1, require_authorized=True
    )
    original_argv = sys.argv
    try:
        sys.argv = [str(V1_WORKER), "--config", str(args.config)]
        return int(v1.main())
    finally:
        sys.argv = original_argv
        v1.validate_static = original_validate
        for name, value in originals.items():
            setattr(evaluation, name, value)


if __name__ == "__main__":
    raise SystemExit(main())
