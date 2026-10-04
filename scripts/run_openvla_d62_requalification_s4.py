#!/usr/bin/env python3
"""Run the bounded, outcome-free S4 D62 substrate requalification."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path("/home/ved/SAVR")
DEFAULT_CONFIG = Path("configs/openvla/d62_requalification_s4_v01.json")
RECOVERY05_WORKER = ROOT / "scripts/run_openvla_semantic_parity_recovery05.py"


class S4TechnicalStop(RuntimeError):
    """A technical condition prevented a valid S4 decision."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def semantic_sha256(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_recovery05() -> Any:
    spec = importlib.util.spec_from_file_location("s3_recovery05_for_s4", RECOVERY05_WORKER)
    if spec is None or spec.loader is None:
        raise S4TechnicalStop("Recovery 05 worker cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_config(config: Mapping[str, Any], v1: Any) -> None:
    if config.get("schema_version") != "openvla-d62-requalification-s4-v1":
        raise S4TechnicalStop("S4 schema changed")
    if config.get("semantic_sha256") != semantic_sha256(config):
        raise S4TechnicalStop("S4 config hash mismatch")
    if config.get("authorization") != {
        "s4_authorized": True,
        "source": "User approved S4 on 2026-09-01",
        "simulator": False,
        "terminal_outcomes": False,
        "expert_actions": False,
        "training": False,
        "automatic_retry": False,
    }:
        raise S4TechnicalStop("S4 authorization changed")
    if config.get("population") != {
        "selection_audit_observations": 8,
        "allfresh_official_controls": 8,
        "recursive_windows": 1,
        "recursive_repetitions": 2,
        "recursive_ages": [1, 2, 3, 4],
        "cadence_steps": 8,
        "terminal_outcomes_accessed": False,
        "expert_actions_accessed": False,
    }:
        raise S4TechnicalStop("S4 population changed")
    if config.get("call_schedule") != {
        "suite_balanced_anchor_calls": 8,
        "suite_balanced_official_calls": 8,
        "suite_balanced_allfresh_calls": 8,
        "recursive_anchor_calls": 2,
        "recursive_transition_calls": 8,
        "reset_official_calls": 1,
        "reset_custom_calls": 2,
        "planned_model_calls": 37,
    }:
        raise S4TechnicalStop("S4 call schedule changed")
    if config.get("substrate") != {
        "historical_id": "D62_BAL_PT1",
        "corrected_candidate_id": "D62_BAL_PT1_S4C_V1",
        "canonical_action_positions": True,
        "instruction_only_positions": True,
        "selection_materiality_rule": (
            "material if any age-1 ordered position, onset assignment, or protected tile "
            "differs across the eight frozen observations"
        ),
        "freeze_new_identity": True,
    }:
        raise S4TechnicalStop("S4 substrate contract changed")
    if config.get("resource_caps") != {
        "gpu_count": 1,
        "model_processes": 1,
        "model_call_hard_cap": 48,
        "planned_model_calls": 37,
        "wall_seconds": 1800,
        "artifact_bytes": 268435456,
        "peak_gpu_memory_mib_strict_max": 23552,
        "downloads": 0,
        "simulator_outcomes": 0,
        "raw_actions_persisted": False,
        "automatic_retry": False,
    }:
        raise S4TechnicalStop("S4 resource boundary changed")
    if config.get("advance") != {
        "next_stage": "S5_CAC_C1_REQUALIFICATION",
        "authorized": False,
        "stop_before_next_stage": True,
    }:
        raise S4TechnicalStop("S4 advance boundary changed")
    if config.get("output_root") != "results/openvla-d62-requalification-s4-v01":
        raise S4TechnicalStop("S4 output root changed")
    for relative, expected in config["authenticated_files"].items():
        path = ROOT / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise S4TechnicalStop(f"authenticated S4 input changed: {relative}")
    s3_path = ROOT / config["s3_parent"]["summary"]
    if not s3_path.is_file() or file_sha256(s3_path) != config["s3_parent"]["summary_sha256"]:
        raise S4TechnicalStop("S3 parent evidence changed")
    s3 = json.loads(s3_path.read_text(encoding="utf-8"))
    if (
        s3.get("semantic_sha256") != v1.semantic_sha256(s3)
        or s3.get("complete") is not True
        or s3.get("passed") is not True
        or s3.get("observations_completed") != 8
        or s3.get("model_calls") != 32
        or s3.get("comparison_count") != 344
        or float(s3.get("maximum_abs_error", 1.0)) > 0.000001
    ):
        raise S4TechnicalStop("S3 parent did not pass")
    manifest = json.loads((ROOT / config["input_manifest"]).read_text(encoding="utf-8"))
    if (
        manifest.get("semantic_sha256") != semantic_sha256(manifest)
        or len(manifest.get("inputs", [])) != 8
        or manifest.get("terminal_outcome_fields_accessed") is not False
        or manifest.get("expert_action_fields_accessed") is not False
        or {row["suite"] for row in manifest["inputs"]}
        != {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
    ):
        raise S4TechnicalStop("S4 input manifest changed")
    data_root = ROOT / manifest["data_root_relative"]
    for row in manifest["inputs"]:
        source = data_root / row["source_path"]
        if not source.is_file() or file_sha256(source) != row["source_sha256"]:
            raise S4TechnicalStop("S4 observation source changed")
        if int(row["step_count"]) <= 40:
            raise S4TechnicalStop("S4 trajectory is too short for the frozen window")
    checkpoint = ROOT / config["model"]["checkpoint_relative"]
    for name, expected in config["model"]["checkpoint_metadata_sha256"].items():
        if not (checkpoint / name).is_file() or file_sha256(checkpoint / name) != expected:
            raise S4TechnicalStop(f"S4 checkpoint changed: {name}")
    if (ROOT / config["output_root"]).exists():
        raise S4TechnicalStop("immutable S4 output already exists")


def selection_digest(ordered: Any, groups: Mapping[Any, int]) -> str:
    payload = {
        "ordered": [int(value) for value in ordered.detach().cpu().tolist()],
        "groups": sorted(
            [getattr(camera, "value", str(camera)), int(tile), int(layer)]
            for (camera, tile), layer in groups.items()
        ),
    }
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f"S4 must start in {ROOT}")
    recovery05 = load_recovery05()
    recovery04 = recovery05.load_recovery04()
    recovery03 = recovery04.load_recovery03()
    recovery02 = recovery03.load_recovery02()
    recovery01 = recovery02.load_recovery01()
    v1 = recovery01.load_v1()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    validate_config(config, v1)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    physical = os.environ.get("OPENVLA_PHYSICAL_GPU_ID", "")
    if not physical.isdigit() or visible != physical or "," in visible:
        raise SystemExit("S4 requires one explicitly selected GPU")
    initial_gpu = v1.selected_gpu_snapshot(int(physical))
    if initial_gpu["memory_used_mib"] > 1024 or initial_gpu["utilization_percent"] > 5:
        raise SystemExit("S4 selected GPU is not sufficiently idle")
    output_root = ROOT / config["output_root"]
    output_root.mkdir(parents=True)
    runtime_cache = output_root / "runtime-cache"
    for name, path in {
        "MPLCONFIGDIR": runtime_cache / "matplotlib",
        "HF_MODULES_CACHE": runtime_cache / "hf-modules",
        "HF_HOME": runtime_cache / "hf-home",
        "TORCH_HOME": runtime_cache / "torch",
        "XDG_CACHE_HOME": runtime_cache / "xdg",
        "TMPDIR": runtime_cache / "tmp",
        "WANDB_DIR": runtime_cache / "wandb",
    }.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["LIBERO_CONFIG_PATH"] = str(ROOT / "configs/pair/libero_runtime")
    sys.path.insert(0, str(ROOT / "src"))
    sys.pycache_prefix = str(runtime_cache / "pycache")
    started = time.monotonic()
    calls = 0
    sampler = v1.AggregateMemorySampler(int(physical), initial_gpu["memory_used_mib"])
    sampler.start()
    checkpoint_baseline = None
    checkpoint = ROOT / config["model"]["checkpoint_relative"]
    restore_checkpoint_exact = None
    evaluation = None
    loader_originals = None
    model = action_head = proprio_projector = processor = None
    config_baseline = None

    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError("S4 wall-time boundary reached")

    signal.signal(signal.SIGTERM, timeout_handler)
    signal.alarm(int(config["resource_caps"]["wall_seconds"]))

    def consume() -> None:
        nonlocal calls
        calls += 1
        if calls > int(config["resource_caps"]["model_call_hard_cap"]):
            raise S4TechnicalStop("S4 model-call cap exceeded")

    try:
        evaluation, loader_originals = recovery01.install_official_loader_guard()
        source = ROOT / "third_party/openvla-oft"
        os.chdir(source)
        sys.path.insert(0, str(source))
        import h5py
        import numpy as np
        import torch
        from experiments.robot.openvla_utils import normalize_proprio, prepare_images_for_vla
        from experiments.robot.robot_utils import set_seed_everywhere
        from savr.acr.v5_d_recovery import capture_checkpoint_baseline, restore_checkpoint_exact
        from savr.cac.c1 import cache_digest
        from savr.openvla.official_semantics import OfficialBoundaryCapture
        from savr.pair.interventions import clone_runtime_cache
        from savr.pair.p3 import P3Profile
        from savr.pair.p3_openvla import (
            PhysicalSourceTracker,
            capture_reused_cache,
            configure_dense,
            configure_profile,
            forward_query,
            ordered_tile_profile_vectorized,
            prepare_query,
            runtime_positions,
            verify_reused_cache,
        )

        set_seed_everywhere(int(config["seed"]))
        checkpoint_baseline = capture_checkpoint_baseline(
            checkpoint, ("config.json", "configuration_prismatic.py", "modeling_prismatic.py")
        )
        cfg = v1.base_config(evaluation, checkpoint, output_root)
        evaluation.validate_config(cfg)
        model, action_head, proprio_projector, noisy, processor = evaluation.initialize_model(cfg)
        model.eval()
        if torch.cuda.device_count() != 1 or action_head is None or proprio_projector is None:
            raise S4TechnicalStop("S4 model/device initialization contract failed")
        config_baseline = (
            model.language_model.config.proportion_attn_var,
            model.language_model.config.reusable_patches,
        )
        model_config = json.loads((ROOT / config["model_config"]).read_text(encoding="utf-8"))
        profile = P3Profile.from_mapping(
            next(
                row
                for row in model_config["profiles"]
                if row["profile_id"] == config["substrate"]["historical_id"]
            )
        )
        manifest = json.loads((ROOT / config["input_manifest"]).read_text(encoding="utf-8"))
        inputs = manifest["inputs"]
        data_root = ROOT / manifest["data_root_relative"]

        def load_window(identity: Mapping[str, Any], count: int) -> list[dict[str, Any]]:
            values = []
            with h5py.File(data_root / identity["source_path"], "r") as handle:
                group = handle[f"data/{identity['original_trajectory_id']}/obs"]
                for ordinal in range(count):
                    step = ordinal * int(config["population"]["cadence_steps"])
                    state = np.concatenate(
                        (np.asarray(group["ee_states"][step]), np.asarray(group["gripper_states"][step]))
                    ).copy()
                    values.append(
                        {
                            "scene": np.asarray(group["agentview_rgb"][step]).copy(),
                            "wrist": np.asarray(group["eye_in_hand_rgb"][step]).copy(),
                            "state": state,
                            "step": step,
                        }
                    )
            if any(row["state"].shape != (8,) for row in values):
                raise S4TechnicalStop("S4 state shape changed")
            return values

        @torch.inference_mode()
        def prepare(identity: Mapping[str, Any], raw: Mapping[str, Any]) -> Any:
            cfg.unnorm_key = identity["normalization_statistics_key"]
            return prepare_query(
                torch_module=torch,
                np=np,
                model=model,
                processor=processor,
                proprio_projector=proprio_projector,
                prepare_images=prepare_images_for_vla,
                normalize_proprio=normalize_proprio,
                cfg=cfg,
                raw_scene=raw["scene"],
                raw_wrist=raw["wrist"],
                raw_state=raw["state"],
                instruction=identity["language_instruction"],
            )

        @torch.inference_mode()
        def official(identity: Mapping[str, Any], raw: Mapping[str, Any]) -> tuple[Any, Any]:
            consume()
            cfg.unnorm_key = identity["normalization_statistics_key"]
            configure_dense(model)
            observation = {
                "full_image": raw["scene"].copy(),
                "wrist_image": raw["wrist"].copy(),
                "state": raw["state"].copy(),
                "prev_images": [raw["scene"].copy(), raw["wrist"].copy()],
            }
            before = {
                "full": v1.array_sha256(observation["full_image"], np),
                "wrist": v1.array_sha256(observation["wrist_image"], np),
            }
            with OfficialBoundaryCapture(model, action_head) as capture:
                result = evaluation.get_action(
                    cfg,
                    model,
                    observation,
                    identity["language_instruction"],
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy,
                    use_film=False,
                )
            if not isinstance(result, tuple) or len(result) != 4:
                raise S4TechnicalStop("S4 official return contract changed")
            actions = np.asarray(result[0], dtype=np.float32)
            if actions.shape != (8, 7) or not np.isfinite(actions).all():
                raise S4TechnicalStop("S4 official actions are invalid")
            after = {
                "full": v1.array_sha256(observation["full_image"], np),
                "wrist": v1.array_sha256(observation["wrist_image"], np),
            }
            expected_state = normalize_proprio(
                raw["state"].copy(), model.norm_stats[cfg.unnorm_key]["proprio"]
            )
            if before != after or not np.array_equal(observation["state"], expected_state):
                raise S4TechnicalStop("S4 official observation isolation failed")
            return actions, capture.exact_values()["action_hidden"]

        @torch.inference_mode()
        def infer(prepared: Any, *, cache: Any, sidecar: bool = False) -> dict[str, Any]:
            consume()
            return forward_query(
                torch_module=torch,
                np=np,
                model=model,
                action_head=action_head,
                cfg=cfg,
                prepared=prepared,
                past_key_values=cache,
                capture_layers=tuple(config["model"]["sidecar_layers"]) if sidecar else (),
                capture_cac=True,
                return_actions=True,
            )

        def cache_disjoint(left: Any, right: Any) -> bool:
            return all(
                a.data_ptr() != b.data_ptr()
                for a, b in zip(
                    (*left.key_cache, *left.value_cache),
                    (*right.key_cache, *right.value_cache),
                    strict=True,
                )
            )

        selection_records = []
        allfresh_records = []
        for identity in inputs:
            raw_anchor, raw_current = load_window(identity, 2)
            anchor_prepared = prepare(identity, raw_anchor)
            current_prepared = prepare(identity, raw_current)
            official_actions, official_hidden = official(identity, raw_current)
            configure_dense(model)
            anchor = infer(anchor_prepared, cache=None, sidecar=True)
            positions = runtime_positions(anchor_prepared, torch)
            canonical_salience = anchor["tap"].salience(
                instruction_positions=positions["instruction"],
                action_positions=positions["action"],
                visual_positions=positions["primary"] + positions["wrist"],
            )
            legacy_salience = anchor["tap"].salience(
                instruction_positions=positions["prompt"],
                action_positions=positions["placeholder"],
                visual_positions=positions["primary"] + positions["wrist"],
            )
            tracker = PhysicalSourceTracker(0, anchor_prepared)
            tracker.add_record(1, current_prepared)
            legacy = ordered_tile_profile_vectorized(
                current=current_prepared,
                tracker=tracker,
                profile=profile,
                previous_salience=legacy_salience,
                torch_module=torch,
            )
            canonical = ordered_tile_profile_vectorized(
                current=current_prepared,
                tracker=tracker,
                profile=profile,
                previous_salience=canonical_salience,
                torch_module=torch,
            )
            legacy_digest = selection_digest(legacy[0], legacy[2])
            canonical_digest = selection_digest(canonical[0], canonical[2])
            protected_changed = legacy[3] != canonical[3]
            selection_records.append(
                {
                    "trajectory_id": identity["trajectory_id"],
                    "suite": identity["suite"],
                    "legacy_salience_sha256": v1.tensor_sha256(legacy_salience),
                    "canonical_salience_sha256": v1.tensor_sha256(canonical_salience),
                    "legacy_selection_sha256": legacy_digest,
                    "canonical_selection_sha256": canonical_digest,
                    "selection_changed": legacy_digest != canonical_digest,
                    "protected_tiles_changed": protected_changed,
                    "selected_positions": int(canonical[0].numel()),
                    "group_count": len(canonical[2]),
                }
            )
            parent_digest = cache_digest(anchor["cache"])
            branch_cache = clone_runtime_cache(anchor["cache"])
            clone_exact = cache_digest(branch_cache) == parent_digest
            clone_disjoint = cache_disjoint(anchor["cache"], branch_cache)
            configure_profile(
                model,
                torch.empty(0, device="cuda:0", dtype=torch.long),
                (0.0, 0.0, 0.0, 0.0),
                torch,
            )
            allfresh = infer(current_prepared, cache=branch_cache)
            hidden_error = float((official_hidden.float() - allfresh["action_hidden"].float()).abs().max().item())
            action_error = float(np.max(np.abs(official_actions - np.asarray(allfresh["actions"], dtype=np.float32))))
            allfresh_records.append(
                {
                    "trajectory_id": identity["trajectory_id"],
                    "suite": identity["suite"],
                    "hidden_max_abs": hidden_error,
                    "action_max_abs": action_error,
                    "official_action_sha256": v1.array_sha256(official_actions, np),
                    "allfresh_action_sha256": v1.array_sha256(allfresh["actions"], np),
                    "parent_cache_unchanged": cache_digest(anchor["cache"]) == parent_digest,
                    "cache_clone_exact": clone_exact,
                    "cache_clone_disjoint": clone_disjoint,
                }
            )
            if hidden_error > 0.000001 or action_error > 0.000001:
                raise S4TechnicalStop("S4 all-fresh path differs from official policy")
            if not all((clone_exact, clone_disjoint, cache_digest(anchor["cache"]) == parent_digest)):
                raise S4TechnicalStop("S4 all-fresh cache isolation failed")
            del anchor, allfresh, branch_cache, tracker, legacy, canonical
            del anchor_prepared, current_prepared, raw_anchor, raw_current
            gc.collect()
            torch.cuda.empty_cache()

        recursive_identity = inputs[2]
        recursive_raw = load_window(recursive_identity, 6)
        recursive_prepared = [prepare(recursive_identity, raw) for raw in recursive_raw]

        def recursive_cycle() -> dict[str, Any]:
            configure_dense(model)
            anchor = infer(recursive_prepared[0], cache=None, sidecar=True)
            positions = runtime_positions(recursive_prepared[0], torch)
            salience = anchor["tap"].salience(
                instruction_positions=positions["instruction"],
                action_positions=positions["action"],
                visual_positions=positions["primary"] + positions["wrist"],
            )
            parent_digest = cache_digest(anchor["cache"])
            cache = clone_runtime_cache(anchor["cache"])
            if not cache_disjoint(anchor["cache"], cache):
                raise S4TechnicalStop("S4 recursive cache clone aliases its parent")
            tracker = PhysicalSourceTracker(0, recursive_prepared[0])
            actions = [anchor["action_record"]["sha256"]]
            selections = []
            sources = []
            maximum_age = 0
            for ordinal in range(1, 5):
                tracker.add_record(ordinal, recursive_prepared[ordinal])
                ordered, proportions, groups, _metadata = ordered_tile_profile_vectorized(
                    current=recursive_prepared[ordinal],
                    tracker=tracker,
                    profile=profile,
                    previous_salience=salience,
                    torch_module=torch,
                )
                snapshots = capture_reused_cache(cache, ordered, profile, torch)
                configure_profile(model, ordered, proportions, torch)
                result = infer(recursive_prepared[ordinal], cache=cache)
                verify_reused_cache(result["cache"], snapshots)
                tracker.advance(ordinal, groups)
                ages = [ordinal - int(source) for source in tracker.sources.values()]
                maximum_age = max(maximum_age, max(ages))
                actions.append(result["action_record"]["sha256"])
                selections.append(selection_digest(ordered, groups))
                sources.append(tracker.digest())
                cache = result["cache"]
            if cache_digest(anchor["cache"]) != parent_digest:
                raise S4TechnicalStop("S4 recursive branch mutated its anchor cache")
            return {
                "action_sha256": actions,
                "selection_sha256": selections,
                "source_sha256": sources,
                "maximum_source_age": maximum_age,
                "final_mixture_count": tracker.mixture_count(),
                "parent_cache_unchanged": True,
                "reused_cache_exact": True,
            }

        first_cycle = recursive_cycle()
        second_cycle = recursive_cycle()
        recursive_exact = first_cycle == second_cycle
        if not recursive_exact or first_cycle["maximum_source_age"] != 4:
            raise S4TechnicalStop("S4 recursive age/determinism contract failed")

        reset_actions, reset_hidden = official(recursive_identity, recursive_raw[5])
        configure_dense(model)
        reset_first = infer(recursive_prepared[5], cache=None)
        configure_dense(model)
        reset_second = infer(recursive_prepared[5], cache=None)
        reset_hidden_error = float(
            (reset_hidden.float() - reset_first["action_hidden"].float()).abs().max().item()
        )
        reset_action_error = float(
            np.max(np.abs(reset_actions - np.asarray(reset_first["actions"], dtype=np.float32)))
        )
        reset_repeat_exact = (
            reset_first["action_record"] == reset_second["action_record"]
            and cache_digest(reset_first["cache"]) == cache_digest(reset_second["cache"])
        )
        if reset_hidden_error > 0.000001 or reset_action_error > 0.000001 or not reset_repeat_exact:
            raise S4TechnicalStop("S4 reset semantics failed")

        peak = sampler.close()
        peak = max(peak, v1.selected_gpu_snapshot(int(physical))["memory_used_mib"])
        if peak >= int(config["resource_caps"]["peak_gpu_memory_mib_strict_max"]):
            raise S4TechnicalStop("S4 strict aggregate-memory boundary reached")
        if calls != int(config["call_schedule"]["planned_model_calls"]):
            raise S4TechnicalStop(f"S4 planned-call accounting changed: {calls}")
        selection_material = any(
            row["selection_changed"] or row["protected_tiles_changed"]
            for row in selection_records
        )
        restoration = restore_checkpoint_exact(checkpoint, checkpoint_baseline)
        checkpoint_baseline = None
        if not all(
            restoration[key]
            for key in ("protected_bytes_restored", "backup_cleanup_complete", "inventory_equal")
        ):
            raise S4TechnicalStop("S4 checkpoint restoration failed")
        evidence = {
            "schema_version": "openvla-d62-requalification-s4-evidence-v1",
            "run_id": config["run_id"],
            "selection_records": selection_records,
            "allfresh_records": allfresh_records,
            "recursive_first": first_cycle,
            "recursive_second": second_cycle,
            "raw_actions_persisted": False,
        }
        evidence["semantic_sha256"] = semantic_sha256(evidence)
        v1.write_once(output_root / "evidence_manifest.json", evidence)
        artifact_bytes = v1.directory_size(output_root)
        if artifact_bytes >= int(config["resource_caps"]["artifact_bytes"]):
            raise S4TechnicalStop("S4 artifact cap exceeded")
        summary = {
            "schema_version": "openvla-d62-requalification-s4-result-v1",
            "run_id": config["run_id"],
            "complete": True,
            "passed": True,
            "historical_substrate_id": config["substrate"]["historical_id"],
            "corrected_substrate_id": config["substrate"]["corrected_candidate_id"],
            "selection_materially_changed": selection_material,
            "selection_changed_observations": sum(
                row["selection_changed"] or row["protected_tiles_changed"]
                for row in selection_records
            ),
            "allfresh_controls": len(allfresh_records),
            "maximum_allfresh_hidden_error": max(row["hidden_max_abs"] for row in allfresh_records),
            "maximum_allfresh_action_error": max(row["action_max_abs"] for row in allfresh_records),
            "recursive_repeat_exact": recursive_exact,
            "maximum_source_age": first_cycle["maximum_source_age"],
            "reset_hidden_max_abs": reset_hidden_error,
            "reset_action_max_abs": reset_action_error,
            "reset_repeat_exact": reset_repeat_exact,
            "model_calls": calls,
            "peak_aggregate_gpu_memory_mib": peak,
            "artifact_bytes": artifact_bytes,
            "evidence_manifest_sha256": file_sha256(output_root / "evidence_manifest.json"),
            "checkpoint_restoration": restoration,
            "simulator_outcomes_accessed": False,
            "expert_actions_accessed": False,
            "terminal_outcomes_accessed": False,
            "raw_actions_persisted": False,
            "automatic_retry": False,
            "elapsed_seconds": time.monotonic() - started,
            "advance": config["advance"],
            "completed_at": utc_now(),
        }
        summary["semantic_sha256"] = semantic_sha256(summary)
        v1.write_once(output_root / "worker_summary.json", summary)
        return 0
    except BaseException as error:
        peak = sampler.close()
        restoration = None
        restoration_error = None
        if checkpoint_baseline is not None and restore_checkpoint_exact is not None:
            try:
                restoration = restore_checkpoint_exact(checkpoint, checkpoint_baseline)
                checkpoint_baseline = None
            except BaseException as restore_error:
                restoration_error = f"{type(restore_error).__name__}: {restore_error}"
        stop = {
            "schema_version": "openvla-d62-requalification-s4-stop-v1",
            "run_id": config.get("run_id"),
            "complete": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "model_calls": calls,
            "peak_aggregate_gpu_memory_mib": peak,
            "checkpoint_restoration": restoration,
            "checkpoint_restoration_error": restoration_error,
            "simulator_outcomes_accessed": False,
            "expert_actions_accessed": False,
            "terminal_outcomes_accessed": False,
            "raw_actions_persisted": False,
            "automatic_retry": False,
            "elapsed_seconds": time.monotonic() - started,
            "stopped_at": utc_now(),
        }
        stop["semantic_sha256"] = semantic_sha256(stop)
        try:
            v1.write_once(output_root / "technical_stop.json", stop)
            descriptor = os.open(
                output_root / "technical_traceback.log",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(traceback.format_exc())
        except FileExistsError:
            pass
        return 1
    finally:
        signal.alarm(0)
        if config_baseline is not None and model is not None:
            model.language_model.config.proportion_attn_var = config_baseline[0]
            model.language_model.config.reusable_patches = config_baseline[1]
        model = action_head = proprio_projector = processor = None
        gc.collect()
        try:
            if "torch" in locals() and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except BaseException:
            pass
        if checkpoint_baseline is not None and restore_checkpoint_exact is not None:
            restore_checkpoint_exact(checkpoint, checkpoint_baseline)
        if evaluation is not None and loader_originals is not None:
            for name, value in loader_originals.items():
                setattr(evaluation, name, value)


if __name__ == "__main__":
    raise SystemExit(main())
