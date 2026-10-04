#!/usr/bin/env python3
"""Execute the single outcome-sealed PAIR P4B confirmation worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path("/home/ved/SAVR")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("ab") as stream:
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()


def aggregate_memory_mib(gpu: int) -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    return int(output)


def base_config(eval_module: Any, checkpoint: Path, run_root: Path) -> Any:
    return eval_module.GenerateConfig(
        pretrained_checkpoint=str(checkpoint),
        task_suite_name="libero_object",
        num_trials_per_task=0,
        seed=0,
        local_log_dir=str(run_root / "logs"),
        use_wandb=False,
        center_crop=True,
        num_open_loop_steps=8,
        num_images_in_input=2,
        use_proprio=True,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        use_vla_cache=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if ROOT != EXPECTED_ROOT:
        raise SystemExit(f"P4B worker refuses to run outside {EXPECTED_ROOT}")
    output_root = args.output_root.resolve()
    expected_output = (ROOT / "results/pair-p4b-confirmatory-v01").resolve()
    if output_root != expected_output or output_root.exists():
        raise SystemExit("P4B immutable output must be new and below project results")
    project_libero_config = ROOT / "configs/pair/libero_runtime"
    configured = os.environ.get("LIBERO_CONFIG_PATH")
    if configured and Path(configured).resolve() != project_libero_config:
        raise SystemExit("P4B refuses a LIBERO configuration outside the project")
    os.environ["LIBERO_CONFIG_PATH"] = str(project_libero_config)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    physical_gpu = os.environ.get("PAIR_PHYSICAL_GPU_ID", "")
    if not visible or "," in visible or not physical_gpu.isdigit():
        raise SystemExit("P4B requires exactly one explicitly selected GPU")
    sys.path.insert(0, str(ROOT / "src"))

    from savr.pair.p4b import expected_counts, semantic_sha256, validate_config

    config_path = (ROOT / args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config, ROOT)
    launch_path = (ROOT / args.launch_manifest).resolve()
    if launch_path != (ROOT / config["artifacts"]["launch_manifest"]).resolve():
        raise SystemExit("P4B launch-manifest path changed")
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    if (
        launch.get("schema_version") != "pair-p4b-launch-manifest-v1"
        or launch.get("status") != "frozen_not_started"
        or launch.get("semantic_sha256") != semantic_sha256(launch)
        or launch.get("configuration_semantic_sha256") != config["semantic_sha256"]
        or launch.get("config_sha256") != file_sha256(config_path)
        or launch.get("code_sha256", {}).get("scripts/run_pair_p4b_worker.py")
        != file_sha256(Path(__file__).resolve())
    ):
        raise SystemExit("P4B launch manifest changed")
    schedule_path = ROOT / config["artifacts"]["schedule"]
    schedule_summary_path = ROOT / config["artifacts"]["schedule_summary"]
    schedule_data = schedule_path.read_bytes()
    schedule_summary = json.loads(schedule_summary_path.read_text(encoding="utf-8"))
    schedule = [json.loads(line) for line in schedule_data.splitlines()]
    counts = expected_counts()
    if (
        launch.get("schedule_sha256") != hashlib.sha256(schedule_data).hexdigest()
        or launch.get("schedule_summary_sha256") != file_sha256(schedule_summary_path)
        or launch.get("preflight_sha256")
        != file_sha256(ROOT / config["artifacts"]["preflight"])
        or
        schedule_summary.get("status") != "completed"
        or schedule_summary.get("schedule_sha256") != hashlib.sha256(schedule_data).hexdigest()
        or schedule_summary.get("counts") != counts
        or len(schedule) != counts["anchors"]
        or any(row["split"] not in {"train", "calibration"} for row in schedule)
        or any(row["run_id"] != config["run_id"] for row in schedule)
    ):
        raise SystemExit("P4B schedule is invalid")

    output_root.mkdir(parents=True)
    intervention_path = output_root / "interventions.sealed.jsonl"
    feature_path = output_root / "features.sealed.jsonl"
    contract_path = output_root / "contracts.sealed.jsonl"
    started_at = utc_now()
    started = time.monotonic()
    intervention_count = 0
    feature_count = 0
    contract_count = 0
    peak_aggregate = aggregate_memory_mib(int(physical_gpu))
    checkpoint_baseline = None
    checkpoint = None

    try:
        source = ROOT / "third_party/vla-cache/src/openvla-oft"
        os.chdir(source)
        sys.path.insert(0, str(source))
        import h5py
        import numpy as np
        import torch
        from experiments.robot.libero import run_libero_eval as evaluation
        from experiments.robot.openvla_utils import normalize_proprio, prepare_images_for_vla
        from experiments.robot.robot_utils import set_seed_everywhere
        from savr.acr.v5_d_recovery import capture_checkpoint_baseline, restore_checkpoint_exact
        from savr.pair.interventions import clone_runtime_cache
        from savr.pair.metrics import normalized_l1_regret
        from savr.pair.p3 import P3Profile, QueryLedger
        from savr.pair.p3_openvla import (
            PhysicalSourceTracker,
            cache_sample_digest,
            configure_dense,
            configure_profile,
            ordered_tile_profile,
            ordered_tile_profile_vectorized,
            prepare_query,
            runtime_positions,
        )
        from savr.pair.p4_openvla import (
            action_digest,
            anchor_instruction_projection,
            arbitrary_group_profile,
            build_router_features,
            forward_normalized_query,
            serialize_router_features,
        )
        from savr.pair.records import semantic_sha256 as record_sha256
        from savr.pair.records import validate_schema
        from savr.pair.types import AtomicGroup, Camera

        set_seed_everywhere(int(config["schedule"]["seed"]))
        model_config = json.loads((ROOT / config["model_config"]).read_text(encoding="utf-8"))
        checkpoint = ROOT / model_config["model"]["checkpoint_relative"]
        checkpoint_baseline = capture_checkpoint_baseline(
            checkpoint, ("config.json", "configuration_prismatic.py", "modeling_prismatic.py")
        )
        cfg = base_config(evaluation, checkpoint, output_root)
        evaluation.validate_config(cfg)
        model, action_head, proprio_projector, _noisy_projector, processor = (
            evaluation.initialize_model(cfg)
        )
        model.eval()
        if torch.cuda.device_count() != 1:
            raise RuntimeError("P4B worker can see more or fewer than one GPU")
        ledger = QueryLedger(
            int(config["resource_caps"]["model_call_hard_cap"]),
            int(config["accounting"]["planned_model_calls"]),
        )
        cap_mib = int(config["gates"]["peak_gpu_memory_mib_strict_max"])
        data_root = ROOT / "data/pair/libero_hdf5/f13aa24a3da8c43c7225569f28c562979fa0e35a"
        intervention_schema = ROOT / "schemas/pair/intervention_record.schema.json"
        feature_schema = ROOT / "schemas/pair/p4_feature_record.schema.json"
        contract_schema = ROOT / "schemas/pair/p4_contract_record.schema.json"
        profile = P3Profile.from_mapping(config["profile"])

        def check_resources() -> None:
            nonlocal peak_aggregate
            aggregate = aggregate_memory_mib(int(physical_gpu))
            reserved = int(torch.cuda.max_memory_reserved() / (1024 * 1024))
            peak_aggregate = max(peak_aggregate, aggregate)
            if aggregate >= cap_mib or reserved >= cap_mib:
                raise RuntimeError("P4B selected GPU reached the strict memory boundary")
            if time.monotonic() - started >= int(config["resource_caps"]["wall_seconds"]):
                raise RuntimeError("P4B wall-time cap reached")
            artifact_bytes = sum(
                path.stat().st_size for path in output_root.rglob("*") if path.is_file()
            )
            if artifact_bytes >= int(config["resource_caps"]["artifact_bytes"]):
                raise RuntimeError("P4B artifact cap reached")

        def normalize_expert(raw: Any, key: str) -> Any:
            values = np.asarray(raw, dtype=np.float32).copy()
            if values.shape != (8, 7):
                raise RuntimeError("P4B expert chunk is not exact 8x7")
            values[:, -1] = np.float32(1.0) - np.clip(
                values[:, -1], np.float32(0.0), np.float32(1.0)
            )
            metadata = model.norm_stats[key]["action"]
            low = np.asarray(metadata["q01"], dtype=np.float32)
            high = np.asarray(metadata["q99"], dtype=np.float32)
            mask = np.asarray(metadata.get("mask", [True] * 7), dtype=bool)
            scaled = np.clip(
                np.float32(2.0) * (values - low) / (high - low + np.float32(1e-8))
                - np.float32(1.0),
                np.float32(-1.0),
                np.float32(1.0),
            )
            result = np.where(mask, scaled, values)
            result = np.where(
                np.asarray(metadata["min"], dtype=np.float32)
                == np.asarray(metadata["max"], dtype=np.float32),
                0.0,
                result,
            ).astype(np.float32)
            if not np.isfinite(result).all():
                raise RuntimeError("P4B normalized expert chunk is nonfinite")
            return result

        def load_anchor(
            row: Mapping[str, Any], *, include_experts: bool = True
        ) -> tuple[list[Any], list[Any]]:
            observations = []
            experts = []
            steps = [row["anchor_original_step"], *row["future_original_steps"]]
            with h5py.File(data_root / row["source_path"], "r") as handle:
                for step in steps:
                    scene = np.asarray(handle[row["primary_camera_dataset"]][step])
                    wrist = np.asarray(handle[row["wrist_camera_dataset"]][step])
                    state = np.concatenate(
                        tuple(np.asarray(handle[path][step]) for path in row["proprio_datasets"])
                    )
                    if scene.ndim != 3 or wrist.ndim != 3 or state.shape != (8,):
                        raise RuntimeError("P4B observation layout changed")
                    observations.append((scene, wrist, state))
                if include_experts:
                    action_data = handle[row["action_dataset"]]
                    for step in row["future_original_steps"]:
                        experts.append(
                            normalize_expert(
                                np.asarray(action_data[step : step + 8]),
                                row["normalization_statistics_key"],
                            )
                        )
            return observations, experts

        @torch.inference_mode()
        def prepare(row: Mapping[str, Any], observation: Sequence[Any]) -> Any:
            cfg.unnorm_key = row["normalization_statistics_key"]
            return prepare_query(
                torch_module=torch,
                np=np,
                model=model,
                processor=processor,
                proprio_projector=proprio_projector,
                prepare_images=prepare_images_for_vla,
                normalize_proprio=normalize_proprio,
                cfg=cfg,
                raw_scene=observation[0],
                raw_wrist=observation[1],
                raw_state=observation[2],
                instruction=row["language_instruction"],
            )

        @torch.inference_mode()
        def infer(prepared: Any, cache: Any = None, *, sidecar: bool = False) -> dict[str, Any]:
            ledger.consume()
            result = forward_normalized_query(
                torch_module=torch,
                model=model,
                action_head=action_head,
                prepared=prepared,
                past_key_values=cache,
                capture_layers=model_config["model"]["sidecar_layers"] if sidecar else (),
            )
            check_resources()
            return result

        def groups_from_mapping(values: Mapping[tuple[Camera, int], int]) -> tuple[AtomicGroup, ...]:
            return tuple(
                sorted(
                    (AtomicGroup(camera, tile, onset) for (camera, tile), onset in values.items()),
                    key=lambda item: (item.onset_layer, item.camera.value, item.tile),
                )
            )

        def previous_gripper_transition(action: Any) -> bool:
            signs = np.asarray(action)[:, -1] >= 0
            return bool(np.any(signs[1:] != signs[:-1]))

        def freeze_record(record: dict[str, Any], schema: Path) -> dict[str, Any]:
            record["record_id"] = record_sha256(record)
            validate_schema(record, schema)
            return record

        # Prepare one bounded technical window without reading its expert labels.
        control_row = schedule[0]
        control_observations, _unused_experts = load_anchor(
            control_row, include_experts=False
        )
        control_prepared = [prepare(control_row, value) for value in control_observations]
        configure_dense(model)
        for _ in range(4):
            infer(control_prepared[0])
        configure_dense(model)
        sidecar_off = infer(control_prepared[0])
        configure_dense(model)
        sidecar_on = infer(control_prepared[0], sidecar=True)
        positions = runtime_positions(control_prepared[0], torch)
        salience = sidecar_on["tap"].salience(
            instruction_positions=positions["instruction"],
            action_positions=positions["action"],
            visual_positions=positions["primary"] + positions["wrist"],
        )
        configure_dense(model)
        dense_control = infer(control_prepared[0])
        configure_profile(
            model,
            torch.empty(0, device="cuda:0", dtype=torch.long),
            (0.0, 0.0, 0.0, 0.0),
            torch,
        )
        allfresh_control = infer(control_prepared[0], clone_runtime_cache(dense_control["cache"]))
        tracker_control = PhysicalSourceTracker(0, control_prepared[0])
        tracker_control.add_record(1, control_prepared[1])
        legacy = ordered_tile_profile(
            current=control_prepared[1],
            tracker=tracker_control,
            profile=profile,
            previous_salience=salience,
            torch_module=torch,
        )
        vectorized = ordered_tile_profile_vectorized(
            current=control_prepared[1],
            tracker=tracker_control,
            profile=profile,
            previous_salience=salience,
            torch_module=torch,
        )
        controls = {
            "sidecar_cache_match": cache_sample_digest(sidecar_off["cache"], torch)
            == cache_sample_digest(sidecar_on["cache"], torch),
            "sidecar_action_max_abs": float(
                np.max(np.abs(sidecar_off["normalized_action"] - sidecar_on["normalized_action"]))
            ),
            "allfresh_cache_match": cache_sample_digest(dense_control["cache"], torch)
            == cache_sample_digest(allfresh_control["cache"], torch),
            "allfresh_action_max_abs": float(
                np.max(
                    np.abs(
                        dense_control["normalized_action"]
                        - allfresh_control["normalized_action"]
                    )
                )
            ),
            "vectorized_legacy_equal": bool(
                torch.equal(legacy[0], vectorized[0]) and legacy[1:] == vectorized[1:]
            ),
        }
        if not all((controls["sidecar_cache_match"], controls["allfresh_cache_match"], controls["vectorized_legacy_equal"])):
            raise RuntimeError("P4B bounded technical controls failed")

        for anchor_index, row in enumerate(schedule):
            observations, experts = load_anchor(row)
            prepared = [prepare(row, value) for value in observations]
            configure_dense(model)
            anchor = infer(prepared[0], sidecar=True)
            positions = runtime_positions(prepared[0], torch)
            anchor_salience = anchor["tap"].salience(
                instruction_positions=positions["instruction"],
                action_positions=positions["action"],
                visual_positions=positions["primary"] + positions["wrist"],
            )
            projection = anchor_instruction_projection(
                prepared[0], seed=int(config["schedule"]["instruction_projection_seed"])
            )
            dense_actions = []
            for current in prepared[1:]:
                configure_dense(model)
                dense_actions.append(infer(current)["normalized_action"])

            base_contract_id = hashlib.sha256(
                f"{row['schedule_id']}|base".encode()
            ).hexdigest()
            if row["category"] == "all_fresh_control":
                forced_base = tuple(() for _ in range(row["horizon"]))
            else:
                forced_base = None

            def run_branch(
                branch_kind: str,
                contract_id: str,
                forced: Sequence[Sequence[AtomicGroup]] | None,
                *,
                write_features: bool,
                repeat_ids: Sequence[str] | None = None,
            ) -> tuple[dict[str, Any], tuple[tuple[AtomicGroup, ...], ...]]:
                nonlocal intervention_count, feature_count, contract_count
                cache = clone_runtime_cache(anchor["cache"])
                tracker = PhysicalSourceTracker(0, prepared[0])
                action_history = {0: anchor["normalized_action"]}
                realized = []
                diagnostics = []
                record_ids = []
                for query_ordinal, current in enumerate(prepared[1:], start=1):
                    tracker.add_record(query_ordinal, current)
                    if forced is None:
                        selected = ordered_tile_profile_vectorized(
                            current=current,
                            tracker=tracker,
                            profile=profile,
                            previous_salience=anchor_salience,
                            torch_module=torch,
                        )
                        ordered, proportions, mapping, _metadata = selected
                        groups = groups_from_mapping(mapping)
                    else:
                        groups = tuple(forced[query_ordinal - 1])
                        ordered, proportions, mapping = arbitrary_group_profile(
                            groups, torch, device=current.projected_patches.device
                        )
                    realized.append(groups)
                    feature_values = None
                    if write_features:
                        feature_values = build_router_features(
                            current=current,
                            tracker=tracker,
                            groups=groups,
                            profile_id=profile.profile_id,
                            horizon=row["horizon"],
                            query_ordinal=query_ordinal,
                            remaining=row["horizon"] - query_ordinal,
                            action_history=action_history,
                            anchor_salience=anchor_salience,
                            gripper_transition=previous_gripper_transition(
                                action_history[query_ordinal - 1]
                            ),
                            global_projection=projection,
                            torch_module=torch,
                        )
                    configure_profile(model, ordered, proportions, torch)
                    branch = infer(current, cache)
                    cache = branch["cache"]
                    source_ages = [
                        query_ordinal
                        - tracker.source_query(group.onset_layer, group.camera, group.tile)
                        for group in groups
                    ] or [0]
                    tracker.advance(query_ordinal, mapping)
                    action_history[query_ordinal] = branch["normalized_action"]
                    diagnostic = normalized_l1_regret(
                        experts[query_ordinal - 1],
                        dense_actions[query_ordinal - 1],
                        branch["normalized_action"],
                        np.ones(8, dtype=bool),
                    )
                    diagnostics.append(diagnostic)
                    mask_payload = {
                        "schedule_id": row["schedule_id"],
                        "branch_kind": branch_kind,
                        "query_ordinal": query_ordinal,
                        "groups": [group.identity for group in groups],
                    }
                    intervention = freeze_record(
                        {
                            "schema_version": "pair-intervention-v1",
                            "run_id": config["run_id"],
                            "phase": "P4B",
                            "split": row["split"],
                            "suite": row["suite"],
                            "task_id": row["task_id"],
                            "trajectory_id": row["trajectory_id"],
                            "query_step": row["future_original_steps"][query_ordinal - 1],
                            "horizon": row["horizon"],
                            "profile_id": profile.profile_id,
                            "mask_id": hashlib.sha256(canonical_bytes(mask_payload)).hexdigest(),
                            "source_ages": source_ages,
                            "valid_action_count": diagnostic.valid_action_count,
                            "dense_expert_l1": diagnostic.dense_expert_l1,
                            "cached_expert_l1": diagnostic.cached_expert_l1,
                            "signed_regret": diagnostic.signed_regret,
                            "positive_regret": diagnostic.positive_regret,
                            "action_l1_distortion": diagnostic.action_l1_distortion,
                            "action_max_abs_distortion": diagnostic.action_max_abs_distortion,
                            "gripper_coordinate_error": diagnostic.gripper_coordinate_error,
                            "expert_action_sha256": action_digest(experts[query_ordinal - 1]),
                            "dense_action_sha256": action_digest(
                                dense_actions[query_ordinal - 1]
                            ),
                            "cached_action_sha256": branch["action_sha256"],
                            "gripper_transition": previous_gripper_transition(
                                experts[query_ordinal - 1]
                            ),
                            "repeat_of_record_id": (
                                repeat_ids[query_ordinal - 1] if repeat_ids is not None else None
                            ),
                            "technical_valid": True,
                            "created_at_utc": utc_now(),
                        },
                        intervention_schema,
                    )
                    append_jsonl(intervention_path, intervention)
                    intervention_count += 1
                    record_ids.append(intervention["record_id"])
                    if feature_values is not None:
                        feature_record = freeze_record(
                            {
                                "schema_version": "pair-p4-feature-v1",
                                "run_id": config["run_id"],
                                "split": row["split"],
                                "contract_id": contract_id,
                                "base_intervention_record_id": intervention["record_id"],
                                "suite": row["suite"],
                                "task_id": row["task_id"],
                                "trajectory_id": row["trajectory_id"],
                                "query_ordinal": query_ordinal,
                                "horizon": row["horizon"],
                                "profile_id": profile.profile_id,
                                "features": serialize_router_features(feature_values),
                                "created_at_utc": utc_now(),
                            },
                            feature_schema,
                        )
                        append_jsonl(feature_path, feature_record)
                        feature_count += 1
                signed = float(np.mean([value.signed_regret for value in diagnostics]))
                contract = freeze_record(
                    {
                        "schema_version": "pair-p4-contract-v1",
                        "run_id": config["run_id"],
                        "split": row["split"],
                        "contract_id": contract_id,
                        "base_contract_id": base_contract_id,
                        "branch_kind": branch_kind,
                        "category": row["category"],
                        "suite": row["suite"],
                        "task_id": row["task_id"],
                        "trajectory_id": row["trajectory_id"],
                        "horizon": row["horizon"],
                        "profile_id": profile.profile_id,
                        "intervention_record_ids": record_ids,
                        "signed_contract_regret": signed,
                        "positive_contract_regret": max(0.0, signed),
                        "mean_action_l1_distortion": float(
                            np.mean([value.action_l1_distortion for value in diagnostics])
                        ),
                        "maximum_action_abs_distortion": float(
                            max(value.action_max_abs_distortion for value in diagnostics)
                        ),
                        "mean_gripper_coordinate_error": float(
                            np.mean([value.gripper_coordinate_error for value in diagnostics])
                        ),
                        "exact_repeat": branch_kind == "repeat",
                        "repeat_of_contract_id": (
                            base_contract_id if branch_kind == "repeat" else None
                        ),
                        "technical_valid": True,
                        "created_at_utc": utc_now(),
                    },
                    contract_schema,
                )
                append_jsonl(contract_path, contract)
                contract_count += 1
                return contract, tuple(realized)

            base_contract, realized = run_branch(
                "base", base_contract_id, forced_base, write_features=row["category"] != "all_fresh_control"
            )
            if row["exact_repeat"]:
                run_branch(
                    "repeat",
                    hashlib.sha256(f"{row['schedule_id']}|repeat".encode()).hexdigest(),
                    realized,
                    write_features=False,
                    repeat_ids=base_contract["intervention_record_ids"],
                )
            if (anchor_index + 1) % 10 == 0:
                print(
                    json.dumps(
                        {
                            "anchors_completed": anchor_index + 1,
                            "intervention_records": intervention_count,
                            "feature_records": feature_count,
                            "contract_records": contract_count,
                            "model_calls": ledger.used,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        ledger.require_complete()
        if (
            intervention_count != counts["intervention_records"]
            or feature_count != counts["feature_records"]
            or contract_count != counts["contract_records"]
        ):
            raise RuntimeError("P4B final record accounting mismatch")
        completed_at = utc_now()
        summary = {
            "schema_version": "pair-p4b-worker-summary-v1",
            "run_id": config["run_id"],
            "status": "completed",
            "configuration_semantic_sha256": config["semantic_sha256"],
            "schedule_sha256": schedule_summary["schedule_sha256"],
            "schedule_summary_sha256": file_sha256(schedule_summary_path),
            "intervention_records": intervention_count,
            "feature_records": feature_count,
            "contract_records": contract_count,
            "model_calls": ledger.used,
            "artifact_sha256": {
                "interventions": file_sha256(intervention_path),
                "features": file_sha256(feature_path),
                "contracts": file_sha256(contract_path),
            },
            "technical_controls": controls,
            "gpu_ids": [int(physical_gpu)],
            "peak_gpu_memory_mib": peak_aggregate,
            "artifact_bytes": sum(
                path.stat().st_size for path in output_root.rglob("*") if path.is_file()
            ),
            "expert_values_persisted": False,
            "raw_model_actions_persisted": False,
            "simulator_accessed": False,
            "locked_test_labels_accessed": False,
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
        }
        summary["semantic_sha256"] = semantic_sha256(summary)
        write_once(output_root / "worker_summary.json", summary)
        print(json.dumps({"status": "completed", "model_calls": ledger.used}), flush=True)
        return 0
    except Exception as error:
        stop = {
            "schema_version": "pair-p4b-technical-stop-v1",
            "run_id": config["run_id"],
            "status": "technical_stop",
            "reason_type": type(error).__name__,
            "reason": str(error),
            "intervention_records": intervention_count,
            "feature_records": feature_count,
            "contract_records": contract_count,
            "peak_gpu_memory_mib": peak_aggregate,
            "started_at_utc": started_at,
            "stopped_at_utc": utc_now(),
        }
        stop["semantic_sha256"] = semantic_sha256(stop)
        write_once(output_root / "technical_stop.json", stop)
        print(json.dumps({"status": "technical_stop", "reason_type": type(error).__name__}))
        return 1
    finally:
        if checkpoint_baseline is not None and checkpoint is not None:
            restore_checkpoint_exact(checkpoint, checkpoint_baseline)


if __name__ == "__main__":
    raise SystemExit(main())
