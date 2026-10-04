#!/usr/bin/env python3
"""Execute the single outcome-blind PAIR P3 physical worker."""

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
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path("/home/ved/SAVR")
DEFAULT_CONFIG = Path("configs/pair/p3_physical_v1.json")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def semantic_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


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
        os.fsync(stream.fileno())


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if ROOT != EXPECTED_ROOT:
        raise SystemExit(f"P3 worker refuses to run outside {EXPECTED_ROOT}")
    project_libero_config = ROOT / "configs/pair/libero_runtime"
    configured_libero_path = os.environ.get("LIBERO_CONFIG_PATH")
    if configured_libero_path and Path(configured_libero_path).resolve() != project_libero_config:
        raise SystemExit("P3 refuses a LIBERO configuration outside the project")
    if not (project_libero_config / "config.yaml").is_file():
        raise SystemExit("P3 project-local LIBERO configuration is missing")
    os.environ["LIBERO_CONFIG_PATH"] = str(project_libero_config)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    physical_gpu = os.environ.get("PAIR_PHYSICAL_GPU_ID", "")
    if not visible or "," in visible or not physical_gpu.isdigit():
        raise SystemExit("P3 requires exactly one explicitly selected GPU")
    sys.path.insert(0, str(ROOT / "src"))

    from savr.acr.v5_d_recovery import capture_checkpoint_baseline, restore_checkpoint_exact
    from savr.pair.p3 import (
        P3Profile,
        QueryLedger,
        block_schedule,
        planned_query_count,
        reject_protected_fields,
        semantic_sha256 as pair_semantic_sha256,
        validate_config,
    )

    config_path = (ROOT / args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    input_path = (ROOT / config["input_manifest"]).resolve()
    inputs_manifest = json.loads(input_path.read_text(encoding="utf-8"))
    supplied = inputs_manifest.get("semantic_sha256")
    if supplied != pair_semantic_sha256(inputs_manifest):
        raise SystemExit("P3 input manifest semantic hash mismatch")
    inputs = inputs_manifest["inputs"]
    validate_config(config, input_count=len(inputs))
    if args.output_root.exists():
        raise SystemExit(f"P3 immutable output already exists: {args.output_root}")
    args.output_root.mkdir(parents=True)
    progress_path = args.output_root / "blocks.jsonl"
    started_at = utc_now()
    started = time.monotonic()
    planned = planned_query_count(config, len(inputs))
    ledger = QueryLedger(int(config["resource_caps"]["model_query_hard_cap"]), planned)
    checkpoint = ROOT / config["model"]["checkpoint_relative"]
    protected = ("config.json", "configuration_prismatic.py", "modeling_prismatic.py")
    checkpoint_baseline = capture_checkpoint_baseline(checkpoint, protected)
    peak_aggregate = aggregate_memory_mib(int(physical_gpu))
    blocks_completed = 0

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
        from savr.pair.features import GroupFeature, RouterFeatures
        from savr.pair.p3_openvla import (
            PhysicalSourceTracker,
            cache_sample_digest,
            capture_reused_cache,
            configure_dense,
            configure_profile,
            forward_query,
            ordered_tile_profile,
            prepare_query,
            runtime_positions,
            verify_reused_cache,
        )
        from savr.pair.router import PairRouter
        from savr.pair.runtime import PairRuntime
        from savr.pair.types import AtomicGroup, Camera

        set_seed_everywhere(int(config["measurement"]["seed"]))
        cfg = base_config(evaluation, checkpoint, args.output_root)
        evaluation.validate_config(cfg)
        model, action_head, proprio_projector, _noisy_projector, processor = (
            evaluation.initialize_model(cfg)
        )
        model.eval()
        if torch.cuda.device_count() != 1:
            raise RuntimeError("P3 worker can see more or fewer than one GPU")
        cap_mib = int(config["resource_caps"]["peak_gpu_memory_mib_strict_max"])

        def check_memory() -> None:
            nonlocal peak_aggregate
            reserved_mib = int(torch.cuda.max_memory_reserved() / (1024 * 1024))
            aggregate = aggregate_memory_mib(int(physical_gpu))
            peak_aggregate = max(peak_aggregate, aggregate)
            if reserved_mib >= cap_mib or aggregate >= cap_mib:
                raise RuntimeError("P3 selected GPU reached the strict 23-GiB stop boundary")

        data_root = ROOT / inputs_manifest["data_root_relative"]

        def load_window(
            identity: Mapping[str, Any], start: int, horizon: int
        ) -> list[dict[str, Any]]:
            path = data_root / identity["source_path"]
            observations = []
            with h5py.File(path, "r") as handle:
                group = handle[f"data/{identity['original_trajectory_id']}/obs"]
                for ordinal in range(horizon + 1):
                    step = (start + ordinal) * 8
                    if step >= int(identity["step_count"]):
                        raise RuntimeError("P3 observation window exceeded original chronology")
                    scene = np.asarray(group["agentview_rgb"][step])
                    wrist = np.asarray(group["eye_in_hand_rgb"][step])
                    state = np.concatenate(
                        (
                            np.asarray(group["ee_states"][step]),
                            np.asarray(group["gripper_states"][step]),
                        )
                    )
                    if scene.ndim != 3 or wrist.ndim != 3 or state.shape != (8,):
                        raise RuntimeError("P3 outcome-blind observation layout changed")
                    observation_hash = hashlib.sha256(
                        scene.tobytes() + wrist.tobytes() + state.astype(np.float32).tobytes()
                    ).hexdigest()
                    observations.append(
                        {
                            "scene": scene,
                            "wrist": wrist,
                            "state": state,
                            "step": step,
                            "observation_sha256": observation_hash,
                        }
                    )
            return observations

        @torch.inference_mode()
        def execute_query(
            observation: Mapping[str, Any],
            identity: Mapping[str, Any],
            *,
            cache: Any = None,
            tracker: PhysicalSourceTracker | None = None,
            profile: P3Profile | None = None,
            query_ordinal: int = 0,
            anchor_salience: Any = None,
            capture_sidecar: bool = False,
            verify_cache: bool = False,
            all_fresh: bool = False,
        ) -> dict[str, Any]:
            if torch.is_grad_enabled():
                raise RuntimeError("P3 inference unexpectedly enabled gradients")
            cfg.unnorm_key = identity["normalization_statistics_key"]
            outer_start = time.perf_counter()
            outer_cuda_start = torch.cuda.Event(enable_timing=True)
            outer_cuda_end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            outer_cuda_start.record()
            prepare_start = time.perf_counter()
            prepared = prepare_query(
                torch_module=torch,
                np=np,
                model=model,
                processor=processor,
                proprio_projector=proprio_projector,
                prepare_images=prepare_images_for_vla,
                normalize_proprio=normalize_proprio,
                cfg=cfg,
                raw_scene=observation["scene"],
                raw_wrist=observation["wrist"],
                raw_state=observation["state"],
                instruction=identity["language_instruction"],
            )
            prepare_wall_ms = (time.perf_counter() - prepare_start) * 1000
            positions = runtime_positions(prepared, torch)
            ordered = None
            groups = None
            mask_metadata = None
            proportions = None
            gate_wall_ms = 0.0
            snapshots = None
            if all_fresh:
                ordered = torch.empty(0, device="cuda:0", dtype=torch.long)
                proportions = (0.0, 0.0, 0.0, 0.0)
                configure_profile(model, ordered, proportions, torch)
            elif profile is None:
                configure_dense(model)
            else:
                if tracker is None or anchor_salience is None:
                    raise RuntimeError("P3 cached query lacks provenance or prior salience")
                tracker.add_record(query_ordinal, prepared)
                gate_start = time.perf_counter()
                ordered, proportions, groups, mask_metadata = ordered_tile_profile(
                    current=prepared,
                    tracker=tracker,
                    profile=profile,
                    previous_salience=anchor_salience,
                    torch_module=torch,
                )
                configure_profile(model, ordered, proportions, torch)
                if verify_cache:
                    snapshots = capture_reused_cache(cache, ordered, profile, torch)
                gate_wall_ms = (time.perf_counter() - gate_start) * 1000
            ledger.consume()
            result = forward_query(
                torch_module=torch,
                np=np,
                model=model,
                action_head=action_head,
                cfg=cfg,
                prepared=prepared,
                past_key_values=cache,
                capture_layers=config["model"]["sidecar_layers"] if capture_sidecar else (),
            )
            sidecar_wall_ms = 0.0
            salience = None
            if capture_sidecar:
                sidecar_start = time.perf_counter()
                salience = result["tap"].salience(
                    instruction_positions=positions["instruction"],
                    action_positions=positions["action"],
                    visual_positions=positions["primary"] + positions["wrist"],
                )
                torch.cuda.synchronize()
                sidecar_wall_ms = (time.perf_counter() - sidecar_start) * 1000
            provenance_start = time.perf_counter()
            if snapshots is not None:
                verify_reused_cache(result["cache"], snapshots)
            if profile is not None and groups is not None and tracker is not None:
                tracker.advance(query_ordinal, groups)
            provenance_wall_ms = (time.perf_counter() - provenance_start) * 1000
            outer_cuda_end.record()
            torch.cuda.synchronize()
            total_wall_ms = (time.perf_counter() - outer_start) * 1000
            total_cuda_ms = float(outer_cuda_start.elapsed_time(outer_cuda_end))
            check_memory()
            expected_length = result["full_sequence_length"]
            if len(result["cache"].key_cache) != 32 or any(
                int(tensor.shape[-2]) != expected_length for tensor in result["cache"].key_cache
            ):
                raise RuntimeError("P3 cache sequence/layer shape changed")
            return {
                "cache": result["cache"],
                "prepared": prepared,
                "salience": salience,
                "action_record": result["action_record"],
                "total_wall_ms": total_wall_ms,
                "total_cuda_ms": total_cuda_ms,
                "decoder_wall_ms": result["wall_ms"],
                "decoder_cuda_ms": result["cuda_ms"],
                "prepare_wall_ms": prepare_wall_ms,
                "gate_wall_ms": gate_wall_ms,
                "sidecar_wall_ms": sidecar_wall_ms,
                "provenance_wall_ms": provenance_wall_ms,
                "active_sequence_length": result["active_sequence_length"],
                "full_sequence_length": result["full_sequence_length"],
                "source_digest": tracker.digest() if tracker is not None else None,
                "source_mixture_count": tracker.mixture_count() if tracker is not None else 1,
                "mask_metadata": mask_metadata,
                "underlying_sdpa_calls": result["tap"].calls if result["tap"] is not None else 0,
            }

        def public_query_record(
            value: Mapping[str, Any], observation: Mapping[str, Any]
        ) -> dict[str, Any]:
            return {
                key: value[key]
                for key in (
                    "action_record",
                    "total_wall_ms",
                    "total_cuda_ms",
                    "decoder_wall_ms",
                    "decoder_cuda_ms",
                    "prepare_wall_ms",
                    "gate_wall_ms",
                    "sidecar_wall_ms",
                    "provenance_wall_ms",
                    "active_sequence_length",
                    "full_sequence_length",
                    "source_digest",
                    "source_mixture_count",
                    "mask_metadata",
                    "underlying_sdpa_calls",
                )
            } | {
                "original_step": observation["step"],
                "observation_sha256": observation["observation_sha256"],
            }

        def run_dense_cycle(
            identity: Mapping[str, Any], observations: list[dict[str, Any]]
        ) -> dict[str, Any]:
            records = [execute_query(observation, identity) for observation in observations]
            return {
                "cycle_wall_ms": sum(item["total_wall_ms"] for item in records),
                "cycle_cuda_ms": sum(item["total_cuda_ms"] for item in records),
                "queries": [
                    public_query_record(value, observation)
                    for value, observation in zip(records, observations, strict=True)
                ],
                "service_count": 0,
                "fallback_count": 0,
            }

        def run_cache_cycle(
            identity: Mapping[str, Any],
            observations: list[dict[str, Any]],
            profile: P3Profile,
            *,
            verify_first: bool = False,
        ) -> dict[str, Any]:
            anchor = execute_query(observations[0], identity, capture_sidecar=True)
            tracker = PhysicalSourceTracker(0, anchor["prepared"])
            cache = anchor["cache"]
            records = [anchor]
            for ordinal, observation in enumerate(observations[1:], start=1):
                value = execute_query(
                    observation,
                    identity,
                    cache=cache,
                    tracker=tracker,
                    profile=profile,
                    query_ordinal=ordinal,
                    anchor_salience=anchor["salience"],
                    verify_cache=verify_first and ordinal == 1,
                )
                cache = value["cache"]
                records.append(value)
            return {
                "cycle_wall_ms": sum(item["total_wall_ms"] for item in records),
                "cycle_cuda_ms": sum(item["total_cuda_ms"] for item in records),
                "queries": [
                    public_query_record(value, observation)
                    for value, observation in zip(records, observations, strict=True)
                ],
                "service_count": len(observations) - 1,
                "fallback_count": 0,
                "final_source_digest": tracker.digest(),
                "maximum_source_mixture_count": max(
                    int(item["source_mixture_count"]) for item in records
                ),
            }

        # Four dense model warmups.
        warm_identity = inputs[0]
        warm_observations = load_window(warm_identity, 0, 3)
        for observation in warm_observations:
            execute_query(observation, warm_identity)

        # Four protected technical controls: sidecar off/on and dense/all-fresh cache paths.
        control_observation = load_window(warm_identity, 0, 0)[0]
        sidecar_off = execute_query(control_observation, warm_identity)
        sidecar_on = execute_query(control_observation, warm_identity, capture_sidecar=True)
        sidecar_control = {
            "off_cache_sample_sha256": cache_sample_digest(sidecar_off["cache"], torch),
            "on_cache_sample_sha256": cache_sample_digest(sidecar_on["cache"], torch),
            "cache_sample_match": cache_sample_digest(sidecar_off["cache"], torch)
            == cache_sample_digest(sidecar_on["cache"], torch),
            "underlying_sdpa_calls": sidecar_on["underlying_sdpa_calls"],
            "off_total_wall_ms": sidecar_off["total_wall_ms"],
            "on_total_wall_ms": sidecar_on["total_wall_ms"],
            "on_sidecar_extract_wall_ms": sidecar_on["sidecar_wall_ms"],
            "incremental_upper_ms": max(
                0.0,
                sidecar_on["total_wall_ms"] - sidecar_off["total_wall_ms"],
                sidecar_on["sidecar_wall_ms"],
            ),
            "off_action_record": sidecar_off["action_record"],
            "on_action_record": sidecar_on["action_record"],
        }
        dense_control = execute_query(control_observation, warm_identity)
        dense_cache_digest = cache_sample_digest(dense_control["cache"], torch)
        all_fresh = execute_query(
            control_observation,
            warm_identity,
            cache=dense_control["cache"],
            all_fresh=True,
        )
        all_fresh_control = {
            "dense_cache_sample_sha256": dense_cache_digest,
            "all_fresh_cache_sample_sha256": cache_sample_digest(all_fresh["cache"], torch),
            "cache_sample_match": dense_cache_digest
            == cache_sample_digest(all_fresh["cache"], torch),
            "dense_action_record": dense_control["action_record"],
            "all_fresh_action_record": all_fresh["action_record"],
        }
        if not sidecar_control["cache_sample_match"] or not all_fresh_control["cache_sample_match"]:
            raise RuntimeError("P3 physical sidecar or all-fresh cache control failed")

        profiles = [P3Profile.from_mapping(value) for value in config["profiles"]]
        for index, profile in enumerate(profiles):
            identity = inputs[index % len(inputs)]
            observations = load_window(identity, 0, 4)
            run_cache_cycle(identity, observations, profile, verify_first=True)

        schedule = block_schedule(config, len(inputs))
        for block in schedule:
            identity = inputs[block.input_index]
            available = int(identity["eligible_query_count"]) - block.horizon
            if available <= 0:
                raise RuntimeError("P3 input lacks the frozen contract horizon")
            start = int(block.block_id[:8], 16) % available
            observations = load_window(identity, start, block.horizon)
            profile = next(item for item in profiles if item.profile_id == block.profile_id)
            arms = {}
            for arm in block.arm_order:
                if arm == "dense":
                    arms[arm] = run_dense_cycle(identity, observations)
                else:
                    arms[arm] = run_cache_cycle(identity, observations, profile)
            record = {
                "schema_version": "pair-p3-block-v1",
                "run_id": config["run_id"],
                "block_id": block.block_id,
                "profile_id": block.profile_id,
                "horizon": block.horizon,
                "repetition": block.repetition,
                "arm_order": block.arm_order,
                "suite": identity["suite"],
                "task_id": identity["task_id"],
                "trajectory_id": identity["trajectory_id"],
                "start_query_ordinal": start,
                "arms": arms,
            }
            reject_protected_fields(record)
            record["semantic_sha256"] = semantic_sha256(record)
            append_jsonl(progress_path, record)
            blocks_completed += 1
            check_memory()

        ledger.require_complete()

        # CPU mock costs: serialized reference router, provenance state, and reset.
        router_profiles = list(dict.fromkeys(item.base_profile_id for item in profiles))
        router = PairRouter.initialize(profiles=router_profiles, seed=17)
        mock_features = RouterFeatures(
            (
                GroupFeature(
                    AtomicGroup(Camera.PRIMARY, 0, 2),
                    profiles[0].base_profile_id,
                    1,
                    (0.0,) * 20,
                ),
            ),
            (0.0,) * 16,
        )
        router_ms = []
        reset_ms = []
        for _ in range(int(config["overhead"]["mock_iterations"])):
            start = time.perf_counter()
            router.predict(mock_features)
            router_ms.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            PairRuntime.reset("p3-overhead")
            reset_ms.append((time.perf_counter() - start) * 1000)

        completed_at = utc_now()
        result = {
            "schema_version": "pair-p3-worker-v1",
            "run_id": config["run_id"],
            "status": "completed",
            "configuration_semantic_sha256": config["semantic_sha256"],
            "input_manifest_semantic_sha256": inputs_manifest["semantic_sha256"],
            "project_revision": subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "selected_gpu": int(physical_gpu),
            "visible_gpu_count": 1,
            "queries_used": ledger.used,
            "queries_planned": ledger.planned,
            "blocks_completed": blocks_completed,
            "blocks_expected": len(schedule),
            "sidecar_control": sidecar_control,
            "all_fresh_control": all_fresh_control,
            "overhead_mock": {
                "iterations": len(router_ms),
                "router_p99_ms": float(np.quantile(router_ms, 0.99)),
                "reset_p99_ms": float(np.quantile(reset_ms, 0.99)),
            },
            "memory": {
                "peak_allocated_mib": int(torch.cuda.max_memory_allocated() / (1024 * 1024)),
                "peak_reserved_mib": int(torch.cuda.max_memory_reserved() / (1024 * 1024)),
                "peak_aggregate_mib": peak_aggregate,
                "strict_limit_mib": cap_mib,
            },
            "protection": {
                "action_values_persisted": False,
                "action_comparisons_performed": False,
                "expert_actions_accessed": False,
                "terminal_outcomes_accessed": False,
                "simulator_used": False,
                "downloads": 0,
            },
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "wall_seconds": time.monotonic() - started,
            "blocks_file_sha256": file_sha256(progress_path),
            "advance": {"next_phase": "P4", "authorized": False},
        }
        reject_protected_fields(result)
        result["semantic_sha256"] = semantic_sha256(result)
        write_once(args.output_root / "worker_summary.json", result)
        print(
            json.dumps({"status": "completed", "queries": ledger.used, "blocks": blocks_completed})
        )
        return 0
    except BaseException as error:
        try:
            peak_aggregate = max(peak_aggregate, aggregate_memory_mib(int(physical_gpu)))
        except Exception:
            pass
        stop = {
            "schema_version": "pair-p3-technical-stop-v1",
            "run_id": config["run_id"],
            "status": "technical_stop",
            "reason": str(error),
            "queries_used": ledger.used,
            "queries_planned": ledger.planned,
            "blocks_completed": blocks_completed,
            "peak_aggregate_mib": peak_aggregate,
            "simulator_used": False,
            "protected_outcome_access": False,
            "automatic_retry": False,
            "completed_at_utc": utc_now(),
        }
        stop["semantic_sha256"] = semantic_sha256(stop)
        write_once(args.output_root / "technical_stop.json", stop)
        raise
    finally:
        restoration = restore_checkpoint_exact(checkpoint, checkpoint_baseline)
        if not (
            restoration["protected_bytes_restored"]
            and restoration["backup_cleanup_complete"]
            and restoration["inventory_equal"]
        ):
            raise RuntimeError("P3 checkpoint metadata restoration failed")


if __name__ == "__main__":
    raise SystemExit(main())
