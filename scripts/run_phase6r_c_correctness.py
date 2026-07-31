#!/usr/bin/env python3
"""Run the bounded Phase 6R-C real-model SAVR 2.0 correctness matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from run_phase4_correctness import (
    CHECKPOINT_RELATIVE,
    CHECKPOINT_REVISION,
    LIBERO_REVISION,
    UPSTREAM_REVISION,
    atomic_json,
    ensure_artifact_cap,
    exact_parity,
    make_state_b,
    require_clean_revision,
    sha256,
    utc_now,
    validate_checkpoint_inventory,
    validate_schemas,
)


EXPECTED_ROOT = Path("/home/ved/SAVR")
RUN_ID = "phase6r-c-correctness-v1"
SUITE = "libero_spatial"
TASK_ID = 0
INITIAL_STATE_ID = 0
SEED = 0
QUERY_CAP = 20
PLANNED_QUERIES = 10
ARTIFACT_CAP_BYTES = 256 * 1024**2
MAX_WALL_SECONDS = 45 * 60
RECOVERY_RUN_ID = "phase6r-c-trace-recovery-v1"
RECOVERY_PLANNED_QUERIES = 8
RECOVERY_TRACE = Path("results/phase6-fr-signals-v1/queries/query_00000000.json")
RECOVERY_TRACE_SHA256 = "ff9f4bfc004b861260e36d61c5eab641356a9c27c25f7ceccf511e04dd687a63"


def query_record(
    *,
    index: int,
    path: str,
    actions: Any,
    timing: Any,
    refresh: bool,
    cache_event: str,
    decision_seconds: float,
    np: Any,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    action_array = np.asarray(actions)
    if action_array.shape != (8, 7) or not np.isfinite(action_array).all():
        raise RuntimeError(f"Query {index} returned invalid actions: {action_array.shape}")
    component_ms = dict(timing.component_device_ms)
    component_counts = dict(timing.component_counts)
    for name in ("vision_backbone", "visual_projector", "language_model", "action_head"):
        component_ms.setdefault(name, 0.0)
        component_counts.setdefault(name, 0)
    return {
        "run_id": RUN_ID,
        "query_index": index,
        "environment_step": 0,
        "status": "completed",
        "path": path,
        "refresh": refresh,
        "cache_event": cache_event,
        "action_shape": list(action_array.shape),
        "actions_sha256": hashlib.sha256(action_array.tobytes()).hexdigest(),
        "timing": {
            "decision_wall_ms": decision_seconds * 1000,
            "query_wall_ms": timing.wall_ms,
            "total_cuda_ms": timing.total_device_ms,
            "component_cuda_ms": component_ms,
            "component_counts": component_counts,
        },
        **(extra or {}),
    }


def assert_counts(record: dict[str, Any], *, visual: int) -> None:
    counts = record["timing"]["component_counts"]
    expected = {
        "vision_backbone": visual,
        "visual_projector": visual,
        "language_model": 1,
        "action_head": 1,
    }
    actual = {name: counts.get(name, 0) for name in expected}
    if actual != expected:
        raise RuntimeError(f"Component counts differ: expected {expected}, found {actual}")


def main() -> int:
    global RUN_ID
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replay-existing-fr-trace",
        action="store_true",
        help="Use the frozen Phase 6 FR trace without simulator execution",
    )
    arguments = parser.parse_args()
    recovery = bool(arguments.replay_existing_fr_trace)
    if recovery:
        RUN_ID = RECOVERY_RUN_ID
    planned_queries = RECOVERY_PLANNED_QUERIES if recovery else PLANNED_QUERIES

    project_root = Path(__file__).resolve().parents[1]
    if project_root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing to run outside {EXPECTED_ROOT}: {project_root}")
    if planned_queries > QUERY_CAP:
        raise SystemExit("Planned queries exceed the frozen Phase 6R-C cap")

    physical_gpu_id = os.environ.get("SAVR_PHYSICAL_GPU_ID")
    selected_uuid = os.environ.get("SAVR_SELECTED_GPU_UUID")
    if not physical_gpu_id or os.environ.get("CUDA_VISIBLE_DEVICES") != physical_gpu_id:
        raise SystemExit("SAVR_PHYSICAL_GPU_ID must exactly match CUDA_VISIBLE_DEVICES")
    if not selected_uuid:
        raise SystemExit("SAVR_SELECTED_GPU_UUID is required")

    def handle_timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError("Phase 6R-C reached its frozen 45-minute wall cap")

    signal.signal(signal.SIGALRM, handle_timeout)
    signal.alarm(MAX_WALL_SECONDS)

    for key, relative in {
        "HF_HOME": "cache/huggingface",
        "HF_HUB_CACHE": "cache/huggingface/hub",
        "LIBERO_CONFIG_PATH": "cache/libero",
        "TORCH_HOME": "cache/torch",
    }.items():
        os.environ[key] = str(project_root / relative)
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
            "PYTHONNOUSERSITE": "1",
            "WANDB_MODE": "disabled",
            "TOKENIZERS_PARALLELISM": "false",
            "TF_CPP_MIN_LOG_LEVEL": "2",
        }
    )

    upstream_root = project_root / "third_party" / "openvla-oft"
    libero_root = project_root / "third_party" / "LIBERO"
    checkpoint = project_root / CHECKPOINT_RELATIVE
    run_dir = project_root / "results" / RUN_ID
    if run_dir.exists():
        raise SystemExit(f"Run ID is immutable and already exists: {run_dir}")

    project_revision = require_clean_revision(project_root)
    upstream_revision = require_clean_revision(upstream_root, UPSTREAM_REVISION)
    libero_revision = require_clean_revision(libero_root, LIBERO_REVISION)
    checkpoint_inventory = validate_checkpoint_inventory(project_root, checkpoint)

    sys.path.insert(0, str(project_root / "src"))
    sys.path.insert(0, str(upstream_root))

    import numpy as np
    import torch  # type: ignore[import-not-found]
    from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
    from jsonschema.validators import validate  # type: ignore[import-untyped]
    from libero.libero import benchmark  # type: ignore[import-not-found]

    from experiments.robot.libero import (  # type: ignore[import-not-found]
        run_libero_eval as upstream_eval,
    )
    from experiments.robot.libero.libero_utils import (  # type: ignore[import-not-found]
        get_libero_env,
    )
    from experiments.robot.openvla_utils import (  # type: ignore[import-not-found]
        normalize_proprio,
    )
    from experiments.robot.robot_utils import (  # type: ignore[import-not-found]
        get_image_resize_size,
        set_seed_everywhere,
    )
    from savr.cache import CacheContext
    from savr.controllers import FullRefreshController
    from savr.integration.openvla_oft import OpenVLAProjectedFeatureAdapter
    from savr.logging import ImmutableRecordStore
    from savr.savr2 import SAVR2Configuration, StateAwareVisualRefresh2Controller
    from savr.timing import ModuleTimingHooks, SynchronizedQueryTimer, TorchCudaEventBackend

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one selected CUDA device must be visible")

    schemas = validate_schemas(project_root, Draft202012Validator, validate)
    store = ImmutableRecordStore(run_dir)
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "run_summary.json"
    protected_names = ("config.json", "configuration_prismatic.py", "modeling_prismatic.py")
    protected_bytes = {name: (checkpoint / name).read_bytes() for name in protected_names}
    protected_hashes = {name: sha256(checkpoint / name) for name in protected_names}
    checkpoint_files_before = {item.name for item in checkpoint.iterdir()}

    manifest: dict[str, Any] = {
        "run_id": RUN_ID,
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "status": "running",
        "policy": "MIXED",
        "savr_git_revision": project_revision,
        "working_tree_clean": True,
        "base_model": {
            "name": "OpenVLA-OFT",
            "checkpoint": str(CHECKPOINT_RELATIVE),
            "revision": CHECKPOINT_REVISION,
        },
        "benchmark": {"name": "LIBERO", "revision": libero_revision, "suite": SUITE},
        "hardware": {
            "physical_gpu_id": physical_gpu_id,
            "selected_gpu_uuid": selected_uuid,
            "cuda_visible_devices": physical_gpu_id,
        },
        "software": {
            "openvla_oft_revision": upstream_revision,
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "configuration": {
            "task_id": TASK_ID,
            "initial_state_id": INITIAL_STATE_ID,
            "seed": SEED,
            "planned_queries": planned_queries,
            "query_cap": QUERY_CAP,
            "rollout_episodes": 0,
            "simulator_resets": 0 if recovery else 1,
            "artifact_cap_bytes": ARTIFACT_CAP_BYTES,
            "wall_cap_seconds": MAX_WALL_SECONDS,
            "checkpoint_inventory": checkpoint_inventory,
            "replay_existing_fr_trace": recovery,
            "recovery_trace": str(RECOVERY_TRACE) if recovery else None,
            "recovery_trace_sha256": RECOVERY_TRACE_SHA256 if recovery else None,
        },
        "command": (
            "scripts/run_phase6r_c_correctness.py --replay-existing-fr-trace"
            if recovery
            else "scripts/run_phase6r_c_correctness.py"
        ),
        "notes": "Phase 6R-C correctness only; no rollout or final-holdout access.",
    }
    validate(manifest, schemas["run_manifest.schema.json"])
    atomic_json(manifest_path, manifest)
    store.write_run_event(0, "RUNNING", {"started_at_utc": manifest["started_at_utc"]})

    cfg = upstream_eval.GenerateConfig(
        pretrained_checkpoint=str(checkpoint),
        task_suite_name=SUITE,
        num_trials_per_task=1,
        seed=SEED,
        local_log_dir=str(run_dir / "logs"),
        use_wandb=False,
        center_crop=True,
        num_open_loop_steps=8,
        num_images_in_input=2,
        use_proprio=True,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
    )
    upstream_eval.validate_config(cfg)
    set_seed_everywhere(SEED)
    torch.cuda.reset_peak_memory_stats()

    model = action_head = proprio_projector = noisy_action_projector = processor = None
    timing_hooks = proprio_handle = current_env = None
    caught_error: BaseException | None = None
    summary: dict[str, Any] = {"run_id": RUN_ID, "status": "failed"}
    started = time.monotonic()
    try:
        os.chdir(upstream_root)
        model, action_head, proprio_projector, noisy_action_projector, processor = (
            upstream_eval.initialize_model(cfg)
        )
        torch.cuda.synchronize()
        if action_head is None or proprio_projector is None:
            raise RuntimeError("Pinned L1/proprio modules were not loaded")

        state_stats = model.norm_stats[cfg.unnorm_key]["proprio"]
        action_stats = model.norm_stats[cfg.unnorm_key]["action"]
        manifest["normalization_statistics"] = {
            "unnorm_key": cfg.unnorm_key,
            "state_q01": [float(value) for value in state_stats["q01"]],
            "state_q99": [float(value) for value in state_stats["q99"]],
            "action_q01": [float(value) for value in action_stats["q01"]],
            "action_q99": [float(value) for value in action_stats["q99"]],
        }
        atomic_json(manifest_path, manifest)

        task_suite = benchmark.get_benchmark_dict()[SUITE]()
        task = task_suite.get_task(TASK_ID)
        task_description = task.language
        if recovery:
            trace_path = project_root / RECOVERY_TRACE
            if sha256(trace_path) != RECOVERY_TRACE_SHA256:
                raise RuntimeError("Frozen Phase 6 FR recovery trace hash differs")
            trace_record = json.loads(trace_path.read_text(encoding="utf-8"))
            if (
                trace_record.get("episode_id") != "fr_task_00_state_00"
                or trace_record.get("configuration_id") != "fr"
                or trace_record.get("environment_step") != 10
            ):
                raise RuntimeError("Frozen Phase 6 FR recovery trace identity differs")
            trace = trace_record["calibration_trace"]
            state_a = np.asarray(trace["state"], dtype=np.float64)
            state_b = state_a.copy()
            perturbation = None
            images = {}
            for name in ("full_image", "wrist_image"):
                shape = tuple(int(value) for value in trace["image_shapes"][name])
                if shape != (32, 32, 3):
                    raise RuntimeError(f"Recovery trace image shape differs for {name}")
                normalized = np.asarray(trace["images"][name], dtype=np.float64).reshape(shape)
                if not np.isfinite(normalized).all() or not (
                    (normalized >= 0).all() and (normalized <= 1).all()
                ):
                    raise RuntimeError(f"Recovery trace image values are invalid for {name}")
                images[name] = np.rint(normalized * 255).astype(np.uint8)
        else:
            resize_size = get_image_resize_size(cfg)
            initial_states = task_suite.get_task_init_states(TASK_ID)
            current_env, task_description = get_libero_env(
                task, cfg.model_family, resolution=cfg.env_img_res
            )
            current_env.reset()
            raw_observation = current_env.set_init_state(initial_states[INITIAL_STATE_ID])
            policy_observation, _ = upstream_eval.prepare_observation(
                raw_observation, resize_size
            )
            current_env.close()
            current_env = None
            state_a = np.asarray(policy_observation["state"], dtype=np.float64).copy()
            state_b, perturbation = make_state_b(state_a, state_stats, np)
            images = {
                "full_image": np.asarray(policy_observation["full_image"]).copy(),
                "wrist_image": np.asarray(policy_observation["wrist_image"]).copy(),
            }

        def observation(state: Any) -> dict[str, Any]:
            return {
                "full_image": images["full_image"].copy(),
                "wrist_image": images["wrist_image"].copy(),
                "state": np.asarray(state).copy(),
            }

        def upstream_query(state: Any) -> Any:
            return upstream_eval.get_action(
                cfg,
                model,
                observation(state),
                task_description,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                use_film=cfg.use_film,
            )

        timer = SynchronizedQueryTimer(TorchCudaEventBackend(torch))
        timing_hooks = ModuleTimingHooks(
            {
                "vision_backbone": model.vision_backbone,
                "visual_projector": model.projector,
                "language_model": model.language_model,
                "action_head": action_head.model,
            },
            timer,
        )
        proprio_inputs: list[Any] = []

        def capture_proprio(_module: Any, inputs: Any) -> None:
            proprio_inputs.append(inputs[0].detach().cpu().clone())

        proprio_handle = proprio_projector.register_forward_pre_hook(capture_proprio)

        fr = OpenVLAProjectedFeatureAdapter(model=model, controller=FullRefreshController())
        fr.begin_context(
            CacheContext("phase6r-c-fr", str(TASK_ID), CHECKPOINT_REVISION, "phase6r-c-fr")
        )
        candidate = SAVR2Configuration(
            configuration_id="savr2-b15",
            image_thresholds={"full_image": 10.0, "wrist_image": 10.0},
            state_thresholds={"translation": 10.0, "orientation": 10.0, "gripper": 10.0},
            action_thresholds={"translation": 10.0, "rotation": 10.0, "gripper": 10.0},
            skip_budget=0.15,
        )
        savr2_controller = StateAwareVisualRefresh2Controller(
            configuration=candidate,
            state_q01=state_stats["q01"],
            state_q99=state_stats["q99"],
            action_q01=action_stats["q01"],
            action_q99=action_stats["q99"],
        )
        savr2 = OpenVLAProjectedFeatureAdapter(model=model, controller=savr2_controller)
        savr2.begin_context(
            CacheContext("phase6r-c-savr2", str(TASK_ID), CHECKPOINT_REVISION, "savr2-b15")
        )

        records: list[dict[str, Any]] = []

        def publish(record: dict[str, Any]) -> None:
            validate(record, schemas["query_record.schema.json"])
            store.write_query(record["query_index"], record)
            records.append(record)
            ensure_artifact_cap(run_dir)

        timer.start()
        upstream_a = upstream_query(state_a)
        timing = timer.finish()
        record = query_record(
            index=0,
            path="unmodified_upstream_state_a",
            actions=upstream_a,
            timing=timing,
            refresh=True,
            cache_event="unmodified",
            decision_seconds=0.0,
            np=np,
        )
        assert_counts(record, visual=1)
        publish(record)

        savr2_record_offset = 1
        if not recovery:
            timer.start()
            fr_result = fr.run_query(
                query=lambda: upstream_query(state_a),
                images=images,
                state=state_a,
                environment_step=0,
            )
            timing = timer.finish()
            fr_parity = exact_parity(upstream_a, fr_result.value, np)
            record = query_record(
                index=1,
                path="wrapped_fr_state_a",
                actions=fr_result.value,
                timing=timing,
                refresh=fr_result.decision.refresh,
                cache_event=fr_result.cache_event,
                decision_seconds=fr_result.decision_seconds,
                np=np,
                extra={"decision": asdict(fr_result.decision), "parity": fr_parity},
            )
            assert_counts(record, visual=1)
            publish(record)
            savr2_record_offset = 2

        for query_index in range(6):
            timer.start()
            result = savr2.run_query(
                query=lambda: upstream_query(state_a),
                images=images,
                state=state_a,
                environment_step=query_index * 8,
            )
            timing = timer.finish()
            if not result.decision.refresh or result.cache_event != "refresh":
                raise RuntimeError(f"SAVR 2.0 warm-up query {query_index} did not refresh")
            record = query_record(
                index=savr2_record_offset + query_index,
                path=f"savr2_fresh_q{query_index}",
                actions=result.value,
                timing=timing,
                refresh=result.decision.refresh,
                cache_event=result.cache_event,
                decision_seconds=result.decision_seconds,
                np=np,
                extra={"decision": asdict(result.decision)},
            )
            assert_counts(record, visual=1)
            publish(record)

        reuse_state = state_a if recovery else state_b
        upstream_reuse_reference = upstream_a
        reuse_record_index = savr2_record_offset + 6
        if not recovery:
            timer.start()
            upstream_reuse_reference = upstream_query(state_b)
            timing = timer.finish()
            record = query_record(
                index=reuse_record_index,
                path="unmodified_upstream_state_b",
                actions=upstream_reuse_reference,
                timing=timing,
                refresh=True,
                cache_event="unmodified",
                decision_seconds=0.0,
                np=np,
            )
            assert_counts(record, visual=1)
            publish(record)
            reuse_record_index += 1

        proprio_before = len(proprio_inputs)
        timer.start()
        reuse = savr2.run_query(
            query=lambda: upstream_query(reuse_state),
            images=images,
            state=reuse_state,
            environment_step=48,
        )
        timing = timer.finish()
        if reuse.decision.refresh or reuse.cache_event != "reuse":
            raise RuntimeError(f"SAVR 2.0 correctness query did not reuse: {reuse.decision}")
        reuse_parity = exact_parity(upstream_reuse_reference, reuse.value, np)
        if len(proprio_inputs) != proprio_before + 1:
            raise RuntimeError("SAVR 2.0 reuse did not execute current proprioception once")
        expected_proprio = torch.tensor(
            normalize_proprio(reuse_state, state_stats), dtype=proprio_inputs[-1].dtype
        ).float().numpy().reshape(-1)
        actual_proprio = proprio_inputs[-1].float().numpy().reshape(-1)
        if not np.array_equal(actual_proprio, expected_proprio):
            raise RuntimeError("SAVR 2.0 reuse did not receive current proprioception")
        if set(reuse.decision.camera_patch_scores) != {"full_image", "wrist_image"}:
            raise RuntimeError("SAVR 2.0 decision omitted a camera patch trace")
        if any(len(values) != 64 for values in reuse.decision.camera_patch_scores.values()):
            raise RuntimeError("SAVR 2.0 decision omitted patch scores")
        record = query_record(
            index=reuse_record_index,
            path="savr2_reuse_trace_state" if recovery else "savr2_reuse_state_b",
            actions=reuse.value,
            timing=timing,
            refresh=reuse.decision.refresh,
            cache_event=reuse.cache_event,
            decision_seconds=reuse.decision_seconds,
            np=np,
            extra={
                "decision": asdict(reuse.decision),
                "parity": reuse_parity,
                "fresh_proprio_array_equal": True,
                "normalized_reuse_state": expected_proprio.tolist(),
            },
        )
        assert_counts(record, visual=0)
        publish(record)

        if len(records) != planned_queries:
            raise RuntimeError("Phase 6R-C query count differs from the frozen plan")
        snapshot = savr2_controller.snapshot()
        if snapshot.completed_reuses != 1 or snapshot.query_index != 7:
            raise RuntimeError(f"SAVR 2.0 counters do not reconcile: {snapshot}")
        summary = {
            "run_id": RUN_ID,
            "status": "completed",
            "query_count": len(records),
            "query_cap": QUERY_CAP,
            "rollout_episode_count": 0,
            "simulator_reset_count": 0 if recovery else 1,
            "exact_fr_parity": True,
            "fr_parity_source": (
                "phase6r-c-correctness-v1 queries 0-1"
                if recovery
                else RUN_ID
            ),
            "exact_reuse_parity": True,
            "reuse_visual_backbone_calls": 0,
            "reuse_visual_projector_calls": 0,
            "fresh_proprio_on_reuse": True,
            "savr2_snapshot": asdict(snapshot),
            "controlled_state_perturbation": perturbation,
            "recovery_trace_sha256": RECOVERY_TRACE_SHA256 if recovery else None,
            "elapsed_seconds": time.monotonic() - started,
            "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    except BaseException as error:
        caught_error = error
        summary = {
            "run_id": RUN_ID,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_seconds": time.monotonic() - started,
        }
    finally:
        signal.alarm(0)
        if current_env is not None:
            current_env.close()
        if proprio_handle is not None:
            proprio_handle.remove()
        if timing_hooks is not None:
            timing_hooks.remove()
        for name, content in protected_bytes.items():
            (checkpoint / name).write_bytes(content)
        new_files = {item.name for item in checkpoint.iterdir()} - checkpoint_files_before
        backup_files = sorted(name for name in new_files if ".back." in name)
        for name in backup_files:
            (checkpoint / name).unlink()
        unexpected_files = sorted(new_files - set(backup_files))
        hashes_after = {name: sha256(checkpoint / name) for name in protected_names}
        checkpoint_restored = hashes_after == protected_hashes and not unexpected_files
        summary.update(
            {
                "checkpoint_restored": checkpoint_restored,
                "unexpected_new_checkpoint_files": unexpected_files,
                "finished_at_utc": utc_now(),
            }
        )
        if not checkpoint_restored:
            summary["status"] = "failed"
            caught_error = caught_error or RuntimeError("Checkpoint restoration audit failed")
            summary["error_type"] = type(caught_error).__name__
            summary["error"] = str(caught_error)
        manifest["status"] = summary["status"]
        manifest["finished_at_utc"] = summary["finished_at_utc"]
        manifest["hardware"].update(
            {
                "visible_gpu_name": torch.cuda.get_device_name(0),
                "visible_gpu_capability": list(torch.cuda.get_device_capability(0)),
                "peak_memory_allocated_bytes": summary.get("peak_gpu_memory_allocated_bytes"),
                "peak_memory_reserved_bytes": summary.get("peak_gpu_memory_reserved_bytes"),
            }
        )
        validate(manifest, schemas["run_manifest.schema.json"])
        atomic_json(summary_path, summary)
        atomic_json(manifest_path, manifest)
        store.write_run_event(
            1,
            summary["status"].upper(),
            {
                "finished_at_utc": summary["finished_at_utc"],
                "checkpoint_restored": checkpoint_restored,
                "query_count": len(store.completed_query_indices()),
            },
        )
        ensure_artifact_cap(run_dir)
        if model is not None:
            del model
        torch.cuda.empty_cache()

    print(json.dumps(summary, indent=2, sort_keys=True))
    if caught_error is not None:
        print(f"Phase 6R-C stopped: {type(caught_error).__name__}: {caught_error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
