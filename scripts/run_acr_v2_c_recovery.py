#!/usr/bin/env python3
"""Run the frozen 41-query V2-C latency recovery."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any


EXPECTED_ROOT = Path("/home/ved/SAVR")
RUN_ID = "acr-v2c-latency-recovery-v01"
PARENT_RUN_ID = "acr-v2c-correctness-latency-v01"
RECOVERY_CONFIG = Path("configs/acr/v2_c_recovery.json")
GATE_CONFIG = Path("configs/acr/v2_c_gate.json")
RECOVERY_QUERY_CAP = 41
REMAINING_WARMUPS = {
    "upstream-fr": 1,
    "dual-path-refresh": 2,
    "dual-path-reuse": 2,
}
TIMED_QUERIES = 36
PARENT_QUERIES = 7
WALL_CAP_SECONDS = 3600
ARTIFACT_CAP_BYTES = 512 * 1024**2


class QueryBudget:
    def __init__(self) -> None:
        self.labels: list[str] = []

    def consume(self, label: str) -> int:
        if len(self.labels) >= RECOVERY_QUERY_CAP:
            raise RuntimeError(f"V2-C recovery query cap exceeded before {label}")
        self.labels.append(label)
        return len(self.labels) - 1


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def validate_recovery_config(config: dict[str, Any]) -> None:
    supplied = config.get("semantic_sha256")
    payload = dict(config)
    payload.pop("semantic_sha256", None)
    if semantic_sha256(payload) != supplied:
        raise RuntimeError("V2-C recovery semantic hash mismatch")
    if config["parent_queries"]["total_consumed"] != PARENT_QUERIES:
        raise RuntimeError("V2-C parent query count changed")
    recovery = config["recovery_queries"]
    if recovery["total"] != RECOVERY_QUERY_CAP or recovery["cumulative_total"] != 48:
        raise RuntimeError("V2-C recovery no longer exhausts exactly 48 cumulative queries")
    if sum(REMAINING_WARMUPS.values()) + TIMED_QUERIES != RECOVERY_QUERY_CAP:
        raise RuntimeError("V2-C recovery schedule is inconsistent")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    if project_root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing to run outside {EXPECTED_ROOT}: {project_root}")
    physical_gpu_id = os.environ.get("SAVR_PHYSICAL_GPU_ID")
    selected_uuid = os.environ.get("SAVR_SELECTED_GPU_UUID")
    if not physical_gpu_id or os.environ.get("CUDA_VISIBLE_DEVICES") != physical_gpu_id:
        raise SystemExit("SAVR_PHYSICAL_GPU_ID must equal CUDA_VISIBLE_DEVICES")
    if not selected_uuid:
        raise SystemExit("SAVR_SELECTED_GPU_UUID is required")

    def timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError("V2-C recovery reached its one-hour wall cap")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(WALL_CAP_SECONDS)
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
            "PYTHONNOUSERSITE": "1",
            "WANDB_MODE": "disabled",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )

    sys.path.insert(0, str(project_root / "src"))
    sys.path.insert(0, str(project_root / "scripts"))
    from run_acr_v2_c import (  # type: ignore[import-not-found]
        CHECKPOINT_RELATIVE,
        CHECKPOINT_REVISION,
        COUNTERBALANCE,
        INSTRUCTION,
        LIBERO_REVISION,
        OPENVLA_REVISION,
        array_sha256,
        directory_size,
        file_sha256,
        require_clean_revision,
        selected_gpu_snapshot,
        summarize_timing,
        synthetic_images,
        utc_now,
        validate_gate_config,
    )

    recovery_config = json.loads((project_root / RECOVERY_CONFIG).read_text())
    gate_config = json.loads((project_root / GATE_CONFIG).read_text())
    validate_recovery_config(recovery_config)
    validate_gate_config(gate_config)
    parent_root = project_root / "results" / PARENT_RUN_ID
    parent_failure = parent_root / "failure" / "record.json"
    parent_manifest = parent_root / "manifest" / "record.json"
    parent_runner = project_root / "scripts/run_acr_v2_c.py"
    expected_hashes = {
        parent_failure: recovery_config["parent_failure_sha256"],
        parent_manifest: recovery_config["parent_manifest_sha256"],
        parent_runner: recovery_config["parent_runner_sha256"],
    }
    for evidence_path, expected in expected_hashes.items():
        if not evidence_path.is_file() or file_sha256(evidence_path) != expected:
            raise RuntimeError(f"V2-C parent evidence changed: {evidence_path}")
    failure = json.loads(parent_failure.read_text())
    expected_labels = [
        "correctness-upstream-refresh",
        "correctness-dual-path-refresh",
        "correctness-acr-v1-reuse",
        "correctness-dual-path-reuse",
        "correctness-cache-mismatch-refresh",
        "correctness-exception-restoration",
        "warmup-00-0-upstream-fr",
    ]
    if (
        failure.get("query_labels") != expected_labels
        or failure.get("query_count") != PARENT_QUERIES
        or "component counts differ" not in failure.get("error", "")
    ):
        raise RuntimeError("V2-C parent control-flow evidence is inconsistent")

    upstream_root = project_root / "third_party/openvla-oft"
    libero_root = project_root / "third_party/LIBERO"
    checkpoint = project_root / CHECKPOINT_RELATIVE
    run_root = project_root / "results" / RUN_ID
    if run_root.exists():
        raise SystemExit(f"Immutable recovery run already exists: {run_root}")
    source_revision = require_clean_revision(project_root)
    require_clean_revision(upstream_root, OPENVLA_REVISION)
    require_clean_revision(libero_root, LIBERO_REVISION)

    sys.path.insert(0, str(upstream_root))
    import numpy as np
    import torch  # type: ignore[import-not-found]
    from experiments.robot.libero import run_libero_eval as upstream_eval  # type: ignore[import-not-found]
    from experiments.robot.robot_utils import set_seed_everywhere  # type: ignore[import-not-found]
    from run_acr_correctness import validate_checkpoint  # type: ignore[import-not-found]
    from savr.acr.controller import ACRController
    from savr.acr.dual_path import DualPathOpenVLAAdapter
    from savr.acr.instrumentation import CameraInstrumentation
    from savr.acr.openvla_oft import TorchTensorOperations
    from savr.acr.records import ImmutableRecordStore
    from savr.acr.signals import normalized_eef_position, prepare_scene_representation
    from savr.acr.types import ACRConfiguration, ACRContext, ACRPolicy
    from savr.timing import ModuleTimingHooks, SynchronizedQueryTimer, TorchCudaEventBackend

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one selected CUDA device must be visible")
    checkpoint_before = validate_checkpoint(project_root, checkpoint)
    protected_names = ("config.json", "configuration_prismatic.py", "modeling_prismatic.py")
    protected_bytes = {name: (checkpoint / name).read_bytes() for name in protected_names}
    checkpoint_names = {item.name for item in checkpoint.iterdir()}

    def restore_checkpoint() -> dict[str, Any]:
        for name, payload in protected_bytes.items():
            (checkpoint / name).write_bytes(payload)
        removed, unexpected = [], []
        for name in sorted({item.name for item in checkpoint.iterdir()} - checkpoint_names):
            if ".back." in name:
                (checkpoint / name).unlink()
                removed.append(name)
            else:
                unexpected.append(name)
        hashes = {name: file_sha256(checkpoint / name) for name in protected_names}
        expected = {
            name: hashlib.sha256(data).hexdigest() for name, data in protected_bytes.items()
        }
        if hashes != expected or unexpected:
            raise RuntimeError("V2-C recovery checkpoint restoration failed")
        return {"hashes": hashes, "removed_loader_backups": removed, "unexpected": unexpected}

    gpu_before = selected_gpu_snapshot(physical_gpu_id)
    if gpu_before["uuid"] != selected_uuid:
        raise RuntimeError("Selected GPU UUID differs from the recovery pre-run snapshot")
    store = ImmutableRecordStore(project_root / "results")
    manifest = {
        "schema_version": "acr.v2-c-recovery-run.v1",
        "run_id": RUN_ID,
        "parent_run_id": PARENT_RUN_ID,
        "status": "running",
        "started_at_utc": utc_now(),
        "configuration_sha256": file_sha256(project_root / RECOVERY_CONFIG),
        "configuration_semantic_sha256": recovery_config["semantic_sha256"],
        "revisions": {
            "savr": source_revision,
            "openvla_oft": OPENVLA_REVISION,
            "libero": LIBERO_REVISION,
            "checkpoint": CHECKPOINT_REVISION,
        },
        "resources": recovery_config["resource_caps"],
        "selected_gpu": gpu_before,
        "command": "scripts/run_acr_v2_c_recovery.py",
    }
    store.write_once(f"{RUN_ID}/manifest", manifest)
    budget = QueryBudget()
    started = time.monotonic()
    model = action_head = proprio_projector = noisy_action_projector = processor = None
    terminal_error: BaseException | None = None
    result: dict[str, Any] = {
        "schema_version": "acr.v2-c-recovery-result.v1",
        "run_id": RUN_ID,
        "parent_run_id": PARENT_RUN_ID,
        "status": "running",
        "correctness_adjudication": {
            "passed_by_parent_control_flow": True,
            "completed_correctness_queries": 6,
            "parent_failure_sha256": recovery_config["parent_failure_sha256"],
            "parent_revision": recovery_config["parent_revision"],
            "parent_runner_sha256": recovery_config["parent_runner_sha256"],
            "hash_details_persisted": False,
        },
    }
    try:
        os.chdir(upstream_root)
        cfg = upstream_eval.GenerateConfig(
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
        )
        upstream_eval.validate_config(cfg)
        set_seed_everywhere(0)
        load_started = time.monotonic()
        model, action_head, proprio_projector, noisy_action_projector, processor = (
            upstream_eval.initialize_model(cfg)
        )
        torch.cuda.synchronize()
        result["model_load_seconds"] = time.monotonic() - load_started
        if model is None or action_head is None or proprio_projector is None or processor is None:
            raise RuntimeError("Pinned L1/proprio/processor modules were not loaded")
        loaded_model: Any = model
        patch_count = int(loaded_model.vision_backbone.get_num_patches())
        stats = loaded_model.norm_stats[cfg.unnorm_key]["proprio"]
        q01, q99 = (
            np.asarray(stats["q01"], dtype=np.float64),
            np.asarray(stats["q99"], dtype=np.float64),
        )
        if q01.shape != (8,) or q99.shape != (8,) or not np.all(q99 > q01):
            raise RuntimeError("Pinned proprioception statistics are invalid")
        q01_values = tuple(float(value) for value in q01)
        q99_values = tuple(float(value) for value in q99)
        state = (q01 + q99) / 2
        scene, wrist = synthetic_images(np)

        def observation() -> dict[str, Any]:
            return {
                "full_image": scene.copy(),
                "wrist_image": wrist.copy(),
                "state": state.copy(),
            }

        def raw_upstream() -> Any:
            return upstream_eval.get_action(
                cfg,
                loaded_model,
                observation(),
                INSTRUCTION,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                use_film=False,
            )

        tensor_ops = TorchTensorOperations(torch)

        def action_finite(value: Any) -> bool:
            return bool(np.isfinite(np.asarray(value)).all())

        frozen_config = ACRConfiguration(
            "sa-dp-acr-t25-h2-b30-v01",
            ACRPolicy.SA_ACR,
            scene_threshold=0.2476380718954248,
            translation_threshold=0.5479944908411765,
            horizon=2,
            hard_reuse_cap=0.30,
        )
        instruction_sha = hashlib.sha256(INSTRUCTION.encode()).hexdigest()

        def make_context(label: str) -> ACRContext:
            return ACRContext(
                episode_id=f"v2c-recovery-{label}",
                attempt_id=f"{RUN_ID}-{label}",
                task_id="synthetic",
                instruction_sha256=instruction_sha,
                checkpoint_id=CHECKPOINT_REVISION,
                upstream_revision=OPENVLA_REVISION,
                configuration_id=frozen_config.configuration_id,
                controller_version=frozen_config.controller_version,
                preprocessing_id="openvla-center-crop-v1",
                action_head_id="l1-regression-8x7",
                dtype="torch.bfloat16",
                device="cuda:0",
                patch_count=patch_count,
            )

        def prime(controller: Any, cache: Any, context: Any, scene_tokens: Any) -> None:
            scene_rep = prepare_scene_representation(scene)
            position = normalized_eef_position(state, q01_values, q99_values)
            action = tuple(0.0 for _ in range(56))
            for index in range(3):
                decision = controller.decide(
                    scene_representation=scene_rep,
                    normalized_eef_position=position,
                    cache_available=index > 0,
                    cache_age=0,
                )
                if not decision.refresh or decision.query_index != index:
                    raise RuntimeError("Recovery priming differs from the frozen controller")
                controller.observe(
                    decision=decision,
                    scene_representation=scene_rep,
                    normalized_eef_position=position,
                    action_chunk=action,
                )
            cache.store(context=context, tokens=scene_tokens, refresh_query_index=2)

        timing_records: list[dict[str, Any]] = []
        scene_references: list[Any] = []
        expected_counts = recovery_config["corrected_component_counts"]

        def execute(path: str, kind: str, repetition: int, order: int) -> None:
            label = f"recovery-{kind}-{repetition:02d}-{order}-{path}"
            budget.consume(label)
            timer = SynchronizedQueryTimer(TorchCudaEventBackend(torch))
            hooks = ModuleTimingHooks(
                {
                    "siglip": loaded_model.vision_backbone.featurizer,
                    "dinov2": loaded_model.vision_backbone.fused_featurizer,
                    "projector": loaded_model.projector,
                },
                timer,
            )
            adapter_result = None
            try:
                if path == "upstream-fr":
                    timer.start()
                    actions = raw_upstream()
                    timing = timer.finish()
                else:
                    controller = ACRController(frozen_config)
                    observer: Any = None
                    if path == "dual-path-refresh" and not scene_references:

                        def capture_scene(value: Any) -> None:
                            scene_references.append(value[:, :patch_count].detach().clone())

                        observer = capture_scene
                    adapter = DualPathOpenVLAAdapter(
                        model=loaded_model,
                        controller=controller,
                        tensor_ops=tensor_ops,
                        instrumentation=CameraInstrumentation(),
                        correctness_mode=False,
                        action_finite_checker=action_finite,
                        projected_tokens_observer=observer,
                    )
                    context = make_context(f"{kind}-{repetition}-{order}-{path}")
                    with adapter.episode(context):
                        if path == "dual-path-reuse":
                            if not scene_references:
                                raise RuntimeError("Reuse recovery lacks its warm-up scene block")
                            prime(controller, adapter.cache, context, scene_references[0])
                        timer.start()
                        adapter_result = adapter.run_query(
                            query=raw_upstream,
                            scene_image=scene,
                            wrist_image=wrist,
                            state=state,
                            state_q01=q01_values,
                            state_q99=q99_values,
                        )
                        timing = timer.finish()
                    actions = adapter_result.value
            finally:
                hooks.remove()
            action_array = np.asarray(actions)
            if action_array.shape != (8, 7) or not np.isfinite(action_array).all():
                raise RuntimeError(f"Recovery {path} returned invalid actions")
            components = dict(timing.component_device_ms)
            counts = dict(timing.component_counts)
            actual = {name: counts.get(name, 0) for name in ("siglip", "dinov2", "projector")}
            if actual != expected_counts[path]:
                raise RuntimeError(
                    f"Recovery {path} counts differ: expected {expected_counts[path]}, found {actual}"
                )
            if adapter_result is not None:
                expected_refresh = path == "dual-path-refresh"
                adapter_result.work.validate(scene_refresh=expected_refresh)
                if adapter_result.decision.refresh != expected_refresh:
                    raise RuntimeError(f"Recovery {path} selected the wrong execution path")
            timing_records.append(
                {
                    "query_index": PARENT_QUERIES + len(budget.labels) - 1,
                    "label": label,
                    "kind": kind,
                    "repetition": repetition,
                    "order": order,
                    "path": path,
                    "actions_sha256": array_sha256(action_array, np),
                    "timing": {
                        "wall_ms": timing.wall_ms,
                        "total_cuda_ms": timing.total_device_ms,
                        "visual_cuda_ms": sum(
                            components[name] for name in ("siglip", "dinov2", "projector")
                        ),
                        "component_cuda_ms": components,
                        "component_counts": counts,
                    },
                    "work": asdict(adapter_result.work) if adapter_result is not None else None,
                }
            )

        for path in ("upstream-fr", "dual-path-refresh", "dual-path-reuse"):
            for repetition in range(REMAINING_WARMUPS[path]):
                execute(path, "warmup", repetition, 0)
        for repetition in range(12):
            for position, path in enumerate(COUNTERBALANCE[repetition % 3]):
                execute(path, "timed", repetition, position)
        if len(budget.labels) != RECOVERY_QUERY_CAP or len(timing_records) != 41:
            raise RuntimeError("Recovery did not consume exactly the remaining 41 queries")
        if not scene_references:
            raise RuntimeError("Recovery never captured a scene reference")

        timing_summary = summarize_timing(timing_records)
        result["queries"] = timing_records
        result["query_labels"] = list(budget.labels)
        result["recovery_query_count"] = len(budget.labels)
        result["cumulative_query_count"] = PARENT_QUERIES + len(budget.labels)
        result["timing_summary"] = timing_summary
        result["latency_pass"] = bool(timing_summary["gates"]["all_pass"])
        result["status"] = "pass" if result["latency_pass"] else "stopped-negative"
        result["checkpoint_restoration"] = restore_checkpoint()
        if validate_checkpoint(project_root, checkpoint) != checkpoint_before:
            raise RuntimeError("Checkpoint inventory changed during V2-C recovery")
        require_clean_revision(project_root, source_revision)
        require_clean_revision(upstream_root, OPENVLA_REVISION)
        require_clean_revision(libero_root, LIBERO_REVISION)
        result["source_trees_restored"] = True
        result["gpu_before"] = gpu_before
        result["gpu_after"] = selected_gpu_snapshot(physical_gpu_id)
        if result["gpu_after"]["uuid"] != selected_uuid:
            raise RuntimeError("Selected GPU UUID changed during recovery")
        result["wall_seconds"] = time.monotonic() - started
        result["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        result["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        result["finished_at_utc"] = utc_now()
        result["result_semantic_sha256"] = semantic_sha256(result)
        projected_size = directory_size(run_root) + len(canonical_bytes(result)) + 1
        if projected_size > ARTIFACT_CAP_BYTES:
            raise RuntimeError("V2-C recovery artifact cap exceeded")
        store.write_once(f"{RUN_ID}/final", result)
        return 0 if result["status"] == "pass" else 2
    except BaseException as error:
        terminal_error = error
        failure_record = {
            "schema_version": "acr.v2-c-recovery-failure.v1",
            "run_id": RUN_ID,
            "parent_run_id": PARENT_RUN_ID,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "recovery_query_count": len(budget.labels),
            "cumulative_query_count": PARENT_QUERIES + len(budget.labels),
            "query_labels": list(budget.labels),
            "rollout_episodes": 0,
            "simulator_resets": 0,
            "recorded_at_utc": utc_now(),
        }
        try:
            store.write_once(f"{RUN_ID}/failure", failure_record)
        except Exception:
            pass
        raise
    finally:
        signal.alarm(0)
        model = None
        action_head = None
        proprio_projector = None
        noisy_action_projector = None
        processor = None
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            restore_checkpoint()
        except Exception as restoration_error:
            print(f"V2-C recovery restoration error: {restoration_error}", file=sys.stderr)
        if terminal_error is not None:
            print(
                f"V2-C recovery failed after {len(budget.labels)}/41 queries: {terminal_error}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    raise SystemExit(main())
