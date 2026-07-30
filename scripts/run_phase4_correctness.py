#!/usr/bin/env python3
"""Run the approved six-query Phase 4 real-model correctness matrix."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_ROOT = Path("/home/ved/SAVR")
RUN_ID = "phase4-correctness-v1"
CHECKPOINT_REVISION = "638918f3d1c2e43a39a8a20772bdb8b91835e4b7"
CHECKPOINT_RELATIVE = Path("checkpoints/openvla-7b-oft-libero-four-suite")
CHECKPOINT_REPORT = Path("reports/runtime/phase2_checkpoint.json")
UPSTREAM_REVISION = "e4287e94541f459edc4feabc4e181f537cd569a8"
LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
SUITE = "libero_spatial"
TASK_ID = 0
INITIAL_STATE_ID = 0
SEED = 0
QUERY_COUNT = 6
ARTIFACT_CAP_BYTES = 256 * 1024**2


class Interrupted(RuntimeError):
    """Raised when the bounded run receives a termination signal."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def git_output(path: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *arguments],
        text=True,
    ).strip()


def require_clean_revision(path: Path, expected: str | None = None) -> str:
    revision = git_output(path, "rev-parse", "HEAD")
    if expected is not None and revision != expected:
        raise RuntimeError(
            f"Revision mismatch for {path.name}: expected {expected}, found {revision}"
        )
    if git_output(path, "status", "--porcelain"):
        raise RuntimeError(f"Refusing to use dirty source tree: {path}")
    return revision


def validate_checkpoint_inventory(project_root: Path, checkpoint: Path) -> dict[str, Any]:
    report_path = project_root / CHECKPOINT_REPORT
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["requested_revision"] != CHECKPOINT_REVISION:
        raise RuntimeError("Checkpoint report requested revision differs")
    if report["resolved_revision"] != CHECKPOINT_REVISION:
        raise RuntimeError("Checkpoint report resolved revision differs")
    mismatched = []
    for item in report["files"]:
        path = checkpoint / item["path"]
        if not path.is_file() or path.stat().st_size != item["size"]:
            mismatched.append(item["path"])
    if mismatched:
        raise RuntimeError("Checkpoint inventory mismatch: " + ", ".join(mismatched))
    selected_hashes = {
        "config.json": "edd5c5cf6d7927e07465cf086ebe41f7b3ec8f3b128a51f71d6db14dad7ad8b1",
        "dataset_statistics.json": (
            "6ec6ef68d0d5bae4cb5f9fc9acb715a22b9f4545e9e9b300d0d88695cd7afec3"
        ),
        "model.safetensors.index.json": (
            "ca8b53fed8133ee2afcd2fc483de8febf7f5bb0f6bcb09f91189772e59e8f659"
        ),
    }
    actual_hashes = {name: sha256(checkpoint / name) for name in selected_hashes}
    if actual_hashes != selected_hashes:
        raise RuntimeError("Checkpoint metadata hashes differ from accepted evidence")
    return {
        "file_count": len(report["files"]),
        "declared_bytes": report["remote_bytes"],
        "selected_hashes": actual_hashes,
    }


def make_state_b(state_a: Any, statistics: dict[str, Any], np: Any) -> tuple[Any, dict[str, Any]]:
    state_b = np.asarray(state_a, dtype=np.float64).copy()
    lows = np.asarray(statistics["q01"], dtype=np.float64)
    highs = np.asarray(statistics["q99"], dtype=np.float64)
    if state_b.shape != (8,) or lows.shape != (8,) or highs.shape != (8,):
        raise RuntimeError("Expected eight-dimensional proprioception statistics")
    if not (np.isfinite(state_b).all() and np.isfinite(lows).all() and np.isfinite(highs).all()):
        raise RuntimeError("State or proprioception statistics are non-finite")
    if not (highs > lows).all():
        raise RuntimeError("Proprioception q99 must exceed q01")
    dimension = 0
    span = highs[dimension] - lows[dimension]
    upper_target = lows[dimension] + 0.75 * span
    lower_target = lows[dimension] + 0.25 * span
    target = upper_target
    if state_b[dimension] == upper_target:
        target = lower_target
    state_b[dimension] = target
    if not (lows[dimension] <= state_b[dimension] <= highs[dimension]):
        raise RuntimeError("Controlled state B is outside the accepted statistic range")
    if np.array_equal(state_a, state_b):
        raise RuntimeError("Controlled state B did not differ from state A")
    return state_b, {
        "dimension": dimension,
        "state_a_value": float(np.asarray(state_a)[dimension]),
        "state_b_value": float(state_b[dimension]),
        "q01": float(lows[dimension]),
        "q99": float(highs[dimension]),
    }


def exact_parity(reference: Any, candidate: Any, np: Any) -> dict[str, Any]:
    reference_array = np.asarray(reference)
    candidate_array = np.asarray(candidate)
    shapes_equal = reference_array.shape == candidate_array.shape
    array_equal = bool(shapes_equal and np.array_equal(reference_array, candidate_array))
    max_absolute_difference = (
        float(np.max(np.abs(reference_array - candidate_array)))
        if shapes_equal and reference_array.size
        else None
    )
    if not array_equal or max_absolute_difference != 0:
        raise RuntimeError(
            "Exact action parity failed: "
            f"shapes_equal={shapes_equal}, array_equal={array_equal}, "
            f"max_absolute_difference={max_absolute_difference}"
        )
    return {
        "reference_shape": list(reference_array.shape),
        "candidate_shape": list(candidate_array.shape),
        "array_equal": array_equal,
        "max_absolute_difference": max_absolute_difference,
    }


def validate_schemas(project_root: Path, validator: Any) -> dict[str, Any]:
    schemas = {}
    for name in (
        "episode_result.schema.json",
        "query_record.schema.json",
        "run_manifest.schema.json",
    ):
        schema = json.loads((project_root / "schemas" / name).read_text(encoding="utf-8"))
        validator.check_schema(schema)
        schemas[name] = schema
    synthetic_episode = {
        "run_id": RUN_ID,
        "episode_id": "schema-audit-only",
        "task": "none",
        "initial_state_id": 0,
        "seed": SEED,
        "status": "completed",
        "success": False,
        "steps": 0,
        "refresh_count": 0,
        "skipped_refresh_count": 0,
        "refresh_rate": 0,
        "latency_ms": {
            "total_episode": 0,
            "policy_median": 0,
            "policy_p95": 0,
        },
    }
    validator(synthetic_episode, schemas["episode_result.schema.json"])
    return schemas


def ensure_artifact_cap(run_dir: Path) -> None:
    size = directory_size(run_dir)
    if size > ARTIFACT_CAP_BYTES:
        raise RuntimeError(f"Phase 4 artifact cap exceeded: {size} bytes")


def query_record(
    *,
    index: int,
    path: str,
    refresh: bool,
    cache_event: str,
    actions: Any,
    timing: Any,
    decision_wall_ms: float,
    np: Any,
    extra: dict[str, Any],
) -> dict[str, Any]:
    action_array = np.asarray(actions)
    if action_array.shape != (8, 7):
        raise RuntimeError(f"Unexpected action shape for query {index}: {action_array.shape}")
    if not np.isfinite(action_array).all():
        raise RuntimeError(f"Non-finite action from query {index}")
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
        "warmup_excluded": False,
        "timing": {
            "decision_wall_ms": decision_wall_ms,
            "query_wall_ms": timing.wall_ms,
            "total_cuda_ms": timing.total_device_ms,
            "component_cuda_ms": component_ms,
            "component_counts": component_counts,
        },
        **extra,
    }


def terminal_placeholders(
    *,
    store: Any,
    query_schema: dict[str, Any],
    validator: Any,
    status: str,
    error: BaseException,
) -> None:
    existing = store.completed_query_indices()
    missing = [index for index in range(1, QUERY_COUNT + 1) if index not in existing]
    for position, index in enumerate(missing):
        record_status = status if status == "interrupted" else (
            "failed" if position == 0 else "interrupted"
        )
        record = {
            "run_id": RUN_ID,
            "query_index": index,
            "environment_step": 0,
            "status": record_status,
            "path": "not_executed",
            "refresh": True,
            "cache_event": "unmodified",
            "action_shape": [1],
            "timing": {
                "decision_wall_ms": 0,
                "query_wall_ms": 0,
                "total_cuda_ms": 0,
                "component_cuda_ms": {},
                "component_counts": {},
            },
            "error_type": type(error).__name__,
            "error": str(error),
        }
        validator(record, query_schema)
        store.write_query(index, record)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    if project_root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing to run outside {EXPECTED_ROOT}: {project_root}")

    physical_gpu_id = os.environ.get("SAVR_PHYSICAL_GPU_ID")
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not physical_gpu_id or visible_gpu != physical_gpu_id:
        raise SystemExit("SAVR_PHYSICAL_GPU_ID must exactly match CUDA_VISIBLE_DEVICES")

    def handle_signal(_signum: int, _frame: Any) -> None:
        raise Interrupted("Phase 4 runner received a termination signal")

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

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
    from savr.controllers import FullRefreshController, VisualOnlyRefreshController
    from savr.integration.openvla_oft import OpenVLAProjectedFeatureAdapter
    from savr.logging import ImmutableRecordStore
    from savr.timing import (
        ModuleTimingHooks,
        SynchronizedQueryTimer,
        TorchCudaEventBackend,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one selected CUDA device must be visible")

    schemas = validate_schemas(project_root, validate)
    Draft202012Validator.check_schema(schemas["query_record.schema.json"])
    run_dir.mkdir(parents=True, exist_ok=False)
    store = ImmutableRecordStore(run_dir)
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "run_summary.json"
    selected_uuid = os.environ.get("SAVR_SELECTED_GPU_UUID")
    if not selected_uuid:
        raise RuntimeError("SAVR_SELECTED_GPU_UUID is required")

    protected_names = (
        "config.json",
        "configuration_prismatic.py",
        "modeling_prismatic.py",
    )
    protected_bytes = {name: (checkpoint / name).read_bytes() for name in protected_names}
    protected_hashes = {name: sha256(checkpoint / name) for name in protected_names}
    checkpoint_files_before = {item.name for item in checkpoint.iterdir()}

    manifest: dict[str, Any] = {
        "run_id": RUN_ID,
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "status": "running",
        "policy": "FR",
        "savr_git_revision": project_revision,
        "working_tree_clean": True,
        "base_model": {
            "name": "OpenVLA-OFT",
            "checkpoint": str(CHECKPOINT_RELATIVE),
            "revision": CHECKPOINT_REVISION,
        },
        "benchmark": {
            "name": "LIBERO",
            "revision": libero_revision,
            "suite": SUITE,
        },
        "hardware": {
            "physical_gpu_id": physical_gpu_id,
            "selected_gpu_uuid": selected_uuid,
            "cuda_visible_devices": visible_gpu,
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
            "real_model_query_cap": QUERY_COUNT,
            "simulator_resets": 1,
            "rollout_episodes": 0,
            "num_open_loop_steps": 8,
            "num_images_in_input": 2,
            "use_proprio": True,
            "use_l1_regression": True,
            "use_diffusion": False,
            "use_film": False,
            "center_crop": True,
            "artifact_cap_bytes": ARTIFACT_CAP_BYTES,
            "checkpoint_inventory": checkpoint_inventory,
        },
        "command": "scripts/run_phase4_correctness.py",
        "notes": "Correctness-only mixed upstream/FR/VOR matrix; no rollout episode.",
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
    summary: dict[str, Any] = {
        "run_id": RUN_ID,
        "query_count_cap": QUERY_COUNT,
        "simulator_reset_count": 0,
        "rollout_episode_count": 0,
        "checkpoint_hashes_before": protected_hashes,
    }
    terminal_status = "failed"
    caught_error: BaseException | None = None
    exit_code = 1
    try:
        os.chdir(upstream_root)
        load_started = time.perf_counter()
        model, action_head, proprio_projector, noisy_action_projector, processor = (
            upstream_eval.initialize_model(cfg)
        )
        torch.cuda.synchronize()
        summary["model_load_seconds"] = time.perf_counter() - load_started
        if action_head is None or proprio_projector is None:
            raise RuntimeError("Pinned L1/proprio modules were not loaded")

        resize_size = get_image_resize_size(cfg)
        task_suite = benchmark.get_benchmark_dict()[SUITE]()
        task = task_suite.get_task(TASK_ID)
        initial_states = task_suite.get_task_init_states(TASK_ID)
        current_env, task_description = get_libero_env(
            task,
            cfg.model_family,
            resolution=cfg.env_img_res,
        )
        current_env.reset()
        summary["simulator_reset_count"] = 1
        raw_observation = current_env.set_init_state(initial_states[INITIAL_STATE_ID])
        policy_observation, _ = upstream_eval.prepare_observation(raw_observation, resize_size)
        current_env.close()
        current_env = None

        state_a = np.asarray(policy_observation["state"], dtype=np.float64).copy()
        proprio_stats = model.norm_stats[cfg.unnorm_key]["proprio"]
        state_b, perturbation = make_state_b(state_a, proprio_stats, np)
        images = {
            "full_image": np.asarray(policy_observation["full_image"]).copy(),
            "wrist_image": np.asarray(policy_observation["wrist_image"]).copy(),
        }
        summary["controlled_state_perturbation"] = perturbation

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

        fr_adapter = OpenVLAProjectedFeatureAdapter(
            model=model,
            controller=FullRefreshController(),
        )
        fr_adapter.begin_context(
            CacheContext(
                episode_id="phase4-fr",
                task_id=str(TASK_ID),
                checkpoint_id=CHECKPOINT_REVISION,
                configuration_id="phase4-fr-l1-two-image",
            )
        )
        vor_adapter = OpenVLAProjectedFeatureAdapter(
            model=model,
            controller=VisualOnlyRefreshController(
                image_threshold=1.0,
                max_reuse_horizon=4,
            ),
        )
        vor_adapter.begin_context(
            CacheContext(
                episode_id="phase4-vor",
                task_id=str(TASK_ID),
                checkpoint_id=CHECKPOINT_REVISION,
                configuration_id="phase4-vor-l1-two-image",
            )
        )

        records: list[dict[str, Any]] = []

        timer.start()
        actions_1 = upstream_query(state_a)
        timing_1 = timer.finish()
        record_1 = query_record(
            index=1,
            path="unmodified_upstream_state_a",
            refresh=True,
            cache_event="unmodified",
            actions=actions_1,
            timing=timing_1,
            decision_wall_ms=0,
            np=np,
            extra={},
        )
        if record_1["timing"]["component_counts"]["vision_backbone"] != 1:
            raise RuntimeError("Query 1 did not execute the vision backbone exactly once")
        validate(record_1, schemas["query_record.schema.json"])
        store.write_query(1, record_1)
        records.append(record_1)
        ensure_artifact_cap(run_dir)

        timer.start()
        result_2 = fr_adapter.run_query(
            query=lambda: upstream_query(state_a),
            images=images,
            state=state_a,
            environment_step=0,
        )
        timing_2 = timer.finish()
        parity_2 = exact_parity(actions_1, result_2.value, np)
        record_2 = query_record(
            index=2,
            path="wrapped_fr_state_a",
            refresh=result_2.decision.refresh,
            cache_event=result_2.cache_event,
            actions=result_2.value,
            timing=timing_2,
            decision_wall_ms=result_2.decision_seconds * 1000,
            np=np,
            extra={"decision": asdict(result_2.decision), "parity": parity_2},
        )
        if not result_2.decision.refresh:
            raise RuntimeError("Wrapped FR query 2 did not refresh")
        if record_2["timing"]["component_counts"]["vision_backbone"] != 1:
            raise RuntimeError("Wrapped FR query 2 visual count differs from one")
        validate(record_2, schemas["query_record.schema.json"])
        store.write_query(2, record_2)
        records.append(record_2)
        ensure_artifact_cap(run_dir)

        timer.start()
        result_3 = fr_adapter.run_query(
            query=lambda: upstream_query(state_a),
            images=images,
            state=state_a,
            environment_step=0,
        )
        timing_3 = timer.finish()
        parity_3 = exact_parity(actions_1, result_3.value, np)
        record_3 = query_record(
            index=3,
            path="wrapped_fr_second_state_a",
            refresh=result_3.decision.refresh,
            cache_event=result_3.cache_event,
            actions=result_3.value,
            timing=timing_3,
            decision_wall_ms=result_3.decision_seconds * 1000,
            np=np,
            extra={"decision": asdict(result_3.decision), "parity": parity_3},
        )
        if not result_3.decision.refresh:
            raise RuntimeError("Wrapped FR query 3 did not refresh")
        if record_3["timing"]["component_counts"]["vision_backbone"] != 1:
            raise RuntimeError("Wrapped FR query 3 visual count differs from one")
        validate(record_3, schemas["query_record.schema.json"])
        store.write_query(3, record_3)
        records.append(record_3)
        ensure_artifact_cap(run_dir)

        timer.start()
        actions_4 = upstream_query(state_b)
        timing_4 = timer.finish()
        record_4 = query_record(
            index=4,
            path="unmodified_upstream_state_b",
            refresh=True,
            cache_event="unmodified",
            actions=actions_4,
            timing=timing_4,
            decision_wall_ms=0,
            np=np,
            extra={},
        )
        validate(record_4, schemas["query_record.schema.json"])
        store.write_query(4, record_4)
        records.append(record_4)
        ensure_artifact_cap(run_dir)

        timer.start()
        result_5 = vor_adapter.run_query(
            query=lambda: upstream_query(state_a),
            images=images,
            state=state_a,
            environment_step=0,
        )
        timing_5 = timer.finish()
        cache_entry = vor_adapter.cache.entry
        if cache_entry is None or vor_adapter.cache.age != 0:
            raise RuntimeError("VOR refresh did not produce an age-zero cache entry")
        cache_metadata = asdict(cache_entry.metadata)
        expected_shape = [
            1,
            model.vision_backbone.get_num_patches()
            * model.vision_backbone.get_num_images_in_input(),
            model.llm_dim,
        ]
        if list(cache_metadata["shape"]) != expected_shape:
            raise RuntimeError(
                f"Cached projected-feature shape differs: {cache_metadata['shape']}"
            )
        record_5 = query_record(
            index=5,
            path="wrapped_vor_refresh_state_a",
            refresh=result_5.decision.refresh,
            cache_event=result_5.cache_event,
            actions=result_5.value,
            timing=timing_5,
            decision_wall_ms=result_5.decision_seconds * 1000,
            np=np,
            extra={
                "decision": asdict(result_5.decision),
                "cache_age_after": vor_adapter.cache.age,
                "cache_metadata": cache_metadata,
            },
        )
        if not result_5.decision.refresh or result_5.cache_event != "refresh":
            raise RuntimeError("Query 5 did not perform the required VOR refresh")
        validate(record_5, schemas["query_record.schema.json"])
        store.write_query(5, record_5)
        records.append(record_5)
        ensure_artifact_cap(run_dir)

        proprio_before = len(proprio_inputs)
        timer.start()
        result_6 = vor_adapter.run_query(
            query=lambda: upstream_query(state_b),
            images=images,
            state=state_b,
            environment_step=0,
        )
        timing_6 = timer.finish()
        parity_6 = exact_parity(actions_4, result_6.value, np)
        if result_6.decision.refresh or result_6.cache_event != "reuse":
            raise RuntimeError("Query 6 did not take the required VOR reuse path")
        if vor_adapter.cache.age != 1:
            raise RuntimeError("VOR cache age did not change from zero to one")
        if len(proprio_inputs) != proprio_before + 1:
            raise RuntimeError("Reuse query did not execute proprioception exactly once")
        expected_normalized_b = normalize_proprio(
            state_b,
            model.norm_stats[cfg.unnorm_key]["proprio"],
        )
        actual_proprio_b = proprio_inputs[-1].float().numpy().reshape(-1)
        expected_proprio_tensor = torch.tensor(
            expected_normalized_b,
            dtype=proprio_inputs[-1].dtype,
        )
        expected_proprio_b = expected_proprio_tensor.float().numpy().reshape(-1)
        if not np.array_equal(actual_proprio_b, expected_proprio_b):
            raise RuntimeError("Reuse query did not receive normalized current state B")
        counts_6 = timing_6.component_counts
        if counts_6.get("vision_backbone", 0) != 0:
            raise RuntimeError("Reuse query executed the vision backbone")
        if counts_6.get("visual_projector", 0) != 0:
            raise RuntimeError("Reuse query executed the visual projector")
        if counts_6.get("language_model", 0) != 1:
            raise RuntimeError("Reuse query did not execute the language model once")
        if counts_6.get("action_head", 0) != 1:
            raise RuntimeError("Reuse query did not execute the action head once")
        refreshed_entry = vor_adapter.cache.entry
        if refreshed_entry is None or asdict(refreshed_entry.metadata) != cache_metadata:
            raise RuntimeError("Cached tensor metadata changed during reuse")
        record_6 = query_record(
            index=6,
            path="wrapped_vor_reuse_state_b",
            refresh=result_6.decision.refresh,
            cache_event=result_6.cache_event,
            actions=result_6.value,
            timing=timing_6,
            decision_wall_ms=result_6.decision_seconds * 1000,
            np=np,
            extra={
                "decision": asdict(result_6.decision),
                "parity": parity_6,
                "cache_age_before": 0,
                "cache_age_after": vor_adapter.cache.age,
                "cache_metadata": cache_metadata,
                "fresh_proprio_array_equal": True,
                "normalized_state_b": expected_proprio_b.tolist(),
            },
        )
        validate(record_6, schemas["query_record.schema.json"])
        store.write_query(6, record_6)
        records.append(record_6)
        ensure_artifact_cap(run_dir)

        if store.completed_query_indices() != set(range(1, QUERY_COUNT + 1)):
            raise RuntimeError("Phase 4 terminal query-record set is incomplete")
        summary.update(
            {
                "status": "completed",
                "query_count": len(records),
                "exact_fr_parity": True,
                "exact_reuse_parity": True,
                "reuse_visual_call_count": 0,
                "fresh_proprio_on_reuse": True,
                "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )
        terminal_status = "completed"
        exit_code = 0
    except BaseException as error:
        caught_error = error
        terminal_status = "interrupted" if isinstance(error, Interrupted) else "failed"
        summary.update(
            {
                "status": terminal_status,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
    finally:
        if current_env is not None:
            current_env.close()
        if proprio_handle is not None:
            proprio_handle.remove()
        if timing_hooks is not None:
            timing_hooks.remove()
        for name, content in protected_bytes.items():
            (checkpoint / name).write_bytes(content)
        files_after_restore = {item.name for item in checkpoint.iterdir()}
        new_files = sorted(files_after_restore - checkpoint_files_before)
        removed_backups = []
        unexpected_new_files = []
        for name in new_files:
            if ".back." in name:
                (checkpoint / name).unlink()
                removed_backups.append(name)
            else:
                unexpected_new_files.append(name)
        hashes_after = {name: sha256(checkpoint / name) for name in protected_names}
        checkpoint_restored = hashes_after == protected_hashes
        summary.update(
            {
                "removed_upstream_backup_files": removed_backups,
                "unexpected_new_checkpoint_files": unexpected_new_files,
                "checkpoint_hashes_after_restore": hashes_after,
                "checkpoint_restored": checkpoint_restored,
                "finished_at_utc": utc_now(),
            }
        )
        if not checkpoint_restored or unexpected_new_files:
            if caught_error is None:
                caught_error = RuntimeError("Checkpoint restoration/inventory audit failed")
            terminal_status = "failed"
            exit_code = 1
            summary["status"] = "failed"
            summary["error_type"] = type(caught_error).__name__
            summary["error"] = str(caught_error)
        if caught_error is not None:
            terminal_placeholders(
                store=store,
                query_schema=schemas["query_record.schema.json"],
                validator=validate,
                status=terminal_status,
                error=caught_error,
            )
        manifest["finished_at_utc"] = summary["finished_at_utc"]
        manifest["status"] = terminal_status
        manifest["hardware"].update(
            {
                "visible_gpu_name": torch.cuda.get_device_name(0),
                "visible_gpu_capability": list(torch.cuda.get_device_capability(0)),
                "peak_memory_allocated_bytes": summary.get(
                    "peak_gpu_memory_allocated_bytes"
                ),
                "peak_memory_reserved_bytes": summary.get(
                    "peak_gpu_memory_reserved_bytes"
                ),
            }
        )
        validate(manifest, schemas["run_manifest.schema.json"])
        atomic_json(summary_path, summary)
        atomic_json(manifest_path, manifest)
        event_status = terminal_status.upper()
        store.write_run_event(
            1,
            event_status,
            {
                "finished_at_utc": summary["finished_at_utc"],
                "checkpoint_restored": checkpoint_restored,
                "query_record_count": len(store.completed_query_indices()),
            },
        )
        ensure_artifact_cap(run_dir)
        if model is not None:
            del model
        torch.cuda.empty_cache()

    print(json.dumps(summary, indent=2, sort_keys=True))
    if caught_error is not None:
        print(
            f"Phase 4 stopped: {type(caught_error).__name__}: {caught_error}",
            file=sys.stderr,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
