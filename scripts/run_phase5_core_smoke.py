#!/usr/bin/env python3
"""Run the approved bounded Phase 5 four-policy trajectory smoke matrix."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter, deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_ROOT = Path("/home/ved/SAVR")
RUN_ID = "phase5-core-smoke-v1"
CHECKPOINT_REVISION = "638918f3d1c2e43a39a8a20772bdb8b91835e4b7"
CHECKPOINT_RELATIVE = Path("checkpoints/openvla-7b-oft-libero-four-suite")
CHECKPOINT_REPORT = Path("reports/runtime/phase2_checkpoint.json")
UPSTREAM_REVISION = "e4287e94541f459edc4feabc4e181f537cd569a8"
LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
SUITE = "libero_spatial"
TASK_ID = 0
SEED = 0
PERIOD = 2
DIAGNOSTIC_THRESHOLD = 1_000_000.0
MAX_REUSE_HORIZON = 2
ARTIFACT_CAP_BYTES = 1024**3
WALL_CAP_SECONDS = 2 * 60 * 60
SCHEDULE = (
    (0, "FR"),
    (1, "PR"),
    (2, "VOR"),
    (0, "SAVR"),
    (1, "FR"),
    (2, "PR"),
    (0, "VOR"),
    (1, "SAVR"),
    (2, "FR"),
    (0, "PR"),
    (1, "VOR"),
    (2, "SAVR"),
)


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


def validate_checkpoint_inventory(
    project_root: Path,
    checkpoint: Path,
) -> dict[str, Any]:
    report = json.loads(
        (project_root / CHECKPOINT_REPORT).read_text(encoding="utf-8")
    )
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


def validate_schemas(
    project_root: Path,
    validator_class: Any,
) -> dict[str, Any]:
    schemas = {}
    for name in (
        "episode_result.schema.json",
        "query_record.schema.json",
        "run_manifest.schema.json",
    ):
        schema = json.loads(
            (project_root / "schemas" / name).read_text(encoding="utf-8")
        )
        validator_class.check_schema(schema)
        schemas[name] = schema
    return schemas


def ensure_bounds(run_dir: Path, run_started: float) -> None:
    size = directory_size(run_dir)
    if size > ARTIFACT_CAP_BYTES:
        raise RuntimeError(f"Phase 5 artifact cap exceeded: {size} bytes")
    elapsed = time.monotonic() - run_started
    if elapsed > WALL_CAP_SECONDS:
        raise RuntimeError(f"Phase 5 wall-clock cap exceeded: {elapsed:.3f} seconds")


def percentile(values: list[float], quantile: float, np: Any) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def raw_robot_state(observation: dict[str, Any], np: Any, quat2axisangle: Any) -> Any:
    state = np.concatenate(
        (
            observation["robot0_eef_pos"],
            quat2axisangle(observation["robot0_eef_quat"]),
            observation["robot0_gripper_qpos"],
        )
    )
    if state.shape != (8,) or not np.isfinite(state).all():
        raise RuntimeError("Trajectory state must be a finite eight-dimensional vector")
    return state


def controller_for_policy(
    policy: str,
    state_statistics: dict[str, Any],
    action_statistics: dict[str, Any],
    controllers: Any,
) -> Any:
    if policy == "FR":
        return controllers.FullRefreshController()
    if policy == "PR":
        return controllers.PeriodicRefreshController(period=PERIOD)
    if policy == "VOR":
        return controllers.VisualOnlyRefreshController(
            image_threshold=DIAGNOSTIC_THRESHOLD,
            max_reuse_horizon=MAX_REUSE_HORIZON,
        )
    if policy == "SAVR":
        return controllers.StateAwareVisualRefreshController(
            image_threshold=DIAGNOSTIC_THRESHOLD,
            state_threshold=DIAGNOSTIC_THRESHOLD,
            action_threshold=DIAGNOSTIC_THRESHOLD,
            max_reuse_horizon=MAX_REUSE_HORIZON,
            state_q01=state_statistics["q01"],
            state_q99=state_statistics["q99"],
            action_q01=action_statistics["q01"],
            action_q99=action_statistics["q99"],
        )
    raise ValueError(f"Unsupported policy: {policy}")


def expected_refresh(
    *,
    policy: str,
    episode_query_index: int,
    cache_age_before: int,
) -> bool:
    if policy == "FR":
        return True
    if episode_query_index == 0:
        return True
    if policy == "PR":
        return episode_query_index % PERIOD == 0
    if policy == "VOR":
        return cache_age_before >= MAX_REUSE_HORIZON
    if policy == "SAVR":
        return episode_query_index < 2 or cache_age_before >= MAX_REUSE_HORIZON
    raise ValueError(f"Unsupported policy: {policy}")


def assert_query_invariants(
    *,
    policy: str,
    episode_query_index: int,
    result: Any,
    timing: Any,
) -> None:
    decision = result.decision
    if decision.query_index != episode_query_index:
        raise RuntimeError("Controller query index is not episode-contiguous")
    if decision.refresh != expected_refresh(
        policy=policy,
        episode_query_index=episode_query_index,
        cache_age_before=decision.cache_age_before,
    ):
        raise RuntimeError(
            f"Unexpected {policy} refresh trajectory at query {episode_query_index}: "
            f"{asdict(decision)}"
        )

    counts = dict(timing.component_counts)
    for name in ("vision_backbone", "visual_projector", "language_model", "action_head"):
        counts.setdefault(name, 0)
    visual_expected = 1 if decision.refresh else 0
    if counts["vision_backbone"] != visual_expected:
        raise RuntimeError(
            f"Vision count differs for {policy} query {episode_query_index}: {counts}"
        )
    if counts["visual_projector"] != visual_expected:
        raise RuntimeError(
            f"Projector count differs for {policy} query {episode_query_index}: {counts}"
        )
    if counts["language_model"] != 1 or counts["action_head"] != 1:
        raise RuntimeError(
            f"Downstream count differs for {policy} query {episode_query_index}: "
            f"{counts}"
        )
    if result.cache_event != ("refresh" if decision.refresh else "reuse"):
        raise RuntimeError("Cache event differs from the effective refresh decision")


def make_query_record(
    *,
    run_id: str,
    global_query_index: int,
    episode_id: str,
    policy: str,
    environment_step: int,
    actions: Any,
    result: Any,
    timing: Any,
    cache_age_after: int,
    np: Any,
) -> dict[str, Any]:
    action_array = np.asarray(actions)
    if action_array.shape != (8, 7) or not np.isfinite(action_array).all():
        raise RuntimeError(
            f"Query action must be finite with shape (8, 7), found {action_array.shape}"
        )
    component_ms = dict(timing.component_device_ms)
    component_counts = dict(timing.component_counts)
    for name in ("vision_backbone", "visual_projector", "language_model", "action_head"):
        component_ms.setdefault(name, 0.0)
        component_counts.setdefault(name, 0)
    return {
        "run_id": run_id,
        "query_index": global_query_index,
        "environment_step": environment_step,
        "status": "completed",
        "path": f"wrapped_{policy.lower()}",
        "refresh": result.decision.refresh,
        "cache_event": result.cache_event,
        "action_shape": list(action_array.shape),
        "actions_sha256": hashlib.sha256(action_array.tobytes()).hexdigest(),
        "episode_id": episode_id,
        "policy": policy,
        "episode_query_index": result.decision.query_index,
        "decision": asdict(result.decision),
        "cache_age_before": result.decision.cache_age_before,
        "cache_age_after": cache_age_after,
        "timing": {
            "decision_wall_ms": result.decision_seconds * 1000,
            "query_wall_ms": timing.wall_ms,
            "total_cuda_ms": timing.total_device_ms,
            "component_cuda_ms": component_ms,
            "component_counts": component_counts,
        },
    }


def validate_complete_matrix(
    episode_records: list[dict[str, Any]],
    query_record_count: int,
) -> dict[str, Any]:
    planned = {(state_id, policy) for state_id, policy in SCHEDULE}
    observed = {
        (int(record["initial_state_id"]), str(record["policy"]))
        for record in episode_records
    }
    if observed != planned or len(episode_records) != len(SCHEDULE):
        raise RuntimeError(
            f"Episode matrix mismatch: missing={sorted(planned - observed)}, "
            f"extra={sorted(observed - planned)}"
        )
    if any(record["status"] != "completed" for record in episode_records):
        raise RuntimeError("A core smoke episode lacks completed terminal status")
    summed_queries = sum(int(record["query_count"]) for record in episode_records)
    if summed_queries != query_record_count:
        raise RuntimeError(
            f"Query record mismatch: episodes={summed_queries}, files={query_record_count}"
        )
    if any(
        record["refresh_count"] + record["reuse_count"] != record["query_count"]
        for record in episode_records
    ):
        raise RuntimeError("Episode refresh/reuse counts do not reconcile")
    return {
        "planned_episode_count": len(SCHEDULE),
        "terminal_episode_count": len(episode_records),
        "query_record_count": query_record_count,
        "success_count": sum(bool(record["success"]) for record in episode_records),
        "policy_successes": {
            policy: sum(
                bool(record["success"])
                for record in episode_records
                if record["policy"] == policy
            )
            for policy in ("FR", "PR", "VOR", "SAVR")
        },
        "policy_refresh_counts": {
            policy: sum(
                int(record["refresh_count"])
                for record in episode_records
                if record["policy"] == policy
            )
            for policy in ("FR", "PR", "VOR", "SAVR")
        },
        "policy_reuse_counts": {
            policy: sum(
                int(record["reuse_count"])
                for record in episode_records
                if record["policy"] == policy
            )
            for policy in ("FR", "PR", "VOR", "SAVR")
        },
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    if project_root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing to run outside {EXPECTED_ROOT}: {project_root}")

    physical_gpu_id = os.environ.get("SAVR_PHYSICAL_GPU_ID")
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    selected_uuid = os.environ.get("SAVR_SELECTED_GPU_UUID")
    if not physical_gpu_id or visible_gpu != physical_gpu_id:
        raise SystemExit("SAVR_PHYSICAL_GPU_ID must exactly match CUDA_VISIBLE_DEVICES")
    if not selected_uuid:
        raise SystemExit("SAVR_SELECTED_GPU_UUID is required")

    def handle_signal(_signum: int, _frame: Any) -> None:
        raise Interrupted("Phase 5 runner received a termination signal")

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
        get_libero_dummy_action,
        get_libero_env,
        quat2axisangle,
    )
    from experiments.robot.robot_utils import (  # type: ignore[import-not-found]
        get_image_resize_size,
        set_seed_everywhere,
    )
    from savr import controllers
    from savr.cache import CacheContext
    from savr.integration.openvla_oft import OpenVLAProjectedFeatureAdapter
    from savr.logging import ImmutableRecordStore
    from savr.timing import (
        ModuleTimingHooks,
        SynchronizedQueryTimer,
        TorchCudaEventBackend,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one selected CUDA device must be visible")

    schemas = validate_schemas(project_root, Draft202012Validator)
    run_dir.mkdir(parents=True, exist_ok=False)
    store = ImmutableRecordStore(run_dir)
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "run_summary.json"
    run_started = time.monotonic()

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
        "policy": "MIXED",
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
            "schedule": [
                {"initial_state_id": state_id, "policy": policy}
                for state_id, policy in SCHEDULE
            ],
            "seed": SEED,
            "period": PERIOD,
            "diagnostic_threshold": DIAGNOSTIC_THRESHOLD,
            "max_reuse_horizon": MAX_REUSE_HORIZON,
            "num_open_loop_steps": 8,
            "num_images_in_input": 2,
            "use_proprio": True,
            "use_l1_regression": True,
            "use_diffusion": False,
            "use_film": False,
            "center_crop": True,
            "episode_cap": len(SCHEDULE),
            "artifact_cap_bytes": ARTIFACT_CAP_BYTES,
            "wall_cap_seconds": WALL_CAP_SECONDS,
            "checkpoint_inventory": checkpoint_inventory,
        },
        "command": "scripts/run_phase5_core_smoke.py",
        "notes": "Diagnostic structural smoke only; thresholds are not calibrated.",
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
    timing_hooks = current_env = None
    terminal_status = "failed"
    caught_error: BaseException | None = None
    exit_code = 1
    summary: dict[str, Any] = {
        "run_id": RUN_ID,
        "status": "running",
        "episode_cap": len(SCHEDULE),
        "checkpoint_hashes_before": protected_hashes,
    }
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

        state_statistics = model.norm_stats[cfg.unnorm_key]["proprio"]
        action_statistics = model.norm_stats[cfg.unnorm_key]["action"]
        controller_for_policy(
            "SAVR",
            state_statistics,
            action_statistics,
            controllers,
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
        resize_size = get_image_resize_size(cfg)
        task_suite = benchmark.get_benchmark_dict()[SUITE]()
        task = task_suite.get_task(TASK_ID)
        initial_states = task_suite.get_task_init_states(TASK_ID)
        current_env, task_description = get_libero_env(
            task,
            cfg.model_family,
            resolution=cfg.env_img_res,
        )

        global_query_index = 0
        for order_index, (state_id, policy) in enumerate(SCHEDULE):
            ensure_bounds(run_dir, run_started)
            episode_id = (
                f"order_{order_index:02d}_task_{TASK_ID:02d}_"
                f"state_{state_id:02d}_{policy.lower()}"
            )
            episode_started_utc = utc_now()
            episode_started = time.perf_counter()
            controller = controller_for_policy(
                policy,
                state_statistics,
                action_statistics,
                controllers,
            )
            adapter = OpenVLAProjectedFeatureAdapter(
                model=model,
                controller=controller,
            )
            adapter.begin_context(
                CacheContext(
                    episode_id=episode_id,
                    task_id=str(TASK_ID),
                    checkpoint_id=CHECKPOINT_REVISION,
                    configuration_id=(
                        f"phase5-{policy.lower()}-diagnostic-"
                        f"state-{state_id}"
                    ),
                )
            )
            current_env.reset()
            observation = current_env.set_init_state(initial_states[state_id])
            action_queue: deque[Any] = deque(maxlen=cfg.num_open_loop_steps)
            query_wall_ms: list[float] = []
            control_wall_ms: list[float] = []
            trigger_counts: Counter[str] = Counter()
            refresh_count = 0
            reuse_count = 0
            episode_query_index = 0
            environment_step = 0
            control_steps = 0
            success = False
            trajectory_digest = hashlib.sha256()
            episode_error: BaseException | None = None

            try:
                max_steps = upstream_eval.TASK_MAX_STEPS[cfg.task_suite_name]
                while environment_step < max_steps + cfg.num_steps_wait:
                    ensure_bounds(run_dir, run_started)
                    if environment_step < cfg.num_steps_wait:
                        observation, _, _, _ = current_env.step(
                            get_libero_dummy_action(cfg.model_family)
                        )
                        environment_step += 1
                        continue

                    if not action_queue:
                        policy_observation, _ = upstream_eval.prepare_observation(
                            observation,
                            resize_size,
                        )
                        images = {
                            "full_image": policy_observation["full_image"],
                            "wrist_image": policy_observation["wrist_image"],
                        }

                        def query() -> Any:
                            return upstream_eval.get_action(
                                cfg,
                                model,
                                policy_observation,
                                task_description,
                                processor=processor,
                                action_head=action_head,
                                proprio_projector=proprio_projector,
                                noisy_action_projector=noisy_action_projector,
                                use_film=cfg.use_film,
                            )

                        timer.start()
                        result = adapter.run_query(
                            query=query,
                            images=images,
                            state=policy_observation["state"],
                            environment_step=environment_step,
                        )
                        timing = timer.finish()
                        actions = np.asarray(result.value)
                        if actions.shape != (8, 7) or not np.isfinite(actions).all():
                            raise RuntimeError(
                                f"Non-finite or malformed action chunk: {actions.shape}"
                            )
                        assert_query_invariants(
                            policy=policy,
                            episode_query_index=episode_query_index,
                            result=result,
                            timing=timing,
                        )
                        query_record = make_query_record(
                            run_id=RUN_ID,
                            global_query_index=global_query_index,
                            episode_id=episode_id,
                            policy=policy,
                            environment_step=environment_step,
                            actions=actions,
                            result=result,
                            timing=timing,
                            cache_age_after=adapter.cache.age,
                            np=np,
                        )
                        validate(query_record, schemas["query_record.schema.json"])
                        store.write_query(global_query_index, query_record)
                        global_query_index += 1
                        episode_query_index += 1
                        query_wall_ms.append(timing.wall_ms)
                        trigger_counts.update(result.decision.triggers)
                        if result.decision.refresh:
                            refresh_count += 1
                        else:
                            reuse_count += 1
                        action_queue.extend(actions)

                    control_started = time.perf_counter()
                    action = upstream_eval.process_action(
                        action_queue.popleft(),
                        cfg.model_family,
                    )
                    if not np.isfinite(action).all():
                        raise RuntimeError("Processed control action is non-finite")
                    observation, _, done, _ = current_env.step(action.tolist())
                    control_wall_ms.append(
                        (time.perf_counter() - control_started) * 1000
                    )
                    control_steps += 1
                    trajectory_digest.update(
                        np.asarray(action, dtype="<f8").tobytes()
                    )
                    trajectory_digest.update(
                        np.asarray(
                            raw_robot_state(observation, np, quat2axisangle),
                            dtype="<f8",
                        ).tobytes()
                    )
                    if done:
                        success = True
                        break
                    environment_step += 1
            except BaseException as error:
                episode_error = error

            episode_seconds = time.perf_counter() - episode_started
            query_count = episode_query_index
            episode_record: dict[str, Any] = {
                "run_id": RUN_ID,
                "episode_id": episode_id,
                "policy": policy,
                "task": f"{SUITE}:{TASK_ID}",
                "initial_state_id": state_id,
                "seed": SEED,
                "status": (
                    "interrupted"
                    if isinstance(episode_error, Interrupted)
                    else "failed" if episode_error is not None else "completed"
                ),
                "success": success if episode_error is None else False,
                "failure_reason": (
                    None if success and episode_error is None
                    else (
                        str(episode_error)
                        if episode_error is not None
                        else "task_not_completed_within_horizon"
                    )
                ),
                "steps": control_steps,
                "query_count": query_count,
                "refresh_count": refresh_count,
                "reuse_count": reuse_count,
                "skipped_refresh_count": reuse_count,
                "refresh_rate": (
                    refresh_count / query_count if query_count else 0.0
                ),
                "trigger_counts": dict(sorted(trigger_counts.items())),
                "trajectory_sha256": trajectory_digest.hexdigest(),
                "latency_ms": {
                    "total_episode": episode_seconds * 1000,
                    "policy_median": percentile(query_wall_ms, 50, np),
                    "policy_p95": percentile(query_wall_ms, 95, np),
                    "control_step_median": percentile(control_wall_ms, 50, np),
                    "control_step_p95": percentile(control_wall_ms, 95, np),
                },
                "peak_gpu_memory_mib": (
                    float(torch.cuda.max_memory_allocated()) / 1024**2
                ),
                "started_at_utc": episode_started_utc,
                "finished_at_utc": utc_now(),
                "error_type": (
                    type(episode_error).__name__ if episode_error is not None else None
                ),
                "error": str(episode_error) if episode_error is not None else None,
            }
            validate(episode_record, schemas["episode_result.schema.json"])
            store.write_episode(episode_id, episode_record)
            ensure_bounds(run_dir, run_started)
            if episode_error is not None:
                raise episode_error
            if query_count < 1:
                raise RuntimeError(f"Episode {episode_id} completed without a query")
            if refresh_count + reuse_count != query_count:
                raise RuntimeError(f"Episode {episode_id} count reconciliation failed")

        episode_records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(store.episode_dir.glob("*.json"))
        ]
        matrix_summary = validate_complete_matrix(
            episode_records,
            len(store.completed_query_indices()),
        )
        summary.update(
            {
                "status": "completed",
                **matrix_summary,
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
        finished_at = utc_now()
        summary.update(
            {
                "removed_upstream_backup_files": removed_backups,
                "unexpected_new_checkpoint_files": unexpected_new_files,
                "checkpoint_hashes_after_restore": hashes_after,
                "checkpoint_restored": checkpoint_restored,
                "artifact_bytes": directory_size(run_dir),
                "elapsed_seconds": time.monotonic() - run_started,
                "finished_at_utc": finished_at,
            }
        )
        limit_error: BaseException | None = None
        if summary["artifact_bytes"] > ARTIFACT_CAP_BYTES:
            limit_error = RuntimeError(
                f"Phase 5 artifact cap exceeded: {summary['artifact_bytes']} bytes"
            )
        elif summary["elapsed_seconds"] > WALL_CAP_SECONDS:
            limit_error = RuntimeError(
                "Phase 5 wall-clock cap exceeded: "
                f"{summary['elapsed_seconds']:.3f} seconds"
            )
        if caught_error is None and limit_error is not None:
            caught_error = limit_error
            terminal_status = "failed"
            exit_code = 1
            summary.update(
                {
                    "status": "failed",
                    "error_type": type(caught_error).__name__,
                    "error": str(caught_error),
                }
            )
        if not checkpoint_restored or unexpected_new_files:
            if caught_error is None:
                caught_error = RuntimeError("Checkpoint restoration/inventory audit failed")
            terminal_status = "failed"
            exit_code = 1
            summary.update(
                {
                    "status": "failed",
                    "error_type": type(caught_error).__name__,
                    "error": str(caught_error),
                }
            )

        manifest["finished_at_utc"] = finished_at
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
        store.write_run_event(
            1,
            terminal_status.upper(),
            {
                "finished_at_utc": finished_at,
                "checkpoint_restored": checkpoint_restored,
                "query_record_count": len(store.completed_query_indices()),
                "episode_record_count": len(list(store.episode_dir.glob("*.json"))),
            },
        )
        if model is not None:
            del model
        torch.cuda.empty_cache()

    print(json.dumps(summary, indent=2, sort_keys=True))
    if caught_error is not None:
        print(
            f"Phase 5 stopped: {type(caught_error).__name__}: {caught_error}",
            file=sys.stderr,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
