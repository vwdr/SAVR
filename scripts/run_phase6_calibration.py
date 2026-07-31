#!/usr/bin/env python3
"""Run a frozen, resumable Phase 6 calibration configuration on TITAN."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_phase5_core_smoke import (
    CHECKPOINT_RELATIVE,
    CHECKPOINT_REVISION,
    LIBERO_REVISION,
    UPSTREAM_REVISION,
    atomic_json,
    directory_size,
    make_query_record,
    percentile,
    raw_robot_state,
    require_clean_revision,
    sha256,
    validate_checkpoint_inventory,
    validate_schemas,
)


EXPECTED_ROOT = Path("/home/ved/SAVR")
SUITE = "libero_spatial"
TASK_IDS = tuple(range(10))
INITIAL_STATE_IDS = tuple(range(10))
SEED = 0
EXPECTED_PAIRINGS = 100
MAX_TERMINAL_ERRORS_PER_SETTING = 3
GLOBAL_ARTIFACT_CAP_BYTES = 2 * 1024**3
MAX_PHASE6_GPU_SECONDS = 48 * 60 * 60


class Interrupted(RuntimeError):
    """Raised for a recoverable termination request."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol",
        "run_id",
        "settings",
        "wall_cap_seconds",
        "artifact_cap_bytes",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Phase 6 config lacks fields: {sorted(missing)}")
    if config["protocol"] not in {
        "PHASE6_CALIBRATION_PROTOCOL.md",
        "PHASE6R_PROTOCOL_V1.md",
        "PHASE6S_PROTOCOL_V1.md",
    }:
        raise ValueError("Configuration names the wrong protocol")
    if not isinstance(config["settings"], list) or not config["settings"]:
        raise ValueError("Configuration must contain at least one setting")
    identifiers: set[str] = set()
    for setting in config["settings"]:
        identifier = str(setting.get("configuration_id", ""))
        if not identifier or identifier in identifiers:
            raise ValueError("Configuration identifiers must be unique and non-empty")
        identifiers.add(identifier)
        policy = setting.get("policy")
        if policy not in {"FR", "PR", "VOR", "SAVR", "SAVR2", "SAVR3"}:
            raise ValueError(f"Unsupported policy in Phase 6 config: {policy}")
        if policy == "PR" and int(setting.get("period", 0)) < 1:
            raise ValueError("PR requires a positive integer period")
        if policy in {"VOR", "SAVR"}:
            if int(setting.get("max_reuse_horizon", 0)) < 1:
                raise ValueError(f"{policy} requires a positive reuse horizon")
            threshold_names = ["image_threshold"]
            if policy == "SAVR":
                threshold_names.extend(["state_threshold", "action_threshold"])
            for name in threshold_names:
                value = float(setting.get(name, -1))
                if value < 0 or not __import__("math").isfinite(value):
                    raise ValueError(f"{policy} requires a finite non-negative {name}")
        if policy in {"SAVR2", "SAVR3"}:
            expected_thresholds = {
                "image_thresholds": {"full_image", "wrist_image"},
                "state_thresholds": {"translation", "orientation", "gripper"},
                "action_thresholds": {"translation", "rotation", "gripper"},
            }
            for field, expected_keys in expected_thresholds.items():
                values = setting.get(field)
                if not isinstance(values, dict) or set(values) != expected_keys:
                    raise ValueError(f"{policy} requires exact {field} keys")
                if any(
                    float(value) < 0
                    or not __import__("math").isfinite(float(value))
                    for value in values.values()
                ):
                    raise ValueError(f"{policy} requires finite non-negative {field}")
            if float(setting.get("skip_budget", -1)) not in {0.05, 0.10, 0.15}:
                raise ValueError(f"{policy} requires a frozen skip budget")
            reversal_veto = setting.get("translation_direction_reversal_veto", False)
            if policy == "SAVR3" and reversal_veto is not True:
                raise ValueError("SAVR3 requires its frozen translation-reversal veto")
            if policy == "SAVR2" and reversal_veto:
                raise ValueError("SAVR2 cannot enable the SAVR3 reversal veto")
    initial_state_ids = config.get("initial_state_ids", list(INITIAL_STATE_IDS))
    if (
        not isinstance(initial_state_ids, list)
        or not initial_state_ids
        or len(set(initial_state_ids)) != len(initial_state_ids)
        or any(not isinstance(value, int) or value not in INITIAL_STATE_IDS for value in initial_state_ids)
    ):
        raise ValueError("Initial-state IDs must be a unique non-empty subset of 0-9")
    if config.get("task_ids", list(TASK_IDS)) != list(TASK_IDS):
        raise ValueError("Phase 6 runner requires the frozen task IDs 0-9")
    if int(config.get("seed", SEED)) != SEED:
        raise ValueError("Phase 6 runner requires seed 0")
    if int(config["wall_cap_seconds"]) < 1:
        raise ValueError("Run wall cap must be positive")
    if not 1 <= int(config["artifact_cap_bytes"]) <= GLOBAL_ARTIFACT_CAP_BYTES:
        raise ValueError("Run artifact cap is outside the Phase 6 bound")
    return config


def controller_for_setting(
    setting: dict[str, Any],
    *,
    state_statistics: dict[str, Any],
    action_statistics: dict[str, Any],
    controllers: Any,
) -> Any:
    policy = setting["policy"]
    if policy == "FR":
        return controllers.FullRefreshController()
    if policy == "PR":
        return controllers.PeriodicRefreshController(period=int(setting["period"]))
    if policy == "VOR":
        return controllers.VisualOnlyRefreshController(
            image_threshold=float(setting["image_threshold"]),
            max_reuse_horizon=int(setting["max_reuse_horizon"]),
        )
    if policy == "SAVR":
        return controllers.StateAwareVisualRefreshController(
            image_threshold=float(setting["image_threshold"]),
            state_threshold=float(setting["state_threshold"]),
            action_threshold=float(setting["action_threshold"]),
            max_reuse_horizon=int(setting["max_reuse_horizon"]),
            state_q01=state_statistics["q01"],
            state_q99=state_statistics["q99"],
            action_q01=action_statistics["q01"],
            action_q99=action_statistics["q99"],
        )
    if policy in {"SAVR2", "SAVR3"}:
        from savr.controllers import Policy
        from savr.savr2 import SAVR2Configuration, StateAwareVisualRefresh2Controller

        return StateAwareVisualRefresh2Controller(
            configuration=SAVR2Configuration(
                configuration_id=str(setting["configuration_id"]),
                image_thresholds=setting["image_thresholds"],
                state_thresholds=setting["state_thresholds"],
                action_thresholds=setting["action_thresholds"],
                skip_budget=float(setting["skip_budget"]),
                translation_direction_reversal_veto=bool(
                    setting.get("translation_direction_reversal_veto", False)
                ),
                policy=Policy(policy),
            ),
            state_q01=state_statistics["q01"],
            state_q99=state_statistics["q99"],
            action_q01=action_statistics["q01"],
            action_q99=action_statistics["q99"],
        )
    raise ValueError(f"Unsupported policy: {policy}")


def assert_component_invariants(result: Any, timing: Any) -> None:
    counts = dict(timing.component_counts)
    for name in ("vision_backbone", "visual_projector", "language_model", "action_head"):
        counts.setdefault(name, 0)
    expected_visual = 1 if result.decision.refresh else 0
    if counts["vision_backbone"] != expected_visual:
        raise RuntimeError(f"Vision count differs from refresh decision: {counts}")
    if counts["visual_projector"] != expected_visual:
        raise RuntimeError(f"Projector count differs from refresh decision: {counts}")
    if counts["language_model"] != 1 or counts["action_head"] != 1:
        raise RuntimeError(f"Downstream component count differs: {counts}")
    expected_event = "refresh" if result.decision.refresh else "reuse"
    if result.cache_event != expected_event:
        raise RuntimeError("Cache event differs from refresh decision")


def assert_savr2_episode_invariants(
    records: list[dict[str, Any]], setting: dict[str, Any]
) -> None:
    """Reconcile every online SAVR 2.0 temporal and prefix-budget decision."""

    if setting.get("policy") not in {"SAVR2", "SAVR3"}:
        return
    budget = float(setting["skip_budget"])
    reuses = 0
    previous_reuse = False
    for index, record in enumerate(records):
        reuse = not bool(record["refresh"])
        if reuse:
            reuses += 1
            if index < 5:
                raise RuntimeError("Safety controller reused before the warm-up boundary")
            if previous_reuse:
                raise RuntimeError("Safety controller produced consecutive reuse decisions")
        if reuses / (index + 1) > budget + 1e-15:
            raise RuntimeError("Safety controller exceeded its episode-prefix skip budget")
        previous_reuse = reuse


def calibration_trace(
    *,
    images: dict[str, Any],
    state: Any,
    actions: Any,
    np: Any,
    prepare_image_representations: Any,
) -> dict[str, Any]:
    representations = prepare_image_representations(images)
    state_array = np.asarray(state, dtype=np.float64)
    action_array = np.asarray(actions, dtype=np.float64)
    if state_array.shape != (8,) or not np.isfinite(state_array).all():
        raise RuntimeError(f"Calibration state is invalid: {state_array.shape}")
    if action_array.shape != (8, 7) or not np.isfinite(action_array).all():
        raise RuntimeError(f"Calibration action chunk is invalid: {action_array.shape}")
    result = {
        "images": {},
        "image_shapes": {},
        "state": state_array.tolist(),
        "actions": action_array.reshape(-1).tolist(),
    }
    for name, values in sorted(representations.items()):
        array = np.asarray(values, dtype=np.float32)
        shape: tuple[int, ...]
        if array.size == 32 * 32 * 3:
            shape = (32, 32, 3)
        elif array.size == 32 * 32:
            shape = (32, 32)
        else:
            raise RuntimeError(f"Unexpected image representation width for {name}")
        result["images"][name] = array.tolist()
        result["image_shapes"][name] = list(shape)
    return result


def record_attempt(
    run_dir: Path,
    *,
    episode_id: str,
    error: BaseException,
    staged_queries: list[dict[str, Any]],
) -> None:
    attempt_dir = run_dir / "attempts"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    index = len(list(attempt_dir.glob(f"{episode_id}_attempt_*.json")))
    payload = {
        "episode_id": episode_id,
        "attempt": index,
        "status": "interrupted",
        "error_type": type(error).__name__,
        "error": str(error),
        "query_count": len(staged_queries),
        "queries": staged_queries,
        "finished_at_utc": utc_now(),
    }
    path = attempt_dir / f"{episode_id}_attempt_{index:02d}.json"
    if path.exists():
        raise RuntimeError(f"Attempt record already exists: {path}")
    atomic_json(path, payload)


def existing_episode_records(store: Any) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(store.episode_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        episode_id = str(record["episode_id"])
        if episode_id in records:
            raise RuntimeError(f"Duplicate episode record: {episode_id}")
        records[episode_id] = record
    return records


def phase6_result_bytes(project_root: Path) -> int:
    total = 0
    for path in (project_root / "results").glob("phase6-*"):
        if path.is_dir():
            total += directory_size(path)
    return total


def phase6_elapsed_seconds(project_root: Path) -> float:
    total = 0.0
    for path in (project_root / "results").glob("phase6-*/run_summary.json"):
        try:
            total += float(json.loads(path.read_text(encoding="utf-8")).get(
                "accumulated_elapsed_seconds", 0.0
            ))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            raise RuntimeError(f"Cannot reconcile Phase 6 elapsed time: {path}")
    return total


def progress_summary(
    *,
    records: dict[str, dict[str, Any]],
    settings: list[dict[str, Any]],
    elapsed: float,
    expected_pairings: int = EXPECTED_PAIRINGS,
) -> dict[str, Any]:
    expected = expected_pairings * len(settings)
    complete = sum(record["status"] == "completed" for record in records.values())
    failed = sum(record["status"] == "failed" for record in records.values())
    terminal = complete + failed
    rate = terminal / elapsed if elapsed > 0 and terminal else 0.0
    remaining = expected - terminal
    return {
        "expected": expected,
        "terminal": terminal,
        "complete": complete,
        "failed": failed,
        "remaining": remaining,
        "elapsed_seconds": elapsed,
        "estimated_remaining_seconds": remaining / rate if rate > 0 else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    if project_root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing to run outside {EXPECTED_ROOT}: {project_root}")
    config_path = arguments.config.resolve()
    if project_root not in config_path.parents:
        raise SystemExit("Phase 6 config must be inside /home/ved/SAVR")
    config = load_config(config_path)
    initial_state_ids = tuple(
        int(value) for value in config.get("initial_state_ids", INITIAL_STATE_IDS)
    )
    expected_pairings = len(TASK_IDS) * len(initial_state_ids)

    physical_gpu_id = os.environ.get("SAVR_PHYSICAL_GPU_ID")
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    selected_uuid = os.environ.get("SAVR_SELECTED_GPU_UUID")
    if not physical_gpu_id or visible_gpu != physical_gpu_id:
        raise SystemExit("SAVR_PHYSICAL_GPU_ID must exactly match CUDA_VISIBLE_DEVICES")
    if not selected_uuid:
        raise SystemExit("SAVR_SELECTED_GPU_UUID is required")

    termination_state = {"publishing": False, "requested": False}

    def handle_signal(_signum: int, _frame: Any) -> None:
        if termination_state["publishing"]:
            termination_state["requested"] = True
            return
        raise Interrupted("Phase 6 runner received a termination signal")

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
    run_dir = project_root / "results" / str(config["run_id"])
    run_dir.mkdir(parents=True, exist_ok=True)

    project_revision = require_clean_revision(project_root)
    upstream_revision = require_clean_revision(upstream_root, UPSTREAM_REVISION)
    libero_revision = require_clean_revision(libero_root, LIBERO_REVISION)
    checkpoint_inventory = validate_checkpoint_inventory(project_root, checkpoint)
    config_sha256 = sha256(config_path)

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
    from savr.signals import prepare_image_representations
    from savr.timing import (
        ModuleTimingHooks,
        SynchronizedQueryTimer,
        TorchCudaEventBackend,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one selected CUDA device must be visible")

    schemas = validate_schemas(project_root, Draft202012Validator)
    store = ImmutableRecordStore(run_dir)
    records = existing_episode_records(store)
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "run_summary.json"
    invocation_started = time.monotonic()
    previous_elapsed = 0.0
    if summary_path.exists():
        previous_elapsed = float(
            json.loads(summary_path.read_text(encoding="utf-8")).get(
                "accumulated_elapsed_seconds", 0.0
            )
        )

    frozen_config = {
        **config,
        "suite": SUITE,
        "task_ids": list(TASK_IDS),
        "initial_state_ids": list(initial_state_ids),
        "seed": SEED,
        "expected_pairings_per_setting": expected_pairings,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "openvla_oft_revision": UPSTREAM_REVISION,
        "libero_revision": LIBERO_REVISION,
        "config_sha256": config_sha256,
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["configuration"] != frozen_config:
            raise RuntimeError("Existing run manifest differs from frozen configuration")
        if manifest["savr_git_revision"] != project_revision:
            raise RuntimeError("Resume revision differs from the original run revision")
    else:
        manifest = {
            "run_id": config["run_id"],
            "started_at_utc": utc_now(),
            "finished_at_utc": None,
            "status": "running",
            "policy": (
                config["settings"][0]["policy"]
                if len(config["settings"]) == 1
                else "MIXED"
            ),
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
            "configuration": frozen_config,
            "command": f"scripts/run_phase6_calibration.py --config {config_path}",
            "notes": "Phase 6 calibration only; no final holdout outcomes.",
        }
    validate(manifest, schemas["run_manifest.schema.json"])
    manifest["status"] = "running"
    manifest["finished_at_utc"] = None
    atomic_json(manifest_path, manifest)
    event_sequence = len(list(store.event_dir.glob("event_*.json")))
    store.write_run_event(event_sequence, "RUNNING", {"started_at_utc": utc_now()})

    protected_names = (
        "config.json",
        "configuration_prismatic.py",
        "modeling_prismatic.py",
    )
    protected_bytes = {name: (checkpoint / name).read_bytes() for name in protected_names}
    protected_hashes = {name: sha256(checkpoint / name) for name in protected_names}
    checkpoint_files_before = {item.name for item in checkpoint.iterdir()}

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
    torch.cuda.reset_peak_memory_stats()

    model = action_head = proprio_projector = noisy_action_projector = processor = None
    timing_hooks = current_env = None
    caught_error: BaseException | None = None
    terminal_status = "failed"
    summary: dict[str, Any] = {}
    try:
        os.chdir(upstream_root)
        load_started = time.perf_counter()
        model, action_head, proprio_projector, noisy_action_projector, processor = (
            upstream_eval.initialize_model(cfg)
        )
        torch.cuda.synchronize()
        model_load_seconds = time.perf_counter() - load_started
        if action_head is None or proprio_projector is None:
            raise RuntimeError("Pinned L1/proprio modules were not loaded")

        state_statistics = model.norm_stats[cfg.unnorm_key]["proprio"]
        action_statistics = model.norm_stats[cfg.unnorm_key]["action"]
        normalization_statistics = {
            "unnorm_key": cfg.unnorm_key,
            "state_q01": [float(value) for value in state_statistics["q01"]],
            "state_q99": [float(value) for value in state_statistics["q99"]],
            "action_q01": [float(value) for value in action_statistics["q01"]],
            "action_q99": [float(value) for value in action_statistics["q99"]],
        }
        if "normalization_statistics" in manifest and (
            manifest["normalization_statistics"] != normalization_statistics
        ):
            raise RuntimeError("Checkpoint normalization statistics changed")
        manifest["normalization_statistics"] = normalization_statistics
        atomic_json(manifest_path, manifest)
        for setting in config["settings"]:
            controller_for_setting(
                setting,
                state_statistics=state_statistics,
                action_statistics=action_statistics,
                controllers=controllers,
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
        settings = config["settings"]
        failed_by_setting = Counter(
            record["configuration_id"]
            for record in records.values()
            if record["status"] == "failed"
        )

        for task_id in TASK_IDS:
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            current_env, task_description = get_libero_env(
                task,
                cfg.model_family,
                resolution=cfg.env_img_res,
            )
            try:
                for state_position, state_id in enumerate(initial_state_ids):
                    pair_index = task_id * len(initial_state_ids) + state_position
                    rotation = pair_index % len(settings)
                    ordered_settings = settings[rotation:] + settings[:rotation]
                    for setting in ordered_settings:
                        configuration_id = setting["configuration_id"]
                        episode_id = (
                            f"{configuration_id}_task_{task_id:02d}_state_{state_id:02d}"
                        )
                        if episode_id in records:
                            continue
                        if failed_by_setting[configuration_id] >= (
                            MAX_TERMINAL_ERRORS_PER_SETTING
                        ):
                            raise RuntimeError(
                                f"Reached terminal-error stop rule for {configuration_id}"
                            )
                        invocation_elapsed = time.monotonic() - invocation_started
                        accumulated = previous_elapsed + invocation_elapsed
                        if accumulated > float(config["wall_cap_seconds"]):
                            raise RuntimeError("Reached frozen run wall-clock cap")
                        other_elapsed = phase6_elapsed_seconds(project_root)
                        if other_elapsed + invocation_elapsed > MAX_PHASE6_GPU_SECONDS:
                            raise RuntimeError("Reached 48-GPU-hour Phase 6 cap")
                        if phase6_result_bytes(project_root) > GLOBAL_ARTIFACT_CAP_BYTES:
                            raise RuntimeError("Reached two-GiB Phase 6 artifact cap")

                        set_seed_everywhere(SEED)
                        controller = controller_for_setting(
                            setting,
                            state_statistics=state_statistics,
                            action_statistics=action_statistics,
                            controllers=controllers,
                        )
                        adapter = OpenVLAProjectedFeatureAdapter(
                            model=model,
                            controller=controller,
                        )
                        adapter.begin_context(
                            CacheContext(
                                episode_id=episode_id,
                                task_id=str(task_id),
                                checkpoint_id=CHECKPOINT_REVISION,
                                configuration_id=configuration_id,
                            )
                        )
                        current_env.reset()
                        observation = current_env.set_init_state(initial_states[state_id])
                        action_queue: deque[Any] = deque(maxlen=cfg.num_open_loop_steps)
                        staged_queries: list[dict[str, Any]] = []
                        query_wall_ms: list[float] = []
                        control_wall_ms: list[float] = []
                        trigger_counts: Counter[str] = Counter()
                        refresh_count = reuse_count = 0
                        environment_step = control_steps = 0
                        success = False
                        trajectory_digest = hashlib.sha256()
                        episode_started = time.perf_counter()
                        episode_started_utc = utc_now()
                        episode_error: BaseException | None = None

                        try:
                            max_steps = upstream_eval.TASK_MAX_STEPS[SUITE]
                            while environment_step < max_steps + cfg.num_steps_wait:
                                invocation_elapsed = time.monotonic() - invocation_started
                                if previous_elapsed + invocation_elapsed > float(
                                    config["wall_cap_seconds"]
                                ):
                                    raise RuntimeError("Reached frozen run wall-clock cap")
                                if environment_step < cfg.num_steps_wait:
                                    observation, _, _, _ = current_env.step(
                                        get_libero_dummy_action(cfg.model_family)
                                    )
                                    environment_step += 1
                                    continue

                                if not action_queue:
                                    policy_observation, _ = (
                                        upstream_eval.prepare_observation(
                                            observation,
                                            resize_size,
                                        )
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
                                    assert_component_invariants(result, timing)
                                    setting_index = next(
                                        index
                                        for index, item in enumerate(settings)
                                        if item["configuration_id"] == configuration_id
                                    )
                                    stable_query_index = (
                                        setting_index * 100_000
                                        + pair_index * 1_000
                                        + len(staged_queries)
                                    )
                                    query_record = make_query_record(
                                        run_id=config["run_id"],
                                        global_query_index=stable_query_index,
                                        episode_id=episode_id,
                                        policy=setting["policy"],
                                        environment_step=environment_step,
                                        actions=actions,
                                        result=result,
                                        timing=timing,
                                        cache_age_after=adapter.cache.age,
                                        np=np,
                                    )
                                    query_record["configuration_id"] = configuration_id
                                    if setting["policy"] == "FR" or bool(
                                        config.get("save_calibration_traces", False)
                                    ):
                                        query_record["calibration_trace"] = (
                                            calibration_trace(
                                                images=images,
                                                state=policy_observation["state"],
                                                actions=actions,
                                                np=np,
                                                prepare_image_representations=(
                                                    prepare_image_representations
                                                ),
                                            )
                                        )
                                    validate(
                                        query_record,
                                        schemas["query_record.schema.json"],
                                    )
                                    staged_queries.append(query_record)
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
                                    raise RuntimeError("Processed action is non-finite")
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
                                        raw_robot_state(
                                            observation,
                                            np,
                                            quat2axisangle,
                                        ),
                                        dtype="<f8",
                                    ).tobytes()
                                )
                                if done:
                                    success = True
                                    break
                                environment_step += 1
                        except BaseException as error:
                            episode_error = error

                        if isinstance(episode_error, Interrupted):
                            record_attempt(
                                run_dir,
                                episode_id=episode_id,
                                error=episode_error,
                                staged_queries=staged_queries,
                            )
                            raise episode_error

                        status = "failed" if episode_error is not None else "completed"
                        query_count = len(staged_queries)
                        episode_record = {
                            "run_id": config["run_id"],
                            "episode_id": episode_id,
                            "configuration_id": configuration_id,
                            "policy": setting["policy"],
                            "task": f"{SUITE}:{task_id}",
                            "initial_state_id": state_id,
                            "seed": SEED,
                            "status": status,
                            "success": success if episode_error is None else False,
                            "failure_reason": (
                                None
                                if success and episode_error is None
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
                                "total_episode": (
                                    time.perf_counter() - episode_started
                                ) * 1000,
                                "policy_median": percentile(query_wall_ms, 50, np),
                                "policy_p95": percentile(query_wall_ms, 95, np),
                                "control_step_median": percentile(
                                    control_wall_ms, 50, np
                                ),
                                "control_step_p95": percentile(
                                    control_wall_ms, 95, np
                                ),
                            },
                            "peak_gpu_memory_mib": (
                                float(torch.cuda.max_memory_allocated()) / 1024**2
                            ),
                            "started_at_utc": episode_started_utc,
                            "finished_at_utc": utc_now(),
                            "error_type": (
                                type(episode_error).__name__
                                if episode_error is not None
                                else None
                            ),
                            "error": (
                                str(episode_error)
                                if episode_error is not None
                                else None
                            ),
                        }
                        validate(
                            episode_record,
                            schemas["episode_result.schema.json"],
                        )
                        termination_state["publishing"] = True
                        try:
                            for query_record in staged_queries:
                                store.write_query(
                                    query_record["query_index"],
                                    query_record,
                                )
                            store.write_episode(episode_id, episode_record)
                        finally:
                            termination_state["publishing"] = False
                        records[episode_id] = episode_record
                        if termination_state["requested"]:
                            raise Interrupted(
                                "Termination was deferred until terminal-record "
                                "publication completed"
                            )
                        if episode_error is not None:
                            failed_by_setting[configuration_id] += 1
                        if query_count < 1:
                            raise RuntimeError(f"{episode_id} has no policy query")
                        if refresh_count + reuse_count != query_count:
                            raise RuntimeError(f"{episode_id} counters do not reconcile")
                        assert_savr2_episode_invariants(staged_queries, setting)
                        if directory_size(run_dir) > int(config["artifact_cap_bytes"]):
                            raise RuntimeError("Reached frozen run artifact cap")

                        elapsed = previous_elapsed + (
                            time.monotonic() - invocation_started
                        )
                        atomic_json(
                            run_dir / "progress.json",
                            progress_summary(
                                records=records,
                                settings=settings,
                                elapsed=elapsed,
                                expected_pairings=expected_pairings,
                            ),
                        )
            finally:
                current_env.close()
                current_env = None

        expected_ids = {
            (
                f"{setting['configuration_id']}_task_{task_id:02d}_"
                f"state_{state_id:02d}"
            )
            for setting in settings
            for task_id in TASK_IDS
            for state_id in initial_state_ids
        }
        if set(records) != expected_ids:
            raise RuntimeError(
                f"Terminal matrix differs: missing={len(expected_ids - set(records))}, "
                f"extra={len(set(records) - expected_ids)}"
            )
        terminal_status = "completed"
        summary.update(
            {
                "status": terminal_status,
                "model_load_seconds_last_invocation": model_load_seconds,
                "progress": progress_summary(
                    records=records,
                    settings=settings,
                    elapsed=previous_elapsed + (
                        time.monotonic() - invocation_started
                    ),
                    expected_pairings=expected_pairings,
                ),
                "setting_summaries": {
                    setting["configuration_id"]: {
                        "terminal_episodes": sum(
                            record["configuration_id"]
                            == setting["configuration_id"]
                            for record in records.values()
                        ),
                        "failed_episodes": sum(
                            record["configuration_id"]
                            == setting["configuration_id"]
                            and record["status"] == "failed"
                            for record in records.values()
                        ),
                        "successes": sum(
                            record["configuration_id"]
                            == setting["configuration_id"]
                            and bool(record["success"])
                            for record in records.values()
                        ),
                        "queries": sum(
                            int(record["query_count"])
                            for record in records.values()
                            if record["configuration_id"]
                            == setting["configuration_id"]
                        ),
                        "refreshes": sum(
                            int(record["refresh_count"])
                            for record in records.values()
                            if record["configuration_id"]
                            == setting["configuration_id"]
                        ),
                        "reuses": sum(
                            int(record["reuse_count"])
                            for record in records.values()
                            if record["configuration_id"]
                            == setting["configuration_id"]
                        ),
                    }
                    for setting in settings
                },
                "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )
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
        files_after = {item.name for item in checkpoint.iterdir()}
        new_files = sorted(files_after - checkpoint_files_before)
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
        invocation_elapsed = time.monotonic() - invocation_started
        accumulated_elapsed = previous_elapsed + invocation_elapsed
        summary.update(
            {
                "run_id": config["run_id"],
                "accumulated_elapsed_seconds": accumulated_elapsed,
                "last_invocation_elapsed_seconds": invocation_elapsed,
                "artifact_bytes": directory_size(run_dir),
                "phase6_artifact_bytes": phase6_result_bytes(project_root),
                "checkpoint_inventory": checkpoint_inventory,
                "checkpoint_hashes_before": protected_hashes,
                "checkpoint_hashes_after_restore": hashes_after,
                "checkpoint_restored": checkpoint_restored,
                "removed_upstream_backup_files": removed_backups,
                "unexpected_new_checkpoint_files": unexpected_new_files,
                "finished_at_utc": utc_now(),
            }
        )
        if not checkpoint_restored or unexpected_new_files:
            terminal_status = "failed"
            caught_error = caught_error or RuntimeError(
                "Checkpoint restoration/inventory audit failed"
            )
            summary.update(
                {
                    "status": "failed",
                    "error_type": type(caught_error).__name__,
                    "error": str(caught_error),
                }
            )
        if summary["artifact_bytes"] > int(config["artifact_cap_bytes"]):
            terminal_status = "failed"
            caught_error = caught_error or RuntimeError("Run artifact cap exceeded")
        if summary["phase6_artifact_bytes"] > GLOBAL_ARTIFACT_CAP_BYTES:
            terminal_status = "failed"
            caught_error = caught_error or RuntimeError("Phase 6 artifact cap exceeded")
        if accumulated_elapsed > float(config["wall_cap_seconds"]):
            terminal_status = "failed"
            caught_error = caught_error or RuntimeError("Run wall-clock cap exceeded")
        summary["status"] = terminal_status
        if caught_error is not None:
            summary["error_type"] = type(caught_error).__name__
            summary["error"] = str(caught_error)
        manifest["status"] = terminal_status
        manifest["finished_at_utc"] = summary["finished_at_utc"]
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
            event_sequence + 1,
            terminal_status.upper(),
            {
                "finished_at_utc": summary["finished_at_utc"],
                "checkpoint_restored": checkpoint_restored,
                "terminal_episode_count": len(records),
            },
        )
        if model is not None:
            del model
        torch.cuda.empty_cache()

    print(json.dumps(summary, indent=2, sort_keys=True))
    if caught_error is not None:
        print(
            f"Phase 6 stopped: {type(caught_error).__name__}: {caught_error}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
