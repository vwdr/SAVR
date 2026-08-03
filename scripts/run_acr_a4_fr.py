#!/usr/bin/env python3
"""Run the frozen ACR A4 upstream-FR LIBERO-Object development matrix."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_ROOT = Path("/home/ved/SAVR")
RUN_ID = "acr-a4-upstream-fr-object-dev00-09-v01"
PHASE = "A4"
POLICY = "upstream-fr"
SUITE = "libero_object"
TASK_IDS = tuple(range(10))
INITIAL_STATE_IDS = tuple(range(10))
SEED = 0
ATTEMPT_CAP = 100
WALL_CAP_SECONDS = 28_800
ARTIFACT_CAP_BYTES = 1024**3
CHECKPOINT_REVISION = "638918f3d1c2e43a39a8a20772bdb8b91835e4b7"
OPENVLA_REVISION = "e4287e94541f459edc4feabc4e181f537cd569a8"
LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
CHECKPOINT_RELATIVE = Path("checkpoints/openvla-7b-oft-libero-four-suite")
CONFIG_RELATIVE = Path("configs/acr/development_fr.json")
STEADY_EXCLUSIONS = frozenset({0, 1, 2})


class Interrupted(RuntimeError):
    """Raised when the bounded runner receives a termination request."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def git_revision(path: Path, expected: str | None = None) -> str:
    revision = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain"], text=True
    ).strip()
    if status:
        raise RuntimeError(f"Refusing to use dirty source tree: {path}")
    if expected is not None and revision != expected:
        raise RuntimeError(f"Expected {expected} at {path}, found {revision}")
    return revision


def selected_gpu_snapshot(physical_gpu_id: str) -> dict[str, Any]:
    fields = "index,uuid,name,memory.total,memory.used,utilization.gpu"
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={physical_gpu_id}",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError("Selected GPU snapshot did not resolve exactly one device")
    values = [value.strip() for value in rows[0].split(",")]
    if len(values) != 6 or values[0] != physical_gpu_id:
        raise RuntimeError("Selected GPU snapshot identity is inconsistent")
    return {
        "index": int(values[0]),
        "uuid": values[1],
        "name": values[2],
        "memory_total_mib": int(values[3]),
        "memory_used_mib": int(values[4]),
        "utilization_percent": int(values[5]),
        "recorded_at_utc": utc_now(),
    }


def validate_config(config: dict[str, Any]) -> None:
    expected = {
        "run_id": RUN_ID,
        "phase": PHASE,
        "policy": POLICY,
        "suite": SUITE,
        "task_ids": list(TASK_IDS),
        "initial_state_ids": list(INITIAL_STATE_IDS),
        "seed": SEED,
        "expected_attempts": ATTEMPT_CAP,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Frozen A4 configuration mismatch: {key}")
    caps = config.get("resource_caps", {})
    if caps != {
        "gpu_count": 1,
        "model_processes": 1,
        "episode_attempts": ATTEMPT_CAP,
        "wall_seconds": WALL_CAP_SECONDS,
        "artifact_bytes": ARTIFACT_CAP_BYTES,
        "downloads_allowed": False,
    }:
        raise ValueError("Frozen A4 resource caps changed")
    if config.get("recovery", {}).get("automatic_episode_retry") is not False:
        raise ValueError("A4 cannot retry an episode automatically")
    if config.get("timing", {}).get("exclude_global_query_indices_from_steady_state") != sorted(
        STEADY_EXCLUSIONS
    ):
        raise ValueError("A4 steady-state timing exclusions changed")
    model = config.get("model", {})
    if (
        model.get("checkpoint_revision") != CHECKPOINT_REVISION
        or model.get("openvla_oft_revision") != OPENVLA_REVISION
        or model.get("libero_revision") != LIBERO_REVISION
        or model.get("num_open_loop_steps") != 8
        or model.get("num_images_in_input") != 2
        or model.get("use_proprio") is not True
        or model.get("use_l1_regression") is not True
        or model.get("use_diffusion") is not False
        or model.get("use_film") is not False
        or model.get("center_crop") is not True
    ):
        raise ValueError("Frozen A4 model configuration changed")
    templates = [
        ("acr-t25-h2-b30", 0.25, 2, 0.30),
        ("acr-t50-h4-b55", 0.50, 4, 0.55),
        ("acr-t70-h8-b75", 0.70, 8, 0.75),
    ]
    observed = [
        (
            item.get("configuration_id"),
            item.get("target_reuse"),
            item.get("horizon"),
            item.get("hard_reuse_cap"),
        )
        for item in config.get("candidate_templates", [])
    ]
    if observed != templates:
        raise ValueError("Frozen A4 candidate templates changed")


def _array_sha256(value: Any, np: Any) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def build_query_record(
    *,
    identity: Any,
    query_index: int,
    environment_step: int,
    global_query_index: int,
    scene_image: Any,
    wrist_image: Any,
    raw_state: tuple[float, ...],
    normalized_state: tuple[float, ...],
    scene_representation: tuple[float, ...],
    patch_scores: tuple[float, ...] | None,
    scene_score: float | None,
    translation_score: float | None,
    gripper_transition_veto: bool | None,
    direction_reversal: bool | None,
    actions: Any,
    timing: Any,
    model_dtype: str,
    model_device: str,
    visual_token_count: int,
    configuration_sha256: str,
    savr_revision: str,
    context_sha256: str,
    audit_sha256: Any,
) -> dict[str, Any]:
    component = dict(timing.component_device_ms)
    visual_cuda = component.get("vision_backbone", 0.0) + component.get("visual_projector", 0.0)
    return {
        "schema_version": "acr.query.v1",
        "run_id": RUN_ID,
        "attempt_id": identity.value,
        "query_id": identity.query_id(query_index),
        "phase": PHASE,
        "policy": POLICY,
        "suite": SUITE,
        "task_id": identity.task_id,
        "initial_state_id": identity.initial_state_id,
        "seed": identity.seed,
        "query_index": query_index,
        "environment_step": environment_step,
        "status": "completed",
        "error": None,
        "decision": {
            "scene_refresh": True,
            "refresh_reasons": ["policy"],
            "cache_age_before": None,
            "cache_age_after": 0,
            "reference_query_index": query_index,
            "scene_score": scene_score,
            "scene_threshold": None,
            "translation_score": translation_score,
            "translation_threshold": None,
            "gripper_transition_veto": gripper_transition_veto,
            "horizon": None,
            "reuse_count_before": 0,
            "query_count_before": query_index,
            "hard_reuse_cap": None,
        },
        "inputs": {
            "scene_image_sha256": audit_sha256(scene_image),
            "wrist_image_sha256": audit_sha256(wrist_image),
            "proprio_sha256": audit_sha256(raw_state),
            "action_sha256": _array_sha256(actions, __import__("numpy")),
            "context_sha256": context_sha256,
            "scene_representation": None,
            "scene_patch_scores": list(patch_scores) if patch_scores is not None else None,
            "proprio_raw": list(raw_state),
            "proprio_normalized": list(normalized_state),
            "direction_reversal": direction_reversal,
        },
        "camera_work": {
            "scene_siglip_calls": 1,
            "scene_dinov2_calls": 1,
            "scene_projector_calls": 1,
            "wrist_siglip_calls": 1,
            "wrist_dinov2_calls": 1,
            "wrist_projector_calls": 1,
            "visual_token_count": visual_token_count,
            "token_order": "scene-wrist",
            "dtype": model_dtype,
            "device": model_device,
            "downstream_calls": 1,
        },
        "timing": {
            "inclusive": True,
            "controller_wall_ms": 0.0,
            "cache_concat_wall_ms": 0.0,
            "scene_visual_cuda_ms": None,
            "wrist_visual_cuda_ms": None,
            "total_visual_cuda_ms": visual_cuda,
            "downstream_cuda_ms": max(0.0, timing.total_device_ms - visual_cuda),
            "query_cuda_ms": timing.total_device_ms,
            "query_wall_ms": timing.wall_ms,
        },
        "provenance": {
            "configuration_sha256": configuration_sha256,
            "savr_revision": savr_revision,
            "openvla_oft_revision": OPENVLA_REVISION,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "recorded_at_utc": utc_now(),
        },
    }


def build_trace_record(
    *,
    identity: Any,
    query_index: int,
    scene_representation: tuple[float, ...],
    normalized_eef_position: tuple[float, float, float],
    actions: Any,
    gripper_transition_veto: bool | None,
    direction_reversals: tuple[bool, bool, bool] | None,
    upstream_component_invocations: dict[str, int],
    encode_float_sequence: Any,
    np: Any,
) -> dict[str, Any]:
    action_values = tuple(float(value) for value in np.asarray(actions).reshape(-1))
    record = {
        "schema_version": "acr.fr-trace-query.v1",
        "run_id": RUN_ID,
        "attempt_id": identity.value,
        "query_id": identity.query_id(query_index),
        "episode_id": identity.episode_id,
        "query_index": query_index,
        "scene_representation": encode_float_sequence(scene_representation).as_record(),
        "normalized_eef_position": list(normalized_eef_position),
        "action_chunk": encode_float_sequence(action_values).as_record(),
        "action_shape": [8, 7],
        "action_sha256": _array_sha256(actions, np),
        "gripper_transition_veto": gripper_transition_veto,
        "translation_direction_reversals": (
            list(direction_reversals) if direction_reversals is not None else None
        ),
        "upstream_component_invocations": upstream_component_invocations,
    }
    record["semantic_sha256"] = value_sha256(record)
    return record


def assert_upstream_query(timing: Any) -> None:
    counts = dict(timing.component_counts)
    expected = {
        "vision_backbone": 1,
        "visual_projector": 1,
        "language_model": 1,
        "action_head": 1,
    }
    if {name: counts.get(name, 0) for name in expected} != expected:
        raise RuntimeError(f"Upstream FR component counts changed: {counts}")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    if project_root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing to run outside {EXPECTED_ROOT}: {project_root}")
    config_path = project_root / CONFIG_RELATIVE
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    physical_gpu_id = os.environ.get("SAVR_PHYSICAL_GPU_ID")
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    selected_uuid = os.environ.get("SAVR_SELECTED_GPU_UUID")
    if not physical_gpu_id or visible_gpu != physical_gpu_id or not selected_uuid:
        raise SystemExit("Selected GPU ID/UUID variables are incomplete or inconsistent")

    def handle_signal(_signum: int, _frame: Any) -> None:
        raise Interrupted("ACR A4 runner received a termination signal")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
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

    upstream_root = project_root / "third_party/openvla-oft"
    libero_root = project_root / "third_party/LIBERO"
    checkpoint = project_root / CHECKPOINT_RELATIVE
    run_dir = project_root / "results" / RUN_ID
    if run_dir.exists():
        raise SystemExit(f"Immutable A4 run already exists: {run_dir}")

    sys.path.insert(0, str(project_root / "src"))
    sys.path.insert(0, str(upstream_root))
    from run_acr_correctness import validate_checkpoint

    import numpy as np
    import torch  # type: ignore[import-not-found]
    from libero.libero import benchmark  # type: ignore[import-not-found]

    from experiments.robot.libero import run_libero_eval as upstream_eval  # type: ignore[import-not-found]
    from experiments.robot.libero.libero_utils import (  # type: ignore[import-not-found]
        get_libero_dummy_action,
        get_libero_env,
        quat2axisangle,
    )
    from experiments.robot.robot_utils import (  # type: ignore[import-not-found]
        get_image_resize_size,
        set_seed_everywhere,
    )
    from savr.acr.records import (
        AttemptIdentity,
        ImmutableRecordStore,
        encode_float_sequence,
        reconcile_episode_counts,
        validate_record,
    )
    from savr.acr.signals import (
        audit_sha256,
        prepare_scene_representation,
        scene_change_from_representations,
        scene_relative_translation,
        transition_signal,
    )
    from savr.signals import normalize_bounds
    from savr.timing import ModuleTimingHooks, SynchronizedQueryTimer, TorchCudaEventBackend
    from run_phase5_core_smoke import raw_robot_state

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one selected CUDA device must be visible")

    run_schema = json.loads((project_root / "schemas/acr_run.schema.json").read_text())
    query_schema = json.loads((project_root / "schemas/acr_query.schema.json").read_text())
    episode_schema = json.loads((project_root / "schemas/acr_episode.schema.json").read_text())
    savr_revision = git_revision(project_root)
    upstream_revision = git_revision(upstream_root, OPENVLA_REVISION)
    libero_revision = git_revision(libero_root, LIBERO_REVISION)
    checkpoint_before = validate_checkpoint(project_root, checkpoint)
    protected_names = ("config.json", "configuration_prismatic.py", "modeling_prismatic.py")
    protected_bytes = {name: (checkpoint / name).read_bytes() for name in protected_names}
    checkpoint_files_before = {item.name for item in checkpoint.iterdir()}
    config_sha256 = file_sha256(config_path)
    gpu_before = selected_gpu_snapshot(physical_gpu_id)
    if gpu_before["uuid"] != selected_uuid:
        raise RuntimeError("Selected GPU UUID changed between approval and launch")

    run_dir.mkdir(parents=True, exist_ok=False)
    store = ImmutableRecordStore(run_dir)
    planned = [
        AttemptIdentity(RUN_ID, POLICY, SUITE.replace("_", "-"), task, state, SEED, 0)
        for task in TASK_IDS
        for state in INITIAL_STATE_IDS
    ]
    started_at = utc_now()
    run_started = time.monotonic()
    manifest = {
        "schema_version": "acr.run.v1",
        "run_id": RUN_ID,
        "phase": PHASE,
        "policy": POLICY,
        "suite": SUITE,
        "scope": "LIBERO-Object tasks 0-9 states 0-9 seed 0 upstream FR",
        "status": "running",
        "configuration_sha256": config_sha256,
        "revisions": {
            "savr": savr_revision,
            "openvla_oft": upstream_revision,
            "libero": libero_revision,
            "checkpoint": CHECKPOINT_REVISION,
        },
        "schemas": {
            "run_sha256": file_sha256(project_root / "schemas/acr_run.schema.json"),
            "query_sha256": file_sha256(project_root / "schemas/acr_query.schema.json"),
            "episode_sha256": file_sha256(project_root / "schemas/acr_episode.schema.json"),
        },
        "population": {
            "task_ids": list(TASK_IDS),
            "initial_state_ids": list(INITIAL_STATE_IDS),
            "seed": SEED,
        },
        "resource_caps": {
            "gpu_count": 1,
            "model_processes": 1,
            "query_attempts": None,
            "episode_attempts": ATTEMPT_CAP,
            "wall_seconds": WALL_CAP_SECONDS,
            "artifact_bytes": ARTIFACT_CAP_BYTES,
            "downloads_allowed": False,
        },
        "planned_attempts": [identity.value for identity in planned],
        "recovery": {
            "mode": "preserve-and-restart",
            "overwrite_allowed": False,
            "resume_incomplete_episode": False,
            "next_attempt_index": 1,
        },
        "artifact_root": str(run_dir),
        "command": "scripts/run_acr_a4_fr.py",
        "host": socket.gethostname(),
        "selected_gpu_id": int(physical_gpu_id),
        "started_at_utc": started_at,
        "finished_at_utc": None,
        "records_sha256": None,
    }
    validate_record(manifest, run_schema)
    store.write_once("manifest", manifest)

    cfg = upstream_eval.GenerateConfig(
        pretrained_checkpoint=str(checkpoint),
        task_suite_name=SUITE,
        num_trials_per_task=10,
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
    hooks = current_env = None
    terminal_status = "failed"
    caught: BaseException | None = None
    attempts_started = 0
    global_query_index = 0
    episode_ids: list[str] = []
    successes = 0
    per_task_success = {str(task): 0 for task in TASK_IDS}
    try:
        os.chdir(upstream_root)
        model, action_head, proprio_projector, noisy_action_projector, processor = (
            upstream_eval.initialize_model(cfg)
        )
        torch.cuda.synchronize()
        if action_head is None or proprio_projector is None:
            raise RuntimeError("Pinned L1/proprio modules were not loaded")
        state_statistics = model.norm_stats[cfg.unnorm_key]["proprio"]
        q01, q99 = state_statistics["q01"], state_statistics["q99"]
        model_parameter = next(model.parameters())
        patch_count = int(model.vision_backbone.get_num_patches())
        visual_token_count = patch_count * 2
        timer = SynchronizedQueryTimer(TorchCudaEventBackend(torch))
        hooks = ModuleTimingHooks(
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
        for task_id in TASK_IDS:
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            current_env, task_description = get_libero_env(
                task, cfg.model_family, resolution=cfg.env_img_res
            )
            try:
                for state_id in INITIAL_STATE_IDS:
                    if attempts_started >= ATTEMPT_CAP:
                        raise RuntimeError("A4 episode-attempt cap exhausted")
                    if time.monotonic() - run_started >= WALL_CAP_SECONDS:
                        raise RuntimeError("A4 wall-time cap reached before scheduling")
                    if directory_size(run_dir) >= ARTIFACT_CAP_BYTES:
                        raise RuntimeError("A4 artifact cap reached before scheduling")
                    identity = AttemptIdentity(
                        RUN_ID, POLICY, SUITE.replace("_", "-"), task_id, state_id, SEED, 0
                    )
                    attempts_started += 1
                    episode_started = time.perf_counter()
                    episode_started_at = utc_now()
                    torch.cuda.reset_peak_memory_stats()
                    current_env.reset()
                    observation = current_env.set_init_state(initial_states[state_id])
                    action_queue: deque[Any] = deque(maxlen=cfg.num_open_loop_steps)
                    environment_step = query_index = control_steps = 0
                    success = False
                    query_records: list[dict[str, Any]] = []
                    trace_records: list[dict[str, Any]] = []
                    query_wall: list[float] = []
                    visual_cuda: list[float] = []
                    steady_query_wall: list[float] = []
                    steady_visual_cuda: list[float] = []
                    trajectory = hashlib.sha256()
                    previous_scene = previous_position = previous_actions = None
                    older_actions = None
                    episode_error: BaseException | None = None
                    try:
                        max_steps = upstream_eval.TASK_MAX_STEPS[cfg.task_suite_name]
                        while environment_step < max_steps + cfg.num_steps_wait:
                            if time.monotonic() - run_started >= WALL_CAP_SECONDS:
                                raise RuntimeError("A4 wall-time cap reached during episode")
                            if environment_step < cfg.num_steps_wait:
                                observation, _, _, _ = current_env.step(
                                    get_libero_dummy_action(cfg.model_family)
                                )
                                environment_step += 1
                                continue
                            if not action_queue:
                                policy_observation, _ = upstream_eval.prepare_observation(
                                    observation, resize_size
                                )
                                raw_state = tuple(
                                    float(value)
                                    for value in np.asarray(policy_observation["state"]).reshape(-1)
                                )
                                normalized_state = normalize_bounds(raw_state, q01, q99)
                                if len(raw_state) != 8 or len(normalized_state) != 8:
                                    raise RuntimeError("A4 proprioception width changed")
                                scene = policy_observation["full_image"]
                                wrist = policy_observation["wrist_image"]
                                representation = prepare_scene_representation(scene)
                                position = (
                                    normalized_state[0],
                                    normalized_state[1],
                                    normalized_state[2],
                                )
                                patch_scores = scene_score = translation_score = None
                                if previous_scene is not None:
                                    change = scene_change_from_representations(
                                        representation, previous_scene
                                    )
                                    patch_scores, scene_score = (
                                        change.patch_scores,
                                        change.top_four_mean,
                                    )
                                if previous_position is not None:
                                    translation_score = scene_relative_translation(
                                        position, previous_position
                                    )
                                decision_transition = (
                                    transition_signal(previous_actions, older_actions)
                                    if previous_actions is not None and older_actions is not None
                                    else None
                                )
                                timer.start()
                                actions = np.asarray(
                                    upstream_eval.get_action(
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
                                )
                                timing = timer.finish()
                                assert_upstream_query(timing)
                                if actions.shape != (8, 7) or not np.isfinite(actions).all():
                                    raise RuntimeError(
                                        f"Malformed A4 action chunk: {actions.shape}"
                                    )
                                trace_transition = (
                                    transition_signal(actions, previous_actions)
                                    if previous_actions is not None
                                    else None
                                )
                                context_hash = value_sha256(
                                    {
                                        "task": task_id,
                                        "state": state_id,
                                        "instruction": task_description,
                                        "checkpoint": CHECKPOINT_REVISION,
                                        "configuration": config_sha256,
                                    }
                                )
                                query_record = build_query_record(
                                    identity=identity,
                                    query_index=query_index,
                                    environment_step=environment_step,
                                    global_query_index=global_query_index,
                                    scene_image=scene,
                                    wrist_image=wrist,
                                    raw_state=raw_state,
                                    normalized_state=normalized_state,
                                    scene_representation=representation,
                                    patch_scores=patch_scores,
                                    scene_score=scene_score,
                                    translation_score=translation_score,
                                    gripper_transition_veto=(
                                        decision_transition.gripper_veto
                                        if decision_transition
                                        else None
                                    ),
                                    direction_reversal=(
                                        any(decision_transition.translation_direction_reversals)
                                        if decision_transition
                                        else None
                                    ),
                                    actions=actions,
                                    timing=timing,
                                    model_dtype=str(model_parameter.dtype),
                                    model_device=str(model_parameter.device),
                                    visual_token_count=visual_token_count,
                                    configuration_sha256=config_sha256,
                                    savr_revision=savr_revision,
                                    context_sha256=context_hash,
                                    audit_sha256=audit_sha256,
                                )
                                validate_record(query_record, query_schema)
                                trace_record = build_trace_record(
                                    identity=identity,
                                    query_index=query_index,
                                    scene_representation=representation,
                                    normalized_eef_position=position,
                                    actions=actions,
                                    gripper_transition_veto=(
                                        trace_transition.gripper_veto if trace_transition else None
                                    ),
                                    direction_reversals=(
                                        trace_transition.translation_direction_reversals
                                        if trace_transition
                                        else None
                                    ),
                                    upstream_component_invocations={
                                        name: int(timing.component_counts.get(name, 0))
                                        for name in (
                                            "vision_backbone",
                                            "visual_projector",
                                            "language_model",
                                            "action_head",
                                        )
                                    },
                                    encode_float_sequence=encode_float_sequence,
                                    np=np,
                                )
                                store.write_once(identity.query_id(query_index), query_record)
                                store.write_once(
                                    f"{identity.query_id(query_index)}/trace", trace_record
                                )
                                query_records.append(query_record)
                                trace_records.append(trace_record)
                                query_wall.append(timing.wall_ms)
                                visual_cuda.append(query_record["timing"]["total_visual_cuda_ms"])
                                if global_query_index not in STEADY_EXCLUSIONS:
                                    steady_query_wall.append(timing.wall_ms)
                                    steady_visual_cuda.append(
                                        query_record["timing"]["total_visual_cuda_ms"]
                                    )
                                previous_scene, previous_position = representation, position
                                older_actions, previous_actions = (
                                    previous_actions,
                                    actions.copy(),
                                )
                                query_index += 1
                                global_query_index += 1
                                action_queue.extend(actions)
                            action = upstream_eval.process_action(
                                action_queue.popleft(), cfg.model_family
                            )
                            if not np.isfinite(action).all():
                                raise RuntimeError("Processed A4 action is non-finite")
                            observation, _, done, _ = current_env.step(action.tolist())
                            control_steps += 1
                            trajectory.update(np.asarray(action, dtype="<f8").tobytes())
                            trajectory.update(
                                np.asarray(
                                    raw_robot_state(observation, np, quat2axisangle), dtype="<f8"
                                ).tobytes()
                            )
                            if done:
                                success = True
                                break
                            environment_step += 1
                    except BaseException as error:
                        episode_error = error
                    counts = {
                        "environment_steps": control_steps,
                        "queries": query_index,
                        "scene_refreshes": query_index,
                        "scene_reuses": 0,
                        "wrist_refreshes": query_index,
                        "scene_siglip_calls": query_index,
                        "scene_dinov2_calls": query_index,
                        "scene_projector_calls": query_index,
                        "wrist_siglip_calls": query_index,
                        "wrist_dinov2_calls": query_index,
                        "wrist_projector_calls": query_index,
                        "downstream_calls": query_index,
                    }
                    reconcile_episode_counts(counts)
                    scientific_failure = episode_error is None and not success
                    episode_record = {
                        "schema_version": "acr.episode.v1",
                        "run_id": RUN_ID,
                        "attempt_id": identity.value,
                        "episode_id": identity.episode_id,
                        "phase": PHASE,
                        "policy": POLICY,
                        "suite": SUITE,
                        "task_id": task_id,
                        "initial_state_id": state_id,
                        "seed": SEED,
                        "status": "failed" if episode_error is not None else "completed",
                        "success": None if episode_error is not None else success,
                        "failure_classification": (
                            "operator"
                            if isinstance(episode_error, Interrupted)
                            else "technical"
                            if episode_error is not None
                            else "scientific"
                            if scientific_failure
                            else None
                        ),
                        "failure_reason": (
                            str(episode_error)
                            if episode_error is not None
                            else "task_not_completed_within_horizon"
                            if scientific_failure
                            else None
                        ),
                        "counts": counts,
                        "timing": {
                            "episode_wall_ms": (time.perf_counter() - episode_started) * 1000,
                            "inclusive_query_wall_ms": query_wall,
                            "steady_query_wall_ms": steady_query_wall,
                            "inclusive_visual_cuda_ms": visual_cuda,
                            "steady_visual_cuda_ms": steady_visual_cuda,
                        },
                        "memory": {
                            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                            "scene_cache_bytes": 0,
                        },
                        "records_sha256": value_sha256(
                            {"queries": query_records, "traces": trace_records}
                        ),
                        "trajectory_sha256": trajectory.hexdigest(),
                        "configuration_sha256": config_sha256,
                        "started_at_utc": episode_started_at,
                        "finished_at_utc": utc_now(),
                    }
                    validate_record(episode_record, episode_schema)
                    store.write_once(identity.episode_id, episode_record)
                    episode_ids.append(identity.episode_id)
                    if episode_error is not None:
                        raise episode_error
                    successes += int(success)
                    per_task_success[str(task_id)] += int(success)
                    if directory_size(run_dir) >= ARTIFACT_CAP_BYTES:
                        raise RuntimeError("A4 artifact cap reached")
            finally:
                current_env.close()
                current_env = None
        if attempts_started != ATTEMPT_CAP or len(episode_ids) != ATTEMPT_CAP:
            raise RuntimeError("A4 terminal matrix did not reconcile to 100 attempts")
        terminal_status = "completed"
    except BaseException as error:
        caught = error
        terminal_status = "interrupted" if isinstance(error, Interrupted) else "failed"
    finally:
        if current_env is not None:
            current_env.close()
        if hooks is not None:
            hooks.remove()
        checkpoint_after: dict[str, Any] | None = None
        upstream_after = libero_after = None
        gpu_after: dict[str, Any] | None = None
        restoration_error: BaseException | None = None
        try:
            for name, content in protected_bytes.items():
                (checkpoint / name).write_bytes(content)
            new_checkpoint_files = {
                item.name for item in checkpoint.iterdir()
            } - checkpoint_files_before
            unexpected = sorted(name for name in new_checkpoint_files if ".back." not in name)
            for name in sorted(new_checkpoint_files):
                if ".back." in name:
                    (checkpoint / name).unlink()
            if unexpected:
                raise RuntimeError(f"Unexpected new checkpoint files after A4: {unexpected}")
            if any(
                (checkpoint / name).read_bytes() != content
                for name, content in protected_bytes.items()
            ):
                raise RuntimeError("A4 checkpoint protected bytes were not restored")
            checkpoint_after = validate_checkpoint(project_root, checkpoint)
            upstream_after = git_revision(upstream_root, OPENVLA_REVISION)
            libero_after = git_revision(libero_root, LIBERO_REVISION)
            gpu_after = selected_gpu_snapshot(physical_gpu_id)
            if gpu_after["uuid"] != selected_uuid:
                raise RuntimeError("Selected GPU UUID changed during A4")
        except BaseException as error:
            restoration_error = error
        elapsed_seconds = time.monotonic() - run_started
        artifact_bytes = directory_size(run_dir)
        if restoration_error is not None:
            caught = caught or restoration_error
            terminal_status = "failed"
        if elapsed_seconds > WALL_CAP_SECONDS:
            caught = caught or RuntimeError("A4 wall-time cap exceeded")
            terminal_status = "failed"
        if artifact_bytes > ARTIFACT_CAP_BYTES:
            caught = caught or RuntimeError("A4 artifact cap exceeded")
            terminal_status = "failed"
        finished_at = utc_now()
        completion = {
            **manifest,
            "status": terminal_status,
            "finished_at_utc": finished_at,
            "records_sha256": value_sha256(episode_ids),
        }
        validate_record(completion, run_schema)
        store.write_once("completion", completion)
        summary = {
            "schema_version": "acr.a4-run-summary.v1",
            "run_id": RUN_ID,
            "status": terminal_status,
            "attempts_started": attempts_started,
            "terminal_episodes": len(episode_ids),
            "successes": successes,
            "per_task_successes": per_task_success,
            "queries": global_query_index,
            "elapsed_seconds": elapsed_seconds,
            "artifact_bytes": artifact_bytes,
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "checkpoint_before": checkpoint_before,
            "checkpoint_after": checkpoint_after,
            "upstream_revision_after": upstream_after,
            "libero_revision_after": libero_after,
            "error_type": type(caught).__name__ if caught else None,
            "error": str(caught) if caught else None,
            "restoration_error": str(restoration_error) if restoration_error else None,
            "finished_at_utc": finished_at,
        }
        store.write_once("summary", summary)
        if model is not None:
            del model
        torch.cuda.empty_cache()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if caught is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
