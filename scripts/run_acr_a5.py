#!/usr/bin/env python3
"""Run one frozen ACR A5 candidate/stage without automatic recovery."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_ROOT = Path("/home/ved/SAVR")
PHASE = "A5"
POLICY = "sa-acr"
SUITE = "libero_object"
TASK_IDS = tuple(range(10))
SEED = 0
ATTEMPT_INDEX = 1
PHASE_ATTEMPT_CAP = 300
PHASE_WALL_CAP_SECONDS = 86_400
PHASE_ARTIFACT_CAP_BYTES = 2 * 1024**3
CHECKPOINT_REVISION = "638918f3d1c2e43a39a8a20772bdb8b91835e4b7"
OPENVLA_REVISION = "e4287e94541f459edc4feabc4e181f537cd569a8"
LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
CHECKPOINT_RELATIVE = Path("checkpoints/openvla-7b-oft-libero-four-suite")
CONFIG_RELATIVE = Path("configs/acr/development_a5.json")
CANDIDATE_RELATIVE = Path("configs/acr/candidates.json")
STEADY_EXCLUSIONS = frozenset({0, 1, 2})
CANDIDATE_IDS = (
    "acr-t25-h2-b30",
    "acr-t50-h4-b55",
    "acr-t70-h8-b75",
)


class Interrupted(RuntimeError):
    """Raised when the bounded runner receives a termination request."""


class ResourceCap(RuntimeError):
    """Raised before a frozen cumulative A5 resource cap is crossed."""


class InvariantFailure(RuntimeError):
    """Raised when camera accounting or immutable-record truth fails."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def phase_artifact_bytes(project_root: Path) -> int:
    return sum(
        directory_size(path)
        for path in (project_root / "results").glob("acr-a5-*")
        if path.is_dir()
    )


def completed_phase_usage(project_root: Path) -> tuple[int, float]:
    attempts = 0
    wall_seconds = 0.0
    for path in sorted((project_root / "results").glob("acr-a5-*/summary/record.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        attempts += int(record.get("attempts_started", 0))
        wall_seconds += float(record.get("elapsed_seconds", 0.0))
    return attempts, wall_seconds


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


def run_id_for(stage: str, candidate_id: str) -> str:
    if stage not in {"stage1", "stage2"} or candidate_id not in CANDIDATE_IDS:
        raise ValueError("Unsupported A5 stage/candidate")
    return f"acr-a5-sa-acr-object-{stage}-{candidate_id}-v01"


def validate_frozen_config(config: dict[str, Any], candidate_payload: dict[str, Any]) -> None:
    if (
        config.get("schema_version") != "acr.a5-development.v1"
        or config.get("phase") != PHASE
        or config.get("policy") != POLICY
        or config.get("suite") != SUITE
        or config.get("task_ids") != list(TASK_IDS)
        or config.get("seed") != SEED
    ):
        raise ValueError("Frozen A5 identity changed")
    source = config.get("candidate_source", {})
    if (
        source.get("path") != "configs/acr/candidates.json"
        or source.get("file_sha256")
        != "8f1503e4f579df9a0a4026b178492566b4ad206830886dfba2418852574567a4"
        or source.get("semantic_sha256")
        != "1cced910aec61f7666c43b90abb12c3da3b804abeb11ca3ea84fb01bc7b279ea"
        or candidate_payload.get("semantic_sha256") != source.get("semantic_sha256")
    ):
        raise ValueError("Frozen A4 candidate source changed")
    expected = [
        ("acr-t25-h2-b30", 0.2476380718954248, 0.5479944908411765, 2, 0.3),
        ("acr-t50-h4-b55", 0.30046895424836606, 0.685919037527938, 4, 0.55),
        ("acr-t70-h8-b75", 0.30046895424836606, 0.685919037527938, 8, 0.75),
    ]
    observed = [
        (
            item.get("configuration_id"),
            item.get("scene_threshold"),
            item.get("translation_threshold"),
            item.get("horizon"),
            item.get("hard_reuse_cap"),
        )
        for item in config.get("eligible_candidates", [])
    ]
    if observed != expected:
        raise ValueError("Frozen A5 candidates changed")
    if any(
        item.get("controller_version") != "acr-controller-v1"
        for item in config.get("eligible_candidates", [])
    ):
        raise ValueError("Frozen A5 controller version changed")
    payload_observed = [
        (
            item.get("configuration_id"),
            item.get("scene_threshold"),
            item.get("translation_threshold"),
            item.get("horizon"),
            item.get("hard_reuse_cap"),
            item.get("status"),
        )
        for item in candidate_payload.get("candidates", [])
    ]
    if payload_observed != [(*item, "DERIVATION_ELIGIBLE") for item in expected]:
        raise ValueError("Canonical A4 candidate payload differs from A5 freeze")
    stages = config.get("stages", {})
    if (
        stages.get("stage1", {}).get("initial_state_ids") != [0, 1, 2]
        or stages.get("stage1", {}).get("attempts_per_candidate") != 30
        or stages.get("stage1", {}).get("run_id_template")
        != "acr-a5-sa-acr-object-stage1-<candidate>-v01"
        or stages.get("stage1", {}).get("advance_gate")
        != {
            "terminal_episodes": 30,
            "successes": 30,
            "per_task_successes": 3,
            "minimum_scene_reuse": 0.15,
            "technical_failures": 0,
        }
        or stages.get("stage2", {}).get("initial_state_ids") != list(range(3, 10))
        or stages.get("stage2", {}).get("attempts_per_candidate") != 70
        or stages.get("stage2", {}).get("run_id_template")
        != "acr-a5-sa-acr-object-stage2-<candidate>-v01"
        or stages.get("stage2", {}).get("eligibility_gate")
        != {
            "maximum_success_loss_vs_fr": 2,
            "maximum_per_task_success_loss_vs_fr": 1,
            "minimum_scene_reuse": 0.4,
            "minimum_visual_cuda_reduction": 0.1,
            "technical_failures": 0,
        }
    ):
        raise ValueError("Frozen A5 stage populations changed")
    if config.get("selection_order") != [
        "highest_paired_success_difference",
        "within_one_percentage_point_highest_query_latency_reduction",
        "highest_visual_cuda_reduction",
        "lowest_horizon",
        "lexicographic_configuration_id",
    ]:
        raise ValueError("Frozen A5 selection order changed")
    if config.get("analysis") != {
        "success_pairing": ["suite", "task_id", "initial_state_id", "seed"],
        "steady_visual_cuda_point": "sum_steady_visual_cuda_ms_divided_by_steady_query_count",
        "steady_query_latency_point": "sum_steady_query_wall_ms_divided_by_steady_query_count",
        "reduction": "one_minus_acr_point_divided_by_upstream_fr_point",
        "fr_source_run_id": "acr-a4-upstream-fr-object-dev00-09-v01",
    }:
        raise ValueError("Frozen A5 analysis semantics changed")
    if config.get("resource_caps") != {
        "gpu_count": 1,
        "model_processes": 1,
        "episode_attempts": PHASE_ATTEMPT_CAP,
        "wall_seconds": PHASE_WALL_CAP_SECONDS,
        "artifact_bytes": PHASE_ARTIFACT_CAP_BYTES,
        "downloads_allowed": False,
    }:
        raise ValueError("Frozen A5 resource caps changed")
    if config.get("timing", {}).get(
        "exclude_global_query_indices_from_steady_state_per_run"
    ) != sorted(STEADY_EXCLUSIONS):
        raise ValueError("Frozen A5 timing exclusions changed")
    if config.get("recovery") != {
        "mode": "preserve-and-restart",
        "automatic_episode_retry": False,
        "resume_incomplete_episode": False,
        "first_attempt_index": ATTEMPT_INDEX,
    }:
        raise ValueError("Frozen A5 recovery semantics changed")
    model = config.get("model", {})
    expected_model = {
        "checkpoint_revision": CHECKPOINT_REVISION,
        "openvla_oft_revision": OPENVLA_REVISION,
        "libero_revision": LIBERO_REVISION,
        "num_open_loop_steps": 8,
        "num_images_in_input": 2,
        "use_proprio": True,
        "use_l1_regression": True,
        "use_diffusion": False,
        "use_film": False,
        "center_crop": True,
    }
    if model != expected_model:
        raise ValueError("Frozen A5 model configuration changed")


def require_stage2_eligibility(project_root: Path, candidate_id: str) -> None:
    path = project_root / "results/acr-a5-stage1-analysis-v01/record/record.json"
    if not path.is_file():
        raise RuntimeError("Stage 2 requires the immutable Stage 1 analysis")
    analysis = json.loads(path.read_text(encoding="utf-8"))
    semantic = dict(analysis)
    claimed = semantic.pop("semantic_sha256", None)
    if claimed != value_sha256(semantic):
        raise RuntimeError("Stage 1 analysis semantic hash changed")
    if (
        analysis.get("schema_version") != "acr.a5-stage1-analysis.v1"
        or analysis.get("disposition") != "ADVANCE_TO_STAGE2"
        or analysis.get("candidate_source_sha256") != file_sha256(project_root / CANDIDATE_RELATIVE)
        or analysis.get("a5_configuration_sha256") != file_sha256(project_root / CONFIG_RELATIVE)
        or analysis.get("analyzer_sha256")
        != file_sha256(project_root / "scripts/analyze_acr_a5.py")
    ):
        raise RuntimeError("Stage 1 analysis provenance changed")
    if candidate_id not in analysis.get("advancing_candidates", []):
        raise RuntimeError(f"Candidate {candidate_id} did not pass Stage 1")
    candidate_result = next(
        (item for item in analysis.get("results", []) if item.get("candidate_id") == candidate_id),
        None,
    )
    if candidate_result is None or candidate_result.get("passed") is not True:
        raise RuntimeError(f"Candidate {candidate_id} lacks a passing Stage 1 record")


def array_sha256(value: Any, np: Any) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def query_record(
    *,
    run_id: str,
    identity: Any,
    result: Any,
    configuration: Any,
    context: Any,
    raw_state: tuple[float, ...],
    normalized_state: tuple[float, ...],
    actions: Any,
    visual_token_count: int,
    model_dtype: str,
    model_device: str,
    configuration_sha256: str,
    savr_revision: str,
    environment_step: int,
    np: Any,
) -> dict[str, Any]:
    decision = result.decision
    timing = result.device_timing
    if timing is None:
        raise InvariantFailure("A5 requires synchronized CUDA timing")
    component_cuda = dict(timing.component_device_ms)
    scene_cuda = sum(
        component_cuda.get(name, 0.0)
        for name in ("scene.siglip", "scene.dinov2", "scene.projector")
    )
    wrist_cuda = sum(
        component_cuda.get(name, 0.0)
        for name in ("wrist.siglip", "wrist.dinov2", "wrist.projector")
    )
    visual_cuda = scene_cuda + wrist_cuda
    cache_concat_wall = sum(
        value
        for name, value in result.work.component_wall_ms.items()
        if "cache" in name or "concat" in name
    )
    record = {
        "schema_version": "acr.query.v1",
        "run_id": run_id,
        "attempt_id": identity.value,
        "query_id": identity.query_id(decision.query_index),
        "phase": PHASE,
        "policy": POLICY,
        "suite": SUITE,
        "task_id": identity.task_id,
        "initial_state_id": identity.initial_state_id,
        "seed": identity.seed,
        "query_index": decision.query_index,
        "environment_step": environment_step,
        "status": "completed",
        "error": None,
        "decision": {
            "scene_refresh": decision.refresh,
            "refresh_reasons": list(decision.reasons),
            "cache_age_before": decision.cache_age_before,
            "cache_age_after": 0 if decision.refresh else (decision.cache_age_before or 0) + 1,
            "reference_query_index": decision.reference_query_index,
            "scene_score": decision.scene_score,
            "scene_threshold": configuration.scene_threshold,
            "translation_score": decision.translation_score,
            "translation_threshold": configuration.translation_threshold,
            "gripper_transition_veto": decision.gripper_transition_veto,
            "horizon": configuration.horizon,
            "reuse_count_before": decision.completed_reuses_before,
            "query_count_before": decision.completed_queries_before,
            "hard_reuse_cap": configuration.hard_reuse_cap,
        },
        "inputs": {
            "scene_image_sha256": result.scene_image_sha256,
            "wrist_image_sha256": result.wrist_image_sha256,
            "proprio_sha256": result.proprio_sha256,
            "action_sha256": array_sha256(actions, np),
            "context_sha256": value_sha256(asdict(context)),
            "scene_representation": None,
            "scene_patch_scores": list(decision.patch_scores) if decision.patch_scores else None,
            "proprio_raw": list(raw_state),
            "proprio_normalized": list(normalized_state),
            "direction_reversal": any(decision.translation_direction_reversals),
        },
        "camera_work": {
            "scene_siglip_calls": result.work.scene_siglip_calls,
            "scene_dinov2_calls": result.work.scene_dinov2_calls,
            "scene_projector_calls": result.work.scene_projector_calls,
            "wrist_siglip_calls": result.work.wrist_siglip_calls,
            "wrist_dinov2_calls": result.work.wrist_dinov2_calls,
            "wrist_projector_calls": result.work.wrist_projector_calls,
            "visual_token_count": visual_token_count,
            "token_order": "scene-wrist",
            "dtype": model_dtype,
            "device": model_device,
            "downstream_calls": result.work.downstream_calls,
        },
        "timing": {
            "inclusive": True,
            "controller_wall_ms": result.controller_wall_ms,
            "cache_concat_wall_ms": cache_concat_wall,
            "scene_visual_cuda_ms": scene_cuda,
            "wrist_visual_cuda_ms": wrist_cuda,
            "total_visual_cuda_ms": visual_cuda,
            "downstream_cuda_ms": max(0.0, timing.total_device_ms - visual_cuda),
            "query_cuda_ms": timing.total_device_ms,
            "query_wall_ms": result.query_wall_ms,
        },
        "provenance": {
            "configuration_sha256": configuration_sha256,
            "savr_revision": savr_revision,
            "openvla_oft_revision": OPENVLA_REVISION,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "recorded_at_utc": utc_now(),
        },
    }
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    parser.add_argument("--candidate", choices=CANDIDATE_IDS, required=True)
    arguments = parser.parse_args()
    stage, candidate_id = arguments.stage, arguments.candidate
    project_root = Path(__file__).resolve().parents[1]
    if project_root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing to run outside {EXPECTED_ROOT}: {project_root}")
    physical_gpu_id = os.environ.get("SAVR_PHYSICAL_GPU_ID")
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    selected_uuid = os.environ.get("SAVR_SELECTED_GPU_UUID")
    if not physical_gpu_id or visible_gpu != physical_gpu_id or not selected_uuid:
        raise SystemExit("Selected GPU ID/UUID variables are incomplete or inconsistent")

    def handle_signal(_signum: int, _frame: Any) -> None:
        raise Interrupted("ACR A5 runner received a termination signal")

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

    config_path = project_root / CONFIG_RELATIVE
    candidate_path = project_root / CANDIDATE_RELATIVE
    config = json.loads(config_path.read_text(encoding="utf-8"))
    candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    if file_sha256(candidate_path) != config["candidate_source"]["file_sha256"]:
        raise RuntimeError("Canonical candidate file hash changed")
    validate_frozen_config(config, candidate_payload)
    if stage == "stage2":
        require_stage2_eligibility(project_root, candidate_id)
    stage_config = config["stages"][stage]
    state_ids = tuple(int(value) for value in stage_config["initial_state_ids"])
    run_attempt_cap = len(TASK_IDS) * len(state_ids)
    run_id = run_id_for(stage, candidate_id)
    candidate_record = next(
        item for item in config["eligible_candidates"] if item["configuration_id"] == candidate_id
    )
    run_configuration_sha256 = value_sha256(
        {
            "a5_config_sha256": file_sha256(config_path),
            "candidate_source_sha256": file_sha256(candidate_path),
            "candidate": candidate_record,
            "stage": stage,
            "state_ids": list(state_ids),
        }
    )
    prior_attempts, prior_wall_seconds = completed_phase_usage(project_root)
    if prior_attempts + run_attempt_cap > PHASE_ATTEMPT_CAP:
        raise ResourceCap("A5 attempt cap would be exceeded by this run")
    if prior_wall_seconds >= PHASE_WALL_CAP_SECONDS:
        raise ResourceCap("A5 wall cap is already exhausted")
    if phase_artifact_bytes(project_root) >= PHASE_ARTIFACT_CAP_BYTES:
        raise ResourceCap("A5 artifact cap is already exhausted")

    upstream_root = project_root / "third_party/openvla-oft"
    libero_root = project_root / "third_party/LIBERO"
    checkpoint = project_root / CHECKPOINT_RELATIVE
    run_dir = project_root / "results" / run_id
    if run_dir.exists():
        raise SystemExit(f"Immutable A5 run already exists: {run_dir}")

    sys.path.insert(0, str(project_root / "src"))
    sys.path.insert(0, str(project_root / "scripts"))
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
    from run_phase5_core_smoke import raw_robot_state
    from savr.acr.controller import ACRController
    from savr.acr.instrumentation import CameraInstrumentation
    from savr.acr.openvla_oft import OpenVLAAsymmetricCameraAdapter, TorchTensorOperations
    from savr.acr.records import (
        AttemptIdentity,
        ImmutableRecordStore,
        reconcile_episode_counts,
        validate_record,
    )
    from savr.acr.types import ACRConfiguration, ACRContext, ACRPolicy
    from savr.signals import normalize_bounds
    from savr.timing import SynchronizedQueryTimer, TorchCudaEventBackend

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
    gpu_before = selected_gpu_snapshot(physical_gpu_id)
    if gpu_before["uuid"] != selected_uuid:
        raise RuntimeError("Selected GPU UUID changed between approval and launch")

    run_dir.mkdir(parents=True, exist_ok=False)
    store = ImmutableRecordStore(run_dir)
    planned = [
        AttemptIdentity(run_id, POLICY, SUITE.replace("_", "-"), task, state, SEED, ATTEMPT_INDEX)
        for task in TASK_IDS
        for state in state_ids
    ]
    started_at = utc_now()
    run_started = time.monotonic()
    manifest = {
        "schema_version": "acr.run.v1",
        "run_id": run_id,
        "phase": PHASE,
        "policy": POLICY,
        "suite": SUITE,
        "scope": f"LIBERO-Object tasks 0-9 states {state_ids[0]}-{state_ids[-1]} seed 0 {candidate_id}",
        "status": "running",
        "configuration_sha256": run_configuration_sha256,
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
            "initial_state_ids": list(state_ids),
            "seed": SEED,
        },
        "resource_caps": {
            "gpu_count": 1,
            "model_processes": 1,
            "query_attempts": None,
            "episode_attempts": PHASE_ATTEMPT_CAP,
            "wall_seconds": PHASE_WALL_CAP_SECONDS,
            "artifact_bytes": PHASE_ARTIFACT_CAP_BYTES,
            "downloads_allowed": False,
        },
        "planned_attempts": [identity.value for identity in planned],
        "recovery": {
            "mode": "preserve-and-restart",
            "overwrite_allowed": False,
            "resume_incomplete_episode": False,
            "next_attempt_index": ATTEMPT_INDEX + 1,
        },
        "artifact_root": str(run_dir),
        "command": f"scripts/run_acr_a5.py --stage {stage} --candidate {candidate_id}",
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
        num_trials_per_task=len(state_ids),
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
    configuration = ACRConfiguration(
        candidate_id,
        ACRPolicy.SA_ACR,
        scene_threshold=float(candidate_record["scene_threshold"]),
        translation_threshold=float(candidate_record["translation_threshold"]),
        horizon=int(candidate_record["horizon"]),
        hard_reuse_cap=float(candidate_record["hard_reuse_cap"]),
    )
    set_seed_everywhere(SEED)
    model = action_head = proprio_projector = noisy_action_projector = processor = None
    adapter = current_env = None
    terminal_status = "failed"
    caught: BaseException | None = None
    attempts_started = 0
    global_query_index = 0
    episode_ids: list[str] = []
    successes = 0
    per_task_success = {str(task): 0 for task in TASK_IDS}
    aggregate_counts = {
        name: 0
        for name in (
            "queries",
            "scene_refreshes",
            "scene_reuses",
            "wrist_refreshes",
            "scene_siglip_calls",
            "scene_dinov2_calls",
            "scene_projector_calls",
            "wrist_siglip_calls",
            "wrist_dinov2_calls",
            "wrist_projector_calls",
            "downstream_calls",
        )
    }
    try:
        os.chdir(upstream_root)
        model, action_head, proprio_projector, noisy_action_projector, processor = (
            upstream_eval.initialize_model(cfg)
        )
        torch.cuda.synchronize()
        if action_head is None or proprio_projector is None or processor is None:
            raise RuntimeError("Pinned L1/proprio/processor modules were not loaded")
        statistics = model.norm_stats[cfg.unnorm_key]["proprio"]
        q01, q99 = statistics["q01"], statistics["q99"]
        model_parameter = next(model.parameters())
        model_dtype, model_device = str(model_parameter.dtype), str(model_parameter.device)
        patch_count = int(model.vision_backbone.get_num_patches())
        visual_token_count = patch_count * 2
        resize_size = get_image_resize_size(cfg)
        task_suite = benchmark.get_benchmark_dict()[SUITE]()
        for task_id in TASK_IDS:
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            current_env, task_description = get_libero_env(
                task, cfg.model_family, resolution=cfg.env_img_res
            )
            try:
                for state_id in state_ids:
                    elapsed = prior_wall_seconds + (time.monotonic() - run_started)
                    if prior_attempts + attempts_started >= PHASE_ATTEMPT_CAP:
                        raise ResourceCap("A5 episode-attempt cap exhausted")
                    if elapsed >= PHASE_WALL_CAP_SECONDS:
                        raise ResourceCap("A5 wall-time cap reached before scheduling")
                    if phase_artifact_bytes(project_root) >= PHASE_ARTIFACT_CAP_BYTES:
                        raise ResourceCap("A5 artifact cap reached before scheduling")
                    identity = AttemptIdentity(
                        run_id,
                        POLICY,
                        SUITE.replace("_", "-"),
                        task_id,
                        state_id,
                        SEED,
                        ATTEMPT_INDEX,
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
                    query_wall: list[float] = []
                    visual_cuda: list[float] = []
                    steady_query_wall: list[float] = []
                    steady_visual_cuda: list[float] = []
                    trajectory = hashlib.sha256()
                    episode_counts = {name: 0 for name in aggregate_counts}
                    peak_scene_cache_bytes = 0
                    episode_error: BaseException | None = None
                    context = ACRContext(
                        episode_id=identity.episode_id,
                        attempt_id=identity.value,
                        task_id=f"{task_id:02d}",
                        instruction_sha256=hashlib.sha256(task_description.encode()).hexdigest(),
                        checkpoint_id=CHECKPOINT_REVISION,
                        upstream_revision=OPENVLA_REVISION,
                        configuration_id=candidate_id,
                        controller_version=configuration.controller_version,
                        preprocessing_id="openvla-center-crop-v1",
                        action_head_id="l1-regression-8x7",
                        dtype=model_dtype,
                        device=model_device,
                        patch_count=patch_count,
                    )
                    adapter = OpenVLAAsymmetricCameraAdapter(
                        model=model,
                        controller=ACRController(configuration),
                        tensor_ops=TorchTensorOperations(torch),
                        instrumentation=CameraInstrumentation(
                            timer=SynchronizedQueryTimer(TorchCudaEventBackend(torch))
                        ),
                    )
                    adapter.begin_context(context)
                    try:
                        max_steps = upstream_eval.TASK_MAX_STEPS[cfg.task_suite_name]
                        while environment_step < max_steps + cfg.num_steps_wait:
                            elapsed = prior_wall_seconds + (time.monotonic() - run_started)
                            if elapsed >= PHASE_WALL_CAP_SECONDS:
                                raise ResourceCap("A5 wall-time cap reached during episode")
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
                                    raise InvariantFailure("A5 proprioception width changed")
                                scene, wrist = (
                                    policy_observation["full_image"],
                                    policy_observation["wrist_image"],
                                )

                                def invoke() -> Any:
                                    return upstream_eval.get_action(
                                        cfg,
                                        model,
                                        policy_observation,
                                        task_description,
                                        processor=processor,
                                        action_head=action_head,
                                        proprio_projector=proprio_projector,
                                        noisy_action_projector=noisy_action_projector,
                                        use_film=False,
                                    )

                                result = adapter.run_query(
                                    query=invoke,
                                    scene_image=scene,
                                    wrist_image=wrist,
                                    state=raw_state,
                                    state_q01=q01,
                                    state_q99=q99,
                                )
                                actions = np.asarray(result.value)
                                if actions.shape != (8, 7) or not np.isfinite(actions).all():
                                    raise InvariantFailure(
                                        f"Malformed A5 action chunk: {actions.shape}"
                                    )
                                result.work.validate(scene_refresh=result.decision.refresh)
                                record = query_record(
                                    run_id=run_id,
                                    identity=identity,
                                    result=result,
                                    configuration=configuration,
                                    context=context,
                                    raw_state=raw_state,
                                    normalized_state=normalized_state,
                                    actions=actions,
                                    visual_token_count=visual_token_count,
                                    model_dtype=model_dtype,
                                    model_device=model_device,
                                    configuration_sha256=run_configuration_sha256,
                                    savr_revision=savr_revision,
                                    environment_step=environment_step,
                                    np=np,
                                )
                                validate_record(record, query_schema)
                                store.write_once(identity.query_id(query_index), record)
                                query_records.append(record)
                                query_wall.append(float(record["timing"]["query_wall_ms"]))
                                visual_cuda.append(float(record["timing"]["total_visual_cuda_ms"]))
                                if global_query_index not in STEADY_EXCLUSIONS:
                                    steady_query_wall.append(query_wall[-1])
                                    steady_visual_cuda.append(visual_cuda[-1])
                                work = record["camera_work"]
                                episode_counts["queries"] += 1
                                episode_counts["scene_refreshes"] += int(result.decision.refresh)
                                episode_counts["scene_reuses"] += int(not result.decision.refresh)
                                episode_counts["wrist_refreshes"] += 1
                                for name in (
                                    "scene_siglip_calls",
                                    "scene_dinov2_calls",
                                    "scene_projector_calls",
                                    "wrist_siglip_calls",
                                    "wrist_dinov2_calls",
                                    "wrist_projector_calls",
                                    "downstream_calls",
                                ):
                                    episode_counts[name] += int(work[name])
                                reconcile_episode_counts(episode_counts)
                                if adapter.cache.entry is not None:
                                    tokens = adapter.cache.entry.tokens
                                    peak_scene_cache_bytes = max(
                                        peak_scene_cache_bytes,
                                        int(tokens.numel() * tokens.element_size()),
                                    )
                                query_index += 1
                                global_query_index += 1
                                action_queue.extend(actions)
                            action = upstream_eval.process_action(
                                action_queue.popleft(), cfg.model_family
                            )
                            if not np.isfinite(action).all():
                                raise InvariantFailure("Processed A5 action is non-finite")
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
                    counts = {"environment_steps": control_steps, **episode_counts}
                    reconcile_episode_counts(counts)
                    scientific_failure = episode_error is None and not success
                    classification = (
                        "operator"
                        if isinstance(episode_error, Interrupted)
                        else "resource"
                        if isinstance(episode_error, ResourceCap)
                        else "invariant"
                        if isinstance(episode_error, InvariantFailure)
                        else "technical"
                        if episode_error is not None
                        else "scientific"
                        if scientific_failure
                        else None
                    )
                    episode_record = {
                        "schema_version": "acr.episode.v1",
                        "run_id": run_id,
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
                        "failure_classification": classification,
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
                            "scene_cache_bytes": peak_scene_cache_bytes,
                        },
                        "records_sha256": value_sha256({"queries": query_records}),
                        "trajectory_sha256": trajectory.hexdigest(),
                        "configuration_sha256": run_configuration_sha256,
                        "started_at_utc": episode_started_at,
                        "finished_at_utc": utc_now(),
                    }
                    validate_record(episode_record, episode_schema)
                    store.write_once(identity.episode_id, episode_record)
                    episode_ids.append(identity.episode_id)
                    for name in aggregate_counts:
                        aggregate_counts[name] += episode_counts[name]
                    if episode_error is not None:
                        raise episode_error
                    successes += int(success)
                    per_task_success[str(task_id)] += int(success)
                    if phase_artifact_bytes(project_root) >= PHASE_ARTIFACT_CAP_BYTES:
                        raise ResourceCap("A5 artifact cap reached")
            finally:
                current_env.close()
                current_env = None
        if attempts_started != run_attempt_cap or len(episode_ids) != run_attempt_cap:
            raise InvariantFailure("A5 terminal matrix did not reconcile")
        reconcile_episode_counts(aggregate_counts)
        terminal_status = "completed"
    except BaseException as error:
        caught = error
        terminal_status = "interrupted" if isinstance(error, Interrupted) else "failed"
    finally:
        if current_env is not None:
            current_env.close()
        adapter = None
        model = action_head = proprio_projector = noisy_action_projector = processor = None
        gc.collect()
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
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
                raise RuntimeError(f"Unexpected new checkpoint files after A5: {unexpected}")
            if any(
                (checkpoint / name).read_bytes() != content
                for name, content in protected_bytes.items()
            ):
                raise RuntimeError("A5 checkpoint protected bytes were not restored")
            checkpoint_after = validate_checkpoint(project_root, checkpoint)
            upstream_after = git_revision(upstream_root, OPENVLA_REVISION)
            libero_after = git_revision(libero_root, LIBERO_REVISION)
            gpu_after = selected_gpu_snapshot(physical_gpu_id)
            if gpu_after["uuid"] != selected_uuid:
                raise RuntimeError("Selected GPU UUID changed during A5")
        except BaseException as error:
            restoration_error = error
        elapsed_seconds = time.monotonic() - run_started
        cumulative_wall = prior_wall_seconds + elapsed_seconds
        artifact_bytes = directory_size(run_dir)
        cumulative_artifact_bytes = phase_artifact_bytes(project_root)
        if restoration_error is not None:
            caught = caught or restoration_error
            terminal_status = "failed"
        if cumulative_wall > PHASE_WALL_CAP_SECONDS:
            caught = caught or ResourceCap("A5 cumulative wall-time cap exceeded")
            terminal_status = "failed"
        if cumulative_artifact_bytes > PHASE_ARTIFACT_CAP_BYTES:
            caught = caught or ResourceCap("A5 cumulative artifact cap exceeded")
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
            "schema_version": "acr.a5-run-summary.v1",
            "run_id": run_id,
            "stage": stage,
            "candidate_id": candidate_id,
            "status": terminal_status,
            "attempts_started": attempts_started,
            "terminal_episodes": len(episode_ids),
            "successes": successes,
            "per_task_successes": per_task_success,
            "counts": aggregate_counts,
            "scene_reuse_rate": (
                aggregate_counts["scene_reuses"] / aggregate_counts["queries"]
                if aggregate_counts["queries"]
                else 0.0
            ),
            "elapsed_seconds": elapsed_seconds,
            "cumulative_wall_seconds": cumulative_wall,
            "artifact_bytes": artifact_bytes,
            "cumulative_artifact_bytes": cumulative_artifact_bytes,
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "checkpoint_before": checkpoint_before,
            "checkpoint_after": checkpoint_after,
            "upstream_revision_after": upstream_after,
            "libero_revision_after": libero_after,
            "configuration_sha256": run_configuration_sha256,
            "error_type": type(caught).__name__ if caught else None,
            "error": str(caught) if caught else None,
            "restoration_error": str(restoration_error) if restoration_error else None,
            "finished_at_utc": finished_at,
        }
        store.write_once("summary", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if caught is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
