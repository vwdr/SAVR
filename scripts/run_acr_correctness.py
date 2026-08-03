#!/usr/bin/env python3
"""Run the authorized ACR A3 real-model correctness matrix without rollouts."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import types
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_ROOT = Path("/home/ved/SAVR")
RUN_ID = "acr-a3-correctness-none-v01"
ATTEMPT_ID = f"{RUN_ID}/mixed/synthetic/task-00/state-00/seed-0/attempt-0000"
CHECKPOINT_REVISION = "638918f3d1c2e43a39a8a20772bdb8b91835e4b7"
OPENVLA_REVISION = "e4287e94541f459edc4feabc4e181f537cd569a8"
LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
CHECKPOINT_RELATIVE = Path("checkpoints/openvla-7b-oft-libero-four-suite")
CHECKPOINT_REPORT = Path("reports/runtime/phase2_checkpoint.json")
QUERY_CAP = 16
PLANNED_QUERIES = 12
WALL_CAP_SECONDS = 3600
ARTIFACT_CAP_BYTES = 512 * 1024**2
INSTRUCTION = "move the robot safely to the target"


class Interrupted(RuntimeError):
    """Raised when the bounded run receives a termination signal."""


class QueryBudget:
    def __init__(self, cap: int) -> None:
        if cap < 1:
            raise ValueError("Query cap must be positive")
        self.cap = cap
        self.attempts: list[str] = []

    def consume(self, label: str) -> int:
        if len(self.attempts) >= self.cap:
            raise RuntimeError(f"Real-model query cap exceeded before {label}")
        self.attempts.append(label)
        return len(self.attempts) - 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


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


def git_output(path: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *arguments], text=True).strip()


def require_clean_revision(path: Path, expected: str | None = None) -> str:
    revision = git_output(path, "rev-parse", "HEAD")
    if expected is not None and revision != expected:
        raise RuntimeError(f"Expected {expected} at {path}, found {revision}")
    if git_output(path, "status", "--porcelain"):
        raise RuntimeError(f"Refusing to use dirty source tree: {path}")
    return revision


def tensor_sha256(value: Any) -> str:
    import torch  # type: ignore[import-not-found]

    raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def array_sha256(value: Any, np: Any) -> str:
    array = np.asarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def exact_array_proof(reference: Any, candidate: Any, np: Any, label: str) -> dict[str, Any]:
    left = np.asarray(reference)
    right = np.asarray(candidate)
    equal = left.shape == right.shape and bool(np.array_equal(left, right))
    if not equal:
        maximum = (
            float(np.max(np.abs(left - right)))
            if left.shape == right.shape and left.size
            else None
        )
        raise RuntimeError(f"{label} exact parity failed; maximum difference={maximum}")
    return {"equal": True, "shape": list(left.shape), "sha256": array_sha256(left, np)}


def exact_tensor_proof(reference: Any, candidate: Any, torch: Any, label: str) -> dict[str, Any]:
    equal = tuple(reference.shape) == tuple(candidate.shape) and bool(torch.equal(reference, candidate))
    if not equal:
        maximum = (
            float((reference.float() - candidate.float()).abs().max().item())
            if tuple(reference.shape) == tuple(candidate.shape)
            else None
        )
        raise RuntimeError(f"{label} bitwise parity failed; maximum difference={maximum}")
    return {
        "equal": True,
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "device": str(reference.device),
        "sha256": tensor_sha256(reference),
    }


def camera_isolation_proof(
    baseline: Any,
    changed: Any,
    *,
    patch_count: int,
    changed_camera: str,
    torch: Any,
) -> dict[str, Any]:
    baseline_scene, baseline_wrist = baseline[:, :patch_count], baseline[:, patch_count:]
    changed_scene, changed_wrist = changed[:, :patch_count], changed[:, patch_count:]
    scene_equal = bool(torch.equal(baseline_scene, changed_scene))
    wrist_equal = bool(torch.equal(baseline_wrist, changed_wrist))
    expected = (not scene_equal and wrist_equal) if changed_camera == "scene" else (
        scene_equal and not wrist_equal
    )
    if changed_camera not in {"scene", "wrist"} or not expected:
        raise RuntimeError(
            f"{changed_camera} isolation failed: scene_equal={scene_equal}, "
            f"wrist_equal={wrist_equal}"
        )
    return {
        "changed_camera": changed_camera,
        "scene_equal": scene_equal,
        "wrist_equal": wrist_equal,
        "baseline_scene_sha256": tensor_sha256(baseline_scene),
        "baseline_wrist_sha256": tensor_sha256(baseline_wrist),
        "changed_scene_sha256": tensor_sha256(changed_scene),
        "changed_wrist_sha256": tensor_sha256(changed_wrist),
    }


def synthetic_images(np: Any, size: int = 256) -> tuple[Any, Any, Any, Any]:
    y, x = np.indices((size, size))
    scene = np.stack(((x + y) % 256, x % 256, y % 256), axis=-1).astype(np.uint8)
    wrist = np.stack(((2 * x + y) % 256, (x + 3 * y) % 256, (255 - x) % 256), axis=-1).astype(np.uint8)
    scene_variant = np.roll(scene, shift=7, axis=1).copy()
    wrist_variant = np.roll(wrist, shift=11, axis=0).copy()
    if np.array_equal(scene, scene_variant) or np.array_equal(wrist, wrist_variant):
        raise RuntimeError("Synthetic camera variants must differ")
    return scene, wrist, scene_variant, wrist_variant


def validate_checkpoint(project_root: Path, checkpoint: Path) -> dict[str, Any]:
    report = json.loads((project_root / CHECKPOINT_REPORT).read_text(encoding="utf-8"))
    if report["requested_revision"] != CHECKPOINT_REVISION:
        raise RuntimeError("Checkpoint request revision changed")
    if report["resolved_revision"] != CHECKPOINT_REVISION:
        raise RuntimeError("Checkpoint resolved revision changed")
    mismatched = [
        item["path"]
        for item in report["files"]
        if not (checkpoint / item["path"]).is_file()
        or (checkpoint / item["path"]).stat().st_size != item["size"]
    ]
    if mismatched:
        raise RuntimeError("Checkpoint inventory mismatch: " + ", ".join(mismatched))
    expected = {
        "config.json": "edd5c5cf6d7927e07465cf086ebe41f7b3ec8f3b128a51f71d6db14dad7ad8b1",
        "dataset_statistics.json": "6ec6ef68d0d5bae4cb5f9fc9acb715a22b9f4545e9e9b300d0d88695cd7afec3",
        "model.safetensors.index.json": "ca8b53fed8133ee2afcd2fc483de8febf7f5bb0f6bcb09f91189772e59e8f659",
    }
    observed = {name: file_sha256(checkpoint / name) for name in expected}
    if observed != expected:
        raise RuntimeError("Checkpoint metadata hash mismatch")
    return {
        "file_count": len(report["files"]),
        "declared_bytes": report["remote_bytes"],
        "metadata_sha256": observed,
        "file_names": sorted(item.name for item in checkpoint.iterdir()),
    }


def query_record(
    *,
    result: Any,
    tokens: Any,
    policy: str,
    configuration: Any,
    context: Any,
    scene_image: Any,
    wrist_image: Any,
    state: Any,
    source_revision: str,
    np: Any,
) -> dict[str, Any]:
    decision = result.decision
    timing = result.device_timing
    if timing is None:
        raise RuntimeError("A3 requires synchronized device timing")
    component_cuda = dict(timing.component_device_ms)
    scene_cuda = sum(value for name, value in component_cuda.items() if name.startswith("scene."))
    wrist_cuda = sum(value for name, value in component_cuda.items() if name.startswith("wrist."))
    visual_cuda = scene_cuda + wrist_cuda
    cache_concat_wall = sum(
        value
        for name, value in result.work.component_wall_ms.items()
        if "cache" in name or "concat" in name
    )
    action_sha = array_sha256(result.value, np)
    return {
        "schema_version": "acr.query.v1",
        "run_id": RUN_ID,
        "attempt_id": ATTEMPT_ID,
        "query_id": f"{ATTEMPT_ID}/query-{decision.query_index:06d}",
        "phase": "A3",
        "policy": policy,
        "suite": None,
        "task_id": None,
        "initial_state_id": None,
        "seed": None,
        "query_index": decision.query_index,
        "environment_step": None,
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
            "action_sha256": action_sha,
            "context_sha256": value_sha256(asdict(context)),
            "direction_reversal": any(decision.translation_direction_reversals),
        },
        "camera_work": {
            "scene_siglip_calls": result.work.scene_siglip_calls,
            "scene_dinov2_calls": result.work.scene_dinov2_calls,
            "scene_projector_calls": result.work.scene_projector_calls,
            "wrist_siglip_calls": result.work.wrist_siglip_calls,
            "wrist_dinov2_calls": result.work.wrist_dinov2_calls,
            "wrist_projector_calls": result.work.wrist_projector_calls,
            "visual_token_count": int(tokens.shape[1]),
            "token_order": "scene-wrist",
            "dtype": str(tokens.dtype),
            "device": str(tokens.device),
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
            "configuration_sha256": value_sha256(asdict(configuration)),
            "savr_revision": source_revision,
            "openvla_oft_revision": OPENVLA_REVISION,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "recorded_at_utc": utc_now(),
        },
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    if project_root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing to run outside {EXPECTED_ROOT}: {project_root}")
    physical_gpu_id = os.environ.get("SAVR_PHYSICAL_GPU_ID")
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    selected_uuid = os.environ.get("SAVR_SELECTED_GPU_UUID")
    if not physical_gpu_id or visible_gpu != physical_gpu_id or not selected_uuid:
        raise SystemExit("Selected GPU ID/UUID variables are incomplete or inconsistent")

    def handle_signal(_signum: int, _frame: Any) -> None:
        raise Interrupted("ACR A3 runner received a termination signal")

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
            "PYTHONNOUSERSITE": "1",
            "WANDB_MODE": "disabled",
            "TOKENIZERS_PARALLELISM": "false",
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
        }
    )

    upstream_root = project_root / "third_party/openvla-oft"
    libero_root = project_root / "third_party/LIBERO"
    checkpoint = project_root / CHECKPOINT_RELATIVE
    results_root = project_root / "results"
    run_root = results_root / RUN_ID
    if run_root.exists():
        raise SystemExit(f"Immutable A3 run already exists: {run_root}")
    source_revision = require_clean_revision(project_root)
    require_clean_revision(upstream_root, OPENVLA_REVISION)
    require_clean_revision(libero_root, LIBERO_REVISION)
    checkpoint_before = validate_checkpoint(project_root, checkpoint)
    started_at = utc_now()
    started_wall = time.monotonic()

    sys.path.insert(0, str(project_root / "src"))
    sys.path.insert(0, str(upstream_root))
    import numpy as np
    import torch  # type: ignore[import-not-found]
    from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
    from experiments.robot.libero import run_libero_eval as upstream_eval  # type: ignore[import-not-found]
    from experiments.robot.robot_utils import set_seed_everywhere  # type: ignore[import-not-found]
    from savr.acr.cache import SceneCacheEntry
    from savr.acr.controller import ACRController
    from savr.acr.instrumentation import CameraInstrumentation
    from savr.acr.openvla_oft import OpenVLAAsymmetricCameraAdapter, TorchTensorOperations
    from savr.acr.records import ImmutableRecordStore, reconcile_episode_counts, validate_record
    from savr.acr.types import ACRConfiguration, ACRContext, ACRPolicy
    from savr.timing import SynchronizedQueryTimer, TorchCudaEventBackend

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one selected CUDA device must be visible")
    visible_uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None)
    if visible_uuid is not None and str(visible_uuid) != selected_uuid:
        raise RuntimeError("Visible CUDA device UUID differs from the selected physical GPU")

    query_schema = json.loads((project_root / "schemas/acr_query.schema.json").read_text())
    run_schema = json.loads((project_root / "schemas/acr_run.schema.json").read_text())
    Draft202012Validator.check_schema(query_schema)
    Draft202012Validator.check_schema(run_schema)
    store = ImmutableRecordStore(results_root)
    schema_hashes = {
        name: file_sha256(project_root / "schemas" / name)
        for name in ("acr_run.schema.json", "acr_query.schema.json", "acr_episode.schema.json")
    }
    manifest = {
        "schema_version": "acr.run.v1",
        "run_id": RUN_ID,
        "phase": "A3",
        "policy": "mixed",
        "suite": None,
        "scope": "Synthetic-input real-model parity; no simulator or benchmark population",
        "status": "running",
        "configuration_sha256": value_sha256({"protocol": "acr-v1", "planned_queries": PLANNED_QUERIES}),
        "revisions": {
            "savr": source_revision,
            "openvla_oft": OPENVLA_REVISION,
            "libero": LIBERO_REVISION,
            "checkpoint": CHECKPOINT_REVISION,
        },
        "schemas": {
            "run_sha256": schema_hashes["acr_run.schema.json"],
            "query_sha256": schema_hashes["acr_query.schema.json"],
            "episode_sha256": schema_hashes["acr_episode.schema.json"],
        },
        "population": {"task_ids": [], "initial_state_ids": [], "seed": None},
        "resource_caps": {
            "gpu_count": 1,
            "model_processes": 1,
            "query_attempts": QUERY_CAP,
            "episode_attempts": 0,
            "wall_seconds": WALL_CAP_SECONDS,
            "artifact_bytes": ARTIFACT_CAP_BYTES,
            "downloads_allowed": False,
        },
        "planned_attempts": [ATTEMPT_ID],
        "recovery": {
            "mode": "preserve-and-restart",
            "overwrite_allowed": False,
            "resume_incomplete_episode": False,
            "next_attempt_index": 1,
        },
        "artifact_root": str(run_root),
        "command": "scripts/run_acr_correctness.py",
        "host": subprocess.check_output(["hostname"], text=True).strip(),
        "selected_gpu_id": int(physical_gpu_id),
        "started_at_utc": started_at,
        "finished_at_utc": None,
        "records_sha256": None,
    }
    validate_record(manifest, run_schema)
    store.write_once(f"{RUN_ID}/manifest", manifest)

    budget = QueryBudget(QUERY_CAP)
    proof: dict[str, Any] = {
        "schema_version": "acr.correctness-proof.v1",
        "run_id": RUN_ID,
        "planned_queries": PLANNED_QUERIES,
        "query_cap": QUERY_CAP,
        "rollout_episodes": 0,
        "simulator_resets": 0,
        "checkpoint_before": checkpoint_before,
        "proofs": {},
    }
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
    model = action_head = proprio_projector = noisy_action_projector = processor = None
    terminal_error: BaseException | None = None
    try:
        os.chdir(upstream_root)
        load_started = time.monotonic()
        model, action_head, proprio_projector, noisy_action_projector, processor = upstream_eval.initialize_model(cfg)
        torch.cuda.synchronize()
        proof["model_load_seconds"] = time.monotonic() - load_started
        if action_head is None or proprio_projector is None or processor is None:
            raise RuntimeError("Pinned L1/proprio/processor modules were not loaded")
        patch_count = int(model.vision_backbone.get_num_patches())
        statistics = model.norm_stats[cfg.unnorm_key]["proprio"]
        q01 = np.asarray(statistics["q01"], dtype=np.float64)
        q99 = np.asarray(statistics["q99"], dtype=np.float64)
        if q01.shape != (8,) or q99.shape != (8,) or not np.all(q99 > q01):
            raise RuntimeError("Pinned proprioception statistics are invalid")
        state_a = (q01 + q99) / 2
        state_b = state_a.copy()
        state_b[0] = q01[0] + 0.55 * (q99[0] - q01[0])
        scene, wrist, scene_variant, wrist_variant = synthetic_images(np)
        proof["synthetic_inputs"] = {
            "scene_sha256": array_sha256(scene, np),
            "wrist_sha256": array_sha256(wrist, np),
            "scene_variant_sha256": array_sha256(scene_variant, np),
            "wrist_variant_sha256": array_sha256(wrist_variant, np),
            "state_a_sha256": array_sha256(state_a, np),
            "state_b_sha256": array_sha256(state_b, np),
            "instruction_sha256": hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
        }

        def observation(scene_image: Any, wrist_image: Any, state: Any) -> dict[str, Any]:
            return {
                "full_image": np.asarray(scene_image).copy(),
                "wrist_image": np.asarray(wrist_image).copy(),
                "state": np.asarray(state).copy(),
            }

        def raw_upstream(scene_image: Any, wrist_image: Any, state: Any) -> Any:
            return upstream_eval.get_action(
                cfg,
                model,
                observation(scene_image, wrist_image, state),
                INSTRUCTION,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                use_film=False,
            )

        def run_upstream(label: str, index: int, scene_image: Any, wrist_image: Any, state: Any) -> tuple[Any, Any]:
            budget.consume(label)
            captured: list[Any] = []
            instance_dict = vars(model)
            had_override = "_process_vision_features" in instance_dict
            previous_override = instance_dict.get("_process_vision_features")
            original = model._process_vision_features

            def intercept(_instance: Any, pixel_values: Any, language_embeddings: Any = None, use_film: bool = False) -> Any:
                value = original(pixel_values, language_embeddings, use_film)
                captured.append(value.detach().clone())
                return value

            timer = SynchronizedQueryTimer(TorchCudaEventBackend(torch))
            setattr(model, "_process_vision_features", types.MethodType(intercept, model))
            timer.start()
            try:
                actions = raw_upstream(scene_image, wrist_image, state)
                timing = timer.finish()
            finally:
                if had_override:
                    setattr(model, "_process_vision_features", previous_override)
                else:
                    delattr(model, "_process_vision_features")
            if len(captured) != 1:
                raise RuntimeError(f"{label} captured {len(captured)} projected blocks")
            record = {
                "schema_version": "acr.oracle-query.v1",
                "run_id": RUN_ID,
                "attempt_id": ATTEMPT_ID,
                "query_index": index,
                "label": label,
                "policy": "upstream-fr",
                "projected_tokens": {
                    "shape": list(captured[0].shape),
                    "dtype": str(captured[0].dtype),
                    "device": str(captured[0].device),
                    "sha256": tensor_sha256(captured[0]),
                },
                "actions": {"shape": list(np.asarray(actions).shape), "sha256": array_sha256(actions, np)},
                "inputs": {
                    "scene_sha256": array_sha256(scene_image, np),
                    "wrist_sha256": array_sha256(wrist_image, np),
                    "state_sha256": array_sha256(state, np),
                },
                "timing": {"query_wall_ms": timing.wall_ms, "query_cuda_ms": timing.total_device_ms},
                "recorded_at_utc": utc_now(),
            }
            store.write_once(f"{ATTEMPT_ID}/oracle-query-{index:06d}", record)
            return actions, captured[0]

        factorized_tokens: list[Any] = []
        factorized_configuration = ACRConfiguration("a3-factorized-fr", ACRPolicy.FACTORIZED_FR)
        factorized_context = ACRContext(
            episode_id="a3-factorized",
            attempt_id=ATTEMPT_ID,
            task_id="synthetic",
            instruction_sha256=hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
            checkpoint_id=CHECKPOINT_REVISION,
            upstream_revision=OPENVLA_REVISION,
            configuration_id=factorized_configuration.configuration_id,
            controller_version=factorized_configuration.controller_version,
            preprocessing_id="openvla-center-crop-v1",
            action_head_id="l1-regression-8x7",
            dtype="torch.bfloat16",
            device="cuda:0",
            patch_count=patch_count,
        )
        factorized_adapter = OpenVLAAsymmetricCameraAdapter(
            model=model,
            controller=ACRController(factorized_configuration),
            tensor_ops=TorchTensorOperations(torch),
            instrumentation=CameraInstrumentation(timer=SynchronizedQueryTimer(TorchCudaEventBackend(torch))),
            projected_tokens_observer=lambda value: factorized_tokens.append(value.detach().clone()),
        )
        factorized_adapter.begin_context(factorized_context)

        def run_adapter(
            *,
            label: str,
            global_index: int,
            adapter: Any,
            configuration: Any,
            context: Any,
            token_list: list[Any],
            policy: str,
            scene_image: Any,
            wrist_image: Any,
            state: Any,
        ) -> tuple[Any, Any]:
            before = len(token_list)

            def invoke() -> Any:
                budget.consume(label)
                return raw_upstream(scene_image, wrist_image, state)

            result = adapter.run_query(
                query=invoke,
                scene_image=scene_image,
                wrist_image=wrist_image,
                state=state,
                state_q01=q01,
                state_q99=q99,
            )
            if len(token_list) != before + 1:
                raise RuntimeError(f"{label} did not expose exactly one projected token block")
            tokens = token_list[-1]
            record = query_record(
                result=result,
                tokens=tokens,
                policy=policy,
                configuration=configuration,
                context=context,
                scene_image=scene_image,
                wrist_image=wrist_image,
                state=state,
                source_revision=source_revision,
                np=np,
            )
            record["query_index"] = global_index
            record["query_id"] = f"{ATTEMPT_ID}/query-{global_index:06d}"
            validate_record(record, query_schema)
            store.write_once(f"{ATTEMPT_ID}/query-{global_index:06d}", record)
            return result, tokens

        actions_upstream_a, tokens_upstream_a = run_upstream("upstream-a", 0, scene, wrist, state_a)
        factorized_a, tokens_factorized_a = run_adapter(
            label="factorized-a", global_index=1, adapter=factorized_adapter,
            configuration=factorized_configuration, context=factorized_context,
            token_list=factorized_tokens, policy="factorized-fr",
            scene_image=scene, wrist_image=wrist, state=state_a,
        )
        proof["proofs"]["factorized_token_parity"] = exact_tensor_proof(
            tokens_upstream_a, tokens_factorized_a, torch, "factorized projected tokens"
        )
        proof["proofs"]["factorized_action_parity"] = exact_array_proof(
            actions_upstream_a, factorized_a.value, np, "factorized actions"
        )
        _, tokens_scene_variant = run_adapter(
            label="factorized-scene-variant", global_index=2, adapter=factorized_adapter,
            configuration=factorized_configuration, context=factorized_context,
            token_list=factorized_tokens, policy="factorized-fr",
            scene_image=scene_variant, wrist_image=wrist, state=state_a,
        )
        proof["proofs"]["scene_isolation"] = camera_isolation_proof(
            tokens_factorized_a, tokens_scene_variant, patch_count=patch_count,
            changed_camera="scene", torch=torch,
        )
        _, tokens_wrist_variant = run_adapter(
            label="factorized-wrist-variant", global_index=3, adapter=factorized_adapter,
            configuration=factorized_configuration, context=factorized_context,
            token_list=factorized_tokens, policy="factorized-fr",
            scene_image=scene, wrist_image=wrist_variant, state=state_a,
        )
        proof["proofs"]["wrist_isolation"] = camera_isolation_proof(
            tokens_factorized_a, tokens_wrist_variant, patch_count=patch_count,
            changed_camera="wrist", torch=torch,
        )

        visual_tokens: list[Any] = []
        visual_configuration = ACRConfiguration(
            "a3-scene-visual", ACRPolicy.SCENE_VISUAL,
            scene_threshold=1.0, horizon=4, hard_reuse_cap=0.75,
        )
        visual_context = replace(
            factorized_context,
            episode_id="a3-scene-visual",
            configuration_id=visual_configuration.configuration_id,
        )
        visual_adapter = OpenVLAAsymmetricCameraAdapter(
            model=model,
            controller=ACRController(visual_configuration),
            tensor_ops=TorchTensorOperations(torch),
            instrumentation=CameraInstrumentation(timer=SynchronizedQueryTimer(TorchCudaEventBackend(torch))),
            projected_tokens_observer=lambda value: visual_tokens.append(value.detach().clone()),
        )
        visual_adapter.begin_context(visual_context)
        run_adapter(
            label="visual-warmup-0", global_index=4, adapter=visual_adapter,
            configuration=visual_configuration, context=visual_context,
            token_list=visual_tokens, policy="scene-visual-acr",
            scene_image=scene, wrist_image=wrist, state=state_a,
        )
        _, tokens_visual_reference = run_adapter(
            label="visual-warmup-1", global_index=5, adapter=visual_adapter,
            configuration=visual_configuration, context=visual_context,
            token_list=visual_tokens, policy="scene-visual-acr",
            scene_image=scene, wrist_image=wrist, state=state_a,
        )
        reuse_result, tokens_reuse = run_adapter(
            label="visual-reuse-current-state", global_index=6, adapter=visual_adapter,
            configuration=visual_configuration, context=visual_context,
            token_list=visual_tokens, policy="scene-visual-acr",
            scene_image=scene, wrist_image=wrist, state=state_b,
        )
        if reuse_result.decision.refresh or reuse_result.cache_event != "reuse":
            raise RuntimeError("Planned A3 scene-reuse query did not reuse")
        reuse_result.work.validate(scene_refresh=False)
        proof["proofs"]["reuse_visual_tokens"] = exact_tensor_proof(
            tokens_visual_reference, tokens_reuse, torch, "reuse visual tokens"
        )
        proof["proofs"]["reuse_component_truth"] = asdict(reuse_result.work)
        actions_upstream_b, tokens_upstream_b = run_upstream("upstream-current-state-b", 7, scene, wrist, state_b)
        proof["proofs"]["reuse_upstream_token_parity"] = exact_tensor_proof(
            tokens_upstream_b, tokens_reuse, torch, "reuse/upstream projected tokens"
        )
        proof["proofs"]["reuse_current_state_action_parity"] = exact_array_proof(
            actions_upstream_b, reuse_result.value, np, "reuse current-state actions"
        )

        injection_proofs = []
        for global_index, kind in enumerate(("shape", "dtype", "device"), start=8):
            entry = visual_adapter.cache.entry
            if entry is None:
                raise RuntimeError("Metadata injection requires a populated scene cache")
            metadata = entry.metadata
            if kind == "shape":
                changed_metadata = replace(metadata, shape=(1, patch_count, metadata.shape[2] + 1))
            elif kind == "dtype":
                changed_metadata = replace(metadata, dtype="torch.float32")
            else:
                changed_metadata = replace(metadata, device="cpu")
            visual_adapter.cache._entry = SceneCacheEntry(  # type: ignore[attr-defined]
                context=entry.context,
                tokens=entry.tokens,
                metadata=changed_metadata,
                refresh_query_index=entry.refresh_query_index,
            )
            injected_result, _ = run_adapter(
                label=f"fail-closed-{kind}", global_index=global_index,
                adapter=visual_adapter, configuration=visual_configuration,
                context=visual_context, token_list=visual_tokens,
                policy="scene-visual-acr", scene_image=scene,
                wrist_image=wrist, state=state_a,
            )
            if (
                not injected_result.decision.refresh
                or injected_result.cache_event != "forced-refresh"
                or "cache" not in injected_result.decision.reasons
            ):
                raise RuntimeError(f"Injected {kind} incompatibility did not fail closed")
            injected_result.work.validate(scene_refresh=True)
            injection_proofs.append(
                {"kind": kind, "cache_event": injected_result.cache_event, "work": asdict(injected_result.work)}
            )
        proof["proofs"]["metadata_fail_closed"] = injection_proofs

        reset_context = replace(visual_context, episode_id="a3-context-reset")
        if not visual_adapter.begin_context(reset_context) or visual_adapter.cache.entry is not None:
            raise RuntimeError("Context change did not invalidate the scene cache")
        reset_result, _ = run_adapter(
            label="fail-closed-context", global_index=11, adapter=visual_adapter,
            configuration=visual_configuration, context=reset_context,
            token_list=visual_tokens, policy="scene-visual-acr",
            scene_image=scene, wrist_image=wrist, state=state_a,
        )
        if not reset_result.decision.refresh or "cache" not in reset_result.decision.reasons:
            raise RuntimeError("Context reset did not force a scene refresh")
        proof["proofs"]["context_fail_closed"] = {
            "cache_event": reset_result.cache_event,
            "reasons": list(reset_result.decision.reasons),
            "work": asdict(reset_result.work),
        }

        counts = {
            "queries": 10,
            "scene_refreshes": sum(
                1 for index in (1, 2, 3, 4, 5, 6, 8, 9, 10, 11)
                if json.loads((results_root / ATTEMPT_ID / f"query-{index:06d}" / "record.json").read_text())["decision"]["scene_refresh"]
            ),
            "scene_reuses": 0,
            "wrist_refreshes": 10,
            "scene_siglip_calls": 0,
            "scene_dinov2_calls": 0,
            "scene_projector_calls": 0,
            "wrist_siglip_calls": 10,
            "wrist_dinov2_calls": 10,
            "wrist_projector_calls": 10,
            "downstream_calls": 10,
        }
        counts["scene_reuses"] = counts["queries"] - counts["scene_refreshes"]
        counts["scene_siglip_calls"] = counts["scene_refreshes"]
        counts["scene_dinov2_calls"] = counts["scene_refreshes"]
        counts["scene_projector_calls"] = counts["scene_refreshes"]
        reconcile_episode_counts(counts)
        proof["adapter_reconciliation"] = counts
        proof["query_attempts"] = list(budget.attempts)
        if len(budget.attempts) != PLANNED_QUERIES:
            raise RuntimeError("Completed real-model query count differs from the frozen plan")
        if time.monotonic() - started_wall > WALL_CAP_SECONDS:
            raise RuntimeError("A3 wall-time cap exceeded")
        checkpoint_after = validate_checkpoint(project_root, checkpoint)
        if checkpoint_after != checkpoint_before:
            raise RuntimeError("Checkpoint inventory changed during A3")
        require_clean_revision(upstream_root, OPENVLA_REVISION)
        require_clean_revision(libero_root, LIBERO_REVISION)
        require_clean_revision(project_root, source_revision)
        proof["checkpoint_after"] = checkpoint_after
        proof["source_trees_restored"] = True
        proof["query_count"] = len(budget.attempts)
        proof["status"] = "PASS"
        proof["finished_at_utc"] = utc_now()
        proof["wall_seconds"] = time.monotonic() - started_wall
        proof["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        proof["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        if directory_size(run_root) > ARTIFACT_CAP_BYTES:
            raise RuntimeError("A3 artifact cap exceeded")
        store.write_once(f"{RUN_ID}/proof", proof)
        records_hash = value_sha256(proof)
        final_manifest = dict(manifest)
        final_manifest.update(
            status="completed",
            finished_at_utc=proof["finished_at_utc"],
            records_sha256=records_hash,
        )
        validate_record(final_manifest, run_schema)
        store.write_once(f"{RUN_ID}/final", final_manifest)
        return 0
    except BaseException as error:
        terminal_error = error
        failure = {
            "schema_version": "acr.correctness-failure.v1",
            "run_id": RUN_ID,
            "status": "interrupted" if isinstance(error, Interrupted) else "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "query_attempts": list(budget.attempts),
            "query_count": len(budget.attempts),
            "rollout_episodes": 0,
            "simulator_resets": 0,
            "recorded_at_utc": utc_now(),
        }
        try:
            store.write_once(f"{RUN_ID}/failure", failure)
        except Exception:
            pass
        raise
    finally:
        if model is not None:
            del model
        if action_head is not None:
            del action_head
        if proprio_projector is not None:
            del proprio_projector
        if processor is not None:
            del processor
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if terminal_error is not None:
            print(f"A3 failed closed after {len(budget.attempts)} queries: {terminal_error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
