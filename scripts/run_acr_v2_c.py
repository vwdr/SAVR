#!/usr/bin/env python3
"""Run the frozen ACR V2-C real-model correctness and latency gate."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import types
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_ROOT = Path("/home/ved/SAVR")
RUN_ID = "acr-v2c-correctness-latency-v01"
CONFIG_RELATIVE = Path("configs/acr/v2_c_gate.json")
CHECKPOINT_RELATIVE = Path("checkpoints/openvla-7b-oft-libero-four-suite")
CHECKPOINT_REVISION = "638918f3d1c2e43a39a8a20772bdb8b91835e4b7"
OPENVLA_REVISION = "e4287e94541f459edc4feabc4e181f537cd569a8"
LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
INSTRUCTION = "move the robot safely to the target"
QUERY_CAP = 48
CORRECTNESS_QUERIES = 6
WARMUP_QUERIES = 6
TIMED_QUERIES = 36
WALL_CAP_SECONDS = 3600
ARTIFACT_CAP_BYTES = 512 * 1024**2
REUSE_WEIGHT = 0.26055045871559634
PATHS = ("upstream-fr", "dual-path-refresh", "dual-path-reuse")
COUNTERBALANCE = (
    PATHS,
    ("dual-path-refresh", "dual-path-reuse", "upstream-fr"),
    ("dual-path-reuse", "upstream-fr", "dual-path-refresh"),
)


class ExpectedRestorationProbe(RuntimeError):
    """Expected downstream exception used to prove restoration."""


class QueryBudget:
    def __init__(self, cap: int = QUERY_CAP) -> None:
        self.cap = cap
        self.labels: list[str] = []

    def consume(self, label: str) -> int:
        if len(self.labels) >= self.cap:
            raise RuntimeError(f"V2-C query cap exceeded before {label}")
        self.labels.append(label)
        return len(self.labels) - 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def git_output(path: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *arguments], text=True).strip()


def require_clean_revision(path: Path, expected: str | None = None) -> str:
    revision = git_output(path, "rev-parse", "HEAD")
    if expected is not None and revision != expected:
        raise RuntimeError(f"Expected revision {expected} at {path}, found {revision}")
    if git_output(path, "status", "--porcelain"):
        raise RuntimeError(f"Refusing to use dirty source tree: {path}")
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
        raise RuntimeError("Selected GPU snapshot did not resolve one device")
    values = [item.strip() for item in rows[0].split(",")]
    if len(values) != 6 or values[0] != physical_gpu_id:
        raise RuntimeError("Selected GPU identity is inconsistent")
    return {
        "index": int(values[0]),
        "uuid": values[1],
        "name": values[2],
        "memory_total_mib": int(values[3]),
        "memory_used_mib": int(values[4]),
        "utilization_percent": int(values[5]),
        "recorded_at_utc": utc_now(),
    }


def tensor_sha256(value: Any) -> str:
    import torch  # type: ignore[import-not-found]

    raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def array_sha256(value: Any, np: Any) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def exact_tensor(reference: Any, candidate: Any, torch: Any, label: str) -> dict[str, Any]:
    equal = tuple(reference.shape) == tuple(candidate.shape) and bool(
        torch.equal(reference, candidate)
    )
    if not equal:
        maximum = None
        if tuple(reference.shape) == tuple(candidate.shape):
            maximum = float((reference.float() - candidate.float()).abs().max().item())
        raise RuntimeError(f"{label} bitwise parity failed; maximum difference={maximum}")
    return {
        "equal": True,
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "sha256": tensor_sha256(reference),
    }


def exact_array(reference: Any, candidate: Any, np: Any, label: str) -> dict[str, Any]:
    left, right = np.asarray(reference), np.asarray(candidate)
    if left.shape != right.shape or not bool(np.array_equal(left, right)):
        maximum = None
        if left.shape == right.shape and left.size:
            maximum = float(np.max(np.abs(left - right)))
        raise RuntimeError(f"{label} exact parity failed; maximum difference={maximum}")
    return {"equal": True, "shape": list(left.shape), "sha256": array_sha256(left, np)}


def synthetic_images(np: Any, size: int = 256) -> tuple[Any, Any]:
    y, x = np.indices((size, size))
    scene = np.stack(((x + y) % 256, x % 256, y % 256), axis=-1).astype(np.uint8)
    wrist = np.stack(((2 * x + y) % 256, (x + 3 * y) % 256, (255 - x) % 256), axis=-1).astype(
        np.uint8
    )
    return scene, wrist


def validate_gate_config(config: dict[str, Any]) -> None:
    supplied = config.get("semantic_sha256")
    payload = dict(config)
    payload.pop("semantic_sha256", None)
    if semantic_sha256(payload) != supplied:
        raise RuntimeError("V2-C configuration semantic hash mismatch")
    caps = config["resource_caps"]
    if caps != {
        "gpu_count": 1,
        "model_processes": 1,
        "model_queries": 48,
        "rollout_episodes": 0,
        "simulator_resets": 0,
        "wall_seconds": 3600,
        "artifact_bytes": 536870912,
        "downloads_allowed": False,
    }:
        raise RuntimeError("V2-C resource caps changed")
    if len(config["correctness_queries"]) != CORRECTNESS_QUERIES:
        raise RuntimeError("V2-C correctness schedule changed")
    timing = config["timing"]
    if timing["untimed_warmups_per_path"] * len(PATHS) != WARMUP_QUERIES:
        raise RuntimeError("V2-C warm-up schedule changed")
    if timing["timed_repetitions_per_path"] * len(PATHS) != TIMED_QUERIES:
        raise RuntimeError("V2-C timed schedule changed")
    if tuple(tuple(order) for order in timing["counterbalance"]) != COUNTERBALANCE:
        raise RuntimeError("V2-C counterbalance changed")


def summarize_timing(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_path: dict[str, list[dict[str, Any]]] = {path: [] for path in PATHS}
    for record in records:
        if record["kind"] == "timed":
            by_path[record["path"]].append(record)
    if any(len(values) != 12 for values in by_path.values()):
        raise RuntimeError("V2-C requires twelve timed records per path")
    summary: dict[str, Any] = {}
    for path, values in by_path.items():
        wall = [float(item["timing"]["wall_ms"]) for item in values]
        visual = [float(item["timing"]["visual_cuda_ms"]) for item in values]
        summary[path] = {
            "wall_ms": wall,
            "visual_cuda_ms": visual,
            "median_wall_ms": statistics.median(wall),
            "mean_wall_ms": statistics.fmean(wall),
            "median_visual_cuda_ms": statistics.median(visual),
            "mean_visual_cuda_ms": statistics.fmean(visual),
        }
    fr = summary["upstream-fr"]["median_wall_ms"]
    refresh = summary["dual-path-refresh"]["median_wall_ms"]
    reuse = summary["dual-path-reuse"]["median_wall_ms"]
    weighted = (1.0 - REUSE_WEIGHT) * refresh + REUSE_WEIGHT * reuse
    gates = {
        "refresh_wall_ratio": refresh / fr,
        "reuse_wall_ratio": reuse / fr,
        "weighted_expected_wall_ratio": weighted / fr,
    }
    gates["refresh_pass"] = gates["refresh_wall_ratio"] <= 1.05
    gates["reuse_pass"] = gates["reuse_wall_ratio"] <= 0.98
    gates["weighted_pass"] = gates["weighted_expected_wall_ratio"] <= 0.98
    gates["all_pass"] = all(gates[name] for name in ("refresh_pass", "reuse_pass", "weighted_pass"))
    return {"paths": summary, "gates": gates, "reuse_weight": REUSE_WEIGHT}


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
        raise TimeoutError("V2-C reached its frozen one-hour wall cap")

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

    config = json.loads((project_root / CONFIG_RELATIVE).read_text())
    validate_gate_config(config)
    upstream_root = project_root / "third_party/openvla-oft"
    libero_root = project_root / "third_party/LIBERO"
    checkpoint = project_root / CHECKPOINT_RELATIVE
    run_root = project_root / "results" / RUN_ID
    if run_root.exists():
        raise SystemExit(f"Immutable V2-C run already exists: {run_root}")
    source_revision = require_clean_revision(project_root)
    require_clean_revision(upstream_root, OPENVLA_REVISION)
    require_clean_revision(libero_root, LIBERO_REVISION)

    sys.path.insert(0, str(project_root / "src"))
    sys.path.insert(0, str(upstream_root))
    sys.path.insert(0, str(project_root / "scripts"))
    import numpy as np
    import torch  # type: ignore[import-not-found]
    from experiments.robot.libero import run_libero_eval as upstream_eval  # type: ignore[import-not-found]
    from experiments.robot.robot_utils import set_seed_everywhere  # type: ignore[import-not-found]
    from run_acr_correctness import validate_checkpoint  # type: ignore[import-not-found]
    from savr.acr.cache import SceneCacheEntry
    from savr.acr.controller import ACRController
    from savr.acr.dual_path import DualPathOpenVLAAdapter
    from savr.acr.instrumentation import CameraInstrumentation
    from savr.acr.openvla_oft import OpenVLAAsymmetricCameraAdapter, TorchTensorOperations
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
        removed = []
        unexpected = []
        for name in sorted({item.name for item in checkpoint.iterdir()} - checkpoint_names):
            if ".back." in name:
                (checkpoint / name).unlink()
                removed.append(name)
            else:
                unexpected.append(name)
        hashes = {name: file_sha256(checkpoint / name) for name in protected_names}
        expected = {
            name: hashlib.sha256(payload).hexdigest() for name, payload in protected_bytes.items()
        }
        if hashes != expected or unexpected:
            raise RuntimeError("Checkpoint loader restoration failed")
        return {"hashes": hashes, "removed_loader_backups": removed, "unexpected": unexpected}

    gpu_before = selected_gpu_snapshot(physical_gpu_id)
    if gpu_before["uuid"] != selected_uuid:
        raise RuntimeError("Selected GPU UUID differs from the pre-run snapshot")
    store = ImmutableRecordStore(project_root / "results")
    manifest = {
        "schema_version": "acr.v2-c-run.v1",
        "run_id": RUN_ID,
        "status": "running",
        "started_at_utc": utc_now(),
        "configuration_sha256": file_sha256(project_root / CONFIG_RELATIVE),
        "configuration_semantic_sha256": config["semantic_sha256"],
        "revisions": {
            "savr": source_revision,
            "openvla_oft": OPENVLA_REVISION,
            "libero": LIBERO_REVISION,
            "checkpoint": CHECKPOINT_REVISION,
        },
        "resources": config["resource_caps"],
        "selected_gpu": gpu_before,
        "command": "scripts/run_acr_v2_c.py",
    }
    store.write_once(f"{RUN_ID}/manifest", manifest)
    budget = QueryBudget()
    started = time.monotonic()
    model = action_head = proprio_projector = noisy_action_projector = processor = None
    result: dict[str, Any] = {
        "schema_version": "acr.v2-c-result.v1",
        "run_id": RUN_ID,
        "status": "running",
        "correctness": {},
        "queries": [],
    }
    terminal_error: BaseException | None = None
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
        result["inputs"] = {
            "scene_sha256": array_sha256(scene, np),
            "wrist_sha256": array_sha256(wrist, np),
            "state_sha256": array_sha256(state, np),
            "instruction_sha256": hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
        }

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
                episode_id=f"v2c-{label}",
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
                    raise RuntimeError("Offline V2-C priming differs from the frozen controller")
                controller.observe(
                    decision=decision,
                    scene_representation=scene_rep,
                    normalized_eef_position=position,
                    action_chunk=action,
                )
            cache.store(context=context, tokens=scene_tokens, refresh_query_index=2)

        def consume_raw_capture(label: str) -> tuple[Any, Any]:
            budget.consume(label)
            captured = []
            original = loaded_model._process_vision_features
            had = "_process_vision_features" in vars(loaded_model)
            previous = vars(loaded_model).get("_process_vision_features")

            def intercept(
                _instance: Any, pixels: Any, language: Any = None, use_film: bool = False
            ) -> Any:
                value = original(pixels, language, use_film)
                captured.append(value.detach().clone())
                return value

            setattr(
                loaded_model,
                "_process_vision_features",
                types.MethodType(intercept, loaded_model),
            )
            try:
                actions = raw_upstream()
            finally:
                if had:
                    setattr(loaded_model, "_process_vision_features", previous)
                else:
                    delattr(loaded_model, "_process_vision_features")
            if len(captured) != 1:
                raise RuntimeError("Upstream correctness oracle did not expose one visual block")
            return actions, captured[0]

        oracle_actions, oracle_tokens = consume_raw_capture("correctness-upstream-refresh")
        scene_reference = oracle_tokens[:, :patch_count].detach().clone()

        # Exact upstream refresh and returned-object identity.
        original_objects: list[Any] = []
        observed_objects: list[Any] = []
        original = loaded_model._process_vision_features
        had = "_process_vision_features" in vars(loaded_model)
        previous = vars(loaded_model).get("_process_vision_features")

        def capture_original(
            _instance: Any, pixels: Any, language: Any = None, use_film: bool = False
        ) -> Any:
            value = original(pixels, language, use_film)
            original_objects.append(value)
            return value

        setattr(
            loaded_model,
            "_process_vision_features",
            types.MethodType(capture_original, loaded_model),
        )
        refresh_adapter = DualPathOpenVLAAdapter(
            model=loaded_model,
            controller=ACRController(frozen_config),
            tensor_ops=tensor_ops,
            correctness_mode=True,
            action_finite_checker=action_finite,
            projected_tokens_observer=observed_objects.append,
        )
        try:
            with refresh_adapter.episode(make_context("correctness-refresh")):
                budget.consume("correctness-dual-path-refresh")
                refresh_result = refresh_adapter.run_query(
                    query=raw_upstream,
                    scene_image=scene,
                    wrist_image=wrist,
                    state=state,
                    state_q01=q01_values,
                    state_q99=q99_values,
                )
        finally:
            if had:
                setattr(loaded_model, "_process_vision_features", previous)
            else:
                delattr(loaded_model, "_process_vision_features")
        if (
            len(original_objects) != 1
            or len(observed_objects) != 1
            or observed_objects[0] is not original_objects[0]
        ):
            raise RuntimeError("Dual-path refresh did not return the exact original tensor object")
        refresh_result.work.validate(scene_refresh=True)
        result["correctness"]["refresh_tokens"] = exact_tensor(
            oracle_tokens, observed_objects[0], torch, "dual-path refresh tokens"
        )
        result["correctness"]["refresh_actions"] = exact_array(
            oracle_actions, refresh_result.value, np, "dual-path refresh actions"
        )
        result["correctness"]["refresh_identity"] = True
        result["correctness"]["refresh_work"] = asdict(refresh_result.work)

        # Version 1 and Version 2 reuse on identical current inputs and cache.
        v1_tokens: list[Any] = []
        v1_controller = ACRController(frozen_config)
        v1_adapter = OpenVLAAsymmetricCameraAdapter(
            model=loaded_model,
            controller=v1_controller,
            tensor_ops=tensor_ops,
            projected_tokens_observer=lambda value: v1_tokens.append(value.detach().clone()),
        )
        v1_context = make_context("correctness-v1-reuse")
        v1_adapter.begin_context(v1_context)
        prime(v1_controller, v1_adapter.cache, v1_context, scene_reference)
        budget.consume("correctness-acr-v1-reuse")
        v1_result = v1_adapter.run_query(
            query=raw_upstream,
            scene_image=scene,
            wrist_image=wrist,
            state=state,
            state_q01=q01_values,
            state_q99=q99_values,
        )
        v2_tokens: list[Any] = []
        v2_controller = ACRController(frozen_config)
        v2_adapter = DualPathOpenVLAAdapter(
            model=loaded_model,
            controller=v2_controller,
            tensor_ops=tensor_ops,
            correctness_mode=True,
            action_finite_checker=action_finite,
            projected_tokens_observer=lambda value: v2_tokens.append(value.detach().clone()),
        )
        v2_context = make_context("correctness-v2-reuse")
        with v2_adapter.episode(v2_context):
            prime(v2_controller, v2_adapter.cache, v2_context, scene_reference)
            budget.consume("correctness-dual-path-reuse")
            v2_result = v2_adapter.run_query(
                query=raw_upstream,
                scene_image=scene,
                wrist_image=wrist,
                state=state,
                state_q01=q01_values,
                state_q99=q99_values,
            )
        if v1_result.decision.refresh or v2_result.decision.refresh:
            raise RuntimeError("Planned V2-C correctness reuse did not reuse")
        v1_result.work.validate(scene_refresh=False)
        v2_result.work.validate(scene_refresh=False)
        result["correctness"]["reuse_tokens"] = exact_tensor(
            v1_tokens[0], v2_tokens[0], torch, "V1/V2 reuse tokens"
        )
        result["correctness"]["reuse_actions"] = exact_array(
            v1_result.value, v2_result.value, np, "V1/V2 reuse actions"
        )
        result["correctness"]["reuse_v1_work"] = asdict(v1_result.work)
        result["correctness"]["reuse_v2_work"] = asdict(v2_result.work)

        # Real cache incompatibility must force the exact upstream refresh path.
        mismatch_controller = ACRController(frozen_config)
        mismatch_adapter = DualPathOpenVLAAdapter(
            model=loaded_model,
            controller=mismatch_controller,
            tensor_ops=tensor_ops,
            correctness_mode=True,
            action_finite_checker=action_finite,
        )
        mismatch_context = make_context("correctness-cache-mismatch")
        with mismatch_adapter.episode(mismatch_context):
            prime(mismatch_controller, mismatch_adapter.cache, mismatch_context, scene_reference)
            entry = mismatch_adapter.cache.entry
            if entry is None:
                raise RuntimeError("Cache mismatch proof lacks a scene entry")
            mismatch_adapter.cache._entry = SceneCacheEntry(  # type: ignore[attr-defined]
                context=entry.context,
                tokens=entry.tokens,
                metadata=replace(entry.metadata, dtype="torch.float32"),
                refresh_query_index=entry.refresh_query_index,
            )
            budget.consume("correctness-cache-mismatch-refresh")
            mismatch = mismatch_adapter.run_query(
                query=raw_upstream,
                scene_image=scene,
                wrist_image=wrist,
                state=state,
                state_q01=q01_values,
                state_q99=q99_values,
            )
        if mismatch.cache_event != "forced-refresh" or not mismatch.decision.refresh:
            raise RuntimeError("Cache metadata mismatch did not force a safe refresh")
        mismatch.work.validate(scene_refresh=True)
        result["correctness"]["cache_mismatch"] = {
            "cache_event": mismatch.cache_event,
            "work": asdict(mismatch.work),
        }

        # Expected downstream failure must restore the original method and cache.
        restoration_adapter = DualPathOpenVLAAdapter(
            model=loaded_model,
            controller=ACRController(frozen_config),
            tensor_ops=tensor_ops,
            correctness_mode=True,
            action_finite_checker=action_finite,
        )
        class_method = type(loaded_model)._process_vision_features

        def fail_after_query() -> Any:
            raw_upstream()
            raise ExpectedRestorationProbe("expected V2-C restoration probe")

        try:
            with restoration_adapter.episode(make_context("correctness-restoration")):
                budget.consume("correctness-exception-restoration")
                restoration_adapter.run_query(
                    query=fail_after_query,
                    scene_image=scene,
                    wrist_image=wrist,
                    state=state,
                    state_q01=q01_values,
                    state_q99=q99_values,
                )
        except ExpectedRestorationProbe:
            pass
        else:
            raise RuntimeError("Expected restoration probe did not fail")
        if (
            "_process_vision_features" in vars(loaded_model)
            or type(loaded_model)._process_vision_features is not class_method
            or restoration_adapter.cache.entry is not None
            or restoration_adapter.controller.query_index != 0
        ):
            raise RuntimeError("Expected exception did not restore exact adapter state")
        result["correctness"]["exception_restoration"] = {
            "restored": True,
            "failure": asdict(restoration_adapter.last_failure)
            if restoration_adapter.last_failure
            else None,
        }
        if len(budget.labels) != CORRECTNESS_QUERIES:
            raise RuntimeError("Correctness query schedule did not consume exactly six queries")

        timing_records: list[dict[str, Any]] = []

        def execute_timing(path: str, kind: str, repetition: int, order: int) -> None:
            label = f"{kind}-{repetition:02d}-{order}-{path}"
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
                    adapter = DualPathOpenVLAAdapter(
                        model=loaded_model,
                        controller=controller,
                        tensor_ops=tensor_ops,
                        instrumentation=CameraInstrumentation(),
                        correctness_mode=False,
                        action_finite_checker=action_finite,
                    )
                    context = make_context(f"timing-{kind}-{repetition}-{order}-{path}")
                    with adapter.episode(context):
                        if path == "dual-path-reuse":
                            prime(controller, adapter.cache, context, scene_reference)
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
                raise RuntimeError(f"Timed {path} returned invalid actions")
            components = dict(timing.component_device_ms)
            counts = dict(timing.component_counts)
            expected_counts = {"siglip": 1, "dinov2": 1, "projector": 1}
            actual_counts = {name: counts.get(name, 0) for name in expected_counts}
            if actual_counts != expected_counts:
                raise RuntimeError(f"Timed {path} component counts differ: {actual_counts}")
            if adapter_result is not None:
                expected_refresh = path == "dual-path-refresh"
                adapter_result.work.validate(scene_refresh=expected_refresh)
                if adapter_result.decision.refresh != expected_refresh:
                    raise RuntimeError(f"Timed {path} selected the wrong execution path")
            visual_cuda = sum(components[name] for name in expected_counts)
            timing_records.append(
                {
                    "query_index": len(budget.labels) - 1,
                    "label": label,
                    "kind": kind,
                    "repetition": repetition,
                    "order": order,
                    "path": path,
                    "actions_sha256": array_sha256(action_array, np),
                    "timing": {
                        "wall_ms": timing.wall_ms,
                        "total_cuda_ms": timing.total_device_ms,
                        "visual_cuda_ms": visual_cuda,
                        "component_cuda_ms": components,
                        "component_counts": counts,
                    },
                    "work": asdict(adapter_result.work) if adapter_result is not None else None,
                }
            )

        for path in PATHS:
            for repetition in range(2):
                execute_timing(path, "warmup", repetition, 0)
        for repetition in range(12):
            order = COUNTERBALANCE[repetition % len(COUNTERBALANCE)]
            for position, path in enumerate(order):
                execute_timing(path, "timed", repetition, position)
        if len(budget.labels) != QUERY_CAP or len(timing_records) != WARMUP_QUERIES + TIMED_QUERIES:
            raise RuntimeError("V2-C did not complete the exact 48-query schedule")

        timing_summary = summarize_timing(timing_records)
        result["queries"] = timing_records
        result["timing_summary"] = timing_summary
        result["query_labels"] = list(budget.labels)
        result["query_count"] = len(budget.labels)
        result["correctness_pass"] = True
        result["latency_pass"] = bool(timing_summary["gates"]["all_pass"])
        result["status"] = "pass" if result["latency_pass"] else "stopped-negative"
        result["checkpoint_restoration"] = restore_checkpoint()
        checkpoint_after = validate_checkpoint(project_root, checkpoint)
        if checkpoint_after != checkpoint_before:
            raise RuntimeError("Checkpoint inventory changed during V2-C")
        require_clean_revision(project_root, source_revision)
        require_clean_revision(upstream_root, OPENVLA_REVISION)
        require_clean_revision(libero_root, LIBERO_REVISION)
        result["source_trees_restored"] = True
        result["gpu_before"] = gpu_before
        result["gpu_after"] = selected_gpu_snapshot(physical_gpu_id)
        if result["gpu_after"]["uuid"] != selected_uuid:
            raise RuntimeError("Selected GPU UUID changed")
        result["wall_seconds"] = time.monotonic() - started
        result["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        result["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        result["finished_at_utc"] = utc_now()
        result["result_semantic_sha256"] = semantic_sha256(result)
        projected_size = directory_size(run_root) + len(canonical_bytes(result)) + 1
        if projected_size > ARTIFACT_CAP_BYTES:
            raise RuntimeError("V2-C artifact cap exceeded")
        store.write_once(f"{RUN_ID}/final", result)
        return 0 if result["status"] == "pass" else 2
    except BaseException as error:
        terminal_error = error
        failure = {
            "schema_version": "acr.v2-c-failure.v1",
            "run_id": RUN_ID,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "query_count": len(budget.labels),
            "query_labels": list(budget.labels),
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
            print(f"V2-C checkpoint restoration error: {restoration_error}", file=sys.stderr)
        if terminal_error is not None:
            print(
                f"V2-C failed closed after {len(budget.labels)} queries: {terminal_error}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    raise SystemExit(main())
