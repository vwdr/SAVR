#!/usr/bin/env python3
"""Run the frozen ACR V3-C real-model correctness and latency gate."""

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
from dataclasses import asdict
from pathlib import Path
from typing import Any


EXPECTED_ROOT = Path("/home/ved/SAVR")
RUN_ID = "acr-v3c-correctness-latency-v01"
CONFIG_RELATIVE = Path("configs/acr/v3_c_gate.json")
CHECKPOINT_RELATIVE = Path("checkpoints/openvla-7b-oft-libero-four-suite")
CHECKPOINT_REVISION = "638918f3d1c2e43a39a8a20772bdb8b91835e4b7"
OPENVLA_REVISION = "e4287e94541f459edc4feabc4e181f537cd569a8"
LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
INSTRUCTION = "move the robot safely to the target"
QUERY_CAP = 64
CORRECTNESS_QUERIES = 8
WARMUP_QUERIES = 8
TIMED_QUERIES = 48
WALL_CAP_SECONDS = 3600
ARTIFACT_CAP_BYTES = 512 * 1024**2
REUSE_WEIGHT = 0.26055045871559634
TOKEN_RTOL = 0.016
TOKEN_ATOL = 1e-5
PATHS = ("sequential-fr", "batched-fr", "v3-refresh", "v3-reuse")
COUNTERBALANCE = (
    PATHS,
    ("batched-fr", "v3-refresh", "v3-reuse", "sequential-fr"),
    ("v3-refresh", "v3-reuse", "sequential-fr", "batched-fr"),
    ("v3-reuse", "sequential-fr", "batched-fr", "v3-refresh"),
)
CORRECTNESS_LABELS = (
    "correctness-input-a-upstream-fr",
    "correctness-input-a-batched-fr",
    "correctness-input-a-v3-refresh",
    "correctness-input-b-upstream-fr",
    "correctness-input-b-batched-fr",
    "correctness-input-b-v3-refresh",
    "correctness-input-a-v2-reuse",
    "correctness-input-a-v3-reuse",
)


class QueryBudget:
    """Consume one immutable identity before every real model call."""

    def __init__(self, cap: int = QUERY_CAP) -> None:
        self.cap = cap
        self.labels: list[str] = []

    def consume(self, label: str) -> int:
        if not label or label in self.labels:
            raise RuntimeError("V3-C query labels must be non-empty and unique")
        if len(self.labels) >= self.cap:
            raise RuntimeError(f"V3-C query cap exceeded before {label}")
        self.labels.append(label)
        return len(self.labels) - 1


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


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


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


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


def array_sha256(value: Any, np: Any) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def tensor_sha256(value: Any) -> str:
    import torch  # type: ignore[import-not-found]

    raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def exact_array(reference: Any, candidate: Any, np: Any, label: str) -> dict[str, Any]:
    left, right = np.asarray(reference), np.asarray(candidate)
    if left.shape != right.shape or not bool(np.array_equal(left, right)):
        maximum = None
        if left.shape == right.shape and left.size:
            maximum = float(np.max(np.abs(left - right)))
        raise RuntimeError(f"{label} exact parity failed; maximum difference={maximum}")
    return {"equal": True, "shape": list(left.shape), "sha256": array_sha256(left, np)}


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
        "device": str(reference.device),
        "sha256": tensor_sha256(reference),
    }


def close_tensor(reference: Any, candidate: Any, torch: Any, label: str) -> dict[str, Any]:
    metadata_equal = (
        tuple(reference.shape) == tuple(candidate.shape)
        and reference.dtype == candidate.dtype
        and reference.device == candidate.device
    )
    close = metadata_equal and bool(
        torch.allclose(reference, candidate, rtol=TOKEN_RTOL, atol=TOKEN_ATOL)
    )
    maximum = None
    mean = None
    if tuple(reference.shape) == tuple(candidate.shape):
        difference = (reference.float() - candidate.float()).abs()
        maximum = float(difference.max().item())
        mean = float(difference.mean().item())
    if not close:
        raise RuntimeError(f"{label} tolerance failed; maximum difference={maximum}")
    return {
        "close": True,
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "device": str(reference.device),
        "rtol": TOKEN_RTOL,
        "atol": TOKEN_ATOL,
        "maximum_absolute_difference": maximum,
        "mean_absolute_difference": mean,
        "reference_sha256": tensor_sha256(reference),
        "candidate_sha256": tensor_sha256(candidate),
    }


def deterministic_inputs(np: Any, size: int = 256) -> dict[str, tuple[Any, Any]]:
    y, x = np.indices((size, size))
    scene = np.stack(((x + y) % 256, x % 256, y % 256), axis=-1).astype(np.uint8)
    wrist = np.stack(
        ((2 * x + y) % 256, (x + 3 * y) % 256, (255 - x) % 256), axis=-1
    ).astype(np.uint8)
    scene_b = np.ascontiguousarray(scene[:, ::-1, (2, 0, 1)])
    wrist_b = np.ascontiguousarray(wrist[::-1, :, (1, 2, 0)])
    return {"input-a": (scene, wrist), "input-b": (scene_b, wrist_b)}


def expected_query_labels() -> tuple[str, ...]:
    labels = list(CORRECTNESS_LABELS)
    for path in PATHS:
        for repetition in range(2):
            labels.append(f"warmup-{path}-{repetition:02d}")
    for repetition in range(12):
        order = COUNTERBALANCE[repetition % len(COUNTERBALANCE)]
        for position, path in enumerate(order):
            labels.append(f"timed-{repetition:02d}-{position}-{path}")
    return tuple(labels)


def validate_gate_config(config: dict[str, Any]) -> None:
    supplied = config.get("semantic_sha256")
    payload = dict(config)
    payload.pop("semantic_sha256", None)
    if semantic_sha256(payload) != supplied:
        raise RuntimeError("V3-C configuration semantic hash mismatch")
    if tuple(config["correctness_queries"]) != CORRECTNESS_LABELS:
        raise RuntimeError("V3-C correctness schedule changed")
    correctness = config["correctness"]
    if (
        correctness["projected_token_rtol"] != TOKEN_RTOL
        or correctness["projected_token_atol"] != TOKEN_ATOL
    ):
        raise RuntimeError("V3-C correctness tolerance changed")
    timing = config["timing"]
    if tuple(timing["paths"]) != PATHS:
        raise RuntimeError("V3-C timing paths changed")
    if tuple(tuple(order) for order in timing["counterbalance"]) != COUNTERBALANCE:
        raise RuntimeError("V3-C counterbalance changed")
    if timing["untimed_warmups_per_path"] * len(PATHS) != WARMUP_QUERIES:
        raise RuntimeError("V3-C warm-up schedule changed")
    if timing["timed_repetitions_per_path"] * len(PATHS) != TIMED_QUERIES:
        raise RuntimeError("V3-C timed schedule changed")
    expected_timing_values = {
        "reuse_weight": REUSE_WEIGHT,
        "bfr_sequential_wall_ratio_max": 0.98,
        "v3_refresh_bfr_wall_ratio_max": 1.02,
        "v3_reuse_bfr_wall_ratio_max": 0.98,
        "v3_weighted_sequential_wall_ratio_max": 0.98,
        "v3_weighted_bfr_wall_ratio_max": 1.0,
        "v3_weighted_visual_cuda_reduction_min": 0.10,
        "outlier_deletion": False,
    }
    if any(timing[key] != value for key, value in expected_timing_values.items()):
        raise RuntimeError("V3-C latency gate changed")
    expected_caps = {
        "gpu_count": 1,
        "model_processes": 1,
        "model_queries": 64,
        "rollout_episodes": 0,
        "simulator_resets": 0,
        "wall_seconds": 3600,
        "artifact_bytes": 536870912,
        "downloads_allowed": False,
    }
    if config["resource_caps"] != expected_caps:
        raise RuntimeError("V3-C resource caps changed")
    if len(expected_query_labels()) != QUERY_CAP:
        raise RuntimeError("V3-C exact query identity schedule is inconsistent")


def summarize_timing(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_path: dict[str, list[dict[str, Any]]] = {path: [] for path in PATHS}
    for record in records:
        if record["kind"] == "timed":
            by_path[record["path"]].append(record)
    if any(len(values) != 12 for values in by_path.values()):
        raise RuntimeError("V3-C requires twelve timed records per path")
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
    sequential = summary["sequential-fr"]
    bfr = summary["batched-fr"]
    refresh = summary["v3-refresh"]
    reuse = summary["v3-reuse"]
    weighted_wall = (1.0 - REUSE_WEIGHT) * refresh["median_wall_ms"] + (
        REUSE_WEIGHT * reuse["median_wall_ms"]
    )
    weighted_visual = (1.0 - REUSE_WEIGHT) * refresh["median_visual_cuda_ms"] + (
        REUSE_WEIGHT * reuse["median_visual_cuda_ms"]
    )
    gates = {
        "bfr_sequential_wall_ratio": bfr["median_wall_ms"] / sequential["median_wall_ms"],
        "v3_refresh_bfr_wall_ratio": refresh["median_wall_ms"] / bfr["median_wall_ms"],
        "v3_reuse_bfr_wall_ratio": reuse["median_wall_ms"] / bfr["median_wall_ms"],
        "v3_weighted_sequential_wall_ratio": weighted_wall / sequential["median_wall_ms"],
        "v3_weighted_bfr_wall_ratio": weighted_wall / bfr["median_wall_ms"],
        "v3_weighted_visual_cuda_reduction": 1.0
        - weighted_visual / sequential["median_visual_cuda_ms"],
        "v3_weighted_wall_ms": weighted_wall,
        "v3_weighted_visual_cuda_ms": weighted_visual,
    }
    gates.update(
        {
            "bfr_sequential_pass": gates["bfr_sequential_wall_ratio"] <= 0.98,
            "v3_refresh_bfr_pass": gates["v3_refresh_bfr_wall_ratio"] <= 1.02,
            "v3_reuse_bfr_pass": gates["v3_reuse_bfr_wall_ratio"] <= 0.98,
            "v3_weighted_sequential_pass": gates["v3_weighted_sequential_wall_ratio"] <= 0.98,
            "v3_weighted_bfr_pass": gates["v3_weighted_bfr_wall_ratio"] <= 1.00,
            "v3_weighted_visual_pass": gates["v3_weighted_visual_cuda_reduction"] >= 0.10,
        }
    )
    pass_keys = [key for key in gates if key.endswith("_pass")]
    gates["all_pass"] = all(bool(gates[key]) for key in pass_keys)
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
        raise TimeoutError("V3-C reached its frozen one-hour wall cap")

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
        raise SystemExit(f"Immutable V3-C run already exists: {run_root}")
    source_revision = require_clean_revision(project_root)
    require_clean_revision(upstream_root, OPENVLA_REVISION)
    require_clean_revision(libero_root, LIBERO_REVISION)

    sys.path.insert(0, str(project_root / "src"))
    sys.path.insert(0, str(upstream_root))
    sys.path.insert(0, str(project_root / "scripts"))
    import numpy as np
    import torch  # type: ignore[import-not-found]
    from experiments.robot.libero import (  # type: ignore[import-not-found]
        run_libero_eval as upstream_eval,
    )
    from experiments.robot.robot_utils import set_seed_everywhere  # type: ignore[import-not-found]
    from run_acr_correctness import validate_checkpoint  # type: ignore[import-not-found]
    from savr.acr.batched_dual_path import (
        BatchedDualPathOpenVLAAdapter,
        BatchedFullRefreshAdapter,
    )
    from savr.acr.cache import SceneTokenCache
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
        removed: list[str] = []
        unexpected: list[str] = []
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
            raise RuntimeError("V3-C checkpoint loader restoration failed")
        return {"hashes": hashes, "removed_loader_backups": removed, "unexpected": unexpected}

    gpu_before = selected_gpu_snapshot(physical_gpu_id)
    if gpu_before["uuid"] != selected_uuid:
        raise RuntimeError("Selected GPU UUID differs from the pre-run snapshot")
    store = ImmutableRecordStore(project_root / "results")
    manifest = {
        "schema_version": "acr.v3-c-run.v1",
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
        "command": "scripts/run_acr_v3_c.py",
    }
    store.write_once(f"{RUN_ID}/manifest", manifest)
    budget = QueryBudget()
    started = time.monotonic()
    model = action_head = proprio_projector = noisy_action_projector = processor = None
    result: dict[str, Any] = {
        "schema_version": "acr.v3-c-result.v1",
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
        q01 = np.asarray(stats["q01"], dtype=np.float64)
        q99 = np.asarray(stats["q99"], dtype=np.float64)
        if q01.shape != (8,) or q99.shape != (8,) or not np.all(q99 > q01):
            raise RuntimeError("Pinned proprioception statistics are invalid")
        q01_values = tuple(float(value) for value in q01)
        q99_values = tuple(float(value) for value in q99)
        state = (q01 + q99) / 2
        inputs = deterministic_inputs(np)
        result["inputs"] = {
            label: {
                "scene_sha256": array_sha256(images[0], np),
                "wrist_sha256": array_sha256(images[1], np),
            }
            for label, images in inputs.items()
        }
        result["inputs"]["state_sha256"] = array_sha256(state, np)
        result["inputs"]["instruction_sha256"] = hashlib.sha256(
            INSTRUCTION.encode()
        ).hexdigest()

        def observation(input_label: str) -> dict[str, Any]:
            scene_image, wrist_image = inputs[input_label]
            return {
                "full_image": scene_image.copy(),
                "wrist_image": wrist_image.copy(),
                "state": state.copy(),
            }

        def raw_upstream(input_label: str) -> Any:
            return upstream_eval.get_action(
                cfg,
                loaded_model,
                observation(input_label),
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

        v2_config = ACRConfiguration(
            "sa-dp-acr-t25-h2-b30-v01",
            ACRPolicy.SA_ACR,
            scene_threshold=0.2476380718954248,
            translation_threshold=0.5479944908411765,
            horizon=2,
            hard_reuse_cap=0.30,
        )
        v3_config = ACRConfiguration(
            "sa-bdp-acr-t25-h2-b30-v01",
            ACRPolicy.SA_ACR,
            scene_threshold=0.2476380718954248,
            translation_threshold=0.5479944908411765,
            horizon=2,
            hard_reuse_cap=0.30,
        )
        instruction_sha = hashlib.sha256(INSTRUCTION.encode()).hexdigest()

        def make_context(label: str, configuration_id: str) -> ACRContext:
            return ACRContext(
                episode_id=f"v3c-{label}",
                attempt_id=f"{RUN_ID}-{label}",
                task_id="synthetic",
                instruction_sha256=instruction_sha,
                checkpoint_id=CHECKPOINT_REVISION,
                upstream_revision=OPENVLA_REVISION,
                configuration_id=configuration_id,
                controller_version=v3_config.controller_version,
                preprocessing_id="openvla-center-crop-v1",
                action_head_id="l1-regression-8x7",
                dtype="torch.bfloat16",
                device="cuda:0",
                patch_count=patch_count,
            )

        def prime(
            controller: Any,
            cache: SceneTokenCache,
            context: ACRContext,
            scene_tokens: Any,
            input_label: str,
        ) -> None:
            scene_image, _ = inputs[input_label]
            scene_rep = prepare_scene_representation(scene_image)
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
                    raise RuntimeError("Offline V3-C priming differs from the frozen controller")
                controller.observe(
                    decision=decision,
                    scene_representation=scene_rep,
                    normalized_eef_position=position,
                    action_chunk=action,
                )
            cache.store(context=context, tokens=scene_tokens, refresh_query_index=2)

        def run_external_timing(call: Any) -> tuple[Any, Any]:
            timer = SynchronizedQueryTimer(TorchCudaEventBackend(torch))
            hooks = ModuleTimingHooks(
                {
                    "siglip": loaded_model.vision_backbone.featurizer,
                    "dinov2": loaded_model.vision_backbone.fused_featurizer,
                    "projector": loaded_model.projector,
                },
                timer,
            )
            timer.start()
            try:
                value = call()
                timing = timer.finish()
            finally:
                hooks.remove()
            return value, timing

        def require_physical_counts(
            timing: Any, expected: tuple[int, int, int], label: str
        ) -> None:
            counts = timing.component_counts
            actual = (
                int(counts.get("siglip", 0)),
                int(counts.get("dinov2", 0)),
                int(counts.get("projector", 0)),
            )
            if actual != expected:
                raise RuntimeError(f"{label} physical component counts differ: {actual}")

        def capture_upstream(input_label: str) -> tuple[Any, Any, Any]:
            label = f"correctness-{input_label}-upstream-fr"
            budget.consume(label)
            captured: list[Any] = []
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
                actions, timing = run_external_timing(lambda: raw_upstream(input_label))
            finally:
                if had:
                    setattr(loaded_model, "_process_vision_features", previous)
                else:
                    delattr(loaded_model, "_process_vision_features")
            if len(captured) != 1:
                raise RuntimeError("Sequential correctness oracle did not expose one visual block")
            require_physical_counts(timing, (2, 2, 1), label)
            return actions, captured[0], timing

        correctness_oracles: dict[str, tuple[Any, Any]] = {}
        for input_label in ("input-a", "input-b"):
            oracle_actions, oracle_tokens, oracle_timing = capture_upstream(input_label)
            correctness_oracles[input_label] = (oracle_actions, oracle_tokens)
            scene_image, wrist_image = inputs[input_label]
            input_result: dict[str, Any] = {
                "upstream_work": dict(oracle_timing.component_counts),
            }

            bfr_tokens: list[Any] = []
            bfr_adapter = BatchedFullRefreshAdapter(
                model=loaded_model,
                tensor_ops=tensor_ops,
                correctness_mode=True,
                action_finite_checker=action_finite,
                projected_tokens_observer=lambda value: bfr_tokens.append(
                    value.detach().clone()
                ),
            )
            bfr_label = f"correctness-{input_label}-batched-fr"
            with bfr_adapter.episode(make_context(bfr_label, "batched-fr-v01")):
                budget.consume(bfr_label)
                bfr_result, bfr_timing = run_external_timing(
                    lambda: bfr_adapter.run_query(lambda: raw_upstream(input_label))
                )
            if len(bfr_tokens) != 1:
                raise RuntimeError("BFR correctness query did not expose one visual block")
            require_physical_counts(bfr_timing, (1, 1, 1), bfr_label)
            bfr_result.work.validate()
            input_result["batched_fr"] = {
                "tokens": close_tensor(oracle_tokens, bfr_tokens[0], torch, bfr_label),
                "scene_block": close_tensor(
                    oracle_tokens[:, :patch_count],
                    bfr_tokens[0][:, :patch_count],
                    torch,
                    f"{bfr_label}-scene",
                ),
                "wrist_block": close_tensor(
                    oracle_tokens[:, patch_count:],
                    bfr_tokens[0][:, patch_count:],
                    torch,
                    f"{bfr_label}-wrist",
                ),
                "actions": exact_array(oracle_actions, bfr_result.value, np, bfr_label),
                "work": asdict(bfr_result.work),
                "physical_counts": dict(bfr_timing.component_counts),
            }

            v3_tokens: list[Any] = []
            v3_controller = ACRController(v3_config)
            v3_adapter = BatchedDualPathOpenVLAAdapter(
                model=loaded_model,
                controller=v3_controller,
                tensor_ops=tensor_ops,
                correctness_mode=True,
                action_finite_checker=action_finite,
                projected_tokens_observer=lambda value: v3_tokens.append(
                    value.detach().clone()
                ),
            )
            v3_label = f"correctness-{input_label}-v3-refresh"
            with v3_adapter.episode(make_context(v3_label, v3_config.configuration_id)):
                budget.consume(v3_label)
                v3_result, v3_timing = run_external_timing(
                    lambda: v3_adapter.run_query(
                        query=lambda: raw_upstream(input_label),
                        scene_image=scene_image,
                        wrist_image=wrist_image,
                        state=state,
                        state_q01=q01_values,
                        state_q99=q99_values,
                    )
                )
                cache_entry = v3_adapter.cache.entry
                if cache_entry is None:
                    raise RuntimeError("V3 correctness refresh did not own a scene cache")
                cache_proof = exact_tensor(
                    v3_tokens[0][:, :patch_count],
                    cache_entry.tokens,
                    torch,
                    f"{v3_label}-cache",
                )
            if len(v3_tokens) != 1 or not v3_result.decision.refresh:
                raise RuntimeError("V3 correctness refresh selected the wrong path")
            require_physical_counts(v3_timing, (1, 1, 1), v3_label)
            v3_result.work.validate()
            input_result["v3_refresh"] = {
                "tokens": close_tensor(oracle_tokens, v3_tokens[0], torch, v3_label),
                "scene_block": close_tensor(
                    oracle_tokens[:, :patch_count],
                    v3_tokens[0][:, :patch_count],
                    torch,
                    f"{v3_label}-scene",
                ),
                "wrist_block": close_tensor(
                    oracle_tokens[:, patch_count:],
                    v3_tokens[0][:, patch_count:],
                    torch,
                    f"{v3_label}-wrist",
                ),
                "actions": exact_array(oracle_actions, v3_result.value, np, v3_label),
                "cache": cache_proof,
                "work": asdict(v3_result.work),
                "physical_counts": dict(v3_timing.component_counts),
            }
            result["correctness"][input_label] = input_result

        oracle_actions_a, oracle_tokens_a = correctness_oracles["input-a"]
        scene_reference = oracle_tokens_a[:, :patch_count].detach().clone()
        scene_a, wrist_a = inputs["input-a"]

        v2_tokens: list[Any] = []
        v2_controller = ACRController(v2_config)
        v2_adapter = DualPathOpenVLAAdapter(
            model=loaded_model,
            controller=v2_controller,
            tensor_ops=tensor_ops,
            correctness_mode=True,
            action_finite_checker=action_finite,
            projected_tokens_observer=lambda value: v2_tokens.append(value.detach().clone()),
        )
        v2_label = "correctness-input-a-v2-reuse"
        v2_context = make_context(v2_label, v2_config.configuration_id)
        with v2_adapter.episode(v2_context):
            prime(v2_controller, v2_adapter.cache, v2_context, scene_reference, "input-a")
            budget.consume(v2_label)
            v2_result, v2_timing = run_external_timing(
                lambda: v2_adapter.run_query(
                    query=lambda: raw_upstream("input-a"),
                    scene_image=scene_a,
                    wrist_image=wrist_a,
                    state=state,
                    state_q01=q01_values,
                    state_q99=q99_values,
                )
            )

        v3_reuse_tokens: list[Any] = []
        v3_reuse_controller = ACRController(v3_config)
        v3_reuse_adapter = BatchedDualPathOpenVLAAdapter(
            model=loaded_model,
            controller=v3_reuse_controller,
            tensor_ops=tensor_ops,
            correctness_mode=True,
            action_finite_checker=action_finite,
            projected_tokens_observer=lambda value: v3_reuse_tokens.append(
                value.detach().clone()
            ),
        )
        v3_reuse_label = "correctness-input-a-v3-reuse"
        v3_reuse_context = make_context(v3_reuse_label, v3_config.configuration_id)
        with v3_reuse_adapter.episode(v3_reuse_context):
            prime(
                v3_reuse_controller,
                v3_reuse_adapter.cache,
                v3_reuse_context,
                scene_reference,
                "input-a",
            )
            budget.consume(v3_reuse_label)
            v3_reuse_result, v3_reuse_timing = run_external_timing(
                lambda: v3_reuse_adapter.run_query(
                    query=lambda: raw_upstream("input-a"),
                    scene_image=scene_a,
                    wrist_image=wrist_a,
                    state=state,
                    state_q01=q01_values,
                    state_q99=q99_values,
                )
            )
        if (
            v2_result.decision.refresh
            or v3_reuse_result.decision.refresh
            or len(v2_tokens) != 1
            or len(v3_reuse_tokens) != 1
        ):
            raise RuntimeError("V2/V3 correctness reuse selected the wrong execution path")
        require_physical_counts(v2_timing, (1, 1, 1), v2_label)
        require_physical_counts(v3_reuse_timing, (1, 1, 1), v3_reuse_label)
        v2_result.work.validate(scene_refresh=False)
        v3_reuse_result.work.validate()
        result["correctness"]["reuse"] = {
            "tokens": exact_tensor(v2_tokens[0], v3_reuse_tokens[0], torch, "V2/V3 reuse"),
            "actions": exact_array(v2_result.value, v3_reuse_result.value, np, "V2/V3 reuse"),
            "oracle_actions": exact_array(
                oracle_actions_a, v3_reuse_result.value, np, "V3 reuse/upstream"
            ),
            "v2_work": asdict(v2_result.work),
            "v3_work": asdict(v3_reuse_result.work),
            "v2_physical_counts": dict(v2_timing.component_counts),
            "v3_physical_counts": dict(v3_reuse_timing.component_counts),
        }
        if tuple(budget.labels) != CORRECTNESS_LABELS:
            raise RuntimeError("V3-C correctness query identities changed")

        timing_records: list[dict[str, Any]] = []

        def execute_timing(path: str, kind: str, repetition: int, order: int) -> None:
            label = (
                f"warmup-{path}-{repetition:02d}"
                if kind == "warmup"
                else f"timed-{repetition:02d}-{order}-{path}"
            )
            budget.consume(label)
            adapter_result: Any = None
            work: dict[str, Any] | None = None
            if path == "sequential-fr":
                actions, timing = run_external_timing(lambda: raw_upstream("input-a"))
                wall_ms = timing.wall_ms
                expected_counts = (2, 2, 1)
            elif path == "batched-fr":
                timer = SynchronizedQueryTimer(TorchCudaEventBackend(torch))
                hooks = ModuleTimingHooks(
                    {
                        "siglip": loaded_model.vision_backbone.featurizer,
                        "dinov2": loaded_model.vision_backbone.fused_featurizer,
                        "projector": loaded_model.projector,
                    },
                    timer,
                )
                adapter = BatchedFullRefreshAdapter(
                    model=loaded_model,
                    tensor_ops=tensor_ops,
                    instrumentation=CameraInstrumentation(timer=timer),
                    correctness_mode=False,
                    action_finite_checker=action_finite,
                )
                context = make_context(label, "batched-fr-v01")
                try:
                    with adapter.episode(context):
                        adapter_result = adapter.run_query(lambda: raw_upstream("input-a"))
                finally:
                    hooks.remove()
                actions = adapter_result.value
                timing = adapter_result.device_timing
                wall_ms = adapter_result.query_wall_ms
                adapter_result.work.validate()
                work = asdict(adapter_result.work)
                expected_counts = (1, 1, 1)
            else:
                timer = SynchronizedQueryTimer(TorchCudaEventBackend(torch))
                hooks = ModuleTimingHooks(
                    {
                        "siglip": loaded_model.vision_backbone.featurizer,
                        "dinov2": loaded_model.vision_backbone.fused_featurizer,
                        "projector": loaded_model.projector,
                    },
                    timer,
                )
                controller = ACRController(v3_config)
                adapter = BatchedDualPathOpenVLAAdapter(
                    model=loaded_model,
                    controller=controller,
                    tensor_ops=tensor_ops,
                    instrumentation=CameraInstrumentation(timer=timer),
                    correctness_mode=False,
                    action_finite_checker=action_finite,
                )
                context = make_context(label, v3_config.configuration_id)
                try:
                    with adapter.episode(context):
                        if path == "v3-reuse":
                            prime(controller, adapter.cache, context, scene_reference, "input-a")
                        adapter_result = adapter.run_query(
                            query=lambda: raw_upstream("input-a"),
                            scene_image=scene_a,
                            wrist_image=wrist_a,
                            state=state,
                            state_q01=q01_values,
                            state_q99=q99_values,
                        )
                finally:
                    hooks.remove()
                actions = adapter_result.value
                timing = adapter_result.device_timing
                wall_ms = adapter_result.query_wall_ms
                expected_refresh = path == "v3-refresh"
                if adapter_result.decision.refresh != expected_refresh:
                    raise RuntimeError(f"Timed {path} selected the wrong execution path")
                adapter_result.work.validate()
                work = asdict(adapter_result.work)
                expected_counts = (1, 1, 1)
            if timing is None:
                raise RuntimeError(f"Timed {path} did not return device timing")
            require_physical_counts(timing, expected_counts, label)
            action_proof = exact_array(oracle_actions_a, actions, np, f"timed-{path}")
            components = dict(timing.component_device_ms)
            visual_cuda = sum(float(components[name]) for name in ("siglip", "dinov2", "projector"))
            timing_records.append(
                {
                    "query_index": len(budget.labels) - 1,
                    "label": label,
                    "kind": kind,
                    "repetition": repetition,
                    "order": order,
                    "path": path,
                    "actions_sha256": action_proof["sha256"],
                    "timing": {
                        "wall_ms": wall_ms,
                        "total_cuda_ms": timing.total_device_ms,
                        "visual_cuda_ms": visual_cuda,
                        "component_cuda_ms": components,
                        "component_counts": dict(timing.component_counts),
                    },
                    "work": work,
                }
            )

        for path in PATHS:
            for repetition in range(2):
                execute_timing(path, "warmup", repetition, 0)
        for repetition in range(12):
            order = COUNTERBALANCE[repetition % len(COUNTERBALANCE)]
            for position, path in enumerate(order):
                execute_timing(path, "timed", repetition, position)
        if tuple(budget.labels) != expected_query_labels():
            raise RuntimeError("V3-C did not complete the exact 64-query identity schedule")
        if len(timing_records) != WARMUP_QUERIES + TIMED_QUERIES:
            raise RuntimeError("V3-C timing record count is incomplete")

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
            raise RuntimeError("Checkpoint inventory changed during V3-C")
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
            raise RuntimeError("V3-C artifact cap exceeded")
        store.write_once(f"{RUN_ID}/final", result)
        return 0 if result["status"] == "pass" else 2
    except BaseException as error:
        terminal_error = error
        failure = {
            "schema_version": "acr.v3-c-failure.v1",
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
            print(f"V3-C checkpoint restoration error: {restoration_error}", file=sys.stderr)
        if terminal_error is not None:
            print(
                f"V3-C failed closed after {len(budget.labels)} queries: {terminal_error}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    raise SystemExit(main())
