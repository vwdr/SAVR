#!/usr/bin/env python3
"""Execute one backend attempt of the frozen V5-D real-tensor gate.

This script is launched only by ``scripts/launch_acr_v5_d.sh`` after a
user-coordinated immutable GPU manifest exists.  It never runs a simulator or
opens task outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import types
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_ROOT = Path("/home/ved/SAVR")
CHECKPOINT_RELATIVE = Path("checkpoints/openvla-7b-oft-libero-four-suite")
OPENVLA_REVISION = "e4287e94541f459edc4feabc4e181f537cd569a8"
LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
CHECKPOINT_REVISION = "638918f3d1c2e43a39a8a20772bdb8b91835e4b7"
INSTRUCTION = "move the robot safely to the target"
TECHNICAL_FALLBACK_EXIT = 20


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def semantic_sha256(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(path: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *arguments], text=True).strip()


def require_clean_revision(path: Path, expected: str | None = None) -> str:
    revision = git_output(path, "rev-parse", "HEAD")
    if expected is not None and revision != expected:
        raise RuntimeError(f"Expected revision {expected} at {path}, found {revision}")
    if git_output(path, "status", "--porcelain"):
        raise RuntimeError(f"Refusing dirty V5-D source tree: {path}")
    return revision


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def selected_gpu_snapshot(physical_id: str) -> dict[str, Any]:
    fields = "index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu"
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={physical_id}",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError("Selected aggregate GPU snapshot did not resolve one device")
    values = [item.strip() for item in rows[0].split(",")]
    if len(values) != 7 or values[0] != physical_id:
        raise RuntimeError("Selected aggregate GPU identity changed")
    return {
        "index": int(values[0]),
        "uuid": values[1],
        "name": values[2],
        "driver_version": values[3],
        "memory_total_mib": int(values[4]),
        "memory_used_mib": int(values[5]),
        "utilization_percent": int(values[6]),
        "recorded_at_utc": utc_now(),
    }


def deterministic_inputs(np: Any, size: int = 256) -> dict[str, tuple[Any, Any]]:
    y, x = np.indices((size, size))
    scene = np.stack(((x + y) % 256, x % 256, y % 256), axis=-1).astype(np.uint8)
    wrist = np.stack(((2 * x + y) % 256, (x + 3 * y) % 256, (255 - x) % 256), axis=-1).astype(
        np.uint8
    )
    scene_b = np.ascontiguousarray(scene[:, ::-1, (2, 0, 1)])
    wrist_b = np.ascontiguousarray(wrist[::-1, :, (1, 2, 0)])
    return {"input-a": (scene, wrist), "input-b": (scene_b, wrist_b)}


def array_sha256(value: Any, np: Any) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def tensor_sha256(value: Any) -> str:
    raw = value.detach().float().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def numeric_comparison(
    reference: Any,
    candidate: Any,
    *,
    np: Any,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    left = np.asarray(reference)
    right = np.asarray(candidate)
    metadata = left.shape == right.shape
    finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
    close = metadata and finite and bool(np.allclose(left, right, rtol=rtol, atol=atol))
    maximum = None if not metadata or not left.size else float(np.max(np.abs(left - right)))
    return {
        "passed": close,
        "shape": list(left.shape),
        "rtol": rtol,
        "atol": atol,
        "maximum_absolute_difference": maximum,
        "reference_sha256": array_sha256(left, np),
        "candidate_sha256": array_sha256(right, np),
    }


def tensor_comparison(
    reference: Any,
    candidate: Any,
    *,
    torch: Any,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    metadata = (
        tuple(reference.shape) == tuple(candidate.shape)
        and reference.dtype == candidate.dtype
        and reference.device == candidate.device
    )
    close = (
        metadata and bool(torch.isfinite(reference).all()) and bool(torch.isfinite(candidate).all())
    )
    close = close and bool(torch.allclose(reference, candidate, rtol=rtol, atol=atol))
    difference = None
    if tuple(reference.shape) == tuple(candidate.shape):
        difference = float((reference.float() - candidate.float()).abs().max().item())
    return {
        "passed": close,
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "device": str(reference.device),
        "rtol": rtol,
        "atol": atol,
        "maximum_absolute_difference": difference,
    }


def gripper_decisions(actions: Any, np: Any) -> list[bool]:
    values = np.asarray(actions)
    if values.shape != (8, 7):
        raise RuntimeError("V5-D final action shape changed")
    return [bool(value > 0.0) for value in values[:, -1]]


def flatten_counters(counters: Any) -> dict[str, int]:
    flat: dict[str, int] = {}
    for group, values in dict(counters).items():
        for key, value in dict(values).items():
            if isinstance(value, (int, float)):
                flat[f"{group}.{key}"] = int(value)
    return flat


def counter_delta(after: dict[str, int], before: dict[str, int], suffix: str) -> int:
    return sum(value - before.get(key, 0) for key, value in after.items() if key.endswith(suffix))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("torch-compile", "raw-cudagraph"), required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing V5-D execution outside {EXPECTED_ROOT}: {root}")
    required_environment = (
        "CUDA_VISIBLE_DEVICES",
        "SAVR_PHYSICAL_GPU_ID",
        "SAVR_SELECTED_GPU_UUID",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TORCH_HOME",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
    )
    if any(not os.environ.get(name) for name in required_environment):
        raise SystemExit("V5-D launch environment is incomplete")
    physical_id = os.environ["SAVR_PHYSICAL_GPU_ID"]
    if os.environ["CUDA_VISIBLE_DEVICES"] != physical_id:
        raise SystemExit("V5-D physical GPU ID must equal CUDA_VISIBLE_DEVICES")
    if any(
        not Path(os.environ[name])
        .resolve()
        .is_relative_to(root / "results" / "acr-v5d-real-tensor-feasibility-v01")
        for name in (
            "HF_HOME",
            "HF_HUB_CACHE",
            "TORCH_HOME",
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
        )
    ):
        raise SystemExit("V5-D caches must stay beneath the immutable run directory")

    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts"))
    from savr.acr.records import ImmutableRecordStore
    from savr.acr.v5_d_runtime import (
        BackendKind,
        BackendWaterfall,
        FrozenQueryLedger,
        MemorySnapshot,
        ResourceEnvelope,
        TechnicalReason,
        V5DEagerReuseExecutor,
        V5DStaticBufferReuseExecutor,
        load_v5_d_freeze,
    )

    config = load_v5_d_freeze(root)
    run_id = config["run_id"]
    run_root = root / "results" / run_id
    store = ImmutableRecordStore(root / "results")
    launch_path = run_root / "launch" / "record.json"
    if not launch_path.is_file():
        raise SystemExit("V5-D immutable launch manifest is missing")
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    if launch.get("semantic_sha256") != semantic_sha256(launch):
        raise SystemExit("V5-D launch-manifest semantic hash mismatch")
    source_revision = require_clean_revision(root)
    if source_revision != launch["execution_revision"]:
        raise SystemExit("V5-D execution revision differs from the launch manifest")
    upstream_root = root / "third_party/openvla-oft"
    libero_root = root / "third_party/LIBERO"
    checkpoint = root / CHECKPOINT_RELATIVE
    require_clean_revision(upstream_root, OPENVLA_REVISION)
    require_clean_revision(libero_root, LIBERO_REVISION)
    expected_hash_paths = {
        **{
            "conda_explicit_sha256": root / "environment/locks/conda-linux-64-explicit.txt",
            "pip_freeze_sha256": root / "environment/locks/pip-freeze.txt",
            "phase1_environment_sha256": root / "environment/phase1-conda.yml",
            "v5_c_runtime_sha256": root / "reports/runtime/acr_v5_c_cpu_executor_verification.json",
            "v5_c_freeze_sha256": root / "configs/acr/v5_c_cpu_executor_freeze.json",
            "v3_c_runtime_sha256": root / "reports/runtime/acr_v3_c.json",
        },
        **{
            "config_json_sha256": checkpoint / "config.json",
            "configuration_prismatic_sha256": checkpoint / "configuration_prismatic.py",
            "modeling_prismatic_sha256": checkpoint / "modeling_prismatic.py",
        },
        **{
            "model_source_sha256": upstream_root / "prismatic/extern/hf/modeling_prismatic.py",
            "openvla_utils_sha256": upstream_root / "experiments/robot/openvla_utils.py",
            "action_heads_sha256": upstream_root / "prismatic/models/action_heads.py",
        },
    }
    expected_hashes = {
        **config["environment_hashes"],
        **config["checkpoint_hashes"],
        **config["upstream_source_hashes"],
    }
    observed_hashes = {name: file_sha256(path) for name, path in expected_hash_paths.items()}
    if observed_hashes != expected_hashes:
        raise SystemExit("V5-D pinned file hash mismatch")
    gpu = selected_gpu_snapshot(physical_id)
    if (
        gpu["uuid"] != os.environ["SAVR_SELECTED_GPU_UUID"]
        or gpu["uuid"] != launch["selected_gpu"]["uuid"]
        or gpu["memory_used_mib"] > config["gpu_selection"]["maximum_memory_used_mib_each_sample"]
        or gpu["utilization_percent"]
        > config["gpu_selection"]["maximum_utilization_percent_each_sample"]
    ):
        raise SystemExit("V5-D selected GPU is no longer eligible")

    os.chdir(upstream_root)
    sys.path.insert(0, str(upstream_root))
    import numpy as np
    import torch  # type: ignore[import-not-found]
    from experiments.robot.libero import run_libero_eval as upstream_eval
    from experiments.robot.openvla_utils import normalize_proprio, prepare_images_for_vla
    from experiments.robot.robot_utils import set_seed_everywhere
    from run_acr_correctness import validate_checkpoint
    from savr.acr.batched_dual_path import BatchedFullRefreshAdapter
    from savr.acr.instrumentation import CameraInstrumentation
    from savr.acr.isolated_controller import IsolatedACRController
    from savr.acr.reuse_executor import (
        EAGER_REUSE_EXECUTOR_VERSION,
        STATIC_REUSE_EXECUTOR_VERSION,
        ReuseCompatibilityKey,
        ReuseExecutionInputs,
    )
    from savr.acr.signals import normalized_eef_position, prepare_scene_representation
    from savr.acr.types import ACRConfiguration, ACRContext, ACRPolicy
    from savr.acr.v5_d_torch_backend import (
        RawCudaGraphCorePair,
        TorchCompileCorePair,
        TorchStaticTensorOperations,
        build_openvla_core_functions,
    )
    from savr.timing import ModuleTimingHooks, SynchronizedQueryTimer, TorchCudaEventBackend

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("V5-D requires exactly one visible logical CUDA device")
    checkpoint_before = validate_checkpoint(root, checkpoint)
    protected_names = ("config.json", "configuration_prismatic.py", "modeling_prismatic.py")
    protected_bytes = {name: (checkpoint / name).read_bytes() for name in protected_names}
    checkpoint_names = {item.name for item in checkpoint.iterdir()}

    def restore_checkpoint() -> dict[str, Any]:
        for name, payload in protected_bytes.items():
            (checkpoint / name).write_bytes(payload)
        removed = []
        for item in checkpoint.iterdir():
            if item.name not in checkpoint_names and (
                item.name.endswith(".bak") or "backup" in item.name.lower()
            ):
                if not item.is_file():
                    raise RuntimeError("Unexpected non-file checkpoint loader artifact")
                item.unlink()
                removed.append(item.name)
        hashes = {name: file_sha256(checkpoint / name) for name in protected_names}
        expected = {
            name: hashlib.sha256(payload).hexdigest() for name, payload in protected_bytes.items()
        }
        unexpected = sorted(
            item.name for item in checkpoint.iterdir() if item.name not in checkpoint_names
        )
        if hashes != expected or unexpected:
            raise RuntimeError("V5-D checkpoint restoration failed")
        return {"hashes": hashes, "removed_loader_backups": sorted(removed)}

    process_token = f"{os.getpid()}-{uuid.uuid4().hex}"
    waterfall = BackendWaterfall(config)
    selection_seconds = (
        datetime.fromisoformat(launch["finished_at_utc"])
        - datetime.fromisoformat(launch["started_at_utc"])
    ).total_seconds()
    prior_attempt_seconds = 0.0
    prior = None
    if args.backend == "torch-compile":
        waterfall.begin(BackendKind.TORCH_COMPILE, process_token=process_token)
    else:
        permit_path = run_root / "raw-transition-permit" / "record.json"
        attempt_path = run_root / "backend-attempt-torch-compile" / "record.json"
        if not permit_path.is_file() or not attempt_path.is_file():
            raise SystemExit("Raw V5-D backend lacks the frozen technical transition evidence")
        prior = json.loads(attempt_path.read_text(encoding="utf-8"))
        permit = json.loads(permit_path.read_text(encoding="utf-8"))
        if (
            prior.get("correctness_records") != 0
            or prior.get("timing_records") != 0
            or permit.get("permitted") is not True
        ):
            raise SystemExit("Raw V5-D transition occurred after output")
        if (
            int(prior["peak_reserved_bytes"])
            > int(config["memory"]["peak_reserved_gib_max"]) * 1024**3
        ):
            raise SystemExit("Raw V5-D transition cannot bypass the peak memory cap")
        prior_attempt_seconds = float(prior["wall_seconds"])
        waterfall.begin(BackendKind.TORCH_COMPILE, process_token=prior["process_token"])
        for label in prior["preparation_labels"]:
            waterfall.record_preparation_launch(label)
        reason = TechnicalReason(prior["technical_reason"])
        if not waterfall.technical_failure(reason, prior["message"]):
            raise SystemExit("Raw V5-D transition is not mechanically permitted")
        waterfall.begin(BackendKind.RAW_CUDAGRAPH, process_token=process_token)

    ledger = FrozenQueryLedger(config)
    model = action_head = proprio_projector = noisy_projector = processor = None
    started = time.monotonic()
    resources = ResourceEnvelope(
        config,
        artifact_bytes=lambda: directory_size(run_root),
        elapsed_before=selection_seconds + prior_attempt_seconds,
    )
    queries: list[dict[str, Any]] = []
    preparation_labels: list[str] = []
    terminal_error: BaseException | None = None
    try:
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
        model, action_head, proprio_projector, noisy_projector, processor = (
            upstream_eval.initialize_model(cfg)
        )
        if model is None or action_head is None or proprio_projector is None or processor is None:
            raise RuntimeError("V5-D pinned model components are incomplete")
        model.eval()
        torch.cuda.synchronize()
        if int(model.vision_backbone.get_num_patches()) != 256:
            raise RuntimeError("V5-D pinned patch count changed")
        stats = model.norm_stats[cfg.unnorm_key]["proprio"]
        q01 = np.asarray(stats["q01"], dtype=np.float64)
        q99 = np.asarray(stats["q99"], dtype=np.float64)
        state = (q01 + q99) / 2
        q01_values = tuple(float(value) for value in q01)
        q99_values = tuple(float(value) for value in q99)
        inputs = deterministic_inputs(np)
        expected_inputs = config["inputs"]
        observed_inputs = {
            "pattern_a_scene_sha256": array_sha256(inputs["input-a"][0], np),
            "pattern_a_wrist_sha256": array_sha256(inputs["input-a"][1], np),
            "pattern_b_scene_sha256": array_sha256(inputs["input-b"][0], np),
            "pattern_b_wrist_sha256": array_sha256(inputs["input-b"][1], np),
            "state_midpoint_sha256": array_sha256(state, np),
            "instruction_sha256": hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
        }
        if any(observed_inputs[name] != expected_inputs[name] for name in observed_inputs):
            raise RuntimeError("V5-D deterministic input identity changed")

        def observation(label: str) -> dict[str, Any]:
            scene, wrist = inputs[label]
            return {"full_image": scene.copy(), "wrist_image": wrist.copy(), "state": state.copy()}

        def raw_upstream(label: str) -> Any:
            return upstream_eval.get_action(
                cfg,
                model,
                observation(label),
                INSTRUCTION,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_projector,
                use_film=False,
            )

        prompt = f"In: What action should the robot take to {INSTRUCTION.lower()}?\nOut:"

        def prepare_reuse(label: str, scene_tokens: Any) -> tuple[ReuseExecutionInputs, Any, int]:
            wrist_image = prepare_images_for_vla([inputs[label][1]], cfg)[0]
            prepared = processor(prompt, wrist_image).to("cuda:0", dtype=torch.bfloat16)
            input_ids = prepared["input_ids"]
            attention_mask = prepared["attention_mask"]
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (
                        input_ids,
                        torch.tensor([[29871]], device=input_ids.device, dtype=input_ids.dtype),
                    ),
                    dim=1,
                )
            number_of_prompt_tokens = int(input_ids.shape[-1]) - 1
            labels = input_ids.clone()
            labels[:] = -100
            prepared_ids, prepared_mask = model._prepare_input_for_action_prediction(
                input_ids, attention_mask
            )
            labels = model._prepare_labels_for_action_prediction(labels, prepared_ids)
            embeddings = model.get_input_embeddings()(prepared_ids)
            action_mask = model._process_action_masks(labels)
            proprio_np = normalize_proprio(state.copy(), stats)
            proprio = torch.as_tensor(proprio_np, device="cuda:0", dtype=torch.bfloat16).reshape(
                1, 8
            )
            reuse = ReuseExecutionInputs(
                compatibility_key=None,
                wrist_pixels=prepared["pixel_values"],
                cached_scene_tokens=scene_tokens,
                prompt_input=prepared_ids,
                prompt_embeddings=embeddings,
                attention_mask=prepared_mask,
                proprioception=proprio,
            )
            return reuse, action_mask, number_of_prompt_tokens

        placeholder_scene = torch.zeros((1, 256, 4096), dtype=torch.bfloat16, device="cuda:0")
        prepared_a, action_mask, prompt_tokens = prepare_reuse("input-a", placeholder_scene)
        if (
            tuple(prepared_a.wrist_pixels.shape) != (1, 6, 224, 224)
            or tuple(prepared_a.prompt_input.shape) != (1, 79)
            or tuple(prepared_a.prompt_embeddings.shape) != (1, 79, 4096)
            or tuple(prepared_a.attention_mask.shape) != (1, 79)
        ):
            raise RuntimeError("V5-D prepared static tensor shape changed")
        eager_cores = build_openvla_core_functions(
            torch_module=torch,
            model=model,
            action_head=action_head,
            proprio_projector=proprio_projector,
            all_actions_mask=action_mask,
            number_of_prompt_tokens=prompt_tokens,
        )
        torch_ops = TorchStaticTensorOperations(torch)

        def make_key(version: str) -> ReuseCompatibilityKey:
            return ReuseCompatibilityKey(
                checkpoint_id=CHECKPOINT_REVISION,
                upstream_revision=OPENVLA_REVISION,
                configuration_id="v5-a100-b40",
                controller_version="acr-isolated-controller-v1",
                executor_version=version,
                preprocessing_id="openvla-center-crop-v1",
                action_head_id="l1-regression-8x7",
                instruction_sha256=config["inputs"]["instruction_sha256"],
                prompt_input_shape=(1, 79),
                dtype="torch.bfloat16",
                device="cuda:0",
                image_height=224,
                image_width=224,
                patch_count=256,
                projected_dimension=4096,
                wrist_shape=(1, 6, 224, 224),
                cached_scene_shape=(1, 256, 4096),
                embedding_shape=(1, 79, 4096),
                attention_mask_shape=(1, 79),
                proprioception_shape=(1, 8),
                action_shape=(1, 8, 7),
                model_training_state=False,
                use_film=False,
                use_diffusion=False,
            )

        eager_executor = V5DEagerReuseExecutor(
            prompt_input_dtype=str(prepared_a.prompt_input.dtype),
            attention_mask_dtype=str(prepared_a.attention_mask.dtype),
            tensor_ops=torch_ops,
            wrist_visual_core=eager_cores.wrist,
            downstream_action_core=eager_cores.downstream,
        )
        eager_executor.prepare(make_key(EAGER_REUSE_EXECUTOR_VERSION))
        resources.set_eager_baseline(
            MemorySnapshot(int(torch.cuda.memory_allocated()), int(torch.cuda.memory_reserved()))
        )
        counters_before = flatten_counters(torch._dynamo.utils.counters)
        if args.backend == "torch-compile":
            pair = TorchCompileCorePair(torch_module=torch, eager=eager_cores)
        else:
            pair = RawCudaGraphCorePair(
                torch_module=torch,
                eager=eager_cores,
                cat_into=lambda destination, values: torch_ops.cat_into(destination, values, dim=1),
            )
        static_executor = V5DStaticBufferReuseExecutor(
            prompt_input_dtype=str(prepared_a.prompt_input.dtype),
            attention_mask_dtype=str(prepared_a.attention_mask.dtype),
            tensor_ops=torch_ops,
            wrist_visual_core=pair.wrist,
            downstream_action_core=pair.downstream,
        )
        static_key = make_key(STATIC_REUSE_EXECUTOR_VERSION)
        static_executor.prepare(static_key)
        owned = static_executor.owned_buffers_for_backend_preparation()
        for name, source in (
            ("wrist_pixels", prepared_a.wrist_pixels),
            ("cached_scene_tokens", prepared_a.cached_scene_tokens),
            ("prompt_input", prepared_a.prompt_input),
            ("prompt_embeddings", prepared_a.prompt_embeddings),
            ("attention_mask", prepared_a.attention_mask),
            ("proprioception", prepared_a.proprioception),
        ):
            torch_ops.copy_(owned[name], source)
        if args.backend == "raw-cudagraph":
            for core in ("wrist", "downstream"):
                for repetition in range(4):
                    label = f"raw-{core}-{repetition}"
                    waterfall.record_preparation_launch(label)
                    preparation_labels.append(label)
        pair.prepare(owned)

        if args.backend == "torch-compile":
            prepared_a = types.SimpleNamespace(**vars(prepared_a))
            prepared_a.compatibility_key = static_key
            for repetition in range(2):
                for name, source in (
                    ("wrist_pixels", prepared_a.wrist_pixels),
                    ("cached_scene_tokens", prepared_a.cached_scene_tokens),
                    ("prompt_input", prepared_a.prompt_input),
                    ("prompt_embeddings", prepared_a.prompt_embeddings),
                    ("attention_mask", prepared_a.attention_mask),
                    ("proprioception", prepared_a.proprioception),
                ):
                    torch_ops.copy_(owned[name], source)
                label = f"compile-wrist-{repetition}"
                waterfall.record_preparation_launch(label)
                preparation_labels.append(label)
                pair.wrist(owned["wrist_pixels"], owned["wrist_tokens"])
                torch_ops.cat_into(
                    owned["combined_tokens"],
                    (owned["cached_scene_tokens"], owned["wrist_tokens"]),
                    dim=1,
                )
                label = f"compile-downstream-{repetition}"
                waterfall.record_preparation_launch(label)
                preparation_labels.append(label)
                pair.downstream(
                    owned["combined_tokens"],
                    owned["prompt_embeddings"],
                    owned["attention_mask"],
                    owned["proprioception"],
                    owned["normalized_actions"],
                )
                torch.cuda.synchronize()
                if repetition == 0:
                    counters_first = flatten_counters(torch._dynamo.utils.counters)
            counters_second = flatten_counters(torch._dynamo.utils.counters)
            graph_breaks = counter_delta(counters_second, counters_before, "graph_break")
            first_graphs = counter_delta(counters_first, counters_before, "unique_graphs")
            second_graphs = counter_delta(counters_second, counters_first, "unique_graphs")
            if graph_breaks:
                raise RuntimeError("V5D_FULL_GRAPH_CAPTURE_ERROR: compiler graph break")
            if first_graphs < 2:
                raise RuntimeError("V5D_VERIFIED_EAGER_FALLBACK: fewer than two compiled graphs")
            if first_graphs > 2 or second_graphs:
                raise RuntimeError("V5D_STATIC_KEY_RECOMPILE: compiler graph count changed")
            compiler_counters = {
                "before": counters_before,
                "after_first": counters_first,
                "after_second": counters_second,
            }
        else:
            compiler_counters = None
        resources.observe_memory(
            MemorySnapshot(int(torch.cuda.memory_allocated()), int(torch.cuda.memory_reserved()))
        )
        resources.check_host_resources()

        selected_config = ACRConfiguration(
            "v5-a100-b40",
            ACRPolicy.SA_ACR,
            scene_threshold=0.30046895424836606,
            translation_threshold=0.685919037527938,
            horizon=1,
            hard_reuse_cap=0.40,
            controller_version="acr-isolated-controller-v1",
        )

        def context(label: str) -> ACRContext:
            return ACRContext(
                episode_id=f"v5d-{label}",
                attempt_id=f"{run_id}-{label}",
                task_id="synthetic",
                instruction_sha256=config["inputs"]["instruction_sha256"],
                checkpoint_id=CHECKPOINT_REVISION,
                upstream_revision=OPENVLA_REVISION,
                configuration_id="v5-a100-b40",
                controller_version="acr-isolated-controller-v1",
                preprocessing_id="openvla-center-crop-v1",
                action_head_id="l1-regression-8x7",
                dtype="torch.bfloat16",
                device="cuda:0",
                patch_count=256,
            )

        zero_action = tuple(0.0 for _ in range(56))

        def primed_controller(label: str, input_label: str, *, post_reuse: bool):
            controller = IsolatedACRController(selected_config)
            ctx = context(label)
            controller.reset(ctx)
            scene = prepare_scene_representation(inputs[input_label][0])
            position = normalized_eef_position(state, q01_values, q99_values)
            for index in range(2):
                decision = controller.decide(
                    scene_representation=scene,
                    normalized_eef_position=position,
                    cache_available=index > 0,
                    cache_age=0,
                )
                if not decision.refresh:
                    raise RuntimeError("V5-D untimed controller priming must refresh")
                controller.observe(
                    decision=decision,
                    scene_representation=scene,
                    normalized_eef_position=position,
                    action_chunk=zero_action,
                )
            cache_age = 0
            if post_reuse:
                reuse = controller.decide(
                    scene_representation=scene,
                    normalized_eef_position=position,
                    cache_available=True,
                    cache_age=0,
                )
                if reuse.refresh:
                    raise RuntimeError("V5-D post-reuse priming did not select reuse")
                controller.observe(
                    decision=reuse,
                    scene_representation=scene,
                    normalized_eef_position=position,
                    action_chunk=zero_action,
                )
                cache_age = 1
            return controller, ctx, scene, position, cache_age

        def bfr_call(input_label: str, *, with_refresh_controller: bool, label: str):
            captured = []
            timer = SynchronizedQueryTimer(TorchCudaEventBackend(torch))
            hooks = ModuleTimingHooks(
                {
                    "siglip": model.vision_backbone.featurizer,
                    "dinov2": model.vision_backbone.fused_featurizer,
                    "projector": model.projector,
                },
                timer,
            )
            adapter = BatchedFullRefreshAdapter(
                model=model,
                tensor_ops=TorchStaticTensorOperationsForBatched(torch),
                instrumentation=CameraInstrumentation(timer=timer),
                projected_tokens_observer=lambda value: captured.append(value.detach().clone()),
                correctness_mode=label.startswith("a-") or label.startswith("b-"),
                action_finite_checker=lambda value: bool(np.isfinite(np.asarray(value)).all()),
            )
            controller_record = None
            primed = None
            if with_refresh_controller:
                primed = primed_controller(label, input_label, post_reuse=True)
            try:
                with adapter.episode(context(label)):
                    started_wall = time.perf_counter()
                    if with_refresh_controller:
                        controller, _, scene, position, age = primed
                        decision = controller.decide(
                            scene_representation=scene,
                            normalized_eef_position=position,
                            cache_available=True,
                            cache_age=age,
                        )
                        if not decision.refresh or "post-reuse-refresh" not in decision.reasons:
                            raise RuntimeError("V5-D refresh path lacks the mandatory latch")
                    result = adapter.run_query(lambda: raw_upstream(input_label))
                    if with_refresh_controller:
                        controller.observe(
                            decision=decision,
                            scene_representation=scene,
                            normalized_eef_position=position,
                            action_chunk=np.asarray(result.value).reshape(-1),
                        )
                        controller_record = asdict(decision)
            finally:
                hooks.remove()
            wall_ms = (time.perf_counter() - started_wall) * 1000.0
            if len(captured) != 1 or result.device_timing is None:
                raise RuntimeError("V5-D Batched-FR capture is incomplete")
            components = dict(result.device_timing.component_device_ms)
            visual = sum(
                float(components.get(name, 0.0)) for name in ("siglip", "dinov2", "projector")
            )
            return {
                "actions": np.asarray(result.value),
                "combined": captured[0],
                "timing": {
                    "wall_ms": wall_ms,
                    "total_cuda_ms": float(result.device_timing.total_device_ms),
                    "visual_cuda_ms": visual,
                    "sequential_cuda_ms": float(result.device_timing.total_device_ms),
                    "wrist_cuda_ms": visual / 2.0,
                    "downstream_cuda_ms": max(
                        0.0, float(result.device_timing.total_device_ms) - visual
                    ),
                },
                "work": asdict(result.work),
                "decision": controller_record,
            }

        class TorchStaticTensorOperationsForBatched:
            def __init__(self, torch_module):
                self.torch = torch_module

            def split(self, value, sections, *, dim):
                return tuple(self.torch.split(value, list(sections), dim=dim))

            def cat(self, values, *, dim):
                return self.torch.cat(tuple(values), dim=dim)

            @staticmethod
            def reshape(value, shape):
                return value.reshape(tuple(shape))

            def all_finite(self, value):
                return bool(self.torch.isfinite(value).all().item())

        def reuse_call(input_label: str, *, optimized: bool, scene_tokens: Any, label: str):
            controller, _, scene, position, age = primed_controller(
                label, input_label, post_reuse=False
            )
            started_wall = time.perf_counter()
            total_start = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)
            wrist_start = torch.cuda.Event(enable_timing=True)
            wrist_end = torch.cuda.Event(enable_timing=True)
            downstream_start = torch.cuda.Event(enable_timing=True)
            downstream_end = torch.cuda.Event(enable_timing=True)
            total_start.record()
            prepared, current_mask, current_prompt_tokens = prepare_reuse(input_label, scene_tokens)
            if not torch.equal(current_mask, action_mask) or current_prompt_tokens != prompt_tokens:
                raise RuntimeError("V5-D current prompt/action mask changed")
            decision = controller.decide(
                scene_representation=scene,
                normalized_eef_position=position,
                cache_available=True,
                cache_age=age,
            )
            if decision.refresh:
                raise RuntimeError("V5-D reuse path selected refresh")
            executor = static_executor if optimized else eager_executor
            key = static_key if optimized else make_key(EAGER_REUSE_EXECUTOR_VERSION)
            prepared = ReuseExecutionInputs(
                compatibility_key=key,
                wrist_pixels=prepared.wrist_pixels,
                cached_scene_tokens=prepared.cached_scene_tokens,
                prompt_input=prepared.prompt_input,
                prompt_embeddings=prepared.prompt_embeddings,
                attention_mask=prepared.attention_mask,
                proprioception=prepared.proprioception,
            )
            original_wrist = executor.wrist_visual_core
            original_downstream = executor.downstream_action_core

            def timed_wrist(pixels, output):
                wrist_start.record()
                original_wrist(pixels, output)
                wrist_end.record()

            def timed_downstream(combined, embeddings, mask, proprio, output):
                downstream_start.record()
                original_downstream(combined, embeddings, mask, proprio, output)
                downstream_end.record()

            executor.wrist_visual_core = timed_wrist
            executor.downstream_action_core = timed_downstream
            try:
                result = executor.run(prepared)
            finally:
                executor.wrist_visual_core = original_wrist
                executor.downstream_action_core = original_downstream
            normalized_float = result.normalized_actions.float()
            host = torch.empty((1, 8, 7), dtype=torch.float32, device="cpu", pin_memory=True)
            host.copy_(normalized_float, non_blocking=True)
            total_end.record()
            total_end.synchronize()
            normalized_np = host.numpy().copy()
            final_actions = model._unnormalize_actions(normalized_np.reshape(8, 7), cfg.unnorm_key)
            controller.observe(
                decision=decision,
                scene_representation=scene,
                normalized_eef_position=position,
                action_chunk=np.asarray(final_actions).reshape(-1),
            )
            wall_ms = (time.perf_counter() - started_wall) * 1000.0
            wrist_ms = float(wrist_start.elapsed_time(wrist_end))
            downstream_ms = float(downstream_start.elapsed_time(downstream_end))
            return {
                "actions": np.asarray(final_actions),
                "normalized": normalized_np,
                "wrist": result.wrist_tokens.detach().clone(),
                "combined": result.combined_tokens.detach().clone(),
                "timing": {
                    "wall_ms": wall_ms,
                    "total_cuda_ms": float(total_start.elapsed_time(total_end)),
                    "visual_cuda_ms": wrist_ms,
                    "wrist_cuda_ms": wrist_ms,
                    "downstream_cuda_ms": downstream_ms,
                    "sequential_cuda_ms": wrist_ms + downstream_ms,
                },
                "work": asdict(result.snapshot.work),
                "buffer_identities": list(result.snapshot.buffer_identities),
                "decision": asdict(decision),
            }

        def write_query(record: dict[str, Any]) -> None:
            record["semantic_sha256"] = semantic_sha256(record)
            store.write_once(f"{run_id}/query-{record['query_index']:03d}", record)
            queries.append(record)
            resources.observe_memory(
                MemorySnapshot(
                    int(torch.cuda.memory_allocated()), int(torch.cuda.memory_reserved())
                )
            )
            resources.check_host_resources()

        waterfall.begin_correctness()
        oracles = {}
        eager_refs = {}
        optimized_a = None
        for label in config["correctness"]["labels"]:
            identity = ledger.consume(label)
            if label.endswith("batched-fr"):
                value = bfr_call(identity.input_label, with_refresh_controller=False, label=label)
                oracles[identity.input_label] = value
                checks = {
                    "combined_shape": tuple(value["combined"].shape) == (1, 512, 4096),
                    "actions_shape": tuple(value["actions"].shape) == (8, 7),
                    "finite": bool(torch.isfinite(value["combined"]).all())
                    and bool(np.isfinite(value["actions"]).all()),
                    "work": value["work"]["mode"] == "batched-fr",
                }
                details = {"combined_sha256": tensor_sha256(value["combined"])}
            elif "eager-reuse" in label:
                oracle = oracles[identity.input_label]
                value = reuse_call(
                    identity.input_label,
                    optimized=False,
                    scene_tokens=oracle["combined"][:, :256, :].detach().clone(),
                    label=label,
                )
                eager_refs[identity.input_label] = value
                token = tensor_comparison(
                    oracle["combined"], value["combined"], torch=torch, rtol=0.016, atol=1e-5
                )
                final = numeric_comparison(
                    oracle["actions"], value["actions"], np=np, rtol=1e-5, atol=1e-6
                )
                checks = {
                    "combined_tokens": token["passed"],
                    "final_actions": final["passed"],
                    "gripper": gripper_decisions(oracle["actions"], np)
                    == gripper_decisions(value["actions"], np),
                    "work": value["work"]["scene_core_calls"] == 0,
                }
                details = {"tokens": token, "final_actions": final}
            else:
                reference = eager_refs[identity.input_label]
                oracle = oracles[identity.input_label]
                value = reuse_call(
                    identity.input_label,
                    optimized=True,
                    scene_tokens=oracle["combined"][:, :256, :].detach().clone(),
                    label=label,
                )
                token = tensor_comparison(
                    reference["combined"], value["combined"], torch=torch, rtol=0.016, atol=1e-5
                )
                normalized = numeric_comparison(
                    reference["normalized"],
                    value["normalized"],
                    np=np,
                    rtol=0.001,
                    atol=0.0001,
                )
                final = numeric_comparison(
                    reference["actions"], value["actions"], np=np, rtol=1e-5, atol=1e-6
                )
                repeat = True
                if label == "a-optimized-repeat":
                    repeat = bool(
                        torch.equal(optimized_a["combined"], value["combined"])
                        and np.array_equal(optimized_a["normalized"], value["normalized"])
                        and np.array_equal(optimized_a["actions"], value["actions"])
                    )
                elif label == "a-optimized-reuse":
                    optimized_a = value
                checks = {
                    "combined_tokens": token["passed"],
                    "normalized_actions": normalized["passed"],
                    "final_actions": final["passed"],
                    "gripper": gripper_decisions(reference["actions"], np)
                    == gripper_decisions(value["actions"], np),
                    "controller": value["decision"] == reference["decision"],
                    "work": value["work"]["scene_core_calls"] == 0
                    and value["work"]["wrist_core_calls"] >= 1
                    and value["work"]["downstream_core_calls"] >= 1,
                    "a_repeat_bitwise": repeat,
                }
                details = {"tokens": token, "normalized": normalized, "final_actions": final}
            record = {
                "schema_version": "acr.v5d-query.v1",
                "run_id": run_id,
                "query_index": len(queries),
                "label": label,
                "kind": "correctness",
                "path": identity.path,
                "input_label": identity.input_label,
                "checks": checks,
                "passed": all(checks.values()),
                "details": details,
            }
            write_query(record)
            waterfall.record_correctness()
            if not record["passed"]:
                raise RuntimeError(f"V5-D correctness failed: {label}")

        waterfall.begin_warmup()
        for identity in ledger.schedule[7:15]:
            consumed = ledger.consume(identity.label)
            if consumed.path == "batched-fr":
                value = bfr_call(
                    consumed.input_label, with_refresh_controller=False, label=consumed.label
                )
            elif consumed.path == "v5-refresh":
                value = bfr_call(
                    consumed.input_label, with_refresh_controller=True, label=consumed.label
                )
            else:
                value = reuse_call(
                    consumed.input_label,
                    optimized=consumed.path == "optimized-reuse",
                    scene_tokens=oracles[consumed.input_label]["combined"][:, :256, :]
                    .detach()
                    .clone(),
                    label=consumed.label,
                )
            write_query(
                {
                    "schema_version": "acr.v5d-query.v1",
                    "run_id": run_id,
                    "query_index": len(queries),
                    "label": consumed.label,
                    "kind": "warmup",
                    "path": consumed.path,
                    "input_label": consumed.input_label,
                    "timing": value["timing"],
                }
            )

        waterfall.begin_timing()
        while ledger.next_identity is not None:
            identity = ledger.consume(ledger.next_identity.label)
            if identity.path == "batched-fr":
                value = bfr_call(
                    identity.input_label, with_refresh_controller=False, label=identity.label
                )
            elif identity.path == "v5-refresh":
                value = bfr_call(
                    identity.input_label, with_refresh_controller=True, label=identity.label
                )
            else:
                value = reuse_call(
                    identity.input_label,
                    optimized=identity.path == "optimized-reuse",
                    scene_tokens=oracles[identity.input_label]["combined"][:, :256, :]
                    .detach()
                    .clone(),
                    label=identity.label,
                )
            reference_actions = oracles[identity.input_label]["actions"]
            parity = numeric_comparison(
                reference_actions, value["actions"], np=np, rtol=1e-5, atol=1e-6
            )
            if not parity["passed"]:
                raise RuntimeError(f"V5-D timed action parity failed: {identity.label}")
            record = {
                "schema_version": "acr.v5d-query.v1",
                "run_id": run_id,
                "query_index": len(queries),
                "label": identity.label,
                "kind": "timed",
                "block": identity.block,
                "position": identity.position,
                "path": identity.path,
                "input_label": identity.input_label,
                "timing": value["timing"],
                "actions_sha256": array_sha256(value["actions"], np),
                "work": value["work"],
                "decision": value["decision"],
            }
            write_query(record)
            waterfall.record_timing()
        ledger.require_complete()
        waterfall.complete()
        restoration = restore_checkpoint()
        if validate_checkpoint(root, checkpoint) != checkpoint_before:
            raise RuntimeError("V5-D checkpoint inventory changed")
        require_clean_revision(root, source_revision)
        require_clean_revision(upstream_root, OPENVLA_REVISION)
        require_clean_revision(libero_root, LIBERO_REVISION)
        final_gpu = selected_gpu_snapshot(physical_id)
        prior_peak_allocated = 0 if prior is None else int(prior["peak_allocated_bytes"])
        prior_peak_reserved = 0 if prior is None else int(prior["peak_reserved_bytes"])
        final = {
            "schema_version": "acr.v5d-run.v1",
            "run_id": run_id,
            "status": "completed",
            "backend": args.backend,
            "configuration_semantic_sha256": config["semantic_sha256"],
            "execution_revision": source_revision,
            "launch_manifest_semantic_sha256": launch["semantic_sha256"],
            "selected_gpu": launch["selected_gpu"],
            "selected_gpu_after": final_gpu,
            "preparation_labels": preparation_labels,
            "compiler_counters": compiler_counters,
            "queries": queries,
            "query_count": len(queries),
            "correctness_pass": True,
            "memory_pass": True,
            "work_pass": True,
            "lifecycle_pass": True,
            "restoration_pass": True,
            "resource_pass": True,
            "peak_allocated_bytes": max(resources.peak.allocated_bytes, prior_peak_allocated),
            "peak_reserved_bytes": max(resources.peak.reserved_bytes, prior_peak_reserved),
            "wall_seconds": resources.elapsed_seconds,
            "checkpoint_restoration": restoration,
            "source_trees_restored": True,
            "simulator_episodes": 0,
            "simulator_resets": 0,
            "downloads": 0,
            "new_task_outcomes": 0,
            "finished_at_utc": utc_now(),
        }
        final["semantic_sha256"] = semantic_sha256(final)
        if len(queries) != 111:
            raise RuntimeError("V5-D final query count changed")
        resources.check_host_resources()
        store.write_once(f"{run_id}/final", final)
        return 0
    except BaseException as error:
        terminal_error = error
        message = str(error)
        technical_reason = None
        if not queries and args.backend == "torch-compile":
            if "V5D_FULL_GRAPH_CAPTURE_ERROR" in message:
                technical_reason = TechnicalReason.FULL_GRAPH_CAPTURE_ERROR
            elif "V5D_STATIC_KEY_RECOMPILE" in message:
                technical_reason = TechnicalReason.STATIC_KEY_RECOMPILE
            elif "V5D_VERIFIED_EAGER_FALLBACK" in message:
                technical_reason = TechnicalReason.VERIFIED_EAGER_FALLBACK
            elif isinstance(error, (MemoryError, torch.cuda.OutOfMemoryError)):
                technical_reason = TechnicalReason.PREPARATION_OOM
            else:
                technical_reason = TechnicalReason.COMPILER_CONSTRUCTION_OR_FIRST_CALL_ERROR
        permitted = False
        if technical_reason is not None:
            permitted = waterfall.technical_failure(technical_reason, message)
        elif waterfall.current_backend is not None:
            waterfall.fail_after_output(message)
        try:
            restoration = restore_checkpoint()
            restoration_error = None
        except Exception as restore_error:
            restoration = None
            restoration_error = str(restore_error)
            permitted = False
        attempt = {
            "schema_version": "acr.v5d-backend-attempt.v1",
            "run_id": run_id,
            "backend": args.backend,
            "process_token": process_token,
            "status": "technical-failure" if technical_reason else "failed",
            "technical_reason": None if technical_reason is None else technical_reason.value,
            "message": message,
            "preparation_labels": preparation_labels,
            "correctness_records": sum(item.get("kind") == "correctness" for item in queries),
            "timing_records": sum(item.get("kind") == "timed" for item in queries),
            "full_query_records": len(queries),
            "wall_seconds": time.monotonic() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "raw_transition_permitted": permitted,
            "restoration": restoration,
            "restoration_error": restoration_error,
            "finished_at_utc": utc_now(),
        }
        attempt["semantic_sha256"] = semantic_sha256(attempt)
        store.write_once(f"{run_id}/backend-attempt-{args.backend}", attempt)
        if permitted:
            permit = {
                "schema_version": "acr.v5d-raw-transition.v1",
                "run_id": run_id,
                "permitted": True,
                "from_backend": "torch-compile",
                "technical_reason": technical_reason.value,
                "compiler_attempt_semantic_sha256": attempt["semantic_sha256"],
                "correctness_records": 0,
                "timing_records": 0,
                "requires_fresh_process": True,
            }
            permit["semantic_sha256"] = semantic_sha256(permit)
            store.write_once(f"{run_id}/raw-transition-permit", permit)
            return TECHNICAL_FALLBACK_EXIT
        return 3
    finally:
        model = action_head = proprio_projector = noisy_projector = processor = None
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.synchronize()
        if terminal_error is not None:
            print(
                f"V5-D stopped: {type(terminal_error).__name__}: {terminal_error}", file=sys.stderr
            )


if __name__ == "__main__":
    raise SystemExit(main())
