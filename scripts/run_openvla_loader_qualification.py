#!/usr/bin/env python3
"""Authenticate the exact OpenVLA runtime without invoking the policy."""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib.util
import inspect
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path("/home/ved/SAVR")
DEFAULT_CONFIG = Path("configs/openvla/loader_qualification_s3l_v1.json")


class LoaderQualificationStop(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def semantic_sha256(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def extract_method_source(source: str, method_name: str) -> str:
    tree = ast.parse(source)
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    ]
    if len(matches) != 1:
        raise LoaderQualificationStop(
            f"checkpoint contains {len(matches)} definitions of {method_name}"
        )
    node = matches[0]
    if node.end_lineno is None:
        raise LoaderQualificationStop(f"checkpoint source extraction failed: {method_name}")
    segment = "\n".join(source.splitlines()[node.lineno - 1 : node.end_lineno])
    return textwrap.dedent(segment).strip()


def validate_config(config: Mapping[str, Any], *, require_authorized: bool = True) -> None:
    if config.get("schema_version") != "openvla-loader-qualification-s3l-v1":
        raise LoaderQualificationStop("loader qualification schema changed")
    if config.get("semantic_sha256") != semantic_sha256(config):
        raise LoaderQualificationStop("loader qualification config hash mismatch")
    authorization = {
        "authorized": require_authorized,
        "source": "User authorized cautious loader-only qualification on 2026-08-31"
        if require_authorized
        else None,
        "policy_calls": 0,
        "actions": 0,
        "simulator": False,
        "outcomes": False,
        "automatic_retry": False,
    }
    if config.get("authorization") != authorization:
        raise LoaderQualificationStop("loader qualification authorization changed")
    if config.get("resource_caps") != {
        "gpu_count": 1,
        "model_processes": 1,
        "policy_calls": 0,
        "vision_encoder_calls": 0,
        "action_head_calls": 0,
        "wall_seconds": 900,
        "artifact_bytes": 268435456,
        "peak_gpu_memory_mib_strict_max": 23552,
        "downloads": 0,
        "simulator_outcomes": 0,
        "automatic_retry": False,
    }:
        raise LoaderQualificationStop("loader qualification resource boundary changed")
    if config.get("advance") != {
        "next_stage": "S3_RECOVERY01",
        "authorized": False,
        "stop_before_next_stage": True,
    }:
        raise LoaderQualificationStop("loader qualification advance boundary changed")
    if config.get("output_root") != "results/openvla-loader-qualification-s3l-v01":
        raise LoaderQualificationStop("loader qualification output root changed")
    if config.get("official_source") != {
        "relative": "third_party/openvla-oft",
        "revision": "e4287e94541f459edc4feabc4e181f537cd569a8",
        "disable_model_logic_sync": True,
    }:
        raise LoaderQualificationStop("loader qualification official source changed")
    for relative, expected in config["authenticated_files"].items():
        path = ROOT / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise LoaderQualificationStop(f"authenticated loader input changed: {relative}")
    checkpoint = ROOT / config["model"]["checkpoint_relative"]
    for name, expected in config["model"]["checkpoint_metadata_sha256"].items():
        if not (checkpoint / name).is_file() or file_sha256(checkpoint / name) != expected:
            raise LoaderQualificationStop(f"loader checkpoint changed: {name}")
    source_root = ROOT / config["official_source"]["relative"]
    revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != config["official_source"]["revision"]:
        raise LoaderQualificationStop("official source revision changed")
    checkpoint_source = (checkpoint / "modeling_prismatic.py").read_text(encoding="utf-8")
    regression = extract_method_source(checkpoint_source, "_regression_or_discrete_prediction")
    prediction = extract_method_source(checkpoint_source, "predict_action")
    if (
        "NUM_PATCHES + NUM_PROMPT_TOKENS" not in regression
        or "- ACTION_DIM * NUM_ACTIONS_CHUNK - 1" in regression
        or "past_key_values" in prediction
    ):
        raise LoaderQualificationStop("checkpoint official action semantics changed")
    if (ROOT / config["output_root"]).exists():
        raise LoaderQualificationStop("immutable loader output already exists")


def selected_gpu_snapshot(gpu: int) -> dict[str, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    values = [int(item.strip()) for item in output.split(",")]
    if len(values) != 2:
        raise LoaderQualificationStop("selected-GPU telemetry changed")
    return {"memory_used_mib": values[0], "utilization_percent": values[1]}


class MemorySampler:
    def __init__(self, gpu: int, initial: int) -> None:
        self.gpu = gpu
        self.peak = initial
        self.error: BaseException | None = None
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.wait(0.2):
            try:
                value = selected_gpu_snapshot(self.gpu)["memory_used_mib"]
                with self.lock:
                    self.peak = max(self.peak, value)
            except BaseException as error:
                self.error = error
                return

    def start(self) -> None:
        self.thread.start()

    def close(self) -> int:
        self.stop.set()
        self.thread.join()
        with self.lock:
            return self.peak


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise LoaderQualificationStop(f"cannot load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f"loader qualification must start in {ROOT}")
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    validate_config(config, require_authorized=True)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    physical = os.environ.get("OPENVLA_PHYSICAL_GPU_ID", "")
    if not physical.isdigit() or visible != physical or "," in visible:
        raise SystemExit("loader qualification requires one explicit selected GPU")
    initial = selected_gpu_snapshot(int(physical))
    if initial["memory_used_mib"] > 1024 or initial["utilization_percent"] > 5:
        raise SystemExit("selected GPU is not sufficiently idle")
    output_root = ROOT / config["output_root"]
    output_root.mkdir(parents=True)
    runtime = output_root / "runtime-cache"
    for key, name in {
        "MPLCONFIGDIR": "matplotlib",
        "HF_MODULES_CACHE": "hf-modules",
        "HF_HOME": "hf-home",
        "TORCH_HOME": "torch",
        "XDG_CACHE_HOME": "xdg",
        "TMPDIR": "tmp",
        "WANDB_DIR": "wandb",
    }.items():
        path = runtime / name
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_MODE": "disabled",
            "PYTHONNOUSERSITE": "1",
        }
    )
    sys.pycache_prefix = str(runtime / "pycache")
    sys.path.insert(0, str(ROOT / "src"))
    started = time.monotonic()
    sampler = MemorySampler(int(physical), initial["memory_used_mib"])
    sampler.start()
    checkpoint = ROOT / config["model"]["checkpoint_relative"]
    checkpoint_baseline = None
    restore_fn = None
    model = action_head = proprio_projector = processor = None
    evaluation = None
    originals: dict[str, Any] = {}

    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError("loader qualification wall limit reached")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(config["resource_caps"]["wall_seconds"]))
    try:
        recovery = load_module(
            ROOT / "scripts/run_openvla_semantic_parity_recovery01.py",
            "openvla_s3_recovery_for_loader",
        )
        v1 = recovery.load_v1()
        evaluation, originals = recovery.install_official_loader_guard()
        from savr.acr.v5_d_recovery import capture_checkpoint_baseline, restore_checkpoint_exact

        restore_fn = restore_checkpoint_exact
        checkpoint_baseline = capture_checkpoint_baseline(
            checkpoint,
            ("config.json", "configuration_prismatic.py", "modeling_prismatic.py"),
        )
        cfg = v1.base_config(evaluation, checkpoint, output_root)
        evaluation.validate_config(cfg)
        model, action_head, proprio_projector, noisy, processor = evaluation.initialize_model(cfg)
        if noisy is not None:
            raise LoaderQualificationStop("loader unexpectedly created a noisy action projector")
        if model is None or action_head is None or proprio_projector is None or processor is None:
            raise LoaderQualificationStop("loader component contract is incomplete")
        import torch

        if torch.cuda.device_count() != 1:
            raise LoaderQualificationStop("loader can see an invalid GPU count")
        checkpoint_text = (checkpoint / "modeling_prismatic.py").read_text(encoding="utf-8")
        runtime_methods = {}
        for method_name in ("_regression_or_discrete_prediction", "predict_action"):
            expected = extract_method_source(checkpoint_text, method_name)
            observed = textwrap.dedent(inspect.getsource(getattr(model, method_name))).strip()
            if observed != expected:
                raise LoaderQualificationStop(f"loaded method differs from checkpoint: {method_name}")
            runtime_methods[method_name] = hashlib.sha256(observed.encode()).hexdigest()
        regression = textwrap.dedent(
            inspect.getsource(model._regression_or_discrete_prediction)
        )
        if (
            "NUM_PATCHES + NUM_PROMPT_TOKENS" not in regression
            or "- ACTION_DIM * NUM_ACTIONS_CHUNK - 1" in regression
        ):
            raise LoaderQualificationStop("loaded action readout is not prompt-derived")
        language_config = model.language_model.config
        if (
            getattr(language_config, "proportion_attn_var", "missing") is not None
            or getattr(language_config, "reusable_patches", "missing") is not None
            or hasattr(cfg, "use_vla_cache")
        ):
            raise LoaderQualificationStop("loaded dense/cache configuration contract failed")
        if file_sha256(checkpoint / "modeling_prismatic.py") != config["model"][
            "checkpoint_metadata_sha256"
        ]["modeling_prismatic.py"]:
            raise LoaderQualificationStop("model logic synchronization occurred during load")
        loaded_class = f"{model.__class__.__module__}.{model.__class__.__name__}"
        if not loaded_class.startswith("transformers_modules.openvla-7b-oft-libero-four-suite."):
            raise LoaderQualificationStop("loaded class did not originate from checkpoint module")
        for name, value in originals.items():
            setattr(evaluation, name, value)
        model = action_head = proprio_projector = processor = noisy = None
        gc.collect()
        torch.cuda.empty_cache()
        restoration = restore_fn(checkpoint, checkpoint_baseline)
        checkpoint_baseline = None
        peak = sampler.close()
        if sampler.error is not None:
            raise LoaderQualificationStop("loader memory sampler failed") from sampler.error
        final_gpu = selected_gpu_snapshot(int(physical))
        peak = max(peak, final_gpu["memory_used_mib"])
        if peak >= int(config["resource_caps"]["peak_gpu_memory_mib_strict_max"]):
            raise LoaderQualificationStop("loader strict memory boundary reached")
        if directory_size(output_root) > int(config["resource_caps"]["artifact_bytes"]):
            raise LoaderQualificationStop("loader artifact boundary reached")
        summary = {
            "schema_version": "openvla-loader-qualification-result-v1",
            "run_id": config["run_id"],
            "complete": True,
            "passed": True,
            "policy_calls": 0,
            "vision_encoder_calls": 0,
            "action_head_calls": 0,
            "actions_produced": 0,
            "simulator_outcomes_accessed": False,
            "loaded_class": loaded_class,
            "runtime_method_sha256": runtime_methods,
            "prompt_derived_action_readout": True,
            "cache_controls_injected_dense_none": True,
            "checkpoint_restoration": restoration,
            "peak_aggregate_gpu_memory_mib": peak,
            "final_selected_gpu": final_gpu,
            "elapsed_seconds": time.monotonic() - started,
            "automatic_retry": False,
            "advance": config["advance"],
            "completed_at": utc_now(),
        }
        summary["semantic_sha256"] = semantic_sha256(summary)
        write_once(output_root / "worker_summary.json", summary)
        return 0
    except BaseException as error:
        peak = sampler.close()
        restoration = None
        restoration_error = None
        if checkpoint_baseline is not None and restore_fn is not None:
            try:
                restoration = restore_fn(checkpoint, checkpoint_baseline)
                checkpoint_baseline = None
            except BaseException as restore_error:
                restoration_error = f"{type(restore_error).__name__}: {restore_error}"
        stop = {
            "schema_version": "openvla-loader-qualification-stop-v1",
            "run_id": config.get("run_id"),
            "complete": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "policy_calls": 0,
            "actions_produced": 0,
            "simulator_outcomes_accessed": False,
            "checkpoint_restoration": restoration,
            "checkpoint_restoration_error": restoration_error,
            "peak_aggregate_gpu_memory_mib": peak,
            "elapsed_seconds": time.monotonic() - started,
            "automatic_retry": False,
            "stopped_at": utc_now(),
        }
        stop["semantic_sha256"] = semantic_sha256(stop)
        write_once(output_root / "technical_stop.json", stop)
        descriptor = os.open(
            output_root / "technical_traceback.log",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(traceback.format_exc())
            stream.flush()
            os.fsync(stream.fileno())
        return 1
    finally:
        signal.alarm(0)
        if evaluation is not None:
            for name, value in originals.items():
                setattr(evaluation, name, value)
        model = action_head = proprio_projector = processor = None
        gc.collect()
        try:
            if "torch" in locals() and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except BaseException:
            pass
        if checkpoint_baseline is not None and restore_fn is not None:
            try:
                restore_fn(checkpoint, checkpoint_baseline)
            except BaseException:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
