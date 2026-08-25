#!/usr/bin/env python3
"""Run one isolated worker of the frozen BRACE-B3 real-model gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path("/home/ved/SAVR")
CONFIG = Path("configs/brace/b3_physical_v1.json")
CHECKPOINT = Path("checkpoints/openvla-7b-oft-libero-four-suite")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


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


def write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def git_output(path: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *arguments], text=True).strip()


def timed_cuda_call(torch: Any, call: Callable[[], Any]) -> tuple[Any, float, float]:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    start.record()
    value = call()
    end.record()
    torch.cuda.synchronize()
    return value, (time.perf_counter() - wall) * 1000, float(start.elapsed_time(end))


def base_config(
    eval_module: Any,
    checkpoint: Path,
    run_root: Path,
    *,
    config_class: Any | None = None,
    **changes: Any,
) -> Any:
    values = {
        "pretrained_checkpoint": str(checkpoint),
        "task_suite_name": "libero_object",
        "num_trials_per_task": 0,
        "seed": 0,
        "local_log_dir": str(run_root / "logs"),
        "use_wandb": False,
        "center_crop": True,
        "num_open_loop_steps": 8,
        "num_images_in_input": 2,
        "use_proprio": True,
        "use_l1_regression": True,
        "use_diffusion": False,
        "use_film": False,
    }
    values.update(changes)
    constructor = config_class or eval_module.GenerateConfig
    return constructor(**values)


def common_model_inputs(model: Any, np: Any) -> tuple[Any, dict[str, tuple[Any, Any]]]:
    from savr.brace.b3_openvla import deterministic_inputs

    stats = model.norm_stats["libero_object"]["proprio"]
    state = (np.asarray(stats["q01"], dtype=np.float64) + np.asarray(stats["q99"], dtype=np.float64)) / 2
    if state.shape != (8,):
        raise RuntimeError("B3 pinned proprioception shape changed")
    return state, deterministic_inputs(np)


def observation(inputs: dict[str, tuple[Any, Any]], state: Any, label: str) -> dict[str, Any]:
    scene, wrist = inputs[label]
    return {
        "full_image": scene.copy(),
        "wrist_image": wrist.copy(),
        "state": state.copy(),
    }


def run_core_fr(config: dict[str, Any], run_root: Path) -> dict[str, Any]:
    source = ROOT / "third_party/openvla-oft"
    os.chdir(source)
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from experiments.robot.libero import run_libero_eval as evaluation
    from experiments.robot.robot_utils import set_seed_everywhere
    from savr.brace.b3_openvla import action_record

    cfg = base_config(evaluation, ROOT / CHECKPOINT, run_root)
    evaluation.validate_config(cfg)
    set_seed_everywhere(0)
    model, action_head, proprio_projector, noisy_projector, processor = evaluation.initialize_model(cfg)
    model.eval()
    state, inputs = common_model_inputs(model, np)
    instruction = config["model"]["instruction"]
    records = []
    references: dict[str, Any] = {}
    for query in range(int(config["measurement"]["core_fr_queries"])):
        label = ("input-a", "input-b", "input-c")[query % 3]

        def call() -> Any:
            return evaluation.get_action(
                cfg,
                model,
                observation(inputs, state, label),
                instruction,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_projector,
                use_film=False,
            )

        actions, wall_ms, cuda_ms = timed_cuda_call(torch, call)
        array = np.asarray(actions, dtype=np.float32)
        if array.shape != (8, 7) or not np.isfinite(array).all():
            raise RuntimeError("B3 optimized FR action output changed")
        if query >= 4:
            records.append({"query": query, "label": label, "wall_ms": wall_ms, "cuda_ms": cuda_ms})
            references.setdefault(label, action_record(array, np))
    return {
        "method": "core_fr",
        "status": "completed",
        "queries": int(config["measurement"]["core_fr_queries"]),
        "timings": records,
        "action_references": references,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "transformers": __import__("transformers").__version__,
    }


def configure_dense(model: Any) -> None:
    model.language_model.config.proportion_attn_var = None
    model.language_model.config.reusable_patches = None


def configure_profile(model: Any, positions: Any, proportions: list[float], torch: Any) -> None:
    schedule = torch.zeros(32, dtype=torch.float32, device="cuda:0")
    for layer, value in zip((2, 6, 9, 11), proportions):
        schedule[layer] = float(value)
    model.language_model.config.reusable_patches = positions
    model.language_model.config.proportion_attn_var = schedule


def assert_cache_shape(cache: Any) -> None:
    if len(cache.key_cache) != 32 or len(cache.value_cache) != 32:
        raise RuntimeError("B3 cache layer count changed")
    if any(tuple(value.shape[-2:]) != (592, 128) for value in cache.key_cache):
        raise RuntimeError("B3 cache key sequence/head shape changed")
    if any(tuple(value.shape[-2:]) != (592, 128) for value in cache.value_cache):
        raise RuntimeError("B3 cache value sequence/head shape changed")


def run_cache_suite(config: dict[str, Any], run_root: Path) -> dict[str, Any]:
    source = ROOT / "third_party/vla-cache/src/openvla-oft"
    os.chdir(source)
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from experiments.robot.libero import run_libero_eval as evaluation
    from experiments.robot.openvla_utils import normalize_proprio, prepare_images_for_vla
    from experiments.robot.robot_utils import set_seed_everywhere
    from savr.brace.b3 import cycle_schedule, summarize_timings
    from savr.brace.b3_openvla import (
        SourceTracker,
        action_record,
        compare_actions,
        dense_or_cached_forward,
        ordered_profile_positions,
        patch_change_scores,
        prepare_query,
        runtime_positions,
    )

    cfg = base_config(evaluation, ROOT / CHECKPOINT, run_root, use_vla_cache=False)
    evaluation.validate_config(cfg)
    set_seed_everywhere(0)
    model, action_head, proprio_projector, noisy_projector, processor = evaluation.initialize_model(cfg)
    model.eval()
    state, inputs = common_model_inputs(model, np)
    instruction = config["model"]["instruction"]
    sidecar_layers = tuple(config["model"]["sidecar_layers"])
    parity = config["parity"]
    eligibility = config["eligibility"]
    pruning_layers = tuple(config["model"]["pruning_layers"])
    query_count = 0

    def prepared(label: str) -> Any:
        scene, wrist = inputs[label]
        return prepare_query(
            torch_module=torch,
            np=np,
            model=model,
            processor=processor,
            proprio_projector=proprio_projector,
            prepare_images=prepare_images_for_vla,
            normalize_proprio=normalize_proprio,
            cfg=cfg,
            raw_scene=scene,
            raw_wrist=wrist,
            raw_state=state,
            instruction=instruction,
        )

    def direct(
        label: str,
        *,
        cache: Any = None,
        capture: bool = False,
        profile: dict[str, Any] | None = None,
        anchor_pixels: Any = None,
        anchor_salience: Any = None,
        wrist_offset: int = 0,
    ) -> dict[str, Any]:
        nonlocal query_count
        outer_start = time.perf_counter()
        outer_cuda_start = torch.cuda.Event(enable_timing=True)
        outer_cuda_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        outer_cuda_start.record()
        prep_start = time.perf_counter()
        value = prepared(label)
        prepare_wall_ms = (time.perf_counter() - prep_start) * 1000
        positions = runtime_positions(value, torch)
        ordered = proportions = None
        gate_wall_ms = 0.0
        if profile is None:
            configure_dense(model)
        else:
            gate_start = time.perf_counter()
            scene_change = patch_change_scores(
                value.preprocessed_pixels[:, :3],
                anchor_pixels[:, :3],
                torch_module=torch,
                epsilon=float(eligibility["cosine_epsilon"]),
                weights=eligibility["patch_change_weights"],
            )
            wrist_change = patch_change_scores(
                value.preprocessed_pixels[:, 3:],
                anchor_pixels[:, 3:],
                torch_module=torch,
                epsilon=float(eligibility["cosine_epsilon"]),
                weights=eligibility["patch_change_weights"],
            )
            ordered, proportions = ordered_profile_positions(
                profile,
                scene_change=scene_change,
                wrist_change=wrist_change,
                salience=anchor_salience,
                torch_module=torch,
                wrist_offset=wrist_offset,
            )
            configure_profile(model, ordered, proportions, torch)
            gate_wall_ms = (time.perf_counter() - gate_start) * 1000
        result = dense_or_cached_forward(
            torch_module=torch,
            np=np,
            model=model,
            action_head=action_head,
            cfg=cfg,
            prepared=value,
            past_key_values=cache,
            capture_layers=sidecar_layers if capture else (),
        )
        salience = None
        sidecar_wall_ms = 0.0
        if capture:
            sidecar_start = time.perf_counter()
            salience = result["tap"].salience(
                instruction_positions=positions["instruction"],
                action_positions=positions["action"],
                visual_positions=positions["scene"] + positions["wrist"],
            )
            torch.cuda.synchronize()
            sidecar_wall_ms = (time.perf_counter() - sidecar_start) * 1000
        outer_cuda_end.record()
        torch.cuda.synchronize()
        query_count += 1
        result.update(
            {
                "total_wall_ms": (time.perf_counter() - outer_start) * 1000,
                "total_cuda_ms": float(outer_cuda_start.elapsed_time(outer_cuda_end)),
                "prepare_wall_ms": prepare_wall_ms,
                "gate_wall_ms": gate_wall_ms,
                "sidecar_wall_ms": sidecar_wall_ms,
                "prepared": value,
                "positions": positions,
                "ordered": ordered,
                "salience": salience,
            }
        )
        return result

    configure_dense(model)
    p0_timings = []
    p0_actions: dict[str, Any] = {}
    for query in range(int(config["measurement"]["cache_p0_queries"])):
        label = ("input-a", "input-b", "input-c")[query % 3]
        result = direct(label)
        assert_cache_shape(result["cache"])
        if query >= 4:
            p0_timings.append(
                {
                    "query": query,
                    "label": label,
                    "wall_ms": result["total_wall_ms"],
                    "cuda_ms": result["total_cuda_ms"],
                }
            )
            p0_actions.setdefault(label, result["actions"])

    configure_dense(model)
    sidecar_off = direct("input-a", capture=False)
    configure_dense(model)
    sidecar_on = direct("input-a", capture=True)
    sidecar_parity = compare_actions(
        sidecar_off["actions"],
        sidecar_on["actions"],
        np=np,
        rtol=float(parity["unnormalized_action_rtol"]),
        atol=float(parity["unnormalized_action_atol"]),
        exact_gripper=bool(parity["gripper_decisions_exact"]),
    )
    sidecar_parity["underlying_sdpa_calls"] = sidecar_on["tap"].calls
    sidecar_parity["sidecar_wall_ms"] = sidecar_on["sidecar_wall_ms"]
    sidecar_parity["dense_wall_overhead_fraction"] = (
        sidecar_on["total_wall_ms"] / sidecar_off["total_wall_ms"] - 1
    )
    if not sidecar_parity["passed"]:
        raise RuntimeError("B3 dense sidecar changed the anchor action")

    corrected_records = []
    corrected_parity = []
    for pair in range(10):
        last = None
        pair_records = []
        for arm in ("anchor", "reuse"):
            obs = observation(inputs, state, "input-a")
            obs["prev_images"] = [inputs["input-a"][0].copy(), inputs["input-a"][1].copy()]

            def official_call() -> Any:
                return evaluation.get_action(
                    cfg,
                    model,
                    obs,
                    instruction,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_projector,
                    use_film=False,
                    last_caches=last,
                )

            cfg.use_vla_cache = True
            output, wall_ms, cuda_ms = timed_cuda_call(torch, official_call)
            actions, last = output[0], output[1]
            query_count += 1
            pair_records.append({"arm": arm, "wall_ms": wall_ms, "cuda_ms": cuda_ms})
            if arm == "reuse":
                corrected_parity.append(
                    compare_actions(
                        p0_actions["input-a"],
                        np.asarray(actions),
                        np=np,
                        rtol=float(parity["unnormalized_action_rtol"]),
                        atol=float(parity["unnormalized_action_atol"]),
                        exact_gripper=bool(parity["gripper_decisions_exact"]),
                    )
                )
        if pair >= 2:
            corrected_records.append({"pair": pair, "queries": pair_records})
    cfg.use_vla_cache = False
    configure_dense(model)

    profile_results: dict[str, Any] = {
        profile["profile_id"]: {"warmup": [], "cycles": []} for profile in config["profiles"]
    }
    warm_parity = []
    for profile in config["profiles"]:
        anchor = direct("input-a", capture=True)
        assert_cache_shape(anchor["cache"])
        tracker = SourceTracker(anchor_query=0)
        cache = anchor["cache"]
        for ordinal, label in enumerate(("input-c", "input-a", "input-a", "input-a")):
            wrist_offset = (
                int(profile["wrist_budgets"][-1])
                if ordinal >= int(profile["wrist_max_age"]) and int(profile["wrist_budgets"][-1])
                else 0
            )
            result = direct(
                label,
                cache=cache,
                profile=profile,
                anchor_pixels=anchor["prepared"].preprocessed_pixels,
                anchor_salience=anchor["salience"],
                wrist_offset=wrist_offset,
            )
            cache = result["cache"]
            assert_cache_shape(cache)
            ordered = tuple(int(value) for value in result["ordered"].tolist())
            tracker.advance(
                ordinal + 1,
                ordered_positions=ordered,
                profile=profile,
                pruning_layers=pruning_layers,
            )
            if ordinal == 0:
                warm_parity.append(
                    {
                        "profile_id": profile["profile_id"],
                        **compare_actions(
                            p0_actions[label],
                            result["actions"],
                            np=np,
                            rtol=float(parity["unnormalized_action_rtol"]),
                            atol=float(parity["unnormalized_action_atol"]),
                            exact_gripper=bool(parity["gripper_decisions_exact"]),
                        ),
                    }
                )
            profile_results[profile["profile_id"]]["warmup"].append(
                {
                    "query": ordinal + 1,
                    "label": label,
                    "wall_ms": result["total_wall_ms"],
                    "source_digest": tracker.digest(),
                }
            )

    profile_by_id = {profile["profile_id"]: profile for profile in config["profiles"]}
    for profile_id, horizon, repetition in cycle_schedule(config):
        profile = profile_by_id[profile_id]
        anchor = direct("input-a", capture=True)
        tracker = SourceTracker(anchor_query=0)
        cache = anchor["cache"]
        accelerated = []
        for ordinal in range(horizon):
            wrist_offset = (
                int(profile["wrist_budgets"][-1])
                if ordinal >= int(profile["wrist_max_age"]) and int(profile["wrist_budgets"][-1])
                else 0
            )
            result = direct(
                "input-a",
                cache=cache,
                profile=profile,
                anchor_pixels=anchor["prepared"].preprocessed_pixels,
                anchor_salience=anchor["salience"],
                wrist_offset=wrist_offset,
            )
            cache = result["cache"]
            ordered = tuple(int(value) for value in result["ordered"].tolist())
            tracker.advance(
                ordinal + 1,
                ordered_positions=ordered,
                profile=profile,
                pruning_layers=pruning_layers,
            )
            accelerated.append(
                {
                    "query": ordinal + 1,
                    "wall_ms": result["total_wall_ms"],
                    "cuda_ms": result["total_cuda_ms"],
                    "active_sequence_length": result["active_sequence_length"],
                    "source_digest": tracker.digest(),
                    "action_sha256": action_record(result["actions"], np)["sha256"],
                    "action_parity": compare_actions(
                        p0_actions["input-a"],
                        result["actions"],
                        np=np,
                        rtol=float(parity["unnormalized_action_rtol"]),
                        atol=float(parity["unnormalized_action_atol"]),
                        exact_gripper=bool(parity["gripper_decisions_exact"]),
                    ),
                }
            )
        cycle_wall = anchor["total_wall_ms"] + sum(item["wall_ms"] for item in accelerated)
        cycle_cuda = anchor["total_cuda_ms"] + sum(item["cuda_ms"] for item in accelerated)
        profile_results[profile_id]["cycles"].append(
            {
                "horizon": horizon,
                "repetition": repetition,
                "anchor_wall_ms": anchor["total_wall_ms"],
                "anchor_cuda_ms": anchor["total_cuda_ms"],
                "cycle_wall_ms": cycle_wall,
                "cycle_cuda_ms": cycle_cuda,
                "accelerated": accelerated,
            }
        )

    expected = (
        int(config["measurement"]["cache_p0_queries"])
        + int(config["measurement"]["attention_parity_queries"])
        + int(config["measurement"]["corrected_vla_cache_queries"])
        + int(config["measurement"]["clean_profile_queries"])
    )
    if query_count != expected:
        raise RuntimeError(f"B3 cache-suite query count {query_count} != {expected}")
    p0_summary = summarize_timings([record["wall_ms"] for record in p0_timings])
    return {
        "method": "cache_suite",
        "status": "completed",
        "queries": query_count,
        "p0_timings": p0_timings,
        "p0_wall_summary": p0_summary,
        "p0_action_references": {
            label: action_record(value, np) for label, value in p0_actions.items()
        },
        "sidecar_parity": sidecar_parity,
        "corrected_vla_cache": {
            "records": corrected_records,
            "all_action_parity": all(record["passed"] for record in corrected_parity),
            "parity_records": corrected_parity,
            "correction": "true_previous_cache_source_and_error_propagation",
        },
        "warm_profile_parity": warm_parity,
        "cache_provenance_reset_invariants": True,
        "profiles": profile_results,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "transformers": __import__("transformers").__version__,
    }


def run_official_comparator(
    method: str, config: dict[str, Any], run_root: Path
) -> dict[str, Any]:
    if method == "vla_adp":
        source = ROOT / "third_party/vla-adp"
    else:
        source = ROOT / "third_party/vla-pruner/src/openvla-oft"
        required = source / "experiments/robot/vla_cache_utils.py"
        if not required.is_file():
            return {
                "method": method,
                "status": "reviewed_technical_exclusion",
                "timing_validity": "excluded_upstream_release_missing_imported_vla_cache_utils",
                "queries": 0,
                "timings": [],
                "action_references": {},
                "peak_allocated_bytes": 0,
                "peak_reserved_bytes": 0,
                "transformers": None,
                "evidence": {
                    "missing_relative_path": "experiments/robot/vla_cache_utils.py",
                    "importing_relative_path": "experiments/robot/openvla_utils.py",
                },
            }
    os.chdir(source)
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from experiments.robot.robot_utils import get_action, set_seed_everywhere
    from savr.brace.b3_openvla import action_record

    set_seed_everywhere(0)
    if method == "vla_adp":
        from experiments.robot.libero import run_libero_eval_prune_v2 as evaluation

        cfg = base_config(
            evaluation,
            ROOT / CHECKPOINT,
            run_root,
            config_class=evaluation.PruneV2GenerateConfig,
            qk_config_json=str(
                source / "experiments/robot/libero/configs/prune_v2_config.json"
            ),
        )
        evaluation._load_qk_config_from_json_if_any(cfg)
        evaluation.validate_config(cfg)
        model, action_head, proprio_projector, noisy_projector, processor = (
            evaluation._initialize_components(cfg)
        )
        evaluation.check_unnorm_key(cfg, model)
        allocation = int(config["measurement"]["vla_adp_queries"])
        validity = "component_timing_only_episode_coupled_dynamic_controller_excluded"
    else:
        from experiments.robot.libero import run_libero_eval as evaluation

        cfg = base_config(
            evaluation,
            ROOT / CHECKPOINT,
            run_root,
            use_vla_cache=False,
            use_vla_pruner=True,
            fastv_r=0.50,
            vla_pruner_mode="semantic_action",
            vla_pruner_semantic_weight=0.5,
            vla_pruner_action_weight=0.5,
            vla_pruner_av_hist_w=3,
            vla_pruner_av_decay=0.8,
        )
        evaluation.validate_config(cfg)
        model, action_head, proprio_projector, noisy_projector, processor = (
            evaluation.initialize_model(cfg)
        )
        allocation = int(config["measurement"]["vla_pruner_queries"])
        validity = "official_temporal_semantic_action_timing"
    model.eval()
    state, inputs = common_model_inputs(model, np)
    instruction = config["model"]["instruction"]
    records = []
    actions = {}
    last_caches = None
    for query in range(allocation):
        label = ("input-a", "input-b", "input-c")[query % 3]

        def call() -> Any:
            arguments = {
                "processor": processor,
                "action_head": action_head,
                "proprio_projector": proprio_projector,
                "noisy_action_projector": noisy_projector,
                "use_film": False,
            }
            if method == "vla_pruner":
                arguments["last_caches"] = last_caches
            return get_action(
                cfg,
                model,
                observation(inputs, state, label),
                instruction,
                **arguments,
            )

        output, wall_ms, cuda_ms = timed_cuda_call(torch, call)
        if method == "vla_pruner":
            predicted = output[0]
            last_caches = output[1]
        else:
            predicted = output
        if query >= 4:
            records.append({"query": query, "label": label, "wall_ms": wall_ms, "cuda_ms": cuda_ms})
            actions.setdefault(label, action_record(np.asarray(predicted), np))
    return {
        "method": method,
        "status": "completed",
        "timing_validity": validity,
        "queries": allocation,
        "timings": records,
        "action_references": actions,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "transformers": __import__("transformers").__version__,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", choices=("core_fr", "cache_suite", "vla_adp", "vla_pruner"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if ROOT != EXPECTED_ROOT:
        raise SystemExit(f"B3 worker refuses to run outside {EXPECTED_ROOT}")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
        raise SystemExit("B3 worker requires one explicitly selected visible GPU")
    sys.path.insert(0, str(ROOT / "src"))
    from savr.acr.v5_d_recovery import capture_checkpoint_baseline, restore_checkpoint_exact
    from savr.brace.b3 import validate_config

    config = json.loads((ROOT / CONFIG).read_text())
    validate_config(config)
    if args.output.exists():
        raise SystemExit(f"Immutable B3 worker output exists: {args.output}")
    protected = ("config.json", "configuration_prismatic.py", "modeling_prismatic.py")
    checkpoint = ROOT / CHECKPOINT
    baseline = capture_checkpoint_baseline(checkpoint, protected)
    started = time.monotonic()
    try:
        if args.method == "core_fr":
            result = run_core_fr(config, args.output.parent)
        elif args.method == "cache_suite":
            result = run_cache_suite(config, args.output.parent)
        else:
            result = run_official_comparator(args.method, config, args.output.parent)
        result.update(
            {
                "schema_version": "brace.b3-worker.v1",
                "run_id": config["run_id"],
                "configuration_semantic_sha256": config["semantic_sha256"],
                "source_revision": git_output(ROOT, "rev-parse", "HEAD"),
                "completed_at_utc": utc_now(),
                "wall_seconds": time.monotonic() - started,
            }
        )
        result["semantic_sha256"] = semantic_sha256(result)
        write_once(args.output, result)
        print(
            json.dumps(
                {"method": args.method, "status": result["status"], "queries": result["queries"]}
            )
        )
        return 0
    finally:
        restoration = restore_checkpoint_exact(checkpoint, baseline)
        if not (
            restoration["protected_bytes_restored"]
            and restoration["backup_cleanup_complete"]
            and restoration["inventory_equal"]
        ):
            raise RuntimeError("B3 checkpoint metadata restoration failed")


if __name__ == "__main__":
    raise SystemExit(main())
