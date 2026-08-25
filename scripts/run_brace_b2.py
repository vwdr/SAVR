#!/usr/bin/env python3
"""Run the frozen CPU/synthetic BRACE-B2 correctness and compatibility gate."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXPECTED_ROOT = Path("/home/ved/SAVR")
CONFIG_RELATIVE = Path("configs/brace/b2_correctness_v2.json")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(dict(value)) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def git_output(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()


def directory_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def validate_config(config: Mapping[str, Any]) -> None:
    supplied = config.get("semantic_sha256")
    payload = dict(config)
    payload.pop("semantic_sha256", None)
    if semantic_sha256(payload) != supplied:
        raise RuntimeError("B2 configuration semantic hash mismatch")
    if config.get("schema_version") != "brace.b2-config.v2":
        raise RuntimeError("B2 configuration schema changed")
    recovery = config.get("recovery")
    if not isinstance(recovery, dict) or recovery.get("supersedes_run_id") != "brace-b2-correctness-v01":
        raise RuntimeError("B2 recovery identity changed")
    if recovery.get("scientific_gates_changed") is not False:
        raise RuntimeError("B2 recovery changed scientific gates")
    contracts = config["contracts"]
    if contracts != {
        "selectable_families": ["P1", "P2"],
        "nonselectable_families": ["P0", "P3", "P4"],
        "horizons": [1, 2, 4],
        "maximum_base_profiles": 6,
        "maximum_scene_only_profiles": 3,
        "maximum_dual_view_profiles": 3,
    }:
        raise RuntimeError("B2 contract boundary changed")
    caps = config["resource_caps"]
    if caps["cuda_visible"] or caps["model_queries"] or caps["policy_outcomes"]:
        raise RuntimeError("B2 resource boundary permits forbidden work")
    if caps["simulator_steps"] or caps["model_checkpoint_dataset_downloads_allowed"]:
        raise RuntimeError("B2 permits simulator or model/dataset work")
    b3 = config["b3_proposal"]
    if (
        b3["maximum_balanced_real_model_queries"] > 500
        or b3["maximum_visible_gpus"] != 1
        or b3["simulator_outcomes"] != 0
        or not b3["requires_separate_authorization"]
    ):
        raise RuntimeError("B3 proposal exceeds Protocol V2.1")


def run_cpu_python(root: Path, environment: Path, code: str) -> dict[str, Any]:
    process_environment = dict(os.environ)
    process_environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(root / "src"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "disabled",
        }
    )
    completed = subprocess.run(
        [str(environment / "bin/python"), "-c", code],
        cwd=root,
        env=process_environment,
        text=True,
        check=True,
        capture_output=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


DYNAMIC_CACHE_CHECK = r'''
import json
from types import SimpleNamespace
import torch, transformers
from transformers import DynamicCache
from savr.brace.cache_adapter import clone_dynamic_cache, position_preserving_index_update, transactional_cache_configuration
cache = DynamicCache()
keys = torch.arange(24, dtype=torch.float32).reshape(1, 2, 4, 3)
values = keys + 100
cache.update(keys, values, 0, {"cache_position": torch.arange(4)})
before = cache.key_cache[0].clone()
clone = clone_dynamic_cache(cache)
clone.key_cache[0].add_(1)
independent = torch.equal(cache.key_cache[0], keys) and not torch.equal(clone.key_cache[0], keys)
config = SimpleNamespace(mode="dense")
try:
    with transactional_cache_configuration(cache, config, {"mode": "reuse", "temporary": 1}) as arm:
        arm.key_cache[0].zero_()
        cache.key_cache[0].add_(2)
        raise RuntimeError("synthetic")
except RuntimeError:
    pass
restored = torch.equal(cache.key_cache[0], before) and config.mode == "dense" and not hasattr(config, "temporary")
cached = torch.arange(24, dtype=torch.float32).reshape(1, 2, 4, 3)
current = torch.full((1, 2, 2, 3), 999.0)
updated = position_preserving_index_update(cached, current, [1, 3])
position_preserving = (
    torch.equal(updated[:, :, 0], cached[:, :, 0])
    and torch.equal(updated[:, :, 2], cached[:, :, 2])
    and torch.equal(updated[:, :, 1], current[:, :, 0])
    and torch.equal(updated[:, :, 3], current[:, :, 1])
)
print(json.dumps({
    "python": __import__("sys").version.split()[0],
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "cuda_initialized": torch.cuda.is_initialized(),
    "clone_independent": independent,
    "transaction_restored": restored,
    "position_preserving_update_present": position_preserving,
}))
'''


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing to run B2 outside {EXPECTED_ROOT}: {root}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise SystemExit("B2 refuses visible CUDA devices")
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONNOUSERSITE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "disabled",
        }
    )
    config = json.loads((root / CONFIG_RELATIVE).read_text())
    validate_config(config)
    caps = config["resource_caps"]

    def timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError("BRACE-B2 reached its frozen wall cap")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(int(caps["wall_seconds"]))
    if git_output(root, "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("Refusing B2 with modified tracked project files")
    source_revision = git_output(root, "rev-parse", "HEAD")
    run_root = root / "results" / config["run_id"]
    if run_root.exists():
        raise SystemExit(f"Immutable B2 run already exists: {run_root}")
    run_root.mkdir(parents=True)
    started = time.monotonic()
    manifest = {
        "schema_version": "brace.b2-manifest.v1",
        "run_id": config["run_id"],
        "status": "running",
        "started_at_utc": utc_now(),
        "source_revision": source_revision,
        "configuration_semantic_sha256": config["semantic_sha256"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "resource_caps": caps,
    }
    write_once(run_root / "manifest.json", manifest)

    sys.path.insert(0, str(root / "src"))
    from savr.brace.baseline import correction_manifest
    from savr.brace.compat import compatibility_disposition, inspect_repository

    try:
        core_root = root / "third_party/openvla-oft"
        if git_output(core_root, "rev-parse", "HEAD") != config["stacks"]["core"][
            "openvla_oft_revision"
        ]:
            raise RuntimeError("core OpenVLA-OFT revision differs")
        if git_output(core_root, "status", "--porcelain"):
            raise RuntimeError("core OpenVLA-OFT source is dirty")

        cache_root = root / "third_party/vla-cache"
        if git_output(cache_root, "rev-parse", "HEAD") != config["stacks"]["vla_cache"][
            "repository_revision"
        ]:
            raise RuntimeError("VLA-Cache revision differs")
        if git_output(cache_root, "status", "--porcelain"):
            raise RuntimeError("VLA-Cache source is dirty")
        evaluator = cache_root / "src/openvla-oft/experiments/robot/libero/run_libero_eval.py"
        correction = correction_manifest(evaluator.read_text())

        comparator_specs = (
            {
                "key": "vla_adp",
                "root": root / "third_party/vla-adp",
                "required_paths": (
                    "experiments/robot/libero/run_libero_eval_prune_v2.py",
                    "experiments/robot/libero/configs/prune_v2_config.json",
                ),
                "license_paths": ("LICENSE.txt",),
                "configuration_paths": (
                    "experiments/robot/libero/configs/prune_v2_config.json",
                ),
                "required_markers": {
                    "experiments/robot/libero/run_libero_eval_prune_v2.py": (
                        "use_dynamic_visual_strategy",
                        "robot0_eef_pos",
                    )
                },
            },
            {
                "key": "vla_pruner",
                "root": root / "third_party/vla-pruner",
                "required_paths": (
                    "src/openvla-oft/experiments/robot/vla_pruner_utils.py",
                    "src/openvla-oft/transformers/src/transformers/models/llama/modeling_llama.py",
                ),
                "license_paths": ("LICENSE",),
                "configuration_paths": (
                    "src/openvla-oft/vla_pruner_srcipts/run_vla_pruner/run_spatial.sh",
                ),
                "required_markers": {
                    "src/openvla-oft/experiments/robot/vla_pruner_utils.py": (
                        "semantic_action",
                        "build_vlapruner_cache_config",
                    )
                },
            },
            {
                "key": "specprune_vla",
                "root": root / "third_party/specprune-vla",
                "required_paths": (
                    "openvla-oft/experiments/robot/spec_prune_vla.py",
                    "openvla-oft/prismatic/extern/hf/modeling_llama.py",
                ),
                "license_paths": ("LICENSE",),
                "configuration_paths": (
                    "openvla-oft/experiments/robot/spec_prune_constants.py",
                ),
                "required_markers": {
                    "openvla-oft/prismatic/extern/hf/modeling_llama.py": (
                        "dynamic_prune",
                        "action_token_count",
                    )
                },
            },
        )
        comparator_records = []
        for spec in comparator_specs:
            key = spec["key"]
            record = inspect_repository(
                spec["root"],
                name=key,
                revision=config["comparators"][key]["revision"],
                required_paths=spec["required_paths"],
                license_paths=spec["license_paths"],
                configuration_paths=spec["configuration_paths"],
                required_markers=spec["required_markers"],
            )
            record["stack_mode"] = config["comparators"][key]["stack_mode"]
            record["disposition"] = compatibility_disposition(
                record, stack_mode=record["stack_mode"]
            )
            if key == "specprune_vla":
                inherited = spec["root"] / "openvla-oft/LICENSE"
                record["inherited_openvla_license_sha256"] = (
                    hashlib.sha256(inherited.read_bytes()).hexdigest() if inherited.exists() else None
                )
                record["license_note"] = (
                    "README badge targets absent top-level LICENSE; the present subtree license "
                    "covers upstream OpenVLA-OFT, not clearly all method-specific additions."
                )
            comparator_records.append(record)
        comparator_records.append(
            {
                "name": "gated_vla_cache",
                "paper": config["comparators"]["gated_vla_cache"]["paper"],
                "official_code": None,
                "license_resolved": False,
                "configuration_resolved": False,
                "disposition": "paper_only_no_official_code_matched_reproduction_required",
            }
        )

        source_bytes = sum(
            directory_size(root / f"third_party/{name}")
            for name in ("vla-adp", "vla-pruner", "specprune-vla")
        )
        if source_bytes > int(caps["source_code_bytes"]):
            raise RuntimeError("B2 comparator source cap exceeded")

        stack_checks = {
            "core_4_40_1": run_cpu_python(
                root, root / "envs/openvla-oft", DYNAMIC_CACHE_CHECK
            ),
            "vla_cache_4_47_0": run_cpu_python(
                root, root / "envs/vla-cache-compat", DYNAMIC_CACHE_CHECK
            ),
        }
        if stack_checks["core_4_40_1"]["transformers"] != "4.40.1":
            raise RuntimeError("core Transformers stack version changed")
        if stack_checks["vla_cache_4_47_0"]["transformers"] != "4.47.0":
            raise RuntimeError("VLA-Cache Transformers stack version changed")
        for record in stack_checks.values():
            if record["cuda_initialized"] or not all(
                record[key]
                for key in (
                    "clone_independent",
                    "transaction_restored",
                    "position_preserving_update_present",
                )
            ):
                raise RuntimeError("a pinned DynamicCache stack failed B2 correctness")

        test_environment = dict(os.environ)
        test_environment["PYTHONPATH"] = f"{root}:{root / 'src'}"
        completed = subprocess.run(
            [str(root / "envs/openvla-oft/bin/python"), "-m", "pytest", "-q", "tests/brace"],
            cwd=root,
            env=test_environment,
            text=True,
            check=True,
            capture_output=True,
        )
        test_terminal = completed.stdout.strip().splitlines()[-1]
        gates = {
            "configuration_authenticated": True,
            "core_and_vla_cache_stacks_isolated": True,
            "dynamic_cache_clone_restore_both_stacks": True,
            "position_preserving_updates_both_stacks": True,
            "corrected_evaluator_previous_source": correction[
                "previous_cache_source_fixed"
            ],
            "corrected_evaluator_error_propagation": correction["episode_errors_propagate"],
            "algorithm_configuration_unchanged": correction[
                "algorithm_configuration_unchanged"
            ],
            "synthetic_adversarial_suite_passed": "passed" in test_terminal,
            "all_comparator_preflights_have_dispositions": len(comparator_records) == 4
            and all(record.get("disposition") for record in comparator_records),
            "cuda_hidden_and_uninitialized": os.environ["CUDA_VISIBLE_DEVICES"] == ""
            and all(not record["cuda_initialized"] for record in stack_checks.values()),
            "no_model_policy_or_simulator_work": True,
            "source_cap_respected": True,
            "b3_cap_is_bounded_and_separately_gated": True,
        }
        summary = {
            "schema_version": "brace.b2-summary.v1",
            "run_id": config["run_id"],
            "status": "accepted_with_comparator_dispositions",
            "started_at_utc": manifest["started_at_utc"],
            "completed_at_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - started,
            "source_revision": source_revision,
            "configuration_semantic_sha256": config["semantic_sha256"],
            "correction": correction,
            "stack_checks": stack_checks,
            "comparator_preflights": comparator_records,
            "source_code_bytes": source_bytes,
            "test_terminal": test_terminal,
            "b3_proposal": config["b3_proposal"],
            "gates": gates,
            "all_gates_passed": all(gates.values()),
            "claim_boundary": (
                "CPU/synthetic correctness only; no model, GPU, simulator, policy outcome, "
                "latency, cache-performance, or positive-paper result."
            ),
        }
        if not summary["all_gates_passed"]:
            raise RuntimeError("B2 summary gates did not all pass")
        summary["semantic_sha256"] = semantic_sha256(summary)
        if directory_size(run_root) > int(caps["artifact_bytes"]):
            raise RuntimeError("B2 artifact cap exceeded")
        write_once(run_root / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        failure = {
            "schema_version": "brace.b2-technical-stop.v1",
            "run_id": config["run_id"],
            "status": "technical_stop",
            "error_type": type(error).__name__,
            "error": str(error),
            "completed_at_utc": utc_now(),
            "source_revision": source_revision,
            "configuration_semantic_sha256": config["semantic_sha256"],
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "model_queries": 0,
            "policy_outcomes": 0,
            "simulator_steps": 0,
        }
        failure["semantic_sha256"] = semantic_sha256(failure)
        write_once(run_root / "technical_stop.json", failure)
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    raise SystemExit(main())
