#!/usr/bin/env python3
"""Audit the pinned official VLA-Cache source in an isolated CPU environment."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_ROOT = Path("/home/ved/SAVR")
VLA_CACHE_REVISION = "a4909880573868dee2769343d52e793c0341678b"
TRANSFORMERS_REVISION = "9a90a37acacf453433168db8d7769b7ea3c40c06"
CORE_TRANSFORMERS_REVISION = "bc339d9ad707454c0c115970db43c260067c61ab"
RUN_ID = "phase5-vla-cache-compatibility-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(path: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *arguments],
        text=True,
    ).strip()


def write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"Audit result is immutable and already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.link(temporary, path)
    temporary.unlink()


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    if project_root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing to run outside {EXPECTED_ROOT}: {project_root}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONNOUSERSITE"] = "1"

    source_root = project_root / "third_party" / "vla-cache"
    openvla_root = source_root / "src" / "openvla-oft"
    eval_path = openvla_root / "experiments" / "robot" / "libero" / "run_libero_eval.py"
    utils_path = openvla_root / "experiments" / "robot" / "vla_cache_utils.py"
    model_path = openvla_root / "prismatic" / "extern" / "hf" / "modeling_prismatic.py"
    output_path = project_root / "results" / RUN_ID / "audit.json"

    revision = git_output(source_root, "rev-parse", "HEAD")
    if revision != VLA_CACHE_REVISION:
        raise RuntimeError(
            f"VLA-Cache revision differs: expected {VLA_CACHE_REVISION}, found {revision}"
        )
    if git_output(source_root, "status", "--porcelain"):
        raise RuntimeError("Pinned VLA-Cache source tree is dirty")

    lock_text = (
        project_root / "environment" / "locks" / "pip-freeze.txt"
    ).read_text(encoding="utf-8")
    if CORE_TRANSFORMERS_REVISION not in lock_text:
        raise RuntimeError("Core Transformers revision is absent from the accepted lock")

    eval_text = eval_path.read_text(encoding="utf-8")
    append_primary = eval_text.index("replay_images.append(img)")
    assign_primary = eval_text.index("prev_img = replay_images[-1]")
    attach_previous = eval_text.index(
        'observation["prev_images"] = [prev_img, prev_img_wrist]'
    )
    previous_frame_aliases_current = (
        append_primary < assign_primary < attach_previous
    )

    except_start = eval_text.index("except Exception as e:")
    metrics_start = eval_text.index("eposode_metrics =", except_start)
    error_block = eval_text[except_start:metrics_start]
    episode_errors_swallowed = "raise" not in error_block

    import numpy as np
    import skimage
    import tokenizers
    import transformers
    from transformers import DynamicCache
    from transformers.models.llama.modeling_llama import LlamaModel

    if transformers.__version__ != "4.47.0":
        raise RuntimeError(
            f"Isolated Transformers version differs: {transformers.__version__}"
        )
    if not tokenizers.__version__.startswith("0.21."):
        raise RuntimeError(
            f"Isolated tokenizers version differs: {tokenizers.__version__}"
        )

    cache_source = inspect.getsource(DynamicCache.update)
    llama_source = inspect.getsource(LlamaModel.forward)
    custom_cache_update_present = (
        "index_copy_" in cache_source and "cache_position" in cache_source
    )
    custom_llama_path_present = (
        "proportion_attn_var" in llama_source and "past_seen_tokens" in llama_source
    )

    sys.path.insert(0, str(openvla_root))
    vla_cache_utils = importlib.import_module(
        "experiments.robot.vla_cache_utils"
    )
    image = np.full((224, 224, 3), 127, dtype=np.uint8)
    static_patches = vla_cache_utils.find_static_patches(image, image)
    patch_utility_passed = len(static_patches) == 150

    findings = {
        "official_previous_frame_aliases_current_frame": previous_frame_aliases_current,
        "official_episode_errors_are_swallowed": episode_errors_swallowed,
        "isolated_transformers_fork_loaded": (
            custom_cache_update_present and custom_llama_path_present
        ),
        "official_patch_utility_passed": patch_utility_passed,
        "core_environment_requires_no_change": True,
        "gpu_episode_executed": False,
        "checkpoint_interface_exercised": False,
    }
    if not all(
        (
            previous_frame_aliases_current,
            episode_errors_swallowed,
            custom_cache_update_present,
            custom_llama_path_present,
            patch_utility_passed,
        )
    ):
        raise RuntimeError(f"Compatibility audit did not match pinned evidence: {findings}")

    result = {
        "run_id": RUN_ID,
        "started_and_finished_at_utc": utc_now(),
        "status": "TECHNICAL_EXCLUSION",
        "claim_boundary": (
            "CPU source/import compatibility only; no VLA-Cache trajectory, "
            "latency, success, or checkpoint-runtime claim."
        ),
        "revisions": {
            "vla_cache": revision,
            "vla_cache_transformers": TRANSFORMERS_REVISION,
            "core_transformers": CORE_TRANSFORMERS_REVISION,
        },
        "packages": {
            "python": sys.version,
            "transformers": transformers.__version__,
            "transformers_path": str(Path(transformers.__file__).resolve()),
            "tokenizers": tokenizers.__version__,
            "skimage": skimage.__version__,
        },
        "source_hashes": {
            "run_libero_eval.py": sha256(eval_path),
            "vla_cache_utils.py": sha256(utils_path),
            "modeling_prismatic.py": sha256(model_path),
        },
        "findings": findings,
        "exclusion_reasons": [
            (
                "The official LIBERO loop appends the current primary/wrist frames "
                "before assigning prev_images from the last replay entries, so "
                "subsequent comparisons receive the current frames as both inputs."
            ),
            (
                "The official episode loop catches runtime exceptions without "
                "re-raising or returning an explicit error status, which violates "
                "the Phase 5 terminal-error evidence requirement."
            ),
        ],
        "decision": (
            "Do not run or report the official evaluator as a valid external "
            "comparison. A reviewed, explicitly labeled correction is required "
            "before any VLA-Cache GPU trajectory evaluation."
        ),
    }
    write_once(output_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
