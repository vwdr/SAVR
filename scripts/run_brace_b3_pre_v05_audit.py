#!/usr/bin/env python3
"""Run the CUDA-hidden exhaustive remaining-path audit before any B3-v05 proposal."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path("/home/ved/SAVR")
OUTPUT = Path("reports/runtime/brace_b3_pre_v05_audit.json")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def function_arguments(path: Path, name: str) -> list[str]:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return [argument.arg for argument in node.args.args]
    raise RuntimeError(f"Missing pinned function: {name}")


def main() -> int:
    started = time.monotonic()
    if ROOT != EXPECTED_ROOT or Path.cwd().resolve() != EXPECTED_ROOT:
        raise SystemExit(f"B3 pre-v05 audit is restricted to {EXPECTED_ROOT}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("B3 pre-v05 audit requires CUDA to be hidden")
    sys.path.insert(0, str(ROOT / "src"))
    from savr.brace.b3 import allowed_project_status, load_config_file

    output = ROOT / OUTPUT
    if output.exists():
        raise SystemExit("Immutable B3 pre-v05 audit already exists")
    cfg = load_config_file(ROOT, Path("configs/brace/b3_physical_v4_recovery.json"))
    raw_status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"]
    )
    if not allowed_project_status(raw_status, "unused-audit-run"):
        raise SystemExit("B3 pre-v05 audit source state is not clean apart from tmp/")

    helper = ROOT / "src/savr/brace/b3_openvla.py"
    worker = ROOT / "scripts/run_brace_b3_worker.py"
    runner = ROOT / "scripts/run_brace_b3.py"
    analyzer = ROOT / "scripts/analyze_brace_b3.py"
    helper_tree = ast.parse(helper.read_text())
    binary_calls = []
    for node in ast.walk(helper_tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr in {"maximum", "minimum"}
        ):
            binary_calls.append({"function": node.func.attr, "line": node.lineno, "arguments": len(node.args)})
    if not binary_calls or any(call["arguments"] != 2 for call in binary_calls):
        raise RuntimeError("B3 tensor min/max arity audit failed")
    worker_source = worker.read_text()
    structural = {
        "direct_inference_decorator": "    @torch.inference_mode()\n    def direct(" in worker_source,
        "direct_gradient_assertion": "if torch.is_grad_enabled():" in worker_source,
        "corrected_previous_cache_source": 'obs["prev_images"]' in worker_source,
        "timed_action_parity": '"action_parity": compare_actions(' in worker_source,
        "immutable_worker_write": "os.O_EXCL" in worker_source,
        "checkpoint_restoration_finally": "    finally:\n        restoration = " in worker_source,
        "exact_cache_query_reconciliation": "if query_count != expected:" in worker_source,
    }
    if not all(structural.values()):
        raise RuntimeError("B3 worker structural audit failed")
    runner_source = runner.read_text()
    analyzer_source = analyzer.read_text()
    terminal_structural = {
        "runner_write_once": "os.O_EXCL" in runner_source,
        "runner_no_query_reassignment": '"unused_queries_not_reassigned"' in runner_source,
        "runner_technical_stop": '"automatic_retry": False' in runner_source,
        "analyzer_write_once": "os.O_EXCL" in analyzer_source,
        "analyzer_blocks_technical_stop": "scientific analysis is prohibited" in analyzer_source,
        "analyzer_conjunctive_gate": "accepted = all(gates.values())" in analyzer_source,
        "analyzer_outcome_boundary": "No simulator outcomes or task-success fields" in analyzer_source,
        "analyzer_stops_before_b4": '"b4_authorized": False' in analyzer_source,
    }
    if not all(terminal_structural.values()):
        raise RuntimeError("B3 runner/analyzer structural audit failed")

    cache_root = ROOT / "third_party/vla-cache/src/openvla-oft"
    model_source = cache_root / "prismatic/extern/hf/modeling_prismatic.py"
    action_source = cache_root / "prismatic/models/action_heads.py"
    robot_source = cache_root / "experiments/robot/robot_utils.py"
    llama_source = ROOT / "envs/vla-cache-compat/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py"
    signatures = {
        "_process_vision_features": function_arguments(model_source, "_process_vision_features"),
        "_process_proprio_features": function_arguments(model_source, "_process_proprio_features"),
        "_build_multimodal_attention": function_arguments(model_source, "_build_multimodal_attention"),
        "_prepare_input_for_action_prediction": function_arguments(
            model_source, "_prepare_input_for_action_prediction"
        ),
        "_prepare_labels_for_action_prediction": function_arguments(
            model_source, "_prepare_labels_for_action_prediction"
        ),
        "predict_action": function_arguments(action_source, "predict_action"),
        "get_action": function_arguments(robot_source, "get_action"),
    }
    expected_prefixes = {
        "_process_vision_features": ["self", "pixel_values", "language_embeddings", "use_film"],
        "_process_proprio_features": ["self", "projected_patch_embeddings", "proprio", "proprio_projector"],
        "_build_multimodal_attention": [
            "self",
            "input_embeddings",
            "projected_patch_embeddings",
            "attention_mask",
        ],
        "_prepare_input_for_action_prediction": ["self", "input_ids", "attention_mask"],
        "_prepare_labels_for_action_prediction": ["self", "labels", "input_ids"],
        "predict_action": ["self", "actions_hidden_states"],
        "get_action": [
            "cfg",
            "model",
            "obs",
            "task_label",
            "processor",
            "action_head",
            "proprio_projector",
            "noisy_action_projector",
            "use_film",
            "last_caches",
        ],
    }
    if any(signatures[name][: len(expected)] != expected for name, expected in expected_prefixes.items()):
        raise RuntimeError("B3 pinned private-model signature changed")
    llama_text = llama_source.read_text()
    backend = {
        "sdpa_class": "class LlamaSdpaAttention" in llama_text,
        "sdpa_primitive": "torch.nn.functional.scaled_dot_product_attention(" in llama_text,
        "pruning_layers": "self.pruning_loc = [2, 6, 9, 11]" in llama_text,
        "position_preserving_update": "cache_position" in llama_text and "selected_reusable_patches" in llama_text,
        "official_no_grad": "with torch.no_grad():" in robot_source.read_text(),
    }
    if not all(backend.values()):
        raise RuntimeError("B3 pinned backend audit failed")

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [str(ROOT / "envs/openvla-oft/bin/python"), "-m", "pytest", "-q", "tests/brace"],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"B3 server audit tests failed:\n{completed.stdout[-4000:]}")
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("B3 pre-v05 audit initialized CUDA")
    statistics = json.loads(
        (ROOT / "checkpoints/openvla-7b-oft-libero-four-suite/dataset_statistics.json").read_text()
    )
    if "libero_object_no_noops" not in statistics:
        raise RuntimeError("B3 checkpoint normalization alias disappeared")
    repositories = {
        key: ROOT / path
        for key, path in {
            "openvla_oft_revision": "third_party/openvla-oft",
            "libero_revision": "third_party/LIBERO",
            "vla_cache_revision": "third_party/vla-cache",
            "vla_adp_revision": "third_party/vla-adp",
            "vla_pruner_revision": "third_party/vla-pruner",
            "specprune_vla_revision": "third_party/specprune-vla",
        }.items()
    }
    repository_gates = {}
    for key, repository in repositories.items():
        revision = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip()
        clean = not subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"], text=True
        ).strip()
        repository_gates[key] = clean and revision == cfg["provenance"][key]
    if not all(repository_gates.values()):
        raise RuntimeError("B3 pinned repository gate failed")
    result = {
        "schema_version": "brace.b3-pre-v05-audit.v1",
        "status": "accepted",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "v04_resolved_configuration_semantic_sha256": cfg["semantic_sha256"],
        "cuda_visible_devices": "hidden",
        "cuda_initialized": False,
        "model_loads": 0,
        "model_queries": 0,
        "simulator_outcomes": 0,
        "protected_outcome_access": False,
        "binary_tensor_calls": binary_calls,
        "worker_structural_gates": structural,
        "runner_analyzer_structural_gates": terminal_structural,
        "pinned_signatures": signatures,
        "backend_gates": backend,
        "repository_gates": repository_gates,
        "pytest_summary": completed.stdout.strip().splitlines()[-1],
        "audited_file_hashes": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in (
                helper,
                worker,
                runner,
                analyzer,
                model_source,
                action_source,
                robot_source,
                llama_source,
            )
        },
        "wall_seconds": time.monotonic() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "v05_authorized": False,
        "b4_authorized": False,
    }
    result["semantic_sha256"] = hashlib.sha256(canonical_bytes(result)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(result) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": result["status"], "pytest": result["pytest_summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
