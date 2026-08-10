from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/acr/v5_d_gpu_feasibility_freeze.json"
RESEARCH = ROOT / "docs/ACR_V5_D_RESEARCH_AND_MEASUREMENT_DESIGN.md"
PROTOCOL = ROOT / "docs/ACR_V5_D_GPU_FEASIBILITY_PROTOCOL.md"


def freeze() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def semantic_sha256(config: dict[str, object]) -> str:
    payload = dict(config)
    payload.pop("semantic_sha256")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_freeze_is_self_authenticating_and_defers_execution() -> None:
    config = freeze()
    assert config["semantic_sha256"] == semantic_sha256(config)
    assert config["status"] == ("FROZEN_BEFORE_BACKEND_IMPLEMENTATION_GPU_SELECTION_OR_EXECUTION")
    assert config["authorized_scope"] == "PROTOCOL_PREPARATION_ONLY"
    assert not any(config["current_authorization"].values())
    assert config["advance_only_to"] == (
        "V5_D_BACKEND_IMPLEMENTATION_AND_EXPLICIT_GPU_COORDINATION"
    )


def test_v5_d_preserves_the_selected_v5_b_and_v5_c_contract() -> None:
    selected = freeze()["selected_method"]
    assert selected["method_id"] == "v5-a100-b40"
    assert selected["controller_version"] == "acr-isolated-controller-v1"
    assert selected["scene_threshold"] == 0.30046895424836606
    assert selected["translation_threshold"] == 0.685919037527938
    assert selected["horizon"] == 1
    assert selected["hard_reuse_cap"] == 0.4
    assert selected["post_reuse_latch"] is True
    assert selected["v5_b_semantic_sha256"] == (
        "8a9f15b818b58ed2868d4b1123a222a4c062507161ab7de911d8d233f3b1efec"
    )
    assert selected["v5_c_semantic_sha256"] == (
        "f7a8d11d4574add57caa630c03463375421d9482984478be769f497b1c9d0b66"
    )


def test_local_environment_and_prior_evidence_hashes_reconcile() -> None:
    hashes = freeze()["environment_hashes"]
    expected_paths = {
        "conda_explicit_sha256": ROOT / "environment/locks/conda-linux-64-explicit.txt",
        "pip_freeze_sha256": ROOT / "environment/locks/pip-freeze.txt",
        "phase1_environment_sha256": ROOT / "environment/phase1-conda.yml",
        "v5_c_runtime_sha256": (ROOT / "reports/runtime/acr_v5_c_cpu_executor_verification.json"),
        "v5_c_freeze_sha256": ROOT / "configs/acr/v5_c_cpu_executor_freeze.json",
        "v3_c_runtime_sha256": ROOT / "reports/runtime/acr_v3_c.json",
    }
    for key, path in expected_paths.items():
        assert hashes[key] == file_sha256(path)


def test_remote_only_pinned_hashes_are_exactly_predeclared() -> None:
    config = freeze()
    assert config["checkpoint_hashes"] == {
        "config_json_sha256": ("edd5c5cf6d7927e07465cf086ebe41f7b3ec8f3b128a51f71d6db14dad7ad8b1"),
        "configuration_prismatic_sha256": (
            "68cc5ae34f1b46af3168d8d479cb81bb776965653453fd904aa8eefb6c8f9f68"
        ),
        "modeling_prismatic_sha256": (
            "f40ee7883e16aab1a2d89b6e8f31cc81f6b8055120b1fefe169e05c7031098fa"
        ),
    }
    assert config["upstream_source_hashes"] == {
        "model_source_sha256": ("b5431a074c0025a12e46dc954a5e18d1d73477babb5ae42e3a12ab4b907f33a6"),
        "openvla_utils_sha256": (
            "eed754d7c5f9821aae2fe0531dbe01df8c11df0d5c79b4aeeb9bb4452124bdf5"
        ),
        "action_heads_sha256": ("8e42df65d6407d64d47457286d9be05466d25d8a0f094aa9dfb90de84a9aa7cc"),
    }


def test_backend_waterfall_cannot_be_used_for_result_shopping() -> None:
    waterfall = freeze()["backend_waterfall"]
    assert waterfall["order"] == ["torch-compile", "raw-cudagraph"]
    assert waterfall["compiler"] == {
        "backend": "inductor",
        "fullgraph": True,
        "dynamic": False,
        "mode": "reduce-overhead",
    }
    assert waterfall["raw_allowed_only_before_correctness"] is True
    assert waterfall["fallback_after_correctness_or_timing"] is False
    assert waterfall["backend_comparison_or_shopping"] is False
    assert waterfall["raw_capture"]["both_cores_required"] is True
    assert waterfall["raw_capture"]["mixed_backends_permitted"] is False
    assert waterfall["fresh_process_on_transition"] is True


def test_correctness_tensor_contract_and_tolerances_are_exact() -> None:
    config = freeze()
    tensors = config["tensor_contract"]
    assert tensors["wrist_pixels"]["shape"] == [1, 6, 224, 224]
    assert tensors["cached_scene_tokens"]["shape"] == [1, 256, 4096]
    assert tensors["combined_tokens"] == {
        "shape": [1, 512, 4096],
        "dtype": "torch.bfloat16",
        "order": "scene-first",
    }
    assert tensors["prepared_input_ids"]["shape"] == [1, 79]
    assert tensors["normalized_actions"]["shape"] == [1, 8, 7]
    correctness = config["correctness"]
    assert correctness["query_count"] == 7
    assert len(correctness["labels"]) == 7
    assert correctness["tolerances"] == {
        "wrist_and_combined_tokens": {"rtol": 0.016, "atol": 0.00001},
        "normalized_actions": {"rtol": 0.001, "atol": 0.0001},
        "unnormalized_actions": {"rtol": 0.00001, "atol": 0.000001},
    }
    assert correctness["a_repeat_bitwise_identical"] is True
    assert correctness["gripper_decisions_exact"] is True
    assert correctness["optimized_reuse_scene_calls"] == 0


def test_timing_schedule_is_complete_balanced_and_within_query_cap() -> None:
    config = freeze()
    timing = config["timing"]
    paths = timing["paths"]
    expected = [list(items) for items in itertools.permutations(paths)]
    assert timing["permutations"] == expected
    assert timing["block_count"] == 24
    assert timing["timed_query_count"] == 96
    for path in paths:
        for position in range(4):
            assert sum(block[position] == path for block in timing["permutations"]) == 6
    warmups = timing["warmups_per_path"] * len(paths)
    assert warmups == timing["warmup_query_count"] == 8
    total = config["correctness"]["query_count"] + warmups + timing["timed_query_count"]
    assert total == config["resource_caps"]["full_model_queries_if_complete"] == 111
    assert config["resource_caps"]["full_model_query_hard_cap"] == 111
    assert timing["outlier_deletion"] is False


def test_efficiency_gates_reconcile_with_the_v5_b_lower_bound() -> None:
    config = freeze()
    gates = config["gates"]
    reuse = config["analysis"]["reuse_weight"]
    refresh_ratio = 1.005452
    target = 0.98
    required_reuse_ratio = (target - (1.0 - reuse) * refresh_ratio) / reuse
    assert gates["optimized_reuse_wall_over_batched_fr_median_max"] == pytest.approx(
        required_reuse_ratio
    )
    assert gates["weighted_wall_over_batched_fr_upper_95_max"] == 0.98
    assert gates["optimized_over_eager_sequential_cuda_upper_95_max"] == 0.96
    assert gates["weighted_visual_cuda_reduction_lower_95_min"] == 0.1
    assert gates["v5_refresh_wall_over_batched_fr_upper_95_max"] == 1.02
    assert gates["maximum_position_median_relative_deviation"] == 0.03
    assert gates["all_gates_conjunctive"] is True


def test_gpu_selection_is_deferred_and_fail_closed() -> None:
    selection = freeze()["gpu_selection"]
    assert selection["selected_during_protocol_preparation"] is False
    assert selection["requires_new_user_coordination"] is True
    assert selection["aggregate_samples"] == 3
    assert selection["seconds_between_samples"] == 10
    assert selection["maximum_utilization_percent_each_sample"] == 5
    assert selection["maximum_memory_used_mib_each_sample"] == 512
    assert selection["selection_rule"] == "lowest-index-eligible-device"
    assert selection["inspect_process_identities_or_commands"] is False
    assert selection["freeze_physical_id_and_uuid_before_model_load"] is True


def test_resources_recovery_and_protected_boundaries_are_frozen() -> None:
    config = freeze()
    caps = config["resource_caps"]
    assert caps["gpu_count"] == 1
    assert caps["model_processes_at_once"] == 1
    assert caps["backend_preparation_core_launch_hard_cap"] == 24
    assert caps["simulator_episodes"] == 0
    assert caps["simulator_resets"] == 0
    assert caps["downloads"] == 0
    assert caps["new_task_outcomes"] == 0
    assert caps["wall_seconds"] == 7200
    assert caps["artifact_bytes"] == 1073741824
    assert config["memory"]["peak_reserved_gib_max"] == 23
    assert config["memory"]["incremental_reserved_gib_over_eager_max"] == 6
    recovery = config["recovery"]
    assert recovery["automatic_retry"] is False
    assert recovery["interrupted_run_stops_for_adjudication"] is True
    assert recovery["restore_patched_methods"] is True
    assert recovery["restore_checkpoint_metadata_byte_for_byte"] is True


def test_docs_state_the_stop_claim_and_authorization_boundaries() -> None:
    research = RESEARCH.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")
    assert "COMPLETE BEFORE V5-D PROTOCOL FREEZE" in research
    assert "FROZEN BEFORE BACKEND IMPLEMENTATION, GPU SELECTION, OR EXECUTION" in protocol
    assert "Raw graphs are not a second statistical candidate" in research
    assert "No task-success field is needed or permitted" in research
    assert "The user approved preparation of this protocol only" in protocol
    assert "Before any device is selected" in protocol
    assert "No automatic retry is allowed" in protocol
    assert "does not authorize an episode" in protocol
