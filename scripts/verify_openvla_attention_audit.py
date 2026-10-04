#!/usr/bin/env python3
"""Reconcile the September 8 CPU attention evidence; no model or GPU imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


AUDIT_RELATIVE = Path("reports/attention_audit_2026-09-08")
SOURCE_HASHES = {
    "original": "3aac24cec583a6ef5f60b6ec634a8bd3c8377784c5c63c8cc14cb6790554c52e",
    "compatibility": "34b00dd58c9887780a7947329cb96468a7fe1427e8fa49dc773ce2c1afc627d4",
}
ARTIFACT_HASHES = {
    "original_full_v1.json": "731e48665e67a68b84bf2cbb8f31674bff44fe78860827e4296ad2e9ce72c091",
    "compatibility_full_v1_stop.json": "9035b764313872f1855fbc367c5958c07448ab064c86c7d819061891a27f89b8",
    "original_components_v2.json": "abc2df05160d2976110a238848d61fe42f514210fee22d617005363c08d09f2b",
    "compatibility_components_v2.json": "ef5cc8bd9b3c73056735340862a3e89790f659178a8d2238df10c07106a2f74c",
    "original_full_v2.json": "882954cc0da11eb997ff1022c82fde8cf062db20e84310cc24b79210e4886b9e",
}
CASES = {"all_valid", "changed_future_token", "right_padding", "changed_padding_token"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def classify(result):
    require(result.get("complete") is True, "probe incomplete")
    for key, expected in {
        "schema_version": "openvla-attention-contract-cpu-v2",
        "device": "cpu", "dtype": "float32", "tiny_model_forwards": 8,
        "openvla_model_loads": 0, "simulator_episodes": 0,
        "weights_downloaded": False, "sdpa_observer_restored": True,
        "cache_production_hidden_difference": 0.0,
    }.items():
        require(result.get(key) == expected, f"invalid {key}")
    calls = result["calls"]
    expected_cases = {(cache, case) for cache in (False, True) for case in CASES}
    require(len(calls) == 8, "incorrect call count")
    require({(c["use_cache"], c["case"]) for c in calls} == expected_cases,
            "duplicate or missing case")
    possible = {"bidirectional", "causal"}
    for call in calls:
        padded = call["case"] in {"right_padding", "changed_padding_token"}
        require(len(call["sdpa_layers"]) == 2, "missing attention layer")
        for layer in call["sdpa_layers"]:
            actual = layer["allowed_keys_by_query"]
            possible &= {
                kind for kind in possible
                if actual == [
                    [int((not padded or j < 5) and (kind == "bidirectional" or j <= i))
                     for j in range(6)] for i in range(6)
                ]
            }
            require(layer["explicit_mask"] is padded, "unexpected mask presence")
    require(len(possible) == 1, "inconsistent or unknown effective attention masks")
    kind = possible.pop()
    for call in calls:
        padded = call["case"] in {"right_padding", "changed_padding_token"}
        for layer in call["sdpa_layers"]:
            require(layer["is_causal"] is (kind == "causal" and not padded),
                    "SDPA flag does not agree with effective mask")
    sensitivities = result["sensitivities"]
    require(len(sensitivities) == 2 and
            {r["use_cache"] for r in sensitivities} == {False, True},
            "missing sensitivity case")
    for record in sensitivities:
        future = record["earlier_tokens_change_when_valid_future_token_changes"]
        pad = record["earlier_tokens_change_when_padding_changes"]
        require(math.isfinite(future) and math.isfinite(pad), "nonfinite sensitivity")
        require(pad == 0.0, "padding changed valid hidden states")
        require(future > 1e-6 if kind == "bidirectional" else future == 0.0,
                "future-token intervention disagrees with attention contract")
    return kind


def reconcile(original_full, original_components, compatibility_components):
    require(original_full["execution"] == "full_model", "missing full reference")
    require(original_components["execution"] == compatibility_components["execution"]
            == "dense_components", "incorrect component execution")
    kinds = [classify(r) for r in (original_full, original_components, compatibility_components)]
    require(kinds == ["bidirectional", "bidirectional", "causal"],
            "recorded mismatch was not reproduced")
    for result, label in ((original_full, "original"), (original_components, "original"),
                          (compatibility_components, "compatibility")):
        require(result["attention_source_sha256"] == SOURCE_HASHES[label],
                "runtime source fingerprint differs")
    require(original_full["calls"] == original_components["calls"],
            "component probe did not reproduce original attention calls")
    require(original_full["hidden_state_sha256"] == original_components["hidden_state_sha256"],
            "component probe did not reproduce original hidden tensors exactly")
    return {
        "complete": True, "attention_semantics_mismatch_confirmed": True,
        "original_attention": kinds[0], "compatibility_attention": kinds[2],
        "original_components_equal_full_model": True,
        "reconciled_v2_tiny_forwards": 24,
        "openvla_model_loads": 0, "simulator_episodes": 0,
        "closed_loop_effect_measured": False,
        "positive_method_result": False,
    }


def verify_files(root):
    directory = root / AUDIT_RELATIVE
    artifacts = {}
    for name, expected in ARTIFACT_HASHES.items():
        raw = (directory / name).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected, f"artifact changed: {name}")
        artifacts[name] = json.loads(raw)
    for name in ("original_full_v2.json", "original_components_v2.json",
                 "compatibility_components_v2.json"):
        require(artifacts[name]["transport_exit_code"] == 0, f"execution failed: {name}")
    failed = artifacts["compatibility_full_v1_stop.json"]
    require(failed["transport_exit_code"] == 1 and failed["result"]["complete"] is False,
            "initial CPU-incompatible forward not preserved")
    result = reconcile(*[artifacts[name]["result"] for name in (
        "original_full_v2.json", "original_components_v2.json",
        "compatibility_components_v2.json",
    )])
    result["authenticated_artifacts"] = len(artifacts)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(verify_files(args.root), sort_keys=True, allow_nan=False))
