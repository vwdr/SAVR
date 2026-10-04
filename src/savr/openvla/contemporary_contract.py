"""Prospective schedule and evidence checks. No task-success acceptance threshold."""
from collections import Counter
import hashlib
import json
import math

DESIGN_SHA = "2052468afb21fd7015542c460ca44c1174eb1d5585321a737ad71b5d45de54ba"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
HORIZONS = dict(zip(SUITES, (220, 280, 300, 520)))
CAPS = dict(model_calls=6000, episodes=88, seconds=21600,
            aggregate_memory_mib=23552, artifact_bytes=536870912)


def require(value, message):
    if not value:
        raise ValueError(message)


def finite(value, lower=0, upper=math.inf, *, strict_lower=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > lower if strict_lower else value >= lower) and value < upper)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def validate_design(design, reference):
    require(design["schema_version"] == "contemporary-reference-design-v1"
            and design["launch_ready"] is False and design["worker_sha256"] is None
            and design["analyzer_sha256"] is None, "design is not an executable config")
    require(design["caps"] == CAPS and design["seed"] == 7
            and design["expected_episodes"] == 88 and design["primary_pairs"] == 40
            and design["native_controls"] == 8, "population/caps changed")
    conditions = reference["episode_conditions"]
    require(len(conditions) == 40 and len({r["condition_id"] for r in conditions}) == 40,
            "reference conditions are not forty unique rows")
    expected = []

    def add(row, arm, role):
        condition = {k: v for k, v in row.items() if k != "arm_order"}
        expected.append(dict(slot=len(expected) + 1, slot_id=f"{role}:{row['condition_id']}:{arm}",
                             condition=condition, arm=arm, role=role, shadow_dense=arm == "native",
                             controller_enabled=arm == "specprune",
                             include_primary_success=role == "primary",
                             include_primary_latency=role == "primary"))

    for s, suite in enumerate(SUITES):
        rows = conditions[10 * s:10 * (s + 1)]
        require(all(r["suite"] == suite and r["seed"] == 7 and r["initial_state_id"] == 0
                    for r in rows), "reference suite/state/seed changed")
        add(rows[0], "native", "native_before")
        for i, row in enumerate(rows):
            order = ("dense", "specprune") if (i + s) % 2 == 0 else ("specprune", "dense")
            for arm in order:
                add(row, arm, "primary")
        add(rows[0], "native", "native_after")
    require(design["episode_slots"] == expected, "frozen episode ordering or membership differs")
    require(design["timing"] == dict(arms=["dense", "specprune"], frames=[0, 1],
                measured_model_calls=128, observation_manifest="configs/pair/p3_inputs_v1.json",
                profile="coarse_two_query_trace", rounds=4, total_model_calls=136,
                trajectories=8, warmup_model_calls=8, warmup_traces_per_arm=2),
            "short-trace timing design changed")
    return True


def timing_slots(observation_ids):
    require(len(observation_ids) == len(set(observation_ids)) == 8, "eight unique timing traces required")
    output = []
    for warmup, rounds in ((True, 2), (False, 4)):
        for round_id in range(rounds):
            for trajectory, identity in enumerate(observation_ids[:1] if warmup else observation_ids):
                order = ("dense", "specprune") if (trajectory + round_id) % 2 == 0 else ("specprune", "dense")
                for arm in order:
                    for frame in (0, 1):
                        output.append(dict(index=len(output) + 1, warmup=warmup, round=round_id,
                                           observation_id=identity, arm=arm, frame=frame))
    return output


def reconcile(design, reference, summary, records, observation_ids):
    """Reject missing/corrupt accounting without looking for favorable scientific outcomes."""
    validate_design(design, reference)
    require(summary.get("complete") is True and summary.get("automatic_retry") is False
            and summary.get("training_performed") is False
            and summary.get("positive_method_result") is False
            and summary.get("checkpoint_unchanged") is True
            and summary.get("authenticated_files_unchanged") is True,
            "incomplete or unauthenticated worker")
    require(finite(summary["elapsed_seconds"], upper=CAPS["seconds"])
            and finite(summary["peak_aggregate_gpu_memory_mib"], upper=CAPS["aggregate_memory_mib"]),
            "resource cap exceeded")
    episodes, timing, parity = records["episodes"], records["timing"], records["parity"]
    require(len(episodes) == summary["episodes"] == 88, "episode count differs")
    queries = 136
    expected_parity = []
    for slot, e in zip(design["episode_slots"], episodes):
        require(e["slot_id"] == slot["slot_id"] and e["condition_id"] == slot["condition"]["condition_id"]
                and e["arm"] == slot["arm"] and e["suite"] == slot["condition"]["suite"],
                "episode identity/order differs")
        n, steps = e["policy_queries"], e["executed_steps"]
        require(type(e["success"]) is bool and type(n) is int and type(steps) is int
                and 0 < n <= steps <= HORIZONS[e["suite"]], "episode shape or horizon differs")
        require(e["controller_enabled"] is slot["controller_enabled"], "controller arm mismatch")
        require(all(type(e[k]) is int and e[k] >= 0 for k in
                    ("replans", "discarded_actions", "remaining_actions")), "invalid action accounting")
        require(8 * n == steps + e["discarded_actions"] + e["remaining_actions"]
                and e["remaining_actions"] <= 7 and e["replans"] <= steps,
                "generated/executed/discarded actions do not reconcile")
        if e["arm"] != "specprune":
            require(e["replans"] == e["discarded_actions"] == 0, "dense/native queue changed")
        require(len(e["queries"]) == n and finite(e["episode_seconds"], strict_lower=True)
                and finite(e["simulator_seconds"]) and finite(e["controller_seconds"])
                and finite(e["nonquery_preparation_seconds"]), "missing episode costs")
        for i, q in enumerate(e["queries"], 1):
            require(q["query"] == i and finite(q["seconds"], strict_lower=True)
                    and type(q["retained_visual_tokens"]) is int
                    and 0 <= q["retained_visual_tokens"] <= 512
                    and type(q["precise"]) is bool, "invalid query costs/state")
            if e["arm"] != "specprune":
                require(q["retained_visual_tokens"] == 512 and q["precise"] is False,
                        "dense/native path changed")
        require(sum(q["seconds"] for q in e["queries"]) <= e["episode_seconds"], "query time exceeds episode")
        queries += n * (2 if e["arm"] == "native" else 1)
        if e["arm"] == "native":
            expected_parity.extend((e["slot_id"], i) for i in range(1, n + 1))
    require([(r["slot_id"], r["query"]) for r in parity] == expected_parity,
            "native shadow accounting differs")
    for p in parity:
        require(p["command_bytes_equal"] is True and p["layers"] == [32, 32]
                and set(p["errors"]) == {"hidden", "normalized", "raw_actions"}
                and all(finite(v) and v <= 1e-6 for v in p["errors"].values()),
                "same-observation parity failed")
        for name in ("observation_sha256", "command_sha256"):
            require(isinstance(p[name], str) and len(p[name]) == 64
                    and all(c in "0123456789abcdef" for c in p[name]), "invalid trace hash")
    expected_timing = timing_slots(observation_ids)
    require(len(timing) == len(expected_timing) == 136, "timing count differs")
    for expected, actual in zip(expected_timing, timing):
        require(all(actual.get(k) == v for k, v in expected.items()), "timing identity/order differs")
        require(finite(actual["seconds"], strict_lower=True) and actual["precise"] is False
                and actual["query"] == actual["frame"] + 1
                and type(actual["retained_visual_tokens"]) is int
                and 0 <= actual["retained_visual_tokens"] <= 512, "timing trace invalid")
        if actual["arm"] == "dense":
            require(actual["retained_visual_tokens"] == 512, "timing dense is not dense")
    require(type(summary["model_queries"]) is int
            and queries == summary["model_queries"] <= CAPS["model_calls"], "total model-call mismatch")
    return True
