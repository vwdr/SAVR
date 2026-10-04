"""Frozen four-arm adaptive screen accounting; science triage never changes execution.

Timing-spec correction (frozen): fixed-384 (and dense-512) stay compressed/dense
from query one; only the adaptive arms run the release's algorithm-required
dense startup (first 3 queries after each episode reset), inside the same
synchronized query boundary. Regimes: 'uniform' for controls, 'warm' (queries
1-3, dense by algorithm) and 'steady' (queries 4+, pruned) for adaptive arms.
The per-arm all-query mean — inclusive of adaptive startup — is the primary
timing comparison; warm and steady means are reported separately.

Parity gate: native shadows run against both dense-512 (<=1e-6, unchanged) and
the backend-aligned adaptive path with fastv_r=0 ('adaptive-zero', native SDPA).
The adaptive-zero bound is not assumed: stage S0 (real-checkpoint adaptive
qualification) measures zero-pruning parity and pins the achieved tolerance;
this config freezes only the acceptance ceiling.
"""
import hashlib
import math
from .contemporary_contract import SUITES, HORIZONS, require, finite

ARMS = ("512", "384", "adaptive_75", "adaptive_50")
RETAINED = {"512": 512, "384": 384, "adaptive_75": 384, "adaptive_50": 256}
CAPS = dict(model_calls=7000, episodes=168, seconds=21600,
            aggregate_memory_mib=23552, artifact_bytes=536870912)
TRIAGE = dict(dense_minimum_successes=39, maximum_success_count_loss=6,
              minimum_mean_time_reduction=.10, preference=[])
ADAPTIVE_ZERO_TOLERANCE_MAX = 1e-3  # acceptance ceiling; S0 pins the achieved bound


def regime(arm, query):
    if arm in ("512", "384", "native"):
        return "uniform"
    return "warm" if query <= 3 else "steady"


def expected_retained(arm, query):
    if arm in ("512", "native"):
        return 512
    if arm == "384":
        return 384
    if regime(arm, query) == "warm":
        return 512
    return RETAINED[arm]


def episode_slots(reference):
    rows = reference["episode_conditions"]
    require(len(rows) == 40 and len({r["condition_id"] for r in rows}) == 40,
            "40 distinct conditions required")
    slots = []

    def add(condition, arm, role):
        slots.append(dict(slot=len(slots) + 1,
                          slot_id=f"{role}:{condition['condition_id']}:{arm}",
                          condition={k: v for k, v in condition.items() if k != "arm_order"},
                          arm=arm, role=role))

    for suite_index, suite in enumerate(SUITES):
        local = rows[10 * suite_index:10 * (suite_index + 1)]
        require(all(r["suite"] == suite and r["initial_state_id"] == 0 and r["seed"] == 7
                    for r in local), "population differs")
        add(local[0], "native", "native_before")
        for i, row in enumerate(local):
            order = ARMS[(i + suite_index) % 4:] + ARMS[:(i + suite_index) % 4]
            for arm in order:
                add(row, arm, "primary")
        add(local[0], "native", "native_after")
    return slots


def timing_slots(ids):
    require(len(ids) == len(set(ids)) == 8, "eight unique trajectories required")
    output = []

    def add(warmup, round_id, trajectory, arm, frame, position):
        output.append(dict(index=len(output) + 1, warmup=warmup, round=round_id,
                           trajectory=trajectory, observation_id=ids[trajectory],
                           arm=arm, frame=frame, position=position,
                           regime=regime(arm, position)))

    # Hardware warm-up: two labeled rounds on the first trajectory as 2-query
    # traces (frames 0 then 1), excluded from every statistic. Adaptive arms
    # run dense here too (queries 1-2 are within their startup window).
    for round_id in range(2):
        order = ARMS[round_id % 4:] + ARMS[:round_id % 4]
        for arm in order:
            for frame in (0, 1):
                add(True, round_id, 0, arm, frame, frame + 1)
    # Measured: per (trajectory, arm) one 7-query trace on shared per-episode
    # state (query == position, frames 0|1 for positions 1..3|4..7). Adaptive
    # queries 1-3 are the required dense startup (also counted); queries 4-7
    # are the steady-state pruned regime measured after it.
    for trajectory in range(8):
        order = ARMS[trajectory % 4:] + ARMS[:trajectory % 4]
        for arm in order:
            for position in range(1, 8):
                add(False, None, trajectory, arm, 0 if position <= 3 else 1, position)
    return output


def validate_config(c, reference, ids):
    require(c["schema_version"] == "adaptive-screen-v1" and c["launch_ready"] is True
            and c["caps"] == CAPS and c["automatic_retry"] is False
            and c["arms"] == list(ARMS) and c["tolerance"] == 1e-6
            and c["output_root"] == "results/adaptive-screen-v03", "screen contract changed")
    require(c["episode_slots"] == episode_slots(reference)
            and c["timing_slots"] == timing_slots(ids), "schedule differs")
    require(c["triage"] == TRIAGE, "triage differs")
    require(type(c["adaptive_zero_tolerance_max"]) in (int, float)
            and math.isfinite(c["adaptive_zero_tolerance_max"])
            and 1e-6 <= c["adaptive_zero_tolerance_max"] <= ADAPTIVE_ZERO_TOLERANCE_MAX,
            "invalid frozen adaptive-zero ceiling")
    return True


def reconcile(c, reference, s, r, ids):
    validate_config(c, reference, ids)
    require(s.get("complete") is True and s.get("training_performed") is False
            and s.get("automatic_retry") is False and s.get("positive_method_result") is False
            and s.get("checkpoint_unchanged") is True and s.get("authenticated_files_unchanged") is True,
            "incomplete evidence")
    require(s["worker_sha256"] == c["worker_sha256"], "worker differs")
    require(finite(s["elapsed_seconds"], upper=CAPS["seconds"])
            and finite(s["peak_aggregate_gpu_memory_mib"], upper=CAPS["aggregate_memory_mib"]),
            "resource cap exceeded")
    zero_tolerance = s.get("adaptive_zero_tolerance")
    require(type(zero_tolerance) in (int, float) and math.isfinite(zero_tolerance)
            and 1e-6 <= zero_tolerance <= c["adaptive_zero_tolerance_max"],
            "S0-pinned adaptive-zero bound absent or outside the frozen ceiling")
    require(len(r["episodes"]) == s["episodes"] == 168 and len(r["timing"]) == 240,
            "counts differ")
    calls = 240
    parity = []
    for slot, e in zip(c["episode_slots"], r["episodes"]):
        require(e["slot_id"] == slot["slot_id"] and e["arm"] == slot["arm"]
                and e["condition_id"] == slot["condition"]["condition_id"]
                and e["suite"] == slot["condition"]["suite"], "episode ordering differs")
        n, steps = e["policy_queries"], e["executed_steps"]
        require(type(e["success"]) is bool and type(n) is int and type(steps) is int
                and 0 < n <= steps <= HORIZONS[e["suite"]], "episode horizon/query mismatch")
        require(e["controller_enabled"] is False and e["replans"] == e["discarded_actions"] == 0
                and type(e["remaining_actions"]) is int and 0 <= e["remaining_actions"] < 8
                and 8 * n == steps + e["remaining_actions"], "queue/controller changed")
        require(len(e["queries"]) == n and finite(e["episode_seconds"], strict_lower=True)
                and all(finite(e[k]) for k in ("simulator_seconds", "controller_seconds",
                                               "nonquery_preparation_seconds")), "episode cost missing")
        for i, q in enumerate(e["queries"], 1):
            require(q["query"] == i and q["retained_visual_tokens"] == expected_retained(e["arm"], i)
                    and q["precise"] is False and finite(q["seconds"], strict_lower=True),
                    "query budget/cost differs")
        require(sum(q["seconds"] for q in e["queries"]) <= e["episode_seconds"],
                "query cost exceeds episode")
        calls += n * (3 if e["arm"] == "native" else 1)
        if e["arm"] == "native":
            parity.extend((e["slot_id"], i, pair) for i in range(1, n + 1)
                          for pair in ("native-vs-512", "native-vs-adaptive-zero"))
    require([(p["slot_id"], p["query"], p["pair"]) for p in r["parity"]] == parity,
            "shadow accounting differs")
    for p in r["parity"]:
        require(p["command_bytes_equal"] is True
                and set(p["errors"]) == {"hidden", "normalized", "raw"}
                and all(finite(v) and v >= 0 for v in p["errors"].values()), "shadow output invalid")
        if p["pair"] == "native-vs-512":
            # Both arms run the released SDPA decoder (output_attentions=False);
            # the released diagnostic must observe each of the 32 layers once.
            require(p["layers"] == [32, 32] and p["sdpa_calls"] >= 1
                    and p["shadow_sdpa_calls"] >= 1
                    and all(v <= 1e-6 for v in p["errors"].values()), "native/512 parity failed")
        else:
            # Backend-aligned adaptive-zero must preserve all native decoder calls.
            require(p["pair"] == "native-vs-adaptive-zero" and p["layers"] == [32, 32]
                    and p["sdpa_calls"] >= 1 and p["shadow_sdpa_calls"] == 32
                    and all(v <= zero_tolerance for v in p["errors"].values()),
                    "adaptive-zero parity exceeds the S0-pinned bound")
        for name in ("observation_sha256", "command_sha256"):
            h = p[name]
            require(isinstance(h, str) and len(h) == 64
                    and all(x in "0123456789abcdef" for x in h), "invalid trace hash")
    for slot, t in zip(c["timing_slots"], r["timing"]):
        require(all(t[k] == v for k, v in slot.items()) and t["query"] == t["position"]
                and t["retained_visual_tokens"] == expected_retained(t["arm"], t["position"])
                and t["precise"] is False and finite(t["seconds"], strict_lower=True),
                "timing order/cost differs")
    require(type(s["model_queries"]) is int and s["model_queries"] == calls <= CAPS["model_calls"],
            "model calls differ")
    return True
