"""Dependency-light contracts for the frozen BRACE-B3 physical gate."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class B3ProtocolError(RuntimeError):
    """Raised when the frozen B3 contract is violated."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def semantic_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def load_config_file(root: Path, relative: Path) -> dict[str, Any]:
    """Load a frozen B3 config or its narrow pre-model recovery overlay."""

    path = root / relative
    value = json.loads(path.read_text())
    if value.get("schema_version") != "brace.b3-recovery.v1":
        validate_config(value)
        return value
    if value.get("semantic_sha256") != semantic_sha256(value):
        raise B3ProtocolError("B3 recovery-record semantic hash mismatch")
    base_relative = Path(value["base_configuration_relative"])
    if base_relative.is_absolute() or ".." in base_relative.parts:
        raise B3ProtocolError("B3 recovery base path is unsafe")
    base = json.loads((root / base_relative).read_text())
    validate_config(base)
    if base["semantic_sha256"] != value["base_configuration_semantic_sha256"]:
        raise B3ProtocolError("B3 recovery base identity changed")
    resolved = dict(base)
    resolved["run_id"] = value["run_id"]
    resolved["recovery"] = {
        "attempt": 2,
        "prior_run_id": value["prior_run_id"],
        "correction": value["correction"],
    }
    resolved["semantic_sha256"] = value["resolved_configuration_semantic_sha256"]
    validate_config(resolved)
    return resolved


def allowed_project_status(raw: bytes, run_id: str) -> bool:
    """Accept only raw NUL-delimited tmp evidence and this run's launch record."""

    entries = [entry.decode("utf-8") for entry in raw.split(b"\0") if entry]
    for entry in entries:
        if len(entry) < 4 or entry[:3] != "?? ":
            return False
        path = entry[3:]
        if path == "tmp" or path.startswith("tmp/"):
            continue
        if path == f"results/{run_id}/launch.json":
            continue
        return False
    return True


def planned_query_count(config: Mapping[str, Any]) -> int:
    measurement = config["measurement"]
    return sum(
        int(measurement[key])
        for key in (
            "core_fr_queries",
            "cache_p0_queries",
            "attention_parity_queries",
            "corrected_vla_cache_queries",
            "clean_profile_queries",
            "vla_adp_queries",
            "vla_pruner_queries",
        )
    )


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "brace.b3-config.v1":
        raise B3ProtocolError("B3 schema changed")
    if config.get("semantic_sha256") != semantic_sha256(config):
        raise B3ProtocolError("B3 semantic hash mismatch")
    if config.get("run_id") not in {"brace-b3-physical-v01", "brace-b3-physical-v02"}:
        raise B3ProtocolError("B3 run identity changed")
    authorization = config["authorization"]
    if authorization != {
        "user_authorized_b3": True,
        "separate_gpu_authorization": True,
        "b4_authorized": False,
    }:
        raise B3ProtocolError("B3 authorization boundary changed")
    profiles = config["profiles"]
    if not 1 <= len(profiles) <= 6:
        raise B3ProtocolError("B3 requires one to six clean profiles")
    if len({profile["profile_id"] for profile in profiles}) != len(profiles):
        raise B3ProtocolError("B3 profile identities must be unique")
    for profile in profiles:
        scene = tuple(int(value) for value in profile["scene_budgets"])
        wrist = tuple(int(value) for value in profile["wrist_budgets"])
        if len(scene) != 4 or len(wrist) != 4:
            raise B3ProtocolError("B3 budgets must map to four pruning layers")
        if tuple(sorted(scene)) != scene or tuple(sorted(wrist)) != wrist:
            raise B3ProtocolError("B3 reuse budgets must be nondecreasing")
        if profile["family"] == "P1" and any(wrist):
            raise B3ProtocolError("P1 may not reuse wrist tokens")
        if profile["family"] not in {"P1", "P2"}:
            raise B3ProtocolError("Only P1/P2 are clean selectable profiles")
    measurement = config["measurement"]
    if measurement["horizons"] != [1, 2, 4]:
        raise B3ProtocolError("B3 horizons changed")
    expected_clean = len(profiles) * (
        int(measurement["profile_warmup_cycles_each"])
        * (1 + int(measurement["profile_warmup_horizon"]))
        + int(measurement["timed_cycles_per_profile_horizon"])
        * sum(1 + int(horizon) for horizon in measurement["horizons"])
    )
    if int(measurement["clean_profile_queries"]) != expected_clean:
        raise B3ProtocolError("B3 clean-profile query accounting changed")
    planned = planned_query_count(config)
    if planned != int(measurement["planned_model_queries"]):
        raise B3ProtocolError("B3 planned query total is inconsistent")
    caps = config["resource_caps"]
    if planned > int(caps["model_query_hard_cap"]) or int(caps["model_query_hard_cap"]) > 480:
        raise B3ProtocolError("B3 query cap exceeds its authorization")
    if (
        int(caps["gpu_count"]) != 1
        or int(caps["model_processes_at_once"]) != 1
        or int(caps["simulator_outcomes"]) != 0
        or bool(caps["protected_outcome_access"])
        or int(caps["downloads"]) != 0
        or bool(caps["automatic_retry"])
    ):
        raise B3ProtocolError("B3 protected resource boundary changed")
    if config["advance"] != {
        "next_phase": "B4",
        "requires_separate_authorization": True,
        "stop_before_next_phase": True,
    }:
        raise B3ProtocolError("B3 advance boundary changed")


@dataclass
class QueryLedger:
    hard_cap: int
    planned_by_method: Mapping[str, int]

    def __post_init__(self) -> None:
        self._used: dict[str, int] = {name: 0 for name in self.planned_by_method}
        if sum(self.planned_by_method.values()) > self.hard_cap:
            raise B3ProtocolError("Planned B3 queries exceed the hard cap")

    def consume(self, method: str, count: int = 1) -> None:
        if method not in self._used or count <= 0:
            raise B3ProtocolError("Invalid B3 query-ledger request")
        if self._used[method] + count > int(self.planned_by_method[method]):
            raise B3ProtocolError(f"B3 method query allocation exceeded: {method}")
        if self.total + count > self.hard_cap:
            raise B3ProtocolError("B3 total query cap exceeded")
        self._used[method] += count

    @property
    def total(self) -> int:
        return sum(self._used.values())

    def record(self) -> dict[str, Any]:
        return {"used": dict(self._used), "total": self.total, "hard_cap": self.hard_cap}


def cycle_schedule(config: Mapping[str, Any]) -> tuple[tuple[str, int, int], ...]:
    """Return fixed-seed randomized profile/horizon/repetition blocks."""

    measurement = config["measurement"]
    rows = [
        (profile["profile_id"], int(horizon), repetition)
        for profile in config["profiles"]
        for horizon in measurement["horizons"]
        for repetition in range(int(measurement["timed_cycles_per_profile_horizon"]))
    ]
    random.Random(int(measurement["seed"])).shuffle(rows)
    return tuple(rows)


def empirical_quantile(values: Sequence[float], probability: float) -> float:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if len(finite) != len(values) or not finite:
        raise B3ProtocolError("B3 timing values must be nonempty and finite")
    if not 0 <= probability <= 1:
        raise B3ProtocolError("Invalid quantile probability")
    position = probability * (len(finite) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return finite[lower]
    weight = position - lower
    return finite[lower] * (1 - weight) + finite[upper] * weight


def summarize_timings(values: Sequence[float]) -> dict[str, float]:
    return {
        "p50": empirical_quantile(values, 0.5),
        "p95": empirical_quantile(values, 0.95),
        "p99": empirical_quantile(values, 0.99),
    }


def reduction(reference: float, candidate: float) -> float:
    if not math.isfinite(reference) or not math.isfinite(candidate) or reference <= 0 or candidate < 0:
        raise B3ProtocolError("Invalid B3 latency reduction inputs")
    return 1.0 - candidate / reference


def profile_speed_gate(
    *,
    p0_accelerated_ms: Sequence[float],
    accelerated_ms: Sequence[float],
    p0_cycle_ms: Sequence[float],
    contract_cycle_ms: Sequence[float],
    minimum_accelerated: float,
    minimum_cycle: float,
) -> dict[str, Any]:
    accelerated_reduction = reduction(
        empirical_quantile(p0_accelerated_ms, 0.5), empirical_quantile(accelerated_ms, 0.5)
    )
    cycle_reduction = reduction(
        empirical_quantile(p0_cycle_ms, 0.5), empirical_quantile(contract_cycle_ms, 0.5)
    )
    return {
        "accelerated_query_reduction": accelerated_reduction,
        "amortized_cycle_reduction": cycle_reduction,
        "passed": accelerated_reduction >= minimum_accelerated
        and cycle_reduction >= minimum_cycle,
    }
