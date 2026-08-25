"""Immutable pre-outcome assignment and intent-to-treat records."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from savr.brace.b1 import semantic_sha256
from savr.brace.types import B2ValidationError


FORBIDDEN_INTENT_FIELDS = {
    "reward",
    "success",
    "terminal_success",
    "current_action",
    "current_logits",
    "current_attention",
    "privileged_state",
}


def freeze_intent_record(record: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "run_id",
        "episode_group",
        "branch_id",
        "assigned_contract",
        "inclusion_probability",
        "assignment_probability",
        "arm_order",
        "configuration_sha256",
    }
    if set(record) & FORBIDDEN_INTENT_FIELDS:
        raise B2ValidationError("intent record contains outcome or forbidden current-forward data")
    if not required <= set(record):
        raise B2ValidationError("intent record is incomplete")
    inclusion = float(record["inclusion_probability"])
    assignment = float(record["assignment_probability"])
    if not 0 < inclusion <= 1 or not 0 < assignment <= 1:
        raise B2ValidationError("assignment and inclusion propensities must be nonzero")
    if tuple(record["arm_order"]) not in (("FR", "CONTRACT"), ("CONTRACT", "FR")):
        raise B2ValidationError("arm order is invalid")
    payload = dict(record)
    payload["schema_version"] = "brace.b2-intent.v1"
    payload["status"] = "precommitted"
    payload["semantic_sha256"] = semantic_sha256(payload)
    return payload


def validate_intent_record(record: Mapping[str, Any]) -> None:
    supplied = record.get("semantic_sha256")
    payload = {key: value for key, value in record.items() if key != "semantic_sha256"}
    if semantic_sha256(payload) != supplied:
        raise B2ValidationError("intent record hash mismatch")
    if record.get("schema_version") != "brace.b2-intent.v1" or record.get("status") != "precommitted":
        raise B2ValidationError("intent record schema or state is invalid")


def experimental_execution_schedule(
    *, horizon: int, abort_at: int | None = None
) -> tuple[str, ...]:
    if horizon not in (1, 2, 4):
        raise B2ValidationError("experimental horizon must be 1, 2, or 4")
    if abort_at is not None and not 0 <= abort_at < horizon:
        raise B2ValidationError("abort index lies outside the assignment horizon")
    schedule = []
    locked = False
    for index in range(horizon):
        if abort_at == index:
            locked = True
        schedule.append("FR" if locked else "CONTRACT")
    return tuple(schedule)


def assert_duplicate_arm_identity(left: Sequence[str], right: Sequence[str]) -> None:
    if tuple(left) != tuple(right):
        raise B2ValidationError("duplicate-arm records are discordant")
