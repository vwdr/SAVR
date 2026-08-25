"""Pure validation primitives for the BRACE-B1 replay harness.

This module deliberately contains no LIBERO, model, CUDA, or filesystem side
effects. The bounded B1 runner supplies JSON-compatible simulator snapshots.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


TRANSCRIPT_SCHEMA = "brace.b1-transcript.v1"
REPLAY_MODE = "env_step_prefix"


class B1ValidationError(ValueError):
    """Raised when immutable replay evidence is incomplete or inconsistent."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise B1ValidationError(f"{label} must be a mapping")
    return value


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise B1ValidationError(f"{label} must be a SHA-256 identifier")
    try:
        int(value, 16)
    except ValueError as error:
        raise B1ValidationError(f"{label} must be hexadecimal") from error
    return value


def freeze_transcript(
    *,
    metadata: Mapping[str, Any],
    initial_snapshot: Mapping[str, Any],
    actions: Sequence[Sequence[float]],
    step_snapshots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create a chained, tamper-evident, complete B1 transcript."""

    frozen_actions = [[float(value) for value in action] for action in actions]
    if not frozen_actions or len(frozen_actions) != len(step_snapshots):
        raise B1ValidationError("actions and step snapshots must be nonempty and aligned")
    if any(len(action) != 7 or not all(math.isfinite(value) for value in action) for action in frozen_actions):
        raise B1ValidationError("every B1 low-level action must contain seven finite values")

    initial = dict(initial_snapshot)
    initial_sha = semantic_sha256(initial)
    previous_chain = semantic_sha256(
        {
            "schema_version": TRANSCRIPT_SCHEMA,
            "metadata": dict(metadata),
            "initial_snapshot_sha256": initial_sha,
        }
    )
    steps: list[dict[str, Any]] = []
    for index, (action, snapshot) in enumerate(zip(frozen_actions, step_snapshots, strict=True)):
        payload = {
            "step_index": index,
            "action": action,
            "action_sha256": semantic_sha256(action),
            "snapshot": dict(snapshot),
            "snapshot_sha256": semantic_sha256(snapshot),
        }
        payload["chain_sha256"] = semantic_sha256(
            {"previous_chain_sha256": previous_chain, "step": payload}
        )
        previous_chain = payload["chain_sha256"]
        steps.append(payload)

    transcript: dict[str, Any] = {
        "schema_version": TRANSCRIPT_SCHEMA,
        "status": "completed",
        "expected_steps": len(frozen_actions),
        "metadata": dict(metadata),
        "initial_snapshot": initial,
        "initial_snapshot_sha256": initial_sha,
        "steps": steps,
        "terminal_chain_sha256": previous_chain,
    }
    transcript["transcript_sha256"] = semantic_sha256(transcript)
    validate_transcript(transcript)
    return transcript


def validate_transcript(transcript: Mapping[str, Any]) -> None:
    """Reject incomplete, reordered, mutated, or self-inconsistent transcripts."""

    record = _require_mapping(transcript, "transcript")
    if record.get("schema_version") != TRANSCRIPT_SCHEMA:
        raise B1ValidationError("unsupported B1 transcript schema")
    if record.get("status") != "completed":
        raise B1ValidationError("B1 transcript is not terminal-complete")
    expected = record.get("expected_steps")
    steps = record.get("steps")
    if not isinstance(expected, int) or expected <= 0:
        raise B1ValidationError("expected_steps must be a positive integer")
    if not isinstance(steps, list) or len(steps) != expected:
        raise B1ValidationError("B1 transcript step count is incomplete")

    metadata = _require_mapping(record.get("metadata"), "metadata")
    initial = _require_mapping(record.get("initial_snapshot"), "initial_snapshot")
    initial_sha = _require_sha(record.get("initial_snapshot_sha256"), "initial snapshot hash")
    if semantic_sha256(initial) != initial_sha:
        raise B1ValidationError("initial snapshot hash mismatch")
    previous_chain = semantic_sha256(
        {
            "schema_version": TRANSCRIPT_SCHEMA,
            "metadata": dict(metadata),
            "initial_snapshot_sha256": initial_sha,
        }
    )
    for index, raw_step in enumerate(steps):
        step = _require_mapping(raw_step, f"step {index}")
        if step.get("step_index") != index:
            raise B1ValidationError("B1 transcript steps are not contiguous")
        action = step.get("action")
        if not isinstance(action, list) or len(action) != 7:
            raise B1ValidationError("B1 transcript action shape is invalid")
        if semantic_sha256(action) != step.get("action_sha256"):
            raise B1ValidationError("B1 transcript action hash mismatch")
        snapshot = _require_mapping(step.get("snapshot"), f"step {index} snapshot")
        if semantic_sha256(snapshot) != step.get("snapshot_sha256"):
            raise B1ValidationError("B1 transcript snapshot hash mismatch")
        chain_payload = {
            "previous_chain_sha256": previous_chain,
            "step": {key: value for key, value in step.items() if key != "chain_sha256"},
        }
        expected_chain = semantic_sha256(chain_payload)
        if expected_chain != step.get("chain_sha256"):
            raise B1ValidationError("B1 transcript chain hash mismatch")
        previous_chain = expected_chain
    if previous_chain != record.get("terminal_chain_sha256"):
        raise B1ValidationError("B1 terminal chain hash mismatch")

    supplied = _require_sha(record.get("transcript_sha256"), "transcript hash")
    payload = {key: value for key, value in record.items() if key != "transcript_sha256"}
    if semantic_sha256(payload) != supplied:
        raise B1ValidationError("B1 transcript semantic hash mismatch")


@dataclass(frozen=True)
class ReconstructionVerdict:
    accepted: bool
    restoration_mode: str
    mismatches: tuple[str, ...]

    def as_record(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "restoration_mode": self.restoration_mode,
            "mismatches": list(self.mismatches),
        }


def _numeric_vector(value: Mapping[str, Any], path: str) -> tuple[float, ...]:
    if value.get("kind") != "numeric":
        raise B1ValidationError(f"{path} is not a numeric snapshot field")
    raw = value.get("values")
    if not isinstance(raw, list):
        raise B1ValidationError(f"{path}.values must be a list")
    vector = tuple(float(item) for item in raw)
    if not all(math.isfinite(item) for item in vector):
        raise B1ValidationError(f"{path} contains nonfinite values")
    shape = value.get("shape")
    if not isinstance(shape, list) or math.prod(int(item) for item in shape) != len(vector):
        raise B1ValidationError(f"{path} shape does not match flattened values")
    return vector


def _compare(
    reference: Any,
    candidate: Any,
    *,
    path: str,
    absolute_tolerance: float,
    relative_tolerance: float,
    mismatches: list[str],
) -> None:
    if isinstance(reference, Mapping):
        if not isinstance(candidate, Mapping):
            mismatches.append(f"{path}:type")
            return
        if reference.get("kind") == "numeric":
            try:
                left = _numeric_vector(reference, path)
                right = _numeric_vector(candidate, path)
            except B1ValidationError:
                mismatches.append(f"{path}:numeric-field-invalid")
                return
            if reference.get("shape") != candidate.get("shape"):
                mismatches.append(f"{path}:shape")
                return
            if reference.get("dtype") != candidate.get("dtype"):
                mismatches.append(f"{path}:dtype")
                return
            if len(left) != len(right):
                mismatches.append(f"{path}:length")
                return
            if any(
                abs(a - b) > absolute_tolerance + relative_tolerance * abs(a)
                for a, b in zip(left, right, strict=True)
            ):
                mismatches.append(f"{path}:values")
            return
        if set(reference) != set(candidate):
            mismatches.append(f"{path}:keys")
            return
        for key in sorted(reference):
            if key == "snapshot_sha256":
                continue
            _compare(
                reference[key],
                candidate[key],
                path=f"{path}.{key}",
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                mismatches=mismatches,
            )
        return
    if isinstance(reference, list):
        if not isinstance(candidate, list) or len(reference) != len(candidate):
            mismatches.append(f"{path}:list")
            return
        for index, (left, right) in enumerate(zip(reference, candidate, strict=True)):
            _compare(
                left,
                right,
                path=f"{path}[{index}]",
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                mismatches=mismatches,
            )
        return
    if isinstance(reference, bool) or isinstance(candidate, bool):
        if reference is not candidate:
            mismatches.append(f"{path}:exact")
        return
    if isinstance(reference, int) or isinstance(candidate, int):
        if type(reference) is not type(candidate) or reference != candidate:
            mismatches.append(f"{path}:exact")
        return
    if isinstance(reference, float) or isinstance(candidate, float):
        try:
            left, right = float(reference), float(candidate)
        except (TypeError, ValueError):
            mismatches.append(f"{path}:numeric")
            return
        if not math.isfinite(left) or not math.isfinite(right):
            mismatches.append(f"{path}:nonfinite")
        elif abs(left - right) > absolute_tolerance + relative_tolerance * abs(left):
            mismatches.append(f"{path}:value")
        return
    if reference != candidate:
        mismatches.append(f"{path}:exact")


def validate_reconstruction(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    restoration_mode: str,
    absolute_tolerance: float,
    relative_tolerance: float = 0.0,
) -> ReconstructionVerdict:
    """Validate replay snapshots and reject every non-prefix restoration mode."""

    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise B1ValidationError("replay tolerances must be non-negative")
    mismatches: list[str] = []
    if restoration_mode != REPLAY_MODE:
        mismatches.append("restoration_mode:not-env-step-prefix")
    _compare(
        dict(reference),
        dict(candidate),
        path="snapshot",
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        mismatches=mismatches,
    )
    return ReconstructionVerdict(
        accepted=not mismatches,
        restoration_mode=restoration_mode,
        mismatches=tuple(mismatches),
    )
