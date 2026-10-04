from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from savr.pair.records import (
    freeze_technical_record,
    records_sha256,
    reject_outcome_fields,
    validate_schema,
)
from savr.pair.runtime import PairRuntime, RuntimeMode
from savr.pair.types import PairValidationError, ProfileSpec


ROOT = Path(__file__).resolve().parents[2]


def technical_record():
    return {
        "schema_version": "pair-technical-v1",
        "run_id": "pair-p2-cpu-v1",
        "phase": "P2",
        "check_id": "all-fresh-parity",
        "status": "passed",
        "synthetic": True,
        "cuda_visible": False,
        "model_accessed": False,
        "checkpoint_accessed": False,
        "simulator_accessed": False,
        "details": {"maximum_absolute_difference": 0.0},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def test_technical_records_validate_hash_and_reproduce():
    record = freeze_technical_record(technical_record())
    validate_schema(record, ROOT / "schemas/pair/technical_record.schema.json")
    assert records_sha256([record]) == records_sha256([record])
    changed = dict(record)
    changed["status"] = "failed_closed"
    with pytest.raises(PairValidationError, match="hash mismatch"):
        validate_schema(changed, ROOT / "schemas/pair/technical_record.schema.json")


def test_malformed_or_nested_outcome_records_fail_closed():
    with pytest.raises(PairValidationError, match="outcome"):
        freeze_technical_record({**technical_record(), "details": {"success": True}})
    with pytest.raises(PairValidationError, match="schema"):
        freeze_technical_record({**technical_record(), "phase": "P3"})
    with pytest.raises(PairValidationError, match="outcome"):
        reject_outcome_fields({"nested": [{"reward": 1}]})
    malformed = freeze_technical_record(technical_record())
    malformed["cuda_visible"] = True
    malformed["record_id"] = records_sha256([])
    with pytest.raises(PairValidationError, match="schema validation"):
        validate_schema(malformed, ROOT / "schemas/pair/technical_record.schema.json")


def test_contract_lifecycle_expiry_abort_and_episode_reset():
    profile = ProfileSpec("D37_BAL", (32, 64, 96, 128), (16, 32, 48, 64))
    runtime = PairRuntime.reset("episode-a")
    active = runtime.start(query=0, profile=profile, horizon=2)
    assert active.mode is RuntimeMode.CONTRACT and active.remaining == 2
    active = active.advance(query=1)
    expired = active.advance(query=2)
    assert expired.mode is RuntimeMode.DENSE and expired.last_query == 2
    active = expired.start(query=3, profile=profile, horizon=1)
    aborted = active.abort(query=4, reason="provenance_mismatch")
    assert aborted.mode is RuntimeMode.DENSE
    assert aborted.abort_history == ("provenance_mismatch",)
    reset = PairRuntime.reset("episode-b")
    assert reset.last_query == -1 and reset.abort_history == ()
    with pytest.raises(PairValidationError, match="contiguous"):
        active.advance(query=9)
