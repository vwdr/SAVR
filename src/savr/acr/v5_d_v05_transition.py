"""Output-independent aggregate telemetry stabilization for V5-D v05."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

from savr.acr.v5_d_runtime import semantic_sha256


class V05TransitionEligibilityError(RuntimeError):
    """The single frozen V05 transition window did not establish eligibility."""


class V05TransitionSampler:
    """Evaluate exactly one fixed aggregate-only transition window."""

    def __init__(
        self,
        *,
        run_id: str,
        rule: Mapping[str, Any],
        expected_index: int,
        expected_uuid: str,
        snapshot: Callable[[str], dict[str, Any]],
        sleep: Callable[[float], None],
        write_once: Callable[[dict[str, Any]], None],
    ) -> None:
        self._run_id = run_id
        self._rule = deepcopy(dict(rule))
        self._expected_index = expected_index
        self._expected_uuid = expected_uuid
        self._snapshot = snapshot
        self._sleep = sleep
        self._write_once = write_once
        self._final: dict[str, Any] | None = None
        self._error: V05TransitionEligibilityError | None = None

    def __call__(self, physical_id: str) -> dict[str, Any]:
        if self._error is not None:
            raise self._error
        if self._final is not None:
            return deepcopy(self._final)
        if physical_id != str(self._expected_index):
            self._error = V05TransitionEligibilityError(
                "V05 transition physical GPU index changed"
            )
            raise self._error

        self._sleep(float(self._rule["initial_discard_seconds"]))
        samples: list[dict[str, Any]] = []
        sample_count = int(self._rule["sample_count"])
        for index in range(sample_count):
            samples.append(deepcopy(self._snapshot(physical_id)))
            if index + 1 < sample_count:
                self._sleep(float(self._rule["sample_interval_seconds"]))

        checks = []
        for sample in samples:
            checks.append(
                {
                    "identity": sample.get("index") == self._expected_index
                    and sample.get("uuid") == self._expected_uuid,
                    "memory": int(sample.get("memory_used_mib", -1))
                    <= int(self._rule["maximum_memory_used_mib_each_sample"]),
                    "utilization": int(sample.get("utilization_percent", -1))
                    <= int(self._rule["maximum_utilization_percent_each_sample"]),
                }
            )
        passed = len(samples) == sample_count and all(all(item.values()) for item in checks)
        record = {
            "schema_version": "acr.v5d-transition-revalidation.v1",
            "run_id": self._run_id,
            "status": "pass" if passed else "technical-stop",
            "backend": "raw-cudagraph",
            "expected_index": self._expected_index,
            "expected_uuid": self._expected_uuid,
            "rule": deepcopy(self._rule),
            "samples": samples,
            "checks": checks,
            "sample_count": len(samples),
            "all_samples_passed": passed,
            "process_identity_inspection": False,
            "allocation_inspection": False,
            "cuda_initialized": False,
            "model_loaded": False,
            "model_queries": 0,
            "automatic_retry_permitted": False,
        }
        record["semantic_sha256"] = semantic_sha256(record)
        self._write_once(record)
        if not passed:
            self._error = V05TransitionEligibilityError(
                "V05 transition eligibility window failed"
            )
            raise self._error
        self._final = deepcopy(samples[-1])
        return deepcopy(self._final)
