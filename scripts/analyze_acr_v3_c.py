#!/usr/bin/env python3
"""Reconcile the immutable ACR V3-C correctness and latency result."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from run_acr_v3_c import (
    CORRECTNESS_LABELS,
    PATHS,
    QUERY_CAP,
    canonical_bytes,
    expected_query_labels,
    summarize_timing,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "reports/runtime/acr_v3_c.json"
MANIFEST_PATH = ROOT / "reports/runtime/acr_v3_c_manifest.json"


def reconcile(result: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    supplied_semantic = result.get("result_semantic_sha256")
    payload = dict(result)
    payload.pop("result_semantic_sha256", None)
    recomputed_semantic = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    if supplied_semantic != recomputed_semantic:
        raise RuntimeError("V3-C result semantic hash mismatch")
    if (
        result.get("run_id") != "acr-v3c-correctness-latency-v01"
        or manifest.get("run_id") != result["run_id"]
        or result.get("query_count") != QUERY_CAP
        or tuple(result.get("query_labels", ())) != expected_query_labels()
        or tuple(result["query_labels"][: len(CORRECTNESS_LABELS)]) != CORRECTNESS_LABELS
    ):
        raise RuntimeError("V3-C run identity or query schedule changed")
    if len(set(result["query_labels"])) != QUERY_CAP:
        raise RuntimeError("V3-C query identities are not unique")

    queries = result.get("queries", [])
    warmups = [record for record in queries if record.get("kind") == "warmup"]
    timed = [record for record in queries if record.get("kind") == "timed"]
    if len(warmups) != 8 or len(timed) != 48:
        raise RuntimeError("V3-C timing record count is incomplete")
    warm_counts = Counter(record["path"] for record in warmups)
    timed_counts = Counter(record["path"] for record in timed)
    if warm_counts != Counter({path: 2 for path in PATHS}) or timed_counts != Counter(
        {path: 12 for path in PATHS}
    ):
        raise RuntimeError("V3-C timing path balance changed")

    for record in queries:
        path = record["path"]
        counts = record["timing"]["component_counts"]
        physical = tuple(int(counts.get(name, 0)) for name in ("siglip", "dinov2", "projector"))
        expected = (2, 2, 1) if path == "sequential-fr" else (1, 1, 1)
        if physical != expected:
            raise RuntimeError(f"V3-C physical call truth changed for {record['label']}")
        work = record.get("work")
        if path == "sequential-fr":
            if work is not None:
                raise RuntimeError("Sequential FR unexpectedly contains adapter work")
        else:
            expected_mode = path
            if path == "v3-refresh":
                expected_mode = "v3-refresh"
            elif path == "v3-reuse":
                expected_mode = "v3-reuse"
            if not isinstance(work, dict) or work.get("mode") != expected_mode:
                raise RuntimeError(f"V3-C logical work changed for {record['label']}")

    derived_timing = summarize_timing(queries)
    if derived_timing != result.get("timing_summary"):
        raise RuntimeError("V3-C timing summary is not reproducible")
    correctness = result.get("correctness", {})
    for input_label in ("input-a", "input-b"):
        for path in ("batched_fr", "v3_refresh"):
            proof = correctness[input_label][path]
            if (
                not proof["tokens"]["close"]
                or proof["tokens"]["maximum_absolute_difference"] != 0.0
                or not proof["actions"]["equal"]
            ):
                raise RuntimeError("V3-C refresh correctness proof changed")
    reuse = correctness.get("reuse", {})
    if not all(reuse[name]["equal"] for name in ("tokens", "actions", "oracle_actions")):
        raise RuntimeError("V3-C reuse correctness proof changed")
    if (
        result.get("status") != "pass"
        or result.get("correctness_pass") is not True
        or result.get("latency_pass") is not True
        or result["timing_summary"]["gates"].get("all_pass") is not True
        or result.get("source_trees_restored") is not True
        or result["checkpoint_restoration"].get("unexpected") != []
    ):
        raise RuntimeError("V3-C positive gate or restoration proof is incomplete")

    action_hashes = {record["actions_sha256"] for record in queries}
    if len(action_hashes) != 1:
        raise RuntimeError("V3-C identical timing inputs returned differing actions")
    gates = result["timing_summary"]["gates"]
    return {
        "schema_version": "acr.v3-c-reconciliation.v1",
        "run_id": result["run_id"],
        "status": result["status"],
        "query_count": result["query_count"],
        "unique_action_hashes": len(action_hashes),
        "correctness_pass": result["correctness_pass"],
        "latency_pass": result["latency_pass"],
        "gates": gates,
        "result_semantic_sha256": supplied_semantic,
    }


def main() -> int:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    print(json.dumps(reconcile(result, manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
