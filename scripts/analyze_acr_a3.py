#!/usr/bin/env python3
"""CPU-only adjudication of the preserved A3 attempt after loader restoration."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_ROOT = Path("/home/ved/SAVR")
RUN_ID = "acr-a3-correctness-none-v01"
ATTEMPT = f"{RUN_ID}/mixed/synthetic/task-00/state-00/seed-0/attempt-0000"
RUNNER_REVISION = "043853ab59f5900703172b40a47617183e741e47"
OPENVLA_REVISION = "e4287e94541f459edc4feabc4e181f537cd569a8"
LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
CONFIG_SHA256 = "edd5c5cf6d7927e07465cf086ebe41f7b3ec8f3b128a51f71d6db14dad7ad8b1"
MODELING_SHA256 = "f40ee7883e16aab1a2d89b6e8f31cc81f6b8055120b1fefe169e05c7031098fa"
PLANNED_LABELS = [
    "upstream-a",
    "factorized-a",
    "factorized-scene-variant",
    "factorized-wrist-variant",
    "visual-warmup-0",
    "visual-warmup-1",
    "visual-reuse-current-state",
    "upstream-current-state-b",
    "fail-closed-shape",
    "fail-closed-dtype",
    "fail-closed-device",
    "fail-closed-context",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(path: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *arguments], text=True).strip()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ordered_control_flow(source: str) -> dict[str, int]:
    markers = [
        'proof["proofs"]["factorized_token_parity"]',
        'proof["proofs"]["factorized_action_parity"]',
        'proof["proofs"]["scene_isolation"]',
        'proof["proofs"]["wrist_isolation"]',
        'proof["proofs"]["reuse_visual_tokens"]',
        'proof["proofs"]["reuse_component_truth"]',
        'proof["proofs"]["reuse_upstream_token_parity"]',
        'proof["proofs"]["reuse_current_state_action_parity"]',
        'proof["proofs"]["metadata_fail_closed"]',
        'proof["proofs"]["context_fail_closed"]',
        "checkpoint_after = validate_checkpoint",
    ]
    positions = {marker: source.index(marker) for marker in markers}
    if list(positions.values()) != sorted(positions.values()):
        raise RuntimeError("Runner correctness/checkpoint gates are not sequential")
    for required in (
        "bitwise parity failed",
        "exact parity failed",
        "isolation failed",
        "did not fail closed",
        "did not reuse",
    ):
        if required not in source:
            raise RuntimeError(f"Runner hard-stop marker is missing: {required}")
    return positions


def camera_counts(record: dict[str, Any]) -> tuple[int, int, int, int, int, int, int]:
    work = record["camera_work"]
    return (
        work["scene_siglip_calls"],
        work["scene_dinov2_calls"],
        work["scene_projector_calls"],
        work["wrist_siglip_calls"],
        work["wrist_dinov2_calls"],
        work["wrist_projector_calls"],
        work["downstream_calls"],
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing to run outside {EXPECTED_ROOT}")
    sys.path.insert(0, str(root / "src"))
    from savr.acr.records import ImmutableRecordStore, validate_record

    run_root = root / "results" / RUN_ID
    failure = load(run_root / "failure/record.json")
    if failure["query_attempts"] != PLANNED_LABELS or failure["query_count"] != 12:
        raise RuntimeError("A3 query attempt ledger differs from the frozen matrix")
    if failure["error"] != "Checkpoint inventory mismatch: config.json, modeling_prismatic.py":
        raise RuntimeError("A3 terminal error is not the diagnosed loader lifecycle defect")
    if failure["rollout_episodes"] != 0 or failure["simulator_resets"] != 0:
        raise RuntimeError("A3 unexpectedly accessed a simulator or rollout")

    query_schema = load(root / "schemas/acr_query.schema.json")
    records: dict[int, dict[str, Any]] = {}
    for index in (*range(1, 7), *range(8, 12)):
        record = load(run_root / f"mixed/synthetic/task-00/state-00/seed-0/attempt-0000/query-{index:06d}/record.json")
        validate_record(record, query_schema)
        records[index] = record
    oracle_zero = load(run_root / "mixed/synthetic/task-00/state-00/seed-0/attempt-0000/oracle-query-000000/record.json")
    oracle_seven = load(run_root / "mixed/synthetic/task-00/state-00/seed-0/attempt-0000/oracle-query-000007/record.json")
    if records[1]["inputs"]["action_sha256"] != oracle_zero["actions"]["sha256"]:
        raise RuntimeError("Factorized/upstream action hashes differ")
    if records[6]["inputs"]["action_sha256"] != oracle_seven["actions"]["sha256"]:
        raise RuntimeError("Reuse/current-state upstream action hashes differ")
    if oracle_zero["projected_tokens"]["sha256"] != oracle_seven["projected_tokens"]["sha256"]:
        raise RuntimeError("Upstream visual tokens changed when only proprioception changed")

    for index in (1, 2, 3, 4, 5, 8, 9, 10, 11):
        if camera_counts(records[index]) != (1, 1, 1, 1, 1, 1, 1):
            raise RuntimeError(f"Fresh-camera accounting mismatch at query {index}")
    if camera_counts(records[6]) != (0, 0, 0, 1, 1, 1, 1):
        raise RuntimeError("Reuse query did not skip exactly the scene path")
    if records[6]["decision"]["scene_refresh"] or records[6]["decision"]["refresh_reasons"]:
        raise RuntimeError("Query 6 was not the required clean scene reuse")
    for index in (8, 9, 10, 11):
        if not records[index]["decision"]["scene_refresh"] or "cache" not in records[index]["decision"]["refresh_reasons"]:
            raise RuntimeError(f"Fail-closed query {index} did not refresh from cache/context")

    runner_source = subprocess.check_output(
        ["git", "-C", str(root), "show", f"{RUNNER_REVISION}:scripts/run_acr_correctness.py"],
        text=True,
    )
    control_positions = ordered_control_flow(runner_source)
    if failure["recorded_at_utc"] <= records[11]["provenance"]["recorded_at_utc"]:
        raise RuntimeError("Terminal checkpoint error did not occur after the final query")

    checkpoint = root / "checkpoints/openvla-7b-oft-libero-four-suite"
    if sha256(checkpoint / "config.json") != CONFIG_SHA256:
        raise RuntimeError("Checkpoint config was not restored")
    if sha256(checkpoint / "modeling_prismatic.py") != MODELING_SHA256:
        raise RuntimeError("Checkpoint modeling source was not restored")
    if list(checkpoint.glob("*.back.20260802_203808")):
        raise RuntimeError("Loader backup files remain after recovery")
    if git(root / "third_party/openvla-oft", "rev-parse", "HEAD") != OPENVLA_REVISION:
        raise RuntimeError("OpenVLA-OFT revision changed")
    if git(root / "third_party/openvla-oft", "status", "--porcelain"):
        raise RuntimeError("OpenVLA-OFT tree is dirty")
    if git(root / "third_party/LIBERO", "rev-parse", "HEAD") != LIBERO_REVISION:
        raise RuntimeError("LIBERO revision changed")
    if git(root / "third_party/LIBERO", "status", "--porcelain"):
        raise RuntimeError("LIBERO tree is dirty")

    artifact_files = sorted(path for path in run_root.rglob("record.json"))
    artifact_digest = hashlib.sha256()
    for path in artifact_files:
        artifact_digest.update(str(path.relative_to(run_root)).encode())
        artifact_digest.update(path.read_bytes())
    record = {
        "schema_version": "acr.a3-adjudication.v1",
        "run_id": RUN_ID,
        "disposition": "PASS_WITH_TECHNICAL_RECOVERY",
        "scientific_proofs_passed": True,
        "original_attempt_status": "failed",
        "technical_root_cause": "Pinned loader changes were audited before restoration instead of after restoration.",
        "additional_model_queries": 0,
        "total_model_queries": 12,
        "query_cap": 16,
        "simulator_resets": 0,
        "rollout_episodes": 0,
        "factorized_action_sha256": records[1]["inputs"]["action_sha256"],
        "reuse_current_state_action_sha256": records[6]["inputs"]["action_sha256"],
        "upstream_visual_sha256": oracle_zero["projected_tokens"]["sha256"],
        "reuse_scene_component_calls": list(camera_counts(records[6])[:3]),
        "reuse_wrist_component_calls": list(camera_counts(records[6])[3:6]),
        "control_flow_attestation": {
            "runner_revision": RUNNER_REVISION,
            "runner_sha256": hashlib.sha256(runner_source.encode()).hexdigest(),
            "ordered_gate_positions": control_positions,
            "terminal_gate": failure["error"],
        },
        "checkpoint_restored": True,
        "checkpoint_hashes": {
            "config.json": CONFIG_SHA256,
            "modeling_prismatic.py": MODELING_SHA256,
        },
        "upstream_trees_clean": True,
        "preserved_record_count_before_adjudication": len(artifact_files),
        "preserved_records_sha256": artifact_digest.hexdigest(),
        "adjudicator_revision": git(root, "rev-parse", "HEAD"),
        "recorded_at_utc": utc_now(),
    }
    ImmutableRecordStore(root / "results").write_once(f"{RUN_ID}/adjudication", record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
