#!/usr/bin/env python3
"""Create and attest the canonical non-interactive LIBERO config for V5-D v02."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_ROOT = Path("/home/ved/SAVR")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing V5-D v02 config preparation outside {EXPECTED_ROOT}: {root}")
    sys.path.insert(0, str(root / "src"))
    from savr.acr.v5_d_recovery import (
        V02_RUN_ID,
        create_libero_config_once,
        semantic_sha256,
        write_json_once,
    )
    from savr.acr.v5_d_runtime import load_v5_d_freeze

    config = load_v5_d_freeze(root)
    if config["run_id"] != V02_RUN_ID:
        raise SystemExit("V5-D v02 resolved run identity changed")
    run_root = root / "results" / config["run_id"]
    launch_path = run_root / "launch" / "record.json"
    if not launch_path.is_file():
        raise SystemExit("V5-D v02 launch manifest is missing")
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    if launch.get("run_id") != config["run_id"] or launch.get(
        "configuration_semantic_sha256"
    ) != config.get("semantic_sha256"):
        raise SystemExit("V5-D v02 launch/config identity mismatch")
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if launch.get("execution_revision") != revision:
        raise SystemExit("V5-D v02 launch revision changed")
    expected_parent = run_root / "cache" / "libero"
    if Path(os.environ.get("LIBERO_CONFIG_PATH", "")).resolve() != expected_parent.resolve():
        raise SystemExit("V5-D v02 LIBERO_CONFIG_PATH changed")
    attestation = create_libero_config_once(root, run_root, config["recovery_v02"])
    record = {
        "schema_version": "acr.v5d-libero-config.v2",
        "run_id": config["run_id"],
        "status": "created-and-verified",
        "created_at_utc": utc_now(),
        "execution_revision": revision,
        "configuration_semantic_sha256": config["semantic_sha256"],
        "launch_manifest_semantic_sha256": launch["semantic_sha256"],
        "config_path": attestation["path"],
        "config_sha256": attestation["sha256"],
        "config_bytes": attestation["bytes"],
        "mapping": attestation["mapping"],
        "created_once": True,
        "overwrite_permitted": False,
        "gpu_inspection": 0,
        "model_queries": 0,
        "simulator_episodes": 0,
        "simulator_resets": 0,
        "downloads": 0,
        "new_task_outcomes": 0,
    }
    record["semantic_sha256"] = semantic_sha256(record)
    write_json_once(run_root / "libero-config" / "record.json", record)
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
