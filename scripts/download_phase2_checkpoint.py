#!/usr/bin/env python3
"""Download and verify the one approved Phase 2A checkpoint revision."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


EXPECTED_ROOT = Path("/home/ved/SAVR")
REPOSITORY_ID = "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
REVISION = "638918f3d1c2e43a39a8a20772bdb8b91835e4b7"
EXPECTED_REMOTE_BYTES = 15_939_168_050
NETWORK_CAP_BYTES = 16 * 1024**3
CHECKPOINT_CAP_BYTES = 18 * 1024**3
ADDITIONAL_PROJECT_CAP_BYTES = 20 * 1024**3


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    if project_root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing to run outside {EXPECTED_ROOT}: {project_root}")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HOME"] = str(project_root / "cache" / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(project_root / "cache" / "huggingface" / "hub")

    checkpoint_dir = (
        project_root / "checkpoints" / "openvla-7b-oft-libero-four-suite"
    )
    report_path = project_root / "reports" / "runtime" / "phase2_checkpoint.json"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    info = api.model_info(REPOSITORY_ID, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise RuntimeError(f"Resolved revision mismatch: {info.sha}")

    remote_files = sorted(
        (
            {"path": sibling.rfilename, "size": sibling.size}
            for sibling in info.siblings
            if sibling.size is not None
        ),
        key=lambda item: item["path"],
    )
    remote_bytes = sum(item["size"] for item in remote_files)
    if remote_bytes != EXPECTED_REMOTE_BYTES:
        raise RuntimeError(
            f"Remote size changed: expected {EXPECTED_REMOTE_BYTES}, got {remote_bytes}"
        )
    if remote_bytes > NETWORK_CAP_BYTES or remote_bytes > CHECKPOINT_CAP_BYTES:
        raise RuntimeError("Pinned checkpoint exceeds an approved resource cap")

    free_before = shutil.disk_usage(project_root).free
    if free_before < ADDITIONAL_PROJECT_CAP_BYTES:
        raise RuntimeError("Insufficient project-filesystem capacity for approved cap")

    resolved_path = Path(
        snapshot_download(
            repo_id=REPOSITORY_ID,
            revision=REVISION,
            local_dir=checkpoint_dir,
            max_workers=4,
        )
    ).resolve()
    if resolved_path != checkpoint_dir.resolve():
        raise RuntimeError(f"Unexpected checkpoint destination: {resolved_path}")

    missing_or_mismatched = []
    for remote_file in remote_files:
        local_path = checkpoint_dir / remote_file["path"]
        if not local_path.is_file() or local_path.stat().st_size != remote_file["size"]:
            missing_or_mismatched.append(remote_file["path"])
    if missing_or_mismatched:
        raise RuntimeError(
            "Checkpoint verification failed: " + ", ".join(missing_or_mismatched)
        )

    checkpoint_bytes = directory_size(checkpoint_dir)
    if checkpoint_bytes > CHECKPOINT_CAP_BYTES:
        raise RuntimeError(
            f"Checkpoint path exceeds cap: {checkpoint_bytes} > {CHECKPOINT_CAP_BYTES}"
        )

    report = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_id": REPOSITORY_ID,
        "requested_revision": REVISION,
        "resolved_revision": info.sha,
        "checkpoint_path": str(checkpoint_dir),
        "remote_file_count": len(remote_files),
        "remote_bytes": remote_bytes,
        "checkpoint_path_bytes": checkpoint_bytes,
        "free_bytes_before": free_before,
        "free_bytes_after": shutil.disk_usage(project_root).free,
        "network_cap_bytes": NETWORK_CAP_BYTES,
        "checkpoint_cap_bytes": CHECKPOINT_CAP_BYTES,
        "additional_project_cap_bytes": ADDITIONAL_PROJECT_CAP_BYTES,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "files": remote_files,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
