"""Read-only source/license/config compatibility preflights for B2."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from savr.brace.types import B2ValidationError


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(path: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *arguments], text=True, stderr=subprocess.STDOUT
    ).strip()


def inspect_repository(
    root: Path,
    *,
    name: str,
    revision: str,
    required_paths: Sequence[str],
    license_paths: Sequence[str],
    configuration_paths: Sequence[str],
    required_markers: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Authenticate one pinned source without importing or executing its method."""

    if not (root / ".git").exists():
        raise B2ValidationError(f"{name} source is absent")
    actual_revision = git_output(root, "rev-parse", "HEAD")
    if actual_revision != revision:
        raise B2ValidationError(f"{name} revision differs from its B2 pin")
    if git_output(root, "status", "--porcelain"):
        raise B2ValidationError(f"{name} source is dirty")
    missing = [relative for relative in required_paths if not (root / relative).is_file()]
    if missing:
        raise B2ValidationError(f"{name} lacks required paths: {missing}")

    marker_results: dict[str, bool] = {}
    for relative, markers in required_markers.items():
        text = (root / relative).read_text(encoding="utf-8", errors="replace")
        marker_results[relative] = all(marker in text for marker in markers)
        if not marker_results[relative]:
            raise B2ValidationError(f"{name} source markers changed in {relative}")

    licenses = {
        relative: file_sha256(root / relative)
        for relative in license_paths
        if (root / relative).is_file()
    }
    configurations = {
        relative: file_sha256(root / relative)
        for relative in configuration_paths
        if (root / relative).is_file()
    }
    missing_configurations = [
        relative for relative in configuration_paths if not (root / relative).is_file()
    ]
    return {
        "name": name,
        "revision": actual_revision,
        "tree_sha256_identity": git_output(root, "rev-parse", "HEAD^{tree}"),
        "clean": True,
        "required_paths_present": True,
        "marker_checks": marker_results,
        "license_files": licenses,
        "license_resolved": bool(licenses),
        "configuration_files": configurations,
        "configuration_resolved": not missing_configurations,
        "missing_configuration_paths": missing_configurations,
    }


def compatibility_disposition(record: Mapping[str, Any], *, stack_mode: str) -> str:
    if not record.get("license_resolved"):
        return "blocked_missing_upstream_license"
    if not record.get("configuration_resolved"):
        return "blocked_missing_upstream_configuration"
    if stack_mode == "core-4.40.1":
        return "source_preflight_passed_core_stack_overlay_required"
    if stack_mode == "vendored-4.47.0":
        return "source_preflight_passed_isolated_stack_required"
    if stack_mode == "paper-only":
        return "paper_only_no_official_code_matched_reproduction_required"
    raise B2ValidationError("unknown compatibility stack mode")
