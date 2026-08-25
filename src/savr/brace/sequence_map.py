"""Runtime-derived multimodal sequence maps with no fixed prompt offsets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from savr.brace.b1 import semantic_sha256
from savr.brace.types import B2ValidationError, Camera


VISUAL_ROLES = {Camera.SCENE.value, Camera.WRIST.value}
NONVISUAL_ROLES = {"special", "text", "proprio", "action", "padding"}


@dataclass(frozen=True)
class SequenceMap:
    length: int
    roles: tuple[str, ...]
    scene_positions: tuple[int, ...]
    wrist_positions: tuple[int, ...]
    text_positions: tuple[int, ...]
    proprio_positions: tuple[int, ...]
    action_positions: tuple[int, ...]
    nonvisual_positions: tuple[int, ...]
    patch_by_position: tuple[tuple[int, int], ...]
    semantic_sha256: str

    def positions(self, camera: Camera) -> tuple[int, ...]:
        return self.scene_positions if camera is Camera.SCENE else self.wrist_positions

    def patch_for_position(self, position: int) -> int:
        mapping = dict(self.patch_by_position)
        if position not in mapping:
            raise B2ValidationError("position is not a mapped visual token")
        return mapping[position]


def derive_sequence_map(
    roles: Sequence[str],
    *,
    patch_ids: Mapping[int, int],
    expected_visual_tokens_per_camera: int | None = None,
) -> SequenceMap:
    """Derive every span from runtime roles and an explicit patch-position map."""

    frozen_roles = tuple(str(role) for role in roles)
    if not frozen_roles:
        raise B2ValidationError("runtime sequence is empty")
    unknown = set(frozen_roles) - VISUAL_ROLES - NONVISUAL_ROLES
    if unknown:
        raise B2ValidationError(f"unknown runtime token roles: {sorted(unknown)}")

    scene = tuple(index for index, role in enumerate(frozen_roles) if role == Camera.SCENE.value)
    wrist = tuple(index for index, role in enumerate(frozen_roles) if role == Camera.WRIST.value)
    text = tuple(index for index, role in enumerate(frozen_roles) if role == "text")
    proprio = tuple(index for index, role in enumerate(frozen_roles) if role == "proprio")
    action = tuple(index for index, role in enumerate(frozen_roles) if role == "action")
    nonvisual = tuple(index for index, role in enumerate(frozen_roles) if role not in VISUAL_ROLES)
    visual = scene + wrist

    if not scene or not wrist or not text or not proprio or not action:
        raise B2ValidationError("runtime sequence lacks a required multimodal role")
    if set(patch_ids) != set(visual):
        raise B2ValidationError("patch mapping must cover exactly the runtime visual positions")
    if set(scene) & set(wrist):
        raise B2ValidationError("scene and wrist positions alias")
    for positions in (scene, wrist):
        patches = sorted(int(patch_ids[position]) for position in positions)
        if patches != list(range(len(positions))):
            raise B2ValidationError("camera patch identifiers must be unique and contiguous")
    if expected_visual_tokens_per_camera is not None and (
        len(scene) != expected_visual_tokens_per_camera
        or len(wrist) != expected_visual_tokens_per_camera
    ):
        raise B2ValidationError("runtime visual-token count differs from the pinned expectation")

    payload = {
        "roles": list(frozen_roles),
        "patch_by_position": sorted((int(key), int(value)) for key, value in patch_ids.items()),
    }
    return SequenceMap(
        length=len(frozen_roles),
        roles=frozen_roles,
        scene_positions=scene,
        wrist_positions=wrist,
        text_positions=text,
        proprio_positions=proprio,
        action_positions=action,
        nonvisual_positions=nonvisual,
        patch_by_position=tuple(payload["patch_by_position"]),
        semantic_sha256=semantic_sha256(payload),
    )
