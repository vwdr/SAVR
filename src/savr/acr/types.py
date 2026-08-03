"""Frozen types shared by the ACR controller, cache, adapter, and records."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum


REFRESH_REASON_ORDER = (
    "cache",
    "invalid-signal",
    "warm-up",
    "scene-change",
    "translation",
    "gripper-transition",
    "horizon",
    "hard-cap",
    "policy",
)


class ACRPolicy(str, Enum):
    FACTORIZED_FR = "factorized-fr"
    SA_ACR = "sa-acr"
    SCENE_VISUAL = "scene-visual-acr"
    SCENE_PERIODIC = "scene-periodic-acr"


@dataclass(frozen=True)
class ACRContext:
    """Every identity that can change scene-token compatibility."""

    episode_id: str
    attempt_id: str
    task_id: str
    instruction_sha256: str
    checkpoint_id: str
    upstream_revision: str
    configuration_id: str
    controller_version: str
    preprocessing_id: str
    action_head_id: str
    dtype: str
    device: str
    patch_count: int = 256
    image_order: tuple[str, str] = ("full_image", "wrist_image")
    number_of_images: int = 2
    center_crop: bool = True

    def __post_init__(self) -> None:
        text_fields = (
            self.episode_id,
            self.attempt_id,
            self.task_id,
            self.instruction_sha256,
            self.checkpoint_id,
            self.upstream_revision,
            self.configuration_id,
            self.controller_version,
            self.preprocessing_id,
            self.action_head_id,
            self.dtype,
            self.device,
        )
        if not all(isinstance(value, str) and value for value in text_fields):
            raise ValueError("Every ACR context identity must be a non-empty string")
        if len(self.instruction_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.instruction_sha256
        ):
            raise ValueError("Instruction identity must be a lowercase SHA-256")
        if self.patch_count < 1:
            raise ValueError("Patch count must be positive")
        if self.image_order != ("full_image", "wrist_image"):
            raise ValueError("ACR Version 1 requires scene-first, wrist-second image order")
        if self.number_of_images != 2:
            raise ValueError("ACR Version 1 requires exactly two images")
        if not self.center_crop:
            raise ValueError("ACR Version 1 requires center crop")


@dataclass(frozen=True)
class ACRConfiguration:
    configuration_id: str
    policy: ACRPolicy
    scene_threshold: float | None = None
    translation_threshold: float | None = None
    horizon: int | None = None
    hard_reuse_cap: float | None = None
    period: int | None = None
    minimum_query_index: int = 2
    controller_version: str = "acr-controller-v1"

    def __post_init__(self) -> None:
        if not self.configuration_id or not self.controller_version:
            raise ValueError("Configuration and controller identities are required")
        for value in (self.scene_threshold, self.translation_threshold):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("ACR thresholds must be finite and non-negative")
        if self.minimum_query_index != 2:
            raise ValueError("ACR Version 1 fixes the first reuse query at index 2")
        if self.policy is ACRPolicy.FACTORIZED_FR:
            if any(
                value is not None
                for value in (
                    self.scene_threshold,
                    self.translation_threshold,
                    self.horizon,
                    self.hard_reuse_cap,
                    self.period,
                )
            ):
                raise ValueError("Factorized FR does not accept refresh thresholds")
            return
        if self.policy is ACRPolicy.SCENE_PERIODIC:
            if self.period is None or self.period < 1:
                raise ValueError("Scene-periodic ACR requires period >= 1")
            if any(
                value is not None
                for value in (
                    self.scene_threshold,
                    self.translation_threshold,
                    self.horizon,
                    self.hard_reuse_cap,
                )
            ):
                raise ValueError("Scene-periodic ACR accepts only a period")
            return
        if self.scene_threshold is None:
            raise ValueError("Scene-gated ACR requires a scene threshold")
        if self.horizon is None or self.horizon < 1:
            raise ValueError("Scene-gated ACR requires a positive horizon")
        if self.hard_reuse_cap is None or not 0 < self.hard_reuse_cap < 1:
            raise ValueError("Scene-gated ACR requires a hard reuse cap in (0,1)")
        if self.period is not None:
            raise ValueError("Scene-gated ACR does not accept a period")
        if self.policy is ACRPolicy.SA_ACR:
            if self.translation_threshold is None:
                raise ValueError("SA-ACR requires a translation threshold")
        elif self.policy is ACRPolicy.SCENE_VISUAL:
            if self.translation_threshold is not None:
                raise ValueError("Scene-visual ACR omits the translation threshold")
        else:  # pragma: no cover - enum exhaustiveness guard
            raise ValueError(f"Unsupported ACR policy: {self.policy}")


@dataclass(frozen=True)
class SceneTensorMetadata:
    shape: tuple[int, ...]
    dtype: str
    device: str
    patch_count: int

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(value < 1 for value in self.shape):
            raise ValueError("Scene tokens require a positive rank-three shape")
        if self.shape[0] != 1 or self.shape[1] != self.patch_count:
            raise ValueError("Scene tokens require batch one and the pinned patch count")
        if self.patch_count < 1 or not self.dtype or not self.device:
            raise ValueError("Scene tensor metadata is incomplete")

    @classmethod
    def from_value(cls, value: object, *, patch_count: int) -> "SceneTensorMetadata":
        try:
            shape = tuple(int(item) for item in getattr(value, "shape"))
        except Exception as error:
            raise ValueError("Scene token tensor lacks valid shape metadata") from error
        return cls(
            shape=shape,
            dtype=str(getattr(value, "dtype", "")),
            device=str(getattr(value, "device", "")),
            patch_count=patch_count,
        )


@dataclass(frozen=True)
class SceneDecision:
    policy: ACRPolicy
    query_index: int
    refresh: bool
    reasons: tuple[str, ...]
    cache_age_before: int | None
    scene_score: float | None
    translation_score: float | None
    gripper_transition_veto: bool | None
    translation_direction_reversals: tuple[bool, bool, bool] = (False, False, False)
    reference_query_index: int | None = None
    completed_queries_before: int = 0
    completed_reuses_before: int = 0
    patch_scores: tuple[float, ...] = ()

    def force_refresh(self, reason: str) -> "SceneDecision":
        reason_set = {*self.reasons, reason}
        unknown = reason_set - set(REFRESH_REASON_ORDER)
        if unknown:
            raise ValueError(f"Unsupported ACR refresh reasons: {sorted(unknown)}")
        reasons = tuple(item for item in REFRESH_REASON_ORDER if item in reason_set)
        return replace(self, refresh=True, reasons=reasons)


@dataclass(frozen=True)
class CameraWork:
    scene_siglip_calls: int = 0
    scene_dinov2_calls: int = 0
    scene_projector_calls: int = 0
    wrist_siglip_calls: int = 0
    wrist_dinov2_calls: int = 0
    wrist_projector_calls: int = 0
    downstream_calls: int = 0
    component_wall_ms: dict[str, float] = field(default_factory=dict)

    def validate(self, *, scene_refresh: bool, query_completed: bool = True) -> None:
        expected_scene = 1 if scene_refresh else 0
        if (
            self.scene_siglip_calls,
            self.scene_dinov2_calls,
            self.scene_projector_calls,
        ) != (expected_scene, expected_scene, expected_scene):
            raise ValueError("Scene component counts do not match the decision")
        if (
            self.wrist_siglip_calls,
            self.wrist_dinov2_calls,
            self.wrist_projector_calls,
        ) != (1, 1, 1):
            raise ValueError("Every query must execute exactly one fresh wrist path")
        if self.downstream_calls != int(query_completed):
            raise ValueError("Downstream component count does not match query completion")
        if any(value < 0 or not math.isfinite(value) for value in self.component_wall_ms.values()):
            raise ValueError("Component wall times must be finite and non-negative")
