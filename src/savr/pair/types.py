"""Frozen identities and fail-closed types for PAIR-VLA."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from savr.brace.types import B2ValidationError


ONSET_LAYERS = (2, 6, 9, 11)
HORIZONS = (1, 2, 4)
TILES_PER_CAMERA = 16
PATCHES_PER_CAMERA = 256


class PairValidationError(B2ValidationError):
    """Raised whenever PAIR cannot prove that an operation is valid."""


class Camera(str, Enum):
    PRIMARY = "primary"
    WRIST = "wrist"


class Split(str, Enum):
    TRAIN = "train"
    CALIBRATION = "calibration"
    LOCKED_TEST = "locked_test"


@dataclass(frozen=True, order=True)
class AtomicGroup:
    camera: Camera
    tile: int
    onset_layer: int

    def validate(self) -> None:
        if not isinstance(self.camera, Camera):
            raise PairValidationError("unsupported camera category")
        if not 0 <= self.tile < TILES_PER_CAMERA:
            raise PairValidationError("tile lies outside the frozen 4x4 grid")
        if self.onset_layer not in ONSET_LAYERS:
            raise PairValidationError("unsupported onset layer")

    @property
    def identity(self) -> str:
        self.validate()
        return f"{self.camera.value}:t{self.tile:02d}:l{self.onset_layer:02d}"


@dataclass(frozen=True)
class ProfileSpec:
    profile_id: str
    primary_budgets: tuple[int, ...]
    wrist_budgets: tuple[int, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProfileSpec":
        result = cls(
            profile_id=str(value["id"]),
            primary_budgets=tuple(int(item) for item in value["primary_budgets"]),
            wrist_budgets=tuple(int(item) for item in value["wrist_budgets"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not self.profile_id:
            raise PairValidationError("profile identity is empty")
        for budgets in (self.primary_budgets, self.wrist_budgets):
            if len(budgets) != len(ONSET_LAYERS):
                raise PairValidationError("profile must cover every frozen onset layer")
            if budgets != tuple(sorted(budgets)):
                raise PairValidationError("profile budgets are not nested")
            if any(value < 0 or value > PATCHES_PER_CAMERA or value % 16 for value in budgets):
                raise PairValidationError("profile budgets must be 16-patch tile multiples")


@dataclass(frozen=True)
class QueryKey:
    suite: str
    task_id: str
    trajectory_id: str
    step: int
    split: Split

    def validate(self) -> None:
        if not all((self.suite, self.task_id, self.trajectory_id)) or self.step < 0:
            raise PairValidationError("query identity is incomplete")
        if not isinstance(self.split, Split):
            raise PairValidationError("unsupported split category")


@dataclass(frozen=True)
class SourcePointer:
    query: int
    age: int

    def validate(self, current_query: int, maximum_age: int = 4) -> None:
        if self.query < 0 or self.query > current_query:
            raise PairValidationError("source query is chronologically invalid")
        if self.age != current_query - self.query or not 0 <= self.age <= maximum_age:
            raise PairValidationError("source age is inconsistent or unsupported")
