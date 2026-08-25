"""Frozen identities and state types for BRACE correctness gates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class B2ValidationError(ValueError):
    """Raised when a B2 correctness or provenance invariant fails."""


class Camera(str, Enum):
    SCENE = "scene"
    WRIST = "wrist"


class ProfileFamily(str, Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class ContractMode(str, Enum):
    ANCHOR = "anchor"
    CONTRACT = "contract"
    EXPERIMENT_FR_LOCK = "experiment_fr_lock"


@dataclass(frozen=True)
class CacheIdentity:
    model_sha256: str
    checkpoint_sha256: str
    sequence_map_sha256: str
    preprocessing_sha256: str
    episode_id: str
    anchor_query: int
    profile_id: str
    dtype: str
    device: str

    def validate(self) -> None:
        for name in (
            "model_sha256",
            "checkpoint_sha256",
            "sequence_map_sha256",
            "preprocessing_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64:
                raise B2ValidationError(f"{name} is not a SHA-256 identifier")
            try:
                int(value, 16)
            except ValueError as error:
                raise B2ValidationError(f"{name} is not hexadecimal") from error
        if not self.episode_id or self.anchor_query < 0 or not self.profile_id:
            raise B2ValidationError("cache identity is incomplete")
        if not self.dtype or not self.device:
            raise B2ValidationError("cache tensor identity is incomplete")


@dataclass(frozen=True)
class LayerBudget:
    layer: int
    scene: int
    wrist: int

    def validate(self) -> None:
        if self.layer < 0 or self.scene < 0 or self.wrist < 0:
            raise B2ValidationError("layer budgets must be nonnegative")


@dataclass(frozen=True)
class Profile:
    profile_id: str
    family: ProfileFamily
    budgets: tuple[LayerBudget, ...]
    scene_change_limit: float
    wrist_change_limit: float
    scene_max_age: int
    wrist_max_age: int
    protected_scene: tuple[int, ...] = ()
    protected_wrist: tuple[int, ...] = ()


@dataclass(frozen=True)
class Contract:
    profile_id: str
    family: ProfileFamily
    horizon: int

    def validate(self) -> None:
        if self.family not in (ProfileFamily.P1, ProfileFamily.P2):
            raise B2ValidationError("only P1/P2 profiles are BRACE-selectable")
        if self.horizon not in (1, 2, 4):
            raise B2ValidationError("contract horizon must be 1, 2, or 4")
        if not self.profile_id:
            raise B2ValidationError("contract profile identity is empty")


@dataclass(frozen=True)
class ContractState:
    mode: ContractMode
    anchor_query: int
    contract: Contract | None = None
    remaining: int = 0
    abort_reason: str | None = None
    experimental: bool = False

    @classmethod
    def anchor(cls, query: int = 0) -> "ContractState":
        if query < 0:
            raise B2ValidationError("anchor query must be nonnegative")
        return cls(mode=ContractMode.ANCHOR, anchor_query=query)

    def start(self, contract: Contract, *, experimental: bool = False) -> "ContractState":
        if self.mode is not ContractMode.ANCHOR:
            raise B2ValidationError("a contract can start only after a dense anchor")
        contract.validate()
        return ContractState(
            mode=ContractMode.CONTRACT,
            anchor_query=self.anchor_query,
            contract=contract,
            remaining=contract.horizon,
            experimental=experimental,
        )

    def advance(self, *, query: int, abort_reason: str | None = None) -> "ContractState":
        if query <= self.anchor_query:
            raise B2ValidationError("contract queries must follow their anchor")
        if self.mode is ContractMode.EXPERIMENT_FR_LOCK:
            remaining = self.remaining - 1
            if remaining <= 0:
                return ContractState.anchor(query)
            return ContractState(
                mode=self.mode,
                anchor_query=self.anchor_query,
                contract=self.contract,
                remaining=remaining,
                abort_reason=self.abort_reason,
                experimental=True,
            )
        if self.mode is not ContractMode.CONTRACT or self.contract is None:
            raise B2ValidationError("no active contract to advance")
        if abort_reason:
            if self.experimental and self.remaining > 1:
                return ContractState(
                    mode=ContractMode.EXPERIMENT_FR_LOCK,
                    anchor_query=self.anchor_query,
                    contract=self.contract,
                    remaining=self.remaining - 1,
                    abort_reason=abort_reason,
                    experimental=True,
                )
            return ContractState.anchor(query)
        remaining = self.remaining - 1
        if remaining <= 0:
            return ContractState.anchor(query)
        return ContractState(
            mode=ContractMode.CONTRACT,
            anchor_query=self.anchor_query,
            contract=self.contract,
            remaining=remaining,
            experimental=self.experimental,
        )


def frozen_value(value: Any) -> Any:
    """Recursively convert JSON-like values to immutable tuples."""

    if isinstance(value, dict):
        return tuple(sorted((str(key), frozen_value(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(frozen_value(item) for item in value)
    return value
