"""Deterministic compact DeepSets-style PAIR risk router."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from savr.pair.features import (
    GLOBAL_CONTINUOUS_NAMES,
    GROUP_CONTINUOUS_NAMES,
    RouterFeatures,
    feature_schema_sha256,
)
from savr.pair.types import Camera, HORIZONS, ONSET_LAYERS, PairValidationError


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _gelu(value: np.ndarray) -> np.ndarray:
    return 0.5 * value * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (value + 0.044715 * value**3)))


def _softplus(value: float) -> float:
    return float(np.log1p(np.exp(-abs(value))) + max(value, 0.0))


@dataclass(frozen=True)
class RouterOutput:
    signed_group_regret: tuple[float, ...]
    signed_contract_regret: float
    positive_regret_q50: float
    positive_regret_q90: float
    dense_action_distortion: float


class PairRouter:
    """NumPy reference router used for CPU correctness and serialized inference."""

    EMBEDDING_WIDTHS = {"camera": 2, "tile": 4, "layer": 3, "profile": 4, "horizon": 2}

    def __init__(
        self,
        *,
        profiles: Sequence[str],
        seed: int,
        group_mean: np.ndarray,
        group_scale: np.ndarray,
        global_mean: np.ndarray,
        global_scale: np.ndarray,
        weights: Mapping[str, np.ndarray],
    ) -> None:
        self.profiles = tuple(str(item) for item in profiles)
        self.seed = int(seed)
        self.group_mean = np.asarray(group_mean, dtype=np.float64)
        self.group_scale = np.asarray(group_scale, dtype=np.float64)
        self.global_mean = np.asarray(global_mean, dtype=np.float64)
        self.global_scale = np.asarray(global_scale, dtype=np.float64)
        self.weights = {
            name: np.asarray(value, dtype=np.float64) for name, value in weights.items()
        }
        self.validate()

    @classmethod
    def initialize(cls, *, profiles: Sequence[str], seed: int = 17) -> "PairRouter":
        if seed not in (17, 29, 43):
            raise PairValidationError("router seed is not in the frozen set")
        profiles = tuple(str(item) for item in profiles)
        if not profiles or len(profiles) > 6 or len(set(profiles)) != len(profiles):
            raise PairValidationError("router profile categories are invalid")
        rng = np.random.default_rng(seed)

        def matrix(rows: int, columns: int) -> np.ndarray:
            limit = np.sqrt(6.0 / (rows + columns))
            return rng.uniform(-limit, limit, size=(rows, columns))

        group_input = len(GROUP_CONTINUOUS_NAMES) + sum(cls.EMBEDDING_WIDTHS.values())
        set_input = 32 * 3 + len(GLOBAL_CONTINUOUS_NAMES)
        weights = {
            "camera_embedding": matrix(len(Camera), cls.EMBEDDING_WIDTHS["camera"]),
            "tile_embedding": matrix(16, cls.EMBEDDING_WIDTHS["tile"]),
            "layer_embedding": matrix(len(ONSET_LAYERS), cls.EMBEDDING_WIDTHS["layer"]),
            "profile_embedding": matrix(len(profiles), cls.EMBEDDING_WIDTHS["profile"]),
            "horizon_embedding": matrix(len(HORIZONS), cls.EMBEDDING_WIDTHS["horizon"]),
            "group_w1": matrix(group_input, 64),
            "group_b1": np.zeros(64),
            "group_w2": matrix(64, 32),
            "group_b2": np.zeros(32),
            "group_head_w": matrix(32, 1),
            "group_head_b": np.zeros(1),
            "set_w1": matrix(set_input, 64),
            "set_b1": np.zeros(64),
            "set_w2": matrix(64, 32),
            "set_b2": np.zeros(32),
            "set_head_w": matrix(32, 4),
            "set_head_b": np.zeros(4),
        }
        return cls(
            profiles=profiles,
            seed=seed,
            group_mean=np.zeros(len(GROUP_CONTINUOUS_NAMES)),
            group_scale=np.ones(len(GROUP_CONTINUOUS_NAMES)),
            global_mean=np.zeros(len(GLOBAL_CONTINUOUS_NAMES)),
            global_scale=np.ones(len(GLOBAL_CONTINUOUS_NAMES)),
            weights=weights,
        )

    @property
    def parameter_count(self) -> int:
        return sum(int(value.size) for value in self.weights.values())

    def validate(self) -> None:
        if (
            not self.profiles
            or len(self.profiles) > 6
            or len(set(self.profiles)) != len(self.profiles)
        ):
            raise PairValidationError("router profile categories are invalid")
        expected_vectors = (
            (self.group_mean, len(GROUP_CONTINUOUS_NAMES), "group mean"),
            (self.group_scale, len(GROUP_CONTINUOUS_NAMES), "group scale"),
            (self.global_mean, len(GLOBAL_CONTINUOUS_NAMES), "global mean"),
            (self.global_scale, len(GLOBAL_CONTINUOUS_NAMES), "global scale"),
        )
        for value, width, name in expected_vectors:
            if value.shape != (width,) or not np.isfinite(value).all():
                raise PairValidationError(f"router {name} is invalid")
        if np.any(self.group_scale <= 0) or np.any(self.global_scale <= 0):
            raise PairValidationError("router standardization scale must be positive")
        expected_shapes = {
            "camera_embedding": (2, 2),
            "tile_embedding": (16, 4),
            "layer_embedding": (4, 3),
            "profile_embedding": (len(self.profiles), 4),
            "horizon_embedding": (3, 2),
            "group_w1": (35, 64),
            "group_b1": (64,),
            "group_w2": (64, 32),
            "group_b2": (32,),
            "group_head_w": (32, 1),
            "group_head_b": (1,),
            "set_w1": (112, 64),
            "set_b1": (64,),
            "set_w2": (64, 32),
            "set_b2": (32,),
            "set_head_w": (32, 4),
            "set_head_b": (4,),
        }
        if set(self.weights) != set(expected_shapes):
            raise PairValidationError("router checkpoint has missing or unexpected tensors")
        for name, shape in expected_shapes.items():
            if self.weights[name].shape != shape or not np.isfinite(self.weights[name]).all():
                raise PairValidationError(f"router tensor {name} is invalid")
        if self.parameter_count >= 250_000:
            raise PairValidationError("router exceeds the frozen parameter cap")

    def _group_rows(self, features: RouterFeatures) -> np.ndarray:
        rows = []
        for item in features.groups:
            continuous = (np.asarray(item.continuous) - self.group_mean) / self.group_scale
            category_rows = (
                self.weights["camera_embedding"][
                    [Camera.PRIMARY, Camera.WRIST].index(item.group.camera)
                ],
                self.weights["tile_embedding"][item.group.tile],
                self.weights["layer_embedding"][ONSET_LAYERS.index(item.group.onset_layer)],
                self.weights["profile_embedding"][self.profiles.index(item.profile_id)],
                self.weights["horizon_embedding"][HORIZONS.index(item.horizon)],
            )
            rows.append(np.concatenate((continuous, *category_rows)))
        return np.stack(rows)

    def predict(self, features: RouterFeatures) -> RouterOutput:
        features.validate(supported_profiles=self.profiles)
        profiles = {item.profile_id for item in features.groups}
        horizons = {item.horizon for item in features.groups}
        if len(profiles) != 1 or len(horizons) != 1:
            raise PairValidationError(
                "one router contract cannot mix profile or horizon categories"
            )
        group_input = self._group_rows(features)
        group_hidden = _gelu(group_input @ self.weights["group_w1"] + self.weights["group_b1"])
        group_embedding = _gelu(group_hidden @ self.weights["group_w2"] + self.weights["group_b2"])
        group_regret = (
            group_embedding @ self.weights["group_head_w"] + self.weights["group_head_b"]
        ).reshape(-1)
        aggregate = np.concatenate(
            (group_embedding.sum(axis=0), group_embedding.mean(axis=0), group_embedding.max(axis=0))
        )
        global_values = (
            np.asarray(features.global_continuous) - self.global_mean
        ) / self.global_scale
        set_input = np.concatenate((aggregate, global_values))
        set_hidden = _gelu(set_input @ self.weights["set_w1"] + self.weights["set_b1"])
        set_embedding = _gelu(set_hidden @ self.weights["set_w2"] + self.weights["set_b2"])
        raw = set_embedding @ self.weights["set_head_w"] + self.weights["set_head_b"]
        q50 = _softplus(float(raw[1]))
        q90 = q50 + _softplus(float(raw[2]))
        output = RouterOutput(
            signed_group_regret=tuple(float(item) for item in group_regret),
            signed_contract_regret=float(raw[0]),
            positive_regret_q50=q50,
            positive_regret_q90=q90,
            dense_action_distortion=_softplus(float(raw[3])),
        )
        if not np.isfinite(
            np.asarray(
                (
                    *output.signed_group_regret,
                    output.signed_contract_regret,
                    q50,
                    q90,
                    output.dense_action_distortion,
                )
            )
        ).all():
            raise PairValidationError("router emitted nonfinite risk")
        return output

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": "pair-router-checkpoint-v1",
            "feature_schema_sha256": feature_schema_sha256(),
            "architecture": "group-64-32/deepsets-64-32/gelu/no-dropout",
            "seed": self.seed,
            "profiles": self.profiles,
            "group_mean": self.group_mean.tolist(),
            "group_scale": self.group_scale.tolist(),
            "global_mean": self.global_mean.tolist(),
            "global_scale": self.global_scale.tolist(),
            "weights": {name: value.tolist() for name, value in sorted(self.weights.items())},
            "parameter_count": self.parameter_count,
        }

    def serialize(self) -> bytes:
        payload = self._payload()
        payload["semantic_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
        return _canonical_bytes(payload) + b"\n"

    def save(self, path: Path) -> str:
        data = self.serialize()
        path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    @classmethod
    def deserialize(cls, data: bytes) -> "PairRouter":
        try:
            payload = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise PairValidationError("router checkpoint is malformed") from error
        supplied = payload.pop("semantic_sha256", None)
        if supplied != hashlib.sha256(_canonical_bytes(payload)).hexdigest():
            raise PairValidationError("router checkpoint semantic hash mismatch")
        if payload.get("schema_version") != "pair-router-checkpoint-v1":
            raise PairValidationError("router checkpoint schema is unsupported")
        if payload.get("feature_schema_sha256") != feature_schema_sha256():
            raise PairValidationError("router feature schema changed")
        result = cls(
            profiles=payload["profiles"],
            seed=int(payload["seed"]),
            group_mean=payload["group_mean"],
            group_scale=payload["group_scale"],
            global_mean=payload["global_mean"],
            global_scale=payload["global_scale"],
            weights=payload["weights"],
        )
        if int(payload.get("parameter_count", -1)) != result.parameter_count:
            raise PairValidationError("router parameter accounting changed")
        return result

    @classmethod
    def load(cls, path: Path) -> "PairRouter":
        return cls.deserialize(path.read_bytes())
