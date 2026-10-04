"""CPU-trainable PyTorch backend for the frozen PAIR router architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as functional

from savr.pair.features import GLOBAL_CONTINUOUS_NAMES, GROUP_CONTINUOUS_NAMES
from savr.pair.types import PairValidationError


LOSS_WEIGHTS = {
    "signed_contract_huber": 1.0,
    "signed_group_huber": 0.5,
    "positive_q50_pinball": 0.25,
    "positive_q90_pinball": 0.5,
    "dense_distortion_huber": 0.25,
}


@dataclass(frozen=True)
class TorchRouterOutput:
    signed_group_regret: torch.Tensor
    signed_contract_regret: torch.Tensor
    positive_regret_q50: torch.Tensor
    positive_regret_q90: torch.Tensor
    dense_action_distortion: torch.Tensor


class TorchPairRouter(nn.Module):
    """Differentiable mirror of the 64/32 group and 64/32 set router."""

    def __init__(self, profiles: Sequence[str], *, seed: int = 17) -> None:
        super().__init__()
        self.profiles = tuple(str(item) for item in profiles)
        if seed not in (17, 29, 43) or not self.profiles or len(self.profiles) > 6:
            raise PairValidationError("torch router identity is outside the frozen design")
        torch.manual_seed(seed)
        self.camera_embedding = nn.Embedding(2, 2)
        self.tile_embedding = nn.Embedding(16, 4)
        self.layer_embedding = nn.Embedding(4, 3)
        self.profile_embedding = nn.Embedding(len(self.profiles), 4)
        self.horizon_embedding = nn.Embedding(3, 2)
        self.group_encoder = nn.Sequential(
            nn.Linear(len(GROUP_CONTINUOUS_NAMES) + 15, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        self.group_head = nn.Linear(32, 1)
        self.set_encoder = nn.Sequential(
            nn.Linear(32 * 3 + len(GLOBAL_CONTINUOUS_NAMES), 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        self.set_head = nn.Linear(32, 4)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        group_continuous: torch.Tensor,
        camera: torch.Tensor,
        tile: torch.Tensor,
        layer: torch.Tensor,
        profile: torch.Tensor,
        horizon: torch.Tensor,
        global_continuous: torch.Tensor,
    ) -> TorchRouterOutput:
        if group_continuous.ndim != 2 or group_continuous.shape[1] != len(GROUP_CONTINUOUS_NAMES):
            raise PairValidationError("torch router group tensor shape is invalid")
        count = group_continuous.shape[0]
        categories = (camera, tile, layer, profile, horizon)
        if any(value.shape != (count,) or value.dtype != torch.long for value in categories):
            raise PairValidationError("torch router categorical tensors are invalid")
        if global_continuous.shape != (len(GLOBAL_CONTINUOUS_NAMES),):
            raise PairValidationError("torch router global tensor shape is invalid")
        if (
            not torch.isfinite(group_continuous).all()
            or not torch.isfinite(global_continuous).all()
        ):
            raise PairValidationError("torch router input contains nonfinite values")
        try:
            inputs = torch.cat(
                (
                    group_continuous,
                    self.camera_embedding(camera),
                    self.tile_embedding(tile),
                    self.layer_embedding(layer),
                    self.profile_embedding(profile),
                    self.horizon_embedding(horizon),
                ),
                dim=1,
            )
        except (IndexError, RuntimeError) as error:
            raise PairValidationError("torch router category is unsupported") from error
        embedding = self.group_encoder(inputs)
        group_regret = self.group_head(embedding).squeeze(-1)
        aggregate = torch.cat(
            (embedding.sum(dim=0), embedding.mean(dim=0), embedding.max(dim=0).values)
        )
        contract_embedding = self.set_encoder(torch.cat((aggregate, global_continuous)))
        raw = self.set_head(contract_embedding)
        q50 = functional.softplus(raw[1])
        q90 = q50 + functional.softplus(raw[2])
        return TorchRouterOutput(
            group_regret,
            raw[0],
            q50,
            q90,
            functional.softplus(raw[3]),
        )


def pinball_loss(prediction: torch.Tensor, target: torch.Tensor, quantile: float) -> torch.Tensor:
    if not 0 < quantile < 1:
        raise PairValidationError("pinball quantile is invalid")
    residual = target - prediction
    return torch.maximum(quantile * residual, (quantile - 1) * residual).mean()


def training_loss(
    output: TorchRouterOutput,
    *,
    signed_group_target: torch.Tensor,
    signed_contract_target: torch.Tensor,
    positive_regret_target: torch.Tensor,
    dense_distortion_target: torch.Tensor,
    weights: Mapping[str, float] = LOSS_WEIGHTS,
) -> torch.Tensor:
    if dict(weights) != LOSS_WEIGHTS:
        raise PairValidationError("router loss weights differ from the P0 freeze")
    loss = functional.huber_loss(output.signed_contract_regret, signed_contract_target, delta=1.0)
    loss = loss + 0.5 * functional.huber_loss(
        output.signed_group_regret, signed_group_target, delta=1.0
    )
    loss = loss + 0.25 * pinball_loss(output.positive_regret_q50, positive_regret_target, 0.5)
    loss = loss + 0.5 * pinball_loss(output.positive_regret_q90, positive_regret_target, 0.9)
    loss = loss + 0.25 * functional.huber_loss(
        output.dense_action_distortion, dense_distortion_target, delta=1.0
    )
    if not torch.isfinite(loss):
        raise PairValidationError("router training loss is nonfinite")
    return loss


def frozen_adamw(model: TorchPairRouter) -> torch.optim.AdamW:
    if model.parameter_count >= 250_000:
        raise PairValidationError("router exceeds the frozen parameter cap")
    return torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
