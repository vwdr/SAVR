from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from savr.pair.router_torch import (  # noqa: E402
    TorchPairRouter,
    frozen_adamw,
    pinball_loss,
    training_loss,
)


def synthetic_inputs():
    return (
        torch.zeros(3, 20),
        torch.tensor([0, 0, 1], dtype=torch.long),
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.zeros(3, dtype=torch.long),
        torch.ones(3, dtype=torch.long),
        torch.zeros(16),
    )


def test_torch_router_cpu_training_path_matches_frozen_optimizer_and_losses():
    assert not torch.cuda.is_initialized()
    first = TorchPairRouter(("D37_BAL",), seed=17)
    second = TorchPairRouter(("D37_BAL",), seed=17)
    assert first.parameter_count < 250_000
    output = first(*synthetic_inputs())
    repeated = second(*synthetic_inputs())
    assert torch.equal(output.signed_group_regret, repeated.signed_group_regret)
    assert torch.equal(output.positive_regret_q90, repeated.positive_regret_q90)
    loss = training_loss(
        output,
        signed_group_target=torch.tensor([0.0, 0.1, -0.1]),
        signed_contract_target=torch.tensor(0.05),
        positive_regret_target=torch.tensor(0.05),
        dense_distortion_target=torch.tensor(0.1),
    )
    optimizer = frozen_adamw(first)
    assert optimizer.defaults["lr"] == 0.001
    assert optimizer.defaults["weight_decay"] == 0.0001
    optimizer.zero_grad()
    loss.backward()
    assert all(parameter.grad is not None for parameter in first.parameters())
    torch.nn.utils.clip_grad_norm_(first.parameters(), 1.0)
    optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.cuda.is_initialized()


def test_pinball_known_answers_and_unsupported_quantile():
    prediction = torch.tensor([0.0, 2.0])
    target = torch.tensor([1.0, 1.0])
    assert pinball_loss(prediction, target, 0.5).item() == pytest.approx(0.5)
    with pytest.raises(Exception, match="quantile"):
        pinball_loss(prediction, target, 1.0)
