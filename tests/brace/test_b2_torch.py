from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import DynamicCache  # noqa: E402

from savr.brace.cache_adapter import (  # noqa: E402
    clone_dynamic_cache,
    position_preserving_index_update,
    transactional_cache_configuration,
)


def populated_cache():
    cache = DynamicCache()
    keys = torch.arange(24, dtype=torch.float32).reshape(1, 2, 4, 3)
    values = keys + 100
    cache.update(keys, values, 0, {"cache_position": torch.arange(4)})
    return cache


def test_real_dynamic_cache_clone_has_no_shared_tensor_storage():
    cache = populated_cache()
    cloned = clone_dynamic_cache(cache)
    original = cache.key_cache[0].clone()
    cloned.key_cache[0].add_(5)
    assert torch.equal(cache.key_cache[0], original)
    assert not torch.equal(cache.key_cache[0], cloned.key_cache[0])


def test_real_dynamic_cache_transaction_restores_configuration_and_cache_on_error():
    cache = populated_cache()
    before = cache.key_cache[0].clone()
    config = SimpleNamespace(cache_mode="dense")
    with pytest.raises(RuntimeError):
        with transactional_cache_configuration(
            cache, config, {"cache_mode": "reuse", "reuse_budget": 2}
        ) as arm:
            arm.key_cache[0].zero_()
            cache.key_cache[0].add_(1)
            raise RuntimeError("synthetic arm failure")
    assert torch.equal(cache.key_cache[0], before)
    assert config.cache_mode == "dense"
    assert not hasattr(config, "reuse_budget")


def test_real_tensor_index_copy_preserves_absolute_sequence_positions():
    cached = torch.arange(24, dtype=torch.float32).reshape(1, 2, 4, 3)
    current = torch.full((1, 2, 2, 3), 999.0)
    updated = position_preserving_index_update(cached, current, [1, 3])
    assert torch.equal(updated[:, :, 0], cached[:, :, 0])
    assert torch.equal(updated[:, :, 2], cached[:, :, 2])
    assert torch.equal(updated[:, :, 1], current[:, :, 0])
    assert torch.equal(updated[:, :, 3], current[:, :, 1])
    assert torch.equal(cached, torch.arange(24, dtype=torch.float32).reshape(1, 2, 4, 3))


def test_expected_transformers_stack_and_custom_position_update_are_present():
    assert transformers.__version__ in {"4.40.1", "4.47.0"}
    import inspect

    source = inspect.getsource(DynamicCache.update)
    assert "cache_position" in source
    assert "index_copy" in source
