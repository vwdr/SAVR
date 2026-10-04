from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from savr.cac.c1 import (
    ALIGNED_RECORD_BYTES,
    SIDECAR_RECORD_BYTES,
    BoundedRecordWriter,
    CACFeatureTensors,
    CACValidationError,
    action_head_penultimate,
    build_full_adapter,
    cache_digest,
    serialize_numeric_record,
    serialize_sidecar,
    transactional_profile,
)
from savr.pair.p3_openvla import PhysicalSourceTracker, PreparedQuery
from savr.pair.p3_openvla import prepare_query
from savr.pair.types import PairValidationError
from savr.brace.cache_adapter import clone_dynamic_cache, transactional_cache_configuration


def _features() -> CACFeatureTensors:
    return CACFeatureTensors(
        current_tiles=torch.zeros(32, 4096, dtype=torch.float16),
        source_deltas=torch.zeros(128, 4096, dtype=torch.float16),
        z_cache=torch.zeros(8, 4096, dtype=torch.float16),
        base_action=torch.zeros(8, 7, dtype=torch.float32),
        proprio=torch.zeros(8, dtype=torch.float32),
        instruction_embedding=torch.zeros(4096, dtype=torch.float16),
        provenance=torch.zeros(128, 12, dtype=torch.uint8),
        source_query_ids=tuple(range(128)),
    )


def _prepared(value: float = 0.0) -> PreparedQuery:
    return PreparedQuery(
        input_embeddings=torch.full((1, 3, 4), value),
        action_mask=torch.zeros((1, 3), dtype=torch.bool),
        projected_patches=torch.full((1, 513, 4), value),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
        normalized_proprio=torch.zeros((1, 8)),
        preprocessed_pixels=torch.zeros((1, 6, 224, 224)),
    )


def test_action_head_penultimate_is_exact() -> None:
    class Block(torch.nn.Module):
        def forward(self, value):
            return value + 0.01

    model = SimpleNamespace(
        layer_norm1=torch.nn.LayerNorm(7 * 4096),
        fc1=torch.nn.Linear(7 * 4096, 4096),
        relu=torch.nn.ReLU(),
        mlp_resnet_blocks=torch.nn.ModuleList([Block(), Block()]),
        layer_norm2=torch.nn.LayerNorm(4096),
        fc2=torch.nn.Linear(4096, 7),
    )
    head = SimpleNamespace(model=model)
    hidden = torch.randn(1, 56, 4096)
    z_cache, action = action_head_penultimate(head, hidden)
    expected = model.fc2(z_cache)
    assert torch.equal(action, expected)
    assert z_cache.shape == (1, 8, 4096)


def test_full_adapter_count_and_zero_initialized_bypass() -> None:
    adapter = build_full_adapter(torch)
    assert sum(parameter.numel() for parameter in adapter.parameters()) == 5_070_599
    output = adapter(_features(), torch.ones(8, 7))
    assert output.shape == (8, 7)
    assert torch.equal(output, torch.zeros_like(output))


def test_tracker_clone_isolation_and_recursive_restart() -> None:
    from savr.pair.types import Camera

    original = PhysicalSourceTracker(0, _prepared())
    duplicate = original.clone()
    duplicate.add_record(1, _prepared(1))
    duplicate.advance(1, {(Camera.PRIMARY, 0): 2})
    assert original.last_query == 0
    assert original.digest() != duplicate.digest()
    restarted = PhysicalSourceTracker(0, _prepared())
    assert original.digest() == restarted.digest()


def test_transaction_restores_profile_after_exception() -> None:
    config = SimpleNamespace(proportion_attn_var="old-a", reusable_patches="old-b")
    model = SimpleNamespace(language_model=SimpleNamespace(config=config))
    with pytest.raises(RuntimeError):
        with transactional_profile(model):
            config.proportion_attn_var = "changed-a"
            config.reusable_patches = "changed-b"
            raise RuntimeError("injected")
    assert (config.proportion_attn_var, config.reusable_patches) == ("old-a", "old-b")


def test_fixed_records_hash_restart_and_exclusive_root(tmp_path: Path) -> None:
    features = _features()
    numeric = serialize_numeric_record(features, torch.zeros(8, 7), torch.ones(8, 7, dtype=torch.uint8))
    identity = hashlib.sha256(b"identity").hexdigest()
    sidecar = serialize_sidecar(
        {
            "contract_id": identity,
            "trajectory_id": identity,
            "anchor_query_id": identity,
            "endpoint_query_id": identity,
            "role": "adapter_fit",
            "suite": "libero_10",
            "horizon": 4,
            "record_index": 0,
        },
        hashlib.sha256(numeric).hexdigest(),
    )
    assert len(numeric) == ALIGNED_RECORD_BYTES
    assert len(sidecar) == SIDECAR_RECORD_BYTES
    root = tmp_path / "records"
    writer = BoundedRecordWriter(root, queue_size=1)
    writer.submit(numeric, sidecar)
    summary = writer.close()
    assert summary["records"] == 1
    assert summary["numeric_bytes"] == ALIGNED_RECORD_BYTES
    assert summary["sidecar_bytes"] == SIDECAR_RECORD_BYTES
    with pytest.raises(CACValidationError, match="already exists"):
        BoundedRecordWriter(root)


def test_query_preparation_fails_closed_if_gradients_are_enabled() -> None:
    with torch.enable_grad(), pytest.raises(PairValidationError, match="gradients disabled"):
        prepare_query(
            torch_module=torch, np=None, model=None, processor=None, proprio_projector=None,
            prepare_images=None, normalize_proprio=None, cfg=None, raw_scene=None,
            raw_wrist=None, raw_state=None, instruction="test",
        )


def test_writer_startup_failure_is_reported_without_deadlock(tmp_path: Path, monkeypatch) -> None:
    def fail_open(*_args, **_kwargs):
        raise OSError("injected writer failure")

    monkeypatch.setattr("savr.cac.c1.os.open", fail_open)
    writer = BoundedRecordWriter(tmp_path / "failed")
    with pytest.raises(CACValidationError, match="writer previously failed"):
        writer.submit(bytes(ALIGNED_RECORD_BYTES), bytes(SIDECAR_RECORD_BYTES))
    with pytest.raises(CACValidationError, match="asynchronous writer failed"):
        writer.close()


def test_cache_transaction_restores_after_injected_exception() -> None:
    cache = SimpleNamespace(
        key_cache=[torch.tensor([1.0])], value_cache=[torch.tensor([2.0])]
    )
    config = SimpleNamespace(proportion_attn_var="dense", reusable_patches=None)
    baseline = clone_dynamic_cache(cache)
    with pytest.raises(RuntimeError, match="injected"):
        with transactional_cache_configuration(
            cache, config, {"proportion_attn_var": "cached", "reusable_patches": [1]}
        ) as branch:
            branch.key_cache[0].add_(10)
            cache.value_cache[0].add_(10)
            raise RuntimeError("injected")
    assert torch.equal(cache.key_cache[0], baseline.key_cache[0])
    assert torch.equal(cache.value_cache[0], baseline.value_cache[0])
    assert config.proportion_attn_var == "dense" and config.reusable_patches is None


def test_cache_digest_is_bit_exact_for_bfloat16() -> None:
    cache = SimpleNamespace(
        key_cache=[torch.tensor([1.0, 2.0], dtype=torch.bfloat16)],
        value_cache=[torch.tensor([3.0, 4.0], dtype=torch.bfloat16)],
    )
    baseline = cache_digest(cache)
    assert baseline == cache_digest(clone_dynamic_cache(cache))
    cache.key_cache[0].view(torch.uint8)[0] ^= 1
    assert baseline != cache_digest(cache)
