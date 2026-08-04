from __future__ import annotations

import math
import threading
from dataclasses import replace

import pytest

from savr.acr.cache import SceneCacheEntry
from savr.acr.controller import ACRController
from savr.acr.dual_path import DualPathOpenVLAAdapter
from savr.acr.records import AttemptIdentity, ImmutableRecordStore, semantic_sha256
from savr.acr.types import ACRConfiguration, ACRContext, ACRPolicy


class FakeTensor:
    def __init__(self, values, shape, *, dtype="float32", device="cpu"):
        self.values = tuple(float(value) for value in values)
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.requires_grad = True
        if math.prod(self.shape) != len(self.values):
            raise ValueError("Fake tensor data/shape mismatch")

    def detach(self):
        return self

    def clone(self):
        copied = FakeTensor(self.values, self.shape, dtype=self.dtype, device=self.device)
        copied.requires_grad = self.requires_grad
        return copied

    def requires_grad_(self, value):
        self.requires_grad = value
        return self

    def tolist(self):
        return list(self.values)


class FakeOps:
    def __init__(self):
        self.finite_checks = []

    @staticmethod
    def split(value, sections, *, dim):
        if dim != 1 or sum(sections) != value.shape[1]:
            raise ValueError("Fake split supports only the tested dimension")
        outer = value.shape[0]
        inner = math.prod(value.shape[2:])
        outputs = []
        start = 0
        for section in sections:
            data = []
            for batch in range(outer):
                base = batch * value.shape[1] * inner
                data.extend(value.values[base + start * inner : base + (start + section) * inner])
            outputs.append(
                FakeTensor(
                    data,
                    (outer, section, *value.shape[2:]),
                    dtype=value.dtype,
                    device=value.device,
                )
            )
            start += section
        return tuple(outputs)

    @staticmethod
    def cat(values, *, dim):
        first = values[0]
        if dim == 2:
            batch, patches, _ = first.shape
            data = []
            for batch_index in range(batch):
                for patch in range(patches):
                    for value in values:
                        width = value.shape[2]
                        start = (batch_index * patches + patch) * width
                        data.extend(value.values[start : start + width])
            return FakeTensor(
                data,
                (batch, patches, sum(value.shape[2] for value in values)),
                dtype=first.dtype,
                device=first.device,
            )
        if dim == 1:
            return FakeTensor(
                tuple(item for value in values for item in value.values),
                (first.shape[0], sum(value.shape[1] for value in values), first.shape[2]),
                dtype=first.dtype,
                device=first.device,
            )
        raise ValueError("Unsupported fake concatenation")

    def all_finite(self, value):
        self.finite_checks.append(value)
        values = value.values if hasattr(value, "values") else value
        return all(math.isfinite(float(item)) for item in values)


class FakeBackbone:
    use_fused_vision_backbone = True

    def __init__(self):
        self.patch_count = 2
        self.number_of_images = 2

    def get_num_images_in_input(self):
        return self.number_of_images

    def get_num_patches(self):
        return self.patch_count

    @staticmethod
    def _features(value, offset):
        mean = sum(value.values) / len(value.values) + offset
        return FakeTensor((mean, mean), (1, 2, 1), dtype=value.dtype, device=value.device)

    def featurizer(self, value):
        return self._features(value, 0)

    def fused_featurizer(self, value):
        return self._features(value, 10)


class FakeModel:
    def __init__(self, ops):
        self.ops = ops
        self.vision_backbone = FakeBackbone()
        self.original_refresh_calls = 0
        self.combined_backbone_calls = 0
        self.combined_projector_calls = 0
        self.original_outputs = []
        self.last_tokens = None
        self.last_state = None

    @staticmethod
    def projector(value):
        projected = tuple(item for item in value.values for _ in range(2))
        return FakeTensor(
            projected,
            (1, value.shape[1], 4),
            dtype=value.dtype,
            device=value.device,
        )

    def _camera(self, pixels):
        regular, fused = self.ops.split(pixels, (3, 3), dim=1)
        return self.ops.cat(
            (
                self.vision_backbone.featurizer(regular),
                self.vision_backbone.fused_featurizer(fused),
            ),
            dim=2,
        )

    def _process_vision_features(self, pixel_values, language_embeddings=None, use_film=False):
        assert language_embeddings is not None and not use_film
        self.original_refresh_calls += 1
        self.combined_backbone_calls += 1
        scene, wrist = self.ops.split(pixel_values, (6, 6), dim=1)
        features = self.ops.cat((self._camera(scene), self._camera(wrist)), dim=1)
        self.combined_projector_calls += 1
        output = self.projector(features)
        self.original_outputs.append(output)
        return output

    def predict(self, pixel_values, language, state, *, actions=None, recurse=None):
        if recurse is not None:
            recurse()
        self.last_tokens = self._process_vision_features(pixel_values, language)
        self.last_state = tuple(state)
        return tuple(0.0 for _ in range(56)) if actions is None else actions


def configuration():
    return ACRConfiguration(
        "sa-dp-acr-t25-h2-b30-v01",
        ACRPolicy.SA_ACR,
        scene_threshold=0.2476380718954248,
        translation_threshold=0.5479944908411765,
        horizon=2,
        hard_reuse_cap=0.30,
    )


def context(**changes):
    value = ACRContext(
        episode_id="episode",
        attempt_id="attempt",
        task_id="task",
        instruction_sha256="0" * 64,
        checkpoint_id="checkpoint",
        upstream_revision="revision",
        configuration_id=configuration().configuration_id,
        controller_version="acr-controller-v1",
        preprocessing_id="preprocessing",
        action_head_id="head",
        dtype="float32",
        device="cpu",
        patch_count=2,
    )
    return replace(value, **changes)


def pixels(scene=1.0, wrist=2.0, *, dtype="float32", channels=12):
    scene_channels = min(channels, 6)
    wrist_channels = max(0, channels - scene_channels)
    values = (scene,) * (scene_channels * 4) + (wrist,) * (wrist_channels * 4)
    return FakeTensor(values, (1, channels, 2, 2), dtype=dtype)


def make_adapter(*, correctness_mode=False, observer=None, action_finite_checker=None):
    ops = FakeOps()
    model = FakeModel(ops)
    adapter = DualPathOpenVLAAdapter(
        model=model,
        controller=ACRController(configuration()),
        tensor_ops=ops,
        correctness_mode=correctness_mode,
        projected_tokens_observer=observer,
        action_finite_checker=action_finite_checker,
    )
    return model, adapter, ops


def run(
    model,
    adapter,
    *,
    scene=1.0,
    wrist=2.0,
    state=(0.0,) * 8,
    actions=None,
    pixel_values=None,
    language=None,
    recurse=None,
):
    values = pixels(scene, wrist) if pixel_values is None else pixel_values
    language = FakeTensor((0.0,) * 20, (1, 5, 4)) if language is None else language
    return adapter.run_query(
        query=lambda: model.predict(values, language, state, actions=actions, recurse=recurse),
        scene_image=[[scene] * 32 for _ in range(32)],
        wrist_image=[[wrist] * 32 for _ in range(32)],
        state=state,
        state_q01=(0.0,) * 8,
        state_q99=(1.0,) * 8,
    )


def warm_to_reuse(model, adapter):
    results = [run(model, adapter, wrist=float(index + 2)) for index in range(4)]
    assert [result.decision.refresh for result in results] == [True, True, True, False]
    return results


def test_refresh_calls_original_once_returns_exact_output_and_owns_scene_cache():
    observed = []
    model, adapter, _ = make_adapter(observer=observed.append)
    original_class_method = FakeModel._process_vision_features
    with adapter.episode(context()):
        result = run(model, adapter)
        assert result.value == (0.0,) * 56
        assert model.last_tokens is model.original_outputs[0]
        assert observed == [model.original_outputs[0]]
        assert model.original_refresh_calls == 1
        assert model.combined_backbone_calls == 1
        assert model.combined_projector_calls == 1
        assert adapter.cache.entry is not None
        cached = adapter.cache.entry.tokens
        assert cached is not model.last_tokens
        assert cached.values == model.last_tokens.values[:8]
        assert cached.requires_grad is False
        result.work.validate(scene_refresh=True)
    assert FakeModel._process_vision_features is original_class_method
    assert "_process_vision_features" not in vars(model)
    assert adapter.cache.entry is None


def test_reuse_uses_cached_scene_and_exactly_one_fresh_wrist_path():
    model, adapter, _ = make_adapter()
    with adapter.episode(context()):
        results = warm_to_reuse(model, adapter)
        reuse = results[-1]
        assert model.original_refresh_calls == 3
        assert reuse.cache_event == "reuse"
        reuse.work.validate(scene_refresh=False)
        assert reuse.work.physical_siglip_calls == 1
        assert reuse.work.physical_dinov2_calls == 1
        assert reuse.work.physical_projector_calls == 1
        assert reuse.work.logical_scene_backbone_calls == 0
        assert reuse.work.logical_wrist_backbone_calls == 1
        assert model.last_tokens.values[:8] == model.original_outputs[-1].values[:8]
        assert model.last_tokens.values[8:] == (5, 5, 15, 15, 5, 5, 15, 15)


def test_physical_and_logical_accounting_truth_table():
    model, adapter, _ = make_adapter()
    with adapter.episode(context()):
        refresh = run(model, adapter)
        for _ in range(2):
            run(model, adapter)
        reuse = run(model, adapter)
    assert (
        refresh.work.physical_fused_backbone_calls,
        refresh.work.physical_siglip_calls,
        refresh.work.logical_scene_backbone_calls,
        refresh.work.logical_wrist_backbone_calls,
    ) == (1, 0, 1, 1)
    assert (
        reuse.work.physical_fused_backbone_calls,
        reuse.work.physical_siglip_calls,
        reuse.work.logical_scene_backbone_calls,
        reuse.work.logical_wrist_backbone_calls,
    ) == (0, 1, 0, 1)


def test_episode_scope_restores_original_after_downstream_exception():
    model, adapter, _ = make_adapter()
    original = FakeModel._process_vision_features
    with pytest.raises(RuntimeError, match="downstream"):
        with adapter.episode(context()):
            run(model, adapter, recurse=lambda: (_ for _ in ()).throw(RuntimeError("downstream")))
    assert "_process_vision_features" not in vars(model)
    assert FakeModel._process_vision_features is original
    assert adapter.cache.entry is None
    assert adapter.controller.query_index == 0
    assert adapter.last_failure is not None and adapter.last_failure.cache_invalidated


def test_nested_episode_and_query_and_concurrent_query_are_rejected():
    model, adapter, _ = make_adapter()
    with adapter.episode(context()):
        with pytest.raises(RuntimeError, match="Nested or concurrent"):
            with adapter.episode(context()):
                pass
        with pytest.raises(RuntimeError, match="Nested dual-path query"):
            run(model, adapter, recurse=lambda: run(model, adapter))
        errors = []

        def other_thread():
            try:
                run(model, adapter)
            except RuntimeError as error:
                errors.append(str(error))

        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join()
        assert errors == ["Concurrent dual-path query use is prohibited"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"image_order": ("wrist_image", "full_image")}, "scene-first"),
        ({"number_of_images": 1}, "exactly two images"),
        ({"center_crop": False}, "center crop"),
    ],
)
def test_context_contract_rejection(change, message):
    with pytest.raises(ValueError, match=message):
        context(**change)


def test_shape_dtype_device_and_patch_count_fail_closed():
    model, adapter, _ = make_adapter()
    with adapter.episode(context()):
        with pytest.raises(RuntimeError, match="pixel tensor shape"):
            run(model, adapter, pixel_values=pixels(channels=11))
    model, adapter, _ = make_adapter()
    with adapter.episode(context()):
        wrong_language = FakeTensor((0.0,) * 20, (1, 5, 4), dtype="float16")
        with pytest.raises(RuntimeError, match="dtype/device"):
            run(model, adapter, language=wrong_language)
    model, adapter, _ = make_adapter()
    with adapter.episode(context()):
        wrong_device = FakeTensor((0.0,) * 20, (1, 5, 4), device="cuda:0")
        with pytest.raises(RuntimeError, match="dtype/device"):
            run(model, adapter, language=wrong_device)
    model, adapter, _ = make_adapter()
    model.vision_backbone.patch_count = 3
    with adapter.episode(context()):
        with pytest.raises(RuntimeError, match="patch count"):
            run(model, adapter)


def test_cache_metadata_mismatch_forces_safe_upstream_refresh():
    model, adapter, _ = make_adapter()
    with adapter.episode(context()):
        for _ in range(3):
            run(model, adapter)
        assert adapter.cache.entry is not None
        entry = adapter.cache.entry
        adapter.cache._entry = SceneCacheEntry(
            context=entry.context,
            tokens=entry.tokens,
            metadata=replace(entry.metadata, dtype="float16"),
            refresh_query_index=entry.refresh_query_index,
        )
        result = run(model, adapter)
        assert result.decision.refresh
        assert result.cache_event == "forced-refresh"
        assert result.work.upstream_two_view_refresh_calls == 1
        assert model.original_refresh_calls == 4


def test_production_avoids_projected_finite_scans_but_checks_action():
    model, adapter, ops = make_adapter(correctness_mode=False)
    with adapter.episode(context()):
        run(model, adapter)
    assert len(ops.finite_checks) == 1
    assert isinstance(ops.finite_checks[0], tuple)


def test_correctness_mode_fully_checks_projected_blocks_and_action():
    model, adapter, ops = make_adapter(correctness_mode=True)
    with adapter.episode(context()):
        run(model, adapter)
    assert len(ops.finite_checks) == 4
    assert sum(isinstance(value, FakeTensor) for value in ops.finite_checks) == 3
    assert isinstance(ops.finite_checks[-1], tuple)


def test_nonfinite_action_fails_preserves_failure_and_does_not_advance():
    model, adapter, _ = make_adapter()
    with adapter.episode(context()):
        with pytest.raises(RuntimeError, match="non-finite"):
            run(model, adapter, actions=(float("nan"),) + (0.0,) * 55)
        assert adapter.controller.query_index == 0
        assert adapter.cache.entry is None
        assert adapter.last_failure is not None
        assert adapter.last_failure.query_index == 0
        assert adapter.last_failure.classification == "invariant"


def test_custom_action_finite_checker_supports_non_tensor_action_outputs():
    checked = []

    def checker(value):
        checked.append(value)
        return all(math.isfinite(float(item)) for item in value)

    model, adapter, ops = make_adapter(action_finite_checker=checker)
    with adapter.episode(context()):
        run(model, adapter)
    assert len(checked) == 1
    assert ops.finite_checks == []


def test_immutable_v2_identity_and_monotonic_recovery(tmp_path):
    identity = AttemptIdentity("acr-v2c-correctness-v01", "sa-dp-acr", "libero-object", 0, 0, 0, 0)
    record = {
        "schema_version": "acr.v2-query.v1",
        "query_id": identity.query_id(0),
        "status": "failed",
        "record_sha256": semantic_sha256({"query_id": identity.query_id(0)}),
    }
    store = ImmutableRecordStore(tmp_path)
    store.write_once(identity.query_id(0), record)
    with pytest.raises(FileExistsError):
        store.write_once(identity.query_id(0), {**record, "status": "completed"})
    pairing = identity.value.rsplit("/attempt-", 1)[0]
    assert store.next_attempt_index(pairing) == 1
