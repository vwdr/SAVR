from __future__ import annotations

import math

import pytest

from savr.acr.controller import ACRController
from savr.acr.instrumentation import CameraInstrumentation
from savr.acr.openvla_oft import OpenVLAAsymmetricCameraAdapter
from savr.acr.types import ACRConfiguration, ACRContext, ACRPolicy
from savr.timing import SynchronizedQueryTimer


class FakeTensor:
    def __init__(self, values, shape, *, dtype="float32", device="cpu"):
        flat = tuple(float(value) for value in values)
        if math.prod(shape) != len(flat):
            raise ValueError("Fake tensor data/shape mismatch")
        self.values = flat
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.requires_grad = True

    def detach(self):
        return self

    def clone(self):
        result = FakeTensor(self.values, self.shape, dtype=self.dtype, device=self.device)
        result.requires_grad = self.requires_grad
        return result

    def requires_grad_(self, value):
        self.requires_grad = value
        return self

    def tolist(self):
        return list(self.values)


class FakeOps:
    @staticmethod
    def split(value, sections, *, dim):
        if dim != 1 or sum(sections) != value.shape[1]:
            raise ValueError("Fake split supports the tested channel/token dimension")
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
            rows = []
            for batch_index in range(batch):
                for patch in range(patches):
                    for value in values:
                        width = value.shape[2]
                        start = (batch_index * patches + patch) * width
                        rows.extend(value.values[start : start + width])
            return FakeTensor(
                rows,
                (batch, patches, sum(value.shape[2] for value in values)),
                dtype=first.dtype,
                device=first.device,
            )
        if dim == 1:
            return FakeTensor(
                tuple(item for value in values for item in value.values),
                (
                    first.shape[0],
                    sum(value.shape[1] for value in values),
                    first.shape[2],
                ),
                dtype=first.dtype,
                device=first.device,
            )
        raise ValueError("Unsupported fake concatenation")

    @staticmethod
    def all_finite(value):
        return all(math.isfinite(item) for item in value.values)


class FakeBackbone:
    use_fused_vision_backbone = True

    @staticmethod
    def get_num_images_in_input():
        return 2

    @staticmethod
    def get_num_patches():
        return 2

    @staticmethod
    def _features(value, offset):
        mean = sum(value.values) / len(value.values) + offset
        return FakeTensor((mean, mean), (1, 2, 1))

    def featurizer(self, value):
        return self._features(value, 0)

    def fused_featurizer(self, value):
        return self._features(value, 10)


class FakeModel:
    def __init__(self):
        self.vision_backbone = FakeBackbone()
        self.last_tokens = None
        self.last_state = None

    @staticmethod
    def projector(value):
        projected = tuple(item for item in value.values for _ in range(2))
        return FakeTensor(projected, (1, 2, 4), dtype=value.dtype, device=value.device)

    def _process_vision_features(self, pixel_values, language_embeddings, use_film=False):
        raise AssertionError("The ACR interception was not installed")

    def predict(self, pixels, language, state, *, fail=False):
        self.last_tokens = self._process_vision_features(pixels, language)
        self.last_state = tuple(state)
        if fail:
            raise RuntimeError("downstream failure")
        return tuple(0.0 for _ in range(56))


def config(policy=ACRPolicy.SA_ACR):
    if policy is ACRPolicy.FACTORIZED_FR:
        return ACRConfiguration("factorized", policy)
    return ACRConfiguration(
        "candidate",
        policy,
        scene_threshold=1.0,
        translation_threshold=1.0,
        horizon=4,
        hard_reuse_cap=0.75,
    )


def context(configuration):
    return ACRContext(
        episode_id="episode",
        attempt_id="attempt",
        task_id="task",
        instruction_sha256="0" * 64,
        checkpoint_id="checkpoint",
        upstream_revision="revision",
        configuration_id=configuration.configuration_id,
        controller_version=configuration.controller_version,
        preprocessing_id="preprocessing",
        action_head_id="head",
        dtype="float32",
        device="cpu",
        patch_count=2,
    )


def make_adapter(policy=ACRPolicy.SA_ACR, *, instrumentation=None):
    configuration = config(policy)
    model = FakeModel()
    adapter = OpenVLAAsymmetricCameraAdapter(
        model=model,
        controller=ACRController(configuration),
        tensor_ops=FakeOps(),
        instrumentation=instrumentation,
    )
    adapter.begin_context(context(configuration))
    return model, adapter


def pixels(scene=1.0, wrist=2.0):
    values = (scene,) * 24 + (wrist,) * 24
    return FakeTensor(values, (1, 12, 2, 2))


def run(model, adapter, *, scene=1.0, wrist=2.0, state=(0.0,) * 8, scene_image=None, fail=False):
    pixel_values = pixels(scene, wrist)
    language = FakeTensor((0.0,) * 20, (1, 5, 4))
    image = [[scene] * 32 for _ in range(32)] if scene_image is None else scene_image
    return adapter.run_query(
        query=lambda: model.predict(pixel_values, language, state, fail=fail),
        scene_image=image,
        wrist_image=[[wrist] * 32 for _ in range(32)],
        state=state,
        state_q01=(0.0,) * 8,
        state_q99=(1.0,) * 8,
    )


def test_camera_factorized_token_order_shape_and_counts():
    model, adapter = make_adapter(ACRPolicy.FACTORIZED_FR)
    result = run(model, adapter)
    assert model.last_tokens.shape == (1, 4, 4)
    assert model.last_tokens.values[:8] == (1, 1, 11, 11, 1, 1, 11, 11)
    assert model.last_tokens.values[8:] == (2, 2, 12, 12, 2, 2, 12, 12)
    result.work.validate(scene_refresh=True)


def test_reuse_skips_scene_and_wrist_is_always_fresh():
    model, adapter = make_adapter()
    first = run(model, adapter, wrist=2)
    second = run(model, adapter, wrist=3)
    third = run(model, adapter, wrist=4, state=(0.1,) + (0.0,) * 7)
    assert first.decision.refresh and second.decision.refresh
    assert not third.decision.refresh
    assert third.cache_event == "reuse"
    assert third.work.scene_siglip_calls == 0
    assert third.work.scene_dinov2_calls == 0
    assert third.work.scene_projector_calls == 0
    assert (
        third.work.wrist_siglip_calls,
        third.work.wrist_dinov2_calls,
        third.work.wrist_projector_calls,
    ) == (1, 1, 1)


def test_reuse_preserves_cached_scene_but_current_wrist_and_proprioception():
    model, adapter = make_adapter()
    run(model, adapter, scene=1, wrist=2)
    run(model, adapter, scene=1, wrist=3)
    cached_scene = model.last_tokens.values[:8]
    current_state = (0.1,) + (0.0,) * 7
    run(model, adapter, scene=9, wrist=4, state=current_state, scene_image=[[1.0] * 32 for _ in range(32)])
    assert model.last_tokens.values[:8] == cached_scene
    assert model.last_tokens.values[8:] == (4, 4, 14, 14, 4, 4, 14, 14)
    assert model.last_state == current_state


def test_changing_one_camera_changes_only_its_block():
    model, adapter = make_adapter(ACRPolicy.FACTORIZED_FR)
    run(model, adapter, scene=1, wrist=2)
    baseline = model.last_tokens.values
    run(model, adapter, scene=5, wrist=2)
    scene_changed = model.last_tokens.values
    run(model, adapter, scene=5, wrist=7)
    wrist_changed = model.last_tokens.values
    assert baseline[:8] != scene_changed[:8] and baseline[8:] == scene_changed[8:]
    assert scene_changed[:8] == wrist_changed[:8] and scene_changed[8:] != wrist_changed[8:]


def test_invalid_scene_signal_forces_refresh():
    model, adapter = make_adapter()
    run(model, adapter)
    result = run(model, adapter, scene_image=[1.0, 2.0])
    assert result.decision.refresh
    assert "invalid-signal" in result.decision.reasons
    assert result.work.scene_projector_calls == 1


def test_exception_restores_method_invalidates_cache_and_does_not_advance():
    model, adapter = make_adapter()
    class_method = FakeModel._process_vision_features
    with pytest.raises(RuntimeError, match="downstream failure"):
        run(model, adapter, fail=True)
    assert "_process_vision_features" not in vars(model)
    assert FakeModel._process_vision_features is class_method
    assert adapter.cache.entry is None
    assert adapter.controller.query_index == 0


def test_context_change_invalidates_cache_and_resets_controller():
    model, adapter = make_adapter()
    run(model, adapter)
    old = adapter.context
    assert old is not None and adapter.cache.entry is not None
    changed = ACRContext(**{**old.__dict__, "episode_id": "new-episode"})
    assert adapter.begin_context(changed)
    assert adapter.cache.entry is None
    assert adapter.controller.query_index == 0


def test_synchronized_component_timing_is_returned():
    class Backend:
        def __init__(self):
            self.event = 0
            self.synchronizations = 0

        def synchronize(self):
            self.synchronizations += 1

        def record_event(self):
            self.event += 1
            return self.event

        @staticmethod
        def elapsed_ms(start, end):
            return float(end - start)

    backend = Backend()
    timer = SynchronizedQueryTimer(backend)
    instrumentation = CameraInstrumentation(timer=timer)
    model, adapter = make_adapter(instrumentation=instrumentation)
    result = run(model, adapter)
    assert result.device_timing is not None
    assert result.device_timing.total_device_ms > 0
    assert result.device_timing.component_counts["scene.siglip"] == 1
    assert result.device_timing.component_counts["wrist.projector"] == 1
    assert backend.synchronizations == 2
