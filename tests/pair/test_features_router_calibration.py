from __future__ import annotations

import json

import numpy as np
import pytest

from savr.pair.calibration import (
    CandidateRisk,
    EmpiricalCalibrator,
    SupportEnvelope,
    conservative_quantile,
    select_maximum_saving,
)
from savr.pair.features import (
    GroupFeature,
    RouterFeatures,
    feature_schema_sha256,
    reject_forbidden_feature_fields,
)
from savr.pair.router import PairRouter
from savr.pair.runtime import calibrated_route
from savr.pair.types import AtomicGroup, Camera, PairValidationError


@pytest.mark.parametrize(
    "field",
    ["expert_actions", "future_frame", "current_dense_logits", "terminal_success"],
)
def test_forbidden_or_leaking_features_are_rejected_recursively(field):
    with pytest.raises(PairValidationError, match="forbidden"):
        reject_forbidden_feature_fields({"allowed": {field: [1, 2]}})


def test_nonfinite_and_unsupported_feature_categories_fail_closed(router_features_factory):
    features = router_features_factory()
    bad_values = list(features.groups[0].continuous)
    bad_values[0] = np.nan
    bad = RouterFeatures(
        (
            GroupFeature(
                features.groups[0].group,
                features.groups[0].profile_id,
                features.groups[0].horizon,
                tuple(bad_values),
            ),
        ),
        features.global_continuous,
    )
    with pytest.raises(PairValidationError, match="nonfinite"):
        bad.validate()
    with pytest.raises(PairValidationError, match="unsupported profile"):
        router_features_factory(profile_id="UNKNOWN").validate(supported_profiles=("D37_BAL",))
    with pytest.raises(PairValidationError, match="unsupported horizon"):
        GroupFeature(AtomicGroup(Camera.PRIMARY, 0, 2), "D37_BAL", 3, (0.0,) * 20).validate()


def test_router_seed_parameter_cap_and_serialized_inference_are_deterministic(
    tmp_path, router_features_factory
):
    features = router_features_factory()
    left = PairRouter.initialize(profiles=("D37_BAL",), seed=17)
    right = PairRouter.initialize(profiles=("D37_BAL",), seed=17)
    different = PairRouter.initialize(profiles=("D37_BAL",), seed=29)
    assert left.parameter_count < 250_000
    assert left.serialize() == right.serialize()
    assert left.serialize() != different.serialize()
    expected = left.predict(features)
    path = tmp_path / "router.json"
    digest = left.save(path)
    loaded = PairRouter.load(path)
    assert loaded.predict(features) == expected
    assert len(digest) == len(feature_schema_sha256()) == 64

    tampered = json.loads(path.read_text())
    tampered["weights"]["set_head_b"][0] += 1
    path.write_text(json.dumps(tampered))
    with pytest.raises(PairValidationError, match="hash mismatch"):
        PairRouter.load(path)


def test_router_rejects_mixed_contract_identity(router_features_factory):
    first = router_features_factory(profile_id="D37_BAL")
    second = router_features_factory(profile_id="D50_BAL")
    router = PairRouter.initialize(profiles=("D37_BAL", "D50_BAL"), seed=17)
    mixed = RouterFeatures((first.groups[0], second.groups[0]), first.global_continuous)
    with pytest.raises(PairValidationError, match="mix"):
        router.predict(mixed)


def test_conservative_calibration_coverage_support_and_maximum_saving(router_features_factory):
    low = router_features_factory(value=0.1)
    high = router_features_factory(value=0.2)
    assert conservative_quantile([0, 1, 2, 3, 4], 0.9) == 4
    calibrator = EmpiricalCalibrator.fit(
        [0.01, 0.02, 0.03, 0.04, 0.05],
        [0.01, 0.02, 0.04, 0.04, 0.08],
        [low, high],
    )
    assert calibrator.residual_correction == pytest.approx(0.03)
    assert calibrator.empirical_coverage >= 0.9
    assert calibrator.support.contains(low)
    outside = router_features_factory(source_age=2, value=0.3)
    assert not calibrator.support.contains(outside)
    with pytest.raises(PairValidationError, match="outside"):
        calibrator.upper_bound(0.01, outside)

    choice = select_maximum_saving(
        [
            CandidateRisk("D37_BAL", 0.1, 0.01, low),
            CandidateRisk("D50_BAL", 0.2, 0.02, high),
            CandidateRisk("outside", 0.9, 0.0, outside),
        ],
        calibrator,
        delta=0.051,
    )
    assert choice is not None and choice.profile_id == "D50_BAL"


def test_runtime_fails_dense_on_oos_nonfinite_gripper_or_excess_risk(router_features_factory):
    training = router_features_factory(value=0.1)
    router = PairRouter.initialize(profiles=("D37_BAL",), seed=17)
    prediction = router.predict(training)
    calibrator = EmpiricalCalibrator.fit(
        [prediction.positive_regret_q90] * 2,
        [prediction.positive_regret_q90] * 2,
        [training, training],
    )
    accepted = calibrated_route(
        router,
        calibrator,
        training,
        delta=prediction.positive_regret_q90 + 1e-12,
    )
    assert not accepted.execute_dense and accepted.fallback_reason is None
    assert calibrated_route(router, calibrator, training, delta=0).execute_dense
    assert calibrated_route(
        router, calibrator, training, delta=1e9, gripper_veto=True
    ).execute_dense
    assert calibrated_route(
        router, calibrator, router_features_factory(value=0.2), delta=1e9
    ).execute_dense


def test_support_fit_requires_finite_examples(router_features_factory):
    with pytest.raises(PairValidationError, match="requires"):
        SupportEnvelope.fit([])
    assert SupportEnvelope.fit([router_features_factory()]).profiles == ("D37_BAL",)
