from __future__ import annotations

import json
from pathlib import Path

import pytest

from savr.pair.p3 import (
    P3Profile,
    QueryLedger,
    block_schedule,
    net_saving_lower,
    one_sided_bootstrap_lower,
    planned_query_count,
    profile_source_diversity,
    reject_protected_fields,
    validate_config,
)
from savr.pair.features import GroupFeature, RouterFeatures
from savr.pair.router import PairRouter
from savr.pair.types import AtomicGroup, Camera
from savr.pair.types import PairValidationError


ROOT = Path(__file__).resolve().parents[2]


def config():
    return json.loads((ROOT / "configs/pair/p3_physical_v1.json").read_text())


def test_p3_config_schedule_and_query_cap_are_exact():
    value = config()
    validate_config(value, input_count=8)
    schedule = block_schedule(value, 8)
    assert len(schedule) == 8 * 3 * 4 == 96
    assert schedule == block_schedule(value, 8)
    assert len({block.block_id for block in schedule}) == 96
    assert planned_query_count(value, 8) == 688


def test_p3_query_ledger_cannot_borrow_or_exceed():
    ledger = QueryLedger(4, 3)
    for _ in range(3):
        ledger.consume()
    ledger.require_complete()
    with pytest.raises(PairValidationError, match="exceeded"):
        ledger.consume()


def test_p3_bootstrap_and_net_speed_known_answers():
    point, lower = one_sided_bootstrap_lower(
        [100, 100, 100, 100], [80, 80, 80, 80], replicates=100, seed=7
    )
    assert point == pytest.approx(0.2) and lower == pytest.approx(0.2)
    assert net_saving_lower(lower, 0.02) == pytest.approx(0.12)


def test_p3_profiles_preserve_source_diversity_and_protection_limits():
    profiles = [P3Profile.from_mapping(value) for value in config()["profiles"]]
    assert all(profile_source_diversity(profile)["mixed_source_possible"] for profile in profiles)
    bad = dict(config()["profiles"][0])
    bad["protected_primary_tiles"] = 15
    with pytest.raises(PairValidationError, match="outside|too few"):
        P3Profile.from_mapping(bad)


def test_p3_timing_aliases_collapse_to_six_router_categories():
    profiles = [P3Profile.from_mapping(value) for value in config()["profiles"]]
    categories = list(dict.fromkeys(profile.base_profile_id for profile in profiles))
    assert len(categories) == 6
    router = PairRouter.initialize(profiles=categories, seed=17)
    features = RouterFeatures(
        (
            GroupFeature(
                AtomicGroup(Camera.PRIMARY, 0, 2),
                profiles[0].base_profile_id,
                1,
                (0.0,) * 20,
            ),
        ),
        (0.0,) * 16,
    )
    assert router.predict(features).positive_regret_q90 > 0


@pytest.mark.parametrize("field", ["success", "reward", "expert_actions", "action_parity"])
def test_p3_records_reject_outcomes_labels_and_action_comparisons(field):
    with pytest.raises(PairValidationError, match="protected"):
        reject_protected_fields({"nested": [{field: True}]})
