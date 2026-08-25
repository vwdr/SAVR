"""Outcome-blind profile validation and suffix-safe nested reuse sets."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

from savr.brace.sequence_map import SequenceMap
from savr.brace.types import B2ValidationError, Camera, Profile, ProfileFamily


def validate_profile(profile: Profile) -> None:
    if not profile.profile_id or profile.family not in (ProfileFamily.P1, ProfileFamily.P2):
        raise B2ValidationError("BRACE base profile must be a named P1 or P2 profile")
    if not profile.budgets:
        raise B2ValidationError("profile has no reuse layer")
    ordered = sorted(profile.budgets, key=lambda item: item.layer)
    if list(profile.budgets) != ordered or len({item.layer for item in ordered}) != len(ordered):
        raise B2ValidationError("profile layers must be unique and increasing")
    previous_scene = previous_wrist = -1
    for budget in ordered:
        budget.validate()
        if budget.scene < previous_scene or budget.wrist < previous_wrist:
            raise B2ValidationError("reuse budgets must be nondecreasing with decoder depth")
        previous_scene, previous_wrist = budget.scene, budget.wrist
    if profile.family is ProfileFamily.P1 and any(item.wrist != 0 for item in ordered):
        raise B2ValidationError("P1 must refresh every wrist token")
    if profile.family is ProfileFamily.P2:
        if profile.wrist_max_age > profile.scene_max_age:
            raise B2ValidationError("P2 wrist age must be no looser than scene age")
        if profile.wrist_change_limit > profile.scene_change_limit:
            raise B2ValidationError("P2 wrist change limit must be no looser than scene")
    if min(
        profile.scene_change_limit,
        profile.wrist_change_limit,
        profile.scene_max_age,
        profile.wrist_max_age,
    ) < 0:
        raise B2ValidationError("profile thresholds and ages must be nonnegative")


def validate_profile_grid(profiles: Sequence[Profile]) -> None:
    if not profiles or len(profiles) > 6:
        raise B2ValidationError("profile grid must contain one to six base profiles")
    if len({profile.profile_id for profile in profiles}) != len(profiles):
        raise B2ValidationError("profile grid identities must be unique")
    if sum(profile.family is ProfileFamily.P1 for profile in profiles) > 3:
        raise B2ValidationError("profile grid exceeds three scene-only profiles")
    if sum(profile.family is ProfileFamily.P2 for profile in profiles) > 3:
        raise B2ValidationError("profile grid exceeds three dual-view profiles")
    for profile in profiles:
        validate_profile(profile)


def build_nested_reuse_sets(
    profile: Profile,
    sequence_map: SequenceMap,
    *,
    eligible: Mapping[int, Mapping[int, bool]],
    change_scores: Mapping[int, Mapping[int, float]],
) -> dict[int, frozenset[int]]:
    """Construct nested sets using eligibility across the complete layer suffix."""

    validate_profile(profile)
    layers = [budget.layer for budget in profile.budgets]
    visual = set(sequence_map.scene_positions + sequence_map.wrist_positions)
    if set(eligible) != set(layers) or set(change_scores) != set(layers):
        raise B2ValidationError("eligibility/change maps must cover every reuse layer")
    for layer in layers:
        if set(eligible[layer]) != visual or set(change_scores[layer]) != visual:
            raise B2ValidationError("layer maps must cover every runtime visual token")

    protected = set(profile.protected_scene) | set(profile.protected_wrist)
    if protected - visual:
        raise B2ValidationError("protected-token map contains a nonvisual position")
    selected: set[int] = set()
    result: dict[int, frozenset[int]] = {}
    for layer_index, budget in enumerate(profile.budgets):
        suffix = layers[layer_index:]
        for camera, limit in ((Camera.SCENE, budget.scene), (Camera.WRIST, budget.wrist)):
            positions = set(sequence_map.positions(camera))
            retained = selected & positions
            candidates = []
            for position in positions - selected - protected:
                if all(bool(eligible[suffix_layer][position]) for suffix_layer in suffix):
                    worst_change = max(float(change_scores[suffix_layer][position]) for suffix_layer in suffix)
                    candidates.append((worst_change, position))
            candidates.sort()
            additions = max(0, limit - len(retained))
            selected.update(position for _score, position in candidates[:additions])
        result[budget.layer] = frozenset(selected)

    previous: frozenset[int] = frozenset()
    for layer in layers:
        current = result[layer]
        if not previous <= current:
            raise B2ValidationError("constructed reuse sets are not nested")
        previous = current
    return result


def summarize_reuse_sets(reuse_sets: Mapping[int, Iterable[int]]) -> dict[str, int]:
    counts = {int(layer): len(set(positions)) for layer, positions in reuse_sets.items()}
    return {
        "layers": len(counts),
        "minimum_reused": min(counts.values(), default=0),
        "maximum_reused": max(counts.values(), default=0),
    }


def validate_exact_source_eligibility(
    *,
    source_queries: Mapping[tuple[int, int], int],
    source_change_scores: Mapping[tuple[int, int, int], float],
) -> None:
    """Reject anchor-only scores when live entries have mixed sources."""

    missing: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (layer, position), source in source_queries.items():
        if (layer, position, source) not in source_change_scores:
            missing[source].append((layer, position))
    if missing:
        raise B2ValidationError("change scores do not resolve every entry's actual source query")
