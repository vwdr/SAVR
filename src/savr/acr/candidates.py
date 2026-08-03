"""Deterministic offline derivation of the three frozen ACR candidates."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from savr.acr.controller import ACRController
from savr.acr.records import canonical_json_bytes, semantic_sha256
from savr.acr.types import ACRConfiguration, ACRContext, ACRPolicy
from savr.acr.signals import scene_change_from_representations, scene_relative_translation


@dataclass(frozen=True)
class FRTraceQuery:
    episode_id: str
    query_index: int
    scene_representation: tuple[float, ...]
    normalized_eef_position: tuple[float, float, float]
    action_chunk: tuple[float, ...]


@dataclass(frozen=True)
class CandidateTemplate:
    configuration_id: str
    target_reuse: float
    horizon: int
    hard_reuse_cap: float


@dataclass(frozen=True)
class DerivedCandidate:
    configuration_id: str
    status: str
    quantile: float | None
    scene_threshold: float | None
    translation_threshold: float | None
    replay_reuse: float | None
    target_reuse: float
    horizon: int
    hard_reuse_cap: float
    trace_sha256: str
    controller_version: str = "acr-controller-v1"


TEMPLATES = (
    CandidateTemplate("acr-t25-h2-b30", 0.25, 2, 0.30),
    CandidateTemplate("acr-t50-h4-b55", 0.50, 4, 0.55),
    CandidateTemplate("acr-t70-h8-b75", 0.70, 8, 0.75),
)
QUANTILE_GRID = tuple(value / 1000 for value in range(500, 1000, 5))


def select_replay_option(
    options: tuple[tuple[float, float, float, float], ...], target_reuse: float
) -> tuple[float, float, float, float]:
    """Apply the frozen distance, lower-reuse, then lower-quantile tie break."""

    if not options:
        raise ValueError("Candidate selection requires at least one valid option")
    return min(
        options,
        key=lambda option: (
            abs(option[0] - target_reuse),
            option[0],
            option[1],
        ),
    )


def _quantile(values: tuple[float, ...], q: float) -> float:
    if not values:
        raise ValueError("Cannot derive a quantile from an empty distribution")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _episodes(trace: tuple[FRTraceQuery, ...]) -> dict[str, tuple[FRTraceQuery, ...]]:
    grouped: dict[str, list[FRTraceQuery]] = {}
    for query in trace:
        grouped.setdefault(query.episode_id, []).append(query)
    result: dict[str, tuple[FRTraceQuery, ...]] = {}
    for episode_id in sorted(grouped):
        queries = tuple(sorted(grouped[episode_id], key=lambda item: item.query_index))
        if tuple(item.query_index for item in queries) != tuple(range(len(queries))):
            raise ValueError(f"FR trace is not contiguous for episode {episode_id}")
        result[episode_id] = queries
    return result


def _adjacent_distributions(
    episodes: dict[str, tuple[FRTraceQuery, ...]],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    scene: list[float] = []
    translation: list[float] = []
    for queries in episodes.values():
        for previous, current in zip(queries, queries[1:]):
            scene.append(
                scene_change_from_representations(
                    current.scene_representation, previous.scene_representation
                ).top_four_mean
            )
            translation.append(
                scene_relative_translation(
                    current.normalized_eef_position, previous.normalized_eef_position
                )
            )
    return tuple(scene), tuple(translation)


def _context(configuration: ACRConfiguration, episode_id: str) -> ACRContext:
    return ACRContext(
        episode_id=episode_id,
        attempt_id=f"offline-{episode_id}",
        task_id="offline",
        instruction_sha256="0" * 64,
        checkpoint_id="offline",
        upstream_revision="offline",
        configuration_id=configuration.configuration_id,
        controller_version=configuration.controller_version,
        preprocessing_id="acr-scene-32-v1",
        action_head_id="offline",
        dtype="offline",
        device="cpu",
        patch_count=1,
    )


def replay_reuse_rate(
    episodes: dict[str, tuple[FRTraceQuery, ...]], configuration: ACRConfiguration
) -> float:
    reuses = 0
    queries_total = 0
    for episode_id, queries in episodes.items():
        controller = ACRController(configuration)
        controller.reset(_context(configuration, episode_id))
        cache_available = False
        cache_age = 0
        for query in queries:
            decision = controller.decide(
                scene_representation=query.scene_representation,
                normalized_eef_position=query.normalized_eef_position,
                cache_available=cache_available,
                cache_age=cache_age,
            )
            if decision.refresh:
                cache_available = True
                cache_age = 0
            else:
                reuses += 1
                cache_age += 1
            queries_total += 1
            controller.observe(
                decision=decision,
                scene_representation=query.scene_representation,
                normalized_eef_position=query.normalized_eef_position,
                action_chunk=query.action_chunk,
            )
    return reuses / queries_total if queries_total else 0.0


def derive_candidates(trace: tuple[FRTraceQuery, ...]) -> dict[str, Any]:
    episodes = _episodes(trace)
    scene_distribution, translation_distribution = _adjacent_distributions(episodes)
    trace_payload = [asdict(query) for query in trace]
    trace_sha256 = semantic_sha256(trace_payload)
    candidates: list[DerivedCandidate] = []
    for template in TEMPLATES:
        options: list[tuple[float, float, float, float]] = []
        for q in QUANTILE_GRID:
            scene_threshold = _quantile(scene_distribution, q)
            translation_threshold = _quantile(translation_distribution, q)
            configuration = ACRConfiguration(
                configuration_id=template.configuration_id,
                policy=ACRPolicy.SA_ACR,
                scene_threshold=scene_threshold,
                translation_threshold=translation_threshold,
                horizon=template.horizon,
                hard_reuse_cap=template.hard_reuse_cap,
            )
            reuse = replay_reuse_rate(episodes, configuration)
            if reuse <= template.hard_reuse_cap:
                options.append((reuse, q, scene_threshold, translation_threshold))
        if not options:
            candidates.append(
                DerivedCandidate(
                    configuration_id=template.configuration_id,
                    status="DERIVATION_INELIGIBLE",
                    quantile=None,
                    scene_threshold=None,
                    translation_threshold=None,
                    replay_reuse=None,
                    target_reuse=template.target_reuse,
                    horizon=template.horizon,
                    hard_reuse_cap=template.hard_reuse_cap,
                    trace_sha256=trace_sha256,
                )
            )
            continue
        selected = select_replay_option(tuple(options), template.target_reuse)
        candidates.append(
            DerivedCandidate(
                configuration_id=template.configuration_id,
                status="DERIVATION_ELIGIBLE",
                quantile=selected[1],
                scene_threshold=selected[2],
                translation_threshold=selected[3],
                replay_reuse=selected[0],
                target_reuse=template.target_reuse,
                horizon=template.horizon,
                hard_reuse_cap=template.hard_reuse_cap,
                trace_sha256=trace_sha256,
            )
        )
    payload: dict[str, Any] = {
        "schema_version": "acr.candidates.v1",
        "trace_sha256": trace_sha256,
        "quantile_grid": list(QUANTILE_GRID),
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    payload["semantic_sha256"] = semantic_sha256(payload)
    return payload


def derive_candidates_bytes(trace: tuple[FRTraceQuery, ...]) -> bytes:
    return canonical_json_bytes(derive_candidates(trace)) + b"\n"
