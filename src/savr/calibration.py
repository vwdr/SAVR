"""Deterministic offline calibration from Full Refresh query traces."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from savr.signals import action_change, image_change, state_change


@dataclass(frozen=True)
class CalibrationQuery:
    """Inputs needed to replay a refresh decision at one policy query."""

    images: Mapping[str, Sequence[float]]
    image_shapes: Mapping[str, tuple[int, ...]]
    state: tuple[float, ...]
    actions: tuple[float, ...]


@dataclass(frozen=True)
class SignalBounds:
    state_q01: tuple[float, ...]
    state_q99: tuple[float, ...]
    action_q01: tuple[float, ...]
    action_q99: tuple[float, ...]


@dataclass(frozen=True)
class ReplayResult:
    refreshes: int
    reuses: int
    decisions: tuple[bool, ...]

    @property
    def query_count(self) -> int:
        return self.refreshes + self.reuses

    @property
    def skip_rate(self) -> float:
        return self.reuses / self.query_count if self.query_count else 0.0

    @property
    def refresh_rate(self) -> float:
        return self.refreshes / self.query_count if self.query_count else 0.0


@dataclass(frozen=True)
class ThresholdCandidate:
    quantile: float
    image_threshold: float
    state_threshold: float | None
    action_threshold: float | None
    max_reuse_horizon: int
    simulated_refreshes: int
    simulated_reuses: int

    @property
    def simulated_skip_rate(self) -> float:
        total = self.simulated_refreshes + self.simulated_reuses
        return self.simulated_reuses / total if total else 0.0


@dataclass(frozen=True)
class PreparedEpisode:
    """Precomputed signal scores for fast exact threshold replay."""

    image_scores: tuple[tuple[float | None, ...], ...]
    state_scores: tuple[float | None, ...]
    action_scores: tuple[float | None, ...]


def _finite_tuple(values: Sequence[Any], *, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must be a non-empty finite sequence")
    return result


def query_from_record(record: Mapping[str, Any]) -> CalibrationQuery:
    """Validate and convert one Phase 6 FR query record."""

    trace = record.get("calibration_trace")
    if not isinstance(trace, Mapping):
        raise ValueError("Query record lacks calibration_trace")
    raw_images = trace.get("images")
    raw_shapes = trace.get("image_shapes")
    if not isinstance(raw_images, Mapping) or not isinstance(raw_shapes, Mapping):
        raise ValueError("Calibration trace lacks image values or shapes")
    if set(raw_images) != set(raw_shapes) or not raw_images:
        raise ValueError("Calibration image values and shapes differ")

    images: dict[str, tuple[float, ...]] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    for name in sorted(raw_images):
        values = _finite_tuple(raw_images[name], name=f"image {name}")
        shape = tuple(int(value) for value in raw_shapes[name])
        if shape not in ((32, 32), (32, 32, 3)):
            raise ValueError(f"Unsupported calibration image shape for {name}: {shape}")
        if math.prod(shape) != len(values):
            raise ValueError(f"Calibration image length differs for {name}")
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError(f"Calibration image {name} is outside [0,1]")
        images[str(name)] = values
        shapes[str(name)] = shape

    state = _finite_tuple(trace.get("state", ()), name="state")
    actions = _finite_tuple(trace.get("actions", ()), name="actions")
    if len(state) != 8:
        raise ValueError("Calibration state must have width 8")
    if len(actions) != 56:
        raise ValueError("Calibration action chunk must have width 56")
    return CalibrationQuery(images=images, image_shapes=shapes, state=state, actions=actions)


def _images_as_nested(query: CalibrationQuery) -> dict[str, Any]:
    """Restore 32x32 images for the production signal implementation."""

    nested: dict[str, Any] = {}
    for name, values in query.images.items():
        shape = query.image_shapes[name]
        channels = 1 if len(shape) == 2 else shape[2]
        iterator = iter(values)
        rows = []
        for _ in range(32):
            row = []
            for _ in range(32):
                pixel = [next(iterator) for _ in range(channels)]
                row.append(pixel[0] if channels == 1 else pixel)
            rows.append(row)
        nested[name] = rows
    return nested


def _representation_change(
    current: CalibrationQuery,
    reference: CalibrationQuery,
) -> float:
    if set(current.images) != set(reference.images):
        raise ValueError("Calibration camera sets differ")
    camera_scores = []
    for name in sorted(current.images):
        left = current.images[name]
        right = reference.images[name]
        if current.image_shapes[name] != reference.image_shapes[name] or len(left) != len(right):
            raise ValueError(f"Calibration camera shape changed: {name}")
        camera_scores.append(
            sum(abs(a - b) for a, b in zip(left, right)) / len(left)
        )
    return sum(camera_scores) / len(camera_scores)


def prepare_episodes(
    episodes: Sequence[Sequence[CalibrationQuery]],
    bounds: SignalBounds,
) -> tuple[PreparedEpisode, ...]:
    """Precompute every possible refresh-reference image score once."""

    prepared = []
    for episode in episodes:
        image_rows: list[tuple[float | None, ...]] = []
        state_scores: list[float | None] = []
        action_scores: list[float | None] = []
        for index, query in enumerate(episode):
            image_rows.append(
                tuple(
                    _representation_change(query, episode[reference_index])
                    if reference_index < index
                    else None
                    for reference_index in range(len(episode))
                )
            )
            state_scores.append(
                None
                if index == 0
                else state_change(
                    query.state,
                    episode[index - 1].state,
                    bounds.state_q01,
                    bounds.state_q99,
                )
            )
            action_scores.append(
                None
                if index < 2
                else action_change(
                    episode[index - 1].actions,
                    episode[index - 2].actions,
                    bounds.action_q01,
                    bounds.action_q99,
                )
            )
        prepared.append(
            PreparedEpisode(
                image_scores=tuple(image_rows),
                state_scores=tuple(state_scores),
                action_scores=tuple(action_scores),
            )
        )
    return tuple(prepared)


def empirical_quantile(values: Sequence[float], quantile: float) -> float:
    """Return the linear empirical quantile used by the frozen protocol."""

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("Quantile must lie in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if not ordered or not all(math.isfinite(value) for value in ordered):
        raise ValueError("Quantile values must be non-empty and finite")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _required_score(value: float | None) -> float:
    if value is None:
        raise ValueError("Required prepared signal score is missing")
    return value


def signal_distributions(
    episodes: Sequence[Sequence[CalibrationQuery]],
    bounds: SignalBounds,
) -> dict[str, tuple[float, ...]]:
    """Collect adjacent-query FR signal distributions."""

    prepared = prepare_episodes(episodes, bounds)
    image_scores = [
        _required_score(episode.image_scores[index][index - 1])
        for episode in prepared
        for index in range(1, len(episode.image_scores))
    ]
    state_scores = [
        float(score)
        for episode in prepared
        for score in episode.state_scores
        if score is not None
    ]
    action_scores = [
        float(score)
        for episode in prepared
        for score in episode.action_scores
        if score is not None
    ]
    if not image_scores or not state_scores or not action_scores:
        raise ValueError("FR traces do not contain enough query history")
    return {
        "image": tuple(image_scores),
        "state": tuple(state_scores),
        "action": tuple(action_scores),
    }


def _distributions_from_prepared(
    prepared: Sequence[PreparedEpisode],
) -> dict[str, tuple[float, ...]]:
    result = {
        "image": tuple(
            _required_score(episode.image_scores[index][index - 1])
            for episode in prepared
            for index in range(1, len(episode.image_scores))
        ),
        "state": tuple(
            float(score)
            for episode in prepared
            for score in episode.state_scores
            if score is not None
        ),
        "action": tuple(
            float(score)
            for episode in prepared
            for score in episode.action_scores
            if score is not None
        ),
    }
    if any(not values for values in result.values()):
        raise ValueError("FR traces do not contain enough query history")
    return result


def replay_prepared_episode(
    episode: PreparedEpisode,
    *,
    image_threshold: float,
    max_reuse_horizon: int,
    state_threshold: float | None = None,
    action_threshold: float | None = None,
) -> ReplayResult:
    """Replay using precomputed scores with production-equivalent comparisons."""

    savr = state_threshold is not None or action_threshold is not None
    if savr and (state_threshold is None or action_threshold is None):
        raise ValueError("SAVR replay requires both state and action thresholds")
    decisions: list[bool] = []
    reference_index: int | None = None
    cache_age = 0
    for index in range(len(episode.image_scores)):
        refresh = reference_index is None
        if reference_index is not None:
            score = episode.image_scores[index][reference_index]
            if score is None:
                raise ValueError("Prepared image score is missing")
            refresh = refresh or score > image_threshold
            if cache_age >= max_reuse_horizon:
                refresh = True
        if savr:
            if index < 2:
                refresh = True
            else:
                assert state_threshold is not None and action_threshold is not None
                state_score_value = episode.state_scores[index]
                action_score_value = episode.action_scores[index]
                if state_score_value is None or action_score_value is None:
                    raise ValueError("Prepared SAVR history score is missing")
                refresh = (
                    refresh
                    or state_score_value > state_threshold
                    or action_score_value > action_threshold
                )
        decisions.append(refresh)
        if refresh:
            reference_index = index
            cache_age = 0
        else:
            cache_age += 1
    refreshes = sum(decisions)
    return ReplayResult(
        refreshes=refreshes,
        reuses=len(decisions) - refreshes,
        decisions=tuple(decisions),
    )


def replay_prepared_episodes(
    episodes: Sequence[PreparedEpisode],
    **kwargs: Any,
) -> ReplayResult:
    decisions = [
        decision
        for episode in episodes
        for decision in replay_prepared_episode(episode, **kwargs).decisions
    ]
    refreshes = sum(decisions)
    return ReplayResult(
        refreshes=refreshes,
        reuses=len(decisions) - refreshes,
        decisions=tuple(decisions),
    )


def replay_episode(
    queries: Sequence[CalibrationQuery],
    *,
    image_threshold: float,
    max_reuse_horizon: int,
    bounds: SignalBounds,
    state_threshold: float | None = None,
    action_threshold: float | None = None,
) -> ReplayResult:
    """Replay exact VOR or SAVR decision semantics on one FR trace."""

    if max_reuse_horizon < 1:
        raise ValueError("Maximum reuse horizon must be positive")
    savr = state_threshold is not None or action_threshold is not None
    if savr and (state_threshold is None or action_threshold is None):
        raise ValueError("SAVR replay requires both state and action thresholds")

    reference_images: dict[str, tuple[float, ...]] | None = None
    previous_state: tuple[float, ...] | None = None
    action_history: list[tuple[float, ...]] = []
    cache_available = False
    cache_age = 0
    decisions: list[bool] = []

    from savr.signals import prepare_image_representations

    for query in queries:
        images = _images_as_nested(query)
        refresh = not cache_available
        if reference_images is not None:
            refresh = refresh or image_change(images, reference_images).mean > image_threshold
        elif cache_available:
            raise ValueError("Replay cache lacks its image reference")
        if cache_available and cache_age >= max_reuse_horizon:
            refresh = True

        if savr:
            if previous_state is None and cache_available:
                refresh = True
            elif previous_state is not None:
                assert state_threshold is not None
                refresh = refresh or state_change(
                    query.state,
                    previous_state,
                    bounds.state_q01,
                    bounds.state_q99,
                ) > state_threshold
            if len(action_history) < 2:
                refresh = True
            else:
                assert action_threshold is not None
                refresh = refresh or action_change(
                    action_history[-1],
                    action_history[-2],
                    bounds.action_q01,
                    bounds.action_q99,
                ) > action_threshold

        decisions.append(refresh)
        if refresh:
            reference_images = prepare_image_representations(images)
            cache_available = True
            cache_age = 0
        else:
            cache_age += 1
        if savr:
            previous_state = query.state
            action_history.append(query.actions)
            action_history = action_history[-2:]

    refreshes = sum(decisions)
    return ReplayResult(
        refreshes=refreshes,
        reuses=len(decisions) - refreshes,
        decisions=tuple(decisions),
    )


def replay_episodes(
    episodes: Sequence[Sequence[CalibrationQuery]],
    **kwargs: Any,
) -> ReplayResult:
    decisions: list[bool] = []
    for episode in episodes:
        decisions.extend(replay_episode(episode, **kwargs).decisions)
    refreshes = sum(decisions)
    return ReplayResult(
        refreshes=refreshes,
        reuses=len(decisions) - refreshes,
        decisions=tuple(decisions),
    )


def derive_savr_candidate(
    episodes: Sequence[Sequence[CalibrationQuery]],
    *,
    bounds: SignalBounds,
    target_skip_rate: float,
    max_reuse_horizon: int,
    quantile_step: int = 1000,
) -> ThresholdCandidate:
    """Apply the frozen common-quantile search for one SAVR grid point."""

    if not 0.0 <= target_skip_rate <= 1.0:
        raise ValueError("Target skip rate must lie in [0, 1]")
    prepared = prepare_episodes(episodes, bounds)
    distributions = _distributions_from_prepared(prepared)
    candidates: list[ThresholdCandidate] = []
    for index in range(quantile_step + 1):
        quantile = index / quantile_step
        thresholds = {
            name: empirical_quantile(values, quantile)
            for name, values in distributions.items()
        }
        replay = replay_prepared_episodes(
            prepared,
            image_threshold=thresholds["image"],
            state_threshold=thresholds["state"],
            action_threshold=thresholds["action"],
            max_reuse_horizon=max_reuse_horizon,
        )
        candidates.append(
            ThresholdCandidate(
                quantile=quantile,
                image_threshold=thresholds["image"],
                state_threshold=thresholds["state"],
                action_threshold=thresholds["action"],
                max_reuse_horizon=max_reuse_horizon,
                simulated_refreshes=replay.refreshes,
                simulated_reuses=replay.reuses,
            )
        )
    return min(
        candidates,
        key=lambda item: (
            abs(item.simulated_skip_rate - target_skip_rate),
            item.simulated_skip_rate > target_skip_rate,
            item.quantile,
        ),
    )


def derive_savr_grid(
    episodes: Sequence[Sequence[CalibrationQuery]],
    *,
    bounds: SignalBounds,
    target_skip_rates: Sequence[float],
    max_reuse_horizons: Sequence[int],
    quantile_step: int = 1000,
) -> dict[tuple[float, int], ThresholdCandidate]:
    """Derive a complete SAVR grid while sharing expensive preprocessing."""

    prepared = prepare_episodes(episodes, bounds)
    distributions = _distributions_from_prepared(prepared)
    thresholds_by_quantile = [
        (
            index / quantile_step,
            {
                name: empirical_quantile(values, index / quantile_step)
                for name, values in distributions.items()
            },
        )
        for index in range(quantile_step + 1)
    ]
    result = {}
    for horizon in max_reuse_horizons:
        candidates = []
        for quantile, thresholds in thresholds_by_quantile:
            replay = replay_prepared_episodes(
                prepared,
                image_threshold=thresholds["image"],
                state_threshold=thresholds["state"],
                action_threshold=thresholds["action"],
                max_reuse_horizon=horizon,
            )
            candidates.append(
                ThresholdCandidate(
                    quantile=quantile,
                    image_threshold=thresholds["image"],
                    state_threshold=thresholds["state"],
                    action_threshold=thresholds["action"],
                    max_reuse_horizon=horizon,
                    simulated_refreshes=replay.refreshes,
                    simulated_reuses=replay.reuses,
                )
            )
        for target in target_skip_rates:
            result[(float(target), int(horizon))] = min(
                candidates,
                key=lambda item: (
                    abs(item.simulated_skip_rate - target),
                    item.simulated_skip_rate > target,
                    item.quantile,
                ),
            )
    return result


def derive_vor_candidates(
    episodes: Sequence[Sequence[CalibrationQuery]],
    *,
    bounds: SignalBounds,
    target_refresh_rate: float,
    max_reuse_horizon: int,
    quantile_step: int = 1000,
) -> tuple[ThresholdCandidate, ...]:
    """Rank the complete VOR threshold grid by closeness to a target budget."""

    prepared = prepare_episodes(episodes, bounds)
    image_scores = _distributions_from_prepared(prepared)["image"]
    candidates = []
    for index in range(quantile_step + 1):
        quantile = index / quantile_step
        threshold = empirical_quantile(image_scores, quantile)
        replay = replay_prepared_episodes(
            prepared,
            image_threshold=threshold,
            max_reuse_horizon=max_reuse_horizon,
        )
        candidates.append(
            ThresholdCandidate(
                quantile=quantile,
                image_threshold=threshold,
                state_threshold=None,
                action_threshold=None,
                max_reuse_horizon=max_reuse_horizon,
                simulated_refreshes=replay.refreshes,
                simulated_reuses=replay.reuses,
            )
        )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                abs(item.simulated_refreshes / (
                    item.simulated_refreshes + item.simulated_reuses
                ) - target_refresh_rate),
                (
                    item.simulated_refreshes
                    / (item.simulated_refreshes + item.simulated_reuses)
                ) < target_refresh_rate,
                item.quantile,
            ),
        )
    )


def select_period(
    query_lengths: Sequence[int],
    *,
    target_refresh_rate: float,
    periods: Sequence[int] = tuple(range(1, 9)),
) -> tuple[int, float]:
    """Select the fixed PR period closest to the declared refresh budget."""

    if not query_lengths or any(length < 1 for length in query_lengths):
        raise ValueError("Query lengths must be positive")
    total_queries = sum(query_lengths)
    candidates = []
    for period in periods:
        if period < 1:
            raise ValueError("Periods must be positive")
        refreshes = sum((length - 1) // period + 1 for length in query_lengths)
        rate = refreshes / total_queries
        candidates.append((period, rate))
    return min(
        candidates,
        key=lambda item: (
            abs(item[1] - target_refresh_rate),
            item[1] < target_refresh_rate,
            item[0],
        ),
    )
