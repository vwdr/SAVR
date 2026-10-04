"""Exact deployment-spaced query indexing without filtered-frame shortcuts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from savr.pair.types import PairValidationError, QueryKey, Split


@dataclass(frozen=True)
class TrajectorySpec:
    suite: str
    task_id: str
    trajectory_id: str
    action_count: int
    split: Split

    def validate(self) -> None:
        if not self.suite or not self.task_id or not self.trajectory_id or self.action_count < 0:
            raise PairValidationError("trajectory metadata is incomplete")
        if not isinstance(self.split, Split):
            raise PairValidationError("trajectory split is unsupported")


@dataclass(frozen=True)
class AlignedQuery:
    key: QueryKey
    action_indices: tuple[int, ...]
    next_step: int | None

    def validate(self, actions_per_query: int = 8) -> None:
        self.key.validate()
        expected = tuple(range(self.key.step, self.key.step + actions_per_query))
        if self.action_indices != expected:
            raise PairValidationError("expert chunk is not aligned to original actions")
        if self.next_step is not None and self.next_step - self.key.step != actions_per_query:
            raise PairValidationError("successive queries are not exactly eight actions apart")


def build_query_index(
    trajectories: Iterable[TrajectorySpec], *, actions_per_query: int = 8
) -> tuple[AlignedQuery, ...]:
    if actions_per_query != 8:
        raise PairValidationError("PAIR is frozen to eight actions per query")
    rows: list[AlignedQuery] = []
    seen: set[tuple[str, str, str]] = set()
    for trajectory in trajectories:
        trajectory.validate()
        identity = (trajectory.suite, trajectory.task_id, trajectory.trajectory_id)
        if identity in seen:
            raise PairValidationError("duplicate trajectory identity")
        seen.add(identity)
        starts = tuple(
            range(0, max(0, trajectory.action_count - actions_per_query + 1), actions_per_query)
        )
        for index, start in enumerate(starts):
            next_step = starts[index + 1] if index + 1 < len(starts) else None
            row = AlignedQuery(
                key=QueryKey(*identity, start, trajectory.split),
                action_indices=tuple(range(start, start + actions_per_query)),
                next_step=next_step,
            )
            row.validate(actions_per_query)
            rows.append(row)
    return tuple(rows)


def assert_split_inheritance(rows: Iterable[AlignedQuery]) -> None:
    inherited: dict[tuple[str, str, str], Split] = {}
    for row in rows:
        row.validate()
        identity = (row.key.suite, row.key.task_id, row.key.trajectory_id)
        previous = inherited.setdefault(identity, row.key.split)
        if previous is not row.key.split:
            raise PairValidationError("trajectory crosses data splits")
