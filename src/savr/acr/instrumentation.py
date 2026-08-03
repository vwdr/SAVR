"""Per-camera component accounting with optional synchronized device timing."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypeVar

from savr.acr.types import CameraWork
from savr.timing import QueryTiming, SynchronizedQueryTimer


T = TypeVar("T")


_COUNT_FIELDS = {
    "scene.siglip": "scene_siglip_calls",
    "scene.dinov2": "scene_dinov2_calls",
    "scene.projector": "scene_projector_calls",
    "wrist.siglip": "wrist_siglip_calls",
    "wrist.dinov2": "wrist_dinov2_calls",
    "wrist.projector": "wrist_projector_calls",
    "downstream": "downstream_calls",
}


class CameraInstrumentation:
    """Count every camera component and measure its CPU/device boundary."""

    def __init__(
        self,
        *,
        timer: SynchronizedQueryTimer | None = None,
        wall_clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.timer = timer
        self.wall_clock = wall_clock
        self.reset()

    def reset(self) -> None:
        self._counts = {field: 0 for field in _COUNT_FIELDS.values()}
        self._wall_ms: dict[str, float] = {}
        self._query_timer_active = False

    def start_query(self) -> None:
        if self._query_timer_active:
            raise RuntimeError("Camera query timing is already active")
        if self.timer is not None:
            self.timer.start()
            self._query_timer_active = True

    def finish_query(self) -> QueryTiming | None:
        if self.timer is None:
            return None
        if not self._query_timer_active:
            raise RuntimeError("Camera query timing is not active")
        try:
            return self.timer.finish()
        finally:
            self._query_timer_active = False

    def call(self, name: str, function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        if name not in _COUNT_FIELDS or name == "downstream":
            raise ValueError(f"Unsupported measured component: {name}")
        started = self.wall_clock()
        if self.timer is not None:
            self.timer.start_component(name)
        try:
            return function(*args, **kwargs)
        finally:
            if self.timer is not None:
                self.timer.stop_component(name)
            elapsed = (self.wall_clock() - started) * 1000.0
            field = _COUNT_FIELDS[name]
            self._counts[field] += 1
            self._wall_ms[name] = self._wall_ms.get(name, 0.0) + elapsed

    def measure_operation(self, name: str, operation: Callable[[], T]) -> T:
        if not name:
            raise ValueError("Operation name is required")
        started = self.wall_clock()
        if self.timer is not None:
            self.timer.start_component(name)
        try:
            return operation()
        finally:
            if self.timer is not None:
                self.timer.stop_component(name)
            self._wall_ms[name] = self._wall_ms.get(name, 0.0) + (
                self.wall_clock() - started
            ) * 1000.0

    def record_downstream(self) -> None:
        self._counts["downstream_calls"] += 1

    def snapshot(self) -> CameraWork:
        return CameraWork(
            scene_siglip_calls=self._counts["scene_siglip_calls"],
            scene_dinov2_calls=self._counts["scene_dinov2_calls"],
            scene_projector_calls=self._counts["scene_projector_calls"],
            wrist_siglip_calls=self._counts["wrist_siglip_calls"],
            wrist_dinov2_calls=self._counts["wrist_dinov2_calls"],
            wrist_projector_calls=self._counts["wrist_projector_calls"],
            downstream_calls=self._counts["downstream_calls"],
            component_wall_ms=dict(self._wall_ms),
        )
