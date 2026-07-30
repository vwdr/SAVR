"""Backend-agnostic synchronized query/component timing."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol


class EventBackend(Protocol):
    def synchronize(self) -> None: ...

    def record_event(self) -> Any: ...

    def elapsed_ms(self, start: Any, end: Any) -> float: ...


@dataclass(frozen=True)
class QueryTiming:
    wall_ms: float
    total_device_ms: float
    component_device_ms: dict[str, float]
    component_counts: dict[str, int]


class TorchCudaEventBackend:
    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module

    def synchronize(self) -> None:
        self.torch.cuda.synchronize()

    def record_event(self) -> Any:
        event = self.torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def elapsed_ms(self, start: Any, end: Any) -> float:
        return float(start.elapsed_time(end))


class SynchronizedQueryTimer:
    """Measure one query and nested named components using device events."""

    def __init__(
        self,
        backend: EventBackend,
        *,
        wall_clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.backend = backend
        self.wall_clock = wall_clock
        self._active = False
        self._total_start: Any = None
        self._wall_start = 0.0
        self._component_stacks: dict[str, list[Any]] = {}
        self._component_events: dict[str, list[tuple[Any, Any]]] = {}

    def start(self) -> None:
        if self._active:
            raise RuntimeError("Query timer is already active")
        self.backend.synchronize()
        self._wall_start = self.wall_clock()
        self._total_start = self.backend.record_event()
        self._component_stacks = {}
        self._component_events = {}
        self._active = True

    def start_component(self, name: str) -> None:
        if not self._active:
            raise RuntimeError("Query timer is not active")
        self._component_stacks.setdefault(name, []).append(
            self.backend.record_event()
        )

    def stop_component(self, name: str) -> None:
        stack = self._component_stacks.get(name)
        if not self._active or not stack:
            raise RuntimeError(f"Component timer stack underflow: {name}")
        end = self.backend.record_event()
        start = stack.pop()
        self._component_events.setdefault(name, []).append((start, end))

    def finish(self) -> QueryTiming:
        if not self._active:
            raise RuntimeError("Query timer is not active")
        if any(self._component_stacks.values()):
            raise RuntimeError("A component timing interval remains open")
        total_end = self.backend.record_event()
        self.backend.synchronize()
        wall_ms = (self.wall_clock() - self._wall_start) * 1000
        total_device_ms = self.backend.elapsed_ms(self._total_start, total_end)
        component_ms = {
            name: sum(
                self.backend.elapsed_ms(start, end) for start, end in intervals
            )
            for name, intervals in self._component_events.items()
        }
        component_counts = {
            name: len(intervals) for name, intervals in self._component_events.items()
        }
        self._active = False
        return QueryTiming(
            wall_ms=wall_ms,
            total_device_ms=total_device_ms,
            component_device_ms=component_ms,
            component_counts=component_counts,
        )


class ModuleTimingHooks:
    """Attach non-mutating module hooks to the currently active query timer."""

    def __init__(self, modules: dict[str, Any], timer: SynchronizedQueryTimer) -> None:
        self.timer = timer
        self.handles = []
        for name, module in modules.items():
            self.handles.append(
                module.register_forward_pre_hook(self._make_before(name))
            )
            self.handles.append(module.register_forward_hook(self._make_after(name)))

    def _make_before(self, name: str) -> Callable[..., None]:
        def before(_module: Any, _inputs: Any) -> None:
            self.timer.start_component(name)

        return before

    def _make_after(self, name: str) -> Callable[..., None]:
        def after(_module: Any, _inputs: Any, _output: Any) -> None:
            self.timer.stop_component(name)

        return after

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
