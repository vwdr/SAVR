"""V04 shared-pool raw CUDA graphs with explicit lifetime enforcement."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from threading import Lock
from typing import Any

from savr.acr.v5_d_runtime import V5DProtocolViolation
from savr.acr.v5_d_torch_backend import OpenVLACoreFunctions


_LAST_EVIDENCE: dict[str, Any] | None = None


def last_raw_graph_evidence() -> dict[str, Any] | None:
    if _LAST_EVIDENCE is None:
        return None
    return {
        "shared_private_pool": True,
        "capture_order": list(_LAST_EVIDENCE["capture_order"]),
        "memory_trace": [dict(row) for row in _LAST_EVIDENCE["memory_trace"]],
    }


class V04SharedPoolRawCudaGraphCorePair:
    """Two sequential graphs sharing one pool and one enforced replay stream."""

    backend_id = "raw-cudagraph"

    def __init__(
        self,
        *,
        torch_module: Any,
        eager: OpenVLACoreFunctions,
        cat_into: Callable[[Any, Sequence[Any]], None],
    ) -> None:
        global _LAST_EVIDENCE
        self.torch = torch_module
        self.eager = eager
        self.cat_into = cat_into
        self._buffers: Mapping[str, Any] | None = None
        self._wrist_graph: Any = None
        self._downstream_graph: Any = None
        self._capture_stream: Any = None
        self._replay_stream_identity: tuple[str, int] | None = None
        self._awaiting_downstream = False
        self._invalidated = False
        self._replay_lock = Lock()
        self.instantiation_modes: list[str] = []
        self._evidence: dict[str, Any] = {"capture_order": [], "memory_trace": []}
        _LAST_EVIDENCE = self._evidence

    def _record_memory(self, stage: str) -> None:
        allocated = getattr(self.torch.cuda, "memory_allocated", None)
        reserved = getattr(self.torch.cuda, "memory_reserved", None)
        if callable(allocated) and callable(reserved):
            self._evidence["memory_trace"].append(
                {
                    "stage": stage,
                    "allocated_bytes": int(allocated()),
                    "reserved_bytes": int(reserved()),
                }
            )

    def _capture(self, label: str, call: Callable[[], None], *, pool: Any = None) -> Any:
        torch = self.torch
        current = torch.cuda.current_stream()
        self._capture_stream.wait_stream(current)
        with torch.cuda.stream(self._capture_stream):
            for _ in range(3):
                call()
        current.wait_stream(self._capture_stream)
        torch.cuda.synchronize()
        self._record_memory(f"{label}-after-warmup")
        graph = torch.cuda.CUDAGraph()
        arguments = {
            "stream": self._capture_stream,
            "capture_error_mode": "global",
        }
        if pool is not None:
            arguments["pool"] = pool
        try:
            with torch.cuda.graph(graph, **arguments):
                call()
            instantiate = getattr(graph, "instantiate", None)
            if callable(instantiate):
                instantiate()
                self.instantiation_modes.append("explicit")
            else:
                if not callable(getattr(graph, "replay", None)):
                    raise V5DProtocolViolation("Captured V04 raw graph is not replayable")
                self.instantiation_modes.append("implicit-capture-end")
            self._evidence["capture_order"].append(label)
            self._record_memory(f"{label}-after-capture")
        except BaseException as error:
            self._record_memory(f"{label}-capture-failed")
            raise RuntimeError(
                f"V5D_V04_GRAPH_CAPTURE_FAILED:{label}:memory_trace="
                f"{self._evidence['memory_trace']}:{error}"
            ) from error
        return graph

    def prepare(self, buffers: Mapping[str, Any]) -> None:
        if self._buffers is not None:
            raise V5DProtocolViolation("V04 raw graph pair is single-prepare")
        required = {
            "wrist_pixels",
            "wrist_tokens",
            "cached_scene_tokens",
            "combined_tokens",
            "prompt_embeddings",
            "attention_mask",
            "proprioception",
            "normalized_actions",
        }
        if not required <= set(buffers):
            raise V5DProtocolViolation("V04 raw graph buffer set is incomplete")
        self._buffers = buffers
        self._capture_stream = self.torch.cuda.Stream()
        self._wrist_graph = self._capture(
            "wrist", lambda: self.eager.wrist(buffers["wrist_pixels"], buffers["wrist_tokens"])
        )
        pool_method = getattr(self._wrist_graph, "pool", None)
        if not callable(pool_method):
            self._invalidated = True
            raise V5DProtocolViolation("Pinned V04 raw CUDA graph pool API is unavailable")
        shared_pool = pool_method()
        if shared_pool is None:
            self._invalidated = True
            raise V5DProtocolViolation("Pinned V04 raw CUDA graph pool token is unavailable")
        self.cat_into(
            buffers["combined_tokens"],
            (buffers["cached_scene_tokens"], buffers["wrist_tokens"]),
        )
        self._downstream_graph = self._capture(
            "downstream",
            lambda: self.eager.downstream(
                buffers["combined_tokens"],
                buffers["prompt_embeddings"],
                buffers["attention_mask"],
                buffers["proprioception"],
                buffers["normalized_actions"],
            ),
            pool=shared_pool,
        )

    def _require_identity(self, name: str, value: Any) -> None:
        if self._buffers is None or value is not self._buffers[name]:
            raise V5DProtocolViolation(f"V04 raw graph {name} pointer changed")

    def _stream_identity(self) -> tuple[str, int]:
        stream = self.torch.cuda.current_stream()
        pointer = getattr(stream, "cuda_stream", None)
        return str(getattr(stream, "device", "unknown")), int(id(stream) if pointer is None else pointer)

    def wrist(self, pixels: Any, output: Any) -> None:
        with self._replay_lock:
            if self._invalidated:
                raise V5DProtocolViolation("V04 shared-pool graph pair is invalidated")
            if self._wrist_graph is None or self._downstream_graph is None:
                raise V5DProtocolViolation("V04 shared-pool graph pair used before preparation")
            if self._awaiting_downstream:
                self._invalidated = True
                raise V5DProtocolViolation("V04 shared-pool wrist replayed out of order")
            self._require_identity("wrist_pixels", pixels)
            self._require_identity("wrist_tokens", output)
            identity = self._stream_identity()
            if self._replay_stream_identity is None:
                self._replay_stream_identity = identity
            elif identity != self._replay_stream_identity:
                self._invalidated = True
                raise V5DProtocolViolation("V04 shared-pool replay stream changed")
            try:
                self._wrist_graph.replay()
            except BaseException:
                self._invalidated = True
                raise
            self._awaiting_downstream = True

    def downstream(
        self,
        combined: Any,
        embeddings: Any,
        attention_mask: Any,
        proprioception: Any,
        output: Any,
    ) -> None:
        with self._replay_lock:
            if self._invalidated:
                raise V5DProtocolViolation("V04 shared-pool graph pair is invalidated")
            if not self._awaiting_downstream:
                self._invalidated = True
                raise V5DProtocolViolation("V04 shared-pool downstream replayed out of order")
            try:
                for name, value in (
                    ("combined_tokens", combined),
                    ("prompt_embeddings", embeddings),
                    ("attention_mask", attention_mask),
                    ("proprioception", proprioception),
                    ("normalized_actions", output),
                ):
                    self._require_identity(name, value)
                if self._stream_identity() != self._replay_stream_identity:
                    raise V5DProtocolViolation("V04 shared-pool replay stream changed")
                self._downstream_graph.replay()
            except BaseException:
                self._invalidated = True
                raise
            self._awaiting_downstream = False
