"""V10 hybrid backend: eager wrist core and one downstream CUDA graph."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from threading import Lock
from typing import Any

from savr.acr.v5_d_runtime import V5DProtocolViolation
from savr.acr.v5_d_torch_backend import OpenVLACoreFunctions


_LAST_V10_EVIDENCE: dict[str, Any] | None = None


def last_v10_hybrid_evidence() -> dict[str, Any] | None:
    if _LAST_V10_EVIDENCE is None:
        return None
    return deepcopy(_LAST_V10_EVIDENCE)


class V10DownstreamOnlyCudaGraphCorePair:
    """Run wrist eagerly and replay one downstream graph on owned buffers."""

    backend_id = "raw-cudagraph"

    def __init__(
        self,
        *,
        torch_module: Any,
        eager: OpenVLACoreFunctions,
        cat_into: Callable[[Any, Sequence[Any]], None],
    ) -> None:
        global _LAST_V10_EVIDENCE
        self.torch = torch_module
        self.eager = eager
        self.cat_into = cat_into
        self._buffers: Mapping[str, Any] | None = None
        self._downstream_graph: Any = None
        self._capture_stream: Any = None
        self._replay_stream_identity: tuple[str, int] | None = None
        self._awaiting_downstream = False
        self._invalidated = False
        self._replay_lock = Lock()
        self.instantiation_modes: list[str] = []
        self._evidence: dict[str, Any] = {
            "wrist_backend": "eager-static-buffer",
            "downstream_backend": "raw-cudagraph",
            "pre_capture_warmup_order": [],
            "pre_capture_warmup_calls": {"wrist": 0, "downstream": 0},
            "capture_attempt_order": [],
            "capture_order": [],
            "graph_objects_created": 0,
            "graph_objects_retained": 0,
            "wrist_capture_count": 0,
            "shared_pool_api_calls": 0,
            "supplied_pool_token": False,
            "empty_cache_calls": 0,
            "preparation_labels": list(self.preparation_labels()),
            "memory_trace": [],
        }
        _LAST_V10_EVIDENCE = self._evidence

    @staticmethod
    def preparation_labels() -> tuple[str, ...]:
        return (
            "raw-wrist-warmup-0",
            "raw-wrist-warmup-1",
            "raw-wrist-warmup-2",
            "raw-downstream-warmup-0",
            "raw-downstream-warmup-1",
            "raw-downstream-warmup-2",
            "raw-downstream-capture-0",
        )

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

    def _stream_identity(self) -> tuple[str, int]:
        stream = self.torch.cuda.current_stream()
        pointer = getattr(stream, "cuda_stream", None)
        return str(getattr(stream, "device", "unknown")), int(
            id(stream) if pointer is None else pointer
        )

    def _warmup(self, label: str, call: Callable[[], None]) -> None:
        current = self.torch.cuda.current_stream()
        self._capture_stream.wait_stream(current)
        try:
            with self.torch.cuda.stream(self._capture_stream):
                for _ in range(3):
                    call()
                    self._evidence["pre_capture_warmup_calls"][label] += 1
            current.wait_stream(self._capture_stream)
            self.torch.cuda.synchronize()
            self._evidence["pre_capture_warmup_order"].append(label)
            self._record_memory(f"{label}-after-pre-capture-warmup")
        except BaseException as error:
            self._invalidated = True
            self._record_memory(f"{label}-pre-capture-warmup-failed")
            raise RuntimeError(
                f"V5D_V10_PRECAPTURE_WARMUP_FAILED:{label}:memory_trace="
                f"{self._evidence['memory_trace']}:{error}"
            ) from error

    def _capture_downstream(self, call: Callable[[], None]) -> Any:
        current = self.torch.cuda.current_stream()
        self._capture_stream.wait_stream(current)
        graph = self.torch.cuda.CUDAGraph()
        self._evidence["graph_objects_created"] += 1
        self._evidence["capture_attempt_order"].append("downstream")
        try:
            with self.torch.cuda.graph(
                graph,
                stream=self._capture_stream,
                capture_error_mode="global",
            ):
                call()
            instantiate = getattr(graph, "instantiate", None)
            if callable(instantiate):
                instantiate()
                self.instantiation_modes.append("explicit")
            else:
                if not callable(getattr(graph, "replay", None)):
                    raise V5DProtocolViolation("Captured V10 downstream graph is not replayable")
                self.instantiation_modes.append("implicit-capture-end")
            self._evidence["capture_order"].append("downstream")
            self._evidence["graph_objects_retained"] = 1
            self._record_memory("downstream-after-capture")
            return graph
        except BaseException as error:
            self._invalidated = True
            self._record_memory("downstream-capture-failed")
            raise RuntimeError(
                "V5D_V10_GRAPH_CAPTURE_FAILED:downstream:memory_trace="
                f"{self._evidence['memory_trace']}:{error}"
            ) from error

    def prepare(self, buffers: Mapping[str, Any]) -> None:
        if self._buffers is not None:
            raise V5DProtocolViolation("V10 hybrid core pair is single-prepare")
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
            raise V5DProtocolViolation("V10 hybrid buffer set is incomplete")
        self._buffers = buffers
        self._capture_stream = self.torch.cuda.Stream()

        def wrist_call() -> None:
            self.eager.wrist(buffers["wrist_pixels"], buffers["wrist_tokens"])

        def downstream_call() -> None:
            self.eager.downstream(
                buffers["combined_tokens"],
                buffers["prompt_embeddings"],
                buffers["attention_mask"],
                buffers["proprioception"],
                buffers["normalized_actions"],
            )

        self._warmup("wrist", wrist_call)
        self.cat_into(
            buffers["combined_tokens"],
            (buffers["cached_scene_tokens"], buffers["wrist_tokens"]),
        )
        self._warmup("downstream", downstream_call)
        self._downstream_graph = self._capture_downstream(downstream_call)

    def _require_identity(self, name: str, value: Any) -> None:
        if self._buffers is None or value is not self._buffers[name]:
            self._invalidated = True
            raise V5DProtocolViolation(f"V10 hybrid {name} pointer changed")

    def wrist(self, pixels: Any, output: Any) -> None:
        with self._replay_lock:
            if self._invalidated:
                raise V5DProtocolViolation("V10 hybrid core pair is invalidated")
            if self._downstream_graph is None:
                raise V5DProtocolViolation("V10 hybrid core pair used before preparation")
            if self._awaiting_downstream:
                self._invalidated = True
                raise V5DProtocolViolation("V10 hybrid wrist called out of order")
            self._require_identity("wrist_pixels", pixels)
            self._require_identity("wrist_tokens", output)
            identity = self._stream_identity()
            if self._replay_stream_identity is None:
                self._replay_stream_identity = identity
            elif identity != self._replay_stream_identity:
                self._invalidated = True
                raise V5DProtocolViolation("V10 hybrid replay stream changed")
            try:
                self.eager.wrist(pixels, output)
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
                raise V5DProtocolViolation("V10 hybrid core pair is invalidated")
            if not self._awaiting_downstream:
                self._invalidated = True
                raise V5DProtocolViolation("V10 hybrid downstream called out of order")
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
                    raise V5DProtocolViolation("V10 hybrid replay stream changed")
                self._downstream_graph.replay()
            except BaseException:
                self._invalidated = True
                raise
            self._awaiting_downstream = False
