"""V06 raw graphs with all eager warm-ups before either graph is retained."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from savr.acr.v5_d_runtime import V5DProtocolViolation
from savr.acr.v5_d_torch_backend import OpenVLACoreFunctions
from savr.acr.v5_d_v04_torch_backend import V04SharedPoolRawCudaGraphCorePair


_LAST_V06_EVIDENCE: dict[str, Any] | None = None


def last_v06_raw_graph_evidence() -> dict[str, Any] | None:
    if _LAST_V06_EVIDENCE is None:
        return None
    return {
        "pre_capture_warmup_order": list(_LAST_V06_EVIDENCE["pre_capture_warmup_order"]),
        "capture_order": list(_LAST_V06_EVIDENCE["capture_order"]),
        "memory_trace": [dict(row) for row in _LAST_V06_EVIDENCE["memory_trace"]],
        "empty_cache_calls": 0,
    }


class V06PreCaptureWarmupRawCudaGraphCorePair(V04SharedPoolRawCudaGraphCorePair):
    """Warm both cores first, then capture both into one ordered private pool."""

    _buffers: Mapping[str, Any] | None
    _capture_stream: Any

    def __init__(
        self,
        *,
        torch_module: Any,
        eager: OpenVLACoreFunctions,
        cat_into: Callable[[Any, Sequence[Any]], None],
    ) -> None:
        global _LAST_V06_EVIDENCE
        super().__init__(torch_module=torch_module, eager=eager, cat_into=cat_into)
        self._evidence["pre_capture_warmup_order"] = []
        _LAST_V06_EVIDENCE = self._evidence

    def _warmup(self, label: str, call: Callable[[], None]) -> None:
        torch = self.torch
        current = torch.cuda.current_stream()
        self._capture_stream.wait_stream(current)
        try:
            with torch.cuda.stream(self._capture_stream):
                for _ in range(3):
                    call()
            current.wait_stream(self._capture_stream)
            torch.cuda.synchronize()
            self._evidence["pre_capture_warmup_order"].append(label)
            self._record_memory(f"{label}-after-pre-capture-warmup")
        except BaseException as error:
            self._invalidated = True
            self._record_memory(f"{label}-pre-capture-warmup-failed")
            raise RuntimeError(
                f"V5D_V06_PRECAPTURE_WARMUP_FAILED:{label}:memory_trace="
                f"{self._evidence['memory_trace']}:{error}"
            ) from error

    def _capture_without_warmup(
        self, label: str, call: Callable[[], None], *, pool: Any = None
    ) -> Any:
        torch = self.torch
        current = torch.cuda.current_stream()
        self._capture_stream.wait_stream(current)
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
                    raise V5DProtocolViolation("Captured V06 raw graph is not replayable")
                self.instantiation_modes.append("implicit-capture-end")
            self._evidence["capture_order"].append(label)
            self._record_memory(f"{label}-after-capture")
        except BaseException as error:
            self._invalidated = True
            self._record_memory(f"{label}-capture-failed")
            raise RuntimeError(
                f"V5D_V06_GRAPH_CAPTURE_FAILED:{label}:memory_trace="
                f"{self._evidence['memory_trace']}:{error}"
            ) from error
        return graph

    def prepare(self, buffers: Mapping[str, Any]) -> None:
        if self._buffers is not None:
            raise V5DProtocolViolation("V06 raw graph pair is single-prepare")
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
            raise V5DProtocolViolation("V06 raw graph buffer set is incomplete")
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

        self._wrist_graph = self._capture_without_warmup("wrist", wrist_call)
        pool_method = getattr(self._wrist_graph, "pool", None)
        if not callable(pool_method):
            self._invalidated = True
            raise V5DProtocolViolation("Pinned V06 raw CUDA graph pool API is unavailable")
        shared_pool = pool_method()
        if shared_pool is None:
            self._invalidated = True
            raise V5DProtocolViolation("Pinned V06 raw CUDA graph pool token is unavailable")
        self.cat_into(
            buffers["combined_tokens"],
            (buffers["cached_scene_tokens"], buffers["wrist_tokens"]),
        )
        self._downstream_graph = self._capture_without_warmup(
            "downstream", downstream_call, pool=shared_pool
        )
