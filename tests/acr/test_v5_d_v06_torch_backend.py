from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from savr.acr.v5_d_runtime import V5DProtocolViolation
from savr.acr.v5_d_torch_backend import OpenVLACoreFunctions
from savr.acr.v5_d_v06_torch_backend import V06PreCaptureWarmupRawCudaGraphCorePair


class Buffer:
    def __init__(self, value=0):
        self.value = value


def buffers():
    return {
        "wrist_pixels": Buffer(2),
        "wrist_tokens": Buffer(),
        "cached_scene_tokens": Buffer(11),
        "combined_tokens": Buffer(3),
        "prompt_embeddings": Buffer(4),
        "attention_mask": Buffer(5),
        "proprioception": Buffer(6),
        "normalized_actions": Buffer(),
    }


class FakeGraph:
    def __init__(self):
        self.call = None
        self.instantiated = False
        self.pool_token = object()

    def instantiate(self):
        self.instantiated = True

    def replay(self):
        assert self.instantiated and self.call is not None
        self.call()

    def pool(self):
        return self.pool_token


class FakeStream:
    _next = 1

    def __init__(self):
        self.cuda_stream = self._next
        FakeStream._next += 1
        self.device = "cuda:0"

    def wait_stream(self, other):
        del other


class FakeCuda:
    def __init__(self):
        self.active_graph = None
        self.synchronizations = 0
        self.graphs = []
        self.capture_calls = []
        self.default_stream = FakeStream()

    def Stream(self):
        return FakeStream()

    def current_stream(self):
        return self.default_stream

    @contextmanager
    def stream(self, stream):
        del stream
        yield

    def synchronize(self):
        self.synchronizations += 1

    def CUDAGraph(self):
        graph = FakeGraph()
        self.graphs.append(graph)
        return graph

    @contextmanager
    def graph(self, graph, *, stream, capture_error_mode, pool=None):
        assert capture_error_mode == "global"
        self.capture_calls.append((graph, stream, pool))
        self.active_graph = graph
        try:
            yield
        finally:
            self.active_graph = None

    def memory_allocated(self):
        return len(self.graphs) * 100

    def memory_reserved(self):
        return len(self.graphs) * 200


class FakeTorch:
    def __init__(self):
        self.cuda = FakeCuda()


def eager(torch, events, *, fail_downstream_call=None):
    counts = {"downstream": 0}

    def wrist(pixels, output):
        captured = torch.cuda.active_graph is not None
        events.append("wrist-capture" if captured else "wrist-warmup")

        def call():
            output.value = pixels.value * 10

        call()
        if captured:
            torch.cuda.active_graph.call = call

    def downstream(combined, embeddings, mask, proprio, output):
        captured = torch.cuda.active_graph is not None
        events.append("downstream-capture" if captured else "downstream-warmup")
        counts["downstream"] += 1
        if not captured and counts["downstream"] == fail_downstream_call:
            raise RuntimeError("injected downstream warm-up failure")

        def call():
            output.value = combined.value + embeddings.value + mask.value + proprio.value

        call()
        if captured:
            torch.cuda.active_graph.call = call

    return OpenVLACoreFunctions(wrist=wrist, downstream=downstream)


def make_pair(*, fail_downstream_call=None):
    torch = FakeTorch()
    events = []
    owned = buffers()

    def cat_into(destination, values):
        events.append("cat")
        destination.value = values[0].value + values[1].value

    pair = V06PreCaptureWarmupRawCudaGraphCorePair(
        torch_module=torch,
        eager=eager(torch, events, fail_downstream_call=fail_downstream_call),
        cat_into=cat_into,
    )
    return torch, pair, owned, events


def downstream(pair, owned):
    pair.downstream(
        owned["combined_tokens"],
        owned["prompt_embeddings"],
        owned["attention_mask"],
        owned["proprioception"],
        owned["normalized_actions"],
    )


def test_v06_warms_both_cores_before_capture_and_shares_pool() -> None:
    torch, pair, owned, events = make_pair()
    pair.prepare(owned)
    core_events = [event for event in events if event != "cat"]
    assert core_events == [
        "wrist-warmup",
        "wrist-warmup",
        "wrist-warmup",
        "downstream-warmup",
        "downstream-warmup",
        "downstream-warmup",
        "wrist-capture",
        "downstream-capture",
    ]
    assert len(torch.cuda.capture_calls) == 2
    first_graph, first_stream, first_pool = torch.cuda.capture_calls[0]
    _, second_stream, second_pool = torch.cuda.capture_calls[1]
    assert first_pool is None
    assert second_pool is first_graph.pool_token
    assert second_stream is first_stream
    assert pair._evidence["pre_capture_warmup_order"] == ["wrist", "downstream"]
    assert pair._evidence["capture_order"] == ["wrist", "downstream"]
    assert [row["stage"] for row in pair._evidence["memory_trace"]] == [
        "wrist-after-pre-capture-warmup",
        "downstream-after-pre-capture-warmup",
        "wrist-after-capture",
        "downstream-after-capture",
    ]


def test_v06_replays_only_in_capture_order_on_one_stream() -> None:
    torch, pair, owned, _ = make_pair()
    pair.prepare(owned)
    for _ in range(3):
        pair.wrist(owned["wrist_pixels"], owned["wrist_tokens"])
        downstream(pair, owned)

    pair.wrist(owned["wrist_pixels"], owned["wrist_tokens"])
    torch.cuda.default_stream = FakeStream()
    with pytest.raises(V5DProtocolViolation, match="stream changed"):
        downstream(pair, owned)


def test_v06_rejects_pointer_or_order_change_and_invalidates() -> None:
    _, pair, owned, _ = make_pair()
    pair.prepare(owned)
    with pytest.raises(V5DProtocolViolation, match="out of order"):
        downstream(pair, owned)
    with pytest.raises(V5DProtocolViolation, match="invalidated"):
        pair.wrist(owned["wrist_pixels"], owned["wrist_tokens"])

    _, pair, owned, _ = make_pair()
    pair.prepare(owned)
    with pytest.raises(V5DProtocolViolation, match="pointer changed"):
        pair.wrist(Buffer(2), owned["wrist_tokens"])


def test_v06_warmup_failure_stops_before_any_capture() -> None:
    torch, pair, owned, _ = make_pair(fail_downstream_call=2)
    with pytest.raises(RuntimeError, match="V5D_V06_PRECAPTURE_WARMUP_FAILED:downstream"):
        pair.prepare(owned)
    assert torch.cuda.capture_calls == []
    assert pair._invalidated is True
    assert pair._evidence["capture_order"] == []
    assert pair._evidence["memory_trace"][-1]["stage"] == ("downstream-pre-capture-warmup-failed")


def test_v06_backend_source_never_calls_empty_cache() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "src/savr/acr/v5_d_v06_torch_backend.py"
    ).read_text(encoding="utf-8")
    assert ".empty_cache(" not in source
