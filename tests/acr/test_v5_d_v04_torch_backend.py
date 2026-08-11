from __future__ import annotations

from contextlib import contextmanager

import pytest

from savr.acr.v5_d_runtime import V5DProtocolViolation
from savr.acr.v5_d_torch_backend import OpenVLACoreFunctions
from savr.acr.v5_d_v04_torch_backend import V04SharedPoolRawCudaGraphCorePair


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


def eager(torch):
    def wrist(pixels, output):
        def call():
            output.value = pixels.value * 10

        call()
        if torch.cuda.active_graph is not None:
            torch.cuda.active_graph.call = call

    def downstream(combined, embeddings, mask, proprio, output):
        def call():
            output.value = combined.value + embeddings.value + mask.value + proprio.value

        call()
        if torch.cuda.active_graph is not None:
            torch.cuda.active_graph.call = call

    return OpenVLACoreFunctions(wrist=wrist, downstream=downstream)


def pair_and_buffers():
    torch = FakeTorch()
    owned = buffers()
    pair = V04SharedPoolRawCudaGraphCorePair(
        torch_module=torch,
        eager=eager(torch),
        cat_into=lambda destination, values: setattr(
            destination, "value", values[0].value + values[1].value
        ),
    )
    pair.prepare(owned)
    return torch, pair, owned


def downstream(pair, owned):
    pair.downstream(
        owned["combined_tokens"],
        owned["prompt_embeddings"],
        owned["attention_mask"],
        owned["proprioception"],
        owned["normalized_actions"],
    )


def test_v04_uses_one_pool_and_one_capture_stream() -> None:
    torch, pair, owned = pair_and_buffers()
    assert len(torch.cuda.capture_calls) == 2
    first_graph, first_stream, first_pool = torch.cuda.capture_calls[0]
    _, second_stream, second_pool = torch.cuda.capture_calls[1]
    assert first_pool is None
    assert second_pool is first_graph.pool_token
    assert second_stream is first_stream
    assert [row["stage"] for row in pair._evidence["memory_trace"]] == [
        "wrist-after-warmup",
        "wrist-after-capture",
        "downstream-after-warmup",
        "downstream-after-capture",
    ]
    pair.wrist(owned["wrist_pixels"], owned["wrist_tokens"])
    downstream(pair, owned)


def test_v04_rejects_downstream_first_and_invalidates() -> None:
    _, pair, owned = pair_and_buffers()
    with pytest.raises(V5DProtocolViolation, match="out of order"):
        downstream(pair, owned)
    with pytest.raises(V5DProtocolViolation, match="invalidated"):
        pair.wrist(owned["wrist_pixels"], owned["wrist_tokens"])


def test_v04_rejects_wrist_twice_and_cross_stream() -> None:
    torch, pair, owned = pair_and_buffers()
    pair.wrist(owned["wrist_pixels"], owned["wrist_tokens"])
    with pytest.raises(V5DProtocolViolation, match="out of order"):
        pair.wrist(owned["wrist_pixels"], owned["wrist_tokens"])

    torch, pair, owned = pair_and_buffers()
    pair.wrist(owned["wrist_pixels"], owned["wrist_tokens"])
    torch.cuda.default_stream = FakeStream()
    with pytest.raises(V5DProtocolViolation, match="stream changed"):
        downstream(pair, owned)


def test_v04_allows_repeated_ordered_queries_on_one_stream() -> None:
    _, pair, owned = pair_and_buffers()
    for _ in range(3):
        pair.wrist(owned["wrist_pixels"], owned["wrist_tokens"])
        downstream(pair, owned)
