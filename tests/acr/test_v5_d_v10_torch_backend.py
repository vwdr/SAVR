from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from savr.acr.v5_d_runtime import V5DProtocolViolation
from savr.acr.v5_d_torch_backend import OpenVLACoreFunctions
from savr.acr.v5_d_v10_torch_backend import V10DownstreamOnlyCudaGraphCorePair


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

    def instantiate(self):
        self.instantiated = True

    def replay(self):
        assert self.instantiated and self.call is not None
        self.call()


class FakeStream:
    _next = 1

    def __init__(self):
        self.cuda_stream = self._next
        FakeStream._next += 1
        self.device = "cuda:0"

    def wait_stream(self, other):
        del other


class FakeCuda:
    def __init__(self, *, fail_capture=False):
        self.fail_capture = fail_capture
        self.active_graph = None
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
        return None

    def CUDAGraph(self):
        graph = FakeGraph()
        self.graphs.append(graph)
        return graph

    @contextmanager
    def graph(self, graph, *, stream, capture_error_mode):
        assert capture_error_mode == "global"
        self.capture_calls.append((graph, stream))
        self.active_graph = graph
        try:
            yield
            if self.fail_capture:
                raise RuntimeError("injected capture failure")
        finally:
            self.active_graph = None

    def memory_allocated(self):
        return len(self.graphs) * 100

    def memory_reserved(self):
        return len(self.graphs) * 200


class FakeTorch:
    def __init__(self, *, fail_capture=False):
        self.cuda = FakeCuda(fail_capture=fail_capture)


def eager(torch, events):
    def wrist(pixels, output):
        events.append("wrist-capture" if torch.cuda.active_graph else "wrist-eager")
        output.value = pixels.value * 10

    def downstream(combined, embeddings, mask, proprio, output):
        captured = torch.cuda.active_graph is not None
        events.append("downstream-capture" if captured else "downstream-eager")

        def call():
            output.value = combined.value + embeddings.value + mask.value + proprio.value

        call()
        if captured:
            torch.cuda.active_graph.call = call

    return OpenVLACoreFunctions(wrist=wrist, downstream=downstream)


def make_pair(*, fail_capture=False):
    torch = FakeTorch(fail_capture=fail_capture)
    events = []
    owned = buffers()

    def cat_into(destination, values):
        events.append("cat")
        destination.value = values[0].value + values[1].value

    pair = V10DownstreamOnlyCudaGraphCorePair(
        torch_module=torch,
        eager=eager(torch, events),
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


def test_v10_prepares_eager_wrist_and_exactly_one_downstream_graph() -> None:
    torch, pair, owned, events = make_pair()
    pair.prepare(owned)
    core_events = [event for event in events if event != "cat"]
    assert core_events == [
        "wrist-eager",
        "wrist-eager",
        "wrist-eager",
        "downstream-eager",
        "downstream-eager",
        "downstream-eager",
        "downstream-capture",
    ]
    assert len(torch.cuda.graphs) == 1
    assert len(torch.cuda.capture_calls) == 1
    assert pair._evidence["pre_capture_warmup_order"] == ["wrist", "downstream"]
    assert pair._evidence["pre_capture_warmup_calls"] == {"wrist": 3, "downstream": 3}
    assert pair._evidence["capture_order"] == ["downstream"]
    assert pair._evidence["wrist_capture_count"] == 0
    assert pair._evidence["shared_pool_api_calls"] == 0
    assert pair._evidence["preparation_labels"] == list(pair.preparation_labels())
    assert pair.preparation_labels() == (
        "raw-wrist-warmup-0",
        "raw-wrist-warmup-1",
        "raw-wrist-warmup-2",
        "raw-downstream-warmup-0",
        "raw-downstream-warmup-1",
        "raw-downstream-warmup-2",
        "raw-downstream-capture-0",
    )


def test_v10_live_query_runs_eager_wrist_then_downstream_replay() -> None:
    _, pair, owned, events = make_pair()
    pair.prepare(owned)
    before = len(events)
    pair.wrist(owned["wrist_pixels"], owned["wrist_tokens"])
    pair.cat_into(
        owned["combined_tokens"],
        (owned["cached_scene_tokens"], owned["wrist_tokens"]),
    )
    downstream(pair, owned)
    assert events[before:] == ["wrist-eager", "cat"]
    assert owned["wrist_tokens"].value == 20
    assert owned["normalized_actions"].value == 46


def test_v10_rejects_order_pointer_and_stream_drift() -> None:
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

    torch, pair, owned, _ = make_pair()
    pair.prepare(owned)
    pair.wrist(owned["wrist_pixels"], owned["wrist_tokens"])
    torch.cuda.default_stream = FakeStream()
    with pytest.raises(V5DProtocolViolation, match="stream changed"):
        downstream(pair, owned)


def test_v10_capture_failure_is_single_attempt_and_fail_closed() -> None:
    torch, pair, owned, _ = make_pair(fail_capture=True)
    with pytest.raises(RuntimeError, match="V5D_V10_GRAPH_CAPTURE_FAILED:downstream"):
        pair.prepare(owned)
    assert len(torch.cuda.capture_calls) == 1
    assert pair._evidence["capture_attempt_order"] == ["downstream"]
    assert pair._evidence["capture_order"] == []
    assert pair._evidence["graph_objects_retained"] == 0
    assert pair._invalidated is True


def test_v10_backend_source_has_no_pool_relaxation_or_empty_cache() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "src/savr/acr/v5_d_v10_torch_backend.py"
    ).read_text(encoding="utf-8")
    assert ".pool(" not in source
    assert 'capture_error_mode="global"' in source
    assert "thread_local" not in source
    assert "relaxed" not in source
    assert "empty_cache(" not in source
