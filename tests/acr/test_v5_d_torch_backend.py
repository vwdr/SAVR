from __future__ import annotations

from contextlib import contextmanager

import pytest

from savr.acr.v5_d_runtime import V5DProtocolViolation
from savr.acr.v5_d_torch_backend import (
    OpenVLACoreFunctions,
    RawCudaGraphCorePair,
    TorchCompileCorePair,
)


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


def eager():
    return OpenVLACoreFunctions(
        wrist=lambda pixels, output: setattr(output, "value", pixels.value * 10),
        downstream=lambda combined, embeddings, mask, proprio, output: setattr(
            output,
            "value",
            combined.value + embeddings.value + mask.value + proprio.value,
        ),
    )


def test_compile_backend_uses_exact_arguments_and_requires_prepare() -> None:
    calls = []

    def compiler(function, **kwargs):
        calls.append(kwargs)
        return function

    pair = TorchCompileCorePair(torch_module=object(), eager=eager(), compile_function=compiler)
    expected = {
        "backend": "inductor",
        "fullgraph": True,
        "dynamic": False,
        "mode": "reduce-overhead",
    }
    assert calls == [expected, expected]
    owned = buffers()
    with pytest.raises(V5DProtocolViolation, match="before preparation"):
        pair.wrist(owned["wrist_pixels"], owned["wrist_tokens"])
    pair.prepare(owned)
    pair.wrist(owned["wrist_pixels"], owned["wrist_tokens"])
    pair.downstream(
        owned["combined_tokens"],
        owned["prompt_embeddings"],
        owned["attention_mask"],
        owned["proprioception"],
        owned["normalized_actions"],
    )
    assert owned["wrist_tokens"].value == 20
    assert owned["normalized_actions"].value == 18


class FakeGraph:
    def __init__(self, cuda):
        self.cuda = cuda
        self.call = None
        self.instantiated = False

    def instantiate(self):
        self.instantiated = True

    def replay(self):
        assert self.instantiated and self.call is not None
        self.call()


class FakeStream:
    def wait_stream(self, other):
        del other


class FakeCuda:
    def __init__(self):
        self.active_graph = None
        self.synchronizations = 0
        self.graphs = []

    def Stream(self):
        return FakeStream()

    def current_stream(self):
        return FakeStream()

    @contextmanager
    def stream(self, stream):
        del stream
        yield

    def synchronize(self):
        self.synchronizations += 1

    def CUDAGraph(self):
        graph = FakeGraph(self)
        self.graphs.append(graph)
        return graph

    @contextmanager
    def graph(self, graph, *, stream, capture_error_mode):
        assert isinstance(stream, FakeStream)
        assert capture_error_mode == "global"
        self.active_graph = graph
        try:
            yield
        finally:
            self.active_graph = None


class FakeTorch:
    def __init__(self):
        self.cuda = FakeCuda()


def test_raw_graph_pair_captures_both_cores_and_rejects_pointer_drift() -> None:
    torch = FakeTorch()
    counts = {"wrist": 0, "downstream": 0}

    def wrist(pixels, output):
        def call():
            counts["wrist"] += 1
            output.value = pixels.value * 10

        call()
        if torch.cuda.active_graph is not None:
            torch.cuda.active_graph.call = call

    def downstream(combined, embeddings, mask, proprio, output):
        def call():
            counts["downstream"] += 1
            output.value = combined.value + embeddings.value + mask.value + proprio.value

        call()
        if torch.cuda.active_graph is not None:
            torch.cuda.active_graph.call = call

    owned = buffers()
    pair = RawCudaGraphCorePair(
        torch_module=torch,
        eager=OpenVLACoreFunctions(wrist=wrist, downstream=downstream),
        cat_into=lambda destination, values: setattr(
            destination, "value", values[0].value + values[1].value
        ),
    )
    pair.prepare(owned)
    assert counts == {"wrist": 4, "downstream": 4}
    assert torch.cuda.synchronizations == 2
    assert pair.instantiation_modes == ["explicit", "explicit"]
    pair.wrist(owned["wrist_pixels"], owned["wrist_tokens"])
    pair.downstream(
        owned["combined_tokens"],
        owned["prompt_embeddings"],
        owned["attention_mask"],
        owned["proprioception"],
        owned["normalized_actions"],
    )
    assert counts == {"wrist": 5, "downstream": 5}
    owned["wrist_pixels"].value = 7
    pair.wrist(owned["wrist_pixels"], owned["wrist_tokens"])
    assert owned["wrist_tokens"].value == 70
    with pytest.raises(V5DProtocolViolation, match="pointer changed"):
        pair.wrist(Buffer(9), owned["wrist_tokens"])
