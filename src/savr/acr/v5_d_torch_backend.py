"""Pinned PyTorch core implementations for the frozen V5-D executor.

PyTorch is injected instead of imported at module import time so the complete
pre-GPU contract remains testable on machines without the pinned CUDA stack.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from savr.acr.v5_d_runtime import V5DProtocolViolation


WRIST_CORE_ID = "acr-v5d-wrist-visual-core-v1"
DOWNSTREAM_CORE_ID = "acr-v5d-downstream-action-core-v1"


class CorePair(Protocol):
    backend_id: str

    def prepare(self, buffers: Mapping[str, Any]) -> None: ...

    def wrist(self, pixels: Any, output: Any) -> None: ...

    def downstream(
        self,
        combined: Any,
        embeddings: Any,
        attention_mask: Any,
        proprioception: Any,
        output: Any,
    ) -> None: ...


class TorchStaticTensorOperations:
    """V5-C static operations backed by the injected pinned torch module."""

    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module

    def allocate(self, shape: Sequence[int], *, dtype: str, device: str) -> Any:
        dtypes = {
            str(self.torch.bfloat16): self.torch.bfloat16,
            str(self.torch.int64): self.torch.int64,
            str(self.torch.bool): self.torch.bool,
        }
        if dtype not in dtypes:
            raise V5DProtocolViolation(f"Unsupported V5-D static dtype: {dtype}")
        if device != "cuda:0":
            raise V5DProtocolViolation("V5-D requires the single visible logical cuda:0")
        return self.torch.empty(tuple(shape), dtype=dtypes[dtype], device=device)

    @staticmethod
    def copy_(destination: Any, source: Any) -> None:
        destination.copy_(source)

    @staticmethod
    def cat_into(destination: Any, values: Sequence[Any], *, dim: int) -> None:
        if dim != 1 or len(values) != 2:
            raise V5DProtocolViolation("V5-D permits only scene-first camera concatenation")
        scene, wrist = values
        scene_length = int(scene.shape[1])
        wrist_length = int(wrist.shape[1])
        if int(destination.shape[1]) != scene_length + wrist_length:
            raise V5DProtocolViolation("V5-D combined-token shape changed")
        destination[:, :scene_length, :].copy_(scene)
        destination[:, scene_length:, :].copy_(wrist)


@dataclass(frozen=True)
class OpenVLACoreFunctions:
    wrist: Callable[[Any, Any], None]
    downstream: Callable[[Any, Any, Any, Any, Any], None]


def build_openvla_core_functions(
    *,
    torch_module: Any,
    model: Any,
    action_head: Any,
    proprio_projector: Any,
    all_actions_mask: Any,
    number_of_prompt_tokens: int,
) -> OpenVLACoreFunctions:
    """Build the exact pinned wrist and downstream tensor-only cores."""

    torch = torch_module
    if bool(getattr(model, "training", True)):
        raise V5DProtocolViolation("V5-D requires model evaluation mode")
    backbone = getattr(model, "vision_backbone", None)
    if backbone is None or not bool(getattr(backbone, "use_fused_vision_backbone", False)):
        raise V5DProtocolViolation("V5-D requires the pinned fused vision backbone")
    if int(backbone.get_num_patches()) != 256 or int(backbone.get_num_images_in_input()) != 2:
        raise V5DProtocolViolation("V5-D pinned vision shape changed")
    if tuple(all_actions_mask.shape) != (1, 79) or all_actions_mask.dtype != torch.bool:
        raise V5DProtocolViolation("V5-D action mask must be static bool [1,79]")
    if number_of_prompt_tokens != 21:
        raise V5DProtocolViolation("V5-D prompt-token boundary changed")
    if action_head is None or proprio_projector is None:
        raise V5DProtocolViolation("V5-D requires L1 action and proprioception modules")

    action_mask = all_actions_mask.unsqueeze(-1)

    def wrist_core(pixels: Any, output: Any) -> None:
        regular, fused = torch.split(pixels, [3, 3], dim=1)
        regular_features = backbone.featurizer(regular)
        fused_features = backbone.fused_featurizer(fused)
        combined_features = torch.cat((regular_features, fused_features), dim=2)
        output.copy_(model.projector(combined_features))

    def downstream_core(
        combined: Any,
        embeddings: Any,
        attention_mask: Any,
        proprioception: Any,
        output: Any,
    ) -> None:
        projected = model._process_proprio_features(combined, proprioception, proprio_projector)
        masked_embeddings = embeddings * ~action_mask
        multimodal_embeddings, multimodal_attention_mask = model._build_multimodal_attention(
            masked_embeddings, projected, attention_mask
        )
        language_output = model.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = language_output.hidden_states[-1]
        patch_tokens = 513
        start = patch_tokens + number_of_prompt_tokens
        hidden = last_hidden[:, start : start + 56, :]
        normalized = action_head.predict_action(hidden).reshape(1, 8, 7)
        output.copy_(normalized)

    return OpenVLACoreFunctions(wrist=wrist_core, downstream=downstream_core)


class TorchCompileCorePair:
    """Compile both frozen cores with the exact predeclared arguments."""

    backend_id = "torch-compile"

    def __init__(
        self,
        *,
        torch_module: Any,
        eager: OpenVLACoreFunctions,
        compile_function: Callable[..., Any] | None = None,
    ) -> None:
        compiler = compile_function or torch_module.compile
        arguments = {
            "backend": "inductor",
            "fullgraph": True,
            "dynamic": False,
            "mode": "reduce-overhead",
        }
        self._wrist = compiler(eager.wrist, **arguments)
        self._downstream = compiler(eager.downstream, **arguments)
        self._prepared = False

    def prepare(self, buffers: Mapping[str, Any]) -> None:
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
            raise V5DProtocolViolation("Compiled V5-D buffer set is incomplete")
        self._prepared = True

    def wrist(self, pixels: Any, output: Any) -> None:
        if not self._prepared:
            raise V5DProtocolViolation("Compiled wrist core used before preparation")
        self._wrist(pixels, output)

    def downstream(
        self,
        combined: Any,
        embeddings: Any,
        attention_mask: Any,
        proprioception: Any,
        output: Any,
    ) -> None:
        if not self._prepared:
            raise V5DProtocolViolation("Compiled downstream core used before preparation")
        self._downstream(combined, embeddings, attention_mask, proprioception, output)


class RawCudaGraphCorePair:
    """Capture and replay both cores against the V5-C owned static buffers."""

    backend_id = "raw-cudagraph"

    def __init__(
        self,
        *,
        torch_module: Any,
        eager: OpenVLACoreFunctions,
        cat_into: Callable[[Any, Sequence[Any]], None],
    ) -> None:
        self.torch = torch_module
        self.eager = eager
        self.cat_into = cat_into
        self._buffers: Mapping[str, Any] | None = None
        self._wrist_graph: Any = None
        self._downstream_graph: Any = None
        self.instantiation_modes: list[str] = []

    def _capture(self, call: Callable[[], None]) -> Any:
        torch = self.torch
        side_stream = torch.cuda.Stream()
        current = torch.cuda.current_stream()
        side_stream.wait_stream(current)
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                call()
        current.wait_stream(side_stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph,
            stream=side_stream,
            capture_error_mode="global",
        ):
            call()
        instantiate = getattr(graph, "instantiate", None)
        if callable(instantiate):
            instantiate()
            self.instantiation_modes.append("explicit")
        else:
            if not callable(getattr(graph, "replay", None)):
                raise V5DProtocolViolation("Captured raw graph is not replayable")
            self.instantiation_modes.append("implicit-capture-end")
        return graph

    def prepare(self, buffers: Mapping[str, Any]) -> None:
        if self._buffers is not None:
            raise V5DProtocolViolation("Raw V5-D graph pair is single-prepare")
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
            raise V5DProtocolViolation("Raw V5-D buffer set is incomplete")
        self._buffers = buffers
        self._wrist_graph = self._capture(
            lambda: self.eager.wrist(buffers["wrist_pixels"], buffers["wrist_tokens"])
        )
        self.cat_into(
            buffers["combined_tokens"],
            (buffers["cached_scene_tokens"], buffers["wrist_tokens"]),
        )
        self._downstream_graph = self._capture(
            lambda: self.eager.downstream(
                buffers["combined_tokens"],
                buffers["prompt_embeddings"],
                buffers["attention_mask"],
                buffers["proprioception"],
                buffers["normalized_actions"],
            )
        )

    def _require_identity(self, name: str, value: Any) -> None:
        if self._buffers is None or value is not self._buffers[name]:
            raise V5DProtocolViolation(f"Raw graph {name} pointer changed")

    def wrist(self, pixels: Any, output: Any) -> None:
        self._require_identity("wrist_pixels", pixels)
        self._require_identity("wrist_tokens", output)
        self._wrist_graph.replay()

    def downstream(
        self,
        combined: Any,
        embeddings: Any,
        attention_mask: Any,
        proprioception: Any,
        output: Any,
    ) -> None:
        for name, value in (
            ("combined_tokens", combined),
            ("prompt_embeddings", embeddings),
            ("attention_mask", attention_mask),
            ("proprioception", proprioception),
            ("normalized_actions", output),
        ):
            self._require_identity(name, value)
        self._downstream_graph.replay()
