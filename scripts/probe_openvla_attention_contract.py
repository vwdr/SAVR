#!/usr/bin/env python3
"""CPU-only behavioral probe of the installed Llama SDPA implementation.

Runs a tiny randomly initialized model, not OpenVLA or the simulator. Prints
JSON to stdout and writes no files. Use independently in both project runtimes.
The SDPA observer forwards the original call unchanged and is always restored.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time


def probe(execution: str) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("explicitly hide CUDA before starting this probe")

    import torch
    import transformers
    from transformers import LlamaConfig, LlamaModel
    from transformers.cache_utils import DynamicCache

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=32, attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    config.proportion_attn_var = None
    config.reusable_patches = None
    model = LlamaModel(config).to(device="cpu", dtype=torch.float32).eval()
    classes = [type(layer.self_attn).__name__ for layer in model.layers]
    if classes != ["LlamaSdpaAttention"] * 2:
        raise RuntimeError(f"unexpected attention classes: {classes}")
    source = Path(inspect.getfile(type(model.layers[0].self_attn))).resolve()
    if not source.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("attention source is outside the selected runtime")

    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    captures = []

    def observed_sdpa(query, key, value, *args, **kwargs):
        if args:
            raise RuntimeError("probe expects explicitly named SDPA options")
        if query.device.type != "cpu":
            raise RuntimeError("non-CPU tensor in CPU-only probe")
        mask = kwargs.get("attn_mask")
        causal = kwargs.get("is_causal", False)
        allowed = torch.ones(query.shape[-2], key.shape[-2], dtype=torch.bool)
        if causal:
            allowed &= torch.tril(torch.ones_like(allowed))
        if mask is not None:
            effective = mask[0, 0]
            allowed &= effective if effective.dtype == torch.bool else effective.eq(0)
        captures.append({
            "is_causal": bool(causal),
            "explicit_mask": mask is not None,
            "allowed_keys_by_query": allowed.to(torch.int8).tolist(),
        })
        return original_sdpa(query, key, value, **kwargs)

    ids = torch.tensor([[1, 5, 9, 13, 17, 21]], device="cpu")
    changed = ids.clone()
    changed[0, -1] = 85
    mask_all = torch.ones_like(ids)
    mask_pad = mask_all.clone()
    mask_pad[0, -1] = 0
    records = []
    outputs = {}

    def dense_components(tokens, mask, cache):
        # The compatibility fork's top-level forward unconditionally starts CUDA
        # timers. Bypass ONLY that top-level orchestration for this CPU probe.
        # Run its actual mask builder, embeddings, decoder layers, and final norm.
        # This is NOT a replacement policy evaluator or a latency measurement.
        hidden = model.embed_tokens(tokens)
        positions = torch.arange(tokens.shape[1], device="cpu")
        mask_args = {
            "attention_mask": mask, "input_tensor": hidden,
            "cache_position": positions, "past_seen_tokens": 0,
        }
        if "output_attentions" in inspect.signature(model._update_causal_mask).parameters:
            mask_args["output_attentions"] = False
        prepared_mask = model._update_causal_mask(**mask_args)
        past = DynamicCache() if cache else None
        for layer in model.layers:
            hidden = layer(
                hidden, attention_mask=prepared_mask,
                position_ids=positions.unsqueeze(0), past_key_value=past,
                output_attentions=False, use_cache=cache,
                cache_position=positions,
            )[0]
        return model.norm(hidden), past

    torch.nn.functional.scaled_dot_product_attention = observed_sdpa
    try:
        with torch.inference_mode():
            for cache in (False, True):
                for case, tokens, mask in (
                    ("all_valid", ids, mask_all),
                    ("changed_future_token", changed, mask_all),
                    ("right_padding", ids, mask_pad),
                    ("changed_padding_token", changed, mask_pad),
                ):
                    captures.clear()
                    if execution == "full_model":
                        output = model(
                            input_ids=tokens, attention_mask=mask,
                            use_cache=cache, output_attentions=False,
                            output_hidden_states=True, return_dict=True,
                        )
                        hidden, past = output.last_hidden_state, output.past_key_values
                    else:
                        hidden, past = dense_components(tokens, mask, cache)
                    if len(captures) != 2:
                        raise RuntimeError("did not capture both attention layers")
                    if not torch.isfinite(hidden).all():
                        raise RuntimeError("nonfinite hidden state")
                    if (past is not None) != cache:
                        raise RuntimeError("unexpected cache-production semantics")
                    outputs[(cache, case)] = hidden.clone()
                    records.append({
                        "case": case, "use_cache": cache,
                        "sdpa_layers": list(captures),
                    })
    finally:
        torch.nn.functional.scaled_dot_product_attention = original_sdpa

    def difference(left, right):
        # Exclude the changed final position itself.
        return float((left[:, :-1] - right[:, :-1]).abs().max())

    sensitivities = []
    for cache in (False, True):
        sensitivities.append({
            "use_cache": cache,
            "earlier_tokens_change_when_valid_future_token_changes": difference(
                outputs[(cache, "all_valid")],
                outputs[(cache, "changed_future_token")],
            ),
            "earlier_tokens_change_when_padding_changes": difference(
                outputs[(cache, "right_padding")],
                outputs[(cache, "changed_padding_token")],
            ),
        })
    return {
        "schema_version": "openvla-attention-contract-cpu-v2",
        "complete": True,
        "execution": execution,
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "attention_source": str(source),
        "attention_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "attention_classes": classes,
        "device": "cpu", "dtype": "float32",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "tiny_model_forwards": len(records),
        "hidden_state_sha256": {
            f"cache_{cache}_{case}": hashlib.sha256(
                hidden.contiguous().numpy().tobytes()
            ).hexdigest()
            for (cache, case), hidden in outputs.items()
        },
        "openvla_model_loads": 0, "simulator_episodes": 0,
        "weights_downloaded": False,
        "sdpa_observer_restored": (
            torch.nn.functional.scaled_dot_product_attention is original_sdpa
        ),
        "cache_production_hidden_difference": float((
            outputs[(False, "all_valid")] - outputs[(True, "all_valid")]
        ).abs().max()),
        "sensitivities": sensitivities,
        "calls": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execution", choices=("full_model", "dense_components"),
        default="full_model",
    )
    args = parser.parse_args()
    started = time.monotonic()
    try:
        result = probe(args.execution)
        result["elapsed_seconds"] = time.monotonic() - started
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except Exception as error:
        print(json.dumps({
            "complete": False, "error_type": type(error).__name__,
            "error": str(error), "elapsed_seconds": time.monotonic() - started,
        }, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
