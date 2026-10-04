"""Scoped, single-query Llama attention intervention; never edits runtime files."""

from contextvars import ContextVar


class LlamaAttentionDiagnostic:
    """Observe released bidirectional attention or impose an explicit causal mask.

    Only SDPA calls originating in the supplied Llama attention modules are
    intercepted. Vision attention is forwarded unchanged. No temporal cache or
    rectangular-query execution is supported by this diagnostic.
    """

    def __init__(self, torch, layers, mode):
        if mode not in {"original", "causal_control"}:
            raise ValueError("unknown attention diagnostic mode")
        self.torch, self.layers, self.mode = torch, list(layers), mode
        self.records = []
        self.active = ContextVar("attention_diagnostic_layer", default=None)
        self.patches = []
        self.original_sdpa = None

    def __enter__(self):
        if self.original_sdpa is not None:
            raise RuntimeError("attention diagnostic cannot be nested or reused")
        for layer in self.layers:
            if type(layer.self_attn).__name__ != "LlamaSdpaAttention":
                raise RuntimeError("expected original LlamaSdpaAttention at every layer")
        self.original_sdpa = self.torch.nn.functional.scaled_dot_product_attention
        self.torch.nn.functional.scaled_dot_product_attention = self._sdpa
        try:
            for index, layer in enumerate(self.layers):
                module = layer.self_attn
                had_local = "forward" in module.__dict__
                local = module.__dict__.get("forward")
                original = module.forward

                def wrapped(*args, _index=index, _original=original, **kwargs):
                    if kwargs.get("output_attentions", False):
                        raise RuntimeError("attention-output fallback is prohibited")
                    token = self.active.set(_index)
                    try:
                        return _original(*args, **kwargs)
                    finally:
                        self.active.reset(token)

                self.patches.append((module, had_local, local))
                module.forward = wrapped
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _sdpa(self, query, key, value, *args, **kwargs):
        index = self.active.get()
        if index is None:
            return self.original_sdpa(query, key, value, *args, **kwargs)
        if args or query.shape[-2] != key.shape[-2]:
            raise RuntimeError("only explicit, square, same-query attention is supported")
        if kwargs.get("is_causal", False):
            raise RuntimeError("reference SDPA is unexpectedly causal")
        if kwargs.get("dropout_p", 0.0) != 0.0:
            raise RuntimeError("inference attention dropout is not zero")
        mask = kwargs.get("attn_mask")
        torch = self.torch
        if mask is not None:
            if mask.ndim != 4 or mask.dtype == torch.bool:
                raise RuntimeError("expected released additive padding-only mask")
            if not torch.equal(mask, mask[:, :, -1:, :].expand_as(mask)):
                raise RuntimeError("reference mask is not bidirectional padding-only")
        if self.mode == "causal_control":
            if mask is None:
                kwargs["is_causal"] = True
            else:
                length = query.shape[-2]
                future = torch.ones((length, length), device=query.device, dtype=torch.bool).triu(1)
                kwargs["attn_mask"] = mask.masked_fill(future, torch.finfo(mask.dtype).min)
                kwargs["is_causal"] = False
        self.records.append({
            "layer": index, "sequence_length": int(query.shape[-2]),
            "reference_is_causal": False, "padding_mask_present": mask is not None,
            "effective_mode": self.mode,
        })
        return self.original_sdpa(query, key, value, **kwargs)

    def __exit__(self, exc_type, exc, traceback):
        for module, had_local, local in reversed(self.patches):
            if had_local:
                module.forward = local
            else:
                del module.forward
        self.patches.clear()
        if self.original_sdpa is not None:
            self.torch.nn.functional.scaled_dot_product_attention = self.original_sdpa
        if exc_type is None and [r["layer"] for r in self.records] != list(range(len(self.layers))):
            raise RuntimeError("attention diagnostic did not observe each layer exactly once")


def execute_episode(evaluation, cfg, env, initial_state, instruction, query, resize_size):
    """Released episode ordering, without its exception-to-task-failure conversion."""
    from collections import deque

    env.reset()
    observation = env.set_init_state(initial_state.copy())
    queue = deque(maxlen=cfg.num_open_loop_steps)
    if cfg.num_open_loop_steps != 8 or cfg.num_steps_wait != 10:
        raise RuntimeError("episode action/wait contract changed")
    for _ in range(cfg.num_steps_wait):
        observation, _, _, _ = env.step(evaluation.get_libero_dummy_action(cfg.model_family))
    steps, calls, success = 0, 0, False
    for _ in range(evaluation.TASK_MAX_STEPS[cfg.task_suite_name]):
        prepared, _image = evaluation.prepare_observation(observation, resize_size)
        if not queue:
            actions = query(prepared, instruction)
            if len(actions) != 8:
                raise RuntimeError("query must return eight actions")
            queue.extend(actions)
            calls += 1
        action = evaluation.process_action(queue.popleft().copy(), cfg.model_family)
        observation, _, done, _ = env.step(action.tolist())
        steps += 1
        if done:
            success = True
            break
    return {"success": success, "executed_steps": steps, "policy_queries": calls}
