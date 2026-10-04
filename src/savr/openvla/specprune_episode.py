"""Episode bridge for the same-checkpoint SpecPrune-OFT adaptation.

Keep the authenticated original evaluator and single-query qualification intact.
History contains both evaluator-resized views BEFORE policy center cropping.
Only the selected historical pair is policy-preprocessed, once, at query time.
No image, decoder hidden state or K/V tensor is reused as a model input.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

from .specprune import EpisodeState


@dataclass(frozen=True)
class CameraPair:
    step: int
    scene: np.ndarray
    wrist: np.ndarray

    @classmethod
    def capture(cls, observation, step):
        images = []
        for key in ("full_image", "wrist_image"):
            image = np.asarray(observation[key])
            if image.shape != (224, 224, 3) or image.dtype != np.uint8:
                raise ValueError("expected evaluator-resized uint8 RGB camera pair")
            image = image.copy()
            image.flags.writeable = False
            images.append(image)
        return cls(step, *images)


def execute_specprune_episode(evaluation, cfg, env, initial_state, instruction,
                             query, resize_size, *, episode_id, state=None,
                             controller=True):
    """Verified action loop plus explicit history/controller state.

    query(observation, instruction, previous_pair, state) returns eight actions
    and must complete exactly one selector (incrementing state.query once).
    Any exception propagates as a technical stop, never a failed task record.
    controller=False retains the original eight-action queue schedule.
    """
    if (cfg.num_open_loop_steps != 8 or cfg.num_steps_wait != 10
            or resize_size != 224 or type(controller) is not bool):
        raise ValueError("episode action/wait/image/controller contract changed")
    state = EpisodeState() if state is None else state
    state.reset(episode_id)  # Reset even when re-running an identical episode ID.
    history, queue = deque(maxlen=6), deque(maxlen=8)
    env.reset()
    observation = env.set_init_state(initial_state.copy())
    for _ in range(10):
        observation, _, _, _ = env.step(evaluation.get_libero_dummy_action(cfg.model_family))
    steps = calls = replans = discarded = 0
    success = False
    for _ in range(evaluation.TASK_MAX_STEPS[cfg.task_suite_name]):
        prepared, _ = evaluation.prepare_observation(observation, resize_size)
        history.append(CameraPair.capture(prepared, steps))
        if not queue:
            previous = history[state.previous_frame_index(len(history))]
            before = state.query
            actions = np.asarray(query(prepared, instruction, previous, state))
            if actions.shape != (8, 7) or not np.isfinite(actions).all():
                raise ValueError("query must return eight finite seven-dimensional actions")
            if state.query != before + 1:
                raise ValueError("query must complete exactly one selector")
            queue.extend(actions.copy())
            calls += 1
        action = np.asarray(evaluation.process_action(queue.popleft().copy(), cfg.model_family))
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("invalid processed action")
        # Unlike the upstream comparator loop, execute ALL early actions too.
        observation, _, done, _ = env.step(action.tolist())
        steps += 1
        if controller and state.observe_executed_action(
                action, acting_steps=steps, queue_remaining=len(queue)):
            replans += 1
            discarded += len(queue)
            queue.clear()
        if done:
            success = True
            break
    return dict(success=success, executed_steps=steps, policy_queries=calls,
                replans=replans, discarded_actions=discarded,
                history_frames=len(history), controller_enabled=controller)


def prepare_query_camera_pairs(observation, previous, prepare_images, cfg):
    """Current and past RGB preparation, each once; return the original PIL pair.

    Current PIL objects are also supplied to the qualified semantic preparer.
    Do not center-crop those objects a second time. Historical pixels supply
    selection signals only, never the vision encoder input.
    """
    current = prepare_images([observation["full_image"].copy(),
                              observation["wrist_image"].copy()], cfg)
    past = prepare_images([previous.scene.copy(), previous.wrist.copy()], cfg)
    for pair in (current, past):
        if len(pair) != 2 or any(np.asarray(v).shape != (224, 224, 3)
                                or np.asarray(v).dtype != np.uint8 for v in pair):
            raise ValueError("policy camera preprocessing contract changed")
    return current, past


class SpecPruneEpisodeQuery:
    """Production query bridge; no loader, checkpoint edits or timing exclusions.

    Dense control uses the SAME direct decoder/head with selection disabled.
    The benchmark caller must time this entire call with device synchronization,
    plus separately report controller/history and complete episode costs.
    This new bridge still requires real multi-query qualification before use
    in a task-success or speed experiment.
    """
    def __init__(self, *, model, head, proprio, processor, cfg, utils,
                 instruction_indexer, enabled=True):
        if type(enabled) is not bool:
            raise ValueError("enabled must be boolean")
        self.model, self.head, self.proprio = model, head, proprio
        self.processor, self.cfg, self.utils = processor, cfg, utils
        self.instruction_indexer, self.enabled = instruction_indexer, enabled
        self.last_query = None

    def __call__(self, observation, instruction, previous, state):
        import torch
        from .official_semantics import (
            prepare_semantic_query, derive_official_layout, select_official_action_hidden,
        )
        from .specprune import low_change_indices, SpecPruneSelection, pruned_decoder_forward

        self.last_query = None
        if not state.started or any(m.training for m in (self.model, self.head, self.proprio)):
            raise ValueError("reset episode and inference-only components required")
        with torch.inference_mode():
            # Dense has no need for the extra historical-image preprocessing.
            if self.enabled:
                current, past = prepare_query_camera_pairs(
                    observation, previous, self.utils.prepare_images_for_vla, self.cfg)
            else:
                current = self.utils.prepare_images_for_vla(
                    [observation["full_image"].copy(), observation["wrist_image"].copy()], self.cfg)
            preparation_calls = 0

            def already_prepared(_images, cfg):
                nonlocal preparation_calls
                if cfg is not self.cfg or preparation_calls:
                    raise ValueError("semantic preparation must consume current views once")
                preparation_calls += 1
                return current

            prepared = prepare_semantic_query(
                torch_module=torch, np_module=np, model=self.model, processor=self.processor,
                proprio_projector=self.proprio, prepare_images=already_prepared,
                normalize_proprio=self.utils.normalize_proprio,
                instruction_indexer=self.instruction_indexer, cfg=self.cfg,
                raw_scene=observation["full_image"], raw_wrist=observation["wrist_image"],
                raw_state=observation["state"], instruction=instruction)
            if preparation_calls != 1:
                raise ValueError("current camera preparation was not consumed")
            layout = derive_official_layout(action_mask=prepared.action_mask, projected_tokens=513,
                                           instruction_token_indices=prepared.instruction_token_indices)
            embeddings, mask = self.model._build_multimodal_attention(
                prepared.input_embeddings * ~prepared.action_mask.unsqueeze(-1),
                prepared.projected_patches, prepared.attention_mask)
            if mask.ndim != 2 or not bool((mask == 1).all()):
                raise ValueError("only unpadded current queries supported")
            low = ()
            if self.enabled:
                budgets = (240, 236) if state.precise else (236, 230)
                low = tuple(np.concatenate([
                    low_change_indices(current[0], past[0], top_k=budgets[0], threshold=.986),
                    low_change_indices(current[1], past[1], top_k=budgets[1], threshold=.98,
                                       wrist=True)]).tolist())
            selector = SpecPruneSelection(layout, state, low, enabled=self.enabled,
                                          dynamic=self.enabled, device=embeddings.device)
            hidden, positions = pruned_decoder_forward(self.model.language_model.model, embeddings, selector)
            action_hidden = select_official_action_hidden(hidden, layout, positions)
            normalized = self.head.predict_action(action_hidden).reshape(8, 7).float().cpu().numpy()
            actions = np.asarray(self.model._unnormalize_actions(normalized, self.cfg.unnorm_key))
            if actions.shape != (8, 7) or not np.isfinite(normalized).all() or not np.isfinite(actions).all():
                raise ValueError("nonfinite or malformed action output")
            self.last_query = dict(query=state.query, previous_frame=previous.step,
                                   precise=state.precise, retained_visual_tokens=int(
                                       ((positions >= 1) & (positions <= 512)).sum()))
            return actions
