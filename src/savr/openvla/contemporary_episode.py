"""Instrumented original action loop; all query preparation remains timed."""
from collections import deque
import time
import numpy as np
from .execution_precision import execution_chunk_float32
from .specprune import EpisodeState
from .specprune_episode import CameraPair


def execute_measured_episode(evaluation, cfg, env, initial_state, instruction,
                             query, resize_size, *, episode_id, controller,
                             synchronize, clock=time.perf_counter, state=None):
    if (cfg.num_open_loop_steps != 8 or cfg.num_steps_wait != 10 or resize_size != 224
            or type(controller) is not bool):
        raise ValueError("original loop contract changed")
    state = EpisodeState() if state is None else state
    state.reset(episode_id)
    history, queue, records = deque(maxlen=6), deque(maxlen=8), []
    simulator_seconds = controller_seconds = nonquery_preparation_seconds = 0.0
    synchronize()
    started = clock()
    before = clock()
    env.reset()
    observation = env.set_init_state(initial_state.copy())
    for _ in range(10):
        observation, _, _, _ = env.step(evaluation.get_libero_dummy_action(cfg.model_family))
    simulator_seconds += clock() - before
    steps = replans = discarded = 0
    success = False
    for _ in range(evaluation.TASK_MAX_STEPS[cfg.task_suite_name]):
        # Start before evaluator resizing, history copying, crop/transfer/selection.
        # Non-query frame preparation is counted separately, never silently omitted.
        needs_query = not queue
        if needs_query:
            synchronize()
        before = clock()
        prepared, _ = evaluation.prepare_observation(observation, resize_size)
        history.append(CameraPair.capture(prepared, steps))
        if needs_query:
            previous = history[state.previous_frame_index(len(history))]
            query_count = state.query
            raw, details = query(prepared, instruction, previous, state)
            actions = execution_chunk_float32(raw)
            if state.query != query_count + 1:
                raise ValueError("each query must advance exactly one selector")
            queue.extend(actions.copy())
            synchronize()
            records.append(dict(query=state.query, seconds=clock() - before,
                                retained_visual_tokens=details["retained_visual_tokens"],
                                precise=details["precise"]))
        else:
            nonquery_preparation_seconds += clock() - before
        before = clock()
        action = np.asarray(evaluation.process_action(queue.popleft().copy(), cfg.model_family))
        if action.shape != (7,) or action.dtype != np.float32 or not np.isfinite(action).all():
            raise ValueError("processed action must be finite float32[7]")
        controller_seconds += clock() - before
        before = clock()
        observation, _, done, _ = env.step(action.tolist())
        simulator_seconds += clock() - before
        steps += 1
        before = clock()
        if controller and state.observe_executed_action(action, acting_steps=steps, queue_remaining=len(queue)):
            replans += 1
            discarded += len(queue)
            queue.clear()
        controller_seconds += clock() - before
        if done:
            success = True
            break
    synchronize()
    return dict(success=success, executed_steps=steps, policy_queries=len(records),
                replans=replans, discarded_actions=discarded, remaining_actions=len(queue),
                history_frames=len(history), controller_enabled=controller, queries=records,
                episode_seconds=clock() - started, simulator_seconds=simulator_seconds,
                controller_seconds=controller_seconds,
                nonquery_preparation_seconds=nonquery_preparation_seconds)
