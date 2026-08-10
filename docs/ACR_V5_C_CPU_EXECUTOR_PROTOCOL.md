# ACR V5-C CPU Executor-Correctness Protocol

Status: **FROZEN BEFORE V5-C IMPLEMENTATION**

Authorization date: 2026-08-10

Machine freeze: `configs/acr/v5_c_cpu_executor_freeze.json`

## 1. Purpose

Implement and mechanically verify the project-owned static-buffer reuse
executor contract selected in
`docs/ACR_V5_C_EXECUTOR_RESEARCH_AND_DESIGN.md`. V5-C establishes software
correctness and lifecycle safety only. It performs no compilation, CUDA graph,
model, simulator, or timing experiment.

## 2. Immutable selected method

The controller remains exactly the V5-B selection:

- method: `v5-a100-b40`;
- controller: `acr-isolated-controller-v1`;
- scene threshold: `0.30046895424836606`;
- translation threshold: `0.685919037527938`;
- horizon: `1`;
- hard prefix reuse cap: `0.40`;
- warm-up: queries `0` and `1`;
- action-derived gripper-transition veto; and
- mandatory post-reuse refresh latch/cache-age agreement.

V5-C may not alter controller code, thresholds, decisions, reason order,
signals, cache semantics, or V5-B evidence.

## 3. Versioned executor identities

Implement new project-owned modules without changing legacy V1/V2/V3 paths:

- reference identity: `acr-reuse-executor-eager-v1`;
- static-plan identity: `acr-reuse-executor-static-v1`;
- integration identity: `ir-sa-acr-static-executor-v1`.

The exact filenames are frozen as:

- `src/savr/acr/reuse_executor.py`;
- `src/savr/acr/isolated_execution_adapter.py`;
- `tests/acr/test_reuse_executor.py`; and
- `tests/acr/test_isolated_execution_adapter.py`.

Exports may be added to `src/savr/acr/__init__.py`. New explicit executor reason
codes may be added to `src/savr/acr/types.py`; no existing reason may be
removed or reordered.

## 4. Executor contract

### Inputs

The executor receives only validated, current query tensors/metadata:

- current wrist pixels;
- compatible cached scene tokens;
- current prompt/input embeddings and masks;
- current proprioceptive tensor when enabled;
- frozen model/action-head handles; and
- complete compatibility context.

It must never receive episode success, task/state identity for decision-making,
or reusable actions.

### Outputs

It returns:

- current combined scene/wrist projected tokens for audit;
- current normalized action tensor;
- executor identity and lifecycle snapshot;
- physical/logical work counts; and
- prelaunch/launch/failure accounting.

CPU/NumPy action conversion remains outside the optimized core.

### Two cores

1. `wrist_visual_core`: wrist pixels → fresh projected wrist tokens.
2. `downstream_action_core`: combined current tokens and current auxiliary
   inputs → fresh normalized action tensor.

The static executor must invoke each core once per completed reuse query. It
must not invoke the scene visual encoder.

## 5. Static-buffer contract

The static executor owns all replay buffers. Preparation fixes shape, dtype,
device, and identities before use. Each run must:

1. verify the compatibility key before any core launch;
2. copy current caller values into owned input buffers;
3. invoke the prepared wrist core once;
4. combine the owned cached-scene and fresh wrist outputs in scene-first order;
5. invoke the prepared downstream core once;
6. return owned/current outputs without exposing stale prior-query values; and
7. update auditable counters only after successful completion.

Tests must prove that caller tensor object identity can change while the
executor-owned buffer identity remains stable and values update correctly.

No production execution path may hash, serialize, write files, allocate a new
large tensor, scan a complete tensor for auditing, or synchronize the device
inside the intended timed core.

## 6. Compatibility key

The immutable key contains:

- checkpoint ID;
- upstream revision;
- configuration/controller/executor IDs;
- preprocessing and action-head IDs;
- instruction SHA-256 and prompt/input shape;
- dtype and device;
- image height/width;
- patch count and projected dimension;
- wrist, cached-scene, embedding/mask, proprioception, and action shapes;
- model evaluation state; and
- FiLM/diffusion flags, both required to be false for this version.

No field may be omitted from equality/hash semantics. Cross-key execution is
rejected before launch.

## 7. Lifecycle and fail-closed behavior

Required states:

- `UNPREPARED`: no optimized execution allowed;
- `PREPARED`: buffers and callable identities validated;
- `ACTIVE`: exactly one non-reentrant execution is in progress; and
- `INVALIDATED`: no execution allowed until a fresh preparation.

Required behavior:

- nested or concurrent calls are rejected;
- reset cannot occur during `ACTIVE`;
- a prelaunch readiness/key mismatch produces `executor-unavailable`, forces
  refresh through the existing eager path, and does not consume a reuse;
- a failure after either core begins produces `executor-failure`, invalidates
  executor and scene cache, leaves the controller unobserved, and stops the
  query;
- no partial failure may silently retry the action query;
- episode exit restores every patched model method even after an exception;
- reset removes all episode bindings and stale outputs; and
- legacy adapters/executors remain byte-for-byte behaviorally unchanged.

## 8. CPU acceptance matrix

V5-C implementation is accepted only if tests prove:

1. exact frozen identities and selected controller configuration;
2. exact eager/static wrist output equivalence on deterministic inputs;
3. exact eager/static combined-token order and value equivalence;
4. exact eager/static normalized-action output equivalence;
5. one fresh wrist and downstream call, zero scene calls on reuse;
6. current proprioception and prompt inputs affect the current result;
7. stable owned-buffer identities across changed caller tensors;
8. no stale output when values change but shapes do not;
9. every compatibility-key field rejects mismatch before launch;
10. FiLM, diffusion, dynamic shape, training mode, missing cache, and invalid
    tensor metadata fail closed;
11. nested/concurrent use and reset-while-active are rejected;
12. prelaunch unavailability forces refresh without observing reuse;
13. postlaunch failure invalidates cache/executor and does not observe the
    controller;
14. exception-safe model-method restoration;
15. episode reset clears bindings/buffers/counters;
16. selected controller decisions match the existing batched adapter on the
    same synthetic trace;
17. maximum reuse streak remains one and prefix cap never exceeds 0.40;
18. no production hot-path hashing, serialization, filesystem I/O, or full-
    tensor audit scan;
19. legacy V3 adapter and V5 controller tests remain unchanged/passing; and
20. complete repository, lint, format, build, bootstrap, and pinned TITAN CPU
    verification pass.

## 9. Machine verification

Add a dependency-free deterministic verifier that exercises lifecycle and
call-order semantics without importing PyTorch. Its output must include:

- executor identities and compatibility-key digest;
- completed reference/static query count;
- core call counts;
- stable-buffer checks;
- parity checks;
- prelaunch/postlaunch failure checks;
- controller trace and maximum streak;
- legacy-separation result; and
- semantic SHA-256.

The verifier output becomes an immutable runtime record only after the full
acceptance matrix passes.

## 10. Exclusions

V5-C does not permit:

- `torch.compile`, CUDA graph capture/replay, CUDA kernels, or GPU selection;
- model loading/querying or simulator reset/episode;
- timing, memory, speed, CUDA-work, or task-success measurement;
- downloads or environment/package changes;
- threshold/controller/signal changes;
- action, hidden-state, or cross-episode cache reuse;
- upstream source modification;
- Goal/reserve/final-population access;
- manuscript modification; or
- modification/deletion of V3/V4/V5-A/V5-B evidence.

## 11. Resources

- GPUs: `0`.
- Model queries: `0`.
- Simulator episodes/resets: `0`.
- Downloads: `0`.
- New task outcomes: `0`.
- New artifact cap: `512 MiB`.
- CPU wall cap for verification: `1,800 seconds`.

All TITAN work remains inside `/home/ved/SAVR`.

## 12. Stop and next gate

Any parity, lifecycle, restoration, controller, cache, or repository failure
stops V5-C. Passing V5-C proves only a correct static execution plan. V5-D then
requires a separate frozen one-GPU protocol, explicit user GPU coordination,
real-tensor parity, memory/capture feasibility, and measured timing.
