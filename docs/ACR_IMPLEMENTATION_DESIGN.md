# ACR Phase A0 Exact Implementation Design

**Date:** 2026-08-02

**Status:** design only; no ACR code exists

**Governing protocol:** `docs/ACR_EXECUTION_PROTOCOL_V1.md`

## 1. Design decision

Implement ACR as project-owned adapters around the pinned OpenVLA-OFT model.
Do not edit the upstream checkout or checkpoint. Preserve the original
`predict_action` path after projected visual tokens have been assembled.

The implementation boundary is the projected visual patch sequence before
current proprioception. The factored path will produce:

```text
[scene projected tokens][wrist projected tokens]
```

The unchanged upstream model will then append current proprioception and run
the complete language/action path.

## 2. Verified upstream factorization

Pinned OpenVLA-OFT revision:
`e4287e94541f459edc4feabc4e181f537cd569a8`.

Pinned source facts:

1. `experiments/robot/openvla_utils.py:745-770` builds an ordered image list
   beginning with `full_image`, appends wrist images, processes each image,
   and concatenates the resulting pixel tensors along channels.
2. `prismatic/extern/hf/modeling_prismatic.py:206-227` splits the concatenated
   tensor into one six-channel group per image, executes each image through
   the SigLIP and DINOv2 towers independently, joins the two tower features
   for that image, and concatenates image blocks along the patch dimension.
3. `modeling_prismatic.py:230-262` applies the projector independently to each
   token along the last dimension.
4. `modeling_prismatic.py:438-447` defines the upstream boundary as vision
   backbone followed by projector.
5. `modeling_prismatic.py:449-458` appends current proprioception only after
   projected visual tokens exist.
6. `modeling_prismatic.py:473-484` inserts the projected visual sequence after
   BOS without reordering it.
7. `modeling_prismatic.py:1002-1019` uses this boundary in real action
   prediction and computes `256 * num_images + proprio` patch positions.

Audited file hashes:

| File | SHA-256 |
|---|---|
| `third_party/openvla-oft/prismatic/extern/hf/modeling_prismatic.py` | `b5431a074c0025a12e46dc954a5e18d1d73477babb5ae42e3a12ab4b907f33a6` |
| `third_party/openvla-oft/prismatic/extern/hf/processing_prismatic.py` | `2474a5c3fdd4ff15234924636dc0ead6419c89f49699193be47dbb87510a5425` |
| `third_party/openvla-oft/experiments/robot/openvla_utils.py` | `eed754d7c5f9821aae2fe0531dbe01df8c11df0d5c79b4aeeb9bb4452124bdf5` |
| `third_party/openvla-oft/experiments/robot/libero/run_libero_eval.py` | `701a20b6ca2942aa36e0e7d567b25647268b936a23af79c5893534f10bec7802` |
| existing `src/savr/integration/openvla_oft.py` | `621df9db66d532a80087c0c3ca2d53d2174a9883a37ed14422d9529bf9231b6c` |

**Feasibility finding:** the two camera inputs are separable before their patch
blocks are concatenated. A project-owned adapter can therefore execute the
two tower/projector paths separately, cache only the scene result, concatenate
scene then wrist tokens, and pass the result into the original downstream
path.

**Unresolved correctness risk:** a projector invocation over one 256-token
block may select a different CUDA GEMM kernel from a single invocation over
the upstream 512-token sequence. Mathematical separability is established,
but bitwise equality is not assumed. Phase A3 must test exact equality. Any
failure stops the project under the protocol; no tolerance may be introduced
without an amendment.

## 3. Project-owned module plan

No file below is to be created until Phase A2 is authorized.

| Planned file | Responsibility |
|---|---|
| `src/savr/acr/__init__.py` | Public ACR types only |
| `src/savr/acr/types.py` | Frozen contexts, decisions, cache metadata, compact trace schemas |
| `src/savr/acr/signals.py` | Deterministic scene score, normalized translation, gripper transition summary |
| `src/savr/acr/controller.py` | SA-ACR, Scene-Visual, Scene-Periodic, and camera-factorized FR decisions |
| `src/savr/acr/cache.py` | One scene projected-token cache with strict compatibility and invalidation |
| `src/savr/acr/openvla_oft.py` | Exception-safe camera factorization and upstream method interception |
| `src/savr/acr/instrumentation.py` | Camera tower/projector counters, synchronized CUDA timings, hashes |
| `src/savr/acr/records.py` | Immutable attempt/query/episode records and reconciliation |
| `src/savr/acr/candidates.py` | Deterministic replay and candidate derivation |
| `src/savr/acr/statistics.py` | Frozen intervals, non-inferiority, Holm adjustment, task-aware summaries |
| `scripts/run_acr_correctness.py` | A3 bounded, no-rollout correctness harness |
| `scripts/run_acr_development.py` | A4-A5 immutable development runner |
| `scripts/run_acr_confirmation.py` | A6 one-shot frozen confirmation runner |
| `scripts/run_acr_transfer.py` | A7 no-retuning transfer runner |
| `scripts/run_acr_final.py` | A8 protected final runner with holdout lock |
| `tests/acr/` | Unit, schema, determinism, reconciliation, and statistics tests |

SAVR1-3 modules remain unchanged. The ACR package may reuse validated pure
signal helpers only when their exact semantics match the protocol; otherwise
it receives independent implementations and tests.

## 4. Camera factorization algorithm

Given the upstream pixel tensor shaped `[B, 12, H, W]` for two fused-backbone
images:

1. Require exactly two images, fused vision enabled, FiLM disabled, batch size
   one, and the pinned patch count.
2. Split channels into:
   - scene group `pixel_values[:, 0:6]`;
   - wrist group `pixel_values[:, 6:12]`.
3. For a camera group:
   - split into regular and fused RGB channels;
   - run the pinned regular featurizer;
   - run the pinned fused featurizer;
   - concatenate tower outputs along hidden dimension;
   - run the unmodified pinned projector;
   - verify `[B, 256, llm_dim]`, dtype, device, finite values, and ownership.
4. Assemble only as `torch.cat([scene_block, wrist_block], dim=1)`.
5. Return `[B, 512, llm_dim]` to the unchanged upstream caller.

Camera-factorized FR performs Step 3 for both cameras on every query. SA-ACR
may substitute only a compatible cached scene block; it always performs Step
3 for the wrist camera.

No pre-projector cache is permitted in Version 1. No token is removed, merged,
masked, reordered, or synthesized.

## 5. Adapter lifecycle and exception safety

The existing SAVR adapter intercepts `_process_vision_features`; ACR will use a
separate adapter to avoid changing historical behavior.

Per policy query:

1. Validate that an ACR context has begun.
2. Compute the decision using raw current observation metadata before model
   interception.
3. Acquire an adapter lock.
4. Save whether the model has an instance-level override for
   `_process_vision_features` and save its value.
5. Install one bound interception method.
6. Execute the otherwise-unmodified upstream query exactly once.
7. Require exactly one visual-boundary invocation.
8. Restore the original method in `finally`, including after exceptions.
9. On any exception or invariant failure, invalidate the scene cache and
   preserve an immutable failed-attempt record.

Nested or concurrent use of the same model instance is prohibited. The lock
protects the temporary method override, not model-level parallel inference.

## 6. Scene cache contract

The cache contains only:

- projected scene token tensor;
- immutable context identity;
- tensor shape, patch count, dtype, device, and model/checkpoint identities;
- last actual scene-refresh query index;
- reference `32 x 32` scene representation and 64 patch scores as needed;
- normalized EEF position at the last actual scene refresh;
- gripper transition context;
- exact cache age and consecutive-reuse counters;
- reference hashes sufficient to diagnose stale or mismatched data.

The tensor must be detached, inference-only, non-aliased with mutable
workspace tensors, and never serialized across episodes.

Any mismatch in task, instruction, episode/attempt, checkpoint, upstream
revision, preprocessing, center-crop flag, image order, number of images,
patch count, shape, dtype, device, action-head configuration, or controller
version invalidates the cache and forces refresh.

## 7. Controller evaluation order

For query index `t`, evaluate refresh reasons in this frozen order and preserve
all active reasons even when the first reason is sufficient:

1. missing/invalid/incompatible cache;
2. invalid or non-finite required metadata/signal;
3. query warm-up (`t < 2`);
4. scene image score `> gamma_scene`;
5. normalized scene-relative EEF translation `> gamma_pos`;
6. gripper transition veto;
7. scene age `>= H_scene`;
8. episode hard scene-reuse cap would be violated.

The exact equality rule is therefore reuse-safe at the thresholds: equality
does not trigger conditions 4 or 5, while equality triggers the horizon. CPU
boundary tests must cover just-below, equal, and just-above values.

Only an actual scene refresh changes the reference image, reference EEF
position, transition context, refresh index, and age origin. A wrist refresh
never changes scene-reference state.

## 8. Always-fresh wrist truth table

| Policy query state | Scene work | Wrist work | Downstream work |
|---|---|---|---|
| New context | fresh | fresh exactly once | current |
| SA-ACR scene refresh | fresh | fresh exactly once | current |
| SA-ACR scene reuse | zero tower/projector work; cached block copied/referenced | fresh exactly once | current |
| Scene-Visual reuse | zero scene work | fresh exactly once | current |
| Scene-Periodic reuse | zero scene work | fresh exactly once | current |
| Any invalid signal/cache | fresh (fail closed) | fresh exactly once | current |
| Exception before completion | cache invalidated | attempt retained | no state guessed on resume |

At least two independent checks enforce this: component call counters and a
hash of the current wrist image recorded on every query. The wrist hash is an
audit signal, never a gate.

## 9. Component instrumentation

Instrumentation must separately count and time:

- scene regular tower calls;
- scene fused tower calls;
- scene projector calls;
- wrist regular tower calls;
- wrist fused tower calls;
- wrist projector calls;
- concatenation/cache-copy overhead;
- controller/signal overhead;
- downstream language/action computation;
- total policy-query latency.

CUDA timing uses paired events on the selected device with explicit
synchronization at defined boundaries. CPU wall time is recorded separately.
No theoretical call-count saving may be reported as measured speedup.

For every scene reuse:

- scene tower calls must be zero;
- scene projector calls must be zero;
- wrist tower and projector calls must each be exactly one path;
- the assembled tensor must still contain 512 visual tokens;
- current proprioception must be appended downstream.

## 10. Compact FR development records

Each A4 upstream-FR query record must be sufficient to replay the controller
without a model, simulator, or outcome access:

- run, attempt, episode, suite, task, state, seed, query, and environment step;
- deterministic `32 x 32` scene representation or a lossless bounded encoding;
- all 64 scene patch changes relative to the replayed reference;
- top-four scene aggregate;
- raw and normalized eight-dimensional current proprioception;
- normalized EEF position and scene-relative translation;
- current action-chunk hash and fixed gripper transition summary;
- logged direction-reversal diagnostic;
- query/component timing and call counts;
- image, state, action, and context hashes;
- schema version and semantic configuration hash.

Task success is held at episode level and is not required to derive controller
thresholds. Replay code consumes only the frozen development population.

## 11. Record identity and reconciliation

Recommended immutable identities:

```text
attempt:  <run>/<policy>/<suite>/task-XX/state-XX/seed-X/attempt-XXXX
episode:  same prefix plus terminal episode record
query:    same prefix plus query-XXXXXX
event:    same prefix plus monotonically increasing event index
```

Every run must reconcile:

- scheduled attempts = terminal episodes + preserved technical failures;
- query count = decision records = downstream query completions or preserved
  query failures;
- wrist refresh count = query count;
- scene refresh count + scene reuse count = query count;
- component calls implied by decisions = measured component calls;
- no duplicate immutable identities;
- progress checkpoint and manifest hashes match the frozen configuration.

Resume may schedule only a never-started pairing. An incomplete attempt is
retained and receives a new attempt identity under the predeclared recovery
rule; it is never overwritten.

## 12. CPU verification plan for Phase A2

Tests required before any real model query:

1. camera channel split and scene-first/wrist-second token order;
2. exact block shapes and combined shape;
3. projector path called once per fresh camera;
4. zero scene calls on a mocked reuse;
5. exactly one wrist path under every decision;
6. current proprioception survives scene reuse;
7. cache identity and all mismatch invalidations;
8. every refresh reason and threshold boundary;
9. warm-up, horizon, hard cap, and reference-update semantics;
10. gripper binarization and transition veto;
11. invalid/non-finite values fail toward refresh;
12. interruption, exception restoration, and recovery identities;
13. deterministic candidate derivation, repeated byte identity, and
    tie-breaking;
14. immutable schemas and reconciliation failure cases;
15. statistical functions cross-checked against an independent implementation;
16. all existing SAVR tests remain unchanged and pass.

Mocks must resemble pinned tensor dimensions and metadata but may not load the
real checkpoint in A2.

## 13. Bounded A3 parity plan

At most 16 real-model policy queries and no simulator episodes:

1. Capture upstream FR projected tokens and action output.
2. Run camera-factorized FR on the byte-identical input.
3. Require bitwise equality for projected tokens and actions.
4. Change only the scene image; require only the scene pre-language block to
   change.
5. Change only the wrist image; require only the wrist block to change.
6. Force scene reuse; require zero scene tower/projector calls, one fresh wrist
   path, and current proprioception.
7. Inject context/shape/device/dtype failures; require fail-closed refresh.
8. Verify upstream and checkpoint hashes before and after.

Any failure is terminal for A3 pending a root-cause report and user-approved
protocol amendment.

## 14. Holdout lock

The final runner must refuse to construct Spatial/Object/Goal/LIBERO-10
states `10-49`, seed `7`, unless all are present and match frozen values:

- selected method/configuration identity;
- semantic configuration hash;
- statistical-plan hash;
- table-shell hash;
- confirmation and transfer gate records;
- explicit phase authorization;
- clean upstream and checkpoint verification.

Reserve seeds `17` and `27` require a separate authorization token and reason.

## 15. Source-boundary exit decision

**PASS for implementation feasibility.**

The exact per-camera source boundary exists without modifying upstream code.
Bitwise parity remains an empirical correctness gate in A3, not a fact inferred
from mathematical separability.
