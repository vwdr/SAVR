# V5-D Backend Implementation and Execution Guide

Status: **V02 RECOVERY IMPLEMENTED AND VERIFIED; GPU NOT SELECTED**

Date: 2026-08-10

Normative protocol: `docs/ACR_V5_D_GPU_FEASIBILITY_PROTOCOL.md`

Base v01 machine freeze semantic SHA-256:
`f445cf5d1a5ec6877ebea46ccc3883a11a676b38cb33a711ee4b74baf22f53f8`

Resolved v02 experiment semantic SHA-256:
`4ae65dda537a5b6dcdf9abd34d79e0a9d7defee834a2a8cc2f7107a659f36076`

V02 preflight semantic SHA-256:
`d7c3ed40cc9d5760a846cb15c688fa5c776cbac8f243d948376d16e64427a695`

## 1. Purpose and authorization boundary

This checkpoint implements the frozen V5-D experiment without selecting a
GPU, loading the model, compiling/capturing CUDA code, or producing timing
output. It prepares the exact code that a later user-coordinated one-GPU phase
may execute.

V01 stopped before model load because LIBERO's first-use configuration prompt
was not initialized for non-interactive execution. V02 corrects that defect
with a canonical create-once run-local config and an outer pre-model technical
stop envelope. The implementation does not authorize the one-GPU phase. The
GPU-selection script contains an explicit coordination environment guard and
the project still requires the user to approve entry before that variable may
be set.

## 2. Added architecture

| Artifact | Responsibility |
|---|---|
| `src/savr/acr/v5_d_runtime.py` | Frozen query ledger, compiler/raw waterfall, cumulative resource envelope, restoration guard, and real mixed-dtype eager/static executors |
| `src/savr/acr/v5_d_recovery.py` | V02 overlay validation, canonical LIBERO config, path/symlink checks, create-once writes, and pre-model stop records |
| `src/savr/acr/v5_d_torch_backend.py` | Exact pinned wrist/downstream cores, `torch.compile` pair, and raw `CUDAGraph` pair |
| `scripts/select_acr_v5_d_gpu.py` | Three aggregate-only samples, frozen eligibility thresholds, lowest-index selection, immutable launch manifest |
| `scripts/prepare_acr_v5_d_libero_config.py` | Create and attest the v02 run-local config before upstream import |
| `scripts/run_acr_v5_d.py` | Pinned-source/hash checks, model setup, backend preparation, seven correctness queries, eight warm-ups, 96 balanced timed queries, restoration, and immutable evidence |
| `scripts/launch_acr_v5_d.sh` | Project-local offline caches, one visible GPU, compile-first process, permitted fresh-process raw transition, finalization |
| `scripts/analyze_acr_v5_d.py` | Deterministic 10,000-block paired bootstrap, ordering-bias analysis, and conjunctive gates |
| `scripts/verify_acr_v5_d.py` | Independent recomputation from raw records without trusting analyzer booleans |
| `scripts/finalize_acr_v5_d.py` | Two byte-identical analyzer executions and immutable independent verification |
| `scripts/verify_acr_v5_d_preflight.py` | Dependency-free deterministic pre-GPU implementation verification |
| `scripts/verify_acr_v5_d_v02_import.py` | Closed-stdin, CUDA-hidden pinned upstream import verification |

No V5-C controller, executor, integration, or evidence file was changed.

## 3. Exact tensor cores

### Wrist core

The wrist core receives current `[1,6,224,224]` bfloat16 pixels, separates the
three SigLIP and three DINOv2 channels, executes both pinned featurizers,
concatenates tower features, applies the pinned projector, and copies the
result into the owned `[1,256,4096]` wrist-token output.

### Downstream core

The downstream core receives scene-first `[1,512,4096]` tokens, current prompt
embeddings/mask, and current `[1,8]` proprioception. It appends the projected
proprio token, zeros action-token embeddings with the frozen `[1,79]` mask,
builds the multimodal sequence, executes the language model, slices exactly 56
action hidden states, applies the pinned L1 head, reshapes to `[1,8,7]`, and
copies into the owned normalized-action output.

CPU transfer, NumPy conversion, unnormalization, controller work,
preprocessing, input copies, timing events, hashing, finite scans, and evidence
writes remain outside both cores.

## 4. Real mixed-dtype correction

V5-C intentionally used a dependency-free fake tensor contract and represented
all buffers with one dtype. The real frozen interface contains:

- bfloat16 images, tokens, embeddings, proprioception, and actions;
- int64 prepared prompt IDs; and
- pinned int64 or bool attention masks.

Reusing the V5-C class directly would either allocate the prompt buffers as
bfloat16 or reject correct real tensors. V5-D therefore adds separate
`V5DEagerReuseExecutor` and `V5DStaticBufferReuseExecutor` subclasses that
preserve V5-C call order, lifecycle, identities, stable ownership, and failure
semantics while enforcing the real per-buffer dtypes. The V5-C source remains
unchanged, so its semantic evidence is intact.

## 5. Backend waterfall

The compile pair calls `torch.compile` twice with exactly:

```python
backend="inductor", fullgraph=True, dynamic=False, mode="reduce-overhead"
```

Two fixed-input preparation passes must yield two unique graphs, no graph
break, and no second-pass graph. Any valid pre-correctness technical failure
is recorded and may create one raw-transition permit.

Raw capture starts in a fresh Python process, revalidates the immutable permit,
source/checkpoint hashes, aggregate GPU eligibility, cumulative time, memory,
artifacts, and preparation launches, then captures both cores against the V5-C
owned pointers. Three side-stream warm-ups and one capture call per core count
as eight preparation launches.

Pinned PyTorch `2.2.0+cu118` has no public `CUDAGraph.instantiate()` method.
For that version, successful capture completion instantiates the graph
internally. The backend records `implicit-capture-end`; if a later pinned API
provides `instantiate()`, it calls and records `explicit`. Either route must
produce a replayable graph before correctness.

No raw transition is possible after one correctness/timing record. Mixed
backends and performance-based switching are impossible in the state machine.

## 6. Exact execution and evidence flow

Only after separate explicit GPU coordination, from `/home/ved/SAVR`:

```bash
SAVR_V5D_GPU_COORDINATION_APPROVED=1 \
  envs/openvla-oft/bin/python scripts/select_acr_v5_d_gpu.py

bash scripts/launch_acr_v5_d.sh
```

The first command records three aggregate samples ten seconds apart and
selects the lowest eligible index. It never requests compute-process or command
information. The second command obtains the physical ID/UUID only from the
immutable manifest, exposes one logical `cuda:0`, redirects every cache beneath
the run directory, enforces offline mode, and launches the frozen waterfall.

The runner writes every completed query separately before advancing. A
successful 111-query run is analyzed twice, compared byte-for-byte, and
independently verified. A negative statistical gate returns a nonzero terminal
status but preserves a valid complete result. A technical/invariant failure
preserves the attempt and stops.

## 7. Analyzer definitions

The analyzer uses the 24 permutation blocks as its resampling unit and seed
`20260810`. It recomputes:

- direct optimized-reuse/Batched-FR median wall ratio;
- weighted wall and total-CUDA ratios at reuse weight
  `0.34180622504322034`;
- optimized/eager sequential-CUDA ratio;
- weighted visual-CUDA reduction;
- refresh/Batched-FR wall ratio; and
- maximum position-specific median deviation across applicable paths/metrics.

No outlier deletion or alternative backend/statistic exists. The independent
verifier separately rebuilds the exact query labels, all 24 permutations,
bootstrap distributions, ordering rows, gate booleans, and disposition.

## 8. Remaining empirical uncertainty

Pre-GPU implementation evidence cannot show that either backend will compile
or capture, fit memory, preserve real values within tolerance, or meet timing
gates. Compiler behavior through the pinned 7B language model is particularly
uncertain. Those are exactly the questions V5-D is designed to answer.

Even a complete V5-D pass would permit only preparation of a later online
development protocol. It would not establish task success or a positive paper
result.
