# PAIR-VLA Phase P0 Freeze

Date: 2026-08-28  
Protocol: `docs/PAIR_VLA_EXECUTION_PROTOCOL_V1.md`  
Protocol SHA-256: `145668e8757ab7a6e9ab88f51d0d1d4903de41fb7db14a5653e3e13c198d47c8`  
Machine-readable authority: `configs/pair/p0_freeze_v1.json`

## 1. Scope and claim

PAIR-VLA is a small, frozen router for provenance-aware reuse of downstream
visual K/V state in OpenVLA-OFT. It is trained from counterfactual
interventions that replace exactly the cache groups that would be stale in the
real recursive system. The expert label is the excess continuous action loss
caused by that actual-source intervention, not visual similarity or terminal
task success.

The primary claim is deliberately scoped:

> On the pinned OpenVLA-OFT/LIBERO stack, PAIR-VLA improves the
> reliability-efficiency frontier over dense inference and the strongest
> faithfully runnable same-stack cache comparator.

This is not a claim of formal robot safety, universal VLA generality, unseen
task generalization, training-free operation, or superiority to a method whose
official implementation is unavailable.

## 2. Novelty and closest-work decision

The primary-source search was refreshed through 2026-08-28 using
`actual-stale`, `source provenance`, `recursive cache`, `expert regret`,
`action-aware token compression`, and `learned VLA caching` combinations.
No located work combines all four defining elements:

1. exact per-group provenance from the recursively realized cache;
2. labels from actual-source stale interventions;
3. expert-relative continuous-action regret across recursive horizons; and
4. a calibrated set-level risk bound used to choose a measured-latency profile.

Action-JND is the closest conceptual neighbor. It learns token importance from
projected visual features and synthetic perturbation tolerance, but it does not
label the mixed-source cache state actually produced by recursive reuse. This
difference is scientifically meaningful only if P4 shows that exact-source,
multi-group labels are stable and improve prediction; those are frozen gates,
not assumed results.

| Work | Version/date audited | Public code/license | Backbone and label | Decision inputs | Timing/evaluation | P0 disposition |
|---|---|---|---|---|---|---|
| VLA-Cache | NeurIPS 2025; repo at `a490988` | Complete official repo; Apache-2.0 | OpenVLA/OFT; heuristic attention/similarity caching | Attention, similarity, cache state | 4 LIBERO suites x 10 tasks, episodes/task not stated; SIMPLER and 4 real tasks; success/FLOPs/CUDA latency/frequency | Strongest faithfully runnable primary cache comparator |
| Learning to Cache (LAC) | arXiv:2602.00686v2, 2026-08-26 | Linked repo `fd191e…` contains README only; no detected license | VLA caching learned with task gradients and optical flow | Learned differentiable signals | 4 LIBERO suites x 10 tasks, trial count not stated; 2 SIMPLER settings x 4 tasks; 4 real tasks; success/FLOPs/CUDA time | Paper-only; no faithful primary comparison |
| Action-JND | arXiv:2608.21247v1, 2026-08-21 | No linked implementation; arXiv license only | OpenVLA-OFT; synthetic perturbation tolerance under continuous weighted L1 | Projected features and learned token estimates | 4 LIBERO suites x 10 tasks on OpenVLA and OFT; adaptive plus 30/40/60/80% reuse and 25/50/75/87.5% pruning; episode count not stated; success/FLOPs/CUDA latency/frequency | Closest paper; optional clearly labeled reproduction only |
| Gated VLA-Cache | arXiv:2608.10824v1, 2026-08-11 | Project page code link disabled | Discrete top-logit margin gate | Top-1/top-2 logit margin | 4 suites x 10 tasks x 50 episodes = 2,000 episodes per method/backbone; OpenVLA and OFT; success/TFLOPs | Paper-only; discrete confidence is not principled for this continuous head |
| SpecPrune | Project-pinned `8091adc…` | Source present; missing license in audit | Token pruning proxy | Model salience | Existing project diagnostic | Not a distributable primary comparator |
| VLA-ADP | Project-pinned `d7094b…` | Source preflight passed; overlay required | Adaptive token pruning | Model proxy | Separate integration | Secondary only after technical validation |
| VLA-Pruner | Project-pinned `84d4b7…` | Source preflight passed; isolated stack | VLA pruning | Learned/pruning proxy | Separate stack | Not a same-stack primary comparison |

Primary records:

- VLA-Cache: <https://openreview.net/pdf?id=QZYZ0Xm58q> and
  <https://github.com/siyuhsu/vla-cache>
- LAC: <https://arxiv.org/html/2602.00686> and
  <https://github.com/JiahanFan/LAC>
- Action-JND: <https://arxiv.org/html/2608.21247>
- Gated VLA-Cache: <https://arxiv.org/html/2608.10824>

The search must be repeated in P9 because all three 2026 neighbors are recent.
Adjacent 2026 work on action-result caching and action self-evaluation was also
screened. It changes the cached object or the reliability task rather than
learning actual-source visual-K/V intervention regret, so it does not remove
the scoped novelty claim.

## 3. Authenticated stack

| Component | Frozen identity |
|---|---|
| Project | `0a274b6ea0c0748adabe386ffa0d61670c4ece7a` |
| OpenVLA-OFT | `e4287e94541f459edc4feabc4e181f537cd569a8` |
| LIBERO | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| VLA-Cache | `a4909880573868dee2769343d52e793c0341678b` |
| Checkpoint repository | `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10` |
| Checkpoint revision | `638918f3d1c2e43a39a8a20772bdb8b91835e4b7` |
| Original LIBERO data revision | `yifengzhu-hf/LIBERO-datasets@f13aa24a3da8c43c7225569f28c562979fa0e35a` |
| Diagnostic filtered data revision | `openvla/modified_libero_rlds@6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551` |
| Runtime | PyTorch 2.2.0+cu118; Transformers 4.47.0; CUDA build 11.8 |

The official OpenVLA-OFT paper used a custom Transformers 4.40.1 environment.
The project retains Transformers 4.47.0 because VLA-Cache and prior BRACE
integration were validated on that same stack. This is a disclosed same-stack
compatibility choice, not an assertion of bit-identical upstream software.

## 4. Exact objective and action semantics

Pinned OpenVLA-OFT source establishes:

- `NUM_ACTIONS_CHUNK = 8`, `ACTION_DIM = 7`;
- continuous hidden states are selected with
  `current_action_mask OR next_actions_mask`;
- the primary fine-tuning loss is `torch.nn.L1Loss` with mean reduction;
- actions use checkpoint `q99` bounds; and
- terminal padding must be removed by the valid-action mask.

For a contract of horizon B, the frozen expert label is

\[
R_B = \frac{1}{B}\sum_{b=1}^{B}
\left[
  \operatorname{L1}(a^*_{q+b},\hat a^{cache}_{q+b})-
  \operatorname{L1}(a^*_{q+b},\hat a^{dense}_{q+b})
\right],
\]

where each L1 is the mean over valid normalized continuous coordinates in the
8 by 7 action chunk. `positive_regret = max(R_B, 0)`.

The training label uses the normalized continuous gripper coordinate. The
execution diagnostic follows the official path: unnormalize; map the gripper
from [0,1] to [-1,1]; sign-binarize; then invert for LIBERO. Expert actions are
never router inputs.

## 5. Chronology and data freeze

`modified_libero_rlds/no_noops` filters near-zero actions. It cannot prove the
original eight-step spacing required by the deployed action-chunk policy, so it
is diagnostic-only.

P1 must acquire only the four original sequential HDF5 suite directories from
`yifengzhu-hf/LIBERO-datasets` at the frozen revision. Hosted byte counts are:

| Suite | Bytes |
|---|---:|
| LIBERO-10 | 13,730,608,904 |
| LIBERO-Goal | 6,373,112,875 |
| LIBERO-Object | 7,444,084,034 |
| LIBERO-Spatial | 6,237,050,764 |
| Total | 33,784,856,577 |

P1 transfer is capped at 34 GiB and new project-local storage at 50 GiB,
including unpacked source (40 GiB), index (2 GiB), and temporary material
(8 GiB). The larger LIBERO-90 directory is excluded.

P1 must authenticate both cameras, original trajectory and step identity,
proprioception, instruction, eight future expert actions, episode boundaries,
padding, no-op/filter identity, revision, and checkpoint normalization. Any
missing requirement rejects the trajectory; systemic failure stops P1 before
label construction.

## 6. Cache intervention freeze

- Patch grid: 16 by 16 per camera.
- Atomic spatial unit: non-overlapping 4 by 4 tile (16 tiles per camera).
- Cameras: primary and wrist.
- Onset layers: 2, 6, 9, and 11.
- Maximum atomic groups: 128 per query.
- Recursive horizons: 1, 2, and 4 queries.
- Maximum source age: 4 queries.
- Contract weights: `beta_b = 1/B`.
- Every group is resolved from exact SourceLedger query, layer, camera, patch,
  and K/V identity.
- Masks are suffix-safe, tile-aligned, and nested.
- Any unresolved identity, invalid nesting, or source age above four causes
  dense fallback.

Primary physical profiles are the six tile-aligned profiles in
`p0_freeze_v1.json`: D37-BAL, D50-BAL, D50-SCENE, D50-LATE, D59-BAL, and
D62-BAL. The historical token-level P2-D25 profile is diagnostic-only because
its budgets are not uniformly tile-aligned. P3 may reject profiles on timing,
memory, or validity, but may not invent a profile after seeing regret or task
success.

## 7. Router feature chronology

Allowed inputs are limited to values available before the cache decision:
exact provenance/age; group identity; current-versus-actual-source raw-patch
summaries; proprioception; model-produced prior actions; previous gripper
transition; contract state; retained prior salience; and compact instruction
information already computed by the pinned path.

The audit found one justified addition: the pinned BRACE/OpenVLA-OFT path calls
`_process_vision_features` before downstream K/V reuse. Therefore compact
current-versus-source projected-tile summaries are available without executing
the downstream computation PAIR proposes to skip. They are allowed, but their
summarization, CPU/GPU transfer, provenance, and router time are charged to
complete-cycle overhead in P3/P6.

The instruction summary is a deterministic 16-dimensional fixed projection of
the already-computed instruction representation. No current dense action,
future action/frame/state, task outcome, hidden dense pass, or expert previous
action is permitted.

## 8. Router and training freeze

The router is a DeepSets-style model:

1. train-split standardization;
2. small categorical embeddings;
3. shared group MLP, widths 64 and 32;
4. signed group-regret output plus group embedding;
5. sum, mean, and max aggregation;
6. concatenate global contract and state features;
7. set MLP, widths 64 and 32; and
8. signed contract regret, positive-regret q50/q90, and distortion outputs.

It uses GELU, no dropout, deterministic inference, fewer than 250,000 trainable
parameters, and no gradients through the VLA. AdamW uses learning rate 1e-3,
weight decay 1e-4, batches of 64 contracts, gradient clipping at 1.0, at most
200 epochs, patience 20, and minimum improvement 1e-4. Seeds are 17, 29, and
43. The selected seed has the lowest calibration composite; within one percent,
the smaller seed wins.

Loss weights are: signed contract Huber 1.0; signed group Huber 0.5; positive
q50 pinball 0.25; positive q90 pinball 0.5; and dense-distortion Huber 0.25.
Huber delta is one standardized target unit.

## 9. Splits, sampling, and protected populations

Complete trajectories are split within each task using SHA-256 and seed
20260828: 70% training, 15% calibration/model selection, and 15% locked offline
test. Deterministic largest-remainder rounding is used. Every derived record
inherits its trajectory split.

P4 sampling seed is 20260829. The cap is 400 anchors and 4,000 VLA forward
calls: 20% fresh/repeat controls, 30% atomic interventions, and 50% multi-group
contracts. Ten percent of non-controls are exact repeats. Immutable identities
include suite, task, trajectory, camera, layer, age, horizon, profile, transition
status, and the ordered actual-source mask.

Protected populations open only in this order: synthetic; timing; demonstration
regret; locked offline regret; development success; confirmatory success.
Outcome-free records use `schemas/pair/intervention_record.schema.json`, which
rejects success fields. Success appears only in terminal episode records. An
analyzer must refuse unblinding until both the expected record count and a
completed immutable run summary exist.

## 10. Gates and statistics

The risk grid is `[0.0005, 0.001, 0.002, 0.004]` normalized mean-L1 units.
Calibration selects the greatest P3-measured complete-cycle saving satisfying:
q90 coverage 85-95%, service at least 70%, total overhead at most 2% of dense
cycle time, net saving at least 10%, memory below 23 GiB, and all fallback
rules. Selection never uses simulator success.

P4 requires, among other frozen checks:

- Spearman point estimate at least 0.30 and one-sided 90% lower bound above
  0.15;
- at least 15% point reduction in positive-regret CVaR90 versus the matched
  best proxy, with a positive one-sided 90% lower bound;
- positive-regret prevalence between 5% and 80%;
- repeat noise median at most 1e-5 and p95 at most 1e-4;
- fresh-control action maximum absolute difference at most 2e-4 and regret
  magnitude at most 1e-5;
- no record contributing more than 20% of positive CVaR mass; and
- measurable source-aware interaction: residual at least twice repeat noise
  and at least 10% horizon-2/4 CVaR improvement.

Confirmatory inference uses two-sided 95% Wilson intervals for individual
success; a 20,000-replicate, suite-stratified paired bootstrap with equal suite
weighting and seed 20260831 for the primary difference; a one-sided 95% lower
bound above -2 percentage points; and exact McNemar as secondary analysis.
Latency uses a 20,000-replicate episode/query-contract block bootstrap. It must
show at least 10% point net saving and a one-sided 95% lower bound above zero.
The hierarchy is non-inferiority, latency, matched-comparator frontier, then
ablations.

P8 size is calculated before P8 outcomes using only P7 paired discordance and
its upper 95% confidence bound. The starting reference is 500 episodes per
suite per method; the minimum is 2,000 and maximum 3,000 per method. If 80%
power needs more than 3,000 per method, execution stops for advisor review.

## 11. Comparator rule

Primary methods are same-stack dense, optimized core-FR as a latency reference,
official or technically validated corrected VLA-Cache, and PAIR-VLA. A PAIR
point is compared to a cache point within two percentage points of measured
complete-cycle saving. Interpolation is allowed only between two predeclared
budgets that bracket the PAIR point; extrapolation is forbidden.

LAC, Action-JND, and Gated VLA-Cache remain paper-only unless a separately
validated, clearly labeled reproduction becomes possible. No missing
implementation is represented as an official baseline.

## 12. Resource and stop freeze

The server reports four 24,576-MiB TITAN RTX GPUs and about 389.6 GB free disk.
No allocation or process was inspected. Prior same-stack BRACE evidence peaked
at 18,419 MiB and achieved a physical complete-cycle saving signal, so a
one-GPU path is credible but not guaranteed.

Every GPU phase has a strict peak-memory stop at 23,552 MiB. P3 is capped at
800 timing queries, six hours, and 2 GiB artifacts. Later caps are fully
specified in the JSON freeze; the maximum new-storage exposure across data and
artifacts remains well below currently reported free disk. GPU IDs must be
recorded, and use requires separate phase approval and coordination.

Any OOM, nonfinite result, invalid timing, chronology/provenance/mask mismatch,
outcome leak, or unauthorized threshold change stops the phase. No later phase
can rescue a failed earlier gate.

## 13. Implementation map (planning only)

Existing components to reuse:

- `src/savr/brace/b3_openvla.py`: pinned OpenVLA/OFT bridge;
- `src/savr/brace/cache_adapter.py`: downstream cache application;
- `src/savr/brace/ledger.py`: exact SourceLedger provenance;
- `src/savr/brace/profiles.py`: physical profiles and nesting;
- `src/savr/brace/records.py`: immutable technical records;
- `src/savr/brace/sequence_map.py`: recursive source mapping; and
- existing raw-patch change utilities.

Planned new modules, which are not authorized in P0:

- `src/savr/pair/types.py`
- `src/savr/pair/chronology.py`
- `src/savr/pair/groups.py`
- `src/savr/pair/features.py`
- `src/savr/pair/interventions.py`
- `src/savr/pair/expert.py`
- `src/savr/pair/router.py`
- `src/savr/pair/calibration.py`
- `src/savr/pair/selector.py`
- `src/savr/pair/runtime.py`
- `src/savr/pair/records.py`
- `src/savr/pair/statistics.py`
- phase-specific runners and analyzers.

## 14. Freeze rule

The JSON freeze is authoritative where prose and machine-readable values
differ. P2 may expose a correctness contradiction, but any scientific change
after this point requires a versioned amendment, documented reason, and new
approval before the affected protected population is opened. P0 does not
authorize P1, downloads, GPUs, implementation, outcomes, commit, push, or local
synchronization.
