# Contribution assessment and minimum useful validation

2026-09-21 UTC. Research/planning only. Supersedes the publication-readiness and
next-step recommendations in `PAPER_CONTRIBUTION_ASSESSMENT_V1.md`.
User decision: hold the advisor email and manuscript; establish the contribution
and minimum additional evidence first. No experiment, training or server access
was performed for this assessment. This document is NOT a GPU-run authorization.

## 1. Decision

We have a functioning accelerated policy, not a nonfunctional robot. We do not
yet have a demonstrated original positive-method contribution. Those statements
are compatible. A useful tradeoff can support research without dominating every
metric, but naming a tradeoff does not supply novelty or independent evidence.

The immediate priority is to resolve **comparative value of the existing fixed
current-frame compression**, not invent another corrector or draft a negative
paper under a positive title. A constrained-platform systems study is a candidate
contribution, not a conclusion: can the simple selector provide a competitive
success–cost point without adaptive-selection/history overhead on our platform?
If an appropriate adaptive baseline is no slower and at least as reliable, that
candidate loses its main justification. Simplicity alone is not enough.

The distinctive feature of our trained version is post-backbone action repair
using all current projected patches while leaving the large model frozen.
This is an identifiable design difference, not a verified first-in-literature
claim. Its useful incremental robot benefit is currently unestablished. Do not
make that module the positive centerpiece on present evidence.

## 2. Evidence already available

The independent local robot-copy verifier passed again: nine evidence hashes,
488 episodes, 10,120 calls, four policy counts and frozen decisions reconcile.
The six frozen gates remain unchanged; their failed conjunction remains failed.

| Existing study | Outcome | Scope |
|---|---|---|
| Robot pilot, states 1/2/3, 120 conditions | Dense 120, fixed-384 117, action-only 118, visual 117 successes; mean controlled times 1201.22, 1017.60, 1018.95, 1019.80 ms | Historically exposed development conditions, one seed |
| Fixed-token screen, state 0, 40 conditions | Dense 39/40, fixed-384 39/40, fixed-256 37/40; respective compression time reductions 15.32% and 32.50% | Separate development screen; do not pool with the pilot |
| Existing SpecPrune adaptation, state 0, 40 conditions | Dense 39/40 vs adaptation 32/40; 53.80% lower controlled mean query time | Genuine existing comparator evidence, not an exact paper reproduction or equal-speed comparison |
| Offline correction, 800 validation observations | Visual L1 5.27% below compression, 0.80% below action-only | No net visual robot-success gain in the pilot |

Sources: `CURRENT_FRAME_ROBOT_PILOT_RESULT_V1.md`,
`FIXED_COMPRESSION_SCREEN_RESULT_V1.md`,
`CONTEMPORARY_REFERENCE_RESULT_V2.md`, and
`CURRENT_FRAME_ADAPTER_FIT_RESULT_V1.md` with its gripper erratum.

The prior assessment incorrectly suggested no published comparator had been run.
The SpecPrune adaptation exists and must be acknowledged. It does not establish
superiority of our slower, more conservative point. Comparisons across the two
state-0 runs remain cross-run development comparisons, not a new jointly
randomized experiment. All states used for these results remain development.

The 15.29% reduction is query latency, approximately 1.180x speedup, not 15.29%
more successful robot work. Both cameras and the full backbone weights remain;
we have not demonstrated a smaller model or proportional memory saving.

## 3. Closest literature and the actual novelty boundary

Primary sources accessed 2026-09-21. Reported publication results are not our
reproductions. No cross-paper hardware-normalized ranking is asserted.

| Source | Relevant overlap | What it means for our contribution |
|---|---|---|
| [VLA-Pruner, v5](https://arxiv.org/html/2511.16449v5) | Adaptive semantic/action-guided visual pruning on OpenVLA-OFT; Table 1 reports 1.46x at 50% retention and 101.05% relative success; RTX 4090, 500 episodes/suite | Strong primary comparison. Fixed visual deletion is not new. Relative success above 100% is not absolute task success. At 25% retention its reported speedup is 1.80x, not approximately 2x; 1.99x at 12.5% comes with 88.27% relative success. These are different tradeoff points. |
| [TEAM-VLA](https://arxiv.org/html/2512.09927v1) | Current-observation-only token expansion and merging on OpenVLA-OFT, without historical-frame buffer | Current-frame-only operation is already established, not our novelty. Its experimental section notes borrowed baseline numbers and A100 hardware, so a table is not a matched TITAN comparison. |
| [SpecPrune-VLA](https://arxiv.org/html/2509.05614v2) | Action-aware selection, layer pruning and controller; OFT/LIBERO | Existing qualified local adaptation is useful context. Preserve documented adaptation and controller differences; never portray the local 32/40 as refuting its published result. |
| [Offline hidden-state recovery preprint](https://arxiv.org/html/2609.19579v1) | Recovers structurally pruned OFT using cached teacher targets and student LoRA/head updates | Compression recovery and cached offline teaching already exist. Our frozen-backbone output-only correction differs from their backbone adaptation, but difference is not evidence of advantage. This is a September 17 preprint, not independently reproduced or assumed peer-reviewed. |
| [ViTaR preprint](https://arxiv.org/abs/2608.15816) | Bounded residual corrections atop a frozen VLA, with tactile information | Frozen-backbone residual correction is not new in general. Different sensor/goal means it is conceptual prior work, not a direct speed comparator. |

The authors' [VLA-Pruner repository](https://github.com/MINT-SJTU/VLA-Pruner)
advertises OpenVLA-OFT code. That does not establish compatibility with our pinned
runtime; integration must be audited before any proposed launch. No code or
weights were cloned/downloaded for this assessment.

FAST (arXiv:2501.09747) concerns action tokenization, not a directly comparable
visual-token selector. EfficientVLA's CogACT/SIMPLER result is contextual, not
a matched comparator. Avoid using an unverified residual-paper repository ID as
the decisive novelty reference. This targeted review does not prove that no
paper uses our exact combination, nor that every possible paper framing fails.

## 4. Corrections to earlier publication claims

- Remove the repeated interpretation that executed gripper agreement worsened.
  Correct disagreement: compression 331/6400; both learned correctors 330/6400.
  The sign-at-zero diagnostic was wrong for executed gripper semantics.
- A shuffled-label control does not isolate the value of visual conditioning.
  The action-only comparison is essential; it currently shows a small offline
  difference and no aggregate robot advantage for the visual adapter.
- A completed pilot with no observed aggregate gain is not proof of an exact
  zero population effect. Do not call it a confirmed null or established mechanism.
- Good controls/provenance make research trustworthy; they are not automatically
  a novel methodology contribution.
- There is no universal rule requiring 500 episodes/suite times three seeds for
  every paper. Standard benchmark coverage is relevant to benchmark claims;
  training-seed replication and evaluation-seed replication answer different
  questions. Sample size must be justified by the claim, precision and resources.
- Do not infer unseen-task generalization from new initial states on the same
  trained tasks. Do not claim equivalence from a confidence interval containing 0.
- The exploratory gate failed at the tested logistic threshold, not every
  possible gate. Its frame-bootstrap interval and tie handling need correction
  before reporting inference; these diagnostics are not the current paper target.

## 5. Smallest decision-oriented next work, in order

### Step A: no-compute comparator specification (before a GPU request)

Prepare a pinned, read-only implementation-difference audit of VLA-Pruner OFT
against our runtime/checkpoint: attention, precision, two-camera input, original
positions, action readout, normalization, chunk execution, history reset,
horizons and timing boundaries. Reuse the existing dense/zero-pruning parity
tests. Do not replace checkpoint code or install upstream dependencies silently.

Specify one primary comparison: plain fixed-384 vs adaptive pruning at matched
75% visual retention. Because equal token counts need not mean equal compute
(pruning at different layers has different costs), also include the published
50%-retention adaptive setting as a competitive reference. Do not tune the
competitor only to a weak setting. If these settings do not bracket the useful
latency region, report that limitation instead of claiming matched latency.

The comparator audit must finish before we promise an execution time or authorize
an experiment. Official code availability is not a compatibility pass.

### Step B: one bounded development comparison, only after explicit approval

Proposed smallest useful screen: four arms — dense, existing fixed-384, adaptive
75% retention, adaptive 50% retention — on the same 40 already-consumed state-0
conditions, all tasks, with balanced order and separate native controls. This is
160 primary episodes, not a confirmatory sample. Rerun dense and fixed-384 in the
same session for this comparison rather than pooling historical numbers.

Include matched-input timing with warmups and repeated blocks, selection/history
overhead, model/head/copies, and separately labelled preprocessing/episode costs.
Record final executed chunks and observation hashes so technical discrepancies
can be localized. Capture memory by arm with explicit measurement definitions;
aggregate GPU memory alone does not establish method-specific memory savings.

Before launch freeze code/config hashes, identities, order, counts, accounting,
resource caps, parity checks, analysis, technical stops and decision table.
Use only ssh titan, /home/ved/SAVR and the coordinated GPU 0 after a fresh scoped
idle check. One run, no automatic retry. No training or new correction model.
This proposal does NOT authorize launch, implementation installs or downloads.

Interpret the full tradeoff, not only a favorable metric:

- Adaptive no slower with no worse observed success: no developmental support
  for a fixed-selector positive advantage; do not scale up just to obtain one.
- Fixed selection appears useful at a distinct cost/quality point: candidate
  systems finding, still requires uncertainty and independent confirmation.
- Tiny or inconsistent differences: inconclusive; no positive claim or automatic
  expansion. Timing alone cannot settle reliability.
- Technical mismatch or changed dense behavior: preserve evidence and stop.

These are new prospective triage rules, not replacements for the completed
adapter experiment's frozen gates. The comparator can defeat the candidate;
it is not a guaranteed route to a paper.

### Step C: confirmation only if Step B supplies a substantive reason

Freeze one claim and one operating point before new evidence: e.g., useful query
latency reduction under an explicitly justified allowed success loss. A success
tolerance is an application judgment, not chosen to accommodate the old 2.5 pp
loss. The manuscript must say tradeoff if quality drops, not equivalent accuracy.

Evaluate dense, fixed compression and the selected strong adaptive reference on
initial conditions not used in design or screening. Audit the historical exposure
ledger before opening them. Keep all four suites represented; do not choose only
the favorable long-horizon suite. Use task-clustered paired analysis, distinguish
simulator repeats from independent task/state units, and plan timing repetition
across sessions. Budget counts from precision/power simulations across plausible
discordance rates, not only the optimistic pilot. Define one fixed stopping rule
or a valid prespecified sequential design; never stop when the result looks good.

There is no honest exact minimum N yet because the claim/tolerance and new paired
discordance are not fixed. To illustrate scale, for independent paired differences
D in {-1,0,1}, Var(D)=q-d^2. A normal planning approximation at d near 0 is
N≈1.96² q/h² for a two-sided 95% half-width h. At q=.05–.10 and h=.025 this is
about 308–615 independent pairs; clustering and power can require more. This is
an illustration, NOT a valid final LIBERO sample-size calculation or launch plan.

If claiming a learned-corrector contribution instead, the old robot pilot does
not satisfy that claim. Matched data/compute action-only and simple residual
controls plus independently replicated fitting/evaluation would be needed.
Do not add those workloads to this faster fixed-compression assessment by default.

## 6. What to do now and what would justify the email

Immediate next artifact: Step A comparator-compatibility and prospective screen
specification. Do not start the manuscript, send the email or run Step B now.
The user preferred a more resolved update over another open-ended advisor query.

After comparative evidence exists, the email can state what useful difference
was actually demonstrated, its limitations and the proposed contribution. If no
positive distinction survives, say so rather than rename negative evidence.
A compelling empirical study is possible in principle, but is not automatically
the positive-solution contribution sought by the user or a guaranteed acceptance.

All prior reports/results remain available. Only this assessment and continuation
status/decision documents are updated locally. No server/GPU access, experiment,
training, manuscript, advisor message or GitHub push was performed.
