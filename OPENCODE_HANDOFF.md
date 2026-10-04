# SAVR / current-frame correction — OpenCode continuation handoff

> **Publication addendum, 2026-10-03:** user authorized the current project
> snapshot on GitHub branch `agent/full-paper-audit`. Read
> `docs/GITHUB_PUBLICATION_SCOPE_2026-10-03.md` for scope/exclusions. This
> supersedes earlier no-push notes for this snapshot only. Read the master
> context and latest scientific checkpoint; no new experiment is authorized.

> **LATEST 2026-10-03 — read `SAVR_MASTER_CONTEXT.md` first.** It consolidates
> the complete history, corrected interpretations, repo navigation and current
> research decision. Both local independent robot/adaptive verifiers passed
> again. The September 26 screen is complete; no confirmation has launched or
> been frozen. The latest user direction is to prepare an advisor update asking
> for guidance on the unresolved original contribution; the email is drafted,
> NOT SENT. Older "running", "no advisor email", and automatic-next-step language
> below is historical. Do not launch from it. No server access or GPU work was
> performed in creating the master guide; last-recorded idle is not a fresh live
> process check. Preserve all safety instructions and immutable evidence.

> LATEST 2026-09-26: user accepted preparing an empirical-paper route, not a new
> pruning-method claim. Read docs/EMPIRICAL_PAPER_BLUEPRINT_V1.md and
> docs/EMPIRICAL_CONFIRMATION_DESIGN_V1.md. Proposed one400-condition, four-arm
> independent study, approximately16–20 GPU-hours, NOT LAUNCHED OR FROZEN.
> Complete exposure certification (including remote-only historical runs), exact
> interval tests, executable contract/config and resource checks before dispatch.
> The95 JSON configs match locally/remotely; inventory is NOT exposure proof.
> New docs/audit artifacts are local only. Do not overwrite with remote status.
> S1 remains completed and verified; no active monitor. No new training, changed
> selector, email or push. Older calls below to seek another research-scope
> decision are superseded by the user's empirical-scope approval; the rejection
> of backend-method novelty is NOT superseded.

> Latest contribution gate: read reports/BACKEND_PRESERVATION_CONTRIBUTION_AUDIT_V1.md.
> Separate score extraction with fast attention is prior art, including an
> explicit adaptation suggestion in VLA-Pruner v1. Do not claim it as our new
> method or launch the proposed confirmation on that premise. No new run is
> active. A narrower empirical question needs a deliberate scope decision.

> LATEST, 2026-09-26: S1 v03 is COMPLETED and verified, not running. Read
> reports/ADAPTIVE_SCREEN_V03_RESULT.md and PROJECT_STATUS.md's completion entry.
> Successes dense/fixed384/adaptive384/adaptive256 =39/39/38/39 of40; controlled
> reductions15.36%/6.76%/15.58%. No fixed advantage over adaptive256 demonstrated.
> Heartbeat deleted. Do not relaunch. No confirmation protocol is frozen and
> no confirmation has launched. Historical active notes below are superseded.

> Current checkpoint, 2026-09-26: qualification v03 PASSED; four-arm screen v03
> is dispatched as owned TITAN PID 3074656. Do NOT launch another worker or edit
> frozen sources/configs. Read reports/ADAPTIVE_SDPA_V03_RUNNING_CHECKPOINT.md
> and PROJECT_STATUS.md's September 26 entry before acting. Heartbeat
> monitor-adaptive-comparison-v03 is active. Keep outcomes blinded until exactly
> 168 episode and 240 timing records plus a complete immutable summary exist.
> These instructions supersede all historical next-step entries below. Preserve
> old evidence; never automatically retry or relaunch old output paths.

Updated 2026-09-21 UTC. This is a restart document for a new agent opened in the
local SAVR project. Read it before acting. It is not a new experiment authorization.

## Latest user decision — takes precedence over older next-step entries

2026-09-21 (later same day): the VLA-Pruner adaptive comparator port is written
and CPU-verified, and the two-stage launch (S0 real-checkpoint adaptive
qualification → S1 four-arm screen) is implemented and frozen. Evidence and
frozen protocol: `reports/VLAPRUNER_COMPARATOR_AUDIT_AND_SCREEN_SPEC_V2.md`
§6, `docs/ADAPTIVE_SCREEN_PROTOCOL_V1.md`, and — for approval —
`reports/ADAPTIVE_SCREEN_READINESS_V1.md`. The port
(`src/savr/openvla/vlapruner_adaptive.py`) and its tests
(`tests/openvla/test_vlapruner_adaptive.py`) pass 17/17 locally (incl.
source-parity vs AST-isolated release code at the two pinned blob SHA-1s) and,
on TITAN's pinned venv (CPU-only), **9 passed + 8 skipped at exit 0** (the 8
skips are the source-parity class because the approved upstream scratch fetch
is Mac-only by policy — behavior/model-level gates all pass on the pinned
environment; always report the TITAN count as 9/8, never "17/17"). The new
adaptive suites (`test_adaptive_qualification.py`, `test_adaptive_screen.py`)
pass 12/12 locally and on TITAN (CPU-only; the three added tests are the
tolerance-separation regression, missing-separated-maxima rejection, and the
analyzer's independent tolerance recomputation), with the fixed-qualification
and compression-screen suites re-verified green. A first GPU-approval request
was reviewed and **held** with one scientific correction, now implemented,
regression-tested, and re-frozen: the S0 adaptive-zero tolerance is calibrated
on zero-pruning (dense-input) parity rows only — `adaptive_zero_tolerance =
max(2 × zero_pruning_observed, 1e-6)` ≤ frozen ceiling `1e-3`; pruning-induced
steady differences are recorded separately (`pruning_shift_observed_max`) and
never calibrate, gate, or inflate the tolerance; the analyzer independently
recomputes the tolerance from the parity rows. The tolerance ceiling was not
increased. Step B is
**frozen but NOT authorized**: GPU launch, in-session timing, and any
experiment still require explicit user approval of the frozen protocol and
GPU-0 coordination. No GPU work has happened.

**Final bug-check (conditional approval asked for "check for bugs first"):**
the review caught one real bug — the S0 prune-boundary gate compared post-prune
**layer lengths** against the retained-visual count (384/256), but the port
always keeps the trailing non-visual tokens (`[0] + top_visual + [513..full)`),
so the true post-prune layer length is `full − (512 − kept)` (477/349 for
605-token inputs). As written it would technical-stop S0 on its first frame-1
mode. Fixed in `run_adaptive_qualification.py` (gate + numeric evidence
`full_tokens`/`post_prune_length`/`layer_lengths` on every mode row),
`reconcile` now independently recomputes the invariant, a regression test
covers it (`test_adaptive_qualification.py` is now 7 tests; adaptive suites =
13 locally), and the protocol/readiness docs were corrected. Re-frozen
(deterministic, twice-identical): S0 cfg `71d0976e…`, S1 cfg `89dc4190…`;
S0 worker `82fdd54e…`, S0 test `1bde8ad6…`, protocol doc `f51f0ff7…`; S1 pins
unchanged. Local: adaptive 13/13, port (`test_vlapruner_adaptive.py`) 17/17,
fixed 3+7 OK, S1↔S0 binding OK. TITAN CPU re-run and byte-identical resync of
the changed files are the next step before any GPU work.

2026-09-21: HOLD the advisor email and manuscript. The user asked to establish
the contribution and smallest additional validation first. Read
`reports/PAPER_CONTRIBUTION_AND_MINIMUM_VALIDATION_V2.md` before continuing.
V1's universal 500-episodes/three-seeds requirement and immediate manuscript
recommendation are superseded, not execution instructions.

The next task is a comparator-compatibility audit and prospective specification
for one matched development comparison of dense, fixed-384, and adaptive pruning
at 75% and 50% retention. This tests whether the existing simple approach offers
a useful constrained-platform tradeoff; it does not assume a positive result.
No GPU run, new fitting, source download/install, email, manuscript or push is
authorized by that plan. Any subsequent confirmation needs justified precision,
an exposure audit and a separately frozen protocol. Do not automatically reopen
the gating/corrector line or claim the offline pilot establishes a general null.

Existing SpecPrune comparator work must not be omitted: see
`reports/CONTEMPORARY_REFERENCE_RESULT_V2.md` (32/40 vs dense 39/40, 53.80% query
reduction). It was an adaptation at a different operating point, not a matched
comparison against our 384-token pilot or proof of defeating the publication.
This turn changed local documentation only; TITAN has not received these latest
OpenCode/Codex documentation updates. Local latest status is authoritative for
continuation; do not overwrite it with an older server copy.

Restart checkpoint 2026-09-21: restart verification passed (see `PROJECT_STATUS.md`
latest entry). Two offline diagnostics were run (CPU-only, saved evidence):
`reports/CURRENT_FRAME_CORRECTION_DIAGNOSTIC_V2.md` supersedes the coverage
diagnostic V1, which had a z-scoring bug (corrected gradient 9.5 pp, weak OOD
association). Confound controls (action-only comparison, mechanical bound) show
the near-base correction "harm" is non-visual-specific and does not explain the
robot failures; per-step deployment actions were not recorded. Cheap
deploy-available signals predict correction benefit only weakly (held-out AUC
0.599) and a logistic gate does not beat always-apply. The open decision is
unchanged in structure: choose (a) an honest compression efficiency–accuracy
study or (b) a narrowly justified correction change, but no cheap offline signal
justifies a conditional-correction gate, so any (b) test requires a new bounded
run that records per-step corrected actions; no new protocol is frozen.

Step A completed 2026-09-21, revised after user-approved released-source
inspection: `reports/VLAPRUNER_COMPARATOR_AUDIT_AND_SCREEN_SPEC_V2.md`
supersedes V1 (paper-only provisional; V1 kept as historical record). The five
V1 blockages are resolved at code level: (1) score aggregation = mean over all
heads at the prune layer's own attention (layer 3); semantic rows = full
prefill block `[0, action_token_start)`, action rows = the 56 placeholders; no
normalization/weights on the executed path; (2) MMDP = cosine distance over the
LLM *input embeddings* of the candidate union, model-dtype, first pick by
second-nearest distance then max-min; (3) layer arithmetic = prune after layer
index 3 (4 full layers + 28 pruned at `fastv_k=3`); EMA deque w=3, γ=0.8,
action vectors captured at layer 15, replacing current action scores from query
≥4; (4) warm-start = the first 3 queries after each episode reset run dense
(`fastv_r=0` forced) and are included in the release's latency averages — our
timing block must separate warm from steady state and cannot use two-frame
traces; (5) dependency surface = OFT pins torch 2.2.0 (matches ours) but a
vendored transformers 4.47.0 fork plus a non-vendored `experiments/robot/
vla_cache_utils` module (absent from the repo tree) are required by the release
entry point — port decision is therefore local translation (SpecPrune-style),
no upstream install. Release-vs-paper reconciliation: `fastv_r` is a pruning
ratio (retention = 1 − fastv_r); released OFT scripts evaluate 25%/12.5%/7.25%
retention, so the paper's 50% point and our matched 75% arm are config-internal
(`fastv_r` 0.5 / 0.25), not released-script points. Source was fetched source-
only (~0.5 MB) to an approved scratch dir outside the repo; nothing committed,
no weights. No launch is authorized; the four-arm draft screen in §6 of V2 is
planning only.

## 1. Start here: actual state and user intent

The user wants useful, original VLA-inference research that can support a
positive-results peer-reviewed paper. Professor Bo Yuan advised developing a
solution, not merely accumulating negative results. The user asked whether a genuine
efficiency–accuracy tradeoff could support a paper without dominating every metric. Separate
measured benefits from plausible future improvements. Never guarantee publication
or manufacture a positive interpretation.

**The most recent robot experiment is COMPLETE. Do not restart it.** Its visual
adapter did not improve overall success over compression alone. There is a real
approximately 15% query-time reduction from compression, with a small observed
success decrease. Whether that simpler contribution is sufficiently original and
competitive for a paper remains unresolved. Failing our particular pilot's
predeclared adapter gates is not a universal prohibition on publishing the study.

The latest user requests were to diagnose possible code/method problems, continue
working, and preserve a full handoff before Codex usage is exhausted. A read-only
audit found an **offline gripper-diagnostic reporting error**, described below.
It does not affect the robot actions, training loss, or observed success counts.
The new direction/protocol after this diagnosis has NOT yet been selected or frozen.

No experiment or transfer is left running by this work. The last owned worker
PID **2016420 exited normally**; that PID is historical, not a reusable target.
The Codex heartbeat `monitor-current-frame-robot-pilot` was deleted after completion.
OpenCode does not inherit Codex tools, automation, or chat context automatically.

## 2. Paths, authority and shared-server safety

Actual local project: **`/Users/veddwivedi/Documents/VLA/SAVR`**.
The older path `/Users/veddwivedi/Documents/SAVR` no longer exists. The OpenCode
project should be the actual local directory above, containing this file.

Remote project: **`/home/ved/SAVR`**, accessed **only through `ssh titan`** using
the user's existing SSH setup and ZeroTier connection. Never request passwords
or print private keys. Do not establish a different access route.

Read `AGENTS.md`. Its original bootstrap/scientific-status wording is historical;
later approved protocols and latest status establish completed work. Its server
safety boundaries still apply:

- All server file operations must remain inside `/home/ved/SAVR`. No unrelated
  university files, environments, jobs, user accounts, services or configuration.
- No sudo, installs, downloads, permission changes, cleanup, process termination,
  reallocation, second GPU, or automatic GPU retry.
- Aggregate telemetry for the previously coordinated GPU 0 is allowed; do not
  enumerate other users' processes or GPUs to acquire resources.
- GPU 0 UUID: `GPU-bb2451d6-2989-a112-5c18-8892943710e4`, TITAN RTX, 24,576 MiB.
  Historical permission is not proof it is currently free. A future approved run
  must repeat the bounded idle/identity checks and stop on resource conflict.
- Original runtime: `/home/ved/SAVR/envs/openvla-oft/bin/python`, Torch
  `2.2.0+cu118`, Transformers `4.40.1`, original Llama SDPA. Do not switch runtimes.
- CPU checks hide CUDA and disable bytecode/cache writes outside the project.
  Runtime-generated HF/Torch/TMP/Matplotlib/etc. caches must stay inside the project.
- Do not modify synced ChatGPT source material under the separate project mirror
  `/Users/veddwivedi/.codex/.chatgpt-projects/g-p-6938d001ab38819185befb0e6b3f4e86/sources/`.

## 3. Local / remote / Git workflow

1. Read local status, protocol, source and evidence. Make scoped local edits with
   a patch tool. Preserve user changes; do not wholesale replace directories.
2. Authenticate the exact destination files before syncing changes to TITAN.
   Use explicit files with `scp` or non-destructive `rsync`; never `--delete`.
3. Run CPU tests with the original remote runtime and CUDA hidden. Freeze new
   configuration, code identities, population, outcomes and caps BEFORE a new run.
4. GPU work, when explicitly within a newly approved bounded protocol, runs on
   TITAN, not the Mac. Retain exclusive output directories/logs and immutable
   evidence. Never overwrite, repair in place or silently repeat a failed run.
5. Copy completed evidence back to the local project and verify hashes. The user
   wants to review everything locally. Preserve logs/configs/source provenance.
6. GitHub push is NOT currently authorized. Remote is `https://github.com/vwdr/SAVR.git`.
   Local branch at handoff: `agent/full-paper-audit`. Much recent work is untracked;
   GitHub is not the up-to-date source of truth. Do not reset, clean, checkout over,
   or pull into this dirty tree. Existing unrelated modified file
   `src/savr/brace/b3_openvla.py` belongs to prior/user work: preserve it.
   Do not infer or change repository visibility.

Local code editing + scoped remote execution + verified local evidence copies is
the current workflow. No need to reinstall anything or clone a second repository.

## 4. Read these first, in order

1. This file and `AGENTS.md`.
2. The TOP of `PROJECT_STATUS.md`, then latest entries in `docs/DECISIONS.md`.
   Older sections explicitly preserved as launch records can say “running” or
   “next”; they are historical, not instructions to relaunch completed experiments.
3. `reports/CURRENT_FRAME_ROBOT_PILOT_RESULT_V1.md` — verified robot results.
4. `reports/CURRENT_FRAME_GRIPPER_DIAGNOSTIC_ERRATUM_V1.md` — latest correction.
5. `docs/CURRENT_FRAME_ROBOT_PILOT_V1.md` and its frozen configuration.
6. `docs/CURRENT_FRAME_LEARNING_PILOT_V1.md`,
   `docs/CURRENT_FRAME_ADAPTER_FIT_V1.md`,
   `reports/CURRENT_FRAME_ADAPTER_FIT_RESULT_V1.md` (read with the erratum).
7. For broader context, `docs/POSITIVE_SOLUTION_COMPREHENSIVE_AUDIT_2026-09-08.md`
   and `docs/STREAMLINED_METHOD_PILOT_PLAN_V1.md`. These are historical research
   motivation, NOT proof of novelty or blanket permission to execute old phases.

Do not read the entire months-long archive before making progress. Use targeted
sources as questions arise. Ignore paper/poster work specific to MIT URTC for now.

## 5. How the project reached this point

- SAVR tried training-free reuse of the whole two-camera visual representation.
  Permissive reuse harmed success; conservative refresh mostly eliminated reuse.
  The primary study included 1,160 episodes plus a separate 50-episode timing pilot.
- Asymmetric camera refresh kept the wrist camera fresh. One matched comparison
  preserved observed 67/70 success with 25.24% scene reuse but no latency gain
  versus batched dense inference (1190.97 vs 1188.28 ms/query).
- BRACE / PAIR / intervening designs explored finer reuse, selection and routing.
  Some technical attempts failed; other completed scientific tests did not support
  the intended reliable acceleration. Read the actual reports if comparing them:
  `reports/BRACE_B3_V05_REPORT.md`, `reports/PAIR_P4B_REPORT.md` and associated
  frozen configs. Do not treat every technical failure as evidence against an idea.
- CAC proposed learned correction of recursively stale K/V effects. Its corrected
  substrate test obtained 0/120 versus dense 68/120 under that older setup and
  stopped before adapter learning. See
  `reports/CAC_C1H_S6_RECOVERY01_GATE_H_STOP.md`.
- Subsequent attention/runtime/semantic work established the current original
  OpenVLA-OFT path. Do not pool the earlier low dense baseline with today's 120/120
  result or attribute all historical failure to caching without qualifying setup.
- We moved to **current-frame token compression plus learned action correction**.
  This is NOT temporal caching, mixed-age K/V repair or a renamed SAVR wrapper.

## 6. Current architecture and training (actual implementation)

Backbone: frozen OpenVLA-OFT 7B checkpoint,
`checkpoints/openvla-7b-oft-libero-four-suite` on TITAN.
Sources: `third_party/openvla-oft` and `third_party/LIBERO`, inside the project.
Do not fetch new versions or weights. Source/checkpoint identities are authenticated
through existing configurations and their predecessor chain.

Every query fully encodes both current camera images (512 projected patches).
The compressed transformer uses **384 actual retained visual tokens**, selected
spatially before its first layer. Original positional indices and all action
readout positions are preserved; no stale states or past-K/V cache is used.
Dense retains all 512. The released action head predicts eight 7-dimensional actions.

The small corrector predicts a residual added to the base eight-action chunk:
`corrected_action = compressed_action + learned_residual`.
Inputs: 8x4096 penultimate released-head features; 8x7 base actions; 8-dimensional
robot state; 4096-dimensional pooled instruction; and, for visual correction,
all 512x4096 current projected patches. No future observation enters the corrector.
Width 256, two blocks, eight attention heads, zero-initialized output.
Action-only ablation omits visual context/cross-attention (3,485,959 parameters).
Visual adapter has 5,071,879 parameters. This is not a parameter-matched ablation.

Key code:

- `src/savr/openvla/fixed_compression.py`: physical token reduction, original SDPA.
- `src/savr/openvla/current_frame_features.py`: exact features and deployment wrapper.
- `src/savr/openvla/current_frame_corrector.py`: model and mean L1 teacher loss.
- `src/savr/openvla/current_frame_records.py`: lossless BF16 / FP32 feature records.
- `src/savr/openvla/current_frame_training_plan.py`: splits, schedules and fit rules.
- `scripts/fit_current_frame_adapters.py`: training, final checkpoints, offline metrics.
- `src/savr/openvla/contemporary_episode.py`: original eight-action queue, timed loop.
- `src/savr/openvla/execution_precision.py`: float32 execution / gripper boundary.
- `scripts/run_current_frame_robot_pilot.py`: frozen four-policy robot comparison.
- `scripts/analyze_current_frame_robot_pilot.py`: frozen completed-run analysis.
- `scripts/verify_current_frame_robot_copy.py`: independent read-only reconciliation.

Fitting: 3,200 observations, 80 per task, from demonstration trajectories; 800
trajectory-disjoint validation observations, 20 per task. Both use historical
training data, not independent final test data. Dense teacher and compressed
features were computed from the SAME current observation. No expert-action target
is substituted for dense prediction. Stored collection is about 17 GB, already
present locally and on TITAN; do not recollect it unnecessarily.

Three matched arms: action-only, visual, within-task shuffled-label visual control.
Seed 7, FP32 AdamW 1e-4, zero weight decay, batch 16, ten epochs / 2,000 updates per
arm. All final checkpoints sealed BEFORE validation, no best-checkpoint selection.
6,000 total updates. Frozen backbone not loaded for research fitting. Training
used deterministic math attention, disabled TF32, no mixed-precision optimizer.
Teacher loss is average L1 over all eight actions and seven dimensions, not a
task-success or “never damage a correct action” objective. Gripper is unnormalized
[0,1] at this boundary, unlike six rescaled motion dimensions.

## 7. Completed evidence and exact outcomes

### Preliminary compression screen

`results/fixed-compression-screen-v01`: state 0, 40 consumed conditions.
Dense 39/40; plain 384 39/40 with 15.32% lower mean controlled query time;
plain 256 37/40 with 32.50% reduction. Predeclared preference selected **384**.
Do not switch to 256 retrospectively to manufacture adapter headroom.

### Collection and fitting

Collection: `results/current-frame-feature-collection-v01` (4,000 records / 8,000
backbone calls). Fit: `results/current-frame-adapter-fit-v01` (three saved models).
Offline L1: base 0.03489550550; action-only 0.03332297899; visual 0.03305615107;
shuffled visual 0.16091179267. Visual improves 5.27% versus base, only 0.80% versus
action-only. This was an offline triage pass, not a robot-performance claim.
Visual improved L1 on 540/800 validation observations, worsened it on 260/800.
Training loss was still decreasing across the ten epochs; this alone does not
establish undertraining or justify extending epochs after seeing outcomes.

### Final completed development pilot

`results/current-frame-robot-pilot-v01`: 120 matched development conditions,
40 tasks, states 1/2/3, seed 7, four policies = 480 primary episodes + eight native
controls = **488 total**. Historically exposed development states, not fresh holdout.

| Policy | Success /120 | Controlled mean query ms |
|---|---:|---:|
| Dense 512 | 120 | 1201.21719 |
| Plain compression 384 | 117 | 1017.59777 |
| Action-only correction | 118 | 1018.95302 |
| Visual correction | 117 | 1019.79798 |

Visual is 15.10% lower time than dense, but 0.216% slower than compression alone.
The approximately 15% speed gain is due to compression, not the corrector.
Native controls 8/8. 16 trained deployment qualifications and 144 same-observation
dense/native shadows all passed. Shadow hidden/normalized/raw maximum error 0.0;
processed commands exact. Final weights loaded and used correctly in these checks.

Per suite (dense / compression / action-only / visual, each out of 30):
Spatial 30/30/30/30; Object 30/30/30/29; Goal 30/30/30/29;
long-horizon LIBERO-10 30/27/28/29.
Visual gains two paired cases and loses two versus compression. Do not select
only long-horizon results for a general superiority claim.

Exact differing cases (task names are LIBERO identifiers):

- Object `pick_up_the_alphabet_soup_and_place_it_in_the_basket`, state 1:
  only visual fails (280-step horizon).
- Goal `push_the_plate_to_the_front_of_the_stove`, state 1:
  only visual fails (300-step horizon).
- LIBERO-10 `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`, state 2:
  dense + visual succeed; compression + action-only fail.
- LIBERO-10 `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket`, state 1:
  dense + visual succeed; compression + action-only fail.
- LIBERO-10 `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate`, state 2:
  dense + action-only succeed; compression + visual fail.

Frozen preliminary-positive criteria: dense >=108 and native 8/8; visual >=3 net
wins versus compression; >=1 net win versus action-only; <=2 losses versus dense;
no suite >2 losses versus compression; >=10% controlled mean time reduction.
Baseline, suite-loss and speed pass; other three fail. These are frozen pilot
criteria, not universal publication requirements. Do not rewrite their outcome.

The descriptive task-cluster 95% interval for visual minus compression is
[-3.33,+2.50] percentage points. One seed, ceiling effects and exposure limit
inference. Independent native trajectories varied in three suites despite exact
same-observation parity. Small success differences can include rollout variation.

10,120 calls; 15,569.69 seconds (~4h20); peak aggregate memory 16,598 MiB;
completed 2026-09-20 04:38:29 UTC; no technical stop. All caps passed.
Controlled timing includes adapter/copies/output work but excludes simulator
observation resizing; online query/whole-episode costs are separately in analysis.

## 8. Confirmed gripper diagnostic error — do not repeat the old interpretation

`scripts/fit_current_frame_adapters.py` line 241 (at handoff) computes disagreement
using `prediction[...,6] > 0` versus `teacher[...,6] > 0`. Its independent saved-
prediction check repeats that formula. The checkpoint's four action-statistics
masks all have last entry false: the gripper is not mapped from [-1,1]. The official
execution does `-sign(2*float32(gripper)-1)`, so the meaningful boundary is **0.5**,
including the exact-tie case. Sources verified via ssh inside the project:

- `checkpoints/openvla-7b-oft-libero-four-suite/dataset_statistics.json`
- `_unnormalize_actions` in that checkpoint's `modeling_prismatic.py`
- `third_party/openvla-oft/experiments/robot/robot_utils.py`
- `third_party/openvla-oft/experiments/robot/libero/run_libero_eval.py`

Read-only recalculation from 800 saved predictions x eight actions:

| Policy | Actual processed-gripper disagreement with teacher |
|---|---:|
| Compression | 331/6400 = 5.171875% |
| Action-only | 330/6400 = 5.156250% |
| Visual | 330/6400 = 5.156250% |
| Shuffled visual | 1963/6400 = 30.671875% |

Old 6.71875% versus 6.81250% values describe a sign-at-zero diagnostic, NOT executed
gripper disagreement. Retract the earlier interpretation that gripper agreement
worsened. This reporting error did not enter optimization, checkpoint selection,
offline triage gates, robot execution or terminal-success analysis. The 117/120
robot outcome and failed adapter criteria remain valid. Preserve old authenticated
files; document a separate erratum and use versioned corrected analysis for future
work. Do not alter frozen source in place: ancestors hash it.

## 9. Diagnosis: known versus not established

No execution bug explaining the robot outcome was found in the scoped audit.
That is not a guarantee that all bugs are absent. Tests verify actual inference,
features, zero residual, trained deployment, weights, action order and normalization.

Plausible research limitations:

- Mean teacher imitation error is not terminal task success. Corrections are always
  applied, including to actions already adequate for the task.
- Fitting used only 3,200 demonstration frames, not states reached by the corrected
  policy. Small action changes can change future observations.
- The incremental visual benefit was small before rollout; small average gains
  and regressions on some states are consistent with mixed closed-loop outcomes.
- Plain compression already succeeded on 117/120 conditions, limiting measurable
  recovery headroom. This does not explain away introduced failures.

Do not assert a proven root cause, “the idea is impossible,” or guaranteed benefit
from longer training. The pilot saved terminal outcomes/query costs and native
shadow hashes, not full per-step corrected actions/images or videos. The exact
mechanism of each failed rollout cannot be reconstructed from those logs alone.
Any targeted reproduction is a NEW diagnostic experiment, not recovery of this run.

## 10. Resume procedure and next decision

Do these safe steps before requesting new compute:

1. Confirm cwd, `AGENTS.md`, latest status/decisions, dirty tree and local evidence.
   Run the read-only local verifier below. Do not rerun the original analyzer:
   it intentionally writes `analysis.json` exclusively, and that file already exists.
2. Read the gripper erratum; ensure future conclusions distinguish the incorrect
   diagnostic from valid robot success. Preserve all immutable evidence/weights.
3. Continue the research decision, NOT another uncontrolled training campaign.
   Assess two explicit possibilities against actual literature and our data:
   (a) an honest current-frame compression efficiency–accuracy study, where novelty
   and competitiveness must be demonstrated against existing simple compression;
   (b) a narrowly justified correction change, supported by evidence about why
   corrections help/hurt, rather than another renamed architecture.
4. A proposed targeted diagnostic should distinguish objective/data coverage,
   rollout variation and implementation behavior. Specify which measurements are
   missing, what result would change the decision, and the smallest bounded test.
   Do not select failed cases and present them as independent performance evidence.
5. Freeze a new prospective protocol for any change in training, loss, policy,
   token budget, population or success tolerance. Seek one clear approval for a
   materially new GPU experiment; avoid repeated micro-approval requests for
   routine CPU research, tests and preparation. The Step B screen protocol is
   now frozen in `reports/VLAPRUNER_COMPARATOR_AUDIT_AND_SCREEN_SPEC_V2.md` §6
   (arms, warm/steady-state timing, parity gates, resource caps, ~1.5–2 h
   runtime estimate, deviations). The immediate decision is whether to approve
   its GPU launch on coordinated GPU 0 (one run, no retry); the manuscript and
   advisor email remain held until the fixed-384 vs adaptive tradeoff is
   measured.
6. Independent confirmation/seed replication and matched competitive methods are
   still required before a positive paper claim. Do not open historically reserved
   states 6+ / final populations merely because they are available. Consult
   `reports/cac_c0_recovery01/simulator_populations_v1.jsonl` for exposure metadata.

Useful source anchors already consulted (read primary papers before relying on them):

- https://www.roboticsproceedings.org/rss21/p017.html — OpenVLA-OFT.
- https://proceedings.mlr.press/v15/ross11a.html — imitation-learning distribution
  shift: motivation, NOT proof that it caused our particular failures.
- https://nips.cc/public/guides/PaperChecklist — claims must match evidence;
  aspirational goals may be discussed explicitly as unachieved.

Do not promise a positive paper based on these citations. A claim can concern a
useful tradeoff rather than dominance on all metrics, but novelty and repeatable
evidence cannot be replaced by optimism about future experiments.

## 11. Commands and authenticated evidence anchors

Read-only local restart checks:

```sh
cd /Users/veddwivedi/Documents/VLA/SAVR
git status --short
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_current_frame_robot_copy.py
```

Example scoped remote CPU checks (not a launch):

```sh
ssh titan 'cd /home/ved/SAVR && env CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src envs/openvla-oft/bin/python -m unittest discover -s tests/openvla -p "test_current_frame*.py" -v'
```

`rg` is available locally but was absent on TITAN; use scoped `grep` there if needed.
No dependency installation is needed. Do not mistake `command not found: rg` for
a failure of the research environment.

Frozen current robot config SHA-256:
`9e3f6728be56821d5da73185f310f1119adda32e2a4fc6e50672cc2e71612c8e`

Robot summary:
`699de8203703ea508ab5668584ed2d7d52452f53af6dc20a539809a50221588e`

Robot analysis:
`4ef6ae0873f5c0d24b5caffcd223a9673d067f951080af4d2492da093e4fba70`

Fit summary:
`c863b31bff5ea15f4c93bed21fda430734711642df56aa3e791e9a402c466627`

Fit analysis:
`d9ed6108eef40d4c4466343e04baea42d58a88ba5e06a44368f89bfcbb64a2de`

Collection summary:
`7bde8cd59846d5f5b54c7b9a8302b30ebe74ef13606eb92aeb754465d5b1a8c3`

Final adapter weights, inside `results/current-frame-adapter-fit-v01`:

- `action_only-final.pt`: `8a01608b9690bf2e62017968fbc2ab4afd86325503ccfd3322994ab1838b8640`
- `visual-final.pt`: `15489080d2752d3cf90a7b9cb016cb4c95c9a395b335b400d63e74c33f6bb951`
- `visual_shuffled-final.pt`: `1524bbd2527a4ad362c1d4dff7ab33a17c2bd2cbfded8bb523c507cd75627cc9`

The shuffled model and earlier tiny diagnostic adapters are not research deployment
candidates. Never replace these final weights with a more favorable checkpoint.

## 12. Communication and handoff maintenance

Keep user updates concise, plain and useful. State what was actually measured,
what remains uncertain, and the next concrete step. Avoid internal phase labels
without explanation. The user is frustrated by repeated technical stops and
unjustified optimism; use small justified experiments, not endless audits.

Before ending a substantial turn, update this handoff's current checkpoint and
latest status/decision entries, without erasing historical results. If a future
job is launched, record exact config hash, output directory, owned PID, launch
time, caps, monitoring mechanism and stopping/unblinding conditions here.
Do not claim background monitoring unless OpenCode actually has a working
scheduled follow-up mechanism; a stopped chat does not monitor by itself.

The latest completed work at this handoff is robot-result verification plus the
gripper reporting erratum and the corrected offline diagnostics
(`reports/CURRENT_FRAME_CORRECTION_DIAGNOSTIC_V2.md`, superseding the buggy
coverage V1). Addressed user review: fit-only z-scoring fix; confound-controlled
adequacy (action-only comparison and mechanical bound); deploy-available signal
predictions are weak (held-out AUC 0.599), so no cheap gate beats always-apply.
**Next: evidence-grounded choice of the next contribution/diagnostic, not
relaunch.** Any (b)-line test needs per-step corrected actions, which the
completed pilot did not save.

Historical paper contribution assessment (SUPERSEDED by V2 and latest user decision):
`reports/PAPER_CONTRIBUTION_ASSESSMENT_V1.md`.
Literature comparison shows published adaptive token-compression for VLA inference
(VLA-Pruner arXiv 2511.16449 on OpenVLA-OFT/LIBERO ~1.46x at 50% retention with
~101% relative success; EfficientVLA NeurIPS 2025 1.93x; VLA-Cache arXiv
2502.02175; FLASHVLA/Token-Expand-Merge/FAST) already covers the efficiency
operating region we measured and more aggressive points. Our fixed 384/512
spatial retention is not a novel or competitive compression method and must not
be claimed as one. What remains defensible: (1) controlled end-to-end latency
methodology including all adapter/head/copy overhead; (2) the honest negative
transfer result (visual corrector -5.27%/-0.80% offline L1 but net 0 / -1
closed-loop success in 120 matched conditions, shuffled control passes); (3)
reproducible-practice reporting (frozen gates, paired conditions, ceiling and
exposure analysis). Peer-review gaps (no work launched): standard LIBERO protocol
~500 eps/suite on untouched split with >=3 seeds; >=1 matched comparator on
identical hardware/protocol; FLOPs/memory; >=3 corrector seeds; held-out
generalization. Next step per user, in order: (i) concise contribution assessment
(done), (ii) LaTeX manuscript draft of the honest empirical-study framing with
figures of all four policies, (iii) warm email draft to Prof. Yuan and Mr. Yang
(not sent). No new GPU runs, training, GitHub push, or frozen-evidence changes.
