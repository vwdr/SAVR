# SAVR master research context and continuation guide

**Snapshot: 2026-10-03, America/New_York.**

**Purpose:** let a new ChatGPT, Codex, OpenCode, or Claude conversation understand the research, locate the evidence, and continue without repeating completed experiments or losing scientific qualifications.

**This document is context, not authorization to run anything.** Read the latest user message and applicable `AGENTS.md` before acting. Historical approvals, launch commands, PIDs, and protocols are not permission for new work.

**Canonical local file:** `/Users/veddwivedi/Documents/VLA/SAVR/SAVR_MASTER_CONTEXT.md`.

**GitHub publication addendum, October 3:** the user subsequently authorized
publishing this project snapshot to `agent/full-paper-audit` at
`https://github.com/vwdr/SAVR`. Read
`docs/GITHUB_PUBLICATION_SCOPE_2026-10-03.md` for included evidence and intentional
exclusions. Earlier uncommitted/GitHub-not-current observations below describe
the pre-publication inspection, not the intended completed branch snapshot.
Repository visibility was verified public; it was not changed. This publishing
authorization does not authorize a new experiment, training, email send or change
to scientific claims. Read the named branch, not the older `main` overview.

## Contents

1. [Start here: the actual checkpoint](#1-start-here-the-actual-checkpoint)
2. [Research objective and question](#2-research-objective-and-question)
3. [People, user preferences, and advisor guidance](#3-people-user-preferences-and-advisor-guidance)
4. [Workspace, safety, Git, and authority](#4-workspace-safety-git-and-authority)
5. [Model, mathematical framework, and measurement definitions](#5-model-mathematical-framework-and-measurement-definitions)
6. [Complete research progression](#6-complete-research-progression)
7. [Current results and permitted interpretations](#7-current-results-and-permitted-interpretations)
8. [Data, splits, exposure, and artifacts](#8-data-splits-exposure-and-artifacts)
9. [Repository navigation and implementation map](#9-repository-navigation-and-implementation-map)
10. [Technical failures, corrections, and prevention](#10-technical-failures-corrections-and-prevention)
11. [Literature and novelty boundary](#11-literature-and-novelty-boundary)
12. [Unexecuted plans and future decisions](#12-unexecuted-plans-and-future-decisions)
13. [Restart procedure and verification](#13-restart-procedure-and-verification)
14. [Paper and poster archive](#14-paper-and-poster-archive)
15. [Latest advisor email draft](#15-latest-advisor-email-draft)
16. [Evidence identifiers](#16-evidence-identifiers)
17. [Open questions and things not to repeat](#17-open-questions-and-things-not-to-repeat)
18. [Updating this document and new-chat prompt](#18-updating-this-document-and-new-chat-prompt)

---

## 1. Start here: the actual checkpoint

### 1.1 What is complete

The most recent GPU study is the **completed backend-aligned four-arm development comparison**, `results/adaptive-screen-v03`, reported in `reports/ADAPTIVE_SCREEN_V03_RESULT.md`. It finished on September 26. Do not relaunch it.

The preceding learned current-frame correction pilot, `results/current-frame-robot-pilot-v01`, is also complete. Its small adapters improved offline imitation of dense-policy actions, but did not improve aggregate robot task success over compression alone.

There are real positive measurements:

- Fixed current-frame compression reduced controlled mean complete-query time by approximately **15%**.
- It obtained **117/120** successes versus dense **120/120** in the learned-correction robot pilot.
- In a separate, previously exposed 40-condition comparison, fixed compression and the stronger adaptive compression arm each obtained **39/40**, the same observed count and success/failure pattern as dense inference.
- The stronger adaptive arm reduced controlled mean query time by **15.58%**, versus **15.36%** for fixed compression. It was substantially faster in steady-state queries.
- A visual correction adapter improved held-out offline action L1 by **5.27%** relative to compression, but produced **117/120** robot successes, the same aggregate count as uncorrected compression.

These are **measured development findings**, not a completed novel-method paper. Static token selection is established. The adaptive selector is derived from published VLA-Pruner. Separating attention-score extraction from optimized attention is also established. Do not claim any of these generic ideas as our invention.

### 1.2 Latest user direction

The user wants a positive-results-oriented peer-reviewed research contribution, not endless renamed wrappers or another negative report. After asking whether more experiments were needed, the user chose to **email Professor Yuan and Mr. Yang with the new results and ask for specific research guidance**. The latest email was drafted in chat, **not sent**. See Section 15.

On September 26, the user had accepted preparing an empirical-paper route. A manuscript blueprint and a candidate independent validation design were written. Later conversation clarified that more replication cannot, by itself, establish novelty. The latest recommendation was to avoid spending the large confirmation budget until the contribution is clear. The user then chose the advisor-update route.

Accordingly, a new chat must not treat the older empirical-plan approval as a standing command to launch the large validation immediately. Start from the latest user direction and ask what action is now requested if unclear.

**Current task that created this file:** documentation and handoff only. No new experiment, training, GPU job, manuscript, email send, commit, or push occurred in the initial creation task. A subsequent explicit user request authorized the GitHub branch publication described in the addendum above.

### 1.3 What is not complete

- No independent large confirmation has launched.
- No final positive-results manuscript has been written or submitted.
- No sufficiently original new compression/correction method has been established.
- No deployed per-step diagnostic has established the causal reason the visual adapter failed to improve success.
- No simulator-state exposure audit has certified a genuinely untouched confirmation population.
- No current advisor response to the latest draft email is available in the supplied context.

### 1.4 Running-state caution

Last recorded state: no experiment is running and the `monitor-adaptive-comparison-v03` heartbeat was deleted after completion. Historical PID `3074656` was the completed S1 worker; `3072450` was S0 qualification. The earlier current-frame pilot PID `2016420` also exited. **These are historical identifiers, not live targets.** Never act on an old PID without authenticating a current owned process.

This handoff creation did **not** connect to TITAN or perform a fresh live process audit. Treat last-recorded idle state as historical knowledge, not proof of current server availability.

### 1.5 Verification performed for this snapshot

On October 3, the two local independent verifiers were rerun successfully:

1. `scripts/verify_current_frame_robot_copy.py`: all nine authenticated artifacts, counts, success totals, per-suite outcomes, controlled timing, and frozen gates reconcile.
2. `scripts/verify_adaptive_screen_v03_independent.py`: 168 episode records, 240 timing records, 288 parity records, 3,728 calls, hashes, source identities, controls, and all four arms reconcile.

Eleven listed configuration, summary, and research-weight digests were also checked directly against the guide, including all three final adapter weights. The full 17 GB feature store and every historical artifact were not rehashed in this handoff turn. Earlier verification is reported as earlier verification, not silently upgraded to a fresh audit.

---

## 2. Research objective and question

### 2.1 Overall objective

Develop an original, useful, experimentally supported contribution to efficient **vision–language–action (VLA) inference** that maintains high robot task performance. A result need not dominate every metric or beat every publication to be useful. It does need a defensible contribution, appropriate comparators, rigorous measurements, and an honest statement of any quality cost.

The user values three properties: novelty, positive measured results, and relevance/impact in the same research area. Novelty means adding original scientific knowledge, not necessarily an unprecedented paradigm.

### 2.2 Original question: temporal reuse

Can robot-state and action information make training-free reuse of earlier visual representations more reliable than visual similarity alone, while reducing complete inference cost?

SAVR initially refreshed the full two-camera projected visual prefix when image change, state change, recent action change, or a maximum reuse horizon required it. Later experiments made reuse more selective by camera, transformer layer, image region, or learned intervention-risk routing.

The practical obstacle was that visual redundancy did not reliably identify which earlier information could be reused without changing consequential actions. Moreover, avoiding some visual computation did not necessarily reduce complete-query latency after implementation overhead.

### 2.3 Revised question: current-frame compression and correction

Can we reduce transformer computation while retaining **current information from both cameras**, then use a small frozen-backbone adapter to correct compressed-policy actions toward the dense policy?

This changed a foundational assumption: the new compressed policy does **not** reuse older visual features or mixed-age K/V states. It computes both current camera encodings, removes selected current visual tokens before transformer processing, and optionally uses current full visual tokens to produce an action residual.

Two separate hypotheses were tested:

1. Current-frame token reduction yields a useful complete-query speed–success tradeoff.
2. A learned visual residual improves robot success beyond compression and beyond an action-only adapter.

Hypothesis 1 has positive development measurements. Hypothesis 2 has offline gains but failed the predeclared robot-benefit gates.

### 2.4 The unresolved publication question

What **new scientific limitation or solution** can this evidence establish beyond known compression and residual adaptation?

The current measurements establish an operating-point tradeoff and an offline-to-closed-loop transfer gap. They do not establish that our simple compression is novel, that our corrector improves robot performance, or that the engineering backend adaptation is new.

A paper can report a favorable result in one dimension and a measured cost in another. It cannot present a plausible future improvement as an achieved result. A confidence interval containing zero does not establish equivalence. More runs can improve precision and generalization evidence, but cannot turn a known algorithm into an original one.

### 2.5 Evidence categories to keep separate

| Category | Example | What it establishes |
|---|---|---|
| Engineering qualification | Exact zero-pruning action parity | Tested integration correctness |
| Systems timing | Lower complete-query time on matched inputs | A measured execution benefit in that timing design |
| Offline action quality | Lower L1 to dense predictions on saved frames | Better imitation on those frames |
| Closed-loop behavior | Terminal task success in LIBERO | Performance of the deployed policy in that population |
| Independent validation | Untouched conditions with prospectively frozen analysis | Stronger generalization/precision evidence, if actually run |
| Novelty assessment | Comparison to prior literature | Whether a contribution is genuinely new; not provided by speed alone |

---

## 3. People, user preferences, and advisor guidance

- **Researcher:** Ved Dwivedi. Project affiliation history includes John P. Stevens High School and Rutgers University.
- **Advisor:** Professor Bo Yuan, Rutgers ECE, Structured Representation and Computing (SRC) Lab.
- **Collaborator/mentor addressed in emails:** Cheng Yang, referred to by the user as Mr. Yang.
- **Publicly mentioned repository:** `https://github.com/vwdr/SAVR`.

### 3.1 Advisor correspondence

- **August 20:** Professor Yuan said negative results are a starting point; research should identify a limitation and develop a solution.
- Ved replied that he would explore solutions involving the wrist/scene camera distinction and coarse reuse limitations.
- **September 3:** Ved sent an update about camera-only and fine-grained internal reuse. The first maintained success but not speed; the second reduced systems time but did not maintain robot success. An updated negative-study paper was attached.
- **September 7:** Professor Yuan said negative results alone were insufficient if the aim was a peer-reviewed research paper; the current content was suitable for a technical internal report.
- Subsequent work moved to current-frame compression and learned correction. The latest draft email updates them on these different experiments, not another repetition of the old whole-prefix findings.

### 3.2 Communication requirements

The user wants concise, direct, understandable communication. Explain what was done and what the numbers mean. Do not drown ordinary explanations in internal phase codes. In a paper, define any necessary technical terms and omit chat-specific jargon.

Do not promise that an experiment will work, that all technical errors have been eliminated, or that a result guarantees publication. Repeated assurances followed by more technical stops have damaged trust. Explain concrete evidence and limits instead.

The user wants effective, fast progress **without** compromised accuracy or post-hoc manipulation. Prefer a useful diagnostic over another large blind run when the mechanism/contribution is unresolved. Do not multiply protocols without a load-bearing scientific reason.

Emails should be warm and kind, simple, condensed, and include the important results with plain explanations. Do not send an email unless explicitly authorized to send it. Drafting does not authorize transmission.

### 3.3 Approval conventions

Many historical phase/run approvals exist. They apply to their frozen scope only. A technical failure was normally followed by an explicit recovery protocol and renewed approval, not an automatic retry. Never reinterpret “get a positive paper” as permission to change gates, hide comparators, or spend arbitrary shared resources.

---

## 4. Workspace, safety, Git, and authority

### 4.1 Exact workspaces

| Purpose | Location |
|---|---|
| Actual local review/development repository | `/Users/veddwivedi/Documents/VLA/SAVR` |
| Remote execution repository | `/home/ved/SAVR` on TITAN |
| Only approved remote access route | Existing `ssh titan` host |
| ChatGPT project mirror | `/Users/veddwivedi/.codex/.chatgpt-projects/g-p-6938d001ab38819185befb0e6b3f4e86` |
| Read-only synced references | `sources/` under that project mirror |
| Original remote runtime | `/home/ved/SAVR/envs/openvla-oft/bin/python` |
| Historical compatibility runtime | `/home/ved/SAVR/envs/vla-cache-compat` |

The earlier local path `/Users/veddwivedi/Documents/SAVR` is obsolete in the current handoff. Do not create another checkout there to make old instructions work.

All relative paths in this document resolve from the **actual local repository**, unless explicitly labeled remote. A path listed for historical evidence may be remote-only; confirm existence before using it. Absence of a local copy does not mean an experiment never ran.

### 4.2 University server boundary

Read the repository's `AGENTS.md` in full. Its safety restrictions remain binding even though its original bootstrap status language is stale.

- Operate remotely only inside `/home/ved/SAVR`.
- Do not inspect, modify, move, rename, delete, copy, or change permissions on unrelated university files or directories.
- Do not inspect or interfere with other people's processes, jobs, services, environments, users, network configuration, or allocations.
- No `sudo`, system-wide installation, broad cleanup, permission changes, process termination, or reprioritization.
- Do not select the entire home directory as the workspace.
- No automatic GPU retry, fallback GPU, or extra workload after a fail-closed stop.
- Downloads of weights, datasets, dependencies, or external source require appropriate explicit approval and storage estimates.
- Runtime caches, temporary files, models, data, and outputs must remain under the project when remote execution is authorized.
- Never request or expose passwords, SSH private keys, tokens, or credentials.

Historical coordinated GPU for the recent experiments: **GPU 0, TITAN RTX, 24,576 MiB**, UUID `GPU-bb2451d6-2989-a112-5c18-8892943710e4`. Recent resource ceilings used 23,552 MiB. Exact inequalities and allowances come from each frozen protocol.

Historical identity does not grant a current allocation. A future approved launch requires its prescribed fresh aggregate identity/idle check and coordination. Do not enumerate other users' workloads to find a GPU. Earlier ACR work used another GPU, but that history is not permission to switch now.

### 4.3 Local/remote workflow

1. Read local status and exact source/evidence. Make scoped, reviewable local edits.
2. Preserve unrelated changes. Use patch-based edits, not broad replacements.
3. Transfer only explicit approved files to matching paths under `/home/ved/SAVR`; authenticate contents. Never use destructive synchronization such as `rsync --delete`.
4. Test against the pinned remote runtime CPU-only when that work is authorized. Hide CUDA for CPU analysis and control thread/cache settings.
5. Freeze claims, schedule, inputs, source hashes, configuration, counts, and resource caps before a newly approved GPU run.
6. Run GPU inference on TITAN, not on the Mac. Preserve exclusive versioned output directories.
7. Sync completed evidence and current status back locally; verify hashes. The user wants reviewable local files.
8. Commit/push only when requested and after separating appropriate code/compact evidence from private or large artifacts. The October 3 follow-up request explicitly authorizes this branch snapshot; see the publication addendum.

Do not pull into a dirty checkout or overwrite local status with an older remote status file. The initial handoff-creation task did not authorize a push; the subsequent October 3 user request does authorize the reviewed publication snapshot.

### 4.4 Git state verified October 3

- Branch: `agent/full-paper-audit`.
- HEAD: `0a274b6ea0c0748adabe386ffa0d61670c4ece7a` (August 25 BRACE documentation commit).
- Origin: `https://github.com/vwdr/SAVR.git`.
- Tracked modifications include `PROJECT_STATUS.md`, `docs/DECISIONS.md`, and `src/savr/brace/b3_openvla.py`.
- Many PAIR/CAC/current-frame/adaptive files are untracked. These files contain important work. Never run `git clean`, reset, or checkout over them.
- Git HEAD is **not** the identity of the code used for September experiments. Frozen per-file hashes and run manifests are the load-bearing provenance.
- GitHub is not guaranteed to contain current local work. Do not imply the latest code/results were pushed.
- The README and old top-level context can describe the original negative study. They are not authoritative for the current scientific checkpoint.
- An old screenshot showed the repo as public, while original instructions required private. Current visibility was not checked here. Do not assert it or change it without the user's direction.

### 4.5 Source authority and conflicts

Priority for behavior: current user instruction, applicable safety instructions, and the exact scope of a newly approved protocol. Priority for measured claims: immutable raw evidence and authenticated manifests, independent reconciliation, corrected result reports, then historical narrative.

This master context supersedes **stale continuation statements**, not historical evidence or safety restrictions. `OPENCODE_HANDOFF.md` and `PROJECT_STATUS.md` preserve old entries saying a worker is running. These are not launch instructions. `docs/DECISIONS.md` is the decision ledger; there is no root `DECISIONS.md` to assume.

---

## 5. Model, mathematical framework, and measurement definitions

### 5.1 Pinned system

| Component | Identity |
|---|---|
| Checkpoint directory | `checkpoints/openvla-7b-oft-libero-four-suite` |
| Checkpoint revision | `638918f3d1c2e43a39a8a20772bdb8b91835e4b7` |
| OpenVLA-OFT revision | `e4287e94541f459edc4feabc4e181f537cd569a8` |
| LIBERO revision | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| Historical VLA-Cache revision | `a4909880573868dee2769343d52e793c0341678b` |
| Original runtime | Torch `2.2.0+cu118`, Transformers `4.40.1`, custom native OFT bidirectional SDPA |
| Historical compatibility runtime | Transformers `4.47.0`, causal attention; not equivalent to original OFT execution |

Two views enter each query: the scene camera (`agentview`) and wrist camera (`eye_in_hand`). The vision pipeline uses SigLIP and DINOv2, then a projector. Each camera contributes a 16×16 patch grid, 256 visual tokens; together there are 512 visual tokens. A proprioceptive token gives 513 projected input tokens before prompt/action layout additions.

The backbone has 32 decoder layers, hidden width 4096. There are 56 action-related hidden states for an eight-step, seven-dimensional action chunk. The released regression head produces normalized actions, which undergo original unnormalization and gripper conversion before deployment.

Nominal queue execution is eight actions per policy query, after ten simulator settling steps. Suite step limits: Spatial 220, Object 280, Goal 300, and `libero_10` 520. `libero_10` denotes the long-horizon suite, not all ten benchmark tasks.

### 5.2 Dense and temporal-reuse notation

Let current observation be `o_t = (I_scene,t, I_wrist,t, s_t, instruction)`. Dense inference is

`V_t = Project(Vision(I_scene,t), Vision(I_wrist,t))`,

`A_dense,t = Head(Backbone(V_t, s_t, instruction))`,

where `A` is an 8×7 normalized chunk.

Whole-prefix temporal reuse replaces current `V_t` with an earlier `V_tau` on selected queries. Camera reuse replaces only one camera representation. Internal reuse substitutes selected earlier K/V states inside transformer layers. Recursively reused states can have different physical source times across layers, cameras, and regions. Such mixed-age states are not equivalent to a single clean prior observation.

SAVR's refresh rule combined image/state/recent-action thresholds and an age horizon. Exact settings live in the original frozen configs and protocols; do not reconstruct them from shorthand labels such as `s25-h2`.

Robot actions influence subsequent observations: `s_(t+1) = Dynamics(s_t, Execute(A_t))`. Therefore a small action change can alter future policy inputs. Offline replay on dense trajectories does not measure a deployed reuse policy's full closed-loop behavior. This is a plausible general mechanism, not proof of the cause of each observed failure.

### 5.3 Current-frame compression

The fixed compressed policy instead uses **current** visual tokens:

`V_t^keep = Select_fixed(V_t; budget B)`,

`A_base,t = Head(Backbone(V_t^keep, s_t, instruction))`.

The fixed selection removes visual tokens before layer 0. Budgets tested were 384/512 and 256/512, balanced between the two cameras. Selection uses 2×2 spatial cells, retaining three or two patches per cell with the implemented rotating-corner pattern. Inspect `spatial_selection.py` for exact deterministic indices. Nonvisual tokens remain and original positions are preserved.

Both vision encodings remain fresh and are still computed. A 25% reduction in visual-token count is **not** a measured 25% total FLOP reduction, memory saving, or latency saving.

### 5.4 Learned current-frame correction

This is distinct from the historical CAC proposal. The trained current-frame corrector has no recursively stale K/V input.

Inputs include:

- `Z_t`, the actual regression head's intermediate eight-step features, shape 8×4096;
- base normalized actions, shape 8×7;
- all 512 current projected visual patches, width 4096;
- pooled instruction-only features, width 4096;
- normalized proprioceptive features, width 8.

The visual adapter uses width 256, two eight-head cross-attention/MLP blocks, camera/row/column embeddings, and eight action-step embeddings. A zero-initialized seven-dimensional output adds a residual to the normalized base action:

`A_corr,t = A_base,t + g_theta(Z_t, A_base,t, V_t, instruction_t, s_t)`.

There is no learned gate, clipping, smoothing, or extra replanning in the tested policy. The backbone is frozen. Feature tensors must be detached/cloned into an autograd-compatible representation before training the small head.

Training target is the **dense policy's normalized action prediction**, not expert action or terminal reward:

`L(theta) = mean_(samples,8 steps,7 dimensions) |A_corr - A_dense|`.

Action-only control has the corresponding nonvisual inputs but no visual cross-attention. Visual head has 5,071,879 parameters; action-only head has 3,485,959. They are not parameter-count matched. Shuffled-label training is a negative control, not a deployed robot arm.

### 5.5 Adaptive comparator

The local VLA-Pruner-derived policy prunes after layer index 3, so four layers process the full sequence and 28 layers process the reduced one. Its score extraction/history logic uses current attention and an action-attention history, with layer-15 history, a three-query deque, and gamma 0.8. First three queries after each episode reset are dense; pruning starts at query 4.

`fastv_r` means **pruning fraction**, so retention is `1 - fastv_r`: 0.25 pruning retains 384 tokens; 0.5 retains 256. History of attention scores does not mean reuse of stale visual K/V representations.

Selection details: head-mean attention at layer 3; pre-action rows for the semantic block, 56 action rows for the action block; diversity selection on LLM input embeddings using cosine-distance MMDP semantics. It preserves nonvisual tokens and absolute positions.

Our backend-aligned port keeps native SDPA for the main output while obtaining auxiliary attention statistics at layers 3 and 15. It is a **local selection adaptation**, not an exact upstream numerical/timing reproduction. The auxiliary score implementation materializes a full score matrix at those layers; no novel fused or memory-linear kernel was implemented.

### 5.6 Critical semantics

- OFT's native transformer attention is bidirectional within the **current inference sequence**. This does not imply access to future physical observations.
- Action hidden-state layout must match the released regression head. A one-token shifted slice previously corrupted the custom path. Use `official_semantics.py` and authenticated layout tests rather than blindly hardcoding a negative slice.
- Token deletion must preserve original rotary positions and the complete nonvisual/action layout.
- Official executed gripper conversion is equivalent to float32 `-sign(2*g - 1)`, with boundary 0.5 and tie output 0. Do not use the sign of normalized `g` at zero as a deployment metric.
- Keep action queue behavior fixed when comparing methods. The older SpecPrune controller replanned and discarded queued actions; fixed/adaptive current-frame policies do not.

### 5.7 Timing and outcomes

Complete-query timing counts all measured policy work, including selection, adapters, heads, copying/output overhead as specified by the protocol. Vision-only time, transformer-only time, controlled full-query time, natural-rollout time, and episode wall time are different quantities.

Time reduction is `(T_dense - T_method) / T_dense`. A 15% time reduction is not a 1.15× speedup; speedup is `T_dense / T_method`.

Hardware warmup may be excluded consistently. Algorithm startup, such as three dense adaptive queries **per episode**, belongs in deployment comparisons. Report startup and steady-state separately too. Never choose the primary timing regime after seeing outcomes.

Terminal success is task completion, not measured collision safety, formal constraints, or certified robot safety. Use reliability/performance language rather than safety claims.

---

## 6. Complete research progression

Internal phase codes below are repository navigation labels, not terminology to insert unexplained into academic prose. “Passed” always means the stated test/gate, not a guarantee of a positive paper.

### 6.1 Initial setup and whole-prefix SAVR

The project started as a training-free state-aware visual-refresh proposal. Bootstrap established the repository, pinned OpenVLA-OFT/LIBERO, tests, server boundaries, frozen plans, and evidence formats. FR means Full Refresh; PR periodic refresh; VOR visual-only refresh. These were intended comparison policies, but not all proceeded to a measured final study.

Whole-prefix experiments used one checkpoint, LIBERO-Spatial, and seed 0. There were **1,160 primary evaluation episodes, plus a separate 50-episode timing pilot**: 100 FR, 900 permissive-grid SAVR, 90 first conservative redesign, and 70 final conservative redesign.

| Setting label | Successes /100 | Visual-prefix reuse % |
|---|---:|---:|
| FR | 100 | 0 |
| s25-h2 | 52 | 34.69 |
| s25-h4 | 21 | 43.29 |
| s25-h8 | 23 | 47.28 |
| s50-h2 | 12 | 56.62 |
| s50-h4 | 4 | 63.26 |
| s50-h8 | 0 | 65.21 |
| s75-h2 | 3 | 64.14 |
| s75-h4 | 1 | 75.00 |
| s75-h8 | 0 | 83.68 |

All permissive settings failed the predeclared two-percentage-point success tolerance. FR used 1,309 queries. The best grid setting did not provide preserved success.

The first conservative redesign tested three 30-episode populations: cap05 30/30 with 0% reuse, cap10 29/30 with 6.72%, cap15 27/30 with 10.57%. The final conservative redesign obtained **69/70**, but only **9/944 queries reused the prefix (0.95%)**. It required 70/70 and at least 5% reuse, failing both. PR/VOR final runs did not occur because eligibility gates were not met; do not invent their results.

Diagnosis found that averaging camera change hid wrist changes: in the best-setting audit, 372 wrist and two scene threshold exceedances occurred among 783 reused queries. Offline dense replay underestimated deployed reuse. Many comparable first-reuse action hashes changed. These are associations and decision-rule defects, not direct proof of a particular physical failure mechanism.

Sources: `reports/PHASE6_CALIBRATION_REPORT.md`, `reports/PHASE6R_A_DIAGNOSIS_REPORT.md`, `reports/PHASE6R_D_STAGE1_REPORT.md`, `reports/PHASE6S_D_VALIDATION_REPORT.md`; compact ledger `docs/evidence/negative_results_summary.csv`; original protocol `docs/SAVR_EXECUTION_PROTOCOL.md` and associated phase documents/configs.

**Conclusion:** whole-prefix reuse did not achieve the required useful success–reuse tradeoff in this specific setting. Do not generalize to every VLA, dataset, caching boundary, or learning-based solution.

### 6.2 Asymmetric camera refresh (ACR)

Motivation: keep the wrist view current because it tracks manipulation closely; reuse only the scene-camera representation. This still recomputes downstream processing on the combined representation.

Initial Object development settings, 30 episodes each:

| Setting | Success | Scene reuse |
|---|---:|---:|
| t25-h2-b30 | 29/30 | 26.06% |
| t50-h4-b55 | 24/30 | 47.40% |
| t70-h8-b75 | 23/30 | 49.44% |

No eligible positive method emerged. Physical timing revealed implementation overhead: one early 48-query, zero-episode systems test gave FR 1,213.54 ms, dual-refresh 1,704.57 ms, dual-reuse 1,700.78 ms. Reuse cut visual CUDA work but not complete cost; a Python/JSON audit added approximately 395 ms. Avoid evaluating an optimized method only against an unnecessarily sequential baseline.

After batching/overhead redesign, a matched Object states 3–9 evaluation ran 70 batched-dense and 70 ACR episodes:

- Both succeeded **67/70**.
- Scene representation reused **337/1,335 queries = 25.24%**.
- Batched dense steady complete-query time: **1,188.28 ms**.
- ACR: **1,190.97 ms**, approximately **0.23% slower**.
- Visual time fell from 114.93 to 105.20 ms, approximately 8.46%, below the 10% visual gate.
- The apparent advantage over sequential FR (1,240.04 ms) did not survive the batched dense comparator.

Sources: `reports/PHASE_A5_REPORT.md`, `reports/PHASE_V2_C_REPORT.md`, `reports/PHASE_V3_D_REPORT.md`, `reports/PHASE_V4_A_REPORT.md`. Protocol family: `docs/ACR_EXECUTION_PROTOCOL_V1.md` and subsequent version/phase files.

Later V4 CPU candidate screening did not establish a usable gate. V5 explored execution optimization, including CUDA graph/compiler integration. The single V10 attempt stopped technically: an unsupported BF16/PTX target on TITAN RTX's sm75 and invalidated graph capture prevented a completed query. It produced six eager component warmups but zero full model queries, timing results, or robot episodes. Reserved memory was approximately 15.38 GiB, below the 23 GiB cap; **memory was not the observed V10 cause**. Source: `reports/PHASE_V5_D_V10_TECHNICAL_STOP_REPORT.md`.

**Conclusion:** preserving success with partial camera reuse was possible in one development population, but a useful complete-query latency improvement was not demonstrated. Compiler stops are not robot-method outcomes.

### 6.3 BRACE: branch-rollout adaptive cache execution

BRACE aimed to learn reuse risk using bounded branch interventions and actual provenance contracts, rather than rely only on image change. Contracts distinguished selected layers/camera regions and finite reuse horizons.

Replay/contract checks preceded the physical B3 test. The completed B3 v05 systems study had 356 queries: 22 FR, 302 cached, and 32 VLAADP. A planned upstream comparator did not run because a required utility was unavailable; the exclusion was documented.

Accelerated-only and cycle-level reductions respectively were approximately 7.60/4.82%, 11.82/7.68%, 11.02/7.30%, and 18.71/12.23% across four tested cache modes. Only the most aggressive listed cycle setting passed its speed gate. All four cached timed-action parity groups had 0/42 matching actions; 168 cached timed actions did not meet the action criterion, although gripper agreement held.

The experiment stopped scientifically before the branch-router evaluation. It did not establish learned-router robot improvement. Source: `reports/BRACE_B3_V05_REPORT.md`, `results/brace-b3-physical-v05`, BRACE protocols and `src/savr/brace/`.

**Important later qualification:** the custom readout/runtime audits in Sections 6.6–6.7 affect official-policy equivalence claims in this lineage. Retain internal systems measurements, but do not present those custom actions as verified original OpenVLA-OFT outputs.

### 6.4 PAIR: provenance-aware intervention-risk learning

PAIR modeled the effect of actual stale-state interventions and attempted a lightweight risk router using expert-action regret. Its outcome was not terminal robot success. The demonstration dataset and split provenance are in Section 8.

An early physical pipeline had approximately 4.09–5.22% overhead, failing the ≤2% budget. A vectorized exact-contract redesign passed the engineering budget. P3R selected the D62-h4 cache operating point after 210 calls, 24 timing blocks, and eight exact recursive checks:

- raw systems gain approximately **24.59%**;
- lower gain estimate approximately 24.40%; conservative net lower estimate 17.05%;
- measured overhead approximately 0.78%;
- visual reuse approximately 48.83%.

The pilot P4 worker produced 3,528 calls, 2,120 intervention records, 800 feature records, and 832 contract records. Some point estimates were encouraging: rank correlation approximately 0.345 and tail-risk reduction approximately 21.48%. But the frozen uncertainty gates failed: lower correlation approximately 0.126 was below the required 0.15, and the tail lower bound did not establish a positive effect.

P4B was a deliberate independent reliability check, not a hidden extension until significance. It produced 1,784 calls, 776 intervention records, 560 features, and 304 contracts:

- Spearman **0.2001**, lower confidence bound **0.0999**, versus required point ≥0.30 and lower >0.15;
- matched-service tail CVaR90 improvement **3.14%**, lower **−1.07%**, versus required ≥15% and lower >0;
- horizon-2/4 improvement **1.62%**, versus required ≥10%;
- five of eight gate groups passed, but the scientific criteria did not.

Stopped before P5. There was no positive PAIR simulator evaluation. Favorable subgroups could not rescue the preregistered primary result.

Sources: `reports/PAIR_P3R_REPORT.md`, `reports/PAIR_P4B_REPORT.md`, `reports/PAIR_P4B_DESIGN_AND_POWER_AUDIT.md`, `reports/pair_p4/`, `reports/pair_p4b/`; protocols `docs/PAIR_VLA_EXECUTION_PROTOCOL_V1.md`, P3R/P4/P4B files. Some raw historical bundles may be remote-only. Later semantics/runtime qualifications apply here too.

### 6.5 CAC: cache action correction, initially proposed

CAC asked whether high-dimensional, recursively mixed-age K/V changes could be repaired through a low-dimensional action residual without reconstructing the full fresh cache. Proposed inputs included current visual information, cached action-head features, and cache provenance. This was not another training-free gate.

C0/C1 implemented tensor/data/control feasibility and resource qualifications. Repeated technical integration failures required explicit recovery work. A headroom evaluation was necessary before collecting/training the research adapter. **The proposed CAC research adapter was not trained at C2**, because the cached substrate failed the robot headroom condition.

Key proposal/logic sources: `docs/CACHE_ACTION_CORRECTION_EXECUTION_PROTOCOL_V2.md`, `docs/LOW_DIMENSIONAL_CACHE_ACTION_CORRECTION_FEASIBILITY_2026-08-30.md`, `docs/FRESH_GROUNDED_CACHE_CORRECTION_DEEP_DIVE_2026-08-30.md`, `docs/CACHE_AWARE_LEARNING_RESEARCH_AUDIT_2026-08-30.md`.

### 6.6 Action-readout correction

An August 31 audit found a real custom-path readout bug. For the audited layout with prompt count P, total sequence length was `N = 513 + P + 57`. The released action hidden-state range was `[N−58, N−2)`, while the old custom tail used `[N−57, N−1)`: one-token shifted, despite still containing 56 states.

The error affected custom BRACE/PAIR/CAC comparisons/features that shared the wrong slice. Internal dense/cache agreement could not expose a shared defect. A tested maximum normalized action discrepancy was approximately 0.1818 before robot evaluation.

Source: `reports/OPENVLA_CUSTOM_PATH_SEMANTIC_AUDIT_2026-08-31.md`. Follow-up semantic/position qualifications corrected the path. Do not hardcode the example slice without checking current sequence layout.

Corrected S5 requalification produced 97 calls, exact 56×4096 hidden-state and action parity with the official helper in that runtime, a measured **22.5969% median complete-cycle headroom**, approximately 10.60 ms feature extraction, and 2.39 ms adapter cost. Peak memory was 16,789 MiB. Source: `reports/CAC_C1_S5_REQUALIFICATION_PASS.md`.

This established integration/timing on that runtime, **not** broad robot success or equivalence to the original runtime. The next discovery matters.

### 6.7 Cached-substrate headroom and attention-runtime correction

The corrected S6 Recovery 01 headroom run used 120 matched conditions, states 0–2 across forty tasks, seed 7: 240 primary dense/cache episodes.

- Runtime-dense: **68/120**.
- D62 cached path: **0/120**.
- Dense per suite: Spatial 14/30, Object 27/30, Goal 13/30, Long-horizon 14/30.
- Cached: zero successes in every suite.
- 8,525 queries; approximately 7 h 9 min elapsed; 16,301 MiB peak.

It stopped scientifically; no extension to states 3–5, no C2 training. Source: `reports/CAC_C1H_S6_RECOVERY01_GATE_H_STOP.md`.

**September 8 attention audit materially changed interpretation:** the compatibility runtime used causal attention, whereas the original OFT runtime used bidirectional attention. Even “dense” in the historical compatibility stack did not preserve original checkpoint execution. Correct action readout against an official helper within that same stack did not certify the entire original computation.

Installed Llama source digests recorded in the audit:

- Original 4.40.1: `3aac24cec583a6ef5f60b6ec634a8bd3c8377784c5c63c8cc14cb6790554c52e`.
- Compatibility 4.47.0: `34b00dd58c9887780a7947329cb96468a7fe1427e8fa49dc773ce2c1afc627d4`.

A small matched attention diagnostic in the original environment used 32 calls and 16 episodes on eight conditions: bidirectional **7/8**, causal **5/8**. All eight normalized action outputs changed; maximum differences were approximately 0.295–1.006. Long-horizon was 2/2 versus 0/2, and there was a common Goal failure. This shows an important computation change on those conditions, not the explanation for every historical failure.

The subsequent original-runtime 40-task baseline, state 0 and seed 7, obtained **39/40**: Spatial 10, Object 10, Goal 9, Long-horizon 10. The failed task involved opening a drawer and placing a bowl inside.

Sources: `reports/OPENVLA_ATTENTION_RUNTIME_AUDIT_2026-09-08.md`, `reports/OPENVLA_ATTENTION_BASELINE_DIAGNOSTIC_V1.md`, `reports/OPENVLA_ORIGINAL_BASELINE_40TASK_RESULT_V1.md`.

**Do not claim:** original OpenVLA-OFT itself succeeds only 68/120; stale K/V alone caused every 0/120 failure; a trained CAC adapter failed; or all learned repair is impossible. These experiments did not establish those claims. Preserve the raw historical results and disclose their substrate qualification.

### 6.8 Contemporary SpecPrune comparison

A local source-based SpecPrune adaptation was qualified against the original semantics. Its September 13 forty-task state-0 development comparison obtained:

- Dense **39/40**.
- SpecPrune **32/40**: Spatial 7, Object 10, Goal 7, Long-horizon 8.
- Eight dense-only successes, one pruning-only success, 31 shared successes.
- Controlled mean query time: dense **1,204.20 ms**, pruning **556.32 ms**, **53.80% reduction**.
- 88 episodes including controls, 136 timing records, 2,255 calls; approximately 49.56 minutes; 16,506 MiB peak.

The SpecPrune controller used 142 replans and discarded 694 queued actions. Current fixed/adaptive policies do not replan. This is a meaningful execution difference. Do not omit the faster method just because it lost more successes, or claim an exact published reproduction.

Source: `reports/CONTEMPORARY_REFERENCE_RESULT_V2.md`; implementation in `specprune.py` and `specprune_episode.py`.

### 6.9 Fixed current-frame compression screen

Next tested deterministic current token reduction before layer 0, with native attention and both fresh images. On forty state-0 conditions:

| Arm | Success | Controlled mean query-time reduction |
|---|---:|---:|
| Dense 512 | 39/40 | — |
| Fixed 384 | 39/40 | 15.32% |
| Fixed 256 | 37/40 | 32.50% |

The predeclared selection chose **384**, not an outcome-dependent later choice. Run had 128 episodes including eight controls, 204 timing records, 2,846 calls, approximately 70.38 minutes, 16,504 MiB peak.

Source: `reports/FIXED_COMPRESSION_SCREEN_RESULT_V1.md`; protocol `docs/CURRENT_FRAME_COMPRESSION_SCREEN_V1.md`.

### 6.10 Current-frame learning qualification, collection, and fitting

Qualification demonstrated feature extraction, plain/zero-initialized adapter parity, and short training mechanics on already exposed observations. It involved 80 calls, 16 observations, and 64 optimizer updates per adapter on only two frames. Those engineering diagnostic weights were **not** the research weights.

Feature collection produced:

- **4,000 samples:** 3,200 fit, 800 validation; 80/20 per task across forty tasks.
- 1,054 fit trajectories and 271 disjoint validation trajectories, drawn from the historically designated training pool.
- 8,000 backbone calls (dense + compressed sequentially).
- Approximately 2 h 34 min, 15,295 MiB peak.
- 17,092,424,025 artifact bytes, about 15.9 GiB, across 4,005 artifacts.

Research fitting trained action-only, visual, and shuffled-label visual heads on the same samples/order/budget: batch 16, ten epochs, 2,000 optimizer updates per head (6,000 total), seed 7, AdamW learning rate 1e−4, weight decay 0, gradient clip 1, FP32 head training. Final checkpoints were used without validation-driven checkpoint selection.

| Predictor | Held-out normalized action L1 to dense |
|---|---:|
| Compressed base | 0.03489551 |
| Action-only correction | 0.03332298 |
| Visual correction | 0.03305615 |
| Shuffled-label visual control | 0.16091179 |

Visual improved **5.271%** versus base, **0.801%** versus action-only. Action-only improved 4.506% versus base. Direction was favorable in the reported offline suite results, but this did not establish robot benefit. Fit elapsed approximately 39.27 minutes; peak training memory 907 MiB because the large backbone was not resident.

Sources: `reports/CURRENT_FRAME_LEARNING_QUALIFICATION_RESULT_V1.md`, `reports/CURRENT_FRAME_FEATURE_COLLECTION_RESULT_V1.md`, `reports/CURRENT_FRAME_ADAPTER_FIT_RESULT_V1.md`; protocols in `docs/CURRENT_FRAME_*`.

### 6.11 Learned-correction robot pilot

Completed September 20, forty tasks × three states (1,2,3), seed 7, four arms. Full results and frozen-gate reconciliation are in Section 7.1.

Main outcome: dense 120/120, compressed 117/120, action-only 118/120, visual 117/120. The visual adapter's offline improvement did not yield aggregate robot improvement. The pilot was technically valid; the adapter-benefit hypothesis failed its predeclared criteria.

No full per-step corrected action/image traces were saved. Do not claim a retrospectively proven action-failure mechanism.

### 6.12 Corrected gripper and offline diagnostic interpretation

The old fitting report used a zero/sign boundary for its gripper disagreement diagnostic. That was not the executed-command boundary. The corrected 0.5-boundary counts over 6,400 commands are:

| Predictor | Executed-gripper disagreements | Percent |
|---|---:|---:|
| Base compression | 331 | 5.171875% |
| Action-only | 330 | 5.156250% |
| Visual | 330 | 5.156250% |
| Shuffled | 1,963 | 30.671875% |

The earlier claim that visual correction worsened the executed gripper is withdrawn. Training loss, weights, deployed conversion, and robot success counts were unaffected. Source: `reports/CURRENT_FRAME_GRIPPER_DIAGNOSTIC_ERRATUM_V1.md`. Do not edit frozen fitting evidence to hide the error.

Offline coverage V1 also contained a real z-scoring bug: validation standardized itself independently of fit data. V2 applies fit-only mean/std to both. Corrected OOD-quartile improvement rates are 0.725/0.705/0.640/0.630: a **9.5 pp** gradient, below the diagnostic's 10 pp criterion; Spearman approximately +0.104.

The initial “dominant explanation” attribution to correcting already-adequate actions was withdrawn. Delta error is mechanically bounded below by negative base error, and action-only correction shows similar near-adequate behavior. The association is not visual-specific and does not establish the cause of robot failures.

Deploy-available feature signals weakly predicted whether visual correction helps: held-out logistic AUC **0.599**, reported interval [0.526,0.677], using trajectory-disjoint testing. The p≥0.5 gate applied correction to approximately 86% of test samples and did not beat always-apply: L1 approximately **0.03481** gated versus **0.03471** always. OOD distance alone was not predictive. Any frame-bootstrap interval must be reviewed for appropriate trajectory clustering before formal inference.

**No deployable beneficial gate was demonstrated.** An “apply only when base error is large” rule cannot be deployed using unavailable dense-teacher error. Do not automatically launch a gated robot test.

Sources: `reports/CURRENT_FRAME_CORRECTION_DIAGNOSTIC_V2.md`, scripts `diagnose_current_frame_correction_coverage.py`, `analyze_correction_benefit_signals.py`, and the versioned diagnostic result directories. V1 is retained but superseded.

### 6.13 VLA-Pruner source audit and zero-pruning failure

Pinned upstream source commit: `84d4b7192c77abf1585610e2f12393319b7ebff9`. The released stack required a vendored Transformers 4.47.0 fork and an absent external utility. A CPU-tested local selection translation was chosen, not a new upstream environment install.

Source audit corrected retention/pruning terminology, attention row definitions, diversity-input semantics, layer arithmetic, history/reset behavior, and the first-three-dense startup. The retained visual count is not the total post-prune sequence length: for a 605-token input, keeping 384/256 visual tokens leaves 477/349 total tokens. A qualification gate originally confused those quantities; it was fixed before dispatch.

On September 25, a 64-call, 16-input diagnostic found:

- Native repeats were exact.
- Adaptive-zero and reference-eager outputs matched each other exactly.
- Both differed from native SDPA: hidden discrepancy up to 1.1875, normalized action 0.09765625, executed command approximately 0.010076; 0/16 exact command matches.
- No gripper disagreement in the 128 compared commands.

The discrepancy was isolated to the backend change on those inputs. It was not permission to remove the parity requirement, and not proof the published eager algorithm was wrong or robot-inferior. Failed v1/v2 qualification evidence was preserved.

The v03 recovery retained native SDPA output and separately extracted selection scores. Its S0 qualification passed: **176 calls, 16 checks, exact zero-pruning actions**, correct 384/256 counts, first-three-dense reset behavior, positions, and 32-layer execution. Approximately 310 seconds and 15,297 MiB peak. The bound was not loosened to excuse pruning differences.

Sources: `reports/VLAPRUNER_COMPARATOR_AUDIT_AND_SCREEN_SPEC_V2.md`, `reports/ADAPTIVE_ZERO_DIAGNOSIS_2026-09-25.md`, `reports/ADAPTIVE_QUALIFICATION_V03_VERIFICATION.md`; protocol `docs/ADAPTIVE_SCREEN_PROTOCOL_V2.md`.

### 6.14 Latest adaptive comparison and novelty audit

S1 v03 completed September 26: dense39/40, fixed38439/40, adaptive38438/40, adaptive25639/40. Complete table in Section 7.2.

Fixed compression did better than adaptive compression at **matched 384-token retention** in this screen. It did **not** demonstrate a useful advantage over the stronger adaptive256 arm. Report both comparisons; do not drop adaptive256 to create a positive story.

A subsequent novelty audit rejected the generic claim that keeping optimized attention outputs while extracting selection scores separately was our new method. VLA-Pruner's earlier version explicitly points to SparseVLM's fast-attention adaptation; SparseVLM, Balanced Token Pruning, and related systems provide prior art.

Source: `reports/BACKEND_PRESERVATION_CONTRIBUTION_AUDIT_V1.md`. A targeted prior-art audit is not exhaustive proof no empirical contribution exists. It is sufficient to reject the previously suggested generic novelty claim.

---

## 7. Current results and permitted interpretations

### 7.1 Current-frame robot pilot: all four policies

Source: `reports/CURRENT_FRAME_ROBOT_PILOT_RESULT_V1.md`; immutable bundle `results/current-frame-robot-pilot-v01`.

| Policy | Success /120 | Success % | Controlled mean complete-query ms | Time reduction vs dense |
|---|---:|---:|---:|---:|
| Original dense | 120 | 100.00 | 1201.217189 | — |
| Fixed384 compression | 117 | 97.50 | 1017.597772 | 15.29% |
| Fixed384 + action-only correction | 118 | 98.33 | 1018.953025 | 15.17% |
| Fixed384 + visual correction | 117 | 97.50 | 1019.797984 | 15.10% |

Each suite has 30 primary conditions:

| Suite | Dense | Compression | Action-only | Visual |
|---|---:|---:|---:|---:|
| Spatial | 30 | 30 | 30 | 30 |
| Object | 30 | 30 | 30 | 29 |
| Goal | 30 | 30 | 30 | 29 |
| Long-horizon | 30 | 27 | 28 | 29 |

Visual versus compression: two gained and two lost successes, net zero. Visual versus action-only: two gains, three losses, net −1. Visual cost approximately 2.20 ms above compression (0.216%).

Frozen criteria, all required together:

| Gate | Required | Observed | Result |
|---|---|---|---|
| Baseline validity | Dense ≥108/120, native controls 8/8 | 120/120 and 8/8 | Pass |
| Incremental visual benefit | ≥3 net successes above compression | 0 | Fail |
| Visual above action-only | ≥1 net success | −1 | Fail |
| Dense success retention | Lose at most two successes | Three lost | Fail |
| No large suite loss vs compression | No suite loses >2 | Largest loss one | Pass |
| Useful complete-query speed | ≥10% reduction | Visual 15.10% | Pass |

Thus the adapter method **did not pass its pilot's conjunction of gates**. This is not a reason to rewrite the gate or call speed alone proof of correction success.

Accounting: **488 episode records** =480 primary+8 controls; 16 qualification records; 272 timing records including 16 excluded hardware-warmup records; **10,120 model calls**. All technical resource/count/hash checks reconciled. Elapsed about 4 h 19 min 30 s; peak GPU memory 16,598 MiB.

Reported descriptive paired bootstrap intervals for success differences:

- Compression−dense: −2.50 pp, interval [−5.00,0.00].
- Visual−compression: 0.00 pp, interval [−3.33,2.50].
- Visual−action-only: −0.83 pp, interval [−4.17,2.50].

These are not a prospective noninferiority test. Conditions share forty tasks with three states each; formal inferential claims require appropriate task/trajectory clustering and an explicit estimand. Native trace variation existed in three suites even when controls succeeded. Do not assert perfect simulator determinism.

### 7.2 Adaptive screen v03: all four policies

Source: `reports/ADAPTIVE_SCREEN_V03_RESULT.md`; immutable bundle `results/adaptive-screen-v03`.

| Policy | Success /40 | Controlled mean ms | Reduction | Adaptive startup ms | Adaptive steady ms | Mean episode wall s |
|---|---:|---:|---:|---:|---:|---:|
| Dense512 | 39 | 1203.646861 | — | — | — | 29.89375 |
| Fixed384 | 39 | 1018.707943 | 15.3649% | No dense startup | — | 26.66135 |
| Adaptive384 | 38 | 1122.292357 | 6.7590% | 1204.668310 | 1060.510392 | 27.84255 |
| Adaptive256 | 39 | 1016.122724 | 15.5797% | 1205.252996 | 874.275020 | 24.32104 |

Per-suite counts, ten conditions each: dense/fixed/adaptive256 =10 Spatial,10 Object,9 Goal,10 Long-horizon; adaptive384 =9,10,9,10. Dense/fixed/adaptive256 have the same observed failure pattern.

The primary controlled timing includes three dense adaptive startup queries per reset. It uses **seven-query traces from two saved frames**, with 24 startup and 32 steady records per adaptive arm, drawn from eight controlled traces. This is not the distribution of natural deployed episode lengths.

Natural-rollout query means were approximately 1207.30/1022.78/1085.94/930.65 ms (dense/fixed/adaptive384/adaptive256). Mean episode wall time additionally includes simulator cost and trajectory length. Every actual episode had at least ten queries; a narrative that fixed wins because these were unusually short episodes is unsupported.

Accounting: **168 episodes** =160 primary+8 controls; **240 timing records** =16 hardware warmup+224 measured; **288 parity records**; **3,728 calls**. Elapsed 5,688.63 s (94.81 min); peak 16,506 MiB. Summary completed September 26 at 19:34:20 UTC. GPU identity and frozen sources reconciled.

All parity errors were exactly zero, including 144 adaptive-zero comparisons. All eight native controls succeeded. Goal before/after traces differed while outcomes matched; other suite control traces matched.

Development gates: dense ≥39 passed; loss at most six successes passed for all compressed arms; minimum 10% complete-query reduction passed fixed384/adaptive256 and failed adaptive384. These permissive development triage gates are not a confirmatory unchanged-success claim. No automatic arm selection occurred.

The analyzer's [0,0] descriptive paired bootstrap intervals for fixed/adaptive256 arise from zero discordances in forty development pairs. **They do not prove zero population loss, equality, or noninferiority.** Report uncertainty honestly with an appropriate prospective procedure if making formal claims.

### 7.3 Claim ledger

| Proposed statement | Status |
|---|---|
| Current-frame compression can reduce measured complete-query cost in our setup | Supported by development measurements |
| Fixed384 achieved high observed task success | Supported, with exact populations/counts |
| Visual correction reduced offline action L1 | Supported on the held-out feature validation set |
| Visual correction improved aggregate robot success | Not supported |
| Cheap confidence/OOD gating improves correction | Not demonstrated |
| Fixed384 is better than the strongest tested adaptive method | Not supported |
| Our adaptive port exactly reproduces published VLA-Pruner | False characterization; disclose local adaptation |
| Our native-backend score extraction is a new general method | Rejected by prior-art audit |
| Current measurements establish success equivalence | Not established |
| We have a finished positive-method peer-reviewed paper | No |
| A useful empirical study could be developed | Possible, but scope/novelty/independent validation need a deliberate decision |

---

## 8. Data, splits, exposure, and artifacts

### 8.1 Demonstration source

Dataset: Hugging Face `yifengzhu-hf/LIBERO-datasets`, revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`.

- Forty HDF5 task files, fifty demonstrations per task: **2,000 trajectories**.
- **338,575 original actions**.
- **41,447 eligible eight-action query windows** and **39,447 adjacent query pairs**.
- Recorded original source size **33,784,856,577 bytes**.
- Original seven-dimensional expert actions; no silent no-op removal or trajectory filtering.
- No timestamps in the source; array position specifies physical order.

Remote source root: `data/pair/libero_hdf5/f13aa24a3da8c43c7225569f28c562979fa0e35a`.

Remote index root: `data/pair/index/f13aa24a3da8c43c7225569f28c562979fa0e35a`, containing `trajectory_index.jsonl`, `query_index.jsonl`, and `split_manifest.json`. Compact local metadata is under `reports/pair_p1/`.

### 8.2 Historical PAIR splits

Hash-sorted per-task assignment, seed 20260828: 35 training, eight calibration, seven locked trajectories per task, totals **1,400 training /320 calibration /280 locked**.

Calibration was consumed in PAIR. The locked pool was sealed at the last recorded checkpoint. Do not open sealed labels casually. Do not label later current-frame validation “the untouched locked test”; it is not.

### 8.3 Current-frame fit/validation

The 3,200 fit and 800 validation samples draw from the historically designated **training** pool, on disjoint trajectories. Some metadata fields still say `split=train` for both. `learning_role` and the frozen learning manifest identify fit versus validation. Do not infer roles from that single field.

Collection result root: `results/current-frame-feature-collection-v01`. This is a large saved feature store. Avoid unnecessary repeated full hashing or loading all 17 GB when compact manifests/verifiers suffice.

Research weights in `results/current-frame-adapter-fit-v01`:

- `action_only-final.pt`
- `visual-final.pt`
- `visual_shuffled-final.pt`

Teacher targets are dense-model predictions. PAIR expert regret targets and current-frame dense imitation are different labels and questions.

### 8.4 Simulator evaluation exposure

Simulator initial-state IDs are not demonstration split IDs.

- All simulator state IDs **0–9** are development in the current planning convention.
- Current-frame pilot used states 1/2/3 across forty tasks, seed 7.
- Original baseline, fixed screen, SpecPrune reference, and adaptive screen all used state 0 development conditions. Do not pool these repeatedly exposed episodes into an independent larger study.
- Candidate IDs 10–49 were proposed for future validation, but are **not certified untouched** across all historical runs yet.
- “Unseen initial condition” does not mean unseen task, policy-training data, or benchmark family.

The metadata audit found 95 JSON configs matched between local and server, 65 local and 114 remote result-directory names, three recognized high-state fields in protected-population plans, and 77 configs without recognized explicit state IDs. Directory/config inventory is not execution/exposure proof. Historical outputs must be reconciled narrowly without opening sealed outcomes.

Source: `reports/CONFIRMATION_METADATA_INVENTORY_V1.json`, `scripts/audit_confirmation_metadata.py`.

### 8.5 Run-bundle conventions

A complete immutable `worker_summary.json`, expected record counts, input/config/source identities, and artifact hashes are required before opening outcomes. A directory, log, or progress counter alone does not establish completion.

During a blinded run, inspect only authorized owned-process health, count-only records, bytes, elapsed time, and aggregate selected-GPU telemetry. Do not read partial successes, timing values, labels, regret, or stage outcome summaries.

If a technical stop occurs, preserve evidence and inspect only the allowed technical diagnostics (`technical_stop.json`, `technical_traceback.log`, or scoped pre-directory terminal log). No automatic retry. A bad scientific outcome is not a technical failure to rerun.

Completed-run analysis is CPU-only after hashes/counts/caps reconcile. Independently recompute gates rather than trusting an analyzer's conclusion alone. Some analyzers create outputs exclusively; do not rerun blindly over an existing result path.

---

## 9. Repository navigation and implementation map

### 9.1 Directory map

| Location | Purpose / caution |
|---|---|
| `AGENTS.md` | Safety and implementation constraints; old proposal wording is historical |
| `PROJECT_STATUS.md` | Long status ledger; top latest entry before old launch notes |
| `OPENCODE_HANDOFF.md` | Earlier focused handoff; has superseded active entries |
| `HANDOFF_FOR_CLAUDE.md` | Earlier cross-agent handoff; not automatic authorization |
| `docs/DECISIONS.md` | Decision ledger |
| `docs/` | Frozen/historical protocols, designs, research audits |
| `reports/` | Human result reports, errata, compact machine audits, terminal logs |
| `results/` | Versioned run bundles; many large/ignored/local or remote-only |
| `configs/` | Frozen/calibration configs, separated by research family |
| `src/savr/` | Original refresh components and later research packages |
| `src/savr/acr/` | Asymmetric camera refresh |
| `src/savr/brace/` | Historical branch/provenance reuse implementation |
| `src/savr/pair/` | Intervention-risk/contract/router work |
| `src/savr/cac/` | Historical cache-action-correction feasibility |
| `src/savr/openvla/` | Original-semantic current-frame/comparator implementation |
| `scripts/` | Preflight, freezing, collection, workers, analysis, verifiers |
| `tests/` | Unit/contract/integration tests; CPU tests do not establish real-model parity alone |
| `schemas/` | Evidence/contract schemas |
| `environment/locks/` | Original dependency records; not a complete description of every later runtime |
| `data/`, `checkpoints/`, `third_party/`, `envs/` | Primarily remote runtime/data; do not assume present locally |
| `tmp/` | Existing scratch material; preserve unless exact cleanup requested |
| `output/pdf/`, `output/tex/`, `output/poster/` | Historical papers/posters; not current method authority |

### 9.2 Current source files

All listed below are under `src/savr/openvla/`:

| File | Role |
|---|---|
| `official_semantics.py` | Released action layout/readout and semantic helpers |
| `execution_precision.py` | Native precision behavior and parity-sensitive execution |
| `offline_camera_inputs.py` | Saved HDF5 camera conversion/orientation |
| `visual_compaction.py` | Token removal with nonvisual layout/position preservation |
| `spatial_selection.py` | Fixed deterministic spatial token selection |
| `fixed_compression.py` | Current-frame compressed policy path |
| `current_frame_features.py` | Actual head/intermediate feature extraction |
| `current_frame_corrector.py` | Visual/action-only residual architectures |
| `current_frame_records.py` | Feature/evidence records |
| `current_frame_sampling.py` | Fit/validation sampling |
| `current_frame_training_plan.py` | Frozen learning-plan semantics |
| `current_frame_robot_contract.py` | Four-arm robot-pilot schedule/accounting |
| `compression_screen_contract.py` | Fixed-budget screen contracts |
| `contemporary_contract.py`, `contemporary_episode.py` | Earlier comparator evaluation integration |
| `specprune.py`, `specprune_episode.py` | SpecPrune local algorithm and replanning controller |
| `vlapruner_adaptive.py` | Pinned-release-derived selection/history semantics |
| `adaptive_sdpa.py` | Native attention output with auxiliary score extraction |
| `adaptive_query.py` | Adaptive query integration |
| `adaptive_screen_contract.py` | Adaptive qualification/screen contracts |

Shared helpers may be pinned by completed runs. Read/extend with a new version where needed; never silently edit a frozen source and claim the old evidence authenticates it.

### 9.3 Current task-to-file map

| Task | Protocol / report | Main scripts |
|---|---|---|
| Original-runtime baseline | `docs/OPENVLA_ORIGINAL_BASELINE_40TASK_PROTOCOL_V1.md`; corresponding result report | `run_openvla_original_baseline.py`, `analyze_openvla_original_baseline.py` |
| Attention diagnostic | `docs/OPENVLA_ATTENTION_BASELINE_DIAGNOSTIC_V1.md`; attention reports | `run_openvla_attention_diagnostic.py`, `analyze_openvla_attention_diagnostic.py` |
| Contemporary SpecPrune reference | `reports/CONTEMPORARY_REFERENCE_RESULT_V2.md` | `run_contemporary_reference_recovery01.py`, `analyze_contemporary_reference_v2.py` |
| Fixed compression | `docs/CURRENT_FRAME_COMPRESSION_SCREEN_V1.md`; fixed qualification/screen reports | `run_fixed_compression_qualification.py`, `run_fixed_compression_screen.py`, corresponding analyzers |
| Learning integration | `docs/CURRENT_FRAME_LEARNING_PILOT_V1.md`; qualification result | `run_current_frame_learning_qualification.py` |
| Feature collection | `docs/CURRENT_FRAME_FEATURE_COLLECTION_V1.md`; collection report | `freeze_current_frame_training_inputs.py`, `collect_current_frame_features.py`, `verify_current_frame_collection_copy.py` |
| Adapter fitting | `docs/CURRENT_FRAME_ADAPTER_FIT_V1.md`; fit report + gripper erratum | `fit_current_frame_adapters.py` |
| Four-arm learned robot pilot | `docs/CURRENT_FRAME_ROBOT_PILOT_V1.md`; readiness/result | `freeze_current_frame_robot_states.py`, `run_current_frame_robot_pilot.py`, `analyze_current_frame_robot_pilot.py`, `verify_current_frame_robot_copy.py` |
| Offline diagnostic | `reports/CURRENT_FRAME_CORRECTION_DIAGNOSTIC_V2.md` | `diagnose_current_frame_correction_coverage.py`, `analyze_correction_benefit_signals.py` |
| Adaptive zero/backend diagnosis | `reports/ADAPTIVE_ZERO_DIAGNOSIS_2026-09-25.md` | `diagnose_adaptive_zero_v1.py` |
| Adaptive S0/S1 v03 | `docs/ADAPTIVE_SCREEN_PROTOCOL_V2.md`; qualification/result reports | `freeze_adaptive_launch.py`, `run_adaptive_qualification.py`, `run_adaptive_screen.py`, corresponding analyzers, `verify_adaptive_screen_v03_independent.py` |
| Contribution assessment | `reports/BACKEND_PRESERVATION_CONTRIBUTION_AUDIT_V1.md`; paper assessment V2 | Read-only research; no launch script implies authorization |
| Candidate confirmation | `docs/EMPIRICAL_CONFIRMATION_DESIGN_V1.md` | `audit_confirmation_metadata.py`; full executable study not frozen |

Script names in this table resolve under `scripts/`. Never execute a `run_*`, `launch_*`, `freeze_*`, or fitting script simply because it is listed here.

### 9.4 Config/result locations for the current completed studies

- `configs/openvla/current_frame_robot_pilot_v1.json` → `results/current-frame-robot-pilot-v01`.
- `configs/openvla/adaptive_qualification_v3.json` → `results/adaptive-qualification-v03`.
- `configs/openvla/adaptive_screen_v3.json` → `results/adaptive-screen-v03`.
- `results/current-frame-feature-collection-v01` contains the saved feature store and manifests.
- `results/current-frame-adapter-fit-v01` contains research weights and fitting evidence.
- `results/fixed-compression-screen-v01` contains the earlier fixed-budget screen.
- `results/adaptive-zero-diagnosis-v01` contains the backend diagnosis.
- `results/contemporary-reference-v02` contains the SpecPrune reference.

Tests to inspect include `tests/openvla/test_official_semantics.py`, spatial/compaction/fixed compression tests, current-frame feature/corrector/record/sampling/training/robot tests, and adaptive SDPA/qualification/screen/VLA-Pruner tests. Report actual pass and skip counts. In recent remote CPU tests, some upstream-scratch parity tests were skipped because approved scratch fixtures were Mac-only. A skipped test is not a passed test.

---

## 10. Technical failures, corrections, and prevention

### 10.1 What repeatedly went wrong

The project crossed several difficult interfaces: cached custom transformer execution, released action-head layouts, attention implementations, rotary positions, raw/normalized action conversion, simulator/controller behavior, and shared GPU resource limits. Small mismatches could invalidate comparisons before the scientific hypothesis was tested.

Not every bad result was technical. Separate:

- Unsupported compiler/graph paths, missing utilities, shape/layout mistakes, precision/backend parity differences: integration issues.
- A correctly executed method failing a predeclared success/routing/benefit gate: scientific outcome.
- A later audit finding a shared reference defect: qualifies earlier interpretation; does not erase evidence.

### 10.2 Correction ledger

| Issue | Correction / present status | Interpretation safeguard |
|---|---|---|
| Average camera-change signal concealed wrist changes | Separate camera attention in later ACR studies | Original gate limitation, not all-reuse impossibility |
| Sequential camera/JSON audit overhead | Batching and scoped timing | Compare against optimized batched dense |
| CUDA compiler/graph unsupported behavior | Preserve V10 technical stop; no automatic fallback | Zero episodes is not a robot result |
| Shifted 56-state action readout | Official layout/readout qualification | Shared custom dense/cache agreement was insufficient |
| Causal compatibility runtime vs native bidirectional OFT | Original runtime restored/audited | Historical 68/120 not original checkpoint baseline |
| Precision/eager vs SDPA output difference | Backend-isolating diagnostic, native output retained | Numerical difference does not itself prove algorithm invalid |
| Visual-count versus full sequence-length gate | Include retained nonvisual tokens | 384 kept visual tokens need not mean sequence length 384 |
| Pruning differences contaminating zero-pruning tolerance | Separate zero-pruning and pruning-shift quantities | Never inflate reference tolerance using actual method changes |
| Offline gripper metric threshold | Official float32 0.5-boundary diagnostic | Counts fixed by erratum, not altered robot evidence |
| Coverage z-score fit/validation mismatch | Single fit-only standardization, versioned V2 outputs | Preserve V1 provenance, withdraw old inference |
| “Dominant explanation” from adequate-action error | Mechanical bound + action-only confound controls | No causal deployment conclusion from offline association |
| Weak benefit predictor/gating | Report failed deployable gate | Dense-teacher error is unavailable at deployment |
| Generic backend novelty claim | Prior-art audit rejected it | Engineering qualification is not scientific novelty |
| Stale running headings/Git HEAD | Latest master checkpoint + source hashes | Do not relaunch from an old PID or use old commit as current code identity |

### 10.3 Required prevention for any future study

1. Specify the scientific question before implementation. Check direct prior art before describing a leading method as novel.
2. Define exact tensor/attention/action/controller semantics and compare to an independent released oracle, not only two custom branches.
3. Exercise real checkpoint/image inputs before large dispatch; CPU mocks do not cover GPU dtype/backend/runtime behavior.
4. Qualify zero-change behavior, original positions, budgets, resets, warm-start, gripper conversion, queue behavior, and camera preprocessing.
5. Separate engineering qualification from training and closed-loop evaluation; diagnostic weights are not research weights.
6. Freeze identities/counts/controls/caps and stop rules. Use exclusive versioned output paths and authenticated immutable summaries.
7. Preserve blinded outcome handling until complete. Report all arms and preregistered gates, including unfavorable ones.
8. Record diagnostic quantities needed for a possible future mechanism study before deployment. Do not presume unsaved actions can later be recovered.
9. Independently recompute results and uncertainty. Distinguish descriptive intervals from valid equivalence/noninferiority claims.
10. Stop when a contribution premise fails. Do not invent a replacement name or more gates to imply progress toward publication.

No finite checklist can guarantee no technical errors. State residual risks honestly and use cheap staged checks to reduce their cost.

---

## 11. Literature and novelty boundary

These are sources used in prior project research, not a freshly exhaustive October 3 literature review. Recheck contemporary publication versions and claims before a manuscript. Cite sources because they support a specific claim, not to reach a count.

### 11.1 Core system and benchmark

- [OpenVLA, official CoRL record](https://proceedings.mlr.press/v270/kim25c.html): base VLA context.
- [OpenVLA-OFT, official RSS record](https://www.roboticsproceedings.org/rss21/p017.html): tested model family/action-chunk inference.
- [LIBERO, official NeurIPS record](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html): benchmark/tasks.
- [VLA-Cache](https://arxiv.org/abs/2502.02175): prior temporal reuse/cache research. Our earlier compatibility implementation is not automatically an exact published reproduction.

### 11.2 Relevant compression comparators and context

- [SpecPrune](https://arxiv.org/html/2509.05614v2): source-based contemporary pruning comparison used here; different controller behavior must be disclosed.
- [VLA-Pruner](https://arxiv.org/html/2511.16449v5): closest adaptive comparator on OpenVLA-OFT/LIBERO. Our local native-SDPA adaptation differs from its released numerical/runtime path.
- [Pinned VLA-Pruner Llama source](https://raw.githubusercontent.com/MINT-SJTU/VLA-Pruner/84d4b7192c77abf1585610e2f12393319b7ebff9/src/openvla-oft/transformers/src/transformers/models/llama/modeling_llama.py): exact audited source identity.
- [TEAM-VLA](https://arxiv.org/html/2512.09927v1): current-frame token merging/compression context.
- [LAC](https://arxiv.org/html/2602.00686): related learned allocation context, not a replicated matched baseline.
- EfficientVLA, FLASHVLA, Token Expand–Merge, and FAST were considered in the earlier paper-contribution assessment. Do not repeat their headline speedups as hardware/checkpoint-matched comparisons. In particular, the cited EfficientVLA result was on CogACT/SIMPLER, not our exact stack. Consult `reports/PAPER_CONTRIBUTION_ASSESSMENT_V1.md` and its V2 successor for qualified context.

### 11.3 Backend-preservation prior art

Sources actually inspected for the rejection of a generic method claim:

- [VLA-Pruner v1](https://arxiv.org/html/2511.16449v1), implementation details: explicitly suggests SparseVLM's FlashAttention adaptation. Its omission from later wording does not erase prior disclosure.
- [SparseVLM full text](https://arxiv.org/html/2410.04417v4) and [ICML record](https://proceedings.mlr.press/v267/zhang25s.html): separate normal fast-attention output and pruning score computation.
- [Balanced Token Pruning, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/file/5aab3631d0d3131281fb88265db69480-Paper-Conference.pdf): score extraction compatible with fast attention.
- [ZipCache, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/7e57131fdeb815764434b65162c88895-Paper-Conference.pdf): related probe-token saliency precedent.
- [ETA-VLA](https://arxiv.org/html/2603.25766): related selected eager layers/fast-attention engineering in a driving setting, not a matched LIBERO comparator.

### 11.4 Conclusions of the contribution audit

Do not claim:

- fixed spatial subsampling is itself a new compression method;
- frozen-backbone residual correction is an unprecedented family;
- separating selection scores from optimized attention output is new;
- a local framework port's action parity proves a new scientific method;
- absence from a few papers establishes novelty or likely acceptance.

Potentially original empirical knowledge includes controlled implementation effects and the observed offline-to-closed-loop transfer gap. Whether that supports a worthwhile peer-reviewed empirical contribution is an open judgment, not an established verdict. Professor Yuan's previous guidance favors a solution contribution. Seek specific guidance rather than dressing replication as invention.

An attention-backend × pruning factorial study was discussed as an **unvalidated option**: native dense, eager dense, native pruned, eager pruned, with selector indices, full cost, and strong selective-score controls. It was neither frozen nor authorized. Current evidence proves numerical differences on tested inputs, not a causal robot-success penalty from switching backend.

---

## 12. Unexecuted plans and future decisions

### 12.1 Historical candidate empirical paper

`docs/EMPIRICAL_PAPER_BLUEPRINT_V1.md` lays out an empirical comparative paper rather than a claimed new pruning algorithm. `docs/EMPIRICAL_CONFIRMATION_DESIGN_V1.md` proposes **one independent study**, not an indefinite sequence of confirmations.

Proposed scope: 400 matched initial conditions across forty tasks, four unchanged arms (dense512, fixed384, adaptive384, adaptive256), totaling 1,600 primary episodes plus sixteen controls, in two sessions. Candidate state pool 10–49 requires full exposure certification. No new selector or fitting was part of this draft.

Proposed primary success estimand: paired terminal-success difference equally averaged over tasks/conditions. Proposed cost estimand: episode-mean complete-query latency, then equally averaged over conditions/tasks, **startup included**. Also report natural query-weighted latency, episode wall time/query count, and separate controlled matched-input timing. An estimation study is not an unchanged-success guarantee.

Planning time: approximately **16–20 GPU hours**, not the earlier two-hour development screen estimate. Provisional bounds discussed two sessions ≤24 h each, one coordinated GPU0, ≤23,552 MiB, ≤2 GiB artifacts, about 70,000 calls. Worst-case schedule accounting subtotal 68,872 calls excluded any additional qualification calls; exact freeze would need to reconcile those.

Simple independent-pair approximations at N=400 suggested success-difference interval halfwidths about 2.19/3.10/4.38 pp for discordance probabilities 0.05/0.10/0.20. These do not guarantee precision under task heterogeneity/clustering.

**Status: DRAFT, NOT EXECUTABLE, NOT FROZEN, NOT LAUNCHED.** The latest conversation shifted to requesting advisor guidance. No task should launch this merely because the design exists.

### 12.2 Work still needed if that route is deliberately resumed

1. State the original question/contribution and why it matters relative to published work.
2. Resolve historical exposure including remote-only outputs; certify exact input manifest without opening protected labels improperly.
3. Fix estimator, confidence intervals, clustering, primary/secondary contrasts, and multiplicity rules prospectively.
4. Implement and CPU-test a new runner/config/analyzer/independent verifier and exact count schedule.
5. Freeze source/config/input/model hashes, controls, ordering, session handling, resource caps, and stop rules.
6. Obtain authorization for the exact study and coordinate selected-GPU availability.
7. Complete the unchanged blinded schedule; no post-hoc population expansion or arm deletion.
8. Verify, analyze, independently reconcile, and interpret all outcomes. Only then write claims supported by the study.

### 12.3 What not to do next automatically

- Do not train another corrector or add a confidence gate without a new rationale and protocol.
- Do not reopen training-free stale K/V routing just because a new chat starts.
- Do not remove the strong adaptive256 comparator.
- Do not turn the rejected backend idea into a “new method” with a different name.
- Do not promise a positive-paper deadline contingent only on one run.
- Do not start a manuscript claiming a solution before the contribution is identified.
- Do not treat a universal “500 episodes per suite and three seeds” as mandatory project law. That earlier assessment was superseded by a more claim-specific V2 planning discussion; precision/generalization requirements must fit the actual claim.

### 12.4 Practical current next step

Help Ved finalize the concise advisor update if requested, and use the advisor's response to select a concrete new scientific target or scoped empirical contribution. Preserve and explain the positive measured operating points now. Be explicit that selecting the next contribution is the remaining bottleneck, not a missing cosmetic paper section.

---

## 13. Restart procedure and verification

### 13.1 Minimal reading order for a new chat

1. This file, especially Sections 1,2,4,7,12.
2. Actual repository `AGENTS.md` in full.
3. Latest top entry of `PROJECT_STATUS.md` and latest `docs/DECISIONS.md` entries.
4. `reports/ADAPTIVE_SCREEN_V03_RESULT.md` and `reports/BACKEND_PRESERVATION_CONTRIBUTION_AUDIT_V1.md`.
5. `reports/CURRENT_FRAME_ROBOT_PILOT_RESULT_V1.md`, gripper erratum, and `reports/CURRENT_FRAME_CORRECTION_DIAGNOSTIC_V2.md`.
6. Exact applicable protocol/config/source when addressing a specific task.
7. Historical reports only as needed, using Section 6 to navigate.

Do not spend the first turn loading the entire multi-month archive or rewriting an old plan. Do not accept old “next phase” statements without checking newer decisions.

### 13.2 Safe local evidence checks

The following are read-only local checks; run from the actual local repo. They do not authorize server access or GPU work:

```sh
cd /Users/veddwivedi/Documents/VLA/SAVR
git status --short
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
python3 -B scripts/verify_current_frame_robot_copy.py
python3 -B scripts/verify_adaptive_screen_v03_independent.py
```

If a verifier fails, preserve evidence and diagnose the mismatch; do not overwrite the expected digest, loosen a gate, regenerate frozen artifacts, or pretend the run is missing. Verifiers should be independent of worker/analyzer computation where possible.

Do not blindly run a whole test suite if it could create/overwrite existing historical output paths, open sealed labels, or import a GPU runtime. Inspect the exact test scope first. CPU execution should hide CUDA and disable unnecessary bytecode/thread/cache side effects, using protocol-specific settings.

### 13.3 What to report on restart

State the verified current checkpoint, what the user actually requested, what can be done now, and what requires a separate decision/authorization. Say whether verification was local-only or remote. A new chat does not inherit the previous assistant's live terminal, automations, or mental context just because it shares project files.

### 13.4 Freeze/blinding checklist for an eventual approved run

Before launch: exact claim → independent population → comparator/controller/backend parity → config/source/input hashes → immutable schedule/counts → CPU and real-model qualification → approved resource availability → exclusive outputs.

While active: owned process health, counts only, artifact bytes, elapsed time, authorized aggregate GPU telemetry. No outcome peeking.

At stop: immutable complete summary required; otherwise technical diagnostics only, preserve, stop without retry, sync status/evidence, report.

At completion: reconcile counts/hashes/frozen sources/caps first; CPU analysis; independent all-arm/gate audit; sync local copies; stop at approved checkpoint. A positive development signal is not a paper-ready confirmation.

No current task authorizes recreating historical heartbeat monitors. Use a monitor only for a newly authorized running job and preserve quiet-on-ordinary-progress notification intent.

---

## 14. Paper and poster archive

The user explicitly set aside URTC-specific paper/poster activity when returning to research mode. Do not restart poster redesign, logo work, font sizing, journal template selection, or URTC submission planning unless asked.

Historical PDFs include:

- `output/pdf/SAVR_Negative_Results_Paper.pdf`.
- `output/pdf/A_Negative_Result_for_Training-Free_Temporal_Visual_Caching_in_OpenVLA-OFT.pdf`.
- `output/pdf/A_Sequential_Negative_Study_of_Temporal_Visual_Reuse_in_OpenVLA_OFT_Inference.pdf`.

Historical LaTeX sources include `output/tex/temporal_reuse_tro/paper.tex`, its `references.bib`, and `output/tex/Temporal_Visual_Reuse_Paper_LaTeX_Source/`. Duplicate source/export folders may exist; establish the intended active source before editing.

The manuscript sent originally was titled *A Negative Result for Training-Free Whole-Prefix Visual Caching in VLA Inference*. Later broader studies covered multiple reuse boundaries. These historical drafts predate some important action-readout/attention-runtime qualifications. **Do not assume every old statement remains valid for a new paper.** Audit affected claims against this guide and original reports.

Academic writing requirements: direct scientific prose, understandable math/architecture, defined terms, no chat-specific slang/phase codes as unexplained method names, no inflated safety/novelty/generalization statements. Use actual evidence and meaningful figures. No performative citations. Add uncertainty where appropriate and complete bibliography/venue/data-availability requirements for an actual submission.

A 1,160 count applies to the primary whole-prefix study only, with a separate 50-episode timing pilot. Do not reuse that total for the whole multi-stage project. Some later studies are calls/offline samples/systems timings rather than robot episodes; no casual grand-total summation.

No final positive-method manuscript or journal submission is complete. The empirical blueprint is a plan, not a paper-ready contribution.

---

## 15. Latest advisor email draft

**Draft only, not sent.** This follows Professor Yuan's September 7 guidance and explains the current-frame change rather than repeating the original negative report. It asks for help identifying a publishable contribution. It does not promise another run or claim a novel solution. No new paper attachment is implied.

**Subject: SAVR progress update and guidance on the next step**

Dear Professor Yuan and Mr. Yang,

I hope you are both doing well! Following your feedback, I explored reducing computation while keeping both camera images current, rather than reusing older visual information.

The new approach processes fewer image representations inside the model. It reduced measured inference time by approximately 15%, while completing 117/120 robot trials, compared with 120/120 for the original policy. These trials covered 40 tasks.

I then trained a small module to correct the compressed model's actions. Its predictions became approximately 5.3% closer to the original policy's predictions on saved observations, but this did not improve overall robot task success.

I also compared our approach with an adaptation of the published VLA-Pruner method. In a separate 40-trial evaluation, the original policy, our compression approach, and the stronger adaptive method each completed 39/40 trials. Our approach reduced inference time by approximately 15%, while the adaptive method reduced it by approximately 16%.

We now have encouraging speed improvements with high observed task success. However, I am getting stuck on how to develop a sufficiently original research contribution: compression is already established, and our correction module has not improved robot performance. I would greatly appreciate your guidance on the specific limitation we should address next to move toward a publishable solution.

Thank you both for your time and support!

Best regards,
Ved Dwivedi

**Technical interpretation behind the simplified wording:** “closer” means lower normalized action L1 to dense predictions, not higher robot success; the 15% figures are controlled complete-query time reductions; 120 and 40 are separate development populations. If revising for precision, retain these meanings.

---

## 16. Evidence identifiers

These identifiers help locate/authenticate the most important evidence. They are full SHA-256 digests, not abbreviated display hashes. Exact artifact filenames/complete inventories are in each immutable summary. A copied digest is not a replacement for verification.

### 16.1 Current-frame robot pilot

```text
configuration: 9e3f6728be56821d5da73185f310f1119adda32e2a4fc6e50672cc2e71612c8e
worker_summary: 699de8203703ea508ab5668584ed2d7d52452f53af6dc20a539809a50221588e
analysis:       4ef6ae0873f5c0d24b5caffcd223a9673d067f951080af4d2492da093e4fba70
```

Fresh local independent verification passed October 3.

### 16.2 Feature collection

```text
configuration: e810cb53f7a836d7a4d9dbb95ee406e919355f70a215e6788d5b760b7863a124
input manifest:2fd13ced3e5a9a2b810eb299a5c1667f8c1de3ee2c5487a4a0683e61847d9ae8
worker_summary:7bde8cd59846d5f5b54c7b9a8302b30ebe74ef13606eb92aeb754465d5b1a8c3
analysis:      ae9ba225dce1d14318b8b7d6c5493657eec307a6142cab2df6f48e27e7eb2273
```

Previously verified collection; not fully rehashed in this snapshot turn.

### 16.3 Adapter fit and research weights

```text
configuration: cafa427d5e5e4a5e5a88f2f93ac85b39b49871b5a689a009f89bc2f6bfc8c031
worker_summary:c863b31bff5ea15f4c93bed21fda430734711642df56aa3e791e9a402c466627
analysis:      d9ed6108eef40d4c4466343e04baea42d58a88ba5e06a44368f89bfcbb64a2de
action-only:   8a01608b9690bf2e62017968fbc2ab4afd86325503ccfd3322994ab1838b8640
visual:        15489080d2752d3cf90a7b9cb016cb4c95c9a395b335b400d63e74c33f6bb951
shuffled:      1524bbd2527a4ad362c1d4dff7ab33a17c2bd2cbfded8bb523c507cd75627cc9
```

Use fit report with the gripper erratum. Do not alter frozen fitting metrics in place.

### 16.4 Adaptive v03 qualification and screen

```text
S0 configuration:  6f3a36decbb18706aa6cf39f2a727eb64601fd53de8efa19b334bf87ea10fc63
S0 worker_summary: 55af20761ff4bbba9a30229b53bee50db6a9fedffbc7f21eb81145a098832948
S1 configuration:  ae1cbd7d0dbaccb3ff14a335cae6b451baa2b6ea62f25968e52c5ea5e3c8c768
S1 worker_summary: e60854bb0b443e9a093c8b58668fc95788ec9a201398d60e1e35ff3c492ee72a
S1 analysis:       0687bb3ef3efd868226c0b61b40be81f7694e37b87e6bf772e85710cc8ee9b58
```

Fresh independent S1 verification passed October 3. Other historical hashes should be retrieved from their own authenticated summaries rather than inferred from filename/version labels.

---

## 17. Open questions and things not to repeat

### 17.1 Genuine unknowns

- What original question/solution would Professor Yuan consider a meaningful next research contribution?
- Is a carefully scoped comparative empirical study worth pursuing, or is new method development necessary for the intended venue?
- Why did small offline action improvements not increase closed-loop success? Teacher mismatch, correction of adequate actions, distribution shift, capacity, and trajectory effects are hypotheses; deployment actions were not recorded sufficiently to decide.
- How stable are current gains across genuinely unexposed states, training seeds, tasks, hardware/session variation, and appropriate confidence analysis?
- What success cost can be justified for a specific application/claim? It must not be chosen after seeing losses merely to call them acceptable.
- Which candidate simulator states are demonstrably untouched by all historical actual executions?
- What is the current remote resource availability, GitHub visibility, and advisor response? This file does not freshly verify them.

### 17.2 Explicitly closed or unsupported shortcuts

- Historical training-free whole-prefix method did not pass the required useful-reuse constraint.
- ACR's saved scene work did not yield latency gain against batched dense in its matched study.
- PAIR P4B did not pass reliability gates; no P5 positive robot result exists.
- CAC research adapter was never trained after the cached substrate failed its headroom stage.
- The 0/120 historical cached result cannot be presented without runtime/readout qualification.
- Current visual corrector has not improved overall robot success.
- No cheap deployable beneficial correction gate has been demonstrated.
- Native-backend score extraction is not a new general contribution.
- Fixed compression did not beat adaptive256 in the latest development comparison.
- No confirmation is frozen, running, or complete; no positive-method manuscript is ready.

### 17.3 Avoid recurring reasoning errors

Do not confuse original method failure with every variant failing. Do not confuse a real speed gain with originality. Do not confuse high success with proven success preservation. Do not confuse repeated state-0 results with independent confirmation. Do not confuse lowered L1 to a teacher with better robot control. Do not confuse pipeline overhead removed with a new algorithm. Do not turn “plausible” into “will work.”

Do not repeatedly present a replacement idea as highly likely to yield a positive paper before code/literature/resource review. If a decisive flaw is found, record its impact clearly and revise the scientific premise, rather than add more reassuring adjectives.

---

## 18. Updating this document and new-chat prompt

### 18.1 Maintenance

Keep one unambiguous latest checkpoint at the top. Preserve historical evidence and errata, but do not stack contradictory “LATEST/running” blocks. Update this file after a meaningful decision, completed study, technical stop, or correction.

For every update record: date, user authorization/scope, exact changed files, whether local or remote work occurred, evidence identities, verification, supported interpretation, and next authorized action. If a plan has not run, continue to label it as a plan.

This file cannot recreate implicit live tool sessions. A new chat should verify scoped current facts rather than trust historical PIDs, cached UI state, or agent memory. It should not need the entire original conversation to make a safe start.

### 18.2 Copy this into a new chat in the Rutgers Research Project

> Continue the SAVR research project. First read `/Users/veddwivedi/Documents/VLA/SAVR/SAVR_MASTER_CONTEXT.md` completely and the repository's `AGENTS.md`. Then read the latest result and contribution-audit reports listed in the start-here section. Treat historical running entries/PIDs and old approvals as obsolete unless the latest checkpoint explicitly says otherwise. Preserve all frozen evidence and unrelated changes. Do not launch experiments, access unrelated university resources, send emails, or push GitHub without the appropriate current authorization. Briefly tell me the actual current research checkpoint, what positive results we have, what remains unresolved for a publishable contribution, and the next action consistent with my latest instruction. Keep the response concise and understandable.

### 18.3 Snapshot creation record

This guide was created October 3 from local repository instructions, result reports, protocols, corrected diagnostics, contribution audits, independent verifiers, and the supplied conversation history. Local robot/adaptive verification passed, all 88 explicit project-path references resolved, and eleven listed config/summary/weight digests matched. A short latest-checkpoint pointer was added to `OPENCODE_HANDOFF.md` and `PROJECT_STATUS.md`; their historical records were preserved. No TITAN connection, new evaluation/training, manuscript modification, email transmission, Git operation changing history, or GitHub push was performed for this documentation task. Synced project `sources/` were not modified.
