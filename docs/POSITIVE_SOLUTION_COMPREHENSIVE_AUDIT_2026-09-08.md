# Audit and decision: a positive-solution direction after SAVR through CAC

Date: 8 September 2026  
Status: research recommendation, not an experiment authorization  
Scope: research evidence, implementation validity, literature, resources, and the next decision. URTC presentation work is excluded.

## 1. Decision in plain language

**Continue the research, but do not resume the failed recursive-cache configuration or promise that another adapter will produce a publishable result.** There is a defensible next hypothesis, not yet a demonstrated positive method.

The recommended direction is **current-frame visual compression with a small learned action-correction module**. Both camera images would be processed at every policy query. Fewer of their visual tokens would pass through the expensive language backbone. A small downstream module would use the current visual features to correct the compressed policy's action. The large model would remain frozen, and its activations would not need to be retained for backward propagation.

The scientific question is specific: **can inexpensive, current visual information recover the control accuracy lost through spatial compression, without retraining the large backbone or consuming the inference saving?** This directly follows the project's difficulty with stale information, but removes recursive cache corruption from the first learning experiment.

There are two prerequisites. First, explain the unexpectedly weak latest dense baseline. Second, establish a faithful published comparator. Published studies report useful caching and compression on OpenVLA-OFT; our results cannot be interpreted as showing that these approaches generally fail. Only after those checks should we train a small prototype.

The potential contribution is a resource-efficient adaptation procedure and a controlled closed-loop study, not the invention of token pruning, action residuals, or distillation. If a published method already provides the same procedure, or if a simple baseline performs equally well, the novelty claim must narrow or disappear. A positive result and acceptance by a journal are separate uncertainties.

## 2. What this audit did and did not establish

The audit examined the repository's experiment reports, selected machine-readable results and manifests, semantic audit, data inventory, inference helpers, simulator worker, launch configuration, and current project-local runtime metadata. It checked relevant primary literature and official implementation documentation. Historical evidence was not rewritten. No new model inference, training, simulator episode, GPU allocation, or retry was launched.

The local research directory is `/Users/veddwivedi/Documents/VLA/SAVR`. Remote inspection used only `ssh titan` and the project under `/home/ved/SAVR`. The earlier poster and manuscript summaries were not treated as the source of experimental truth.

This is a decision audit, not a full independent rerun or a proof that every line of code is correct. Hash agreement authenticates which artifact was read; it does not establish scientific correctness. Similarly, a passing tensor comparison does not establish full-episode equivalence. Remaining empirical questions are named below rather than hidden behind a readiness label.

### Evidence hierarchy

1. Completed, reconciled outcomes from the corrected implementation support claims about those exact runs.
2. Historical measurements affected by a semantic bug remain records of the historical implementation, not authenticated evidence about the released policy.
3. Technical stops establish an execution limitation, not a task-performance result.
4. Published results support feasibility in their own settings. They are not forecasts for TITAN.
5. The proposed method, numerical planning targets, and resource estimates below are explicitly prospective.

## 3. What we actually learned

| Investigation | What was tested | Main evidence | Defensible interpretation |
|---|---|---|---|
| Whole-prefix SAVR | Reuse the complete two-camera visual representation, with increasingly conservative refresh rules | 1,160 primary episodes plus 50 timing-pilot episodes. Nine permissive configurations: 34.69–83.68% reuse, 0–52% success. Conservative configuration: 69/70 successes, 9/944 reused queries | The tested rules did not give useful reuse while preserving success in that setup |
| Asymmetric camera refresh | Reuse the scene-camera representation while updating the wrist view | Matched batched dense and reuse: 67/70 successes each. Scene reuse 25.2434%. Mean query time 1188.2837 versus 1190.9691 ms | Observed success was maintained, but the fair batched comparison showed no end-to-end acceleration |
| Subsequent camera-controller redesign | Replay six refresh candidates and build a static executor | V4 replay failed its declared gates, including a horizon/streak specification contradiction. V5 CPU executor contracts passed | Neither stage established a new closed-loop or GPU-latency result |
| Fine-grained executor development | Selective internal computation and GPU execution optimizations | The final V10 attempt stopped on execution/compiler problems, before completed-query measurements or episodes | It was a technical stop, not a negative task-success experiment |
| BRACE | Finer cache execution, intended to support adaptive reuse decisions | Completed numerical screen failed action-parity criteria; an independent later audit found a shifted action readout | Historical action-quality conclusions are provisional for the released policy |
| PAIR | Learn which reuse interventions are tolerable from paired action effects | Exploratory signal weakened in a separate confirmation. Confirmation rank correlation 0.2001; tail-risk improvement 3.14%, with lower bound −1.07% | The tested historical router did not meet its criteria. The later readout bug limits generalization to correct OpenVLA-OFT |
| CAC systems qualification | Expose adapter inputs and test corrected recursive execution | Corrected 97-call test: 22.5969% median gross cycle saving; small adapter forward and feature extraction measured; semantic controls passed | There was engineering headroom, not evidence that learned correction worked |
| Corrected CAC closed-loop screen | Dense versus the corrected recursive cache before adapter training | 120 paired conditions: dense 68/120, cache 0/120. All 12 reconciliation checks passed | This exact untrained cached policy failed badly. The frozen protocol stopped before training |

Internal sources: [E1]–[E9] below. The populations, metrics, and implementation versions differ. They must not be pooled as repeated trials of a single method.

### 3.1 The original visual-similarity assumption was inadequate

The SAVR archive reports that averaging visual changes across cameras could conceal wrist-camera changes. First action differences also appeared when reuse began. These observations support examining camera-specific information and action sensitivity. They do not independently identify the causal contribution of each factor, nor do they establish a general impossibility result for reuse.

### 3.2 The camera experiment exposed a timing-comparator problem

Scene-camera reuse looked better against sequential full refresh, but that advantage disappeared against batched full refresh. Its 1190.9691 ms mean was approximately 0.226% slower than the 1188.2837 ms batched reference. A component-level vision saving was insufficient. Future claims must use the strongest matched implementation and include every operation needed to produce an executable action. [E2]

### 3.3 BRACE and PAIR are not clean evidence that learning cannot work

The independent semantic audit found that the official regression head used hidden states corresponding to `[-58:-2]`, while the custom path used `[-57:-1]`. The latter shifted all 56 action-readout positions. Earlier controls compared several executions of the same faulty helper and therefore agreed with one another. Action-attention and instruction-position bookkeeping also required correction. [E6]

This matters twice. It invalidates treating internal agreement as an independent reference check. It also weakens earlier statements that the project had established a fundamental inability of cheap features or learned routing to identify useful reuse. PAIR's confirmation was disappointing, but it was performed on a policy implementation later found not to match the released action readout. Repeating all of PAIR is not the recommended next investment; acknowledging this limitation is essential.

PAIR already involved a learned router. Therefore, “we have only tried training-free methods” is inaccurate. The more precise statement is that **we have not completed a corrected, closed-loop evaluation of a trained action-repair policy**.

### 3.4 CAC did not test the proposed trained solution

The corrected cache screen completed normally. Dense success by suite was Spatial 14/30, Object 27/30, Goal 13/30, and Long 14/30. The cache succeeded in none of the 120 conditions. The predeclared stopping rule was correctly applied. [E8]

However, the result is not an experiment in which a trained correction adapter failed. Training never began. The protocol required an already viable cached starting policy, and this one did not qualify. That was a resource-allocation decision, not a theorem that no adapter could learn from a poor starting policy. The old report's categorical language about what can be trained should be understood in that limited procedural sense.

There is still a practical reason not to resume it: repairing a policy with zero observed successes and recursive, mixed-age internal states is a much harder initial learning target than correcting moderate, nonrecursive compression error.

### 3.5 Some failures came from our experimental process

The V4 report records a direct contradiction: prose treated horizon two as preventing consecutive reuse, while the implemented controller allowed two reuses and the frozen gate required a maximum streak of one. Applying the frozen stop preserved integrity, but did not turn the contradiction into evidence against a scientific mechanism. The protocol needed a semantic example before execution. [E12]

V5's CPU static-executor checks then passed, while later real GPU capture failed. CPU parity did not establish compatibility with the actual GPU operations. The last V10 failure occurred well below the memory cap; treating every later stop as insufficient GPU memory would be wrong. [E3][E12]

Across these examples, independent reference checks arrived too late, readiness assertions exceeded what the tests exercised, and increasingly elaborate infrastructure delayed a decisive learning experiment. Immutable artifacts and fail-closed rules are strengths worth retaining. They cannot replace a simple reference implementation, a faithful end-to-end test, or a clear distinction between technical feasibility and scientific effectiveness.

## 4. The unresolved dense-baseline problem comes first

The latest dense result was 56.67%. Official documentation reports 96.8% average success for the combined four-suite policy in the authors' evaluation. These are not matched experiments: hardware, initial conditions, evaluation breadth, seeds, and runtime differ. Nevertheless, the discrepancy is too large to silently accept when defining a new improvement claim. [E8][^1]

The static worker inspection found the expected two images, center crop, proprioception, eight-action execution, ten initialization steps, and official action postprocessing. It did not identify a demonstrated cause of the low dense success. Exact single-observation action parity rules out some helper errors; it does not validate the entire observation-to-environment loop.

Current project-runtime metadata read on TITAN:

| Component | Installed version |
|---|---|
| PyTorch | 2.2.0+cu118 |
| Transformers | 4.47.0 |
| Tokenizers | 0.21.1 |
| robosuite | 1.4.1 |
| MuJoCo | 2.3.7 |

The authors document their custom Transformers 4.40.1 stack. The older repository lock also does not describe every aspect of the active compatibility environment. This is a **candidate source of discrepancy, not a diagnosis**. A pinned version string alone is insufficient: custom source hashes, attention implementation, checkpoint code, and loader behavior matter. [^1]

The next baseline check must compare two genuinely independent paths on the same already-consumed development conditions: the released evaluator and the project's corrected evaluator. Capture camera preprocessing, prompt IDs, normalized state, action-head inputs, normalized actions, executed actions, and simulator transitions at the first divergence. Replaying exactly the same observation and then exactly the same action separates policy and environment differences. Compare reset behavior, environment reuse, seed handling, gripper conversion, action queue, and maximum episode length.

Use project-local environments without replacing the historical one. Do not modify the checkpoint's files as a normal loading operation. If an upstream-compatible stack is required, its installation is a separately authorized implementation step. Do not rerun until the result resembles a published number. If both independently qualified paths remain weak, document that finding and decide whether this checkpoint/runtime is a suitable reference before investing in a new method.

## 5. Literature: what is established and where novelty remains possible

The literature search covered adaptive caching, current-frame token compression, small correction heads, distillation, early exit, and closed-loop data collection. The table distinguishes direct overlap from supporting evidence. The recent arXiv studies are preprints unless independently identified otherwise; their results have not been reproduced here.

| Primary work | Relevance to this decision | Consequence for our claim |
|---|---|---|
| OpenVLA-OFT, RSS 2025 [^2] | Efficient continuous action chunks and the actual base architecture | Compare against OFT itself, not an older slower decoding baseline |
| VLA-Cache, NeurIPS 2025 [^3] | Published adaptive token caching | Generic temporal reuse is established; our custom cache is not automatically a faithful reproduction |
| LightVLA, ICRA 2026 per official repository [^4] | Learned visual-token pruning on OFT | “Learn which tokens matter” is not new. Official training requirements exceed our default single-card budget |
| Compressor-VLA, November 2025 preprint [^5] | Instruction-conditioned global/local visual compression on OFT | A learned resampler or camera-detail argument alone is not an adequate novelty claim |
| LAC, January 2026 preprint [^6] | Learns a token selector and cache ratio with a frozen backbone and task gradients | “Frozen backbone plus learned caching” is already studied; frozen weights do not eliminate backward activation memory |
| AC²-VLA, January 2026 preprint [^7] | Action-conditioned routing, pruning, skipping, and self-distillation | State/action guidance and dense-teacher distillation cannot be claimed as inventions |
| Action-JND, August 2026 preprint [^8] | Action-tolerance learning for caching/pruning, including separate camera estimators on OFT | Particularly close to earlier ideas. Reports useful OFT caching, contradicting a broad dismissal based on our custom cache |
| A2C2, September 2025 preprint [^9] | Current observations drive a small residual correction of delayed action chunks | Strong conceptual support and major prior-art overlap. Its problem is chunk delay, not same-query spatial compression |
| Latent Bridge, May 2026 preprint [^10] | Predicts changing features and trains on learner-induced states | Cache reconstruction and one-round distribution correction already exist; transferring its results to OFT is unjustified |
| DeeR-VLA, NeurIPS 2024; Shallow-π, January 2026 preprint [^11][^12] | Early exit and depth distillation | Early exit is a credible alternative, not a fresh novelty claim, and introduces another readout/training boundary |
| DAgger, AISTATS 2011 [^13] | Addresses the distribution change caused by a learner's actions | Offline error must not substitute for closed-loop evaluation; teacher queries on learner states are established methodology |
| DivPrune, CVPR 2025 [^14] | Current-token diversity selection without training | A useful comparator for token selection, but its original multimodal-language results do not prove robot success |

Two concrete comparisons illustrate the resource and evidence limits. Compressor-VLA trains compressors and the action head along with backbone adapters on eight A100 GPUs for 150,000 steps. It supports the usefulness of learned current-frame compression, but not a claim that our downstream-only approximation will achieve the same result. A2C2's LIBERO experiment uses SmolVLA and a 32-million-parameter correction module, not our OFT compression setting. [^5][^9]

Action-JND reports an OFT dense baseline of 95.05% and several caching configurations retaining similar success. That is important counterevidence to our historical narrative. It is not proof our implementation is wrong or that reproducing the paper will be straightforward. Original VLA-Cache and later OFT adaptations must be distinguished and tied to exact public code. [^8]

### Novelty statement that is supportable now

The proposed study would investigate **downstream-only recovery of current-frame compression error, with a fresh visual bypass, actual hard-compressed training inputs, and closed-loop evaluation under a single 24-GB-class GPU constraint**. In the primary works examined, this exact study was not established. That is a bounded search finding, not a priority guarantee.

The individual ingredients are established. A publishable contribution would require showing that the particular inexpensive training boundary produces useful results, explaining when the visual bypass is necessary, and comparing against relevant alternatives. A residual attached to pruning without convincing ablations could be judged an application of existing methods. We should solicit methodological feedback before a large campaign, not only after writing the paper.

## 6. Why prioritize this direction over the alternatives?

| Option | Advantage | Main obstacle | Decision |
|---|---|---|---|
| Another handcrafted refresh/router rule | Reuses much code | Repeats tested assumptions; historical evidence compromised; novelty crowded | Do not prioritize |
| Resume CAC on the zero-success recursive cache | Closest to previous proposal | Severe initial deficit, changing cache provenance, unresolved dense reference | Do not resume under the old protocol |
| End-to-end cache-aware OFT adaptation | Can change the representation, not just its output | Backward memory, training cost, implementation complexity, close prior work | Reserve for a separately justified resource expansion |
| Current-frame compression plus downstream correction | No recursive state; small training graph; direct connection to observed information-loss problem | Compression may discard information the correction cannot recover; novelty conditional | Preferred bounded hypothesis |
| Fresh early-exit policy distillation | Avoids temporal staleness and can remove many layers | New exit semantics; strong existing work; larger behavior-learning change | Secondary option, not a simultaneous project |
| Replace OFT with a small VLA and retrain | Training may become easier | Changes the research question and ecosystem; needs additional validation | Explicit broader pivot only |

This ranking is an engineering and scientific judgment, not a measured probability. The recommendation removes one known source of difficulty while keeping the robot, images, dataset, checkpoint, and action interface familiar. It does not make compression intrinsically harmless.

## 7. Minimal proposed method

### 7.1 Architecture and inference

Let the current observation be two images and proprioception, with instruction \(\ell\). The frozen visual encoder and projector produce \(V_t\in\mathbb{R}^{512\times4096}\). A deterministic selection operator \(P_k\) keeps \(k\) current visual tokens for the large backbone. All instruction, proprioception, and structurally identified action-readout positions remain present.

The compressed backbone and original action head produce normalized action chunk \(A_t^k\in\mathbb{R}^{8\times7}\) and action-head penultimate features \(Z_t^k\in\mathbb{R}^{8\times4096}\). A small corrector predicts

\[
\widehat A_t=A_t^k+g_\theta(Z_t^k,A_t^k,V_t,s_t,e(\ell)).
\]

Every tensor describes the current query. There are no reused K/V tensors, mixed ages, cache reset horizons, branch rollouts, or learned refresh gates. Only the existing eight-action chunk is executed before the next observation query. The current-image features sent directly to the corrector are not run through the large backbone a second time.

```text
Current scene + wrist images → frozen vision encoder → current visual tokens
                                                       ├─ fewer tokens → frozen large model
                                                       │                  → base action + action features
                                                       └─ lightweight visual projection ─────────────┐
Current state + instruction ────────────────────────────────────────────────────────────────────────┤
                                                                                  small corrector ←┘
                                                                                         ↓
                                                                                 corrected action chunk
```

Start with a fixed compression budget and no adaptive routing. Two modest diagnostic budgets, 384 and 256 of 512 tokens, are proposed; these are experimental settings, not discovered optimal values. Use a documented per-camera spatially stratified selection and an established diversity-selection comparator. Preserve original positions and camera ordering. A fair evaluation cannot rely solely on beating a deliberately weak uniform selector.

The first corrector can use the existing 256-wide, two-block cross-attention design as a starting implementation, with eight action queries. Remove all cache-age/source-delta machinery. Retain current camera and patch-position information. Initialize the output to zero so that the initial corrected policy exactly reproduces its compressed base. Use all current patch features initially; pooling them aggressively would introduce a second information bottleneck before testing the hypothesis. A smaller pooled variant is an ablation, not an assumed equivalent.

The final parameter count and latency must be remeasured. The old 5.07-million-parameter CAC timing does not transfer automatically to a different context length and dtype.

### 7.2 Training target and data

For each training observation, run an independently qualified dense teacher and the **actual hard-compressed** student. Store their detached inputs/features and dense target action. Train only the small corrector:

\[
\mathcal L(\theta)=\mathbb E_{o\sim\mathcal D}
\left[\frac1{56}\|A^k(o)+g_\theta(o)-A^D(o)\|_1\right].
\]

Use the released normalization and action execution conventions, with gripper errors tracked separately. No generic “safety” interpretation follows from this regression objective. Teacher labels are targets for behavioral preservation, not ground-truth optimal actions.

Begin with this one objective. Do not initially add a critic, uncertainty router, layer scheduler, reconstruction loss, reinforcement learning, or multiple auxiliary objectives. Include dense-input identity examples so that the correction can learn not to alter an already correct computation. Preserve the reference policy's output conventions rather than adding an untested clipping or smoothing rule.

After an offline prototype, collect a bounded set of simulator trajectories executed by the corrected student. Query the dense teacher on those **same visited observations** without executing the teacher's action. Add the resulting examples for one prespecified refinement round. This is a DAgger-style step, not a novelty claim. It reduces one distribution mismatch, but a teacher may itself fail to recover from states outside its training experience. [^13]

Demonstration actions remain useful for auditing teacher disagreement. They cannot be substituted as action labels for arbitrary off-demonstration learner states. Chunk horizons, no-op handling, terminal truncation, and image orientation must follow explicit contracts.

### 7.3 Why the training boundary fits the resources better

Frozen weights upstream of a learned input module can still require a large backward graph. Here, all learned parameters are downstream of detached backbone outputs. Dense and compressed forward passes can run sequentially, and adapter training can run separately from the large model using numeric feature records. This is the reason for the proposed memory advantage, not merely the adapter's parameter count.

### 7.4 Main ways it could fail

The compressed backbone may lose instruction grounding or subtle geometry that the small corrector cannot reconstruct. Teacher imitation may fit demonstrations but fail after the student changes its own observations. The corrector may memorize tasks or robot states without using images. Gripper switching may remain wrong despite small average action error. The additional visual projection may consume the entire latency benefit on TITAN. A stronger published selector may outperform the new method without training. Any of these findings would defeat or substantially narrow the positive-paper claim.

An ablation removing the fresh visual input tests whether the bypass actually matters. A dense-plus-corrector control tests whether any improvement is generic adaptation rather than recovery from compression. Matched teacher data and training effort are required in both comparisons.

## 8. Resource and data budget

TITAN is a shared university server. The existing operating boundary is one explicitly coordinated GPU, with a 23-GiB/23,552-MiB cap. The latest run used 16,301 MiB; the corrected adapter systems test peaked at 16,789 MiB. Those measurements support forward feasibility, not guaranteed training feasibility. No current allocation or unrelated process was inspected in this audit. [E7][E8]

The project has 40 task files, 2,000 demonstrations, 338,575 actions, and 41,447 eligible eight-action query records. The recorded split is 1,400 training, 320 calibration, and 280 locked trajectories. Previous PAIR calibration use means the 320 trajectories are no longer a pristine final test. Before reuse, verify the exposure ledger for the 280 locked trajectories and simulator initial conditions. Demonstration IDs and simulator state IDs must not be assumed to correspond. [E9]

| Resource | Budget implication |
|---|---|
| Demonstrations already present | Approximately 31.5 GiB of source files; no new acquisition needed for a first prototype |
| Full current visual features | 512 × 4096 × 2 bytes = 4 MiB per query, before action features and metadata |
| Initial 4,000-query feature set | Approximately 16 GiB of visual features; budget 20 GiB including overhead |
| All 41,447 queries | Approximately 162 GiB for visual features alone; do not materialize by default |
| Approximate 5M-parameter adapter | FP32 weights, gradients, and two Adam moments alone total about 80 MB; activations and temporary buffers are additional |
| Dense plus compressed labels | Two sequential forwards per observation; generation may cost more than adapter optimization |
| Full OFT LoRA recipe | Official example needs about 25 GB even at batch size one, above our 23-GiB limit [^1] |

Use FP32 for the small adapter initially rather than assuming native BF16 training support or compiler compatibility on this GPU generation. Keep the large model in its qualified inference configuration. Avoid adding compilation, CUDA graph capture, quantization, model sharding, or simultaneous teacher/student model copies to the first study.

At 1.2–3.0 seconds per forward as a planning range, 8,000 forwards for 4,000 paired labels would take roughly 2.7–6.7 GPU-hours before data overhead. This is an estimate, not a new benchmark. The completed 240-episode CAC screen took 7.17 hours, so robust evaluation can require days. Measure throughput on a small fixed batch before setting a calendar. A read-only filesystem check showed roughly 308 GB available on the shared project volume; that is not an allocation or permission to consume it.

## 9. Staged plan, with decisions that end branches

All proposed thresholds below are decision-design suggestions. They do not modify historical frozen gates. Exact schedules, budgets, and analysis code must be frozen before the corresponding new outcomes are opened.

### Stage 0 — Establish an independent reference and reproducible snapshot

Audit the baseline as specified in Section 4, using consumed development states. Resolve checkpoint identity, runtime sources, preprocessing, action readout, simulator settings, and seed policy. Include full-episode released-evaluator comparison, not only an anchor tensor. Preserve all original evidence if a discrepancy is found.

Also create a reviewed source/manifest snapshot. Both the local and remote tracked HEAD observed in the audit were `0a274b6`, while many later PAIR/CAC files are untracked and several tracked files are modified. This does not prove those files are absent from every GitHub branch, but it means a commit ID alone does not identify the current experimental source. Do not blanket-stage the repository. Include relevant code, tests, source hashes, and configurations deliberately.

Exit: a defensible dense reference and an authenticated source snapshot. If the low dense result remains unexplained, stop method development and report the unresolved baseline. A debugging fix alone is not the novel paper contribution.

### Stage 1 — Establish a useful compression target and published comparator

Run a short, frozen development screen of dense inference, two modest current-frame compression budgets, and a relevant published OFT comparator. Verify all-token retention equals dense inference. Check original token positions and every action-readout position after compaction. Token masking without actual shorter computation does not count as acceleration.

The published comparator should come from an authenticated OFT adaptation where possible. A port of original VLA-Cache must be labeled a port and independently checked. LightVLA's released model is useful as a separate reference if authorized and compatible, but its separately trained weights do not provide a same-checkpoint ablation. [^3][^4][^8]

Exit: at least one current-frame compression setting with measured timing headroom and a noncatastrophic accuracy deficit suitable for a small recovery study. As a resource policy, target at least 10% complete-query saving before correction and avoid a setting more than approximately 15 percentage points below the qualified dense reference in the pilot. These are triage values, not proof that worse settings cannot be learned.

If none qualifies, stop this candidate rather than adding more gates or selectors indefinitely. If an existing method already resolves the problem, reproduce it and identify a genuine remaining limitation before inventing an extension.

### Stage 2 — Train the smallest decisive prototype

Use one selected compression setting and at most 4,000 training queries initially. Generate actual compressed features and dense labels. Test feature serialization, target alignment, tiny-batch overfitting, nonzero gradients, zero-init identity, and finite memory before the main fit. A shuffled-label control should not improve held-out action agreement.

Train a small action-feature-only corrector and the current-vision corrector using the same labels. Evaluate both in a bounded development rollout population; then perform at most one planned learner-state refinement round. This stage must reach real trained-policy behavior. Do not spend several more named phases on proxy-only evaluations.

Exit: improvement over the same compressed policy without correction, evidence that current visual input adds value, and remaining end-to-end savings. Failure to improve after the bounded refinement ends this method version. A lower offline loss alone does not pass.

### Stage 3 — Confirm the scientific contribution

Freeze the method and test across all four suites, multiple independently trained adapter seeds, and an exposure-audited test population. Include dense, uncorrected compression, action-only correction, fresh-vision correction, and the strongest compatible published comparator. Include a dense-plus-corrector control and at least one alternate token budget/selector as targeted ablations.

Do not label new states within pretrained LIBERO tasks as unseen-task generalization. Claims beyond this checkpoint should require a second independently trained checkpoint or model, with acquisition and compute approved separately. Real-robot generalization is out of scope unless new resources become available.

Exit: a reproducible success–latency improvement attributable to the proposed change. If only one easy suite or seed improves, report it as a limited pilot rather than a broad positive solution.

### Stage 4 — Decide whether the evidence supports a paper

The core figure should show success versus complete-query latency, with uncertainty and identical comparison populations. Additional evidence should isolate the visual bypass, training data amount, distribution correction, and compression budget. Publish the evaluated source, model/adapter provenance, training cost, split/exposure ledger, raw outcomes, and analysis scripts. Retain the prior negative investigation as motivation, not as proof of universal failure.

Do not begin a full manuscript on the assumption that Stage 2 must succeed. A clear method improvement may support a focused applied study without beating every large model, but peer review will still judge novelty, scope, and reproducibility.

## 10. What counts as a positive result?

Define paired success difference \(\Delta S=S_{\mathrm{method}}-S_{\mathrm{dense}}\) and net timing saving \(G=1-T_{\mathrm{method}}/T_{\mathrm{dense}}\). A reasonable prospective target is a positive lower confidence bound for at least a 10% complete-query saving while the lower bound for \(\Delta S\) remains above a predeclared −5-percentage-point noninferiority margin. The margin requires scientific justification; it must not be selected after seeing results.

Also demonstrate a gain over a strong compression baseline. Recovering from 0% to 20% while dense succeeds 90% of the time is not the intended result. Beating only a handicapped sequential baseline is not sufficient. A lower training-memory requirement can contribute to a paper, but needs measured end-to-end cost and meaningful task performance, not a parameter-count argument.

Use paired episode outcomes and report discordant pairs. Account for task clustering and training-seed variation; do not treat thousands of queries as thousands of independent success trials. For orientation only, if 10% of pairs disagree, the simple independent-pair standard error near equal success is about \(\sqrt{0.1/n}\). At 120 pairs that is about 2.9 percentage points before task/seed effects, too imprecise for many narrow claims. Final sample size should come from a prespecified power analysis using development estimates and available untouched states.

Measure latency on a fixed observation corpus with randomized/interleaved method order, warmup, explicit GPU synchronization, and all necessary preprocessing, token selection, correction, transfer, and postprocessing. Report mean, median, and tail latency. Separately report deployment episode cost and success. Failed episodes alter the observed query distribution and length, so their timing average is not a clean same-input speed comparison. Timing runs must exclude teacher queries and heavy audit hooks; correctness runs must still verify the same execution path.

LIBERO's synchronous simulator normally waits for inference. Faster computation there does not by itself demonstrate improved real-time reaction or safety under physical delay. Avoid those claims without a separate delay-aware protocol.

## 11. Failure controls that address our repeated problems

| Risk | Required control |
|---|---|
| Shared helper hides the same bug in both arms | Independent released evaluator; deliberately failing regression test for the old readout error |
| All-token path passes but compact path is wrong | Explicit original-position map, camera order, prompt-length variation, action-state capture, position/mask assertions |
| Loader changes checkpoint metadata | Read-only checkpoint contract, isolated dynamic-code cache, source manifests; no routine in-place patching |
| Smoke test never exercises the actual failing route | Exercise teacher labeling, hard compression, serialization, backward update, reload, and one complete episode before scale |
| Small parameter count mistaken for small training memory | Measure one complete optimizer step with the intended batch, dtype, and feature length |
| Soft-mask training differs from physical pruning | Train from actual hard-compressed forward outputs |
| Feature-target or action-normalization misalignment | Immutable query identity, camera hashes, suite statistics, eight-step action and gripper tests |
| Test leakage through prior experiments | Trajectory-level split and exposure ledger; development reuse disclosed; locked evaluation opened once |
| Apparent learning comes from task memorization | State-only and action-only controls; task-level reporting; second-checkpoint or task-transfer claim only when tested |
| Offline improvement fails in robot control | Early closed-loop prototype and bounded learner-state labeling |
| GPU integration failures repeat | No new compiler/graph/sharding stack in the prototype; project-local pinned runtime; preflight on actual executed operations |
| Timing saving comes from instrumentation differences | Same timing boundary and execution mode; matched optimized dense baseline |
| Outcomes dictate a new gate after every run | Freeze versioned hypotheses, settings, counts, criteria, and analysis before outcomes; preserve all failed versions |
| Jobs disappear or stop silently | Atomic progress and summary artifacts; health-only monitoring during frozen runs; technical stop distinguished from scientific failure |
| Research becomes an endless redesign sequence | One selected setting, one bounded refinement, explicit stopping decision; no automatic new acronym or retry |

No checklist guarantees an error-free experiment. The purpose is to make likely failures cheap to detect, isolate their causes, and prevent a technical recovery from silently changing the scientific question.

## 12. Bottom-line assessment

**Engineering feasibility:** credible for sequential forward-label generation and detached small-module training, subject to measured memory and runtime qualification. Full OFT retraining is not the default feasible route.

**Chance of preserving closed-loop performance:** unresolved. Existing work makes learning-based recovery plausible, but neither our data nor a literature analogy supports a numerical success probability. The zero-success recursive policy is a reason to change the starting problem, not evidence that the new corrector will work.

**Novelty:** possible as a focused, resource-efficient study of downstream correction under current-frame compression, but the broad ingredients are already established. The strongest novelty test is a comparison and ablation showing an effect that the nearest existing approaches do not already provide.

**Recommendation:** authorize baseline qualification first, then a bounded compression-and-learning prototype if qualification succeeds. Retain the negative evidence, stop the failed cache branch, and make the next investment a small actual learning experiment rather than another long sequence of proxy gates. This is a defensible route toward a positive paper, not an assurance of one.

## Internal evidence index

Paths below are relative to the research repository. Historical reports retain their original terminology; this audit's interpretations take precedence only for the present recommendation, not for the frozen records.

- **[E1]** `docs/NEGATIVE_RESULTS_PAPER_ARCHIVE.md` and its companion data: whole-prefix evaluation, timing pilot, and diagnostic findings.
- **[E2]** `reports/PHASE_V3_D_REPORT.md`: matched batched dense/scene-camera comparison.
- **[E3]** `reports/PHASE_V5_D_V10_TECHNICAL_STOP_REPORT.md`: compiler/execution stop, no completed-query or episode result.
- **[E4]** `reports/BRACE_B3_V05_REPORT.md`: completed numerical screen; later semantic qualification caveat applies.
- **[E5]** `reports/PAIR_P3R_REPORT.md`, `reports/pair_p4/`, `reports/PAIR_P4B_REPORT.md`: systems, exploratory and confirmation evidence; later semantic caveat applies.
- **[E6]** `reports/OPENVLA_CUSTOM_PATH_SEMANTIC_AUDIT_2026-08-31.md`; `reports/OPENVLA_SEMANTIC_PARITY_S3_RECOVERY05_PASS.md`; `reports/OPENVLA_D62_S4_RECOVERY01_PASS.md`: independent readout diagnosis and correction.
- **[E7]** `reports/CAC_C1_S5_REQUALIFICATION_PASS.md`: corrected 97-call systems qualification, not learned-policy performance.
- **[E8]** `reports/CAC_C1H_S6_RECOVERY01_GATE_H_STOP.md`; `results/cac-c1h-headroom-s6-v02-recovery01/analysis.json`; matching immutable worker evidence. Analysis SHA-256: `7944802c69eb977c04eeebb7fa7bbe0ee038ed0cc06d292632643bbcd3cc755b`.
- **[E9]** `reports/PAIR_P1_REPORT.md`, `reports/pair_p1/`, `configs/pair/p1_data_v1.json`: source data, normalization, chronology and split audit.
- **[E10]** `src/savr/pair/p3_openvla.py`, `src/savr/cac/c1.py`, `scripts/run_cac_c1h_worker.py`, `configs/cac/c1h_headroom_s6_v02_recovery01.json`: examined active interfaces and worker behavior.
- **[E11]** `PROJECT_STATUS.md`, `docs/DECISIONS.md`, runtime package metadata, local/remote Git status: current state and provenance limitations. Status headers and tracked HEAD alone are not sufficient summaries of later work.
- **[E12]** `reports/PHASE_V4_A_REPORT.md`, `reports/PHASE_V5_C_REPORT.md`: controller specification contradiction and CPU-only executor qualification, respectively.

Authenticated experiment identities from the existing records: checkpoint revision `638918f3d1c2e43a39a8a20772bdb8b91835e4b7`; OpenVLA-OFT revision `e4287e94541f459edc4feabc4e181f537cd569a8`; LIBERO revision `8f1084e3132a39270c3a13ebe37270a43ece2a01`; demonstration revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`. Later compatibility code also requires its own source hashes.

## Primary references and verification notes

Sources checked on 8 September 2026. Official reports and code are distinguished from external preprints. Search-result aggregators were used for discovery, not as the basis for technical claims. An OpenReview verification page prevented direct access to its VLA-Cache PDF during this audit; the authors' primary project page was available. No downloaded checkpoint or third-party reported number was independently benchmarked here.

[^1]: Kim et al., [official OpenVLA-OFT LIBERO evaluation and training documentation](https://github.com/moojink/openvla-oft/blob/main/LIBERO.md). Used for reference success, evaluation conventions, environment differences, and the documented training-memory requirement. Live documentation is not a replacement for the experiment's pinned source.
[^2]: Kim et al., [Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success](https://www.roboticsproceedings.org/rss21/p017.html), Robotics: Science and Systems XXI, 2025. Used to identify the base inference/training method and proper comparison family.
[^3]: Xu et al., [VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching](https://vla-cache.github.io/), authors' project page identifying NeurIPS 2025 publication. Used for adaptive caching prior art and original evaluation scope.
[^4]: Jiang et al., [The Better You Learn, The Smarter You Prune: Towards Efficient Vision-language-action Models via Differentiable Token Pruning](https://arxiv.org/html/2509.12594), September 2025; [official LightVLA code and requirements](https://github.com/LiAutoAD/LightVLA), identifying ICRA 2026 acceptance. Used for direct learned-pruning overlap and resource requirements.
[^5]: Gao et al., [Compressor-VLA: Instruction-Guided Visual Token Compression for Efficient Robotic Manipulation](https://arxiv.org/html/2511.18950), November 2025 preprint. Sections 3 and 4.1 establish the global/local compressor and eight-A100 fine-tuning recipe.
[^6]: [Learning to Accelerate Vision-Language-Action Models through Adaptive Visual Token Caching](https://arxiv.org/html/2602.00686), January 2026 preprint. Sections 3.1–3.3 establish learned selector/ratio modules with a frozen backbone and task-gradient training.
[^7]: Yu et al., [AC²-VLA: Action-Context-Aware Adaptive Computation in Vision-Language-Action Models for Efficient Robotic Manipulation](https://arxiv.org/html/2601.19634), January 2026 preprint. Sections 3–4 establish routing, compaction, position mapping, and self-distillation; experiments use CogACT, not our OFT configuration.
[^8]: [Just Noticeable Difference Modeling for Token Compression in Vision-Language-Action Models](https://arxiv.org/html/2608.21247), August 2026 preprint. Sections IV–V and Table II establish action-tolerance-based compression, separate view estimators, and reported OFT comparisons. Recent and not independently reproduced.
[^9]: Sendai et al., [Leave No Observation Behind: Real-time Correction for VLA Action Chunks](https://arxiv.org/html/2509.23224), September 2025 preprint. Sections 3–4 establish the residual mechanism, latest-observation conditioning, demonstration targets, and SmolVLA experiment.
[^10]: [Latent Bridge: Feature Delta Prediction for Efficient Dual-System Vision-Language-Action Model Inference](https://arxiv.org/html/2605.02739), May 2026 preprint. Used for feature-reconstruction overlap and learner-state refinement, not a transferable OFT performance estimate.
[^11]: Yue et al., [DeeR-VLA: Dynamic Inference of Multimodal Large Language Models for Efficient Robot Execution](https://papers.neurips.cc/paper_files/paper/2024/file/67b0e7c7c2a5780aeefe3b79caac106e-Paper-Conference.pdf), NeurIPS 2024. Used for early-exit prior art.
[^12]: Jeon et al., [Shallow-π: Knowledge Distillation for Flow-based VLAs](https://arxiv.org/abs/2601.20262), January 2026 preprint. Used for depth-distillation prior art; its flow-based setting differs from OFT.
[^13]: Ross, Gordon, and Bagnell, [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html), AISTATS, PMLR 15:627–635, 2011. Used for the learner-induced distribution problem and data aggregation rationale, not a finite-sample success guarantee.
[^14]: Alvar et al., [DivPrune: Diversity-based Visual Token Pruning for Large Multimodal Models](https://openaccess.thecvf.com/content/CVPR2025/papers/Alvar_DivPrune_Diversity-based_Visual_Token_Pruning_for_Large_Multimodal_Models_CVPR_2025_paper.pdf), CVPR 2025, pp. 9392–9401. Used for a concrete nonlearned token-selection comparator, not evidence of VLA task success.
