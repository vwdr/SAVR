# IR-SA-ACR Gated Evaluation Roadmap

Status: **ROADMAP APPROVED; EXPERIMENT-SPECIFIC FREEZES STILL REQUIRED**

Approval date: 2026-08-10
Applies after: V5-A software-correctness checkpoint

## 1. Objective

Determine honestly whether IR-SA-ACR can preserve task success while reducing
meaningful visual work and measured inference time. The roadmap prioritizes a
positive-results route but does not guarantee or manufacture a positive
result. Every stopped or negative phase remains part of the record.

The user's approval authorizes the logical next planning and CPU-only steps in
this roadmap. It does not waive scientific freezes, protected-data boundaries,
download controls, or shared-server safety. Before selecting or using a GPU,
the project must still pause for user coordination rather than inspect shared
allocations.

## 2. Global principles

1. **Success first.** Task success is the primary constraint; efficiency is
   secondary.
2. **Freeze before output.** Candidate families, gates, estimands, exclusions,
   and stop rules are committed before relevant outputs are read.
3. **Development and confirmation are separate.** Object development can
   inform selection; protected Goal/final data cannot.
4. **Paired comparisons.** Use the same task, seed, initial state, and order for
   Full Refresh and IR-SA-ACR whenever supported.
5. **Fail closed.** Missing/invalid signals, state disagreement, technical
   ambiguity, or incomplete evidence cannot promote a candidate.
6. **One variable family at a time.** Controller selection and executor
   optimization are measured separately before combination.
7. **Preserve negatives.** V3/V4 evidence and all future stopped runs remain
   immutable.
8. **No post-hoc relabeling.** A change made after output creates a new method
   version; it is not part of the original freeze.
9. **Full provenance.** Record code/config hashes, environment, hardware,
   selected GPU, seeds, commands, artifacts, timestamps, and failures.
10. **Bounded resources.** Each phase defines maximum queries, episodes,
    storage, and GPU count before execution.

## 3. Status labels

- `NOT_STARTED`: no phase-specific freeze or execution.
- `FREEZE_IN_PROGRESS`: protocol is being written; no outputs may be read.
- `FROZEN`: protocol/config committed before execution.
- `RUNNING_OUTCOME_BLIND`: only health and terminal-record counts visible.
- `COMPLETE_ELIGIBLE`: all required gates pass.
- `STOPPED_NEGATIVE`: technically valid evidence failed a scientific gate.
- `STOPPED_TECHNICAL`: integrity/completion failed; no scientific conclusion.
- `NOT_RUN_INELIGIBLE`: upstream gate prevented the phase.

## 4. Phase V5-B — Output-blind development screening

Status: `NOT_STARTED`; protocol preparation authorized.

### Purpose

Determine whether any small, predeclared threshold/cap candidate produces a
plausible reuse region under the corrected state machine before GPU work.

### Before execution

- Freeze the exact development-only input records and prove they exclude Goal
  and final populations.
- Freeze the candidate-generation rule and compact candidate count.
- Freeze all thresholds, prefix caps, warm-up, signal preprocessing, tie
  handling, candidate ordering, and promotion gates.
- Hash every input, config, script, and expected schema.
- Ensure the analysis script cannot expose episode success fields.

Prior V4 outputs may justify the method family but cannot silently select the
best V5 threshold. If any prior evidence is used quantitatively, declare it in
the freeze and reserve a new independent confirmation population.

### Permitted measurements

- decision sequence and reason counts;
- maximum reuse streak;
- prefix cap compliance;
- reuse positions and cache-age agreement;
- theoretical scene-encoding work under the predeclared accounting formula;
  and
- deterministic replay equality.

Success labels and protected outcomes remain sealed.

### Required gates

- trace grammar and every software invariant pass;
- maximum streak equals one;
- no cross-episode cache/reference state;
- candidate meets predeclared minimum reuse and theoretical visual-work margin;
- candidate is not dominated under the frozen selection rule; and
- two independent executions produce identical semantic output.

If no candidate passes, stop negative. Do not relax gates in place.

### Deliverables

Frozen protocol/config, input manifest, deterministic runtime JSON, analysis
report, decision entry, and exact selected candidate or explicit no-selection.

## 5. Phase V5-C — Optimized executor correctness on CPU

Status: `NOT_RUN_INELIGIBLE` until V5-B selects a candidate.

### Purpose

Ensure that reduced logical scene work becomes a genuinely distinct execution
path without altering outputs or controller semantics.

### Required implementation separation

- keep Full Refresh as the oracle;
- retain a non-optimized IR-SA-ACR reference path;
- implement optimization behind a separate versioned executor identity;
- do not change controller thresholds while optimizing execution; and
- instrument scene, wrist, downstream, synchronization, transfer, and total
  time separately.

### Correctness gates

- wrapped Full Refresh matches the base model within frozen tolerances;
- optimized refresh matches reference refresh;
- optimized reuse matches reference reuse for identical cached inputs;
- cache tensor identity, shape, dtype, device, context, and lifetime are valid;
- controller decisions and reason codes are identical across executors;
- failures cannot corrupt or advance the cache/controller state;
- deterministic/adversarial tests and complete repository tests pass; and
- memory growth stays within a predeclared bound.

CPU results cannot establish GPU speed. Passing only makes a bounded GPU test
eligible.

## 6. Phase V5-D — Bounded one-GPU correctness and timing margin

Status: `V02_RECOVERY_VERIFIED_NOT_RUN`. V5-C passed; v01 stopped during
upstream LIBERO import before model load because its run-local config file was
absent. V02 now passes canonical-config, pre-model-stop, complete regression,
and TITAN closed-stdin import gates. GPU execution remains unauthorized.

### Mandatory pause

Before choosing a GPU, stop for user coordination. Do not inspect unrelated
jobs or allocations. Use at most one explicitly selected GPU, record its ID,
and do not interfere with other university work.

### Purpose

Test real-model parity, cache correctness, and whether the optimized path has a
credible timing margin before simulator episodes.

### Freeze requirements

- exact model checkpoint and hashes;
- GPU ID, query budget, tensor inputs, warm-up count, repetitions, order, and
  timing/synchronization method;
- maximum artifact/storage estimate;
- numerical parity tolerances;
- memory, visual-work, sequential-time, and wall-time gates; and
- immutable fail/recovery procedure.

### Gates

- real-tensor refresh/reuse output parity passes;
- trace and cache invariants pass;
- no OOM, NaN, unexpected fallback, or cross-query contamination;
- measured visual work clears the predeclared margin;
- sequential and wall-time ratios clear their predeclared margins with
  uncertainty intervals; and
- repeated ordering checks do not indicate material measurement bias.

Failure stops before simulator work. A speed result without correctness is
ineligible.

## 7. Phase V5-E — Paired Object development

Status: `NOT_RUN_INELIGIBLE` until V5-D passes.

### Purpose

Estimate paired task-success and efficiency on the development suite and decide
whether exactly one frozen V5 method is eligible for independent confirmation.

### Design

- predeclare task/seed/episode set, policy order, episode budget, timeout,
  failure handling, and immutable restart rules;
- pair Full Refresh and IR-SA-ACR on identical initial conditions;
- hide aggregate success and efficiency outcomes until the complete terminal
  record count and run summary exist;
- monitor only own runner health, artifact size, record count, elapsed time,
  and aggregate telemetry for the selected GPU;
- use one primary candidate; diagnostic variants require separately frozen
  budgets and cannot replace it after outputs; and
- preserve every terminal episode record, including technical failures.

### Primary estimands

- paired success difference versus Full Refresh and confidence interval;
- scene reuse fraction and distribution;
- measured visual-work reduction;
- sequential-time ratio and confidence interval;
- wall-time ratio and confidence interval; and
- technical-failure rate.

### Promotion gates

The candidate must simultaneously satisfy the predeclared success
non-inferiority gate, visual-work margin, sequential-time margin, wall-time
margin, technical-validity gate, and all controller/cache invariants. Exact
numeric margins must be justified and frozen before results.

No composite score can compensate for a failed primary gate.

## 8. Phase V5-F — Independent Goal confirmation

Status: `NOT_RUN_INELIGIBLE` until V5-E passes every promotion gate.

### Purpose

Test the single frozen method on an unopened population not used for method or
threshold selection.

### Rules

- no method, threshold, executor, metric, or gate change after unsealing;
- predeclare sample size/power rationale and analysis;
- use paired Full Refresh comparison and the same provenance standard;
- remain outcome-blind until all terminal records and the completed summary
  exist;
- reconcile every frozen gate mechanically; and
- treat a failed confirmatory gate as a negative result, not a tuning prompt.

Passing V5-F supports a scoped empirical paper claim. It does not establish
real-robot or cross-model generalization.

## 9. Phase V5-G — Ablations, sensitivity, and failure analysis

Status: `NOT_RUN_INELIGIBLE` until a primary method is frozen; confirmatory
claim priority remains with V5-F.

Required ablations:

- visual-only gate;
- + normalized end-effector translation;
- + action-history-derived gripper veto;
- direction-reversal veto as a separately labeled diagnostic, if frozen;
- latch removed only in a bounded diagnostic that cannot be promoted;
- hard-cap sensitivity;
- executor reference versus optimized path; and
- eligible neighboring thresholds frozen before outcome access.

Report success and efficiency jointly. Analyze failures by task, seed, episode
length, reason code, reuse position, gripper proximity, and time since refresh.
Avoid causal language unless the design supports it.

## 10. Phase V5-H — Claim audit and manuscript update

Status: `NOT_RUN_INELIGIBLE` until all intended evidence is reconciled.

Before editing the manuscript:

- map every numerical claim to a result cell, config, code revision, and
  analysis version;
- distinguish development, confirmation, ablation, and diagnostic evidence;
- ensure stopped/negative results are not silently omitted;
- verify citations against primary sources;
- update title, abstract, introduction, method, results, discussion,
  limitations, and conclusion consistently; and
- have an independent reproducibility pass confirm tables and figures.

## 11. On-task controls

The following controls prevent drift or hallucination:

- one active phase and one authoritative protocol at a time;
- phase checklist with explicit entry and exit gate;
- machine-readable configuration plus semantic hashes;
- automated schema and invariant tests;
- separate output production and result reconciliation scripts;
- outcome blindness until completeness;
- immutable evidence directories and append-only decisions;
- status updates after every freeze, execution, failure, and merge;
- GitHub review/CI before merge;
- synchronize merged main to local Documents/SAVR and `/home/ved/SAVR`;
- compare exact Git commit across all three locations; and
- stop on contradiction between prose, configuration, code, or evidence.

## 12. Approval and immediate next checkpoint

The user approved the logical next steps on 2026-08-10. The immediate action
after this documentation checkpoint is to draft and freeze the Phase V5-B
output-blind screening protocol. No replay output may be read until that freeze
is committed and verified. Later GPU selection still requires explicit server
coordination under `AGENTS.md`.
