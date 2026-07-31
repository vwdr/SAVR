# Phase 6R SAVR Redesign Roadmap

Status: PHASE 6R-B THROUGH 6R-E AUTHORIZED

Roadmap date: 2026-07-31

## Purpose

Phase 6R pursues a scientifically defensible SAVR redesign after the original
Phase 6 calibration stop. The objective is a meaningful positive-results
paper, but no positive conclusion is assumed or guaranteed.

The original Phase 6 protocol, artifacts, and negative findings remain
immutable and visible. Initial-state IDs `10-49`, seeds `7/17/27`, and all
other final-holdout outcomes remain untouched.

## Approval rule

The user granted blanket authorization for all remaining Phase 6 work on
2026-07-31. Phase boundaries remain mandatory and are announced, recorded,
validated, and synchronized, but no further approval pause is required before
Phase 7. Work still stops for a safety boundary, material scope change,
unexpected resource requirement, or scientific decision outside this roadmap.

At each checkpoint:

1. reconcile evidence and immutable artifacts;
2. update status, milestone, decision, and claim records;
3. run the declared validation suite;
4. synchronize GitHub, TITAN, and the local review copy;
5. report safety/resource use and unresolved limitations;
6. request authorization for the next phase.

## Phase 6R-A — Forensic diagnosis

Use only existing Phase 6 calibration artifacts. Analyze online/offline
skip-rate transfer, task-level outcomes, reuse timing, action divergence,
camera aggregation, threshold proximity, reuse streaks, and evidence gaps.

Exit gate:

- reproducible diagnosis artifact and report;
- ranked observed mechanisms and plausible mechanisms clearly separated;
- concrete requirements for SAVR 2.0;
- no method change, GPU rollout, or final-holdout access.

## Phase 6R-B — Redesign and protocol

Use the Phase 6R-A diagnosis and primary-source research to define SAVR 2.0.
Freeze the controller semantics, development population, candidate funnel,
success and efficiency requirements, online stop rules, instrumentation,
resource bounds, and claim boundary before outcome collection.

Exit gate:

- redesign is training-free and preserves the validated cache boundary;
- every new rule addresses a documented failure mechanism;
- conservative candidate and stopping rules are predeclared;
- user reviews the frozen redesign protocol before implementation.

## Phase 6R-C — Implementation and correctness

Implement SAVR 2.0 without deleting or overwriting SAVR 1.0. Expand unit,
synthetic replay, recovery, logging, and schema tests. Run only the bounded
real-model checks declared by the Phase 6R-B protocol.

Exit gate:

- all CPU and bounded real-model checks pass;
- wrapped FR parity remains intact;
- every reuse safety rule is directly tested;
- no calibration or holdout rollout has begun.

## Phase 6R-D — Conservative staged calibration

Use only declared development/calibration data. Begin with low reuse targets
and a predeclared online funnel. Promote candidates mechanically; retain every
failure. Do not infer online safety from FR replay alone.

Exit gate:

- a candidate satisfies the frozen success constraint and practical efficiency
  floor, or the redesign stops;
- no final-holdout outcome is inspected;
- no threshold, margin, or promotion rule is relaxed after outcomes.

## Phase 6R-E — Baselines, selection, and power

Match PR and VOR to the eligible SAVR budget, run required development
ablations, select one SAVR configuration by a frozen rule, and calculate the
confirmatory sample size and resources.

Exit gate:

- one defensible primary configuration exists for each method;
- SAVR has a non-negligible success-efficiency rationale;
- power and resource requirements are feasible;
- otherwise stop before Phase 7.

## Phases 7-11

- Phase 7: freeze the confirmatory protocol.
- Phase 8: run the untouched multi-suite final evaluation.
- Phase 9: run predeclared ablations and sensitivity analyses.
- Phase 10: reproduce every analysis and audit every paper claim.
- Phase 11: complete the evidence-based manuscript.

## On-task controls

- exactly one phase may be active;
- one branch and one checkpoint report per phase;
- immutable run directories and input/output hashes;
- explicit protocol version and code revision in every run;
- tests before GPU work and small screens before large matrices;
- predeclared resource caps and stop rules;
- final holdout protected from tuning;
- calibration, exploratory, and confirmatory evidence labeled separately;
- no unsupported claim added to the manuscript;
- all server writes confined to `/home/ved/SAVR`.
