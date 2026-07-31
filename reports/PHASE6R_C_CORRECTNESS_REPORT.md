# Phase 6R-C SAVR 2.0 Correctness Report

Status: COMPLETE — ALL REQUIRED CORRECTNESS GATES PASSED

Report date: 2026-07-31

## Scope

Phase 6R-C implemented SAVR 2.0 separately from SAVR 1.0 and validated the
frozen signal, controller, cache, logging, and real-model behavior. It did not
run calibration or rollout episodes and did not inspect the final holdout.

## CPU evidence

- `88` tests passed locally and on TITAN, including `9` server subtests;
- all SAVR 1.0 tests remained unchanged and passed;
- local per-camera patch aggregation and vetoes passed;
- grouped state/action and gripper-transition checks passed;
- warm-up, stable-fresh, isolated-reuse, and prefix-budget boundaries passed;
- invalid data, cache incompatibility, reset identity, and immutable decision
  records passed;
- Ruff, mypy, compilation, schema, and diff checks passed.

## First real-model fixture

Run `phase6r-c-correctness-v1` executed ten model queries, one simulator reset,
and zero rollout episodes on GPU `0`, UUID
`GPU-bb2451d6-2989-a112-5c18-8892943710e4`.

Exact FR parity and all published component counts passed. The intended reuse
was correctly vetoed because the real action chunk contained both open and
closed gripper commands. The run stopped as `failed`, its `29,917` bytes of
evidence remain immutable, and no controller rule was weakened.

- elapsed: `26.126` seconds;
- checkpoint restoration: passed;
- unexpected checkpoint files: none;
- summary SHA-256:
  `9f675f602237d0809ea5e1591d816995ce97ce3167ae88e607e9404932812729`;
- manifest SHA-256:
  `15dce134a03fcf42f77ed8c53fc68975fcfe4b0b5e777d21975a054dd81124ff`.

## Predeclared recovery

The recovery used the hashed existing Phase 6 FR trace specified in
`reports/PHASE6R_C_CORRECTNESS_RECOVERY_PLAN.md`. Run
`phase6r-c-trace-recovery-v1` executed eight model queries, zero simulator
resets, and zero rollout episodes on the same responsibly selected GPU.

All recovery gates passed:

- terminal status: `completed`;
- query records: `8/8`, indices `0-7`;
- refresh/reuse decisions: `7/1`;
- reuse triggers: none;
- reuse signals stable: true;
- reuse vision-backbone calls: `0`;
- reuse visual-projector calls: `0`;
- reuse language-model calls: `1`;
- reuse action-head calls: `1`;
- current proprioception on reuse: exact;
- reuse/unmodified actions for the same input: exactly equal;
- controller counters: seven completed controller queries, one reuse, stable
  counter reset to zero;
- artifact bytes: `30,861`;
- elapsed: `15.980` seconds;
- peak allocated GPU memory: `16,089,727,488` bytes (`15,344 MiB`);
- checkpoint restoration: passed;
- unexpected checkpoint files: none.

Recovery provenance:

- SAVR revision: `fa7a7d04c0ec544066a5eba908cc2fec147dbbde`;
- trace SHA-256:
  `ff9f4bfc004b861260e36d61c5eab641356a9c27c25f7ceccf511e04dd687a63`;
- summary SHA-256:
  `9b58b58ef11de5f594066bde4d45c3f56548960431b83c48d25715fdf6e46ef9`;
- manifest SHA-256:
  `64f675228c0b2fcfcb193c965e3f19ec0fd6f552156641dbd14d14b79f504260`.

## Resource and integrity reconciliation

- cumulative Phase 6R-C real-model queries: `18/20`;
- cumulative real-model elapsed time: `42.106` seconds;
- GPU count per run: one;
- rollout episodes: zero;
- recovery simulator resets: zero;
- both run directories were below the `256 MiB` cap;
- checkpoint and protected-source restoration passed;
- GPU memory returned to its pre-run aggregate level;
- final-holdout states and seeds were not executed or inspected;
- nothing outside `/home/ved/SAVR` was modified.

## Conclusion

Phase 6R-C is complete. The first fixture failure remains visible and confirms
the transition veto; the separately predeclared recovery supplies the missing
real reuse evidence. Phase 6R-D may derive frozen candidates and begin its
staged development-only calibration. This correctness result does not imply a
positive task-success or efficiency result.
