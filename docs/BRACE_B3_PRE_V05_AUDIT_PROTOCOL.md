# BRACE-B3 Pre-v05 Exhaustive Technical Audit

Date: 2026-08-25  
Status: COMPLETE; ACCEPTED

## Purpose

V01--V04 ended at four different technical integration boundaries before a
complete B3 result existed. Before proposing any v05 attempt, audit every B3
path that can be exercised without loading the model or using a GPU. This is a
technical-readiness audit, not a BRACE experiment and not evidence for or
against the method.

## Immutable boundaries

- Operate on TITAN only through `ssh titan` and only inside `/home/ved/SAVR`.
- Hide CUDA for the complete audit; initialize no CUDA context.
- Load no model or checkpoint weights and issue zero model queries.
- Use no simulator and inspect no protected success or outcome field.
- Preserve all prior run evidence and the checkpoint.
- Do not authorize or create a B3-v05 run identity.
- B4 remains unauthorized.
- Write the server audit once to
  `reports/runtime/brace_b3_pre_v05_audit.json`.

## Required checks

1. Run the entire BRACE test suite in the pinned OpenVLA-OFT environment with
   CUDA hidden and bytecode writes disabled.
2. Exercise patch-change scoring on real PyTorch tensors at zero and nonzero
   dynamic range, including an ineligible-candidate failure.
3. Exercise the SDPA sidecar through the real PyTorch primitive for the exact
   32-call/three-layer contract, verify unchanged outputs and finite salience,
   and prove call-count drift fails closed while restoring the primitive.
4. Exercise all four frozen profiles on real tensors, including P2 wrist-source
   rotation, every horizon, all 72 randomized timed cycles, deterministic
   source provenance, nested layer budgets, and age bounds.
5. Reconcile the complete 388-query allocation and 420-query hard cap.
6. Parse every `torch.maximum` and `torch.minimum` call in the B3 helper and
   reject invalid binary arity.
7. Reconcile worker-level inference mode, disabled-gradient assertion,
   corrected previous-image source, timed action parity, immutable artifact
   writes, and checkpoint restoration structure.
8. Re-read the pinned private OpenVLA/cache helper signatures, SDPA backend,
   pruning locations, position-preserving update path, official no-gradient
   path, checkpoint normalization alias, and all pinned repository revisions.
9. Require the main repository to be clean except for preserved `tmp/`, and
   require every pinned third-party repository to be clean.
10. Authenticate the exact audited sources and the final audit record.

## Acceptance gate

The audit is accepted only if every required check passes in one CUDA-hidden
server execution, CUDA remains uninitialized, and the immutable record states
zero model loads, model queries, simulator outcomes, and protected-outcome
access. Any failure is documented and corrected before a new audit record; it
does not authorize a GPU retry.

Even an accepted audit cannot guarantee that the full model path will succeed.
It only removes all currently testable integration failures. Residual risks
include real checkpoint tensor shapes, live cache mutation semantics, GPU
memory, and timing behavior. A separately reviewed and explicitly authorized
v05 would still be required to test those risks.

## Completion record

The immutable audit passed 43 BRACE tests on TITAN with CUDA hidden and
uninitialized. All structural, backend, private-interface, repository, and
authentication gates passed with zero model loads, model queries, simulator
outcomes, or protected-outcome access. Evidence is recorded in
`reports/BRACE_B3_PRE_V05_AUDIT_REPORT.md` and
`reports/runtime/brace_b3_pre_v05_audit.json`. V05 and B4 remain unauthorized.
