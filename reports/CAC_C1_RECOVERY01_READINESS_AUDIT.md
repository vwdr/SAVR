# CAC C1 Recovery 01 Pre-Execution Readiness Audit

**Date:** 2026-08-31  
**Scope:** technical and methodological readiness only  
**Decision:** ready except for explicit Recovery 01 authorization; no GPU run performed

## Audit conclusion

Recovery 01 is now designed to test C1 without repeating the first attempt's
memory error and without biasing the gross timing gate. The recovery preserves
the D62 method, observations, 95-call schedule, scientific gates, protected
populations, and one-GPU boundary. Eighteen CPU-only tests pass in the pinned
TITAN runtime. The read-only TITAN preflight passes every check except the
intentionally absent user-authorization field.

This audit does not claim that CAC will pass C1. It establishes that a pass or
failure should be attributable to the frozen C1 contracts rather than a known
implementation defect.

## Problems found and corrections

| Risk | Failure or bias | Correction and verification |
|---|---|---|
| Gradient retention | The first attempt retained vision autograd graphs and exhausted GPU memory before call 1 | Query preparation is forced into inference mode; the shared helper fails closed when gradients are enabled; dedicated test passes |
| Excess live tensors | Control outputs and multiple recursive caches could remain referenced and inflate later peaks | Large results/caches are explicitly released after each control; recursive-cycle return values contain hashes/booleans only; allocator cache is cleared between sections |
| Incomplete timing | Precomputed visual features would measure only the decoder and overstate complete-cycle cache savings | Every timed dense and D62 query now includes image preprocessing, visual encoding, proprio projection, decoder execution, mask selection, sidecar work, and synchronization |
| Transient memory peaks | Point telemetry could miss a short-lived peak | A background aggregate-GPU sampler is combined conservatively with PyTorch's maximum reserved memory and initial aggregate usage; the strict 23,552-MiB gate applies to the maximum |
| Shared-GPU contention | Another workload could make the selected GPU unsafe | Launch requires aggregate memory at most 1,024 MiB; preflight requires utilization at most 10%; runtime aggregate monitoring stops fail-closed at the strict limit; no process identities are inspected |
| Cache-branch contamination | Dense shadow, cache branch, tracker, salience, RNG, or action state could cross-mutate | Parent and branch caches receive full raw-byte digests; every K/V tensor clone must have disjoint storage; tracker, salience, CPU/CUDA/Python/NumPy RNG, and absent action queue are checked |
| Reset/all-fresh ambiguity | A nominal reset could still carry cache provenance, or all-fresh could differ silently | Full K/V and action equality are checked; reset provenance must be fresh, age zero, phase zero, source zero, and have zero source deltas; two complete recursive runs must reproduce exactly |
| Tensor/API drift | Action head, tokenizer, camera ordering, or feature layouts could change | Exact tokenizer offsets, `fc2(Z_C)` reproduction at `1e-6`, shapes, dtypes, devices, finite values, camera/tile order, source IDs, age cap, and eight-action cadence fail closed |
| Partial or blocked writes | The asynchronous writer could deadlock or silently leave malformed records | Writer startup is synchronized, descriptors are closed on partial failure, errors propagate without queue deadlock, records are fixed-size and exclusive, hashes stream in bounded memory, and injected-failure tests pass |
| Checkpoint-loader mutation | The model loader temporarily stages metadata in the checkpoint | All three metadata hashes and absence of stale backups are checked before load; exact baseline restoration occurs before a completed summary and again on error through `finally`; restored inventory/hashes are recorded |
| Writes outside the project | Matplotlib, Hugging Face, Torch, or temporary caches could write to the user home | Runtime cache, module cache, temporary directory, bytecode, and logging paths are redirected under the immutable result root; hub/network access is forced offline |
| Data or code drift | A changed model helper, observation source, or failed-attempt record could invalidate the run | Recovery config authenticates all code, tests, configs, source-stop evidence, loader backups, checkpoint metadata, and selected HDF5 files before execution |
| Budget drift or runaway execution | Call arithmetic, disk use, or a hung call could exceed scope | Schedule arithmetic is recomputed as exactly 95 calls under the 160-call cap; at least 8 GiB free space is required; artifacts remain capped at 4 GiB; SIGTERM becomes a Python timeout exception so cleanup runs; launch will also use a 3,600-second external timeout |
| Outcome leakage | Technical validation could accidentally use expert actions or terminal success | Only observation groups, proprioception, model outputs, hashes, and timing are accessed; no simulator, expert-action dataset, success field, training, or protected outcome is opened or persisted |
| Unauthorized retry/advance | A recovery or C1H could begin implicitly | Pending authorization is machine-enforced before the result directory is created; automatic retry is false; output root is versioned; C1H remains unauthorized |

## Tests and read-only preflight

- TITAN CPU-only tests: **18/18 passed**.
- Configuration semantic hash: passed.
- Authenticated file hashes: passed.
- Failed attempt preserved and recovery root absent: passed.
- Checkpoint metadata and no stale loader backups: passed.
- Selected observation-source hashes: passed.
- Exact call accounting: passed.
- Pinned Python runtime: passed.
- Storage margin: passed.
- GPU 0 at preflight: 6 MiB aggregate, 0% utilization.
- Authorization lock: deliberately false; worker refuses to create the recovery root.

## Residual risks that cannot be removed before C1

1. A power loss, kernel failure, or uncatchable `SIGKILL` can bypass Python
   cleanup. The next preflight will detect any checkpoint drift or stale backup
   and refuse further work; it will not auto-repair or retry.
2. Another university user may allocate the selected GPU after preflight. The
   aggregate sampler and strict memory gate minimize interference, but cannot
   reserve an unmanaged shared GPU. Any OOM remains a preserved technical stop.
3. The pinned compatibility runtime emits an upstream Transformers-version
   warning already present in the authenticated PAIR runs. C1 intentionally
   tests the same deployed stack; changing it here would invalidate continuity.
4. C1 can still fail scientifically or architecturally: insufficient D62 gross
   headroom, excessive memory, nonexact cache behavior, or tensor-contract drift
   are legitimate C1 failures rather than recovery defects.

## Execution boundary

The approved execution, if separately authorized, is one versioned Recovery 01
attempt on one idle GPU, launched with the frozen configuration and a one-hour
timeout. It stops after C1 analysis and before C1H. No automatic recovery is
allowed.
