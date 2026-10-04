# Cache Action Correction Phase C1 Report

**Date:** 2026-08-31  
**Decision:** **C1 COMPLETE — PASS; STOP BEFORE C1H**  
**Execution:** one preserved zero-call technical stop, then one authorized Recovery 01

## Outcome

CAC passed the tensor, chronology, isolation, storage, memory, and gross
complete-cycle headroom requirements of Protocol V2. This establishes that the
frozen CAC representation can be extracted correctly and that the proposed
adapter fits within the deployed D62 systems budget. It is not evidence that CAC
repairs actions or preserves terminal task success; those questions remain
unopened.

## Recovery history

The first C1 attempt stopped before any VLA call because query preparation
retained vision autograd graphs. Its immutable evidence remains at
`results/cac-c1-tensor-feasibility-v01/`. Recovery 01 forced inference-mode
preparation, bounded tensor lifetimes, corrected complete-cycle timing, hardened
cache/write/checkpoint controls, and preserved every scientific setting and
gate. The user separately authorized the single recovery attempt.

## Exit-gate reconciliation

| Gate | Requirement | Observed | Decision |
|---|---:|---:|---:|
| Planned VLA calls | 95; hard cap 160 | 95 | pass |
| Dense/all-fresh action tolerance | at most `1e-6` | `0.0` | pass |
| `fc2(Z_C)` reproduction | at most `1e-6` | `0.0` | pass |
| Maximum physical-source age | exactly 4 at h4; never above 4 | 4 | pass |
| D62 h4 gross complete-cycle saving | at least 8% | 22.41% median | pass |
| Peak aggregate GPU memory | below 23,552 MiB | 16,785 MiB | pass |
| Nonfinite tensors | zero | zero | pass |
| Temporary evidence | below 4 GiB | 97,972,847 bytes | pass |
| Adapter parameters | exactly 5,070,599 | 5,070,599 | pass |
| Fixed numeric record | exactly 1,380,352 bytes | 1,380,352 bytes | pass |

Every machine gate in `result.json` is true.

## Timing and predicted CAC cost

Timing includes image preprocessing, visual encoding, proprio projection,
decoder execution, D62 mask selection, sidecar extraction, and synchronization.
No timed visual features were precomputed.

| Horizon | Dense median | D62 median | Gross saving |
|---:|---:|---:|---:|
| 2 | 3,903.52 ms | 3,177.73 ms | 18.68% |
| 4 | 6,526.31 ms | 5,063.72 ms | 22.41% |

- CAC feature extraction: 9.03 ms median per cached query.
- Full zero-initialized adapter forward: 2.38 ms per query.
- Adapter parameter storage in bfloat16: 10,141,198 bytes.
- Median action-head capture overhead: 5.28 ms CUDA / 5.31 ms wall.

The h4 feature-plus-adapter estimate is approximately 45.7 ms over four cached
queries, small relative to the observed gross cycle saving. This is a systems
feasibility estimate, not the final C5 latency result.

## Correctness and isolation

The following checks all passed:

- exact sidecar-off/on action and full K/V equality;
- exact dense/all-fresh full K/V equality;
- exact recursive repeat actions and source histories;
- fresh reset provenance, age zero, phase zero, source zero, and zero deltas;
- full raw-byte equality of cloned K/V tensors with disjoint storage;
- unchanged parent cache, tracker, salience, CPU/CUDA/Python/NumPy RNG, and
  absent offline action queue;
- exact camera/tile ordering, source IDs, chronology, eight-action cadence,
  normalization, dtypes, devices, and finite tensors; and
- bitwise zero correction from the zero-initialized full adapter.

## Streaming and artifact integrity

- feature record SHA-256:
  `06590b082e20ec20602de2c4900432e5dd9b96be4bdda1e0cacdcde4861e32dc`;
- sidecar SHA-256:
  `bc7eda7d19b76cf9a3ff42897155813d279abbd1a208dd3fcc4734de4648d261`;
- instruction embedding SHA-256:
  `5a48946a9c150c8729f34a2e81a2491b082d0375cc61637fb26d20c258104d3a`;
- result semantic SHA-256:
  `9c326b39fbc39c6d56451fe4e2b94b46c016c69406196b74e9f1def27605ce5b`.

All hashes were independently reconciled after completion. Runtime caches were
confined under the result root.

## Protection and restoration

- GPU: physical GPU 0 only;
- training: none;
- simulator: unused;
- expert actions: unopened;
- terminal outcomes: unopened;
- persisted action values: none;
- automatic retry: none;
- checkpoint `config.json`, `configuration_prismatic.py`, and
  `modeling_prismatic.py`: restored exactly;
- loader-created backups: removed only after content verification; and
- changes outside `/home/ved/SAVR`: none remaining.

## Phase decision

All C1 exit gates pass. **Stop before C1H.** C1H is the terminal
repair-opportunity screen and requires separate explicit user approval. A C1
pass makes CAC technically feasible; it does not yet establish a positive-paper
result or authorize training.
