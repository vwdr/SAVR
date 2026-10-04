# PAIR-VLA Phase P2 Report

Date: 2026-08-28  
Phase: P2 — CPU/synthetic implementation correctness  
Protected-population access: none  
GPU use: none  
Decision: **PASS P2; STOP before P3**

## Result

PAIR-VLA now has a fail-closed, synthetic-tested implementation of the frozen
query alignment, intervention, provenance, risk-routing, calibration, and
recording contracts. The implementation extends BRACE through separate
`savr.pair` modules and does not alter accepted BRACE code or evidence.

The authoritative TITAN gate passed 69 focused tests: 25 PAIR tests and 44
BRACE regressions. `CUDA_VISIBLE_DEVICES` remained empty; the CPU training test
also asserted that PyTorch CUDA was uninitialized before and after optimization.
No model, checkpoint, demonstration, simulator, task outcome, or network source
was accessed.

## Implemented contracts

- Original-action query indexing at exact eight-action deployment spacing,
  complete-chunk exclusion, and trajectory-level split inheritance.
- Frozen 4x4 tiles, primary/wrist atomic groups, onset layers, exact physical
  profile budgets, and suffix-safe nested masks.
- Exact-clone recursive source ledgers with per-layer/camera/tile ownership,
  mixed source ages, bounded source retention, live-source eviction refusal,
  dense reset, and abort reset.
- All-fresh identity and actual-source stale tensor interventions on synthetic
  `[camera, patch, feature]` tensors.
- Exact valid-mask normalized expert L1, signed and positive regret, dense-action
  distortion, maximum distortion, and gripper-coordinate diagnostics.
- Deployment-available group/global features with recursive rejection of expert,
  future, outcome, and hidden-current-dense fields.
- A deterministic NumPy 64/32 group plus 64/32 DeepSets reference router with
  learned categorical embeddings, fewer than 250,000 parameters, canonical
  checkpoint serialization, and semantic-hash verification.
- A CPU PyTorch training backend with the frozen Huber/pinball loss weights,
  AdamW settings, gradient clipping path, deterministic seeds, and no VLA
  parameters or gradients.
- Conservative empirical q90 correction, support-envelope detection,
  maximum-saving selection, and mandatory dense fallback outside support.
- Outcome-protected technical records, strict JSON Schema validation,
  semantic hashes, and contract lifecycle/fallback behavior.

## Adversarial coverage

The tests explicitly cover scene/wrist isolation; tile/layer off-by-one errors;
missing source records; mixed ages; live-source ring eviction; nonnested and
duplicate masks; expert/future/current-dense leakage; terminal padding; empty
valid masks; nonfinite actions/features/tensors; unsupported profiles,
horizons, and categories; contract expiry, abort, and episode reset;
deterministic seeds and checkpoint reload; malformed and nested-outcome records;
all-fresh dense identity; and actual-stale source resolution.

## Verification

| Gate | Evidence | Decision |
|---|---|---|
| New PAIR correctness tests | 25/25 passed on TITAN | PASS |
| Existing relevant BRACE tests | 44/44 passed on TITAN | PASS |
| Local focused gate | 56 passed, 9 optional-dependency skips | PASS |
| Local full repository regression | 447 passed, 9 optional-dependency skips | PASS |
| CUDA hidden/uninitialized | Empty visibility; CPU torch assertions passed | PASS |
| Protected populations untouched | Config and record report zero access | PASS |
| Malformed/unsupported states fail closed | Adversarial test matrix passed | PASS |
| Serialized inference deterministic | Same-seed bytes and reloaded outputs match | PASS |
| Schema/hash reproducibility | Local and TITAN manifest identity both `9cd16e29...e7229d` | PASS |
| Artifact cap | 86,516 bytes versus 1,073,741,824-byte cap | PASS |

An additional non-gating full-repository run on TITAN produced 457 passes and
one unrelated legacy ACR preflight failure. That old test requires V10 result
paths not to exist; TITAN correctly contains the already-completed V10 evidence.
The evidence was not changed or removed. The equivalent full local repository
run is green, and all PAIR/BRACE tests on TITAN are green.

## Evidence

- `configs/pair/p2_cpu_v1.json`
- `scripts/verify_pair_p2.py`
- `schemas/pair/technical_record.schema.json`
- `reports/pair_p2/artifact_manifest.json`
- `reports/pair_p2/technical_record.json`
- `reports/pair_p2/run_summary.json`
- `reports/PAIR_P2_SEMANTIC_MANIFEST.json`

The artifact manifest covers 20 implementation, test, schema, configuration,
and verifier files totaling 86,516 bytes. Its semantic SHA-256 is
`9cd16e29f0df1cd173187fdbda36ca0043ff279de2d975d5dcef22aae2e7229d`.

## Boundary

P2 did not use a GPU, run the VLA, read checkpoint contents, read demonstration
data, open simulator outcomes, use the network, modify the manuscript, alter
accepted BRACE evidence, commit, push, or begin P3. P3 is an outcome-blind
physical-headroom phase and requires separate explicit user approval.
