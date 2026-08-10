# IR-SA-ACR Implementation and Provenance Ledger

Status: **AUDITED DOCUMENTATION OF THE V5-A SOFTWARE CHECKPOINT**

Ledger date: 2026-08-10
Repository: private `vwdr/SAVR`
Research/design merge: `6007d6f` (PR #64)
Implementation merge: `cf8b731` (PR #65)

## 1. Purpose

This ledger records what changed, why it changed, how it was checked, what did
not change, and which conclusions the evidence permits. It is the provenance
bridge between the implementation and a future formal report.

## 2. Development chronology

| Checkpoint | Action | Scientific purpose | Result |
|---|---|---|---|
| V3-D | Paired Object development with legacy SA-BDP-ACR | Test the initial asymmetric refresh route | Completed negative: success preserved, but frozen visual-reduction and wall-time gates failed |
| V4-A | Output-blind six-candidate diagnosis | Seek a predeclared redesign without opening protected data | Stopped negative: every candidate violated at least one frozen gate; no implementation selected |
| V5-R | Primary-source and code audit | Correct the discovered mismatch between “no consecutive reuse” prose and horizon-2 behavior | Separate latch-based method frozen before code |
| V5-A | Implement and CPU-verify IR-SA-ACR | Establish the corrected temporal semantics without producing new benchmark outcomes | Software checkpoint passed; empirical performance remains unknown |

No V3/V4 result was deleted, relabeled, or reinterpreted to make V5 appear
positive.

The formal specification clarifies one notation detail from the frozen design:
`R* (U R)*` was a shorthand for forced refresh after reuse, not the exact
finite-prefix language. The exact accepted language is
`R*(UR+)*(epsilon|U)`, which permits extra refreshes and termination after one
reuse while still forbidding `UU`. This is a documentation correction only;
the frozen evidence and executable controller are unchanged.

### Documentation checkpoint added after V5-A

The present checkpoint adds four report-facing artifacts: the exact formal
method specification, this implementation/provenance ledger, the manuscript
translation and claim guide, and the gated evaluation roadmap. It also adds
`tests/acr/test_v5_documentation.py` and updates project status, milestones,
and decisions. It changes no controller, adapter, frozen configuration, runtime
evidence, prior report, protected data, or manuscript source.

## 3. Exact V5-R design-freeze changes

Merge `6007d6f` added the research audit, frozen protocol, machine-readable
configuration, freeze tests, and status/decision records before the controller
was implemented.

| Artifact | Role | SHA-256 at V5-A checkpoint |
|---|---|---|
| `docs/ACR_V5_RESEARCH_AUDIT.md` | Primary-source synthesis, exact bug diagnosis, rejected alternatives, claim boundary | `7777a92a00048aee3432dc6dd5c3a44bd4d995724d8f06c0f2be01dcfa089027` |
| `docs/ACR_V5_ISOLATED_REUSE_PROTOCOL.md` | Normative controller contract, acceptance matrix, exclusions, resource cap | `b9145afc07a25076c857e7d2b90185ec657a5d7bcbbd9b4175f8c459e01736e4` |
| `configs/acr/v5_isolated_reuse_freeze.json` | Machine-readable identity, horizon, invariants, exclusions | `06f662f97150c47d5477a8a988ac15b98a36dcbd04650cb984c8097c4e12373a` |
| `tests/acr/test_v5_isolated_reuse_freeze.py` | Prevent accidental drift between freeze and implementation | `f08dcdd1c45ddb45625757f96e2e5f15d69f6767f4accde34022b112f3b94118` |

## 4. Exact V5-A implementation changes

Merge `cf8b731` contained 797 added and 18 removed lines across twelve files.
The removals were status-text replacements, not deletion of experimental
evidence.

| Artifact | Exact change and principle | SHA-256 at merge |
|---|---|---|
| `src/savr/acr/isolated_controller.py` | New, separately versioned controller; requires horizon 1; owns post-reuse latch; checks external cache age; rejects forged consecutive reuse; exposes/reset state | `fdfdda7d5939b0ddb74a15d77b0931aad901521dfb453934283157a0341d1c56` |
| `src/savr/acr/types.py` | Added only `post-reuse-refresh` and `isolation-state-mismatch` reason codes | `7bdfa98a3ec7cbb4e5dda9e91d68abf280cab53ae21f6905757ca40812c72143` |
| `src/savr/acr/__init__.py` | Exported the new controller and snapshot types | `beddbedc8aa36646bee3d1f96295f82a5ddb951261778e9fa4fe413e408b2fab` |
| `tests/acr/test_isolated_controller.py` | New adversarial/state-machine suite covering configuration, trace grammar, mismatch, failure, reset, context, gripper, hard cap, random traces, and legacy separation | `59eedaf9dfd6ab3116e285eb80af1b233ae9b888f9e473800c9665519136b31f` |
| `tests/acr/test_batched_dual_path.py` | Added adapter-level proof that the controller follows actual refresh/reuse cache observations | `87da40c87f3bf76d14e2a9c837409fc8fc3ce58b57c29d6e4449b5059ed495ec` |
| `scripts/verify_acr_v5_isolation.py` | Dependency-free deterministic verifier suitable for the pinned server Python | `9791bdff4b23f9723ffeb2d574286574dca6db3a7309f12fb27756cfb46c3387` |
| `tests/acr/test_v5_cpu_verifier.py` | Checks verifier schema, determinism, trace properties, and semantic digest | `4d86cf6ba95a1c6d214415a3dd1971f1dd7bea67a7ead2399586a0aec670bd7a` |
| `reports/runtime/acr_v5_cpu_verification.json` | Immutable machine-readable CPU result | `7f28416d70d6f2b527741546eb350e35540202b1fdc4ecd38594244eba2a66c7` |
| `reports/PHASE_V5_A_CORRECTION_REPORT.md` | Human-readable acceptance and claim-boundary report | `9bbcc0574259ab60a62d65f56cb79f9ea7f78a6a3f2aebd1100ac9fe991063f9` |
| `PROJECT_STATUS.md`, `docs/MILESTONES.md`, `docs/DECISIONS.md` | Recorded V5-A completion and next-phase prohibition; did not change results | Versioned by merge `cf8b731` |

## 5. Intentionally unchanged implementation

| Artifact | Reason it remained unchanged | Checkpoint SHA-256 |
|---|---|---|
| `src/savr/acr/controller.py` | Preserves historical ACR semantics and permits an explicit legacy comparison showing that horizon 2 can reach streak two | `42557c51c2fbe57ccce42d1215cf9c405a719ae0aa6da738b82c1607cf811529` |
| `src/savr/acr/batched_dual_path.py` | Existing adapter already accepts the common decision/observation interface and implements fresh wrist + cached-or-fresh scene paths | `b242072800bed4d2ce5d27226bc09fafd968a77dd2b6e13a583ec38ad38ad1ba` |
| V3/V4 configs, reports, runtime records | These are immutable negative evidence and must remain available for auditing or a negative-results route | Preserved in Git history and repository |
| Manuscript `.tex` source | V5 has no benchmark result and the user did not authorize a manuscript rewrite | Unmodified by V5-R/V5-A |

## 6. Requirement-to-evidence matrix

| Normative requirement | Test/evidence |
|---|---|
| Only controller identity `acr-isolated-controller-v1` | configuration rejection tests; freeze test |
| Only horizon 1 | configuration rejection tests; freeze test |
| No completed `UU` trace | stable trace, randomized trace, forged-reuse tests; machine record |
| Completed reuse forces next refresh | stable trace and adapter integration tests |
| Cache/latch disagreement forces refresh | age-zero-after-reuse and general mismatch tests |
| Failed/unobserved query cannot clear latch | unobserved-refresh test |
| Reset clears episode state | reset test and snapshot assertions |
| Context, warm-up, invalid-signal, gripper, threshold, hard-cap rules retained | targeted controller tests and inherited common input types |
| Legacy semantics remain separate | legacy horizon-2 trace reaches maximum streak two |
| Common adapter compatibility | batched adapter integration test |
| Reproducible semantic output | deterministic verifier test and semantic SHA-256 |
| Repository regressions absent | 253 repository tests passed at V5-A merge; formatting/lint/bootstrap checks passed |

## 7. Machine verification record

The deterministic record contains:

- 128 corrected queries;
- 51 completed reuses;
- maximum corrected reuse streak of one;
- maximum corrected prefix reuse fraction of 0.40;
- a preserved legacy trace with maximum reuse streak two; and
- semantic SHA-256
  `7dcde7e8b96ba7fe79f1eed0cd6a73661e0d0977678f3581062902b445f7de2b`.

The same dependency-free verifier output was reproduced under TITAN's pinned
Python 3.10 environment, and source compilation passed. The TITAN system did
not provide `pytest`, so the server verification did not claim a second full
test-suite run.

## 8. Resource and safety accounting

V5-R and V5-A used:

- zero GPUs;
- zero model loads or model queries;
- zero simulator resets or episodes;
- zero new benchmark outcomes;
- zero dataset/model downloads;
- zero protected Goal/final-population access; and
- zero manuscript modifications.

All durable server work was limited to `/home/ved/SAVR`. No system
configuration, unrelated process, service, university file, permission, or GPU
allocation was inspected or changed. During the documentation-sync checkpoint,
however, the
verification command briefly wrote its own generated semantic JSON to
`/tmp/savr-v5-doc-sync-verify.json`, which was outside the permitted project
path. The deviation was detected immediately; that exact temporary file was
removed and its absence verified. No unrelated path or content was inspected or
changed. Future remote verification output must be written beneath
`/home/ved/SAVR` or streamed directly without creating an external temporary
file.

## 9. Reproduction procedure

From the pinned repository revision:

```bash
python scripts/verify_acr_v5_isolation.py
python -m pytest
python -m compileall -q src scripts/verify_acr_v5_isolation.py
```

The first command is the minimal dependency-free semantic check. The complete
test command requires the repository development dependencies. Reproduction
must record Python/dependency versions, Git revision, platform, timestamp, and
the generated semantic digest.

## 10. Change-control rules from this checkpoint

Every future method or executor change must add a ledger entry containing:

1. date, branch, commit, and pull request;
2. predeclared hypothesis and acceptance/stop gates;
3. exact files and configuration fields changed;
4. before/after hashes for normative artifacts;
5. tests and adversarial cases added;
6. resources, data populations, seeds, and hardware used;
7. all observed failures and negative outcomes;
8. claim boundary; and
9. synchronization status for GitHub, local Documents/SAVR, and TITAN.

No post-output code or threshold change may be described as predeclared. Such a
change starts a new version and requires a new protected confirmation set.

## 11. Terminology map

| Term | Meaning |
|---|---|
| SAVR | Umbrella project: training-free, state-aware visual computation reuse |
| ACR | Asymmetric Camera Refresh: scene and wrist camera computation are treated differently |
| SA-BDP-ACR | V3 legacy state-aware batched dual-path implementation |
| IR-SA-ACR | V5 isolated-reuse state-aware ACR controller used by the existing batched path |
| Refresh (`R`) | Encode current scene and wrist observations |
| Reuse (`U`) | Reuse cached scene representation while refreshing wrist and downstream action computation |
| Software-correctness result | Proof by tests that code follows the frozen state-machine contract |
| Experimental result | Predeclared model/simulator measurement of success or efficiency; none yet exists for V5 |

## 12. Ledger conclusion

V5-A is a valid, reproducible software correction, not a positive-results
paper result. The formal report may describe the method and CPU invariants as
implemented facts. It must label task success, reuse, visual-work reduction,
and speed as open empirical questions until the gated evaluation protocol is
completed.

## 13. V5-B output-blind screening checkpoint

Preflight merge `e901fae` froze six candidates and the exact outcome-free A4
trace digest before replay. The committed analyzer executed twice per candidate
and the independent verifier passed. Machine result
`reports/runtime/acr_v5_b.json` has file SHA-256
`46d2033d1ea409062bbe6cc57afc46359c1047fa68c565ed6216ca8121c88080`
and semantic SHA-256
`8a9f15b818b58ed2868d4b1123a222a4c062507161ab7de911d8d233f3b1efec`.

The frozen rule selected `v5-a100-b40` with 35.48% reuse and 17.74%
theoretical logical visual-work reduction. Maximum streak was one and all
integrity counts were zero. This is positive offline mechanism evidence only;
success fields remained sealed and no GPU, model, simulator, download, new
task outcome, protected population, or manuscript change was used.
