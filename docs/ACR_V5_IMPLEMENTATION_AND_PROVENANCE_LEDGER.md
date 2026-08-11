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

## 14. V5-C executor design freeze

The V5-C research/design checkpoint preserves the V5-B controller and selects
a project-owned static-buffer executor contract with separate wrist-visual and
downstream-action GPU cores. It rejects whole-`predict_action` capture because
the pinned host path includes CPU/NumPy transfers and dynamic logic. The freeze
records executor identities, complete compatibility fields, lifecycle,
prelaunch/postlaunch failure semantics, 20 CPU acceptance requirements, and a
future compiler/raw-CUDA-graph waterfall.

This checkpoint adds only
`docs/ACR_V5_C_EXECUTOR_RESEARCH_AND_DESIGN.md`,
`docs/ACR_V5_C_CPU_EXECUTOR_PROTOCOL.md`,
`configs/acr/v5_c_cpu_executor_freeze.json`, freeze tests, and status/decision
updates. It changes no controller, adapter, executor, result, model, simulator,
environment, protected data, or manuscript.

## 15. V5-C static executor implementation checkpoint

Implementation commit `7159f4d475e90ac2f3454be298d4bc92939e69c0`
adds the reference executor, owned-static-buffer executor, isolated execution
adapter, adversarial tests, deterministic verifier, and immutable machine
record. The executor source SHA-256 is
`d2ff398a0933ba8f6c0e6bd3e1e782f928d0357f0864a6309912d9b513ec67e3`;
the integration source SHA-256 is
`50f9791cf8d5e3c128f33f7b327f4e00c878e36add217ed637225e0e1255f260`.

The machine record semantic SHA-256 is
`f7a8d11d4574add57caa630c03463375421d9482984478be769f497b1c9d0b66`.
It was reproduced exactly by local Python and TITAN system Python 3.10.12. The
complete local suite passed 293 tests; compilation, bootstrap, package build,
change-scoped formatting, and source/test lint passed. TITAN's system Python
has no pytest, so no environment was changed; its dependency-free verifier,
compilation, and bootstrap gates passed.

An initial implementation attempted to add executor reasons to shared
`src/savr/acr/types.py`. The full regression suite correctly detected that this
would change a source digest embedded in immutable V5-A evidence. The change
was removed before publication, prior evidence stayed untouched, and executor
reason insertion was isolated in the new integration adapter. This is the
required evidence-preserving resolution, not an omitted failure.

V5-C used zero GPU, model query, simulator episode/reset, download, environment
change, new outcome, protected population, or manuscript edit. It establishes
only CPU software-contract correctness and advances only to V5-D protocol
preparation.

## 16. V5-D real-tensor protocol freeze

The V5-D protocol checkpoint preserves the selected controller and V5-C
executor interfaces and freezes the next real-model feasibility question
before implementation or output. It adds the research/measurement design,
normative protocol, machine-readable freeze, drift tests, protocol report, and
status/decision updates. It changes no controller, executor, adapter, result,
model, simulator, environment, protected data, or manuscript.

The machine freeze semantic SHA-256 is
`f445cf5d1a5ec6877ebea46ccc3883a11a676b38cb33a711ee4b74baf22f53f8`.
It fixes exact real tensor shapes, seven correctness queries, two warm-ups per
path, all 24 four-path permutations, 96 timed queries, the 111-query hard cap,
paired-bootstrap seed/gates, compiler/raw technical waterfall, memory/resource
limits, recovery, and the claim boundary.

Preparation used zero GPU, model query, simulator episode/reset, download, new
outcome, protected population, or manuscript edit. A reconciliation check
caught inaccurate copied hash suffixes before publication; every affected hash
was corrected against the authoritative repository or exact pinned TITAN file.
The next checkpoint is backend implementation with CPU/fake-backend tests.
GPU selection remains prohibited until that code merges and the user is
explicitly coordinated.

During that pre-GPU implementation, deterministic input regeneration found
that the six frozen A/B image, instruction, and midpoint hashes also had
incorrect copied suffixes. They were corrected against the immutable V3-C
machine record before backend publication or output. This changed the freeze
semantic digest but no scientific method, threshold, schedule, tolerance,
gate, or resource boundary.

## 17. V5-D pre-GPU implementation checkpoint

The implementation checkpoint adds a separately versioned V5-D orchestration
and PyTorch backend layer, aggregate GPU selector, launch wrapper, exact runner,
paired analyzer, independent verifier/finalizer, deterministic preflight, and
adversarial tests. Validated V5-C source and evidence remain unchanged.

The implementation discovered that the fake V5-C executor's single dtype could
not represent the real frozen bfloat16 plus int64/bool contract. Separate
V5-D-only eager/static subclasses preserve the V5-C execution/lifecycle
semantics while enforcing each real dtype. It also corrected the six
deterministic input hash suffixes against immutable V3-C before GPU output and
records pinned PyTorch 2.2's implicit capture-end graph instantiation.

Preflight semantic SHA-256 is
`db097ca8cab44d474a65e22888a72da8c4c6e2489a31188abea67c7ed55bff98`.
It verifies all 111 query identities, compile-first/fresh-process raw rules,
post-output fallback prohibition, hot-core host-side-effect exclusion,
aggregate-only selector, project-local caches, and zero simulator/outcome
paths.

This checkpoint used zero GPU, model query, simulator episode/reset, model,
dataset, or TITAN download, new outcome, protected population, or manuscript
edit. Local package validation used an isolated temporary build environment
without changing the repository or persistent environments. It advances only
to explicit user coordination before aggregate GPU selection.

## 18. V5-D v01 zero-query technical stop

The user explicitly authorized entry into the bounded one-GPU phase. The
aggregate-only selector took the frozen three samples, found all four devices
eligible, and selected the lowest physical index, GPU 0. Its selected samples
were consistently 6 MiB used and 0% utilization. Launch-manifest semantic
SHA-256 is
`194b5fbae6cdf8b0d987ef153040b2d162b7c15932b7c824a916d8ce44fab165`.

The launcher created an empty project-local `LIBERO_CONFIG_PATH` and invoked
the compile-first runner. Before model load, importing the pinned upstream
LIBERO evaluation module imported LIBERO itself. Because `config.yaml` did not
exist, LIBERO entered its interactive first-use dataset-path prompt. Closed
stdin caused `EOFError`. The process returned nonzero and correctly did not
enter raw fallback.

This exposed a preflight coverage gap: project-local cache-root placement was
tested, but a fresh absent LIBERO config and closed-stdin import were not. v01
is preserved and cannot be retried. It contains zero model queries,
backend-preparation launches, correctness records, warm-ups, timed queries,
simulator operations, downloads, or outcomes. No checkpoint write or
restoration was required; source and checkpoint trees stayed clean. Post-stop
selected-GPU telemetry returned 6 MiB used and 0% utilization.

The curated technical-stop semantic SHA-256 is
`edf5872fa818f5806601f52143cb17cec7dd4974e03cc4e2ed43c3d042fb4412`.
The proposed v02 route changes only non-interactive LIBERO configuration and
pre-import failure recording while preserving every method, backend,
correctness, timing, statistical, resource, and claim gate. v02 requires a new
run ID, merged pre-GPU correction, and separate execution authorization.

## 19. V5-D v02 pre-GPU recovery implementation

Recovery implementation commit `9f187122dfa8441c8156a9083c8e507e5cb1bedd`
adds a compact machine overlay rather than modifying the v01 freeze. Resolution
changes the run ID to `acr-v5d-real-tensor-feasibility-v02`, links the immutable
v01 stop digest, and permits only canonical LIBERO configuration plus an outer
pre-model stop envelope. The selected method, pinned stack, source/checkpoint
hashes, backend waterfall, inputs, tensors, correctness schedule/tolerances,
timing permutations, bootstrap/gates, GPU selection rule, memory/resources,
recovery, and claim boundary remain byte-structurally equal to v01.

The config helper emits deterministic JSON-valid-YAML with exactly five pinned
LIBERO paths. It refuses path escape, symlinks, changed keys/bytes, overwrite,
or a second create. The launch wrapper invokes it before either backend, and
the runner verifies its independent attestation before upstream import. An
uncaught pre-model failure now writes a zero-query immutable record and exits
4; only exact exit 20 can activate the frozen raw transition.

All 329 local tests passed. The deterministic v02 preflight semantic SHA-256 is
`d7c3ed40cc9d5760a846cb15c688fa5c776cbac8f243d948376d16e64427a695`.
On TITAN, the pinned upstream evaluation module imported with stdin closed,
interactive input forbidden, and `CUDA_VISIBLE_DEVICES` empty.
`torch.cuda.is_initialized()` remained false before and after. Import-preflight
semantic SHA-256 is
`a3ffc574631e8e250ab8021c0f8b99e0bf329a1e82d085499fb8e19747dd3490`.

This checkpoint used zero GPU inspection, model load/query, simulator
instance/reset/episode, download, new task outcome, protected population, or
manuscript edit. It advances only to explicit user coordination before v02
aggregate GPU selection.

## 20. V5-D v02 pre-correctness technical stop

The user explicitly authorized the bounded v02 one-GPU run. Three
aggregate-only samples found all four TITAN RTX devices eligible and selected
physical GPU 0 by the frozen lowest-index rule. Launch-manifest semantic
SHA-256 is
`0c723dd5ff93c0dfe4544dd2f50b6e7ff91409fb3ba47059508701fa8081cec8`.

Canonical LIBERO initialization passed and the pinned model loaded. The first
compiler preparation call failed before correctness because Triton emitted
BF16 PTX that requires `sm_80` or newer, while TITAN RTX is `sm_75`. The
technical attempt semantic SHA-256 is
`2eb417293ece405c8c161ab275926318766698c600e9d31a1ea89ad56934ec68`.
The curated technical-stop semantic SHA-256 is
`0a30bd847bf2e1549c376200e559a23c670b33c0b01215926c90a15704487661`.

Raw fallback was not started. Although the compiler failure was technically
eligible, the restoration guard found two loader-created
`.back.<timestamp>` files outside its cleanup allowlist and failed closed. The
protected files themselves had already been restored to their exact frozen
hashes. The two duplicates were independently hash-verified and removed, after
which the pinned checkpoint validator and all three source-tree cleanliness
checks passed.

V02 contains one compiler preparation launch but zero full model queries,
correctness records, warm-ups, timed records, simulator operations, downloads,
or task outcomes. No reward or success field was accessed. It is neither
positive nor negative method evidence. V02 remains immutable. The proposed
v03 correction is limited to exact loader-backup restoration coverage and
requires a new reviewed implementation checkpoint plus separate GPU
authorization.

## 21. V5-D v03 pre-GPU restoration recovery

Implementation commit `a772c72a7e3eba693ee348875bd8c565444a0819`
introduces the v03 immutable run ID and exact restoration helper. The overlay
links the curated v02 technical stop and preserves every scientific section
byte-structurally. Its resolved semantic SHA-256 is
`a9447cd385b4229e54cf85ba8fc7e06e4b4d283b9ac5c655e0c5201fb5d3f297`.

Before model initialization, the runner captures protected bytes/hashes and
the top-level checkpoint baseline. Restoration accepts only new regular,
non-symlink files matching the frozen protected-name suffixes, requires backup
content to match its original, rejects pre-existing drift, restores protected
bytes, removes only verified artifacts, and requires exact final inventory.
The attempt record distinguishes byte restoration from backup cleanup. Any
failure clears raw-transition permission.

All 341 local tests passed. The boundary-compliant TITAN repeat passed 341
tests and 9 subtests with its temporary root under `/home/ved/SAVR/results/`.
The initial TITAN pytest invocation had used pytest's default external
temporary directory; after detection, no external path was inspected or
manually changed, and that invocation is excluded from acceptance evidence.

The deterministic preflight semantic SHA-256 is
`f25d7f2dd743bf0c4fbe8a56420ba94d136c8cfc6b5baaeedf473e5a5a6ab163`.
The closed-stdin pinned import semantic SHA-256 is
`f8b9002d0998345c7f1a423003a180cfc5a406d52f85e1f3ac2d1f60fe22cdc5`;
the curated CPU-verification semantic SHA-256 is
`37587ccd329cfc68672ef048a7422fb8825944a2a61260d1631b8532c2efdf95`.
CUDA remained uninitialized. This checkpoint used zero GPU inspection, model
load/query, simulator operation, download, outcome access, or manuscript edit.
It advances only to explicit user coordination before v03 GPU selection.
