# ACR V5-D v03 Recovery Plan

Status: **PROPOSED; IMPLEMENTATION AND GPU EXECUTION NOT AUTHORIZED**

Date: 2026-08-10

## 1. Objective and unchanged scientific contract

V03 will answer the same frozen V5-D feasibility question. It must preserve
the selected `v5-a100-b40` controller, checkpoint and dependency revisions,
deterministic tensors, four measured paths, compile-first/raw-second backend
waterfall, 111-query schedule, tolerances, bootstrap seed and procedure,
timing gates, memory/resource caps, and claim boundary.

V03 may correct only checkpoint-loader artifact restoration and extend
preflight coverage for the observed failure. It may not tune the method,
change thresholds, choose a result-favorable backend, access a simulator, or
inspect task outcomes.

## 2. Exact restoration correction

Before model initialization, capture the checkpoint inventory, protected
bytes, and protected hashes. After any attempt:

1. restore the three protected files byte-for-byte;
2. calculate the exact set difference from the pre-load inventory;
3. accept only regular, non-symlink files matching the loader's observed
   protected-name form
   `(config.json|modeling_prismatic.py).back.YYYYMMDD_HHMMSS` or the already
   frozen `.bak`/`backup` forms;
4. reject every directory, symlink, unrelated name, changed baseline file, or
   non-allowlisted artifact;
5. remove only the verified new allowlisted artifacts;
6. require the complete inventory and protected hashes to equal the pre-load
   baseline; and
7. make the operation idempotent so a second validation changes nothing.

The attempt record must distinguish protected-byte restoration from auxiliary
backup cleanup. Raw fallback may be permitted only when both gates pass.

## 3. Required pre-GPU verification

The v03 implementation checkpoint must add tests that inject:

- the exact two `.back.20260810_212317` forms observed in v02;
- multiple permitted timestamps and already-supported backup forms;
- an unexpected regular file, directory, and symlink;
- a changed protected file and a changed pre-existing non-protected file;
- partial cleanup failure; and
- repeated restoration.

Every negative case must fail closed without granting a raw-transition permit.
The exact observed v02 artifact set must restore successfully to the frozen
inventory. Run the full regression suite and deterministic preflight locally
and on the pinned TITAN environment without initializing CUDA.

## 4. Hardware/backend handling

V02 established that the pinned TorchInductor BF16 core cannot compile for a
TITAN RTX (`sm_75`). V03 must record the selected device's compute capability
before model load, but it must not silently skip or replace the frozen
compile-first backend. On the current TITAN node, the expected pre-output
compiler technical failure may transition to a fresh-process raw-CUDA-graph
attempt only after exact restoration succeeds.

If the university provides a separately configured Ampere-or-newer node, its
host, project boundary, pinned environment, checkpoint hashes, and single-GPU
availability must be validated and documented before selection. Access is not
assumed from the user's general cluster permission. Moving to such a node
requires a separately frozen execution-environment amendment; it cannot be a
post-result backend choice.

## 5. New immutable identity and acceptance checkpoint

Implementation requires a new run ID,
`acr-v5d-real-tensor-feasibility-v03`, linking the v02 curated technical-stop
digest. V01 and v02 remain immutable and excluded from the v03 query budget.

Before any v03 GPU inspection or execution, require:

- reviewed and merged source/config/test changes;
- a resolved v03 freeze proving all scientific sections equal v02;
- clean SAVR, OpenVLA-OFT, LIBERO, and checkpoint inventories;
- deterministic local and pinned-environment preflight records;
- confirmation that no GPU/model/simulator/outcome access occurred during the
  correction checkpoint; and
- explicit user authorization for one-GPU execution.

## 6. V03 execution and stop rules

After authorization, take only aggregate device telemetry and select one idle
GPU by the frozen rule. Preserve the compiler attempt. A raw transition must
use a fresh process and may occur only after a permitted pre-output compiler
technical failure plus exact checkpoint restoration.

Any output-bearing compiler failure, raw capture failure, parity failure,
memory/resource breach, restoration uncertainty, or source/checkpoint drift
stops V5-D. Only a complete independently verified V5-D pass may advance to
planning V5-E. Even that pass would establish bounded real-tensor feasibility,
not task success or a positive paper result.
