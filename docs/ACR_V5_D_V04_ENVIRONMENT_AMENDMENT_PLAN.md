# ACR V5-D v04 Compatible-Environment Amendment Plan

Status: **PROPOSED; IMPLEMENTATION AND EXECUTION NOT AUTHORIZED**

Date: 2026-08-11

## 1. Purpose

V03 established that the current TITAN RTX node cannot execute either frozen
optimized backend: `sm_75` cannot compile the BF16 TorchInductor core, and raw
capture exhausts 24 GB before correctness. V04 should change only the execution
environment so the already frozen method can finally reach its real-tensor
correctness and timing gate.

The selected controller, checkpoint, dependencies, tensor inputs, four paths,
compile-first/raw-second waterfall, 111-query schedule, tolerances, bootstrap,
gates, and claim boundary must remain unchanged.

## 2. Required target capability

Before any workload, identify one university GPU node with:

- CUDA compute capability at least 8.0, so the observed BF16 PTX requirement is
  supported;
- at least 32 GiB physical memory, providing explicit headroom over the
  observed 23.2246 GiB raw reservation; an A100 40 GB or equivalent is the
  preferred target;
- a project-owned writable root explicitly approved for SAVR;
- no need for `sudo`, system-wide changes, or interference with other jobs;
  and
- a method to select one idle device using only aggregate telemetry.

These are entry requirements, not performance claims. Exact hardware identity,
driver, CUDA runtime, PyTorch/Triton stack, free storage, and checkpoint hashes
must be recorded before freezing v04.

## 3. Reproducibility and anti-shopping controls

Choose the target environment before model output. Do not test several GPU
types and retain the best timing. Establish one primary node/device class and
one contingency class in advance. Preserve the compiler-first waterfall and
all failure rules.

If reproducing the pinned environment on the new node requires a dependency
change, document and validate it before execution. Prove the scientific
configuration remains byte-structurally equal to v03. V01–v03 remain immutable
and excluded from the v04 query budget.

## 4. Pre-GPU checkpoints

V04 implementation must stop before GPU selection and demonstrate:

1. the exact cluster host and writable project boundary are authorized;
2. source, dependency, checkpoint, and dataset identities are pinned;
3. no download or copy begins without a storage estimate and explicit scope;
4. deterministic CPU/preflight verification passes with temporary files inside
   the approved project root;
5. the v04 run ID and result paths are unused;
6. the v03 technical-stop digest is linked; and
7. the complete repository and remote workspace are clean.

## 5. Execution and stopping rule

After separate authorization, select at most one GPU and run v04 exactly once.
Only a complete, independently verified 111-record result may advance to V5-E.
Any compiler/raw failure, memory breach, parity failure, restoration uncertainty,
or resource violation stops before simulator work.

Access to another cluster is not inferred from general permission. The exact
hostname, authentication route, and approved project directory are required
before this plan can be implemented.
