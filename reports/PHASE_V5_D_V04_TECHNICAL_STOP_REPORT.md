# Phase V5-D v04 Technical-Stop Report

Status: **TECHNICAL STOP BEFORE RAW MODEL LOAD — NO METHOD-PERFORMANCE RESULT**

Date: 2026-08-11

## 1. Execution outcome

After explicit authorization, the frozen aggregate selector chose physical GPU
0. All three selection samples showed 6 MiB used and 0% utilization.

The compiler process loaded the model and repeated the anticipated first-call
TorchInductor failure: its generated BF16 PTX requires `sm_80` or newer, while
TITAN RTX is `sm_75`. Exact checkpoint restoration passed, the two verified
timestamped loader backups were removed, and the fresh-process raw transition
was authorized.

The raw process then applied its frozen immediate GPU revalidation. Memory had
already returned to 6 MiB, but that single sample reported 33% utilization,
above the 5% eligibility limit. It stopped before model load, CUDA-graph
preparation, or any full query. A later aggregate-only check of the same GPU
showed 6 MiB and 0% utilization.

## 2. Scientific interpretation

V04 did not exercise the shared-pool backend. It contains zero raw preparation
launches, full model queries, correctness records, schedule warm-ups, timed
records, simulator operations, rewards, success fields, or task outcomes. The
analyzer and finalizer were not run.

This is neither positive nor negative evidence about the selected state-aware
visual-refresh method, nor evidence that shared-pool capture does or does not
fit beneath 23 GiB. It is a transition-control technical stop: a single
immediate post-compiler utilization sample rejected the otherwise idle selected
GPU before raw model load.

## 3. Integrity and resource audit

- Execution revision:
  `f95d34397e4839a9b9f1f59793e370973de5ccac`.
- Selected GPU: physical 0, UUID
  `GPU-bb2451d6-2989-a112-5c18-8892943710e4`, compute capability 7.5.
- Compiler preparation: one launch; 80.4387 seconds.
- Raw preparation: zero launches; raw model not loaded.
- Maximum concurrent model processes: one.
- Post-stop selected-GPU telemetry: 6 MiB used, 0% utilization.
- No simulator, download, task outcome, unrelated process, or allocation was
  accessed.

Checkpoint restoration is exact. No loader backup remains. The protected
hashes and SAVR, OpenVLA-OFT, and LIBERO source trees are clean.

## 4. Immutable evidence

- Launch semantic SHA-256:
  `3eaa5c0bb62cf972280e526788c830190a0cf8f3105a2d1b50c6639a1dcdc455`.
- LIBERO-config semantic SHA-256:
  `c94ca0149b512fbc8e2e72ab58e070b240e4f13bfc4cffddc40f42c296224321`.
- Compiler-attempt semantic SHA-256:
  `fb634f4b93557ea71ce6a82a250a63a467ba4b73bd97d462c177d51f3ec335df`.
- Raw-transition semantic SHA-256:
  `da3f75c751a6c2eb2a8db06836abec91084bd24bbe4cc747b525bbfb8033ffb3`.
- Raw pre-model stop semantic SHA-256:
  `b8b8c347d07eeaeedb5f942ba1c2aa1f7b045c1e29a0037b8b046ada4318a80c`.
- Curated technical-stop semantic SHA-256:
  `a3515180022df7938b50956851a2ca05b698819da38b387ddc23b54e59769811`.
- Curated record: `reports/runtime/acr_v5_d_v04_technical_stop.json`.

The complete create-once run evidence remains on TITAN under
`results/acr-v5d-real-tensor-feasibility-v04/`.

## 5. Disposition

`STOP_NO_AUTOMATIC_RETRY_PREPARE_SEPARATELY_FROZEN_TRANSITION_RECOVERY`

V04 will remain immutable. V5-E remains ineligible. Any next attempt requires a
new run identity and a separately researched transition rule that preserves
the aggregate-only, noninterference safeguard while avoiding a decision from
one immediate post-process utilization sample. The scientific method, backend,
query schedule, memory cap, correctness gates, and timing gates must remain
unchanged.
