# Streamlined route to an informative trained-method pilot

Date: 2026-09-12. User approved phase-level execution of the streamlined plan.
Earlier immutable protocols and results remain unchanged. This update removes
the requirement to wait for all comparator outcomes before implementing and
CPU-testing the proposed corrector. It does not waive correctness or evidence gates.

## Work now, with overlapping preparation

1. Targeted dense precision diagnostic: use a new execution boundary matching
   the forty-task baseline's float32 conversion. Run native-controlled and
   bridge-controlled episodes with shadow evaluation of both methods on every
   queried observation. Compare the final gripper-processed commands byte for
   byte and hash both observation/command traces. Keep the exact 78-step/10-query
   reference gate. This replaces another full single-query qualification cycle.
2. Implement the current-frame corrector and its action-only ablation while
   the diagnostic/comparator runs. CPU-test zero-init identity, detached teacher
   and backbone inputs, inference-mode tensor conversion, actual optimizer steps,
   and use of visual information. Synthetic fitting is an implementation test,
   not an experiment on LIBERO and not evidence for the research hypothesis.
3. After dense correctness is resolved, freeze the paired comparator evaluation
   and current-frame compression screen. Include complete-query and episode
   cost. Continue feature extraction/trainer preparation while these run.
4. Freeze a small actual trained-method pilot: dense, compression alone, action-
   only correction, fresh-visual correction, with the qualified published
   comparator as the external reference. Same development conditions, matched
   training labels/budget, and prespecified decision criteria. Cap initial
   training inputs at 4,000 queries; use a smaller throughput/overfit sample
   first. No locked evaluation data may be opened for this pilot.
5. Expand only after an improvement in closed-loop behavior remains useful after
   the corrector's measured overhead. One planned learner-state refinement round
   is permitted only after its data collection and evaluation are frozen. No
   repeated outcome-driven threshold changes or claims from offline loss alone.

No need for user approval after routine tests or code edits within these phases.
Stop for material scientific changes, resource conflicts or failed correctness
gates. Do not treat this as permission to automatically retry failed GPU runs.
Use only project-local files and one coordinated GPU; no downloads or installs.

## Frozen targeted diagnostic

Config: `configs/openvla/dense_precision_trace_v1.json`.
SHA-256: `2fcc0591cb04adeb28aace93745a07411f8b376908bdd79111737dc02f0a693f`.
Worker: `scripts/run_dense_precision_trace.py`.
Output: `results/dense-precision-trace-v01`, exclusive creation, no resumption.

Same already-consumed Spatial task/state/seed as the stopped integration run.
Order: native controls first episode, precision-aligned dense bridge controls
second. On each observation run native then bridge, without compression, and
execute only the designated arm's float32 chunk. Capture head inputs, normalized
and raw actions; tolerance 1e-6. Final processed commands must match byte-for-byte.
Both episodes must match success/78 steps/10 queries and their complete query
observation/command hash sequences. Intended 40 calls; hard cap 112 calls, two
episodes, 1,200 seconds, aggregate GPU memory below 23,552 MiB, artifacts below
256 MiB. No compressed outcomes are read or produced by this diagnostic.

Nine new CPU boundary/reconciliation tests passed before launch. Original
source/checkpoint and new file hashes must verify. GPU 0 must be freshly idle.
Monitor only owned health/counts/bytes/elapsed/aggregate telemetry while running.
On a stop preserve technical evidence; no automatic retry or gate change. On
completion independently reconcile before proceeding. This is not a latency run.

## Corrector starting implementation

`CurrentFrameCorrector`: eight action queries, current compressed action-head
penultimate features, normalized base actions, current normalized state and a
current instruction embedding. The visual version cross-attends to all 512
current projected patches, with camera and 16-by-16 position embeddings. Default
width 256, two blocks, eight heads; all weights FP32. Output is an unconstrained,
zero-initialized residual added to the normalized base chunk. No cache features,
clipping, routing or extra losses. The action-only ablation removes visual
projection/cross-attention, with matched data and optimization budget planned.

Inputs and labels are detached. Inference-mode features are cloned into normal
training tensors to avoid saving inference tensors in autograd. No backbone
backward graph is built. The default-size parameter count and real optimizer
memory/latency must be measured rather than borrowed from CAC. Input validation
and copying costs must be included in eventual end-to-end timing.

Training objective remains mean L1 to qualified dense normalized action chunks.
Actual hard-compressed model features are required for data generation; synthetic
CPU tensors do not satisfy this requirement. Corrector code alone does not mean
that a trained or evaluated proposed method exists.
