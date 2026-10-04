# Contemporaneous-reference implementation and hook-preflight stop

Local date: 2026-09-12. No positive-method result.

## Implemented and CPU verified

- New worker and CPU analyzer, separate from every historical frozen worker.
- Exact 88-slot development schedule validation, balanced D/S ordering and eight
  native controls. No historical exact-rollout-length requirement is imposed.
- Same-input native/dense head/action tolerance remains 1e-6; processed float32
  commands must match byte-for-byte. Native diagnostic costs are not primary timing.
- Measured action loop follows the qualified original loop, including all early
  actions, settling, queue handling, controller replanning and repeated-ID reset.
  Query timing includes preparation; episode costs include other preparation,
  simulator work and controller costs. Generated/discarded/executed actions reconcile.
- Analyzer requires completed authenticated evidence, all 88 episodes and all
  136 timing calls. It reports unfavorable outcomes without a success-based stop
  or exclusion. Native repeatability discordance is flagged. Bootstrap units are
  tasks stratified by suite and paired timing trajectories, not individual queries.
- Feature-record format for the corrector preserves BF16 bits without converting
  them to FP16. Dense labels and compressed features require matching observation
  identifiers and a sample-level training allowlist. Shapes distinguish the
  eight penultimate action features from the 56 action-head input tokens.
  This is serialization infrastructure, NOT a qualified GPU extractor or trainer.

86 targeted CPU tests passed in TITAN's existing original environment with CUDA
hidden, including 24 new tests. No skips in that remote run. These tests did not
validate the stored-image/native-image boundary adequately, as the stop demonstrates.

## Frozen attempt and observed failure

Config: `configs/openvla/contemporary_hook_qualification_v1.json`.
SHA-256: `e642cffcd44e82e97999b3a24a0b6002f1a800844b8d9c6087dfbcc19166984a`.
Worker: `scripts/run_contemporary_reference.py`.
Owned PID 1247115, GPU 0. Started 2026-09-13 02:23:21 UTC.
Caps: 80 model calls, zero episodes, 1,800 seconds, aggregate GPU memory below
23,552 MiB, artifacts below 256 MiB. Source/checkpoint preflight passed before launch.

The worker stopped after loading the model, before the first model query:
`ValueError: expected evaluator-resized uint8 RGB camera pair`.
Elapsed after launch: 21.61 seconds. Peak aggregate memory: 14,763 MiB.
Zero model queries, zero robot episodes, zero training. No completed worker summary.
The process exited and GPU 0 returned to 6 MiB and 0% utilization.

## Cause and responsibility

The new worker called `CameraPair.capture` on raw HDF5 frames. That helper is
specifically for live evaluator-resized 224x224 images. All eight authenticated
training trajectories store both views at 128x128, with uint8 RGB. The prior
offline bridge retained these raw images until the released policy preprocessing
converted them. The new worker incorrectly imported the live-loop constraint
into the offline timing/qualification path. This was an assistant implementation
mistake, not a demonstrated model, research-method, CUDA, memory or server failure.

The preflight checked source/data hashes, but not this input-schema compatibility.
CPU fixtures used 224x224 camera arrays. They therefore missed the integration
boundary despite passing. The post-stop audit read shapes/dtypes for the eight
already consumed training trajectories only, not expert actions or locked data.

## Required recovery, not executed

1. Preserve this worker, config, evidence and failure status unchanged.
2. In a new version, separate raw offline camera-pair capture from the live
   224x224 helper. Validate and copy the authenticated 128x128 raw RGB arrays;
   retain released current/historical policy preprocessing exactly once. Do not
   weaken the live guard or add an undocumented extra resize/crop.
3. Move real input shape/dtype/frame-count checks ahead of model initialization.
   Exercise frame 0 and frame 1 in all eight sources on CPU before GPU allocation.
4. Add 128x128 offline versus 224x224 live regression cases, preparation-once and
   observation-immutability tests. Keep both paths in the preflight coverage matrix.
5. Freeze a separate recovery config/output, rerun source preflight and request
   authorization for one bounded recovery. No automatic retry was launched.

The 88-episode comparison was NOT started. No previous failed result was
reclassified. The learned corrector still has no robot-data training or closed-loop
result. All evidence and code are synced locally. No GitHub push or heartbeat.
No server files outside `/home/ved/SAVR` were modified; only explicitly selected
GPU aggregate telemetry and the owned runner's process health were inspected.
