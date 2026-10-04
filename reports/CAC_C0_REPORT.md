# Cache Action Correction Phase C0 Report

**Date:** 2026-08-31  
**Decision:** **C0 COMPLETE — PASS; STOP BEFORE C1**  
**Execution:** CPU-only C0 Recovery 01 after preserved technical stop 01

## Outcome

CAC C0 completed the literature, identity, population, method, manifest,
statistics, recovery, and resource freeze required by Protocol V2. The
versioned recovery corrected only the scheduler assertion documented in
`CAC_C0_TECHNICAL_STOP_01.md`; it changed no scientific setting.

C0 establishes that the experiment is specified and resource-feasible. It is
not evidence that CAC repairs actions or preserves closed-loop success. C1 is
not authorized.

## Literature and novelty decision

The primary-source audit through 2026-08-30 found no materially identical method
containing all five narrow CAC elements. Residual action correction is not new:
A2C2 is the closest action-space neighbor; Latent Bridge is the closest learned
latent/K/V neighbor; Action-JND, LAC, AC2-VLA, and Gated VLA-Cache are the
closest action-aware cache selection/routing methods. CAC's surviving question
is specifically whether current accelerated-path visual evidence plus exact
recursive mixed-age cache provenance can directly predict the dense action
effect without reconstructing fresh K/V or issuing a hidden dense call.

VLA-Cache is the executable substrate. Public AC2-VLA and Latent Bridge code
uses materially different backbones/training systems. No official Gated
VLA-Cache, Action-JND, or LAC implementation was found that was both compatible
with the pinned OpenVLA-OFT checkpoint and admissible under the one-GPU/no-new-
large-download boundary. These methods will be compared conceptually; no
empirical-superiority claim over an unreproduced method is permitted.

Evidence: `docs/CAC_C0_LITERATURE_COLLISION_AUDIT_2026-08-30.md`.

## Authenticated identities and protected populations

- project base: `0a274b6ea0c0748adabe386ffa0d61670c4ece7a`
- OpenVLA-OFT: `e4287e94541f459edc4feabc4e181f537cd569a8`
- VLA-Cache: `a4909880573868dee2769343d52e793c0341678b`
- LIBERO: `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- checkpoint revision: `638918f3d1c2e43a39a8a20772bdb8b91835e4b7`
- dataset revision: `f13aa24a3da8c43c7225569f28c562979fa0e35a`
- exact `D62_BAL_PT1` profile and vectorized source helper reauthenticated
- P1 trajectory/query/split indexes reauthenticated
- all 280 locked-test demonstrations remained sealed
- all state-ID 10-49 outcomes remained sealed

## Frozen manifests

| Manifest population | Count |
|---|---:|
| Trajectory roles | 2,000 |
| C2 primary contracts | 3,600 |
| C2 all-fresh controls | 240 |
| C2 exact repeats | 360 |
| C3 contracts | 13,200 |
| C7 locked offline contracts | 2,880 |
| Simulator population rows | 2,240 |

The trajectory roles are exactly 960 adapter-fit, 200 architecture-selection,
240 checkpoint-validation, 320 development-calibration, and 280 locked-test.
Every locked contract contains identity-only scheduling metadata and a null
transition label.

Two LIBERO-Goal tasks have no eligible demonstrated gripper-transition windows
in the permitted C2 train roles:

- `open_the_middle_drawer_of_the_cabinet`; and
- `push_the_plate_to_the_front_of_the_stove`.

Their zero availability is recorded rather than manufactured or silently
replaced. Every task with at least eight available transition contracts meets
the eight-contract minimum.

Canonical recovery evidence: `reports/cac_c0_recovery01/`.

## Frozen method and systems contract

- D62 cameras, tile grid, onset layers `(2,6,9,11)`, recursive physical-source
  semantics, age-four reset, and eight-action cadence are unchanged.
- Feature records contain 32 current tile tokens, 128 layer/camera/tile source
  deltas, eight `Z_C` tokens, the cached action, target residual, proprioception,
  provenance, and validity mask. Fixed payload is 1,378,328 bytes, aligned to
  1,380,352 bytes; sidecars are capped at 512 bytes.
- The full two-block width-256 cross-attention candidate contains exactly
  5,070,599 trainable parameters and an estimated 233,601,024 linear/attention
  MACs per query.
- The frozen pooled summary has 3,521 inputs. Ridge has 24,654 parameters and
  the MLP has 1,806,855 parameters.
- The dense-minus-cache target, support guard, Smooth-L1/BCE losses, gripper
  threshold 0.5, three seeds, optimizer, schedule, matched controls, and four
  independently trained ablations are explicit in the C0 configuration.
- The fixed C7 analyzer rejects incomplete populations and enforces G3-G6 in
  order, including per-suite/task reliability checks, hierarchical timing, and
  Holm correction.

## Resource accounting

| Phase | Exact capped work | Forecast |
|---|---:|---:|
| C1 | 160 model calls | 1 hour; 4 GiB |
| C1H | at most 480 episodes | 10 hours |
| C2 | 17,640 model calls | 8 hours; 5,799,628,800 bytes (5.40 GiB) |
| C3 collection | 57,200 model calls | 24 hours |
| C3 + C7 feature storage | 16,080 records | 22,204,293,120 bytes (20.68 GiB) |
| C4 | at most 600 model calls | 2 hours |
| C5 | at most 480 new episodes | 12 hours |
| C6 | 4,800 teacher calls and at most 480 episodes | 15 hours combined |
| C7 primary | 4,800 episodes | 96 hours |
| C7 locked construction | 12,480 model calls | 6 hours |

The original C3 allowance of up to 26,000 contracts could not coexist with the
60,000-call cap because each independent recursive contract requires `h+2`
calls. C0 therefore lowered C3 to 13,200 exact contracts and 57,200 calls. No
cap was raised. Project free space at audit time was approximately 349 GB;
peak GPU use remains capped strictly below 23,552 MiB.

## Power boundary

The analytic paired calculation gives approximately 81.2% power at true
CAC-dense equality and 10% discordance. The required task-then-state simulation
was more conservative under the historical task-heterogeneity pattern:

| Discordance | True difference | Simulated power (100 outer datasets) |
|---:|---:|---:|
| 5% | 0 pp | 78% |
| 10% | 0 pp | 40% |
| 15% | 0 pp | 32% |
| 20% | 0 pp | 10% |

At true differences of -1 pp or -2 pp, simulated power was lower still. The
simulation uses the exact 20,000-replicate task-then-state bootstrap inside each
outer dataset and reports Wilson intervals. Its outer sample is deliberately
reported as coarse planning evidence, not a result.

The lower hierarchical forecast governs: final dense noninferiority is
plausibly powered only if paired discordance is very low. C5/C6 must update the
forecast from observed paired discordance without changing the margin,
population, or final analyzer. This limitation does not block C1, but it is a
material positive-paper risk.

Evidence: `reports/cac_c0_power_simulation_v1.json` and
`reports/cac_c0_recovery01/power_table_v1.json`.

## Recovery, tests, and protection

- first attempt root preserved empty: yes
- Recovery 01 root separate and immutable: yes
- expanded TITAN C0 tests: 6/6 passed
- generated manifest semantic hashes: all passed independent validation
- generated count/role/population checks: all passed
- automatic retries: none
- GPU/model/simulator calls: zero
- terminal outcomes: zero
- locked-test action values: unopened
- state-ID 10-49 outcomes: unopened
- downloads: zero
- changes outside `/home/ved/SAVR`: none

## Phase decision

Every C0 exit field is explicit and sealed. Compact C1/C2 storage is below
10 GiB, C3 storage is below 50 GiB, and the protected populations remain
sealed. **C0 passes and stops before C1.** C1 requires separate user approval.
