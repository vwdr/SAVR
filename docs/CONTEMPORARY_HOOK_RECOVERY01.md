# Stored-image boundary recovery 01

Date: 2026-09-12 local. The user approved the correction and one frozen GPU
recovery check. No automatic retry. Stop after this technical qualification;
do not dispatch the 88-episode evaluation in this recovery authorization.

The failed v01 worker/config/evidence remain unchanged. Its zero-query failure
was a programming error: the live-camera 224x224 guard was applied to raw 128x128
demonstration images. The recovery introduces a separately validated raw camera
pair for offline inputs. It preserves exact uint8 RGB bytes, copies both views,
and does not pre-resize or pre-crop. The released preprocessing still handles
each current and historical view once. The live 224x224 guard remains strict.

Before loading any model, verify actual frames 0 and 1 from all eight previously
consumed training trajectories: source identity, split, unique trajectory IDs,
frame availability, image dtype/shape, state dtype/shape and finite state values.
No expert actions or held-out data are read. A separate CPU-only preflight calls
the released preprocessing and existing current/history wrapper, checks exact
pixel equality against direct released preprocessing, counts preparation calls
and verifies that raw inputs remain unchanged. Freeze the 16-check report hash.
The worker reloads and compares all 16 observation hashes before GPU/model setup.

The GPU check is unchanged: 80 total model calls, 24 ordered trajectory/arm
checks, zero simulator episodes, zero training. Compare native on frame 0 and
dense/SpecPrune on reset two-frame traces with and without audit hooks. Require
byte-identical processed commands, identical selection metadata/prior indices,
32 observed layers, complete records, source/checkpoint preservation and budgets.
The independent CPU reconciler rejects missing, reordered, duplicated or failed
checks and incorrect model-call accounting. No scientific success threshold.

Caps remain 1,800 seconds, aggregate selected-GPU memory strictly below 23,552
MiB, output below 256 MiB. Select GPU 0 only after fresh aggregate idle telemetry;
the worker repeats that check. No unrelated allocation/process inspection.

Worker: `scripts/run_contemporary_reference_recovery01.py`.
Configuration: `configs/openvla/contemporary_hook_qualification_v2_recovery01.json`.
Output: `results/contemporary-hook-qualification-v02-recovery01`, exclusive create.
Real-input preflight: `scripts/preflight_contemporary_inputs_recovery01.py`.
Keep monitors blinded to partial hook outcomes. After completion verify hashes
and all counts. On failure preserve technical evidence and stop without retry.
Sync code/evidence/status to the local SAVR folder. No GitHub push or changes
outside `/home/ved/SAVR` on TITAN. Passing is an implementation result only, not
a trained-corrector result, speedup or positive-paper result.
