# Bounded live SpecPrune integration check

Date: 2026-09-11. User authorized continuation of the integration check.
This is not the matched development performance experiment or adapter training.

Frozen configuration: `configs/openvla/specprune_episode_qualification_v1.json`.
SHA-256: `ae52f5f8d535d5f786a050ce46800dbaff4560f007ee7e88bb0c0dd9bf599de7`.
Worker: `scripts/run_specprune_episode_qualification.py`.
Output: `results/specprune-episode-qualification-v01`, created exclusively.

## Fixed sequence and gates

Use frames 0 and 1 of each of the eight authenticated, previously consumed
training trajectories. Their complete source files remain hash-authenticated.
For each trajectory run, in order:

1. Native and new dense bridge on frame 0.
2. Native and new dense bridge on frame 1.
3. First compressed query on frame 0 with its own pair as history.
4. Second compressed query on frame 1 with frame 0 as history.
5. Reset the same episode-state object and repeat the first compressed query.

Exactly 56 offline model calls. Dense and reset identity must match head inputs,
normalized actions and unnormalized actions within the unchanged 1e-6 tolerance.
All calls must execute 32 original SDPA layers. Prior selection indices and
confidence must persist within the two-query sequence and reset across episodes.
These are implementation gates; compressed actions need not equal dense actions.

Then run dense/controller-off and compressed/controller-on simulator episodes,
in that order, on the first frozen forty-task development condition:
`pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`,
LIBERO-Spatial, initial state 0, seed 7. Match the initial-state SHA-256 in the
configuration. Preserve the original 220-step horizon and ten settling actions.
Dense must match the prior native reference: success, 78 executed steps, 10
queries. Compressed task success is recorded but is NOT an integration pass gate.
Record actual replans, discarded actions and model calls. Never select a new
task or alter pruning in response to this run's outcomes.

## Limits and execution

One selected TITAN GPU 0, UUID frozen in the configuration. Require fresh
aggregate idle telemetry immediately before launch. At most 512 model calls,
two complete episodes, 1,800 seconds, aggregate GPU memory strictly below
23,552 MiB, and artifacts below 256 MiB. TensorFlow GPU use is disabled, caches
are project-local, model loading is offline and metadata mutation is disabled.
Runtime/checkpoint, prior source archive and old results remain untouched.

The launch preflight verifies the original qualification and all new source
hashes, plus availability of both fixed frames. Forty-eight targeted comparator
CPU tests passed, including eight new corrupt-fixture reconciliation tests.
The prior 65-test episode/core/original-runtime suite passed in the preceding
implementation step. The new tests do not replace real-model qualification.

While running, inspect only owned process health, count-only progress, bytes,
elapsed time and selected aggregate GPU telemetry. Read outcomes only after
immutable completion. If execution fails, preserve the technical stop and log;
never automatically retry. Independently verify counts, artifact hashes and
all gates before accepting completion. Sync evidence/status locally and stop
before launching the matched development benchmark.

No latency claim follows from this instrumented check. The native attention
observer and head capture deliberately remain enabled for validation. No positive
research claim follows from one comparator task, even if both arms succeed.
