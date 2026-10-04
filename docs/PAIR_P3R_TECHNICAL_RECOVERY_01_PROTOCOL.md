# PAIR P3R Technical Recovery 01 Protocol

## Authorization and scope

This protocol authorizes exactly one recovery of the zero-query technical stop
`pair-p3r-vectorized-v01`. It does not authorize P4, a scientific redesign, a
second recovery, or any change to the P3R frontier or decision gates.

## Authenticated parent evidence

- Parent P3 analysis:
  `results/pair-p3-physical-v03-recovery01/analysis.json`
- Failed P3R stop record:
  `results/pair-p3r-vectorized-v01/technical_stop.json`
- Failed-stop SHA-256:
  `2d4b34e3171c2190cbd881f2aace3c94c2e014cb7c6e6deed4a9136fca85743b`
- Failure usage: 0 model queries, 0 timed blocks, no model load, no protected
  outcomes, and no simulator.

## Sole technical corrections

1. Launch with the existing authenticated runtime
   `/home/ved/SAVR/envs/vla-cache-compat/bin/python`, which contains the pinned
   compatibility dependencies.
2. Resolve the output directory to an absolute path below `/home/ved/SAVR`
   before the worker changes its working directory.
3. Require CUDA-hidden preflight to import the pinned OpenVLA/LIBERO evaluation
   module successfully in that same runtime without loading the model.

No dependency installation, download, checkpoint change, source revision change,
or deletion of the failed-run evidence is permitted.

## Frozen scientific design

The recovery preserves P3R V1 exactly:

- profiles: `D59_BAL_PT1`, `D62_BAL_PT1`;
- horizons: 2 and 4;
- six paired repetitions per profile/horizon;
- exact legacy/vectorized equivalence controls;
- 210 planned model queries and a hard cap of 220;
- randomized paired arm order and no outlier deletion;
- raw saving lower bound greater than zero;
- net-saving lower bound at least 10%;
- total overhead at most 2%;
- service rate at least 70%;
- strict selected-GPU memory below 23,552 MiB;
- one GPU, one process, two-hour wall cap, 1 GiB artifact cap;
- no expert actions, action comparisons, simulator outcomes, downloads, or
  automatic retry.

## Execution boundary

- New immutable run identifier: `pair-p3r-vectorized-v02-recovery01`.
- Use any idle GPU selected only from aggregate telemetry and record its ID.
- Preflight and result directories must be absent before execution.
- A technical failure stops the phase without another retry.
- On 210/210 completion, run the frozen P3R analyzer exactly once, reconcile all
  gates, preserve and synchronize the evidence, and stop before P4.
