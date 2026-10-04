# SpecPrune real-checkpoint qualification readiness

Date: 2026-09-11. Preflight passed on TITAN; this is not a result.

- Eight new fail-closed accounting/parity tests passed locally and on TITAN.
  All 27 source/port/qualification tests in the targeted TITAN run passed.
  The preceding full CPU port regression had 44 passing tests.
- Preflight authenticated 46 source/checkpoint/input files plus all eight
  consumed training observation sources. It did not load a model.
- Selected GPU 0 aggregate telemetry: TITAN RTX, 24,576 MiB total, 6 MiB used,
  0% utilization. The worker repeats this check immediately before loading.
- Frozen configuration SHA-256:
  `8adc9dde86d08f8d717c96e4017250229c71d7f5062ebe697cbb4c53eb75e168`.
- Worker SHA-256:
  `1dad144a4a5e88159ba0d7cf726fa464cee3da44dfbf3ddab2f9d1200816b2f3`.
- Port SHA-256:
  `edf4d8886915237fe5cc8c7a636e41d08bfcb81e450ae5f96a1d9ac508f63c17`.
- Exactly 40 calls, eight observations, no simulator or training. Error tolerance
  1e-6; 30-minute wall cap; aggregate memory below 23,552 MiB; artifacts below
  256 MiB. No retry. Output: `results/specprune-real-qualification-v01`.
- No installed code, checkpoint or historical evidence modified. All server
  operations restricted to `/home/ved/SAVR` through `ssh titan`.

Protocol: `docs/SPECPRUNE_REAL_CHECKPOINT_QUALIFICATION_V1.md`.
No claims about closed-loop performance or speed are authorized by this record.
