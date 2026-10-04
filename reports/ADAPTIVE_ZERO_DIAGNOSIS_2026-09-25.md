# Completed technical diagnosis and backend-aligned recovery

User authorized resolving qualification, running the four-method comparison,
then one justified, prospectively specified confirmation. No positive guarantee.

## Verified diagnosis

`results/adaptive-zero-diagnosis-v01`: complete, 16 inputs, 64 model calls,
zero episodes, 141.859 s, peak GPU0 memory 16,271 MiB. All three artifact hashes
verified locally. No retry. Full source identities recorded in launch.json.

| Comparison | Hidden max abs | Normalized action max abs | Executed command max abs | Exact rows |
|---|---:|---:|---:|---:|
| Native repeat | 0 | 0 | 0 | 16/16 |
| Adaptive-zero vs native SDPA | 1.1875 | 0.09765625 | 0.010076046 | 0/16 |
| Reference eager vs native SDPA | 1.1875 | 0.09765625 | 0.010076046 | 0/16 |
| Adaptive-zero vs reference eager | 0 | 0 | 0 | 16/16 |

Gripper disagreements: zero across all 128 saved action entries for each
comparison. Therefore the earlier gripper-boundary explanation was not supported
on these inputs. The mismatch is reproduced by the attention-backend change
alone. This is NOT evidence that the executed continuous differences are harmless.
The old 1e-3 bound cannot honestly accommodate this eager implementation.

Diagnostics SHA256:
`a0eb97cbea0657c3926d08ed274cba4d3966997e312344b5eac7000d71d845d0`.
Launch SHA256:
`84569a8a407f5f60fe8135ab84085db7d79480005dad5995e1abb523f4aefac2`.

## Recovery, not tolerance relaxation

New `adaptive_sdpa.py` keeps native SDPA outputs and extracts probabilities at
the two layers actually consumed by the selection algorithm. Shared original
runtime/reference files remain untouched. Selection rules remain the existing
pinned translation; this is a backend-aligned adaptation, not exact reproduction
of upstream speed/behavior. All extraction/selection overhead is timed.

Exact executed-action parity is reinstated. The maximum numerical tolerance is
unchanged. Worker and reconciliation both require 32 decoder SDPA calls. Fresh
v03 result paths and v3 config filenames prevent overwrite. The earlier v01/v02
failure metadata/logs are now copied locally. Local and server pre-recovery source
archives preserve the differing old code/config states. No GitHub push.

CPU test run on pinned TITAN runtime: 35 discovered, 27 passed, eight skipped
(upstream source fixtures unavailable on TITAN). Includes actual 32-layer
zero/startup native parity and fourth-query pruning at both budgets, not only
record fixtures. Re-run after final cleanup before freezing/launch.

Next: freeze hashes, authenticated CPU preflight, one v03 S0 qualification.
Only after completed independent reconciliation may the authorized S1 comparison
run. No automatic retry on technical failure. See protocol V2 and recovery plan.
