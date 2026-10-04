# Current-frame feature collection v1: dataset verification

Verified 2026-09-18 UTC. Worker completed 2026-09-16 01:23:11.633113 UTC.

## Result and scope

Dataset generation and the frozen CPU analysis passed. Independent server-side
and local reconciliation passed. The full local evidence transfer is complete
and all 4,005 summary-listed artifact hashes match. This is a data-preparation
milestone, not a trained-method result or evidence of improved robot success.

| Frozen requirement | Observed evidence |
| --- | --- |
| Exactly 4,000 records | 4,000, with unique ordered sample identities |
| Exactly 8,000 model calls | 8,000, two per observation |
| Role counts | 3,200 fit; 800 validation |
| Task balance | 40 tasks, 80 fitting / 20 validation observations each |
| No trajectory leakage | 1,054 fitting and 271 validation trajectories, disjoint |
| No fitting or simulator episodes | 0 optimizer steps; 0 episodes |
| Immutable source/sample/artifact contracts | Frozen analyzer passed; independent verifier checked 4,005 hashes |
| Elapsed time below 8 hours | 9,215.599 seconds (about 2 h 34 min) |
| Aggregate GPU memory below 23,552 MiB | Peak 15,295 MiB on GPU 0 |
| Artifacts below 20 GiB | 17,092,424,025 bytes before summary |
| Initial/free-space guards | Worker completed without a guard stop; local transfer began with 73,570,216 KiB free |
| No retry | Single completed collection, automatic retry disabled |

Both roles derive exclusively from the historical training split. Role membership
must come from the frozen manifest, not simply from the `split=train` field.
No calibration or locked observations were used. Teacher outputs are normalized
dense-model actions, not expert demonstrations. Features use the fixed 384-token
student and identical current inputs. No temporal cache was introduced.

## Evidence identities

- Run: `results/current-frame-feature-collection-v01`.
- Config SHA-256: `e810cb53f7a836d7a4d9dbb95ee406e919355f70a215e6788d5b760b7863a124`.
- Input manifest SHA-256: `2fd13ced3e5a9a2b810eb299a5c1667f8c1de3ee2c5487a4a0683e61847d9ae8`.
- Worker summary SHA-256: `7bde8cd59846d5f5b54c7b9a8302b30ebe74ef13606eb92aeb754465d5b1a8c3`.
- Analysis SHA-256: `ae9ba225dce1d14318b8b7d6c5493657eec307a6142cab2df6f48e27e7eb2273`.
- Technical log: `reports/current-frame-feature-collection-v01-terminal.log`.

The frozen analyzer ran in the original project environment with CUDA hidden,
bytecode writing disabled and one OpenBLAS thread:

```sh
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
envs/openvla-oft/bin/python scripts/collect_current_frame_features.py \
--analyze results/current-frame-feature-collection-v01
```

The independent copy checker is `scripts/verify_current_frame_collection_copy.py`.
Its first execution on TITAN verified the server source; an initial output key
was named `local_copy_verified` even there. The key has been corrected to
`copy_verified` with an explicit `verified_root` to avoid confusing source
verification with destination verification. No frozen experiment code was changed.

## Completed transfer and next checkpoint

Local rsync PID 72647 (terminal session 16277) exited successfully. The complete
run is at `/Users/veddwivedi/Documents/VLA/SAVR/results/current-frame-feature-collection-v01`.
No delete option was used. The completed analysis and technical log are also
local. On 2026-09-18, the independent local copy checker passed every one of the
4,005 listed hashes, exact samples, 3,200/800 roles, 40-task balance, disjoint
1,054/271 trajectory counts, 8,000-call accounting and resource limits. Both
summary and analysis hashes match the authenticated server source. Local free
space after transfer was 57,066,452 KiB. No action errors or losses were examined.
The collection checkpoint is complete; retire its heartbeat after final status sync.

Stop before research fitting. Next freeze the matched optimizer, training budget,
shuffle/seed, shuffled-label control and checkpoint selection rule under
`docs/CURRENT_FRAME_LEARNING_PILOT_V1.md`. Offline data or training loss alone
cannot establish incremental closed-loop benefit over compression alone.

All remote work stayed inside `/home/ved/SAVR`. No new GPU job, unrelated server
operation, retry, GitHub push or change to frozen scientific gates was performed.
