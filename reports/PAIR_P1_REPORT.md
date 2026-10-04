# PAIR-VLA Phase P1 Report

Date: 2026-08-28  
Phase: P1 — data acquisition and semantic alignment  
Outcome access: demonstration metadata only  
GPU use: none  
Decision: **PASS P1; STOP before P2**

## Result

The original sequential LIBERO demonstrations provide a valid, reproducible
path for PAIR-VLA's eight-action expert labels. All four approved suites were
downloaded into `/home/ved/SAVR` at the frozen Hugging Face revision. Every
source file matches the published byte count and LFS SHA-256.

The CPU-only audit covered all 40 tasks, 2,000 complete trajectories, 338,575
original environment actions, and 41,447 eligible eight-action query records.
It produced 39,447 adjacent query pairs separated by exactly eight original
environment actions.

## Authenticated source

| Field | Value |
|---|---|
| Repository | `yifengzhu-hf/LIBERO-datasets` |
| Revision | `f13aa24a3da8c43c7225569f28c562979fa0e35a` |
| License | CC BY 4.0, as declared by the official LIBERO project |
| Suites | `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` |
| HDF5 files | 40 |
| Source bytes | 33,784,856,577 |
| Local/hash mismatches | 0 |

The source is the original sequential HDF5 release. The filtered
`modified_libero_rlds/no_noops` dataset was not used and no no-op filtering was
applied.

## Semantic findings

Each task file contains 50 demonstrations. Every trajectory has:

- primary `agentview_rgb` and wrist `eye_in_hand_rgb` images;
- aligned 7-D actions;
- 6-D end-effector state and 2-D gripper state, forming the required 8-D
  proprioception;
- joint, robot, and simulator states;
- a single final terminal marker;
- original ordered array position as the environment-step identity;
- task instruction and environment metadata; and
- source-file and trajectory hashes.

The source has no timestamps. Exact discrete chronology is nevertheless
authenticated by the original per-trajectory array order: observation/action
index `t` precedes action `t`, and successive deployed queries use `t` and
`t+8`. The index stores those original integer positions; it never substitutes
adjacent filtered frames.

The official OpenVLA-OFT data path was executed from revision
`e4287e94541f459edc4feabc4e181f537cd569a8` with:

1. `libero_dataset_transform` for the gripper convention;
2. checkpoint suite-specific q01/q99 statistics and
   `normalize_action_and_proprio`; and
3. `chunk_act_obs(window_size=1, future_action_window_size=7)`.

An independent float32 NumPy reconstruction was evaluated for every
trajectory. Maximum absolute differences for actions, proprioception, and
eight-action chunks were all exactly `0.0`, within the frozen `1e-6` absolute
tolerance.

Official chunking drops starts with fewer than eight remaining actions. P1
therefore emits only complete 8x7 primary chunks with all 56 validity entries
true; incomplete terminal starts are excluded rather than padded or silently
repeated. The terminal remainder class (0 through 7 original actions) is
preserved. All eight classes occur in the data and were audited with both
cameras.

## Split and leakage result

Complete trajectories were hash-sorted independently inside every task using
seed `20260828` and the frozen 70/15/15 rule. With 50 demonstrations per task,
the deterministic allocation is 35 training, 8 calibration, and 7 locked
test.

| Split | Trajectories |
|---|---:|
| Training | 1,400 |
| Calibration | 320 |
| Locked test | 280 |
| Total | 2,000 |

Every query inherits its parent trajectory split. Duplicate trajectory IDs,
cross-split trajectories, and query inheritance failures are all zero.

## Storage and resource reconciliation

| Item | Actual bytes | Approved cap |
|---|---:|---:|
| Original data root, including transfer metadata | 33,784,910,717 | — |
| Index and P1 audit root | 53,556,485 | — |
| Combined primary P1 data/index | 33,838,467,202 | 53,687,091,200 |
| Source network payload | 33,784,856,577 | 36,507,222,016 |

Remaining project-filesystem capacity after P1 was 355,776,028,672 bytes.
TensorFlow reported zero visible GPUs, `CUDA_VISIBLE_DEVICES` was empty, and no
GPU identity, allocation, process, or outcome was used.

## Gate adjudication

| Gate | Evidence | Decision |
|---|---|---|
| Exactly eight original actions between q and q+1 | 39,447 indexed pairs; every delta equals 8 | PASS |
| No filtered-frame shortcut | Original HDF5 revision; no filtering applied | PASS |
| Expert chunks match official preprocessing | All-transform maximum absolute difference 0.0 | PASS |
| Both cameras and correct proprioception | Validated on all 2,000 trajectories; sample content hashes for all 40 tasks | PASS |
| Trajectory-level split inheritance | 1,400/320/280, with zero leakage/inheritance failures | PASS |
| Hashes and counts reconcile | 40/40 files and 33,784,856,577/33,784,856,577 bytes; all SHA-256 values match | PASS |
| Storage within cap | 33,838,467,202 bytes versus 53,687,091,200-byte cap | PASS |

## Artifacts

Repository evidence:

- `configs/pair/p1_data_v1.json`
- `scripts/audit_pair_p1_data.py`
- `reports/pair_p1/source_manifest.json`
- `reports/pair_p1/split_manifest.json`
- `reports/pair_p1/audit_report.json`
- `reports/pair_p1/artifact_manifest.json`
- `reports/pair_p1/run_summary.json`

Large project-local indexes retained only on TITAN:

- `data/pair/index/f13aa24a3da8c43c7225569f28c562979fa0e35a/trajectory_index.jsonl`
- `data/pair/index/f13aa24a3da8c43c7225569f28c562979fa0e35a/query_index.jsonl`

Their authenticated hashes are in `artifact_manifest.json`; images are not
duplicated in the index.

## Boundary

P1 did not implement PAIR-VLA, open simulator outcomes, use a GPU, alter the
manuscript, remove existing files, commit, push, or start P2. P2 requires new
explicit approval and is limited to CPU/synthetic implementation correctness.
