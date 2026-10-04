# Current-frame offline gripper diagnostic: erratum

Recorded 2026-09-21 UTC after the completed robot pilot. This is a post-hoc
reporting correction, not a change to frozen code, training, gates or evidence.

## Finding

The offline fitting diagnostic in `scripts/fit_current_frame_adapters.py`
(line 241 at audit) compares prediction and teacher gripper values using `> 0`.
Its independent saved-prediction verification repeats that formula. This does
not measure disagreement between the commands actually sent to the simulator.

The four LIBERO action-statistics masks in the frozen checkpoint have their
gripper entry set to false. The gripper remains on the [0,1] scale, whereas the
six motion dimensions are unnormalized. Official execution converts the gripper
using `-sign(2*g - 1)` after float32 conversion. Its boundary is 0.5, not zero;
an exact 0.5 value produces zero and must not be silently assigned to either side.

Sources inspected through `ssh titan`, entirely inside `/home/ved/SAVR`:

- `checkpoints/openvla-7b-oft-libero-four-suite/dataset_statistics.json`
- `_unnormalize_actions` in the checkpoint's `modeling_prismatic.py`
- `third_party/openvla-oft/experiments/robot/robot_utils.py`
- `third_party/openvla-oft/experiments/robot/libero/run_libero_eval.py`
- Local frozen fitting source and saved validation predictions.

## Corrected descriptive calculation

The read-only audit evaluated all 800 saved validation observations, eight
commands each, with float32 conversion and the actual three-valued execution
rule. Denominator: 6,400 commands per policy, compared with dense teacher commands.

| Policy | Disagreeing commands | Disagreement |
|---|---:|---:|
| Plain compression | 331 / 6400 | 5.171875% |
| Action-only correction | 330 / 6400 | 5.156250% |
| Visual correction | 330 / 6400 | 5.156250% |
| Shuffled-label visual control | 1963 / 6400 | 30.671875% |

The compressed predictions contained one exact 0.5 tie; teacher predictions
contained none. The calculation includes the tie as an actual zero command.
These are descriptive offline agreement counts, not independent robot trials.

The previously reported 6.71875% (compression) and 6.81250% (visual) values are
sign-at-zero diagnostic values. They must not be interpreted as executed-gripper
error. The earlier interpretation that visual correction worsened executed
gripper agreement is withdrawn. The corrected difference is only one command
out of 6,400 and is not evidence of meaningful robot-performance improvement.

## What this does and does not change

This diagnostic did not enter the L1 optimization loss, checkpoint selection,
offline eligibility inequalities, robot action conversion or terminal-success
analysis. The completed robot results remain dense 120/120, compression 117/120,
action-only 118/120 and visual 117/120. The frozen visual-adapter positive-signal
criteria still fail. No robot rerun is needed to correct this reporting metric.

The audit did not identify an execution defect explaining the mixed robot
outcomes. That is not proof that all defects are absent. Demonstration-only
training, average imitation loss, always-on correction and rollout variability
are plausible limitations, not established causal explanations. The existing
pilot lacks full corrected-policy per-step action/image traces, so it cannot
resolve the exact mechanism of each failed episode retrospectively.

## Preservation and future use

Keep authenticated fitting sources, summaries, predictions, weights and analyses
unchanged. Read `CURRENT_FRAME_ADAPTER_FIT_RESULT_V1.md` and older status/decision
entries together with this erratum. Future metric implementations should use a
new version, reproduce official execution semantics including float32 ties,
and include boundary tests before any new freeze. Do not patch the frozen fitting
source in place because downstream provenance authenticates it.

The next research decision remains open. A useful efficiency–accuracy tradeoff
can be a legitimate contribution without winning every metric, but the actual
tradeoff, originality, uncertainty and limitations must be supported. Potential
future success is a hypothesis, not a measured result.
