# Phase 6S-D SAVR3 Validation Report

Status: STOPPED NEGATIVE — PREDECLARED POSITIVE GATE NOT MET

Date: 2026-07-31

## Frozen evaluation

- Policy: `SAVR3`
- Configuration: `savr3-rv-w375-b15`
- Suite: LIBERO-Spatial tasks `0-9`
- Initial states: `3-9`
- Seed: `0`
- Fixed episodes: `70`
- Selected GPU: physical ID `0`, UUID
  `GPU-bb2451d6-2989-a112-5c18-8892943710e4`

The configuration and gates were frozen in `docs/PHASE6S_PROTOCOL_V1.md`
before SAVR3 implementation or outcome collection. No threshold, rule,
pairing, margin, or episode was changed or rerun.

## Result

| Measure | Result | Frozen gate | Pass |
|---|---:|---:|---|
| Terminal episodes | 70/70 | 70/70 | yes |
| Successful episodes | 69/70 | 70/70 | no |
| Per-task success | tasks 0-8: 7/7; task 9: 6/7 | every task 7/7 | no |
| Queries | 944 | — | — |
| Visual refreshes | 935 | — | — |
| Visual reuses | 9 | — | — |
| Skip rate | 0.9534% | at least 5% | no |
| Technical failures | 0 | 0 | yes |
| Invariant errors | 0 | 0 | yes |
| Exact visual-call reduction | 9 backbone and 9 projector calls skipped | one each per reuse | yes |

The only unsuccessful episode was
`savr3-rv-w375-b15_task_09_state_04`. It reached the 220-step horizon with
one reuse among 28 policy queries. It completed without a runtime or
instrumentation error. All nine episodes containing a reuse are preserved;
eight succeeded and this one did not.

The independent validator returns `positive_method_result = false`. SAVR3
therefore does not support the predeclared positive method claim. It preserved
high success but was too conservative to provide meaningful visual-compute
savings, and it also missed the exact success gate.

## Integrity and resource audit

- Every reuse invoked zero vision-backbone and projector calls.
- All 944 language-model and action-head paths executed once per query.
- The run used one model process and one GPU.
- Elapsed GPU-run time: `3784.63` seconds (about `1.05` hours).
- Result artifacts: `170,580,474` bytes, below the 1 GiB cap.
- Peak allocated GPU memory: `16,090,075,136` bytes.
- Protected checkpoint files were restored to their exact pre-run hashes.
- No unexpected checkpoint file remained.
- GPU 0 returned to 6 MiB aggregate use and 0% utilization.
- No final-holdout outcome was accessed.

## Evidence hashes

- Run summary SHA-256:
  `bb54eef3620f136c6fcd121b573f58641f4677d92ff02e31ba2f33aba13ad046`
- Manifest SHA-256:
  `2703dd2939fbd75b5a4ddc08625c2c79451336515abb7098e83ca64cf5553234`
- Analysis file SHA-256:
  `de570a1b79c7e7e50bf5193f5bf2d2f7048c2336abf10c0dd0b460db51f3e789`
- Embedded semantic analysis SHA-256:
  `a9099897c667a654322f708ef91a5589cd7ffe410d847eb0c699aba6acf27b9b`

## Frozen stop

Do not tune or rerun SAVR3, search another local wrist threshold, run matched
baselines, advance to Phase 6S-E, or inspect the final holdout. A further
attempt would require a materially different method and new independent
development evidence; the current states `0-9` can no longer be described as
fresh evidence for another post-hoc SAVR variant.
