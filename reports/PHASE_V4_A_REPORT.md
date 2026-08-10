# ACR Version 4 Phase V4-A Report

Status: **STOPPED NEGATIVE; V4-B INELIGIBLE**

Date: 2026-08-10

## Outcome

The frozen CPU-only diagnosis completed after two preserved fail-closed
technical recoveries. All six predeclared candidates were replayed twice with
byte-identical outputs, all input integrity checks passed, and no candidate
passed every controller and complete-method gate. The mechanical disposition
is `STOP_BEFORE_V4_B`; no controller or executor was selected.

This is an offline replay and source-feasibility result. It is not a new
closed-loop task-success or model-latency result.

## Candidate gate results

| Candidate | Reuse (95% lower) | Predicted visual reduction (95% lower) | Max streak | Required reuse wall ratio | Result |
|---|---:|---:|---:|---:|---|
| `v4-a050-h2-b40-g` | 35.25% (33.98%) | 11.97% (11.51%) | 2 | 0.9162 | Ineligible |
| `v4-a050-h2-b40-gr` | 26.34% (25.04%) | 8.83% (8.36%) | 2 | 0.8818 | Ineligible |
| `v4-a075-h2-b40-g` | 35.70% (34.46%) | 12.13% (11.67%) | 2 | 0.9176 | Ineligible |
| `v4-a075-h2-b40-gr` | 28.48% (27.15%) | 9.58% (9.10%) | 2 | 0.8920 | Ineligible |
| `v4-a100-h2-b40-g` | 35.76% (34.48%) | 12.15% (11.68%) | 2 | 0.9176 | Ineligible |
| `v4-a100-h2-b40-gr` | 29.44% (28.00%) | 9.92% (9.42%) | 2 | 0.8956 | Ineligible |

The closest candidate, `v4-a100-h2-b40-g`, met the reuse point/lower-bound,
predicted visual-reduction, gripper-transition, and source-feasibility gates.
It failed the frozen maximum-reuse-streak gate because it produced streaks of
two rather than exactly one. Its current executor predicts a 0.9983 weighted
wall ratio versus BFR, short of the 0.98 complete-method target; a future
executor would need a per-reuse wall ratio no worse than 0.9176 to make that
target under the frozen conservative calculation.

## Diagnosis

1. The preflight's prose described horizon 2 as preventing consecutive reuse,
   but the implemented controller semantics permit two consecutive reuse
   decisions before the horizon refresh. The machine-frozen maximum-streak-one
   gate controls, so every candidate fails. This contradiction is preserved;
   the horizon or gate was not changed after outputs were observed.
2. Adding a translation-direction-reversal veto reduces the measured
   transition exposure, but all three such candidates fall below the 35% reuse
   and 12% predicted visual-reduction targets. It is too conservative for the
   frozen efficiency objective.
3. Existing V3-D timing confirms that the current reuse path does not supply
   the required wall margin. Source inspection supports a testable
   project-owned static-buffer GPU core, but not a performance, capture-safety,
   memory, or numerical-equivalence claim.
4. Only 250 of 1,204 paired BFR/V3 query indices had matching action hashes;
   none of 314 V3 reuse indices or 295 post-reuse indices matched. Because the
   policies generated different closed-loop states, this is descriptive
   divergence and not causal proof that reuse caused a particular failure.

## Integrity and resources

- Machine result: `reports/runtime/acr_v4_a.json`
- File SHA-256: `6bb940cf6d68e64a33854fc98fffb39edc7158a78daecac622d076abe7aae012`
- Semantic SHA-256: `e7749e524ea39674a31654204dc879002b129fb8dfef6d89e66e89a38a22ffd8`
- V3-D input records SHA-256: `85f2c648ce06a40557d34bdc51e55e1f4dc59a8021d94e30ddfbc8a2bc11ed1c`
- A4 trace SHA-256: `577d683be265af7919deeb58cbbb895c9b6b1975b093c845b39e34a353fb0d69`
- GPU/model queries/simulator episodes/downloads: `0/0/0/0`
- Protected outcomes opened: `0`
- No manuscript change was made.

The first invocation stopped before candidate computation because completion
metadata intentionally lacked a semantic hash. Recovery 1 stopped after
volatile computation but before result construction because legacy A5 query
records lacked semantic hashes. Both stops and their narrowly scoped fixes
were committed before the next invocation. Recovery 2 delegated A5 integrity
to the original committed analyzer and produced the immutable result above.

## Decision

Apply the predeclared negative stop. Do not implement V4-B, silently change
horizon 2 to horizon 1, relax the streak gate, add a candidate, or reinterpret
offline replay as a positive result. Any further route must begin as a new,
independently predeclared method/protocol and must preserve this V4-A result.
