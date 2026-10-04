# Backend-aligned adaptive screen — frozen recovery V2, 2026-09-25

Supersedes V1 for the new v03 qualification and screen only. Failed v01/v02
artifacts and old configs are preserved. User authorized diagnosis, comparison
and one justified confirmation on 2026-09-25. No automatic retry or fallback.

## Diagnosis and implementation choice

The completed 64-call diagnostic (16 consumed inputs, zero episodes) found exact
native repeatability and exact adaptive-zero/reference-eager equality on all
inputs. Eager/native differences reached 1.1875 in action hidden states and
0.09765625 in normalized actions; executed continuous commands differed by up
to 0.010076046, with zero gripper disagreements in these 128 actions. This does
not prove backend differences harmless in closed loop. The existing numerical
ceiling is NOT enlarged and exact command equality is restored as a hard gate.

The adaptive comparator now preserves the native SDPA output at EVERY layer.
It separately computes attention probabilities only at layer 3 (selection) and
layer 15 (history) using the existing NativeAttentionScores implementation.
Selection arithmetic, two camera budgets, absolute positions, history w=3 and
decay=0.8, first-three-dense startup, checkpoint and action head are unchanged.
All extraction, CPU history copies and selection overhead remain timed.

This is an explicitly **backend-aligned local VLA-Pruner selection adaptation**,
NOT an exact upstream timing or numerical reproduction, and not a new pruning
algorithm. We do not claim a favorable comparison refutes or beats the published
method. Score arithmetic parity and native-output parity are separate checks.
Changing the numerical backend can change token rankings; report this limitation.

## Required tests and qualification

Execute real pinned 32-layer tiny-model CPU tests: zero and startup outputs
must equal native SDPA exactly, both adaptive budgets must actually prune on
query four, history must reset, and nested SDPA wrappers must restore correctly.
Retain the source-selection tests (report skips honestly). Reference modules
and prior experiment implementations remain unchanged.

S0 v03: same 16 inputs, 176 exact calls, zero episodes; cap 256 calls, 3600 s,
23,552 MiB GPU-0 aggregate memory, 256 MiB artifacts. Every adaptive decoder
must produce 32 observed SDPA calls, scoped to the decoder, not the vision
encoder. Audited and unaudited outputs must match. Zero-pruning executed
commands must match native exactly. The original <=1e-3 numerical ceiling is
unchanged; its measured bound uses ONLY zero-pruning rows. Pruning-induced
differences are recorded separately. Post-prune token lengths include retained
visual AND all nonvisual tokens. Analyze S0 CPU-only before launching S1.

## Four-method development screen

S1 v03: dense512, fixed384, adaptive384, adaptive256. Same frozen 40 previously
consumed state-0 conditions, seed7, balanced arm rotation; 160 primary plus eight
native controls =168 episodes. Same 240 timing slots (16 hardware warmup excluded,
224 measured). Fixed compression starts at query one; adaptive startup lasts
three queries and is included in its all-query mean. Native shadows require exact
zero-pruning commands and the S0-pinned numerical bound, with native SDPA witnesses.

The controlled timing corpus repeats two saved frames into seven-query traces.
It is a microbenchmark, not representative natural-trajectory timing. Report warm,
steady and all-query measurements, plus observed episode costs; do not use only
the most favorable timing regime. Limits remain 7000 calls, six hours, 168 episodes,
23,552 MiB aggregate memory, 512 MiB artifacts, GPU0 only after fresh idle/UUID check.

Freeze source/configuration hashes before launch. Hide outcomes until completed
168 episode and 240 timing records plus immutable summary. Reconcile all hashes,
counts, caps, four arms and controls before CPU analysis. On a technical stop,
preserve technical evidence and stop without retry. Never convert errors to task
failures or weaken scientific thresholds after observing outcomes.

## Confirmation decision

No automatic arm selection or positive claim. A useful development tradeoff may
justify ONE separate confirmation study; lack of one does not justify repeated
screens. Before confirmation, fix the claim, precision-based sample size, independent
evaluation conditions, seeds, relevant comparator(s), paired uncertainty analysis,
resource cap and stopping rule. Describe exposed tasks/states accurately. The
existing user authorization covers pursuing this plan, not guaranteeing a positive
result or writing unsupported manuscript claims. No GitHub push or advisor email.
