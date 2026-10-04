# CAC C1H Recovery 02 Readiness Audit

Date: 2026-08-31  
Run identity: `cac-c1h-headroom-v03-recovery02`  
Verdict: authorized and ready for one bounded attempt

## Exact recovery scope

Recovery 02 addresses only the pre-episode integration stop recorded in
`cac-c1h-headroom-v02-recovery01`. It does not change the frozen population,
paired schedule, D62 substrate, terminal-success measurement, Gate H, or the
stop-before-C2 boundary.

The source stop is authenticated by complete file and semantic SHA-256 values:
0 episode attempts, 1 pre-episode model query, 0 progress records, and no
partial terminal outcomes.

## Corrected contracts

The pinned VLA-Cache action helper requires `prev_images` even with caching
disabled, returns `(actions, cache, images, metrics)`, and normalizes the input
state in place. Recovery 02 therefore:

1. clones the standard prepared observation so upstream mutation cannot alter
   the custom-helper comparison input;
2. supplies independent copies of the current scene and wrist images as the
   first-query `prev_images` value;
3. validates and unpacks the complete four-item return before using actions;
4. releases the official cache before the custom helper is invoked;
5. runs exact official-versus-custom dense parity plus D62 anchor and reuse
   controls before the first terminal episode; and
6. persists a traceback inside the immutable result root for any future
   technical stop.

The controls now use exactly 4 model calls. Exact worst-case accounting is
9,964 calls for Stage 1 and 19,924 calls cumulatively, below the unchanged
20,000-call cap.

## Verification

- 39 focused C1H/C1/P3 checks pass on TITAN.
- The complete suite records 514 passes plus 9 subtest passes.
- Its sole failure is the already-documented unrelated ACR-V10 preflight test,
  whose historical pre-attempt assertion is invalid after V10 completed.
- Worker and analyzer compile; the bounded launcher passes shell validation.
- The new observation-cloning and four-item return contracts have direct unit
  tests, including protection against in-place state mutation.
- The exact real-model dense and D62 paths are placed before all terminal
  episodes and fail closed if any contract, action shape, finiteness, memory,
  or parity check fails.

## Authorization and boundary

The user explicitly approved Recovery 02. One attempt is permitted; automatic
retry is false. Partial outcomes stay sealed until an exact stage completes.
C2, locked state IDs 10--49, training, downloads, and unrelated TITAN access
remain prohibited.

Frozen configuration semantic SHA-256:
`38a4e5cb5f61d91c594ba082805213fd5fde26fa84e11624aafddc18321b035b`.
