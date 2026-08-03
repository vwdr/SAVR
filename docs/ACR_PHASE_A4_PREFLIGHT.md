# ACR Phase A4 Preflight

**Authorization:** user approved Phase A4 on 2026-08-02.

**Outcome access at freeze:** none.

## Frozen scope

- Run ID: `acr-a4-upstream-fr-object-dev00-09-v01`
- Oracle: unmodified upstream two-view Full Refresh
- Population: LIBERO-Object tasks `0-9`, initial states `0-9`, seed `0`
- Planned attempts: exactly 100, one per task/state pairing
- ACR rollouts: zero
- Protected populations: Goal, LIBERO-10, all states `10-49`, and reserve seeds remain untouched

The runner records a deterministic `32 x 32` scene representation, normalized
end-effector position, complete action chunk in a hashed compact companion
trace, transition summary, component counts, synchronized timings, and exact
provenance for every query. The frozen query schema's per-camera fields count
logical camera-block work; the companion trace separately records the actual
single upstream backbone/projector module invocations. Raw images and videos
are disabled.

## Hard resources

- one explicitly selected GPU and one model process;
- 28,800 seconds total phase wall time;
- 100 attempted episodes, including any technical attempt;
- 1,073,741,824 result bytes;
- no downloads and no writes outside `/home/ved/SAVR`.

There is no automatic episode retry. A technical or invariant failure is
preserved and stops the phase because a replacement would exceed the exact
100-attempt matrix. A scientific task failure is terminal and is not rerun.
CPU-only analysis recovery may write a new monotonically named analysis
attempt after preserving an interrupted attempt; it cannot rerun an episode or
change derivation semantics.

## Mechanical gate

Before candidate derivation, the run must contain exactly 100 reconciled
terminal episodes, zero technical/accounting failures, at least 90 successes,
and at least 8 successes for every task. Failure stops A4 without an ACR
rollout.

If the FR gate passes, the committed analyzer reconstructs the trace twice
independently and derives exactly the three Protocol V1 templates on the frozen
quantile grid. Both serialized derivations must be byte-identical. No template,
threshold, tie-break, horizon, or reuse cap may be added or changed.

## Publication boundary

The configuration, runner, analyzer, tests, and this preflight must merge and
be synchronized to TITAN before the first Object episode. The runner revision,
configuration hash, checkpoint identity, GPU identity, and schema hashes are
then fixed in the immutable run manifest.
