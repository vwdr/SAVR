# Backend-aligned qualification v03 — verification

Completed on 2026-09-26. This is an implementation qualification, not a robot
performance result or a positive-method finding.

## Evidence

- Output: `results/adaptive-qualification-v03`.
- Configuration SHA-256:
  `6f3a36decbb18706aa6cf39f2a727eb64601fd53de8efa19b334bf87ea10fc63`.
- Immutable worker-summary SHA-256:
  `55af20761ff4bbba9a30229b53bee50db6a9fedffbc7f21eb81145a098832948`.
- 176 model calls, 16 observation/frame checks, zero simulator episodes.
- Worker elapsed time: 309.974637 seconds. Peak aggregate GPU-0 memory:
  15,297 MiB. Both are within frozen caps.
- Exact zero-pruning executed-command equality: 16/16. Maximum zero-pruning
  hidden, normalized-action and raw-action difference: **0**. The numerical
  bound remains 1e-6, below the unchanged 1e-3 ceiling.
- Both budgets passed eight steady-state checks each: 384 and 256 visual
  tokens, with 512 tokens for the first three queries. All 32 layers retained
  the native SDPA output path. Layer lengths, original positions, no-cache
  behavior and audit-hook neutrality passed.
- Pruning-induced maximum difference (across recorded representation fields):
  6.8125. This is not a numerical-parity error or a success measurement and
  does not enter the parity tolerance.

## Independent verification

A separate local standard-library calculation, without importing the worker or
analyzer, verified all six artifact hashes, nine frozen source hashes, config
identity, summary identity, stream/bundle equality, exact count/order, zero
errors and command matches, both budgets, every layer length, structural flags,
resource caps and the separately recorded pruning-shift maximum. All passed.
The remote frozen CPU analyzer also completed successfully (exit 0), reporting
`technical_qualification_passed=true`, 176 calls, zero observed parity error and
the same summary hash. S1 dispatch still requires its own CPU preflight.

## Interpretation

The earlier eager-versus-native attention mismatch is absent on these tested
inputs after preserving native attention outputs. This does not establish robot
success or equivalence to upstream VLA-Pruner. The next authorized stage is the
four-arm development screen, conditional on the CPU analyzer and preflight.
Old failed runs, diagnostic evidence and frozen v03 source/configuration are
preserved. No automatic retry, training, manuscript edit or GitHub push.
