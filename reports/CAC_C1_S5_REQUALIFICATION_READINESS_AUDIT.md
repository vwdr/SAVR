# CAC C1 S5 Requalification Readiness Audit

Date: 2026-09-01  
Decision: ready for one authorized GPU attempt; no GPU selected during package verification

## Conclusion

The corrected C1 package is ready. It preserves every historical C1 tensor,
chronology, isolation, streaming, timing, memory, and headroom control while
adding the independent official-policy boundary required by S5. The frozen
schedule is exactly 97 calls: 2 new official/custom parity calls plus all 95
historical C1 calls.

## Corrections relative to historical C1

- Uses the S3-qualified official loader guard and official OpenVLA-OFT source
  tree; no model-logic synchronization from the cache fork is allowed.
- Uses corrected cache/action selection with explicit compact absolute-position
  mapping and instruction-only semantic positions.
- Runs the released evaluator before C1 warmup, independently captures its
  exact 56x4096 action-head input and normalized output, and compares hidden,
  normalized, and unnormalized actions at `1e-6`.
- Verifies image-copy isolation and the documented normalized-state mutation.
- Uses corrected substrate identity `D62_BAL_PT1_S4C_V1`; historical profile
  values are reused only because S4 found 0/8 selection changes.
- Seals semantic technical-stop evidence and restores checkpoint metadata on
  any caught failure before writing the stop.

## Frozen controls retained

- 4 warmup calls;
- sidecar off/on and dense/all-fresh action/cache controls;
- four paired action-head hook controls;
- two deterministic recursive age-1--4 cycles with reset;
- clean branch cache/tracker/salience/RNG isolation;
- complete visual-plus-decoder timing at horizons 2 and 4;
- exact `fc2(Z_C)` reproduction and corrected feature extraction;
- zero-initialized 5,070,599-parameter adapter bypass;
- bounded 1,380,352-byte numeric record and instruction sidecar;
- 8% median h4 gross-headroom gate, 23,552 MiB memory gate, and 4 GiB
  artifact cap.

## Verification

- Configuration semantic hash:
  `c029aea2add0c12c0dbdcc6f8b684613243929fcc85db63e340c58cd24a25a23`.
- 79 focused CUDA-hidden TITAN tests passed.
- Authenticated file, S3 parent, S4 parent, loader-guard, source-data,
  checkpoint, schedule, runtime, storage, and output-absence checks passed.
- No stale checkpoint backup exists.
- No active `-57:-1` selector exists.
- No GPU was selected and no model call, simulator call, outcome access,
  training, download, or runtime write occurred during verification.

## Remaining bounded risks

The real run can still stop because of shared-GPU contention, an upstream
runtime failure, or a genuine C1 gate failure. These cannot be eliminated with
static tests. Aggregate telemetry is checked immediately before launch and
sampled during execution; all failures are fail-closed and cannot retry.

## Boundary

One S5/C1 GPU attempt is authorized. Stop before C1H regardless of outcome.
C1H, C2, simulator work, training, and automatic recovery remain unauthorized.
