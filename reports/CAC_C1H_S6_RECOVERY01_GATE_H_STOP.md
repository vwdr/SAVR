# CAC C1H S6 Recovery 01 Gate-H Result

**Date:** 2026-09-02  
**Run:** `cac-c1h-headroom-s6-v02-recovery01`  
**Classification:** valid scientific stop  
**Decision:** `STOP`; do not proceed to C2

## Result

Recovery 01 completed the exact frozen Stage-1 population: 120 paired
conditions and 240 terminal episodes, with one dense and one corrected-D62 arm
per condition. The worker completed without a technical stop and the independent
analyzer passed all 12 reconciliation gates.

| Arm | Successes | Episodes | Success rate |
|---|---:|---:|---:|
| Dense OpenVLA-OFT | 68 | 120 | 56.67% |
| `D62_BAL_PT1_S4C_V1` | 0 | 120 | 0.00% |

The dense-minus-D62 gap was 56.67 percentage points. Dense favored all four
suites:

| Suite | Dense | D62 |
|---|---:|---:|
| LIBERO Spatial | 14/30 (46.67%) | 0/30 (0.00%) |
| LIBERO Object | 27/30 (90.00%) | 0/30 (0.00%) |
| LIBERO Goal | 13/30 (43.33%) | 0/30 (0.00%) |
| LIBERO-10 | 14/30 (46.67%) | 0/30 (0.00%) |

## Gate H

Stage 1 independently triggered three frozen stop conditions:

- dense success was below 75%;
- D62 success was below 50%;
- the dense-minus-D62 gap exceeded 35 percentage points.

The prespecified extension was therefore not opened. Gate H returned `stop`,
and C2 has no eligible next phase and is not authorized.

## Validity and integrity

- mandatory outcome-free qualification: pass, exactly 4 calls, all official
  hidden/normalized/unnormalized parity errors 0.0;
- actual pre-episode official-boundary controls: pass, all parity errors 0.0;
- terminal records: exactly 240; progress records: exactly 240;
- model queries: 8,525, within the 20,000 cap;
- peak aggregate GPU-0 memory: 16,301 MiB, below 23,552 MiB;
- elapsed time: 25,796.42 seconds;
- outcomes sealed until exact Stage-1 completion: pass;
- checkpoint restoration: exact;
- GPU 0 after completion: 6 MiB, 0% utilization;
- extension opened: false; C2 started: false; automatic retry: false.

All analyzer gates passed: worker completeness, episode count, extension rule,
worker/analyzer decision agreement, C2 boundary, progress count, query cap,
memory cap, official dense parity, corrected official boundary, checkpoint
restoration, and outcome sealing.

## Interpretation

This is a scientific result, not a technical failure. The corrected D62
implementation retains its measured systems savings and exact semantic
qualification, but recursive closed-loop reuse collapsed task success across
every tested suite. A low-dimensional action corrector cannot be responsibly
trained as the planned solution on a substrate with 0/120 successes and a
56.67-point reliability deficit. Under the frozen CAC V2 protocol, the
D62-based positive-solution route ends here.

This result does not prove that all learned cache correction is impossible. It
shows that this exact corrected whole-prefix D62 substrate, horizon, and reset
policy is not a viable basis for C2 under the declared constraints.

## Evidence

- qualification summary SHA-256:
  `aeef5da66c36d2415ac68994fedb16e4885421538b751db0b6b962b293f7f4c6`
- Stage-1 schedule SHA-256:
  `94e3f17f10231ff3276335a967b500875f5fc224c464d328adf523ba73832252`
- terminal records SHA-256:
  `6d89a447cc68f892a37556a076c5069ad7c4da4fa0ab3e3505a98c4774c14bae`
- worker summary SHA-256:
  `d6112f13a9cf4942d73eb29a592df239f4f22482667cf9ae0a88248ab2f6c3b9`
- analysis SHA-256:
  `7944802c69eb977c04eeebb7fa7bbe0ee038ed0cc06d292632643bbcd3cc755b`
- analysis semantic SHA-256:
  `1ea1eacd360f22064d01f422b90432ddc5b2a6caa796847de2e212747a541916`

