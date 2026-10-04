# CAC C1 S5 Requalification Pass

Date: 2026-09-01  
Run: `cac-c1-s5-requalification-v01`  
Decision: S5/C1 complete—pass; stop before C1H

## Result

Corrected CAC C1 passed every frozen tensor, semantic, chronology, isolation,
streaming, timing, storage, and memory gate. This re-establishes CAC systems
feasibility downstream of the corrected official OpenVLA action readout and
corrected D62 substrate `D62_BAL_PT1_S4C_V1`.

This is not evidence that a trained adapter repairs cached actions or preserves
closed-loop task success. No training or simulator was used.

## Independent official-policy boundary

The released evaluator and corrected custom dense path matched exactly on the
frozen C1 anchor:

- official and custom action-hidden shapes: `[1,56,4096]`;
- hidden maximum absolute error: 0.0;
- normalized-action maximum absolute error: 0.0;
- unnormalized-action maximum absolute error: 0.0;
- observation-copy/state-mutation contract: pass;
- independent reversible official oracle: pass.

This boundary ran before all C1 warmup and internal controls.

## C1 gates

- Calls: exactly 97 planned and completed; hard cap 160.
- Dense/all-fresh normalized-action error: 0.0.
- `fc2(Z_C)` reproduction error: 0.0.
- Sidecar action/cache equality: exact.
- Recursive age-1--4 repetitions: exact.
- Reset phase/provenance: exact.
- Maximum physical-source age: 4.
- Branch cache clone: byte-exact and storage-disjoint.
- Parent cache/tracker/salience and CPU/CUDA/Python/NumPy RNG: unchanged.
- Zero-initialized correction adapter: bitwise bypass.
- Adapter parameters: 5,070,599; bfloat16 bytes: 10,141,198.
- Numeric record: exactly 1,380,352 bytes.
- Median horizon-4 gross complete-cycle saving: 22.5969%, above 8%.
- Feature extraction: 10.5964 ms median.
- Full adapter forward: 2.3883 ms.
- Peak aggregate GPU memory: 16,789 MiB, below 23,552 MiB.
- Artifacts before summary: 90,698,062 bytes, below 4 GiB.

All nine machine gates in `result.json` are true.

## Comparison with historical C1

The corrected result remains consistent with the prior engineering conclusion:

- historical h4 gross saving: 22.41%; corrected: 22.60%;
- historical peak memory: 16,785 MiB; corrected: 16,789 MiB;
- historical adapter time: 2.38 ms; corrected: 2.39 ms.

The corrected numeric feature record has a new SHA-256, as expected because
the official action states/base actions were corrected. The instruction
embedding hash is unchanged. Historical C1 remains preserved but is superseded
for official-equivalence-dependent claims.

## Evidence integrity

- Result SHA-256:
  `44c2d69bcc3ccd0983b75a75639140cb39d85c457fe6d162ee5b37cffbcd084e`.
- Result semantic SHA-256:
  `c4cd5bc67e181fb0e18e0e5f4b78750a6eccb635c9b73aadd0a6709f570fd46c`.
- Numeric feature record SHA-256:
  `baaf2f784f97f7440d2d4759d76d0a53fcc3274031700a92dc4ce8630cca5e60`.
- Sidecar SHA-256:
  `1d25537076bfc5e90ad975867587b9b0cbb8869ad4656fcb0ab4cbe5dab577bb`.
- Instruction embedding SHA-256:
  `5a48946a9c150c8729f34a2e81a2491b082d0375cc61637fb26d20c258104d3a`.

Checkpoint metadata and inventory were restored exactly, the loader backup was
verified and removed, and GPU 0 returned to 6 MiB/0%. No expert actions,
terminal outcomes, raw action values, downloads, training, or automatic retry
occurred. Runtime writes remained under `/home/ved/SAVR`.

The first shell invocation was rejected before preflight because the transferred
launcher lacked an executable bit. It created no output and made zero model
calls. The authenticated script was then invoked explicitly through `bash`;
this changed no scientific or server-permission setting and produced the sole
S5 model attempt reported above.

## Boundary

S5/C1 is complete. Under the semantic requalification protocol, C1H is now
eligible but remains unauthorized. C1H is the first outcome-bearing screen of
whether sufficient repair opportunity exists; it requires separate approval.
C2, training, and any positive-paper claim remain blocked.
