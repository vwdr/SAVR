# OpenVLA Semantic Parity S3 Recovery 05 Pass

Date: 2026-09-01  
Classification: semantic-foundation qualification pass; not a method result

## Result

Recovery 05 completed the frozen S3 semantic-parity stage successfully. The
released OpenVLA-OFT evaluator, the corrected dense no-cache control, the dense
cache-producing control, and the sidecar-instrumented cache control were tested
on all eight frozen observations (two from each LIBERO suite).

- Observations: 8 of 8
- Model calls: 32 of 32
- Ordered comparisons: 344 of 344 passed
- Comparisons per observation: 43
- Maximum absolute error: `1.4014237548209962e-08`
- Frozen tolerance: `1e-6`
- First mismatch: none
- Simulator or terminal outcomes accessed: no
- Expert actions accessed: no
- Raw actions persisted: no

This establishes that the corrected custom dense forward uses the released
evaluator's semantic action-readout boundary on this frozen offline population.
It also establishes that producing a cache, and installing the observation-only
sidecar, did not change hidden states or actions beyond the frozen tolerance.
It does not establish closed-loop task success, cache reuse safety, adapter
quality, or a positive paper result.

## Required qualifications

Before loading OpenVLA, Recovery 05 passed all of the following:

- Authenticated local configuration and 20 source/input files
- CUDA-hidden remote preflight
- 46 focused remote tests
- Prior 22-boundary comparator qualification
- Prior pinned-runtime `None`/`False`/`True` cache qualification
- New full dense-helper qualification using GPU tensors and a synthetic model

The full-helper qualification traversed the actual dense helper four times and
observed forwarded modes `[None, False, True, True]`. The explicit control
returned no cache; both cache paths returned caches; the sidecar context entered
and exited; and all synthetic actions were identical. It used zero OpenVLA model
loads and zero OpenVLA model calls.

## Integrity and resources

- Run: `openvla-semantic-parity-s3-v06-recovery05`
- Elapsed model-attempt time: 46.73 seconds
- Peak aggregate GPU-0 memory: 16,955 MiB (< 23,552 MiB cap)
- Checkpoint restoration: exact, complete, and backup-free
- Post-run GPU 0 state: 6 MiB, 0% utilization
- Automatic retry: false
- Worker summary SHA-256:
  `19d5519d1a11574a3ed02904b55464ea9e7ee4ed7857e62ab10d8f85073ddf9e`
- Comparison manifest SHA-256:
  `6e99421120fb68d5ee4ab29b07b75c1c0c4758aec9883403c2d36d55954db839`
- Full-helper qualification SHA-256:
  `e64e2999d37c35f12ce34facbaa0bec9698797334ccfb0a068bf2d12132fbd8a`

## Boundary

S3 is complete. The next protocol stage is S4 D62 semantic requalification,
but it remains unauthorized and unstarted. C1H and C2 also remain blocked until
the protocol's intervening qualifications pass.
