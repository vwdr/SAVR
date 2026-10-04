# OpenVLA Semantic Parity S3 Recovery 02 Technical Stop

Date: 2026-09-01  
Classification: `TECHNICAL_HARNESS_STOP_NO_SEMANTIC_OR_METHOD_RESULT`  
Status: `ATTEMPT_CONSUMED_NO_AUTOMATIC_RETRY`

## What Recovery 02 established

Both Recovery 02 corrections worked as intended. The previously failing
normalized-proprio boundary passed after canonical shape exposure, and the
worker successfully wrote a strict-JSON technical stop when a later error
occurred. The checkpoint was restored exactly and the run produced a complete,
authenticated technical record.

## New stop

The first observation again completed its four scheduled model calls. The
ordered harness passed 15 comparisons through the dynamic `action_hidden`
boundary, then reached `normalized_actions`.

At this boundary the official capture is a CUDA tensor while the custom path
intentionally stores its action-head output as a CPU NumPy array. The generic
comparison helper only recognized the both-tensors case. For the mixed case it
called `numpy.asarray` directly on the CUDA tensor, which PyTorch rejects:

`TypeError: can't convert cuda:0 device type tensor to numpy.`

This is a deterministic evidence-harness type-conversion error. It is not an
action-value mismatch and does not support a semantic or method conclusion.

## Integrity and boundary

- model calls: 4;
- completed observations: 0;
- ordered comparisons passed before the stop: 15;
- simulator outcomes, expert actions, and raw actions: none;
- peak aggregate GPU-0 memory: 16,413 MiB;
- checkpoint protected files and inventory restored exactly;
- checkpoint restoration error: none;
- post-stop GPU 0: 6 MiB and 0% utilization; and
- automatic retry: false.

The immutable technical stop has SHA-256
`1a0198579be84e5f1606b7658d673256218cdeef698ceddc62f43a34c5f623fb`
and semantic SHA-256
`8c9f419b1b6a017cde3c60d015feea1d4557ec5e3801fbba0010ced6b36cf10d`.

## Recovery requirement

Any Recovery 03 must normalize only heterogeneous comparison inputs—not model
computation—by detaching a tensor, converting it to float32 on CPU, and then
comparing it with the NumPy value. Before another model attempt, its audit must
enumerate every S3 boundary's runtime representation and test all relevant
tensor/array combinations, including CUDA-to-CPU conversion without running
the policy. The population, call schedule, semantic gates, loader, resources,
and protected-data boundaries must remain unchanged.

Recovery 03 is not authorized. S4, D62/C1, C1H, C2, simulator work, and
training remain blocked.

Evidence:
`results/openvla-semantic-parity-s3-v03-recovery02/technical_stop.json` and
`technical_traceback.log` in the same directory.
