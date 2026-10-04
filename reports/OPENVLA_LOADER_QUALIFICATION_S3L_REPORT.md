# OpenVLA Loader Qualification S3L Report

Date: 2026-09-01  
Classification: `ZERO_CALL_TECHNICAL_QUALIFICATION`  
Status: `PASS_STOP_BEFORE_S3_RECOVERY01`

## Purpose

S3 v01 stopped before its first policy call because its loader assumed two
VLA-Cache-only configuration fields existed. A deeper audit also found that
the v01 loader entered the cache fork's source tree, which could synchronize
modified model logic into the checkpoint. Before permitting another semantic
parity attempt, S3L isolated and tested only the replacement loader.

S3L was authorized for one model load on one selected GPU and prohibited all
policy, vision-encoder, action-head, action-production, simulator, and outcome
calls. It could not authorize or launch S3 Recovery 01.

## Pre-execution verification

- The local package audit found and corrected two static-gate defects before
  TITAN was touched: method-source indentation normalization and a false
  positive in the prohibited-call scanner.
- The corrected CUDA-hidden remote preflight passed against the exact official
  source revision, checkpoint metadata, authenticated files, authorization,
  resource limits, and empty immutable output root.
- Eight focused remote tests passed.
- GPU 0 was selected only after aggregate telemetry reported 6 MiB and 0%
  utilization.

## Result

The single S3L attempt passed:

- loaded class:
  `transformers_modules.openvla-7b-oft-libero-four-suite.modeling_prismatic.OpenVLAForActionPrediction`;
- the runtime source of `_regression_or_discrete_prediction` and
  `predict_action` exactly matched the authenticated checkpoint source;
- action readout was prompt-derived and did not use the cache fork's legacy
  tail slice;
- model-logic synchronization did not occur;
- the absent cache controls were injected only as dense `None` values;
- policy calls: 0;
- vision-encoder calls: 0;
- action-head calls: 0;
- actions produced: 0;
- simulator outcomes accessed: false;
- elapsed time: 32.66 seconds;
- peak aggregate GPU memory: 15,275 MiB, below the strict 23,552 MiB limit;
- checkpoint protected files and inventory were restored exactly, and the
  loader-created backup was removed; and
- the immediate post-run aggregate check reported GPU 0 at 6 MiB and 0%.

The checkpoint's own runtime warning notes that it was authored against
Transformers 4.40.1 and Tokenizers 0.19.1, while the shared compatibility
environment uses 4.47.0 and 0.21.1. This does not invalidate the planned S3
within-runtime parity comparison because both official and corrected paths use
the same frozen environment. It remains a limitation for claims about
cross-version equivalence and must be retained in later interpretation.

## Integrity and boundary

The immutable summary is
`results/openvla-loader-qualification-s3l-v01/worker_summary.json` with file
SHA-256
`185baf9f4f46272f9f4566e52dd7299be118e7e022a49c5d63704ba76a1863c2`
and semantic SHA-256
`c51db9011d7c105232a31c73c8faaed6de59236bb257ce4ad16d38b2dcaf2915`.

S3L is a technical qualification, not a semantic-parity or method result. It
establishes that the corrected official loader can safely reach the exact
authenticated model under the current resource boundary. S3 Recovery 01 has
not run and remains explicitly unauthorized. S4, D62/C1 requalification,
C1H, C2, simulator evaluation, and training remain blocked.
