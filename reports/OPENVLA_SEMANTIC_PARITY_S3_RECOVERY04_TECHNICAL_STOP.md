# OpenVLA Semantic Parity S3 Recovery 04 Technical Stop

Date: 2026-09-01  
Classification: technical integration stop; no semantic or method result

## Outcome

Recovery 04 did not complete S3. The mandatory selected-GPU cache-mode
qualification passed, but the single authorized OpenVLA attempt stopped
fail-closed after two model calls and before completing the first observation.
No simulator outcomes, expert actions, terminal outcomes, or raw actions were
accessed or persisted.

## What passed

- Local compilation, shell validation, and focused tests passed.
- The remote CUDA-hidden preflight passed.
- All 40 focused remote tests passed.
- The selected-GPU synthetic qualification verified the pinned runtime:
  `use_cache=None` returned a cache, `use_cache=False` returned no cache, and
  `use_cache=True` returned a cache.
- The qualification used three tiny synthetic Llama calls, zero OpenVLA model
  loads, zero OpenVLA model calls, and no simulator data.
- The Recovery 04 OpenVLA attempt loaded the exact frozen checkpoint and
  executed two calls before stopping.

## Stop diagnosis

Recovery 04 translated the legacy custom-path `None` argument to explicit
`False` before invoking `structurally_aligned_dense_forward`. That helper still
contained an older input guard allowing only `None` or `True`, so it raised:

`OpenVLASemanticError: dense parity use_cache must be None or true`

The synthetic qualification exercised the translation wrapper against a probe
and the pinned Llama runtime, but it did not exercise the wrapper through this
higher-level helper guard. This gap explains why the preflight passed while the
model attempt stopped. The stop occurred before the no-cache custom forward
reached the language model; it therefore provides no scientific evidence about
official/custom semantic parity or the cache method.

## Integrity

- Run: `openvla-semantic-parity-s3-v05-recovery04`
- Model calls: 2 of 32
- Completed observations: 0 of 8
- Peak aggregate selected-GPU memory: 15,699 MiB
- Automatic retry: false
- Checkpoint restoration: exact and complete
- Post-stop GPU 0 state: 6 MiB, 0% utilization
- Cache qualification summary SHA-256:
  `c603a06d10f8c23c4058a7c28bbb79e8be436f8b432c3b1fd1ab815b3a4da765`
- Technical stop SHA-256:
  `ad9b9396a6fa5c06b82817ad3be4875307d182fa7b9dca056f90c34172bd530e`
- Technical traceback SHA-256:
  `c9e63f93bef009a5b69ff860c9fa9f59030064d267be69031ec1f864e734be14`

## Required boundary before another attempt

No Recovery 05 attempt is authorized. A future recovery must make the dense
helper's accepted cache modes and the model-level cache modes one explicit
contract, then exercise the entire helper path with a synthetic model stub—not
only the translation wrapper and Llama runtime. That qualification must prove
that the helper accepts `False`, forwards it unchanged, returns no cache for the
control, preserves `True` for cache and sidecar modes, and leaves official
evaluation behavior unchanged. Scientific inputs, call schedule, tolerance,
resource caps, and S4/C1H/C2 boundaries must remain frozen.
