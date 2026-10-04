# OpenVLA Semantic Parity S3 Recovery 03 Readiness Audit

Date: 2026-09-01  
Status: `GPU_COMPARATOR_QUALIFIED_ONE_MODEL_ATTEMPT_AUTHORIZED`

Recovery 03 replaces narrow comparison fixes with a complete representation
contract for all 22 unique S3 boundaries. Eighteen boundaries require
CUDA-tensor/CUDA-tensor comparison, `normalized_actions` requires an explicit
CUDA-bfloat16-to-CPU-float32/NumPy comparison, and three action/determinism
boundaries require CPU NumPy/NumPy comparison. Unknown boundaries or type/device
drift stop fail-closed.

Before model loading:

- every source, checkpoint, prior-stop, input, authorization, resource, and
  scientific-contract hash reconciled;
- CUDA-hidden preflight passed;
- 37 focused remote tests passed;
- a separate selected-GPU micro-qualification exercised all 22 contracts,
  actual CUDA-to-CPU float32 conversion, shape mismatch, NaN, strict-JSON
  sealing, and representation-drift rejection;
- the qualification passed in 1.38 seconds with zero model loads and zero
  model calls; and
- GPU 0 returned to 6 MiB and 0% utilization.

The comparator qualification is preserved at
`results/openvla-comparator-qualification-s3q-v01/worker_summary.json` with
semantic SHA-256
`f0f97238fc5e5fd2665db5ecac80f439e46e4f85909ef0128563e856b10e8290`.

The approved model attempt retains the original eight observations, 32-call
cap, order, semantic boundaries, `1e-6` gate, official loader, one-GPU limit,
and outcome protections. No automatic retry or later stage is authorized.
