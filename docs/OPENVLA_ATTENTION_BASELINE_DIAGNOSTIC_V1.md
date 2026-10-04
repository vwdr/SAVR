# Original-runtime attention baseline diagnostic

Date: 9 September 2026. Status: bounded diagnostic authorized by the user's
instruction to continue and delegated choice of a single available GPU.
This is not training, a cache-method retry, or a positive-paper experiment.

## Question and intervention

Does the causal attention used by the later compatibility stack change the
released policy's actions and closed-loop behavior? Use the existing original
OpenVLA-OFT environment and checkpoint. Compare its native bidirectional Llama
attention with an explicitly causal control in the same process, holding all
weights, preprocessing, normalization, action-head indexing, and episode rules
fixed. This isolates attention from other differences between environments.

The intervention wraps only the 32 Llama SDPA modules, not vision attention.
For unpadded input it activates causal SDPA; for padded input it adds an upper
triangular exclusion while preserving padding. No runtime/checkpoint file is
edited. Every wrapper restores in `finally`. Requests for attention outputs,
which could trigger the original runtime's eager fallback, are rejected.

## Frozen population and execution

1. Use all eight already-consumed training observations in
   `configs/pair/p3_inputs_v1.json`, observation step zero. Authenticate the
   source HDF5 files. Read only their images and robot state, not expert actions.
2. Run exactly four policy calls per observation: original evaluator with
   boundary capture, corrected dense helper, causal-control evaluator, and
   restored original evaluator. Require matching shapes, finite values, and
   maximum absolute discrepancy no greater than 1e-6 at all compared reference
   and restoration boundaries. A disagreement stops before simulation. Record
   actual attention classes and the effective mode at every Llama layer.
3. Select two lexicographically first task names per suite from the old
   `headroom_stage1` manifest at initial-state ID zero. These are eight already
   consumed conditions, two each from Spatial, Object, Goal, and Long. Resolve
   tasks by name, not the manifest's global task indices. The exact condition
   records are copied into the new configuration before launch.
4. Run original and causal control on every condition. Alternate their order
   within each suite so four pairs begin with each arm. Reset the random seed
   to 7 before each arm; create a fresh environment, use the same initial state,
   wait ten steps, and execute eight-action chunks. Retain the released image
   preparation, gripper conversion, and per-suite step limits.
5. Do not expose intermediate task outcomes or use them to change the schedule.
   Progress contains only completed-episode and query counts. Write the complete
   16-episode record and final summary only after reconciliation. On a technical
   failure preserve its traceback, keep partial outcomes undisclosed, and stop
   without an automatic retry.

The plain-language experimental arms are **original attention** and **causal
attention control**. Neither reuses visual computation. A difference is evidence
about the attention change, not an estimate of how well a corrected cache or
trained adapter would perform.

## Resource and server boundary

GPU 0 was selected after the user's delegated choice, using only aggregate
GPU readings. UUID: `GPU-bb2451d6-2989-a112-5c18-8892943710e4`.
Recheck its UUID, memory at most 1,024 MiB, and utilization at most 5% immediately
before launch. If unavailable, do not evict another job or silently move cards.

One model, one GPU, aggregate memory strictly below 23,552 MiB. Limit the
PyTorch allocator to 23,000 MiB and disable TensorFlow GPU allocation; TensorFlow
is used only for image preprocessing. This study makes no latency claim.
All caches, temporary files, logs, and outputs stay under `/home/ved/SAVR`.

Hard limits: 32 offline calls, 16 episodes, 696 total policy calls, 90 minutes
after preflight, and 512 MiB of artifacts. The call ceiling is 32 plus four
times (ceil(220/8) + ceil(280/8) + ceil(300/8) + ceil(520/8)) = 696.
The wall limit is a ceiling, not a claim that the diagnostic needs 90 minutes.

No installations, downloads, training, new held-out data, unrelated process
inspection, or modification of old experiments is authorized by this diagnostic.
Disable only the original loader's checkpoint-writing convenience functions;
use the existing authenticated checkpoint metadata. Hash checkpoint weights,
heads, processing metadata, and source before and after the attempt.

## Verification before launch

Eleven CPU tests must pass in the original TITAN environment with CUDA hidden:
native observer neutrality, future-token information flow, padding exclusion,
restoration after a normal call and after an exception, exclusion of vision
attention from the intervention, rejection of attention-output fallback,
episode queue/termination/horizon behavior, propagation of technical errors,
and exact fake-environment trace agreement with the actual released evaluator's
episode function. Passing CPU tests does not itself qualify the real model.

Freeze the configuration's authenticated source/input hashes after tests. Reject
changed hashes and an existing output directory. Preserve the initial test
setup correction (the first local invocation lacked `PYTHONPATH=src`) as a
development issue, not a GPU attempt or a scientific result. All 11 tests passed
on TITAN without skips; three episode tests also passed locally, with eight
runtime-dependent tests skipped there.

## Interpretation and stopping point

Analyze only when `worker_summary.json` is complete and all 16 records and 32
offline calls reconcile with its hashes. Report raw paired counts, discordant
pairs, and suite breakdowns. Do not characterize eight conditions per arm as a
precise benchmark estimate. Do not compare this selected subset directly with
the earlier 120-condition percentage as if the populations were equal.

If causal control changes actions and worsens paired success, that supports an
attention-specific contribution to the runtime problem. It does not establish
that attention explains the entire historical deficit. If the original policy
also remains weak, continue independent baseline diagnosis. If original behavior
looks restored, qualify it on a larger fixed development population before
selecting a learned method or claiming a positive result.

Stop after this diagnostic and its report. Do not start compression, adapters,
CAC training, or a new benchmark campaign automatically.
