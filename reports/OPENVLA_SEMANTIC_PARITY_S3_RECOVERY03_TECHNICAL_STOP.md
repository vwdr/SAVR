# OpenVLA Semantic Parity S3 Recovery 03 Technical Stop

Date: 2026-09-01  
Classification: `POST_COMPARISON_HARNESS_STOP_NO_STAGE_RESULT`  
Status: `ATTEMPT_CONSUMED_NO_AUTOMATIC_RETRY`

## What passed

Recovery 03's complete representation repair worked. Its separate selected-GPU
micro-qualification passed all 22 unique boundary contracts with zero model
calls. In the model attempt, all 43 ordered comparisons for the first
observation then passed, including:

- preprocessing, tokens, images, proprio, visual and multimodal features;
- official versus custom action hidden states;
- normalized and unnormalized actions;
- `use_cache` mode action/hidden determinism; and
- sidecar-off/on determinism.

These passes are encouraging diagnostic evidence, but the observation was not
marked complete and the eight-observation S3 gate did not finish.

## Stop cause

After the comparisons, the original harness asserted that
`use_cache=None` must return no K/V cache. In the pinned Transformers runtime,
the language-model implementation explicitly resolves `None` to
`self.config.use_cache`. That default is enabled, so `None` correctly returned
a cache and the harness stopped with:

`ParityTechnicalStop: use_cache=None unexpectedly returned a cache`.

This assertion is inconsistent with the runtime API and with S3's actual goal:
test that producing K/V does not change hidden states or actions. A true
no-cache control must pass `use_cache=False`; cache-producing controls must pass
`use_cache=True`. The stop is therefore a post-comparison harness-contract
error, not a semantic mismatch.

## Remaining-path audit

The code after this stop contains only four classes of checks:

1. cache presence/absence;
2. sidecar and structural-layout invariants;
3. resource/completion arithmetic; and
4. manifest, checkpoint, and summary sealing.

The sidecar context already completed its 32-layer capture check during the
four calls. Structural layout was already derived and validated for each custom
forward. True-cache behavior follows the pinned runtime's explicit `use_cache`
branch. Resource, restoration, strict-JSON, and terminal-stop sealing have been
exercised by the preceding qualifications and stops. The incorrect `None`
absence assertion is the only identified remaining execution-path defect.

## Integrity

- model calls: 4;
- completed observations: 0;
- ordered semantic comparisons passed before the stop: 43;
- simulator outcomes, expert actions, and raw actions: none;
- peak aggregate GPU-0 memory: 16,413 MiB;
- checkpoint restored exactly with no restoration error;
- GPU 0 returned to 6 MiB and 0% utilization; and
- automatic retry: false.

The immutable stop has SHA-256
`f539b5b7c68381b51fc2bfbce43d3cdda56f96cea2f2f0cd4121e19580cbe9b0`
and semantic SHA-256
`bc7ed9d1098816297dd23417d376582210975743f698b1f1c11f5b5ab28e5085`.

## Required correction

Any Recovery 04 must replace only the no-cache control's `None` with explicit
`False`, retain explicit `True` for cache and sidecar controls, and rename the
schedule accordingly. It must source-audit and test the runtime's
`None`/`False`/`True` resolution plus every remaining post-comparison invariant
before another model attempt. Model computation, official reference, inputs,
semantic comparisons, tolerance, resource limits, and protected-data rules
remain unchanged.

Recovery 04 is not authorized. S4, D62/C1, C1H, C2, simulator work, and
training remain blocked.

Evidence:
`results/openvla-semantic-parity-s3-v04-recovery03/technical_stop.json` and
`technical_traceback.log` in the same directory.
