# PAIR-VLA Phase P3 Technical Stop 02

Date: 2026-08-29  
Attempt: `pair-p3-physical-v02`  
Decision: **TECHNICAL STOP; NO SCIENTIFIC RESULT; NO AUTOMATIC RETRY**

## What happened

The corrected project-local LIBERO configuration worked, the pinned checkpoint
loaded on GPU 0, and the first warm-up model forward executed. The worker then
requested `output.last_hidden_state`, but the pinned Transformers 4.47.0
`CausalLMOutputWithPast` exposes the required tensor as
`output.hidden_states[-1]` only when `output_hidden_states=True`. The mismatch
raised an exception before any timing block.

## Preserved evidence and interpretation

- `results/pair-p3-physical-v02/technical_stop.json`
- Timed blocks completed: 0 of 96
- Completed-query ledger: 0 of 688
- Actual model forwards attempted: 1 warm-up forward
- Expert actions, simulator state, and terminal outcomes accessed: none
- Automatic retry: false
- Checkpoint metadata backups remaining: none

The V02 stop record's 6-MiB peak value is the launch-time aggregate sample,
not a valid peak for the attempted forward: the exception occurred before the
post-query memory sampler. It must not be cited as V02 model memory evidence.

## Corrective design for a separately authorized V03 attempt

The forward path now matches the already-validated BRACE/OpenVLA integration:
it requests hidden states and reads the final tensor from
`output.hidden_states[-1]`. This is a pinned API correction, not a method or
timing-frontier change. Failed-query accounting is moved immediately before
the model forward, and the exception path resamples aggregate GPU memory before
writing a technical stop.

V03 must have a new immutable run identity, pass the complete CPU preflight,
recheck aggregate GPU availability, and receive explicit user approval. V02
remains immutable and is not timing or scientific evidence.
