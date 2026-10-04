# Independent runtime audit: attention mismatch

Date: 8 September 2026  
Classification: confirmed implementation-semantic discrepancy; closed-loop effect not yet measured  
Scope: baseline investigation for the proposed positive-solution direction. No presentation work.

## Finding

The two project environments do not implement the same dense attention operation.
The original OpenVLA-OFT SDPA implementation uses bidirectional attention. The
later compatibility implementation uses causal attention, including when
temporal reuse is disabled. This is a change to the policy computation, not
merely a different software version or a timing optimization.

The distinction was verified by executing the installed attention components on
CPU, observing the masks actually passed to attention, and changing a future
input token. It agrees with the authors' [pinned Transformers implementation](https://github.com/moojink/transformers-openvla-oft/blob/bc339d9ad707454c0c115970db43c260067c61ab/src/transformers/models/llama/modeling_llama.py#L672).
The original source deliberately removes causal masking while retaining padding
exclusion. Its relevant SDPA call sets `is_causal=False`.

This supplies a concrete explanation to test for the unexpectedly weak later
dense baseline. It does **not** establish how many failures the mismatch caused,
whether correcting it will make caching effective, or whether the proposed
learned method will work.

## What differs mathematically

For query position i and key position j, attention is

\[
\operatorname{softmax}(QK^\top/\sqrt d + M)V.
\]

In the released SDPA path, M permits every valid key position. In the
compatibility path, M additionally blocks j > i. The multimodal sequence places
visual tokens before later instruction/action positions. Causal masking therefore
changes which current information can influence their representations. It also
changes interactions between action positions. This changes the computation
even with identical checkpoint weights, images, action-head indices, and no
temporal cache reuse.

“Bidirectional” here refers to positions **within the current policy query**. It
does not mean access to future robot observations or future episode outcomes.

## Source and historical provenance

All server inspection stayed within `/home/ved/SAVR` through `ssh titan`.

| Runtime | Transformers | Installed `modeling_llama.py` SHA-256 |
|---|---|---|
| `envs/openvla-oft` | 4.40.1 | `3aac24cec583a6ef5f60b6ec634a8bd3c8377784c5c63c8cc14cb6790554c52e` |
| `envs/vla-cache-compat` | 4.47.0 | `34b00dd58c9887780a7947329cb96468a7fe1427e8fa49dc773ce2c1afc627d4` |

Both report PyTorch 2.2.0+cu118. The installed original source converts a
padding-aware causal mask to a padding-only mask at lines 719–723, and calls SDPA
with `is_causal=False` at line 731. The compatibility source lacks that conversion
and uses `is_causal=causal_mask is None and q_len > 1` at line 677.
When an explicit padding mask is present, its SDPA flag is false but the mask
itself remains triangular. Checking the flag alone would miss the discrepancy.

The compatibility source hash exactly matches the hash recorded in
`reports/OPENVLA_D62_S4_V01_TECHNICAL_STOP.md` during the earlier requalification.
The completed CAC screen's launcher explicitly selects this environment.
The S3 “official” loader imports the original evaluator but runs it inside this
same compatibility environment. It changes loader/configuration compatibility,
not Llama attention masking.

The checkpoint's Llama construction forwards the selected attention backend.
The inspected loader does not explicitly choose a backend; installed Transformers
prefers supported SDPA. The previous summary does not independently record every
live attention class and effective mask. The next real-checkpoint check must
record those directly rather than rely on default-dispatch inference.

Additional static exclusions: the current checkpoint and pinned repository's
Prismatic sources differ in two diffusion-scheduler lines, not the L1 regression
path used here. The inspected evaluation path has the expected two images,
crop setting, eight-action queue, state normalization, gripper conversion, and
initialization steps. This is not an exhaustive proof of the whole evaluator.

## Executed CPU tests

The probe uses a randomly initialized 24,736-parameter, two-layer Llama model,
six input positions, FP32, seed 7, and one CPU thread. No 7B checkpoint, dataset,
simulator, or GPU allocation is involved. It compares four cases with cache
production both disabled and enabled: unchanged inputs, a changed valid final
token, right padding, and a changed padded token.

| Measurement | Original runtime | Compatibility runtime |
|---|---:|---:|
| Effective attention without padding | All 6 keys per query | Only current/earlier keys |
| Earlier hidden-state max change after changing a valid future token | 0.6519075036 | 0.0 |
| Earlier hidden-state max change after changing a padded token | 0.0 | 0.0 |
| Hidden-state difference caused solely by producing a cache | 0.0 | 0.0 |

The sensitivity numbers describe synthetic hidden states, **not robot action
error or task success**. Their role is to corroborate the observed information
flow, not estimate the full-model effect size.

The first original full-model probe completed eight forwards. The first
compatibility full-model probe stopped at unconditional CUDA timing code with
CUDA hidden, before any decoder layer ran. That diagnostic stop is preserved.
No GPU was enabled to work around it.

The revised CPU probe invokes each runtime's actual embedding, mask builder,
decoder layers, and final normalization directly, bypassing top-level CUDA
timers. It does not replace or patch the attention operation. Eight component
executions completed in each runtime. A further eight original full-model
executions matched the original component executions exactly: all masks and all
eight full hidden-tensor SHA-256 values agree. This validates the component
probe against the runnable original reference for this synthetic population.
It does not establish full compatibility-runtime GPU equivalence.

Totals: 32 completed tiny-model evaluations across the exploratory and v2
probes; one preserved CPU-incompatible top-level attempt; zero OpenVLA model
loads, zero simulator episodes, zero new task-success results. No installed
runtime or checkpoint source was edited.

## Why the earlier checks missed this

The earlier action-readout correction was necessary, but insufficient. Both the
“official” evaluator and custom helper were executed using the same altered
Transformer dependency. Agreement between their tensors authenticated that
shared implementation, not equivalence to the original OpenVLA-OFT stack.

This was a gap in our validation design. Increasing the number of comparisons
inside the same dependency could not establish independence. The new reference
must include the model's dependencies and effective attention operation, not
only the evaluator's top-level Python source and action-head slice.

## Consequences for the research record

1. Preserve every earlier result, configuration, and frozen stopping decision.
   Do not overwrite measurements or silently relabel them as corrected runs.
2. Treat the latest dense 68/120 and cached 0/120 as measurements of the
   historical compatibility-stack policies. Their interpretation as a clean
   comparison of the released policy and its intended cached variant is not
   established. The mismatch's actual contribution remains unmeasured.
3. The same dependency undermines claims of full released-policy equivalence
   in the later BRACE/PAIR/CAC qualification chain. Earlier BRACE/PAIR results
   also retain the separately documented action-readout problem.
4. Do not automatically invalidate the earlier whole-prefix and camera work.
   Those used the original environment in relevant launch paths. Their actual
   runtime provenance must be assessed separately; today's finding is not
   evidence that every experiment shared this mismatch.
5. Do not resume CAC training or claim a positive result from discovering a
   bug. Do not declare the new compression/repair candidate validated either.
6. The first prerequisite in the September 8 direction audit remains active:
   establish an independently defensible dense baseline before selecting the
   next learned method or qualifying a published comparator.

## Next bounded check

Use the already-present original environment and checkpoint first. Do not edit
the compatibility environment in place and do not simply flip `is_causal`:
padding masks also need the right semantics, and cached compaction is a
separate problem. No installations or checkpoint downloads are needed for the
first diagnostic.

### A. Original-stack action/reference check

- Use the same eight previously consumed observations used by S3. Do not open
  new held-out trajectories or select observations based on action differences.
- Verify exact checkpoint/source hashes and resolve the real loaded attention
  classes, effective masks, image preprocessing, instruction IDs, normalized
  state, action-head inputs, and executed actions.
- Compare the original evaluator with the corrected dense helper **inside the
  original runtime**, with no inherited caching sidecars or legacy loader chain.
- Disable only checkpoint-mutating loader convenience operations. Load the
  authenticated checkpoint code without modifying its files or `auto_map`.
  Preserve the released evaluator's native return type.
- Keep `output_attentions=False`: the original SDPA implementation falls back
  to another attention implementation if this flag is enabled. Observe the
  actual operation without changing it. Capture hooks must restore cleanly.

### B. Isolate attention from other version differences

On those same observations, a temporary, explicitly recorded causal-attention
control in the **original** runtime can hold weights and all other computation
fixed. Its implementation must first reproduce the two mask contracts on tiny
inputs, preserve padding, and be restricted to Llama attention rather than
vision attention. Restore it and verify original outputs again. Do not present
a cross-version difference alone as an isolated causal effect of attention.

### C. Short closed-loop diagnostic, then a decision

Only after A/B, freeze a small paired schedule from previously consumed
development conditions, distributed across all four suites. Record the exact
conditions, arm ordering, call/time/memory ceilings, and analysis before running.
Use unchanged official observation/action/simulator conventions. Distinguish
technical errors from task failures. Do not expand or rerun based on partial
successes. A small diagnostic can identify a large restoration effect, not
establish the published benchmark percentage or a paper-quality success rate.

If original-stack dense behavior improves, qualify that baseline on a larger
fixed development population and reconsider the comparator and method choice.
If it remains weak, continue diagnosis of the baseline rather than train an
adapter to compensate for an unexplained reference discrepancy. Neither outcome
authorizes repeating every historical cache experiment.

GPU coordination remains required before this step. Use only one user-confirmed
GPU, below 23,552 MiB, within `/home/ved/SAVR`. No GPU ID has been selected for
this new diagnostic. A complete GPU harness/schedule has not yet been frozen,
and no GPU run, training, or automatic retry has been launched.

## Reproduction and artifacts

Run `scripts/probe_openvla_attention_contract.py` through `ssh titan`, with stdin
supplied from the local script, once per project runtime. Set
`CUDA_VISIBLE_DEVICES=""`, `PYTHONDONTWRITEBYTECODE=1`, offline Hugging Face modes,
project-local cache/temp paths, and one CPU thread. Use `--execution full_model`
only in the original runtime and `--execution dense_components` in both.
The script prints JSON and has no artifact-writing code. Its SDPA observer
forwards the original arguments unchanged and is restored in `finally`.

The original probe source SHA-256 was
`8610003418be2c47a051e9f946c819ba7b2eb5059e70722b126b1767ef9d2228`.
The revised source SHA-256 is
`7f63018c8c0aceb075fab66faf0041f0dcb5ffc6faf10a958b68f89f0ea0252d`.

Raw captured outputs, including the initial compatibility stop, are in
`reports/attention_audit_2026-09-08/`. The five exact artifact hashes are pinned
in `scripts/verify_openvla_attention_audit.py`. Run:

```bash
python3 -B scripts/verify_openvla_attention_audit.py
python3 -B -m unittest discover -s tests/openvla -p test_attention_audit.py -v
```

Verification completed: five authenticated artifacts, original component/full
agreement, distinct attention contracts, and 12 passing regression tests.
Tests cover missing/duplicate cases, padding leakage, incorrect causal flags,
nonfinite sensitivity, cache-production changes, observer restoration, source
identity, and component/reference tensor disagreement. These tests do not
certify the unexecuted GPU harness.

Changes this turn are local diagnostic code, captured evidence, this report,
and an additive status/decision correction. No manuscript, poster, historical
run evidence, installed server environment, or checkpoint was modified. No
GitHub push was made.
