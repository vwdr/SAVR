# SpecPrune-OFT source audit and visual-compaction qualification

Date: 2026-09-10. Status: source audit and CPU foundation complete; the published
comparator is **not yet integrated or empirically qualified**.

## Decision

Use the official SpecPrune-VLA OFT code as the next comparator implementation
reference. Do not run its stock loader/evaluator against our checkpoint.
Build a documented same-checkpoint adaptation on the authenticated original OFT
runtime. Its pruning-disabled path must reproduce that policy before a benchmark.
This is implementation work within the approved comparator step, not a new
research pivot. The baseline result remains 39/40 on consumed development states.

## Sources and provenance

Official repository: <https://github.com/alexwhz-sjtu/SpecPrune-VLA>.
Frozen revision: `8091adc4b574ce9008d49a1dc9a210f4eec314c1`.
Paper: <https://arxiv.org/abs/2509.05614>.
The repository identifies the work as ICML 2026; its venue label is not an
independent reproduction or a guarantee of performance on our hardware.

Eight selected upstream text files are preserved, unmodified, in
`reports/specprune_source_audit_v1/upstream/`. Their Git blob identifiers match
the pinned upstream tree. Complete SHA-256 values and the checkpoint source
identity are recorded in `source_contract_audit.json` beside that directory.
The upstream MIT license and source-file notices are retained. No dependency,
model, dataset, or checkpoint download or installation occurred.

Reference checkpoint source SHA-256:
`f40ee7883e16aab1a2d89b6e8f31cc81f6b8055120b1fefe169e05c7031098fa`.
Original runtime LLaMA source SHA-256:
`3aac24cec583a6ef5f60b6ec634a8bd3c8377784c5c63c8cc14cb6790554c52e`.

## Findings that affect our comparison

| Item | Observed source behavior | Required treatment |
|---|---|---|
| Action-head readout | Upstream selects `[-57:-1]`. Our checkpoint selects the 56 positions immediately preceding the action placeholders, equivalent to `[-58:-2]` for this dense layout. Input-extension operations have identical ASTs. | Map the authenticated checkpoint's absolute readout positions through compaction. This is an explicit adaptation, not an undocumented fix to upstream. |
| Attention outputs | Upstream requests attention matrices; our qualified reference does not. Its SDPA implementation falls back to manual attention when matrices are requested. | Validate the actual path and masks. Preserve native bidirectional policy output; any auxiliary attention calculation needs neutrality, score-equivalence, and cost checks. Static inspection alone does not establish that the upstream runtime is causal. |
| Loader side effects | The upstream loader invokes checkpoint auto-map and model-logic update helpers. | Never invoke those mutating helpers on the authenticated checkpoint. Use isolated imports and project-local caches. |
| Timing population | The inspected evaluator accumulates timing for successful episodes after the first success. | Our latency comparison must include a frozen set of complete queries regardless of episode success, with all selection/controller overhead. Published timings cannot be transplanted to TITAN. |
| Controller presets | Code computes ordinary local top-k values 24/19 and precise-mode values 15/12. The precise-mode name is not evidence that it always retains more tokens. | Preserve and label the pinned source preset; trace actual retained counts. Reconcile prose/code differences without outcome-driven tuning. |
| Dynamic selection | Importance updates occur at layers 14/19/24, while pruning starts at layer 10. The first pruning point therefore precedes those updates. | Characterize zero-score ties and exact source behavior with synthetic traces before claiming algorithmic equivalence. Do not silently move layer indices. |
| State and positions | The code uses previous selection indices, episode-level confidence, layerwise masks and absolute positions. | Test reset, first query, mode switches, query/key map alignment and retained nonvisual tokens. Reuse of token indices is distinct from reuse of stale K/V values. |

The readout observation is a source incompatibility with **our** authenticated
checkpoint. It is not evidence that the authors' published results are invalid.
No pretrained SpecPrune action or closed-loop result was measured in this audit.

## Implemented and tested

`scripts/audit_specprune_source.py` authenticates the pinned eight-file archive
and reference source, compares the action input extension and readout ASTs, and
emits the machine-readable audit. Source drift is rejected. Three independent
synthetic prompt lengths demonstrate the one-position difference.

`src/savr/openvla/visual_compaction.py` provides a shared, visual-only position
map. It retains BOS, both camera views, proprioception, language, all action
readout and placeholder states, and the stop token. Repeated compaction cannot
restore already removed tokens. It preserves absolute rotary positions and
delegates action extraction to the existing authenticated semantic helper.
It does not select tokens or implement SpecPrune's layerwise algorithm.

TITAN CPU tests, with CUDA hidden, passed with no skips:

- Six source-audit tests, including source identity and drift rejection.
- Nine position-map tests, including repeated compaction, protected tokens,
  malformed selections, camera deletion, and cross-layout rejection.
- Five tensor tests on a random two-layer, width-32 LLaMA in the original
  project runtime. Retain-all output was exactly equal to native dense output.
  Compaction from 512 to 256 visual tokens produced a finite 349-token sequence
  for the synthetic 605-token input. All 56 readout positions were preserved.
  A changed future token still affected earlier states, checking bidirectionality.

These twenty tests are CPU integration evidence, not real-checkpoint parity,
robot success, latency measurements, or a method efficacy result. On the Mac,
the five tensor tests were skipped because the necessary runtime is absent;
they were then executed successfully on TITAN. No local runtime was installed.

## Remaining work and stopping rules

The next step is the isolated pruning/controller port and synthetic equivalence
tests, followed by a bounded real-checkpoint parity check. Exact experimental
configuration and source hashes must be frozen before that run. The detailed
sequence is in `docs/SPECPRUNE_COMPARATOR_QUALIFICATION_V1.md`.

Do not train the corrector, use held-out data, benchmark a partially ported
comparator, reinterpret historical results, or claim a positive result at this
checkpoint. Preserve technical stops without automatic retry. Neither the
checkpoint nor installed attention code was edited. All server actions stayed
inside `/home/ved/SAVR`; only CPU work ran. No GPU workload, GitHub push,
manuscript change, or poster change occurred.
