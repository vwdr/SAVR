# Step A: VLA-Pruner comparator compatibility audit and prospective screen specification

2026-09-21 UTC. No-compute audit per
`reports/PAPER_CONTRIBUTION_AND_MINIMUM_VALIDATION_V2.md` §5 Step A and the latest
user decision (hold manuscript/email; resolve the fixed-384 vs adaptive-pruning
tradeoff question first). Nothing below authorizes a GPU run, a download, an
install, a training job or any server access. No code or weights of VLA-Pruner
were cloned, downloaded or installed. The audit is based on the released paper
text (arXiv 2511.16449v5) and our own pinned runtime sources.

## 1. Scope and framing of the comparison question

Question: **does the existing simple fixed current-frame compression offer a
useful constrained-platform success–cost point compared to adaptive pruning on
the same hardware and observation stream?** Primary comparison: plain fixed-384
(384 visual tokens, pre-layer-0 deletion) versus adaptive VLA-Pruner at 75%
visual-token retention (384 of 512 visual tokens) — matched token count, not
necessarily matched compute. Competitive reference: the published 50%-retention
adaptive setting (256 tokens). Equal token counts do not imply equal compute or
equal latency; this is a timing measurement, not a token-count theorem.

## 2. VLA-Pruner mechanism as pinned from the paper (v5)

- Prunes visual tokens at transformer-layer index K with K=3 by default
  (`§4.3`); per the paper's FLOPs derivation, the first K−1 layers process the
  full sequence and the remaining layers process the shortened sequence. Exact
  layer-boundary arithmetic must be confirmed from released source.
- Importance has two components (`§4.2`):
  - Semantic relevance S_vl: text→vision attention from last-layer prefill.
  - Action relevance S_act: action→vision attention averaged over the latter
    half of layers, smoothed across previous control steps by a finite-window EMA:
    Ŝ_act^t = Σ_{i=1..w} γ^i S_act^{t−i} / Σ_{i=1..w} γ^i, with window w=3 and
    decay γ=0.8. Warmed up for w steps (`§4.3`).
- Combine-then-Filter (`§4.2.2`): take top-M̃ of each score as candidate sets,
  union them, then select the retained subset of size M̃ by a max-min dispersion
  greedy over pairwise cosine distances (first token by maximal second-nearest
  distance). Complexity is O(M̃·|C_dual|) distance computations per query.
- Retained tokens keep their original absolute positions (standard pruning
  convention; the paper does not re-index positions). Requires attention-score
  capture at the prune layer and the relevance layers; stores historical action
  attention across queries (reported as negligible memory).
- Evaluated on OpenVLA and OpenVLA-OFT on LIBERO (Table 1) and real-world OFT
  tasks; released repository `github.com/MINT-SJTU/VLA-Pruner` advertises OFT
  support. Publication settings: 50% / 25% / 12.5% retention; 75% retention is
  inside the method's domain (ρ ∈ (0,1]) but is not one of its published table
  points.

## 3. Our pinned runtime (what the comparison runs on)

Pinned facts from verified reports/source (`CURRENT_FRAME_ROBOT_PILOT_RESULT_V1.md`,
`CONTEMPORARY_REFERENCE_RESULT_V2.md`, `FIXED_COMPRESSION_SCREEN_RESULT_V1.md`,
`src/savr/openvla/*`):

- Released OpenVLA-OFT checkpoint; original runtime Torch 2.2.0+cu118,
  Transformers 4.40.1, 32 LlamaSdpaAttention layers, batch 1, unpadded current
  queries, bidirectional attention; no K/V cache; inference-only contracts that
  reject training mode/change.
- Fixed compression: full two-camera encoding per query (513 projected tokens =
  512 visual + proprio/boundary), stratified static selection to 256/384/512
  before decoder layer zero, original rotary position IDs preserved for retained
  tokens, protected nonvisual positions (0 and 513+); action readout via
  `select_official_action_hidden` on the pinned 56-placeholder layout; 8×7
  action chunks per query; per-episode explicit state reset.
- Timing methodology (all existing arms): matched-input timing module, 8
  two-frame training traces, warm-ups, 4 repeated rounds, first-query/second-
  query separation, all adapter/selection/copy/output overhead included;
  separately labelled evaluator preprocessing.
- Existing parity machinery: dense/zero-pruning parity performed for the fixed
  selector (512 = dense control), native/shadow twin-query parity with
  hidden/normalized/raw error ≤1e-6 and byte-identical float32 chunks, trace
  hashes, and a neutral SDPA score-capture sidecar
  (`SemanticSDPASidecarTap`, `NativeAttentionScores`) that does not alter outputs.

## 4. Implementation-difference audit: VLA-Pruner requirement vs our runtime

Legend: ✔ compatible / ⚠ needs pinning from released source / ✖ blocked without
code access.

| # | Dimension | VLA-Pruner requirement | Our runtime capability | Verdict |
|---|---|---|---|---|
| 1 | Attention capture | Softmax attention scores at prune layer (K≈3), last prefill layer, and latter-half layers (action→vision) | SDPA score-capture sidecar already proven neutral and count-checked (32-layer, batch-1, unpadded invariants) | ✔ mechanism exists; ⚠ head/row aggregation rules (mean over heads? text rows? action rows) must be pinned to match source |
| 2 | Pruning layer placement | Delete after first K−1 full-sequence layers (K=3) | `fixed_decoder_forward` deletes before layer 0; SpecPrune port deletes mid-forward per-layer | ✔ feasible; ⚠ claimed FLOPs assume K chosen identically; at matched 75% retention ours runs all 32 layers short vs their early full layers — compute differs and must be measured, not assumed equal |
| 3 | Precision / dtype | Paper reports fp32-softmax captures and BF16-ish inference; release may assume newer torch/transformers | Pinned Torch 2.2.0+cu118, Transformers 4.40.1, BF16 model, FP32 score capture for selection | ⚠ no silent install; a local port (as done for SpecPrune) must compile under pinned env; parity tests gate it |
| 4 | Two-camera input | OFT dual camera; scores over all 512 visual positions | Full two-camera encoding per query; positions 1..256 / 257..512 per camera | ✔ compatible in principle; ⚠ camera-specific handling in release (wrist offsets) must be pinned |
| 5 | Original positions | Retained tokens keep absolute rotary IDs | Explicitly preserved by our compaction (absolute_positions map) | ✔ same convention |
| 6 | Action readout | Head input rows unchanged (visual-only pruning) | `select_official_action_hidden` uses pinned readout positions; protected nonvisual positions guaranteed | ✔ compatible; ⚠ action rows used for S_act (the 56 placeholders vs readout rows) must match |
| 7 | Normalization | None (dropped tokens only) | Normalized output identical path; no change needed | ✔ |
| 8 | Chunk execution | OFT 8-action parallel decode | Fixed 8×7 chunks/query in harness; same policy interface | ✔ compatible |
| 9 | History reset | EMA/action-attention history across queries (w=3) | Harness requires explicit per-episode reset; cross-episode leakage rejected by contract | ✔ reset semantics exist; ⚠ confirm release resets at episode boundaries, not globally |
| 10 | Horizons | LIBERO per-suite horizons; method agnostic | Same suites/horizon definitions reused from pilot/screen | ✔ |
| 11 | Timing boundaries | Selection overhead, EMA bookkeeping, warm-start w steps | Prior arms: warm-up + 4 rounds, first/second query split; all overhead included | ⚠ two-frame traces give ≤1 prior step of history, below w=3; steady-state EMA cost is not characterized by the old timing block — trace length must be extended or warm-up regime reported explicitly |
| 12 | Parity (dense control) | ρ=1.0 retention should reproduce dense byte-identically | Existing dense/native/shadow parity machinery reusable unchanged | ✔ reuse; must also add adaptive-zero-pruning (=dense) parity before any adaptive arm runs |

## 5. Pinned blockages that require released-source access (post-approval)

The following cannot be resolved from the paper text and will block an honest
matched timing run until the released repository is inspected (explicit approval
to download is required; V2 §5 forbids silent installs):

1. Exact head/row aggregation for S_vl and S_act (mean over heads; which text
   rows; placeholder rows vs readout rows) on the OFT 56-placeholder layout.
2. MMDP cosine-distance representation (projected patch features? hidden?),
   dtype, and greedy implementation (exact cost per query in our env).
3. Layer-boundary arithmetic for K=3 (which layers run full) and whether the
   EMA buffer persists across control steps/traces in their evaluator.
4. Warm-start behavior of the first w queries per episode (dense run? partial
   pruning?) and whether it is excluded from or included in their latency.
5. Dependency surface of the released code (torch/transformers versions) versus
   our pinned environment; port decision (local translation, SpecPrune-style)
   versus upstream install.

Until items 1–5 are pinned, no execution time can be promised and no launch is
authorized. Code availability does not constitute a compatibility pass.

## 6. Prospective Step B screen specification (draft — NOT authorized)

To be frozen in full only after §5 resolves and after explicit approval.

- Arms (4): dense 512; existing fixed-384 (exact existing selector/contract);
  adaptive VLA-Pruner 75% retention (384 of 512); adaptive VLA-Pruner 50%
  retention (256 of 512). No corrector, no training, no replanning.
- Population: the same 40 already-consumed state-0 development conditions
  (10 per suite × 4 suites), one initial state per task, seed 7, released four-
  suite checkpoint — identical to `FIXED_COMPRESSION_SCREEN_RESULT_V1.md` and
  `CONTEMPORARY_REFERENCE_RESULT_V2.md`. Rerun dense and fixed-384 in-session for
  this comparison; do not pool historical numbers.
- Order/labels: balanced rotating arm order per suite; separate native controls
  bracketing suites (8 native episodes, not pooled into primary success).
- Timing: matched-input module extended so VLA-Pruner arms are characterized in
  their warm regime (first ≤w queries) and, with longer traces, a steady-state
  regime; all selection, EMA and MMDP overhead included; preprocessing and
  episode costs separately labelled.
- Recorded per primary episode: final executed chunks and observation hashes so
  technical discrepancies localize; per-arm memory with explicit measurement
  definitions; total calls and caps (prior runs: per-run call caps, 23,552 MiB
  memory cap, elapsed caps — reuse the same bounded values).
- Technical stops: any change in dense behavior relative to in-session dense
  control, parity failures for adaptive-zero-pruning≡dense, nonfinite outputs,
  cap breaches, or runtime/version drift → stop and preserve evidence.
- Decision table (new prospective triage rules, not frozen gates from the
  completed adapter experiment):
  - Adaptive no slower with no worse observed success → no developmental
    support for a fixed-selector positive advantage; do not scale up to obtain
    one.
  - Fixed-384 useful at a distinct cost/quality point → candidate systems
    finding; still requires uncertainty and independent confirmation.
  - Tiny/inconsistent differences → inconclusive; no positive claim or
    automatic expansion.
  - Technical mismatch or changed dense behavior → preserve evidence and stop.
- Server discipline (unchanged): `ssh titan` only, only `/home/ved/SAVR`,
  coordinated GPU 0 after a fresh scoped idle check, one run, no automatic retry,
  no installs, no downloads, no GPU-claiming without stopping for coordination.
- This section does NOT authorize launch. It becomes an executable plan only
  after the §5 audit resolves and the user explicitly approves the frozen
  protocol.

## 7. What this document does not claim

- It does not claim our fixed selector beats VLA-Pruner or any published method.
- It does not claim matched compute at matched token counts.
- It does not treat VLA-Pruner's published numbers as our reproductions; no
  cross-paper hardware-normalized ranking is asserted.
- It does not re-authorize the gating/corrector line, a learned-correctors
  paper, or any scaling-up of the comparison.

## 8. Local-only change record

This audit is a new local report. No server access, no GPU, no download, no
install, no training, no experiment, no manuscript, no email, no GitHub push.
Updated: `OPENCODE_HANDOFF.md` and `PROJECT_STATUS.md` checkpoints.