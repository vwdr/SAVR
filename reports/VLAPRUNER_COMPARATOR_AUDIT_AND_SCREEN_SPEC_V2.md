# Step A (V2): VLA-Pruner comparator audit — released-source verdict and prospective screen spec

2026-09-21 UTC. Supersedes `reports/VLAPRUNER_COMPARATOR_AUDIT_AND_SCREEN_SPEC_V1.md`
(the paper-only provisional audit) for the §5 blockage resolutions and the §6
screen spec. V1 remains as a historical record. Per
`reports/PAPER_CONTRIBUTION_AND_MINIMUM_VALIDATION_V2.md` §5 Step A and the
user's explicit approval, the released VLA-Pruner repository was inspected —
**source files only, no weights, no install, no launch, no server access**.
Nothing below authorizes a GPU run. No positive result is assumed.

## 0. Source-inspection record (approved, bounded)

- Repository: `github.com/MINT-SJTU/VLA-Pruner`, branch `main`, pinned commit
  `84d4b7192c77abf1585610e2f12393319b7ebff9` (2026-06-04, "Update README.md").
  Full-repo git size per GitHub API: 21,775 KB (~21 MB), dominated by vendored
  transformers trees. License: MIT.
- Fetched (raw files, ~0.5 MB total) into the approved scratch directory
  `/private/var/folders/1k/ct_z8c1d3wn3jv4p559_b3dm0000gn/T/opencode/vla_pruner_src`
  — outside the git repo, nothing committed, nothing pushed, no checkpoint or
  weight files were downloaded. Storage estimate honored: source-only fetch is
  ~2% of the repo git size.
- The OFT tree (`src/openvla-oft/`) is the relevant comparator target; our
  runtime is OpenVLA-OFT (audit V1 §3). The plain-OpenVLA tree
  (`src/openvla/`) was not used for this assessment.

## 1. Scope and framing of the comparison question (unchanged)

Question: **does the existing simple fixed current-frame compression offer a
useful constrained-platform success–cost point compared to adaptive pruning on
the same hardware and observation stream?** Primary comparison: plain fixed-384
(384 visual tokens, pre-layer-0 deletion) versus adaptive VLA-Pruner at 75%
visual-token retention (384 of 512 visual tokens) — matched token count, not
necessarily matched compute. Competitive reference: adaptive at 50% retention
(256 tokens). Equal token counts do not imply equal compute or equal latency;
this is a timing measurement, not a token-count theorem.

**Release-vs-paper reconciliation (new, from source):** in the released OFT
code the pruning knob is `fastv_r` = *pruning ratio*; retention = 1 − fastv_r
(script comments + `_vlapruner_image_indices`/`_fastv_pruning_indices`
arithmetic: `num_keep = round(image_len * (1.0 - prune_ratio))`, per image
span). Released OFT run scripts evaluate retention 25% / 12.5% / 7.25%
(`fastv_r` 0.75 / 0.875 / 0.9275). The paper's Table-1 settings (50% / 25% /
12.5% retention, per audit V1 §2) are not all covered by the released OFT
scripts: **the released OFT scripts do not include the 50%-retention point**
(config accepts any ρ∈(0,1]; the 50% point would be `fastv_r=0.5`). 75%
retention — our primary matched arm — is inside the method's domain but is not
a released-script point either (`fastv_r=0.25`).

## 2. VLA-Pruner mechanism AS EXECUTED in the release (supersedes V1 §2)

V1 pinned the mechanism from the paper text. The released source confirms most
of it but corrects four points (marked "correction"). Executed path:

`run_libero_eval.py` → `get_action` (robot_utils) → `get_vla_action`
(openvla_utils) → `vla.predict_action` → `_regression_or_discrete_prediction`
→ `language_model.fastv_forward(..., fastv_config=...)` in the **vendored
transformers fork** (`src/openvla-oft/transformers`, `__version__ = "4.47.0"`).

- Config (`GenerateConfig`, `_configure_attention_pruning`): `use_vla_pruner`,
  `fastv_k=3` (default; scripts pass 3), `fastv_r` (prune ratio, scripts
  0.75/0.875/0.9275), `vla_pruner_layer=15` (scripts pass 15), mode
  `semantic_action`, semantic/action weight config 0.5/0.5, `av_hist_w=3`,
  `av_decay=0.8`, seed 7, 8-action open-loop chunks, per-episode
  `model.reset_av_history()` (run_episode).
- **Prune layer and layer-boundary arithmetic (blockage 3 resolved):**
  `fastv_layer = min(fastv_k, len(layers)-1)` = layer index 3 (0-based).
  Layer 3 runs at full length (its output attention is used to decide), then
  `hidden_states = hidden_states.index_select(1, keep_indices)` and layers
  4..31 (28 layers) run on the shortened sequence. The release's own FLOPs
  helper uses `full_layers = pruning_layer + 1 = 4`, `pruned_layers = 28`.
  So at matched retention our fixed-384 runs **all 32 layers short** while the
  adaptive release runs 4 full + 28 short; compute differs and must be
  measured, not assumed equal from token counts.
- **Score computation (blockage 1 resolved; one correction):**
  `attention_avg = torch.mean(last_layer_attention, dim=1)[0]` — mean over all
  heads, taken from the *prune layer's own* attention (layer 3), not a
  separate "last prefill layer". Per camera span
  (`_visual_token_spans`: two spans of 256 tokens when 2 images):
  - Semantic: `attention_avg[0 : action_token_start, image span].mean(dim=0)`
    — mean over all prefill rows (not only text rows; the span includes BOS /
    image / proprio / text tokens) for each visual column.
  - Action: `attention_avg[action_start : action_end, image span].mean(dim=0)`
    — mean over the 56 action-placeholder rows (`action_horizon=0` → all 56;
    layout matches our pinned 513-projected / 56-placeholder convention).
  - **Correction:** the executed path does **not** min–max normalize and does
    **not** apply the 0.5/0.5 weights; the normalization/weighted-sum code in
    `vla_pruner_utils.py` (`_normalize_scores`,
    `compute_semantic_action_reuse_indices`) is **dead code in the release**
    (imported by no module on the executed path). Selection uses raw means.
- **Temporal EMA (blockage 4 resolved; semantics confirmed):** when
  `use_temporal` (true for `semantic_action` mode), `av_hist` (deque, maxlen
  = w = 3, decay γ = 0.8) stores per-query action→visual vectors captured at
  **layer 15** (`vla_pruner_layer`) from each query's forward. The next
  query's action scores are **replaced** by the decay-weighted history
  (`γ^0..γ^(len-1)`, most recent weighted 1.0), built only when
  `len(av_hist) >= 3`. **Warm-start:** for the first w = 3 queries after each
  episode reset, `historical_attention is None` → `fastv_r` is forced to 0.0 →
  **those queries run fully dense** (no pruning), still with
  `output_attentions=True` across all 32 layers. From query 4 onward pruning
  engages with EMA-guided action relevance. The release's own latency metric
  (`metrics["time_elapsed"]`, wall-clock around `predict_action` only) **mixes
  these dense warm-up queries into its per-step averages.** An honest matched
  timing run must therefore separate warm-up (first ≤3 queries per fresh
  episode) from steady-state (≥4-th query), and cannot rely on the old
  two-frame traces (≤1 prior step).
- **Combine-then-Filter (blockage 2 resolved):** per span, take
  `prefill_scores.topk(span_keep)` and `action_scores.topk(span_keep)`, form
  the union, and if `|union| > span_keep` run
  `_redundancy_minization` (the MMDP greedy in the released code) on the
  candidate visual tokens *using the LLM input embeddings*
  (`inputs_embeds[0, image span]` indexed at the candidate positions).
  Distances: 1 − cosine similarity of the L2-normalized embeddings (computed
  in the model's BF16 dtype in this release, no fp32 cast). Greedy: first
  selected token = argmax of each candidate's *second-nearest* distance
  (`topk(distances, 2, largest=False).values[1,:]`), then iteratively the
  token maximizing `min(distance, chosen)`; per-query cost ≈ one (N×N;
  N = |union| ≤ 2·span_keep) cosine-matrix build plus `span_keep` row-select
  + min/argmax passes — O(span_keep·|C_dual|) distance operations as claimed.
- **Positions preserved (confirmed):** `keep_indices` keep original absolute
  indices; after pruning `position_ids = keep_indices.unsqueeze(0)` and a fresh
  causal mask is built; the action readout is unchanged
  (`last_hidden_states[:, -56-1 : -1]` on the pinned 56-placeholder layout).
  Same absolute-position convention as our compaction.
- **Dependency surface (blockage 5 resolved):** OFT tree pins `torch==2.2.0`
  (matches our pinned Torch 2.2.0+cu118), `torchvision==0.17.0`,
  `torchaudio==2.2.0`, `tokenizers==0.21.1`, `peft==0.11.1`, `timm==0.9.10`,
  `sentencepiece==0.1.99`, `tensorflow==2.15.0`, python ≥3.8, and installs the
  **vendored transformers fork (version 4.47.0)** as
  `transformers @ file:./transformers`. **New fact:** `openvla_utils.py`
  imports `from .vla_cache_utils import find_static_patches,
  task_relevant_selection, get_layer_mask_schedule` at module level and
  `vla_pruner_utils.py` imports `draw_patches_overlay` from the same module,
  but **`experiments/robot/vla_cache_utils.py` is not vendored in this repo**
  (absent from the full recursive tree). As-shipped, the OFT entry point
  cannot even import without that module on the path (it comes from the
  VLA-Cache sibling codebase). A local port (SpecPrune-style, our own harness)
  does not need that module; it needs only the `fastv_forward` selection
  semantics, which our neutral SDPA sidecar already captures.

## 3. Our pinned runtime (unchanged, still valid)

Repeated for context from V1 §3 (verified sources unchanged):
released OpenVLA-OFT checkpoint; Torch 2.2.0+cu118, Transformers 4.40.1,
32 LlamaSdpaAttention layers, batch 1, unpadded, no K/V cache, per-episode
reset, 8×7 action chunks, fixed 513-projected layout with protected nonvisual
positions, neutral attention-capture sidecar, existing dense/zero-pruning and
native/shadow parity machinery (≤1e-6 / byte-identical float32, trace hashes).

**Compatibility note:** the release's vendored transformers is 4.47.0; ours is
4.40.1. We do not install their fork — a local translation of the (small)
selection logic must compile under our pinned environment and be gated by the
existing parity tests, exactly as SpecPrune was.

## 4. Implementation-difference audit — updated verdicts

Legend: ✔ compatible / ⚠ needs care in port / ✖ impossible.

| # | Dimension | Verdict now (source-pinned) |
|---|---|---|
| 1 | Attention capture | ✔ — mean over heads at prune layer 3 for both score terms; sidecar capture proven neutral. ⚠ semantic rows = full prefill block [0, action_start), not text-only |
| 2 | Pruning layer placement | ✔ feasible; ⚠ compute differs at matched tokens (4 full + 28 short vs our 32 short); must be measured |
| 3 | Precision / dtype | ⚠ MMDP distances in model BF16 in release; we may match dtypes explicitly and report; parity gates required for our 4.40.1 translation |
| 4 | Two-camera input | ✔ per-camera spans (256/camera), independent span_keep — matches our position split (1..256 / 257..512) |
| 5 | Original positions | ✔ confirmed: `position_ids = keep_indices` after prune |
| 6 | Action readout | ✔ unchanged 56-placeholder rows; S_act rows = action_start..action_end (56 rows) |
| 7 | Normalization | ✔ no normalization on the executed path (dead code only) |
| 8 | Chunk execution | ✔ 8×7 chunks/query, same policy interface |
| 9 | History reset | ✔ confirmed per-episode `reset_av_history()`; cross-episode leakage rejected by contract |
| 10 | Horizons | ✔ same suites/horizon definitions |
| 11 | Timing boundaries | ⚠ warm-start = first 3 queries per fresh episode run dense (fastv_r forced 0); release latency mixes them into averages; steady-state requires ≥4-query traces with per-episode reset; report warm vs steady separately |
| 12 | Parity (dense control) | ✔ reuse; adaptive-zero-pruning≡dense parity required before any adaptive arm |

## 5. The five V1 blockages — resolved status

1. Head/row aggregation — **resolved:** mean over heads (dim=1) on prune-layer
   attention; semantic rows [0, action_start); action rows [action_start,
   action_end) (56 placeholders); no normalization on executed path.
2. MMDP representation — **resolved:** cosine distance over the LLM *input
   embeddings* of the candidate union, model-dtype, two-phase greedy (first by
   second-nearest distance, then max-min), per-span budgets.
3. K-layer arithmetic and EMA lifetime — **resolved:** prune after layer 3
   (4 full layers, 28 pruned); EMA deque w=3, γ=0.8, action vectors from
   layer 15, replaces current action scores from query ≥4 per episode.
4. Warm-start — **resolved:** first w=3 queries per (reset) episode run dense
   with `fastv_r=0`; included in release latency averages; not separable in
   their logged metric, so our timing block must reproduce the regime.
5. Dependency surface — **resolved:** torch 2.2.0 matches; vendored
   transformers fork is 4.47.0 (differ from our 4.40.1); the OFT entry point
   additionally requires a non-vendored `vla_cache_utils` module — port
   decision therefore is local translation (SpecPrune-style), no upstream
   install. MIT license; no weights needed.

With 1–5 pinned, an honest matched timing run is now definable; V1's "no
execution time can be promised" gate is lifted only for a port that passes the
parity gates below.

## 6. Frozen Step B screen specification (frozen after CPU verification — GPU launch still requires explicit approval)

Frozen 2026-09-21 after the CPU verification below. This section becomes
executable only after the user explicitly approves launch and coordinates
GPU 0. No GPU work has been performed.

### 6.1 Port and verification evidence (CPU-only, completed)

The adaptive comparator is a local, SpecPrune-style translation on our pinned
runtime — no upstream install, no weights:
`src/savr/openvla/vlapruner_adaptive.py` (selection mirrors, EMA action
history, `adaptive_decoder_forward`, per-episode `AdaptiveQueryState`).
Tests: `tests/openvla/test_vlapruner_adaptive.py`.

- **Local (Mac, `savr_cpu` venv: torch 2.14.0 / transformers 4.40.1 /
  numpy 2.5.3, python3.12):** 17/17 pass, including `ReleasedSourceParityTests`
  that AST-isolate the six released LlamaModel selection methods and the two
  released prismatic EMA/history methods from the pinned commit and compare
  our port against them on real tensors (fp32 and bf16).
- **TITAN pinned venv (CPU only, `CUDA_VISIBLE_DEVICES=""`, transformers
  4.40.1 / torch 2.2.0):** `Ran 17 tests` at exit 0 — **9 passed, 8 skipped**
  (never reported as "17/17"). The 8 skips are exactly the source-parity class: the approved
  upstream scratch fetch lives only on the dev Mac (release sources are policy-
  excluded from the repo and the server); the source-parity evidence therefore
  comes from the local run, while every behavior and model-level gate
  (byte-parity, counts/positions, warm/reset, EMA arithmetic, rejection modes)
  runs on the authoritative pinned environment. The upstream `test_pinned_source_hash`
  pins git blob SHA-1s: `modeling_llama.py`
  `b245f276dcfe9895fed160ff53b95ab7c9caa1f1`, `modeling_prismatic.py`
  `b6222c12b5c3c1552a59a046f866f4c7ce4f2f09` at commit
  `84d4b7192c77abf1585610e2f12393319b7ebff9`.
- Gates that passed: dense-ratio-zero forward is **byte-identical** to the
  native forward `(1,56,32)` incl. all 32 layer attentions (fp32 and bf16);
  retained counts are the actual kept tokens — 384 at `fastv_r=0.25`, 256 at
  `fastv_r=0.5` — with `position_ids=keep_indices`, `cache_position=arange(keep)`;
  first 3 queries per episode reset run dense (`fastv_r` forced 0, release
  semantics: history length, not query counter, drives the ratio); layer-15
  record matches the pinned `_update_action_attention_history` in dense and
  pruned branches (zero-filled pruned visual); EMA guided
  `(1·q3+0.8·q2+0.64·q1)/2.44` end-to-end; re-selection matches the released
  code's `prefill_scores`/`action_scores`/`kept_visual_tokens`/`image_spans`
  for both dtypes and for historical-guided and uniform modes; raw means, no
  min-max normalization, no 0.5/0.5 weighting on the executed path.

### 6.1.1 Two-stage screen implementation and CPU-only verification (2026-09-21)

New frozen artifacts (all byte-identical on TITAN under `/home/ved/SAVR`,
SHA-256 pinned in the configs):

- `src/savr/openvla/adaptive_screen_contract.py` — frozen four-arm contract:
  168 episode slots (160 primary + 8 native) and 240 timing slots (16 warmup +
  224 measured) with `validate_config`, `episode_slots`, `timing_slots`,
  `expected_retained`, `TRIAGE` (checks only + `positive_method_result=False`,
  `preference=[]`, no auto-selection), and `reconcile`.
- `src/savr/openvla/adaptive_query.py` — released-path bridge
  (`AdaptivePruneQuery`) + `count_sdpa()` witness context manager; the
  4.40.1 `output_attentions=True` eager-vs-SDPA delegation is captured at
  import (`_SDPA_REAL`) so adaptive calls measure zero SDPA invocations.
- `scripts/run_adaptive_qualification.py` / `analyze_adaptive_qualification.py`
  — **Stage S0**: real-checkpoint, native-audited qualification that pins the
  achieved adaptive-zero eager-vs-SDPA tolerance
  (`max(2×zero_pruning_observed, 1e-6)`, frozen ceiling `1e-3`) before any
  experimental outcome is scored. The tolerance is calibrated on the
  zero-pruning (dense-input) parity rows only; pruning-induced steady
  differences are recorded separately (`pruning_shift_observed_max`) and never
  calibrate or gate the tolerance.
- `scripts/run_adaptive_screen.py` / `analyze_adaptive_screen.py` — **Stage S1**:
  the four-arm screen gated in-session on S0; per-native-episode three-call
  shadow (native, 512, adaptive-zero) with witness rows; timing-and-episode
  loops per contract.
- `scripts/freeze_adaptive_launch.py` — deterministic freeze generator;
  `configs/openvla/adaptive_qualification_v1.json` and
  `configs/openvla/adaptive_screen_v1.json`; `docs/ADAPTIVE_SCREEN_PROTOCOL_V1.md`.
- Tests: `tests/openvla/test_adaptive_qualification.py` (6), `test_adaptive_screen.py` (6).

CPU verification:

- **Local (savr_cpu venv):** 12/12 adaptive tests pass (9 original + 3
  tolerance-separation regressions); the 17-test port suite passes locally.
  The full local `tests/openvla` discovery shows pre-existing GPU/server-asset
  suites that cannot run on the Mac CPU venv (unchanged files, untouched by
  this work).
- **TITAN pinned venv (CPU only, `CUDA_VISIBLE_DEVICES=""`):** new adaptive
  suites **12 passed, 0 skipped, exit 0**; port suite rerun **9 passed, 8
  skipped, exit 0** (source-parity class skipped by policy); regression rerun
  of `test_fixed_compression_qualification.py` (3 ok) and
  `test_compression_screen.py` (7 ok).
- Freeze determinism: regenerating the configs twice yields identical SHA-256.

### 6.2 Frozen configuration

- Arms (4): dense 512; fixed-384 (existing selector/contract unchanged);
  adaptive at 75% visual retention (`fastv_r=0.25` → 384 of 512); adaptive at
  50% retention (`fastv_r=0.5` → 256 of 512). No corrector, no training, no
  replanning. 75%/50% points are config-internal: released scripts only cover
  25/12.5/7.25% retention (§1).
- Adaptive config: `fastv_k=3` (prune after layer index 3; layers 0-3 full,
  4-31 short), `vla_pruner_layer=15`, `av_hist_w=3`, `av_decay=0.8`, mode
  `semantic_action`, seed 7, 8×7 open-loop chunks. Selection: head-mean
  attention at layer 3, raw means; semantic rows `[0, action_start)`, action
  rows = the 56 readout rows; per-camera spans `[1,257)`/`[257,513)`; top-k
  union per span, MMDP on LLM input embeddings (cosine, model dtype) when the
  union exceeds the budget; `position_ids=keep_indices` after prune; EMA
  replaces action scores from query ≥4; first 3 queries per episode reset run
  dense (`fastv_r` forced 0); `reset_av_history()` per episode.
- Population: same 40 already-consumed state-0 development conditions
  (10 per suite × 4 suites), 1 initial state per task, seed 7, released
  four-suite checkpoint — identical to `FIXED_COMPRESSION_SCREEN_RESULT_V1.md`
  and `CONTEMPORARY_REFERENCE_RESULT_V2.md`. Dense and fixed-384 are rerun
  in-session; historical numbers are not pooled.
- Order/labels: balanced rotating arm order per suite; separate native
  controls bracket the suites (8 native episodes, not pooled into primary
  success). **168 episode slots total = 160 primary (40 per arm) + 8 native**;
  **240 timing slots = 16 warmup (excluded) + 224 measured**.
- Frozen configs: `configs/openvla/adaptive_qualification_v1.json` (S0) and
  `configs/openvla/adaptive_screen_v1.json` (S1), generated deterministically
  by `scripts/freeze_adaptive_launch.py` (§6.1.1). Prior-chain hashes pinned:
  contemporary reference `afa1bc94…`, fixed-qualification summary
  `ad197c96…`, reference summary `6c8fc825…`.

### 6.3 Timing protocol (corrected, frozen per the Oct-verified contract)

- **Fixed-384 and dense-512 run compressed/dense from query one.** Only the
  adaptive arms run the release's algorithm-required dense startup (first 3
  queries after each episode reset). The **per-arm all-query mean — adaptive
  startup included — is the primary timing comparison across all arms**, so
  adaptive's required startup cannot inflate its comparison unfairly or be
  hidden from it; warm (queries 1–3) and steady (queries ≥4) means are
  reported separately for the adaptive arms as labeled context.
- 240 timing slots: **16 warmup** (2 rounds × 4 arms × frames 0,1 on the first
  trajectory as single 2-query traces, excluded from every statistic) + **224
  measured** (8 trajectories × 4 arms × 7 positions). One 7-query trace per
  (trajectory, arm) on shared per-episode state; positions 1–3 use frame 0,
  positions 4–7 use frame 1 (`offline_camera_inputs` exposes frames 0,1);
  `query == position` per trace; regime `uniform` for the controls, `warm` /
  `steady` for adaptive arms.
- All query preprocessing, transfers, selection/EMA/MMDP, decoder/head work,
  validation and dtype conversion stay inside the synchronized boundary;
  preprocessing and episode costs separately labelled; hardware warm-up is
  excluded consistently and identified separately. Timing buckets: per-query
  mean/median/p95 per arm per regime; all-query mean is the primary comparison;
  query counts logged per episode.

### 6.4 Parity gates (before success measurement) and completeness checks

**Stage S0 gate (real-checkpoint, runs first in the bounded launch; S1 never
starts without it):**

- 16 inputs (8 trajectories × frames 0,1), **176 exact model calls** (cap
  256); per input: native dense audited (`sdpa_calls ≥ 1`), adaptive-zero
  (`fastv_r=0`) plain + audited on both frames, plus a 4-query trace (3 dense
  + 1 steady) for each adaptive arm at frames 0,0,0,1.
- Gates: hooks neutrality (parallel plain/audited traces, identical query
  numbering); zero-vs-native parity (`hidden/normalized/raw`); structural
  execution contract (retention `512,512,512,384|256`; prune boundary at
  layer 3 — layers 0–3 full then 4–31 short; post-prune layer length =
  `full − (512 − retained_visual)` since trailing non-visual tokens survive,
  i.e. lengths `[full]*4 + [full−512+kept]*28`;
  kept positions on every post-prune layer; no cache reuse);
  adaptive-zero eager attention witness `== 0`; native witness `≥ 1`.
- **Tolerance separation (S0 rule):** `adaptive_zero_tolerance =
  max(2×zero_pruning_observed, 1e-6)` where `zero_pruning_observed` is the
  largest eager-vs-SDPA gap from the **adaptive-zero parity rows only** (dense
  inputs; zero-pruning comparisons). It must be **≤ the frozen ceiling
  `1e-3`**, else S0 marks `technical_stop` and S1 is refused. Pruning-induced
  steady differences vs dense (a different, shorter computation) are recorded
  and reported separately — `pruning_shift_observed_max` — and never
  calibrate, gate, or inflate the tolerance. The achieved tolerance becomes
  the adaptive-zero parity bound in S1.

**Stage S1 gates (per native shadow query):**

1. Native and dense-512 twin calls — unchanged: `≤1e-6` on hidden/normalized/
   raw and byte-equal commands; both record `sdpa_calls ≥ 1`.
2. Adaptive-zero (`fastv_r=0`, fresh episode state) twin — parity across the
   same three outputs bounded by the S0-achieved tolerance; `sdpa_calls == 0`
   (eager path, same overhead as `output_attentions=True` adaptive calls).
3. Completeness: every record carries its arm, per-query kept-token count
   (actual retained tokens, not config labels — 512/384/256), positions,
   layer-3 attention hash, EMA history hash, and observation/trace hashes;
   parity rows carry `observation_sha256`, `command_sha256`,
   `command_bytes_equal`, and both SDPA witnesses (attention audit).

### 6.5 Resource caps and runtime estimate

- **Stage S0** (adaptive qualification, first): exact **176 model calls**
  (16 × [native + zero-plain + zero-audited] + 8 frame-1 × 2 modes × 8
  parallel trace calls), ceiling `model_calls=256`, `episodes=0`,
  `seconds=3600`, `aggregate_memory_mib=23552`,
  `artifact_bytes=268435456`. Estimates: ~176 calls ≈ 5–10 min plus model
  load; runtime ~20–40 min total. Output `results/adaptive-qualification-v01`
  (fresh `mkdir(exist_ok=False)`, immutable).
- **Stage S1** (screen, only after S0 passes): `model_calls=7000`,
  `episodes=168`, `seconds=21600`, `aggregate_memory_mib=23552`,
  `artifact_bytes=536870912`. Peak aggregate GPU-0 memory ≤ 23,552 MiB
  (unchanged precedent; measured dense peak 16,504–16,506 MiB). Estimated
  calls ≈ 240 timing + 160 primary + 8 native × 3 shadow bridges ≈ **3,900**;
  runtime estimate ≈ **1.5–2 h** scaled from the fixed screen (4,222.59 s /
  2,846 calls ≈ 1.48 s/call) plus adaptive selection/EMA overhead and the
  eager-attention cost counted honestly; hard stop at the 6 h cap.
- No retries, no automatic fallback, no installs/downloads on the server, no
  sudo, GPU 0 (`GPU-bb2451d6-2989-a112-5c18-8892943710e4`) only, coordinated
  after a fresh scoped idle check. Output `results/adaptive-screen-v01`.

### 6.6 Technical stops and decision table

Stops (preserve evidence): any dense-behavior change vs the in-session dense
control, parity failures, nonfinite outputs, cap breaches, runtime/version
drift; equal token counts never substitute for measured compute. S0-specific:
`adaptive_zero_tolerance > 1e-3` → technical stop; no S1 launch. S1-specific:
S0 summary incomplete/stopped/hash-mismatched, or the S0-bound config sha
mismatch at gate → refuse launch.

Decision table (prospective triage, not frozen gates):

- Adaptive no slower with no worse observed success → no developmental support
  for a fixed-selector positive advantage; do not scale up to obtain one.
- Fixed-384 useful at a distinct cost/quality point → candidate systems
  finding; still requires uncertainty and independent confirmation.
- Tiny/inconsistent differences → inconclusive; no positive claim or automatic
  expansion.
- Technical mismatch or changed dense behavior → preserve evidence and stop.

### 6.7 Deviations documented

- Local translation on transformers 4.40.1 (release fork is 4.47.0); parity
  gates are byte-identity (model-level) + native/shadow on the pinned runtime;
  no upstream install.
- 75% and 50% retention are config-internal comparison points, not released
  script points.
- Source-parity tests skip on the server (upstream scratch is Mac-only by
  policy); parity evidence comes from the local run; behavior/model-level
  gates run on the pinned environment (both green).
- Warm-start dense queries are algorithm-required **for the adaptive arms
  only** (fixed-384 stays compressed from query one) and are included in the
  primary all-query timing comparison across all arms; warm and steady-state
  are reported separately so the release's own mixed per-step latency metric
  is not reproduced as a single ambiguous number.
- Attention audit: 4.40.1 `LlamaSdpaAttention.forward` delegates to the eager
  implementation whenever `output_attentions=True` (pinned upstream blob
  `b245f276…`). S0 measures the eager-vs-SDPA gap on the real checkpoint
  (adaptive-zero arm) and pins the achieved tolerance ≤ `1e-3` before any
  outcome is scored; in-session SDPA witnesses record `0` for every adaptive
  call and `≥ 1` for native/dense calls, so the attention overhead is counted
  honestly rather than assumed equal.

### 6.8 Launch authorization

GPU launch, in-session timing, and any experiment run still require explicit
user approval of this frozen protocol (§6) and GPU-0 coordination. Nothing in
this section authorizes execution by itself. The readiness gate for that
approval is `reports/ADAPTIVE_SCREEN_READINESS_V1.md`, which pins the frozen
hashes, the exact bounded two-stage command, the S0/S1 stopping conditions and
the honest 9/8 CPU-test wording.

## 7. What this document does not claim

- No claim that our fixed selector beats VLA-Pruner or any published method.
- No claim of matched compute at matched token counts (4-full+28-short vs
  32-short; warm-up dense queries).
- The paper's published numbers are not our reproductions; no cross-paper
  hardware-normalized ranking is asserted. We do not port or run the release
  end-to-end; we translate its described selection semantics and verify parity.
- No re-authorization of the gating/corrector line, a learned-correctors
  paper, or scaling-up of the comparison.

## 8. Local-only change record

This turn changed documentation and added the local adaptive port and its
tests. Release source was inspected with explicit user approval (source only,
~0.5 MB, scratch dir outside the repo, nothing committed/pushed, no weights, no
install, no server/GPU access, no training, no experiment, no manuscript, no
sent email). New: `src/savr/openvla/vlapruner_adaptive.py` and
`tests/openvla/test_vlapruner_adaptive.py` (copied into `/home/ved/SAVR` to run
the fix on the pinned environment — CPU only). Updated this report §6 to the
frozen protocol with verification evidence, resource caps and runtime
estimate; updated `OPENCODE_HANDOFF.md`, `PROJECT_STATUS.md`. Note
`run_10.sh`/`run_spatial.sh` confirm the released OFT seed is 7 and
`num_trials_per_task` 50 in their protocol; our screen uses 1 state-0 condition
per task (40 total). §6 is frozen but not authorized for launch; GPU execution
requires explicit approval and GPU-0 coordination.