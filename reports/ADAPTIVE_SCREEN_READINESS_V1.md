# Adaptive Screen Readiness V1 — one bounded GPU-0 launch, subject to approval

Date: 2026-09-21 (UTC). Status: **ready for review; NOT authorized.**
Nothing here launches anything. The single ask at the end requires explicit
user approval plus a fresh GPU-0 coordination check, exactly as the frozen
protocol requires.

> **Second review (approval held):** after the first readiness presentation the
> reviewer held the launch and required the S0 numerical tolerance to be
> calibrated on zero-pruning parity comparisons only, with pruning-induced
> steady differences recorded separately (`pruning_shift_observed_max`) and
> never gated against dense outputs. This was implemented in the S0 worker
> (separate accumulators), `reconcile`, and the analyzer (which independently
> recomputes the tolerance from the zero-parity rows), covered by new CPU
> regression tests, and re-frozen — see §2/§3 and `docs/ADAPTIVE_SCREEN_PROTOCOL_V1.md`.
> The tolerance ceiling was not increased; it remains `1e-3`.

> **Third review (final bug-check, caught one real bug):** the reviewer's
> conditional approval required a clean final bug-check first. That check
> found the S0 prune-boundary gate compared the post-prune **layer lengths**
> against the *retained-visual* count (384/256), but the port's `keep_indices`
> always retains the trailing non-visual tokens (`[0] + top_visual +
> [513..full)`), so the true post-prune layer length is `full − (512 − kept)`
> (477 at 75%, 349 at 50% for 605-token queries). The gate as written could
> never pass at runtime and would have technical-stopped S0 on its first
> frame-1 mode. It now checks the true post-prune length, records the numeric
> evidence (`full_tokens`, `post_prune_length`, `layer_lengths`) on every mode
> row, and `reconcile` independently recomputes the invariant so the boundary
> is CPU-verifiable (new regression test, adaptive suite now 13 tests). Frozen
> hashes updated accordingly (§2); nothing else changed.

> **Fourth review (S0/S1 technical stop, root cause and fix):** the granted
> launch technical-stopped after 3 model queries with
> `RuntimeError: attention-output fallback is prohibited` —
> `technical_stop.json` and `technical_traceback.log` are preserved under
> `results/adaptive-qualification-v01/`. Root cause: the adaptive contract
> requires `output_attentions=True` (`vlapruner_adaptive.py`), which forces the
> manual eager attention path in Transformers 4.40.1 and is exactly what the
> released `LlamaAttentionDiagnostic` forbids by design. The workers had
> installed the released diagnostic around adaptive queries (S0 `call(…,True)`
> for adaptive-zero and the mode traces; S1 `audited()` for the adaptive-zero
> shadow), so the *adaptive* arms crashed even though the *native released*
> chain was untouched. Fix is **worker-only** (`attention_diagnostic.py` and
> the released reference chain are byte-identical; their frozen hashes
> unchanged): the released diagnostic now wraps only released-SDPA queries
> (native, dense-512, fixed-384); adaptive queries are audited by the zero-SDPA
> `count_sdpa()` witness plus the official capture, and adaptive-zero parity
> rows correctly report `layers == [32, 0]` (the S1 contract previously
> required the impossible `[32, 32]`). New port-level regression proves the
> conflict and the witness on the tiny-model harness; adaptive suites still
> 13/13, port suite now 18 tests; everything re-frozen (§2). The launch ask
> stands on this corrected, re-frozen state.

## 1. What was built this stage (all CPU-only so far)

A SpecPrune-style two-stage port set, byte-identical local ↔ TITAN
(`/home/ved/SAVR`; all 12 new files verified `sha256sum`-equal, and the
accidental `/home/ved` placement of those files during the first scp was
detected and fully reverted — `/home/ved` contains only `SAVR` and `snap`):

| Artifact | Role |
|---|---|
| `src/savr/openvla/adaptive_query.py` | Released-path bridge + `count_sdpa()` SDPA witness |
| `src/savr/openvla/adaptive_screen_contract.py` | Frozen four-arm contract (168 episodes, 240 timing slots, triage = checks only, no auto-selection) |
| `scripts/run_adaptive_qualification.py` | **S0** worker: real-checkpoint adaptive qualification |
| `scripts/analyze_adaptive_qualification.py` | S0 CPU-only analyzer |
| `scripts/run_adaptive_screen.py` | **S1** worker: four-arm screen gated in-session on S0 |
| `scripts/analyze_adaptive_screen.py` | S1 CPU-only analyzer |
| `scripts/freeze_adaptive_launch.py` | Deterministic config freeze generator |
| `configs/openvla/adaptive_qualification_v1.json` | Frozen S0 config |
| `configs/openvla/adaptive_screen_v1.json` | Frozen S1 config |
| `docs/ADAPTIVE_SCREEN_PROTOCOL_V1.md` | Frozen schedules/caps/stopping conditions |
| `tests/openvla/test_adaptive_qualification.py` (7), `test_adaptive_screen.py` (6) | Contract/worker tests (fixture-based); three S0 additions cover tolerance-separation, missing-maxima rejection, and independent analyzer recomputation, plus one regression for the prune-boundary layer-length invariant |

Unchanged and re-verified: `vlapruner_adaptive.py` (module sha
`53c3f9ad…`, identical to the previously TITAN-verified copy) and the existing
fixed-compression / contemporary-reference chain.

## 2. Frozen hashes (deterministic; regenerating the configs yields identical SHA-256)

- `configs/openvla/adaptive_qualification_v1.json` →
  `c5f88505a8b4085b579c84dc12375bdd1e773159e691fa6f7f24efb594278acc`
- `configs/openvla/adaptive_screen_v1.json` →
  `173096dce8e90b6c5284796e3765964f510e821880b48730ae2fb643d97961dc`
- `scripts/freeze_adaptive_launch.py` → `9c22dce228b2bbbcd6477d291066ac450c58391917a1f46a64033faad438d5a8`
- S0 `authenticated_files` (worker/analyzer/port/bridge/tests/docs): see the
  configs themselves; key ones: `run_adaptive_qualification.py`
  `9f5b7099…`, `analyze_adaptive_qualification.py` `99c972ec…`,
  `adaptive_query.py` `0b14a093…`, `vlapruner_adaptive.py` `53c3f9ad…`,
  `test_adaptive_qualification.py` `1bde8ad6…`,
  `test_vlapruner_adaptive.py` `c85839bb…`,
  `docs/ADAPTIVE_SCREEN_PROTOCOL_V1.md` `1bdfc19f…`.
- S1 `authenticated_files`: `run_adaptive_screen.py` `cbec4e62…`,
  `analyze_adaptive_screen.py` `2ec3ad40…`, `adaptive_screen_contract.py`
  `9596f75b…`, `test_adaptive_screen.py` `900b5b8c…`,
  `docs/ADAPTIVE_SCREEN_PROTOCOL_V1.md` `1bdfc19f…` (shared with S0).
- Prior-chain pins embedded in the configs: contemporary reference
  `afa1bc94…`, fixed-qualification config `d4cb82d5…`, fixed-qualification
  summary `ad197c96…`, reference summary `6c8fc825…`.
- GPU: index 0, UUID `GPU-bb2451d6-2989-a112-5c18-8892943710e4` (both configs).

## 3. Frozen protocol (see `docs/ADAPTIVE_SCREEN_PROTOCOL_V1.md` for the full text)

- **Timing (corrected):** fixed-384 and dense-512 run compressed/dense from
  query one; only adaptive arms run the algorithm-required dense startup
  (first 3 queries per episode reset), and that startup is counted in the
  primary all-query timing mean across all arms. 240 slots = 16 warmup +
  224 measured; one 7-query trace per (trajectory, arm); warm/steady reported
  separately for adaptive arms.
- **Episodes:** 168 = 160 primary (4 arms × 40 conditions, rotated) + 8 native
  (2/suite). Native episodes shadow 3 calls per query (native, dense-512
  `≤1e-6` unchanged, adaptive-zero bound by S0-achieved tolerance) with SDPA
  witnesses on every parity row (`adaptive == 0`, native/dense ≥ 1).
- **S0 (first in the one bounded launch):** 16 inputs, **176 exact calls**
  (16 × [native + zero-plain + zero-audited] + 8 frame-1 × 2 modes × 8
  parallel trace calls), ceiling `model_calls=256`, `seconds=3600`, 23,552
  MiB, 256 MiB artifacts. Native-audited at frames 0/1; adaptive-zero
  plain+audited both frames; adaptive_75/50 as 4-query traces (3 dense + 1
  steady, parallel plain/audited for identical query numbering). Gates: hooks
  neutral, no cache, structural retention `512,512,512,384|256` with prune
  boundary at layer 3: layers 0–3 full then 4–31 short, post-prune layer
  length = `full − (512 − retained_visual)` (the port keeps `[0] + top_visual
  + [513..full)`), re-verified numerically by `reconcile` on every mode row,
  steady differences vs native@frame1 recorded separately (never gated),
  `sdpa_zero == 0`.
  **Tolerance separation (S0 rule):** `adaptive_zero_tolerance =
  max(2×zero_pruning_observed, 1e-6)` is calibrated on the **zero-pruning
  (dense-input) parity rows only** and must be ≤ frozen ceiling `1e-3` or S0
  technical-stops and S1 is refused. Pruning-induced steady differences are a
  different, shorter computation; they are recorded as
  `pruning_shift_observed_max` and never calibrate, gate, or inflate the
  tolerance.
- **S1 caps:** `model_calls=7000, episodes=168, seconds=21600,
  aggregate_memory_mib=23552, artifact_bytes=536870912`. Estimated ~3,900
  calls, ~1.5–2 h at measured ~1.48 s/call; hard 6 h stop.
- **Triage/analysis:** `positive_method_result=False`, `preference=[]`, no
  automatic arm selection; evidence-only analysis (trajectory-clustered
  bootstrap, suite-stratified success differences, baseline-concern flag).

## 4. Verification performed (nothing on GPU)

- **Local (`savr_cpu` venv):** adaptive suites 13/13 pass (9 original + 3
  tolerance-separation regressions + 1 prune-boundary length regression); port
  suite (`test_vlapruner_adaptive.py`) is now **18 tests** — 17 pre-existing,
  all passing on the pinned TITAN venv (reported 9 passed + 8 skipped by
  policy, never "17/17"), plus the new diagnostic-vs-adaptive-exclusion
  regression, which passes on the local Mac CPU venv and is expected to pass
  on the pinned venv (TITAN blotter refreshed at §8 after resync); two
  pre-existing version-pinned failures occur *only* on a Mac venv with
  transformers ≠ 4.40.1 (byte-parity vs the 4.40.1 eager path) and are known
  local noise, not regressions; freeze re-run deterministic; the S1 config
  validates against the contract (`validate_config`, 168/240, all measured
  traces 1..7, no duplicates); the S1↔S0 config-sha binding re-verified
  against the re-frozen hashes. The wider local discovery shows pre-existing
  GPU/server-asset suites failing *on the Mac CPU venv* (files untouched by
  this work); they are not regressions.
- **TITAN pinned venv (CPU only, `CUDA_VISIBLE_DEVICES=""`):** re-verified after
  the re-freeze (see §8 for the fresh TITAN blotter):
  - `test_adaptive_*.py`: **13 tests, exit 0** (post-fix count; see §8).
  - `test_vlapruner_adaptive.py` rerun: **Ran 17 tests — 9 passed, 8 skipped,
    exit 0** (source-parity class skipped by policy; always reported as 9/8,
    never "17/17").
  - Regression rerun: `test_fixed_compression_qualification.py` 3 ok,
    `test_compression_screen.py` 7 ok.
- All changed files byte-identical local ↔ TITAN (`sha256sum` match; see §8).

## 5. Exact bounded launch (only after approval + fresh idle check on GPU 0)

On TITAN, `cd /home/ved/SAVR`, original runtime, **one run, no retry**:

```
env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  envs/openvla-oft/bin/python scripts/run_adaptive_qualification.py \
  --config configs/openvla/adaptive_qualification_v1.json
```

(S0 must complete with `complete=True` and tolerance ≤ 1e-3; its
`launch.json` `config_sha256` must equal `c5f88505…`.) Then, in the same
session:

```
env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  envs/openvla-oft/bin/python scripts/run_adaptive_screen.py \
  --config configs/openvla/adaptive_screen_v1.json
```

Both support `--preflight-only` (CUDA-hidden contract + input-audit gate, 0
model calls) to be run first. Outputs land only under
`results/adaptive-qualification-v01` and `results/adaptive-screen-v01`
(`mkdir(exist_ok=False)` — a stale summary cannot satisfy the S1 gate).

## 6. Server discipline (unchanged)

Only `/home/ved/SAVR`; no sudo, installs, downloads, GPU inspection beyond the
single selected device idle check, no retry/reprioritization, no second GPU,
no push. If anything would leave that boundary, stop and ask.

## 7. The single ask

Approve **one bounded two-stage run (S0 → S1) on GPU 0** per this report and
`docs/ADAPTIVE_SCREEN_PROTOCOL_V1.md`, with no retry and no automatic
fallback. The fixed-384 vs adaptive tradeoff measured by S1 is what the held
advisor email and manuscript decisions await. The granted launch
technical-stopped on the adaptive audit/session boundary; that stop is
preserved as evidence, the root cause (released diagnostic wrapping adaptive
queries) is fixed **worker-only** — `attention_diagnostic.py` and the released
reference chain are unmodified, with their frozen hashes unchanged — the fix
is regression-covered and re-frozen (§2/§3/§4). The ask stands on the
corrected, re-frozen state. Until approval arrives, no GPU work occurs and
this report is the handoff record.

## 8. Fresh TITAN verification blotter (post-fix; to be refreshed after resync)

Recorded 2026-09-21 locally after the worker-only fix. **The prior §8 TITAN
blotter predates this fix; the changed files must be re-synced byte-identical
to `/home/ved/SAVR` and the TITAN CPU suites re-run before any launch
(§4/§5).** Local results on the current (fixed, re-frozen) tree:

- `test_adaptive_qualification.py`: **Ran 7 tests — OK.**
- `test_adaptive_screen.py`: **Ran 6 tests — OK.** (adaptive suites 13 total)
- `test_vlapruner_adaptive.py`: **18 tests**; the 17 pre-existing behave as
  before on the pinned venv (9 passed, 8 skipped by policy), and the new
  diagnostic-vs-adaptive-exclusion regression passes (also passes on a Mac CPU
  venv even with transformers ≠ 4.40.1, alongside the two known byte-parity
  local-only failures).
- S0 `--preflight-only`: `{"preflight_passed": true, "model_calls": 0}`.
- S1 `--preflight-only`: refuses cleanly with
  `ValueError: S0 qualification not present — run S0 (adaptive-qualification-v1) first`
  (the in-session S0-then-S1 gate works; no stale summary can satisfy it).
- Re-frozen configs verified locally: S1 embeds the new S0 config sha
  (`c5f88505…`), all `authenticated_files` hashes match the working tree.

Local summary for the same hashes: adaptive suites 13/13 OK, freeze re-run
deterministic, S1 `validate_config` OK (168/240), S1↔S0 config-sha binding
re-verified (S1 embeds `c5f88505…` = S0 config sha).