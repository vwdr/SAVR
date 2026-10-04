# SAVR — continuation handoff for CLAUDE

> Latest: reports/BACKEND_PRESERVATION_CONTRIBUTION_AUDIT_V1.md rejects the
> generic backend-preserving-pruning novelty claim based on primary prior art.
> No new GPU run/confirmation is active. Preserve completed results. Do not
> reinterpret an engineering fix or replication as a demonstrated new method.

> LATEST, 2026-09-26: S1 v03 COMPLETED, verified, worker exited, heartbeat deleted.
> Read reports/ADAPTIVE_SCREEN_V03_RESULT.md. Dense/fixed384/adaptive384/adaptive256
> successes39/39/38/39 of40, controlled reductions15.36%/6.76%/15.58% versus dense.
> No fixed superiority over adaptive256 established. No confirmation launched
> or protocol frozen. Do not relaunch; the active notes below are historical.

> Current checkpoint, 2026-09-26: qualification v03 PASSED and screen v03 is
> dispatched as owned TITAN PID 3074656. Do NOT launch another worker or change
> frozen code/configs. Read reports/ADAPTIVE_SDPA_V03_RUNNING_CHECKPOINT.md and
> PROJECT_STATUS.md's September 26 entry first. Heartbeat
> monitor-adaptive-comparison-v03 is active. Outcomes remain blinded until 168
> episode and 240 timing records plus complete immutable summary exist. These
> instructions supersede every historical next-step/authorization entry below.

**Prepared 2026-09-22 (local) by the outgoing agent for the incoming agent (CLAUDE).**
Read this whole file once before acting. It is the authoritative "where we are" for the
SAVR adaptive chain. The older `OPENCODE_HANDOFF.md` (557 lines) is the codex-era
historical guide — **keep it, do not delete it**; this file supersedes its "next step"
entries with the current frozen state.

**Provenance discipline that governs everything (AGENTS.md):** SAVR is a proposal, not a
validated method. Never invent results/measurements/citations/implementation status.
Distinguish hypotheses / plans / observations / measured evidence. Preserve provenance
(commands, config, code rev, model/checkpoint identity, benchmark version, seeds, hardware,
timestamps) for every experiment. Treat task success as the primary safety/performance
constraint; efficiency gains are not sufficient if success degrades. Do not modify the
manuscript or the released/reference chain unless the user explicitly asks.

---

## The single fact to internalize before anything else

The two-stage adaptive GPU chain (**S0 `adaptive-qualification-v1` → S1 `adaptive-screen-v1`**)
is **frozen and on hold**. The last real-GPU run stopped at the **worker's own technical
stop**: the adaptive arm's whole-bridge zero-SDPA witness is **unsatisfiable** on the real 7B
because the released SigLIP/vision-prep path inside that too-wide window legitimately calls
released SDPA. Root cause is **already diagnosed** and the user **approved the fix (Option A:
decoder-scoped witness)**. The fix is **NOT yet implemented**, configs are **NOT yet
re-frozen**, and **NO GPU run is authorized** until:

1. Option A is implemented **worker-only** (decoder-scoped zero-SDPA witness, symmetric with
   the released arm).
2. A regression proves satisfiability on a real released-SDPA-in-non-decoder path (the gap the
   earlier CPU suites missed).
3. Configs are re-frozen (NEW SHAs), re-synced byte-identical to TITAN.
4. The changed-chain diff is presented for **fresh launch approval by the user**.
5. **Only after that fresh approval:** one bounded GPU-0 run (deterministic seed 7, no
   auto-retry, no relaunch without approval, S1 only after S0 completes).

Copy-paste the one-line summary from §2 of the older handoff: **GPU-0 launch is on hold
pending a fresh decision. Real-GPU stop = worker's own witness (`ValueError … SDPA`), 3 model
queries, peak 16,367 MiB, 51.6 s. S1 never launched.** Evidence preserved in
`results/adaptive-qualification-v01/technical_stop.json` + `technical_traceback.log` (real-GPU
stop) and the crashed/stopped evidence dirs. **Do not delete those.**

---

## A. The technical stop — measured evidence (real-GPU, honest, read-only snapshot)

On **2026-09-21** the bounded two-stage relaunch (log `relaunch-bounded-s0s1-20260921-194455.log`,
`bounded_s0s1.nohup.out`, deterministic seed 7, `CUDA_VISIBLE_DEVICES=0`, pinned venv
`envs/openvla-oft/bin/python`, transformers 4.40.1 + torch 2.2.0+cu118, GPU-0 by unique UUID
`GPU-bb2451d6-2989-a112-5c18-8892943710e4`) passed the worker's **own** fresh idle gate
(memory ≤ 1024 MiB, util ≤ 5%), loaded the real OpenVLA-7B, ran **3 model queries**, peaked
**16,367 MiB**, and stopped at the worker's own technical stop after **51.6 s**. The canonical
stop evidence is preserved and must NEVER be deleted:
`results/adaptive-qualification-v01/technical_stop.json`:
```json
{
  "automatic_retry": false,
  "complete": false,
  "elapsed_seconds": 51.58355179056525,
  "error": "adaptive audited path must not call released SDPA",
  "error_type": "ValueError",
  "model_queries": 3,
  "peak_aggregate_gpu_memory_mib": 16367
}
```
plus the in-dir `technical_traceback.log` (full traceback) and the mirrored evidence dirs
`results/adaptive-qualification-v01-crashed-evidence/` + `-crashed-evidence` variants (keep all).
Also preserved: the 18:16Z idle-gate **refusal** record (0 model calls, no technical_stop written
— clean refusal, distinct from this stop). S1 **never launched**.

### A.1 Root cause (read-only diagnosis — the honest truth)
The adaptive arm's zero-SDPA witness was scoped to the **whole multimodal bridge**
(SigLIP vision prep + projector + decoder) via `count_sdpa()` in
`src/savr/openvla/adaptive_query.py`. On the real 7B, the released SigLIP vision-prep /
projector path legitimately calls released SDPA inside that too-wide window → a
whole-bridge zero is **unsatisfiable by construction on the real model**. The earlier
CPU/tiny-model suites "passed" only because the tiny hidden model never dispatched the real
SigLIP/prep attention path → **false all-clear**; they did NOT prove the fix on the real
model's attention wiring (`attn_implementation="sdpa"`, `LlamaSdpaAttention` in 4.40.1).
The **decoder arm itself is provably released-SDPA-free** in 4.40.1 (with
`output_attentions=True`, `LlamaSdpaAttention` falls back to the manual eager
`super().forward` path → 0 SDPA), so a **decoder-scoped** zero is satisfiable — symmetric
with the released arm's decoder-scoped 32-record diagnostic.

### A.2 The approved fix — **Option A (decoder-scoped witness, recommended)** — NOT YET IMPLEMENTED
Rescope the adaptive arm's zero-SDPA witness to the **decoder forward / `decoder.layers`**
only, symmetric with the released arm. Worker-only change. This is what CLAUDE must implement:
1. `src/savr/openvla/adaptive_query.py` — add a **decoder-scoped** zero-witness helper
   (count SDPA only inside the decoder forward, not the whole bridge), sharing the same
   counter machinery / contract semantics; keep the whole-bridge counter only for where the
   released/reference arm needs it (i.e., **don't silently weaken the released arm**).
2. `scripts/run_adaptive_qualification.py` (S0 `call()` adaptive arm) — switch the
   `require(sdpas[0] == 0, …)` witness to the decoder-scoped count, symmetric with the
   released arm's decoder-scoped contract (~line 270 area; keep the action-head capture +
   finite-values audits unchanged on both arms).
3. `scripts/run_adaptive_screen.py` (S1 `shadow()` adaptive arm) — same decoder-scoped
   witness so S1 is symmetric with S0 and satisfiable.
4. Keep the released/reference chain + released-arm diagnostic (`src/savr/openvla/attention_diagnostic.py`)
   byte-identical and untouched.

**Then (in strict order, NO GPU until past step 5):**
- Add the regression: prove the **decoder-scoped** zero witness is satisfiable even when
  released SDPA fires in the non-decoder (vision-prep) part of the bridge — the exact gap
  the earlier tiny-CPU regression missed. Add to `tests/openvla/test_adaptive_screen.py` /
  `tests/openvla/test_adaptive_qualification.py` (whichever suite covers the adaptive arm).
- Re-freeze deterministically: `envs/openvla-oft/bin/python scripts/freeze_adaptive_launch.py`
  → NEW config SHAs (S0 and S1; S1 embeds the new S0 sha). Record new SHAs.
- Re-sync to TITAN byte-identical and verify hashes.
- Run local + TITAN CPU suites (pinned venv) → must pass.
- **Present the changed-chain diff for FRESH launch approval. Do NOT launch GPU until then.**
