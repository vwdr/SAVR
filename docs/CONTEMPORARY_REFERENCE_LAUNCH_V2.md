# Contemporaneous-reference evaluation v2 launch

Authorized by the user's 2026-09-13 continuation after the completed recovery.
This is the existing 88-episode design, not a new method or a changed population.

Configuration: `configs/openvla/contemporary_reference_evaluation_v2.json`.
SHA-256: `afa1bc948880b072b74c9f150e7dbab77fe90468040746c73b82af603f91ec67`.
Worker: `scripts/run_contemporary_reference_recovery01.py`, unchanged since the
80-call completed hook qualification. The frozen config authenticates that
summary, actual-frame CPU preflight, sources, checkpoint and input identities.
Analyzer: `scripts/analyze_contemporary_reference_v2.py`.
Output: `results/contemporary-reference-v02` (exclusive creation).

Forty previously consumed conditions each receive dense and SpecPrune episodes,
with balanced arm order. Eight bracketing native episodes include dense shadows
on the same observation; they are not extra primary tasks or primary timing
samples. A separate 136-call controlled timing module includes eight labeled
warm-ups and 128 measured calls. No new method training occurs in this run.

Prelaunch checks: 96 targeted CPU tests, actual released preprocessing on all
16 frozen stored frames, completed 80-call/24-check hook qualification, evaluation-
mode config verification, synthetic v2 analyzer reconciliation, source/checkpoint/
input preflight and freshly idle aggregate selected-GPU telemetry. The worker
repeats input/hash/availability checks before model loading.

Caps: one selected GPU, 6,000 total calls including native shadows and warm-ups,
88 episodes, six hours, aggregate GPU memory <23,552 MiB, artifacts <512 MiB.
No exclusions, outcome-based retries, threshold changes or early scientific
decisions. Exceptions and exceeded caps remain technical stops, not task failures.

While active inspect only owned process health, progress/episode/timing record
counts, artifact bytes, elapsed time and aggregate GPU telemetry. Do not inspect
partial successes, individual latency values, parity outcomes or aggregate results.
On completed immutable worker summary, verify exact 88/136 counts and all hashes,
then run the CPU analyzer with CUDA hidden. Report all conditions and native
repeatability flags; this is development characterization, not corrector efficacy.
If stopped early, inspect technical_stop.json/technical_traceback.log only, preserve
all evidence and stop without retry. If launch fails before a run directory exists,
inspect only the dedicated technical terminal log. No model restart without approval.

The quiet local completion monitor should notify only on completion, failure or
required user action. It must sync reviewed evidence/status to the local SAVR
folder and delete itself once handled. Do not automatically launch the compression
screen or train the corrector after this evaluation. No GitHub push, unrelated
server activity or edits to historical evidence. Server work only via `ssh titan`
and inside `/home/ved/SAVR`.
