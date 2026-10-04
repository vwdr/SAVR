# GitHub branch publication scope — October 3, 2026

The user explicitly requested publishing the current project so a new chat can
read the codebase and context without local filesystem access.

## Entry points

- Repository: https://github.com/vwdr/SAVR
- Branch: `agent/full-paper-audit` (not `main`).
- Read `SAVR_MASTER_CONTEXT.md`, `AGENTS.md`, the README's current-status section,
  and the latest result/contribution reports before acting.
- Historical status/launch entries remain for provenance. Do not launch a worker
  or infer current permission from them.

The repository was verified **public** before preparing this push. Visibility
was not changed. Publication does not send an email, create a new study,
access TITAN, or alter frozen experiment evidence.

## Included

Research source, tests, scripts, schemas, frozen configurations, protocols,
status/decision history, handoff files, master context and existing `AGENTS.md`.
Small historical paper/poster sources and review outputs are retained as archives,
not current validated claims. The README directs readers to the current context
and important qualifications.

Completed-run root-level JSON/JSONL evidence is included for:

- `results/current-frame-robot-pilot-v01`
- `results/adaptive-screen-v03`
- `results/adaptive-qualification-v03`
- `results/current-frame-adapter-fit-v01`
- `results/current-frame-feature-collection-v01` (metadata/manifests, not tensors)
- `results/current-frame-correction-coverage-diagnostic-v01` (superseded, retained)
- `results/current-frame-correction-coverage-diagnostic-v02`
- `results/current-frame-correction-benefit-signals-v01`
- `results/adaptive-zero-diagnosis-v01`
- `results/fixed-compression-screen-v01`
- `results/contemporary-reference-v02`

The outcome-free `reports/cac_c0_recovery01/simulator_populations_v1.jsonl`
is also included because the robot verifier authenticates it. Its 2,240 records
contain condition IDs, task/suite/state/seed metadata and hashes, not actions,
rewards, regret, success or calibration labels. It is a population declaration,
not evidence that all declared conditions were executed.

These records permit CPU reconciliation of the two latest robot studies without
the backbone, dataset or training feature tensors. Immutable bytes/hashes are
not edited for publication. Absolute runtime paths inside original records are
provenance, not instructions to access a university machine or proof that those
files exist in a fresh checkout.

## Excluded intentionally

- Model checkpoints and learned `.pt` weights, datasets/HDF5 files, raw feature
  tensors, and runtime caches/environments.
- `tmp/` scratch files and Git backup bundles.
- Backup `.tar.gz` files and redundant source ZIP packages (unpacked sources remain).
- Raw JSONL contracts/trajectory-role schedules under `reports/cac_c0_recovery01/`,
  including protected/locked-pool material; the outcome-free simulator population
  manifest and compact freeze/power summaries remain.
- Other ignored historical raw run directories, credentials, private keys and
  local environment secrets.

Nothing excluded was deleted locally. Some historical raw evidence exists only
on TITAN. The master guide and reports disclose its location/qualification; this
branch is not a complete dataset/weight archive or a standalone GPU environment.
Training/collection verifiers requiring excluded tensors or weights cannot be
promised to pass on the GitHub-only copy. Exact weight hashes remain documented.

## Verification

Before staging, candidate text/source/config/evidence files were scanned for
private-key blocks and common GitHub, Hugging Face, OpenAI, AWS and literal-secret
patterns without printing secret values. No credential-pattern findings were
reported. This is a bounded automated check, not a guarantee against every
possible secret format. Python source syntax and the two independent completed
robot-study verifiers were checked. A clean committed snapshot is checked before
push to confirm that required tracked evidence is sufficient.

No failed scientific criterion, frozen source/configuration, action weight,
success count, or authenticated result was changed to make this release work.
New user authorization is required for future experiments and shared resources.

## New-chat instruction

> Read the `agent/full-paper-audit` branch of https://github.com/vwdr/SAVR,
> beginning with `SAVR_MASTER_CONTEXT.md` and `AGENTS.md`. Read the linked latest
> result and contribution-audit reports. Summarize our current research checkpoint
> and unresolved publication contribution before proposing action. Do not rely
> on `main`, old running entries, or historical experiment approvals. Do not
> launch experiments, send email or change shared university resources unless
> currently authorized. Tell me if your GitHub connector cannot access this
> branch rather than assuming it can.
