# Fixed current-frame compression: integration qualification passed

Date: 2026-09-13. Run: `results/fixed-compression-qualification-v01`.
User authorized the gentler-compression phase after the completed dense/SpecPrune
reference evaluation. This is a technical qualification, not a method result.

## Checks and observed result

The single GPU attempt completed all 112 model calls and all 16 stored-frame
records without a technical stop. Sixteen same-input comparisons of the new
512-token path against the qualified dense direct decoder had zero maximum
head/normalized/raw-action discrepancy and byte-identical processed commands.
All 48 plain-versus-diagnostic comparisons (16 frames x three budgets) produced
identical commands and query metadata. Both current views and state stayed unchanged.

The 384/256 paths actually shortened every one of the 32 decoder layers. Recorded
absolute rotary IDs matched the frozen selector, preserved all nonvisual states,
and used no past K/V. Every action chunk was finite. Compressed action equality
to dense was deliberately not required: its closed-loop effect is the next test.

Before launch, 53 targeted CPU tests passed, including nine new tests covering
fixed compression and corrupted qualification evidence. They checked exact dense
identity, protected/readout positions for varied prompt lengths, original
bidirectional attention, absence of cross-query persistence, actual shorter
layer inputs, preprocessing and float32 action processing. CPU source/checkpoint
and actual 128x128 stored-image schema verification passed before model loading.

Independent CPU analysis on TITAN and local reconciliation both passed. Source
and artifact hashes were verified, and frozen JSON evidence was synced locally.
Completed summary time: 2026-09-13T21:16:29.865757+00:00. Owned PID 1331329 exited.
Elapsed runtime: 179.88 seconds; peak aggregate GPU-0 memory: 15,295 MiB.
No simulator episode, teacher-data generation, adapter training or automatic retry.

## Frozen provenance

- Configuration: `configs/openvla/fixed_compression_qualification_v1.json`.
- Config SHA-256: `d4cb82d508b16a923270d4e979eced5f243b4a902d851b71772bc9702aa862b4`.
- Worker SHA-256: `41f8c6ff97655187f748b691efad11f90ed1d16829b2a1158e933c2b119ce19c`.
- Query module SHA-256: `5f685ab74c18890b0556a233ea398933e920869272f079f46e68298f29f9881a`.
- Summary SHA-256: `ad197c96e75d8c6c753d4c786c924aa49415d5e5632873ff81fcbe9a197ca5a9`.
- Analysis SHA-256: `d21c2aea4cbb5c2154986ce72fceb1d055b352a347ee9a90fd4ab6a66db74a2e`.
- GPU 0 UUID: `GPU-bb2451d6-2989-a112-5c18-8892943710e4`.
- Unchanged runtime: Torch 2.2.0+cu118 / Transformers 4.40.1, released four-suite
  OpenVLA-OFT checkpoint and its previously authenticated source and input chain.
- Caps respected: 112 calls, zero episodes, 1,800 seconds, <23,552 MiB, <256 MiB.

## Next authorized step

The 128-episode robot screen and CPU analyzer are implemented. Seven additional
screen tests passed (60 targeted tests in total across this phase), covering
schedule/count errors, task failures as data, baseline/control concerns, timing
triage, the preference for 384 and no automatic training after a failed screen.
Its executable configuration authenticates this completed qualification:
`configs/openvla/fixed_compression_screen_v1.json`, SHA-256
`cc0100823806caac8770f4bb533dda181985b6224a3275997af0793b19590ac3`.

Three fresh arms on 40 consumed conditions plus eight native controls, 204 timing
calls, no training. The full protocol and predeclared selection criteria are in
`docs/CURRENT_FRAME_COMPRESSION_SCREEN_V1.md`. A passing integration check does
not establish that compression preserves success or that a corrector will work.

All server work used ssh titan and project-local files. No unrelated university
files, processes, allocations or configuration were changed. No GitHub push.
