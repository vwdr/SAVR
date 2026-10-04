# CAC C1H Recovery 02 Technical Stop 03

Date: 2026-08-31  
Run: `cac-c1h-headroom-v03-recovery02`  
Classification: pre-episode integration-parity stop; not a method result

## Exact stop boundary

Recovery 02 passed model loading and the repaired official observation/return
contracts. It then stopped at the prespecified official-versus-custom dense
action parity check:

- maximum absolute action difference: `0.18178841471672058`
- allowed tolerance: `0.000001`
- episode attempts: 0
- model queries: 2
- progress records: 0
- partial terminal outcomes opened: false
- D62 anchor/reuse control reached: false
- peak aggregate selected-GPU memory: 16,437 MiB
- completed worker summary: absent
- automatic retry: false

No paired dense/D62 episode, terminal-success measurement, stage summary, or
Gate-H decision exists.

## Integrity and cleanup

Checkpoint protected bytes and inventory were restored exactly, loader backups
were removed, and GPU 0 returned to 6 MiB/0% aggregate use. Evidence is
preserved immutably at `results/cac-c1h-headroom-v03-recovery02/`.

- technical-stop file SHA-256:
  `db6e09bff9e772fa4b2ac3676624efa5b9f55e19d24bc126a790f2e94e5b9e66`
- traceback file SHA-256:
  `785fa33a4a62e9698b354381d44980024674ce36685ae7a0363c8843113deadd`
- technical-stop semantic SHA-256:
  `486fffb9aa9212de591f0e7bf105c2be2c1b2bdf492db044538fbf042f48994a`
- frozen configuration semantic SHA-256:
  `38a4e5cb5f61d91c594ba082805213fd5fde26fa84e11624aafddc18321b035b`

## Interpretation

The previous `prev_images` failure is fixed. The new stop exposes a deeper
integration discrepancy: the custom cache-capable forward path used by C1/C1H
does not reproduce the pinned released evaluator's dense action for the same
simulator observation within the frozen tolerance. C1 established internal
all-fresh reproduction inside the custom path; it did not establish equivalence
to this released evaluator entry point.

This is not evidence for or against CAC's repair effectiveness because no D62
or terminal evaluation occurred. It does mean the current C1H implementation
cannot be treated as an authenticated drop-in evaluation of the pinned policy.
The parity discrepancy must be explained before any further simulator attempt.

The authorized attempt is consumed. There is no automatic retry, and C2 remains
unauthorized.
