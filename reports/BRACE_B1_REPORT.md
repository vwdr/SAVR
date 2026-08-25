# BRACE-B1 Replay Harness Report

Date: 2026-08-25

Run: `brace-b1-replay-v01`

Decision: **ACCEPTED**

## Scope

B1 tested only whether complete scripted LIBERO prefixes can be reconstructed
deterministically in fresh simulator instances. It used the project-local
CPU/OSMesa stack with CUDA hidden. It did not load a VLA, install a cache,
query a model, inspect a GPU, download an asset, or evaluate a policy. B2 was
not started.

## Frozen identity

- Source revision: `e441af06fb9e8004978255b0db793ef179dd77b7`
- LIBERO revision: `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- Configuration semantic SHA-256:
  `37952f345bcaecceffe4695fecea8d48c2b67ed19d19222d64fd29eeda1dc04c`
- Summary semantic SHA-256:
  `daf140fd5cbba93dc6cfa5fed2ecf2612f8c0c1b5d39203237a596848b3f8341`
- Summary file SHA-256:
  `8428b19cd117388d523b83e77187b937a824844b8f47526e1bc9fa4418a625bc`

## Results

All frozen gates passed:

- three complete, hash-chained transcripts covered free motion, contact-rich
  dynamics, and a gripper transition;
- 18/18 fresh prefix reconstructions matched at early, middle, and late
  boundaries;
- 9/9 identical next-action probes matched between fresh replicas;
- 3/3 modified-action prefixes were rejected;
- 3/3 direct-simulator-state-only reconstructions were rejected with
  structural mismatches beyond the restoration-mode label;
- transcript completeness, ordering, version, and hash validation passed; and
- independent post-run reconciliation revalidated all transcripts, checks,
  summary hashes, counts, and resource accounting.

The scripted scenarios produced 0.0256867 m maximum end-effector displacement,
a nonempty contact signature, and 0.0605043 rad gripper-joint span. The attempt
used 27 of 30 allowed environment instances, 198 of 240 simulator steps,
701,560 bytes of artifacts, and 284.40 of 1,800 allowed seconds.

## Interpretation and boundary

B1 supports only hypothesis H1: exact simulator reconstruction through the
scripted prefix boundary is available when the full action prefix is replayed.
It also confirms that setting simulator state alone is insufficient. This is
infrastructure evidence, not a positive BRACE method result and not evidence
that caching is accurate, fast, predictable, or competitive.

B2 is now the next protocol-eligible phase, but it requires separate user
authorization. No work outside `/home/ved/SAVR` was modified on TITAN.
