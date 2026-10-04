# OpenVLA D62 S4 Recovery 01 Protocol

Date: 2026-09-01  
Parent attempt: `openvla-d62-requalification-s4-v01`  
Purpose: repair only the compact hidden-position integration boundary

## Frozen diagnosis

The parent attempt stopped on its first recursive reuse transition because the
pinned VLA-Cache fork compacted reused visual positions while the canonical
action selector required a dense full-length hidden sequence. The fork returns
the surviving absolute `cache_position` vector. Recovery 01 must use that
explicit mapping; it may not infer positions from sequence length or use a
negative tail slice.

## Allowed correction

For every custom cached forward:

1. obtain the final absolute-position vector returned by the pinned fork;
2. prove it is rank one and aligned one-to-one with the final hidden sequence;
3. prove it is sorted, unique, nonnegative, and within the full sequence;
4. prove all 56 canonical action-readout positions remain present;
5. map each canonical position to its compact tensor offset; and
6. pass those 56 states, in official order, to the unchanged action head.

Dense output retains its structural full-length contract. The old
`[-57:-1]` convention remains prohibited.

## Mandatory pre-model qualification

Before loading the OpenVLA checkpoint, a tiny randomly initialized Llama from
the exact pinned VLA-Cache environment must run on the selected GPU:

- one dense anchor producing a cache;
- one cache-reuse call with visual-position pruning enabled;
- direct verification that active hidden length decreases;
- direct verification that the returned position vector exactly matches the
  active hidden length and is sorted/unique;
- proof that deliberately designated action positions survive;
- exact sentinel comparison between position-mapped selection and expected
  action positions;
- rejection tests for missing action positions, duplicate maps, and mismatched
  map length; and
- zero checkpoint loads, simulator calls, outcomes, expert actions, or raw
  policy actions.

The qualification writes one immutable summary and must return the selected
GPU to its pre-run aggregate memory state. Failure consumes the recovery and
must not launch OpenVLA.

## Frozen S4 attempt

If and only if the qualification passes, run the unchanged S4 design from a
new immutable root:

- 8 suite-balanced selection audits;
- 8 official/all-fresh controls;
- 2 identical recursive cycles over ages 1--4;
- 1 official plus 2 custom reset controls;
- exactly 37 planned model calls, hard cap 48;
- one selected GPU and one model process;
- 30-minute wall cap and 23,552 MiB strict peak-memory cap;
- no simulator, terminal outcome, expert action, training, downloads, or raw
  action persistence; and
- exact checkpoint restoration and no automatic retry.

The D62 profile, observations, order, seeds, selection-materiality rule,
parity tolerance, cache/source gates, and corrected candidate identity remain
unchanged from v01.

## Recovery exit gate

Recovery 01 passes only if:

- the tiny-model compact-position qualification passes;
- all eight official/all-fresh controls pass;
- both age-1--4 recursive cycles are exact and mutually deterministic;
- cache provenance, clone isolation, parent immutability, source age, reset,
  and memory gates pass;
- selection materiality is reported from the complete population; and
- one complete immutable worker summary and evidence manifest reconcile.

Any failure is a technical or scientific stop as appropriate. No result may be
inferred from partial call progress. Stop before S5/C1 regardless of outcome.

## Authorization boundary

Protocol construction, source repair, CUDA-hidden tests, and readiness review
are permitted. Selecting a GPU for the tiny-model qualification and launching
the new OpenVLA attempt require one explicit Recovery 01 approval. That
approval authorizes one qualification and, only if it passes, one S4 model
attempt. It does not authorize an additional retry or any later stage.
