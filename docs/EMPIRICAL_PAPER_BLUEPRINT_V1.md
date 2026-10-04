# Current-frame visual compression: evidence-led paper blueprint

2026-09-26. User approved the fastest defensible empirical-paper route. This is
a writing/contribution specification, not a claim that publication is assured.
No new algorithm is proposed. No advisor email is to be prepared or sent now.

## Working title and central question

**Current-Frame Visual Compression in OpenVLA-OFT: A Controlled Study of
Inference Cost and Closed-Loop Performance**

Question: How do fixed spatial subsampling and a backend-aligned adaptive
selector compare when task success, selection overhead, startup and complete
query latency are measured together on the same deployment?

The useful output is a reproducible account of the tradeoff and its measurement
conditions. It is not the claim that visual pruning, static subsampling or
separate attention-score extraction is new. Whether the empirical findings add
enough knowledge for a specific venue remains an editorial/research judgment.
More episodes improve evidence; they do not create novelty.

## Claim-to-evidence map

| Claim | Available evidence | Permitted wording / missing evidence |
|---|---|---|
| Current-frame compression can accelerate this deployed policy | Verified four-arm development screen: fixed384 and adaptive256 reduce controlled mean query time by 15.36% and 15.58% | Positive measured acceleration on the tested inputs; not universal speedup |
| High observed robot success can coexist with acceleration | Dense, fixed384 and adaptive256 each 39/40 on matched development conditions | Equal observed counts, NOT proven equal population success; independent states and nondegenerate uncertainty needed |
| Fixed selection has less cost than adaptive selection at equal retained-token count | Fixed384 1018.71 ms vs adaptive384 1122.29 ms in the same controlled test | Equal token count is not equal FLOPs: selection depth and scoring also differ; do not attribute the entire difference to one cause |
| Fixed selection is the best method | Contradicted by adaptive256's lower measured latency with the same observed success pattern | Do not make this claim or omit adaptive256 |
| Startup changes how latency should be interpreted | Adaptive256: 1205.25 ms startup, 874.28 ms steady, 1016.12 ms combined in a seven-query test | Clearly label synthetic seven-query weighting; actual development episodes all required at least ten queries |
| Backend alignment fixes the tested integration mismatch | 64-call diagnostic and subsequent 176-call qualification with exact zero-pruning commands | Tested engineering property; not a new method and not evidence of inferior upstream robot success |
| Learned correction makes the robot better | Earlier visual-corrector pilot: 117/120, equal to uncorrected compression and below action-only 118/120 | Unsupported. Report as a separate negative transfer observation if included; offline action improvement is not robot improvement |
| The result generalizes broadly | One checkpoint, GPU class and benchmark family; exposed development states | Unsupported; new states still do not establish new-task, new-model or real-robot generalization |

Source of the current comparison: `results/adaptive-screen-v03`, authenticated
by `reports/ADAPTIVE_SCREEN_V03_RESULT.md` and the independent verifier. Reverified
2026-09-26: 168 episodes, 240 timing rows, 3728 calls; all frozen checks reconcile.
Use full hashes from that report, not newly invented provenance.

## Paper structure and figure specification

1. **Introduction.** Explain inference cost, then distinguish compressing the
   current observation from reusing old observations. State the empirical
   question and limited contribution directly. Do not retell chat history.
2. **Related work.** Credit OpenVLA-OFT, LIBERO, VLA-Pruner and relevant
   token-reduction work. Credit SparseVLM/related fast-attention compatibility
   where describing the implementation. Cite only sources actually used.
   Existing contribution and comparator audits contain the verified sources.
3. **Compared policies.** Define dense512, fixed384, adaptive384 and adaptive256.
   Give token location/index handling, selection depth, history/reset rule,
   three-query adaptive startup, unchanged checkpoint/head and action queue.
   Label the adaptive path a local backend-aligned selection adaptation, not an
   exact numerical/timing reproduction of the released publication.
4. **Experimental design.** Separate development from prospective validation.
   State checkpoint, software, GPU, all task/state IDs, seeds, arm order,
   action horizons, controls, timing boundaries and exclusions. Separate
   algorithm startup from hardware warmup. Explain uncertainty units.
5. **Results.** Report all four arms. Present success and latency together;
   do not lead with one favorable percentage and hide quality costs. Keep
   independent validation tables empty until the fixed experiment completes.
6. **Discussion and limitations.** Explain when simpler selection is useful
   and where the stronger adaptive setting is better, only as supported.
   Discuss the offline-to-robot correction gap separately, without pooling
   populations or claiming a diagnosed causal mechanism.
7. **Conclusion and reproducibility.** Summarize measured scope, not a general
   verdict on VLA compression. Include artifact access and adaptation details.

Figures should answer questions rather than decorate the paper:

- **Policy diagram:** both current cameras → projected tokens → fixed or
  adaptive selection → unchanged policy/head → eight-step action chunk → robot.
  Mark where tokens are removed and where adaptive history is reset.
- **Joint success–latency plot:** all four arms, uncertainty, and separate
  development/validation panels. No connected frontier implying untested points.
- **Timing decomposition:** startup and steady query latency with the exact
  mixture weights beside the overall result; natural-rollout costs separately.
- **Optional correction transfer panel:** paired offline and robot comparisons
  from the earlier study, with its different population labelled explicitly.

Use field terminology, define every acronym, and omit internal phase names from
the main scientific narrative. No unmeasured safety, FLOPs or memory-savings
claims. Aggregate GPU memory is a resource check, not per-arm memory reduction.

## Completion rules

Draft background/methods from existing evidence; fill independent results only
after verification. Do not rewrite old frozen manuscripts or results. Before
submission, audit every abstract sentence against an evidence row, verify all
citations and comparison identities, and inspect the compiled figures/tables.
The paper may contain both positive acceleration and negative findings. It
must not be described as a new successful method unless new evidence actually
establishes that different claim.

Next execution document: `docs/EMPIRICAL_CONFIRMATION_DESIGN_V1.md`.
