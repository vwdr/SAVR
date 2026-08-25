# BRACE-B2 Implementation Freeze

Date: 2026-08-25

Status: Frozen before TITAN B2 execution

Governing protocol: `docs/BRACE_EXECUTION_PROTOCOL_V2_1.md`

Configuration semantic SHA-256:
`220c0b6b15253e0ecbea34010258d580cecb90f459d2d55da8bb10c01760285a`

## Scope

B2 is CPU/synthetic correctness work only. It may download pinned public source
code inside `/home/ved/SAVR/third_party`, but may not download a model,
checkpoint, or dataset; load a VLA; run a simulator; inspect or use a GPU; or
collect a policy outcome. B3 remains separately gated.

## Frozen implementation matrix

| Component | Required B2 evidence |
|---|---|
| Runtime sequence map | Derive scene, wrist, text, proprioception, action, and nonvisual positions from runtime roles; reject camera/token permutations and fixed-offset assumptions |
| Patch/source operator | Bounded L1/cosine score against each entry's actual source; known-answer zero, constant, one-zero, nonfinite, and bounds tests |
| Dense sidecar | Detached post-RoPE Q/K attention with exact scale/mask and runtime query positions; synthetic parity only; real backend/action/timing parity deferred to B3 |
| Profile mechanics | P1 wrist-refresh rule, stricter P2 wrist limits, at most three profiles per family and six total, nondecreasing budgets, nested reuse, complete suffix eligibility |
| Contract/runtime | P0--P4 identities, 1/2/4-query expiry, abort-to-FR, experimental FR lock, episode reset, P3/P4 nonselectability |
| Provenance | Immutable complete multimodal source records, per-layer/token ownership and age, exact-source drift envelopes, ring buffer that cannot evict live sources |
| Cache isolation | Exact independent clone and failure-safe restoration for both pinned DynamicCache forks, transactional model configuration, absolute-position index updates, zero cross-arm mutation |
| Branch records | Nonzero inclusion/assignment probabilities, immutable pre-outcome intent records, randomized valid arm order, duplicate-arm equality, FR fill after abort |
| VLA-Cache correction | Isolated true previous-cache-source frame history and exception propagation; no cache algorithm or configuration change |
| Comparator preflights | Authenticate revision, tree, required source, license, and configuration for VLA-ADP, VLA-Pruner, and SpecPrune-VLA; record Gated VLA-Cache's official-code disposition |

## Pinned stacks and sources

- Core OpenVLA-OFT: `e4287e94541f459edc4feabc4e181f537cd569a8`
- Core Transformers 4.40.1:
  `bc339d9ad707454c0c115970db43c260067c61ab`
- VLA-Cache: `a4909880573868dee2769343d52e793c0341678b`
- VLA-Cache Transformers 4.47.0:
  `9a90a37acacf453433168db8d7769b7ea3c40c06`
- VLA-ADP: `d7094b09a4996847772c1fa975f09d863e1b759a`
- VLA-Pruner: `84d4b7192c77abf1585610e2f12393319b7ebff9`
- SpecPrune-VLA: `8091adc4b574ce9008d49a1dc9a210f4eec314c1`
- Gated VLA-Cache: arXiv `2608.10824`; no official code repository was
  discoverable in the pre-freeze search.

## Predeclared comparator dispositions

- VLA-ADP is expected to require an isolated overlay on the core 4.40.1 stack;
  its released `pyproject.toml` contains an author-local Transformers path, so
  direct installation is not a valid reproducibility path.
- VLA-Pruner vendors a customized Transformers 4.47.0 tree and therefore
  requires an isolated stack rather than replacing the proven core stack.
- SpecPrune-VLA's README advertises MIT and links `LICENSE`, but the pinned
  repository has no top-level `LICENSE`. Its `openvla-oft/LICENSE` is the
  upstream OpenVLA-OFT MIT notice; it does not unambiguously license every
  method-specific addition. Record a missing-upstream-license disposition.
- Gated VLA-Cache is a paper-only comparator at this freeze. A later matched
  reproduction must be labeled as such and cannot be called official code.

These dispositions are not hidden failures and do not convert a paper-only or
license-ambiguous comparator into an executable result. Protocol V2.1 permits
individually reviewed technical exclusions at the later comparison gate.

## Resource and next-phase boundary

B2 is capped at 1 GiB of comparator source, 16 MiB of run artifacts, and 1,800
seconds. The post-B2 B3 proposal is capped at 480 balanced real-model queries,
one separately authorized GPU, zero simulator outcomes, and 23 GiB peak memory.
This document does not authorize B3.
