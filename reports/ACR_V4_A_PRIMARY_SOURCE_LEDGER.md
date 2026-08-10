# ACR V4-A Primary-Source Ledger

Date: 2026-08-10

| Source | Supported relevance to V4-A | Boundary |
|---|---|---|
| [OpenVLA](https://arxiv.org/abs/2406.09246) | Confirms the base VLA and fused DINOv2/SigLIP visual architecture used by the pinned project stack. | Does not establish that this project's cache is safe or fast. |
| [VLA-Cache](https://arxiv.org/abs/2502.02175) and [official code](https://github.com/siyuhsu/vla-cache) | Establishes training-free adaptive visual-token caching as a relevant efficiency direction. | Its cache design and evaluation do not validate ACR's camera-block controller. |
| [DepthCache](https://arxiv.org/abs/2603.10469) | Shows recent training-free VLA visual-token reuse/merging work and motivates comparing against broader visual redundancy methods. | Changes token computation and is not evidence for the frozen ACR executor. |
| [VLN-Cache](https://arxiv.org/abs/2603.07080) | Supports the general relevance of dynamics- and saliency-aware cache vetoes. | Navigation setting and method differ from LIBERO manipulation and ACR. |
| [EfficientVLA](https://arxiv.org/abs/2506.10100) | Provides context on training-free pruning/caching/compression for VLA inference. | Broader token/layer changes fall outside V4-A's frozen method boundary. |
| [PyTorch `torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html) | Documents the `reduce-overhead` mode and its CUDA-Graph use. | Compilation is not guaranteed to graph every query or deliver a particular speedup. |
| [PyTorch CUDA semantics](https://docs.pytorch.org/docs/main/notes/cuda.html) | Documents static-shape/control and CPU-overhead considerations for CUDA Graphs. | Source compatibility still requires project-specific correctness and timing tests. |
| [NVIDIA CUDA Graph constraints](https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/constraints.html) and [PyTorch best practices](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/best-practices.html) | Supports the audit requirements for static buffers, fixed control, asynchronous GPU work, and avoiding capture-time CPU synchronization. | The audit establishes only a testable boundary, not capture success or performance. |

## Synthesis

Primary work supports visual redundancy reuse as an active efficiency route,
but it does not rescue the frozen V4-A controller gate. Official CUDA-Graph
guidance supports testing a project-owned fixed-shape GPU core while keeping
CPU preprocessing, controller logic, validation, and eager fallback outside
capture. That executor remains a hypothesis because V4-A authorized no GPU,
model query, implementation, or timing measurement.
