# [ICML 2026] *SpecPrune-VLA*: Accelerating Vision-Language-Action Models via Action-Aware Self-Speculative Pruning

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2509.05614)
[![Code](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/alexwhz-sjtu/SpecPrune-VLA#)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

**SpecPrune-VLA** is a two-level action-aware token pruning framework for accelerating Vision-Language-Action (VLA) policies while preserving task performance. The core idea is to remove redundant visual tokens by exploiting both the spatial-temporal consistency of robotic control and the hierarchical attention pattern formed inside VLA models.

![Overview](assets/overview.png)

## Introduction

Modern VLA models rely on dense visual inputs from one or multiple camera views. In typical robotic manipulation settings, visual tokens account for more than 80% of the model input sequence. However, many of these tokens correspond to static backgrounds, unchanged regions, or task-irrelevant objects. This creates substantial input redundancy and makes inference unnecessarily expensive and sloe.

### Key Insight
In this paper, we observe two key properties of VLA inference:

1. **Spatial-temporal consistency.** Consecutive inference steps in robotic control are highly correlated. The globally important visual tokens selected in one step often overlap strongly with those needed in the next step.

2. **Hierarchical attention pattern.** As the VLA model goes deeper, its visual attention evolves from broad and scattered patterns to task-relevant regions, and eventually concentrates on action-centric interaction areas. Shallow layers behave like early filters, middle layers aggregate task information, and deeper layers form compact action-centric representations.

![Insights](assets/insight.png)

Based on these observations, we propose **two-level action-aware pruning**, which combines action-level static pruning, layer-level dynamic pruning, and an action-aware controller.

## Method

### 1. Action-Level Static Pruning

SpecPrune-VLA reuses globally important visual tokens from the previous inference step as action-level draft information. These tokens provide stable action-centric cues for the current step. While shallow (3~12) layers attend to irrelevant tokens such as background, the first two layers is reliable for using text-guided attention to update local task-relevant visual evidence.

### 2. Layer-Level Dynamic Pruning

After early filtering, SpecPrune-VLA dynamically scores remaining visual tokens using action-aware importance. As deeper layers form more compact action-centric representations, redundant tokens are progressively removed through a layer-adaptive pruning schedule.

### 3. Action-Aware Controller

The action-aware controller adjusts pruning strength according to predicted action mode (fine/coarse mode). It keeps more visual tokens during precise manipulation and enables stronger pruning during less sensitive motion phases.

## Experiments

We evaluate SpecPrune-VLA across multiple VLA backbones, simulation benchmarks and real world.

On **OpenVLA-OFT** in the **LIBERO** benchmark, SpecPrune-VLA achieves up to **1.57x inference speedup** while maintaining competitive success rates. On **Dexbotic-OFT (DB-OFT)** in **SimplerEnv Visual Matching**, SpecPrune-VLA achieves **1.44x speedup**.

![Main Results](assets/exp_main.png)

On **CogACT** in **SimplerEnv Visual Matching**, SpecPrune-VLA further demonstrates strong compatibility with different VLA architectures. When combined with uniform feature caching (UFC) with a cache interval of 5 to accelerate the diffusion action expert, SpecPrune-VLA achieves **1.42x speedup**.

![CogACT Results](assets/exp_cogact.png)

In real world task, SpecPrune-VLA show larger potential and achieves 1.70x speedup on a Flexiv Rizon4 arm.
![Real Results](assets/real.png)

## Environments

Our experiments cover diverse manipulation scenes, multi-view observations, and task settings across LIBERO and SimplerEnv.

![Environments](assets/env.png)

## Installation
See `SETUP.md`
