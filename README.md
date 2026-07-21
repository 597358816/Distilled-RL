
# Distilled Reinforcement Learning for LLM Post-Training

<p align="center">
  <a href="https://arxiv.org/abs/2607.17247">
    <img src="https://img.shields.io/badge/arXiv-2607.17247-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://github.com/597358816/Distilled-RL">
    <img src="https://img.shields.io/badge/Code-GitHub-black.svg" alt="GitHub">
  </a>
</p>

This repository contains the implementation of **Distilled Reinforcement Learning (Distilled RL)**, a unified post-training framework that integrates teacher supervision directly into the reinforcement learning objective.

Standard reinforcement learning relies on coarse-grained outcome rewards, while on-policy distillation usually encourages the student to imitate the teacher distribution unconditionally. Distilled RL instead uses the teacher to redistribute the policy-gradient signal at the token level, providing selective and fine-grained guidance while preserving reward-driven optimization.

## Overview

<p align="center">
  <img src="overview.png" width="95%" alt="Overview of Distilled RL">
</p>

Distilled RL consists of three components:

1. **Reverse importance sampling with clipping**, which measures the teacher's relative preference for each student-generated token.
2. **Negative sample reset**, which disables teacher reweighting on negative-advantage trajectories.
3. **Sequence-level geometric normalization**, which removes sequence-level scale bias while preserving relative token preferences.

## Method

Given a prompt $q$ and a response $o_i$ sampled from the old student policy, the standard policy ratio is

```math
r_{i,t}(\theta)
=
\frac{
\pi_{\theta}(o_{i,t} \mid q, o_{i,1:t-1})
}{
\pi_{\mathrm{old}}(o_{i,t} \mid q, o_{i,1:t-1})
}
```

The response-level advantage is estimated using group-normalized rewards:

```math
A_i
=
\frac{
R_i - \mathrm{mean}(\{R_j\}_{j=1}^{G})
}{
\mathrm{std}(\{R_j\}_{j=1}^{G})
}.
```

### Reverse Importance Sampling

We measure the teacher's relative preference for each student-generated token using

```math
\rho_{i,t}
=
\frac{
\pi_{\mathrm{teacher}}(o_{i,t} \mid q, o_{i,1:t-1})
}{
\pi_{\theta_{\mathrm{old}}}(o_{i,t} \mid q, o_{i,1:t-1})
}.
```

To prevent extreme teacher–student likelihood ratios, we apply symmetric clipping:

```math
\bar{\rho}_{i,t}
=
\mathrm{clip}
\left(
\rho_{i,t},
\epsilon_{\rho}^{-1},
\epsilon_{\rho}
\right).
```

### Sequence-Level Geometric Normalization

The clipped ratios are normalized within each response:

```math
\widetilde{\rho}_{i,t}
=
\frac{
\bar{\rho}_{i,t}
}{
\exp
\left(
\frac{1}{|o_i|}
\sum_{s=1}^{|o_i|}
\log \bar{\rho}_{i,s}
\right)
}.
```

The normalized ratios satisfy

```math
\left(
\prod_{t=1}^{|o_i|}
\widetilde{\rho}_{i,t}
\right)^{1/|o_i|}
=
1.
```

This normalization removes the sequence-level mean shift in log importance ratios while preserving the teacher's relative preferences across tokens.

### Negative Sample Reset

Teacher guidance is applied only to positive-advantage responses:

```math
w_{i,t}
=
\begin{cases}
\widetilde{\rho}_{i,t}, & A_i > 0, \\
1, & A_i \leq 0.
\end{cases}
```

For negative-advantage responses, the update reduces to the original RL objective.

### Distilled RL Objective

For responses sampled from the old student policy, the final policy optimization objective is

```math
\mathcal{J}_{\mathrm{DistilledRL}}(\theta)
=
\mathbb{E}
\left[
\frac{1}{G}
\sum_{i=1}^{G}
\frac{1}{|o_i|}
\sum_{t=1}^{|o_i|}
\min
\left(
r_{i,t}(\theta) w_{i,t} A_i,
\hat{r}_{i,t}(\theta) w_{i,t} A_i
\right)
\right],
```

where the clipped policy ratio is

```math
\hat{r}_{i,t}(\theta)
=
\mathrm{clip}
\left(
r_{i,t}(\theta),
1-\epsilon_{\mathrm{low}},
1+\epsilon_{\mathrm{high}}
\right).
```

Unlike KL-based on-policy distillation, Distilled RL does not treat the teacher as an unconditional imitation target. Instead, the teacher selectively redistributes the reward-driven policy-gradient signal at the token level.

## Main Results

We evaluate Distilled RL on three student models using Qwen3-8B-GRPO as the teacher. The table below reports the average Pass@1 over ten mathematical reasoning benchmarks.

| Student Model | Base | OPD | RL | OPD+RL | Distilled RL |
|---|---:|---:|---:|---:|---:|
| DeepSeek-R1-Distill-Qwen-1.5B | 31.70 | 35.27 | 36.86 | 36.54 | **40.00** |
| Qwen3-1.7B | 39.86 | 45.21 | 44.76 | 44.89 | **46.37** |
| Qwen3-4B | 46.33 | 55.97 | 57.40 | 56.38 | **58.96** |

Distilled RL consistently improves over standard RL, OPD, and their direct combination across different student scales and teacher–student settings.

## Training Setup

- **Teacher:** Qwen3-8B-GRPO
- **Students:**
  - DeepSeek-R1-Distill-Qwen-1.5B
  - Qwen3-1.7B
  - Qwen3-4B
- **Training dataset:** DAPO-17K
- **RL algorithm:** GRPO
- **Rollout group size:** 8
- **Maximum response length:** 8192
- **Teacher ratio clipping threshold:** \(\epsilon_\rho=3\)

The implementation is built on top of [EasyR1](https://github.com/hiyouga/EasyR1) and [VeRL](https://github.com/volcengine/verl).

## Requirements

### Software

Clone the repository:

```bash
git clone https://github.com/597358816/Distilled-RL.git
cd Distilled-RL
```

Install the required dependencies:

```bash
pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 vllm==0.8.3 transformers==4.51.2
pip install ray==2.48.0 tensordict==0.9.1 pydantic==2.11.7
pip install flash-attn
pip install -e .
pip install tensorboard
```
