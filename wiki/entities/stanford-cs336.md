---
type: entity
title: Stanford CS336
created: 2026-09-12
updated: 2026-09-15
tags:
  - course
  - stanford
sources: 1
---

# Stanford CS336

**定义**：斯坦福课程《Language Modeling from Scratch: Language Modeling from First Principles》（[spring2025](https://stanford-cs336.github.io/spring2025/)），系统讲授从零构建语言模型。

## 课程背景

> wiki 外补注（候选来源即上方课程页，2026-09-15 检索）：Tatsunori Hashimoto 与 Percy Liang 主讲，5 学分、极重实现（implementation-heavy）——带学生走完语言模型的每一环。19 讲覆盖：tokenization 与工具 → 架构与超参（含 MoE）→ GPU / Triton 内核 / 并行训练 → scaling laws → 推理 → 评测 → 数据 → 对齐（SFT/RLHF 与 RL）。五个作业各占一环：①Basics（从零实现 tokenizer、模型架构、优化器并训练最小 LM）②Systems（剖析基准、Triton 实现 FlashAttention2、分布式训练）③Scaling（组件消融 + 拟合 scaling law）④Data（Common Crawl 原始转储 → 可用预训练数据）⑤Alignment & Reasoning RL（SFT + RL 解数学题，可选实现 DPO）。

## 与本库的关联

- Assignment 1「从零实现 tokenizer」即 [[bpe-tokenization|BPE]] 的从零训练，是 [[jun-yu-tan]] 那篇优化长文的作业背景
- 作业 5 的可选 DPO 部分对应 [[dpo]]
- 课程作业驱动的实现 + 基准文化：学生在真实语料上做复杂度分析与测量

## 来源

- [[building-a-fast-bpe-tokenizer-from-scratch]]

## 相关

- [[jun-yu-tan]] ｜ [[bpe-tokenization]] ｜ [[dpo]] ｜ [[tinystories]]
