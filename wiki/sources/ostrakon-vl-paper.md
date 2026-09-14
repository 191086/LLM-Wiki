---
type: source
title: Ostrakon-VL：面向餐饮零售的领域多模态大模型（论文）
created: 2026-09-14
updated: 2026-09-14
tags:
  - paper
  - mllm
  - domain-model
  - data-curation
  - benchmark
sources: 1
raw: raw/Ostrakon-VL.pdf
---

# Ostrakon-VL: Towards Domain-Expert MLLM for Food-Service and Retail Stores

> 原文：[[raw/Ostrakon-VL.pdf]] ｜ 类型：论文（arXiv:2601.21342v1）｜ 日期：2026-01-29 ｜ 机构：Rajax Network Technology（淘宝闪购 / 阿里）｜ 代码：github.com/Ostrakon-VL/Ostrakon-VL

## 一句话总结

基于 Qwen3-VL-8B 全参微调的餐饮零售（FSRS）领域多模态模型「三件套」：模型 [[ostrakon-vl|Ostrakon-VL]] + 首个 FSRS 基准 [[shopbench|ShopBench]] + 数据清洗管线 [[quad-data-curation|QUAD]]（69.25M→3.40M、压缩 20.4× 反涨 2.5 分），让 8B 模型在 ShopBench（60.1）反超 30 倍大的 Qwen3-VL-235B（59.4）。

## 关键要点

- **动机**：通用 MLLM 在 FSRS 场景（监管巡检、低分辨率监控、随手拍摄）存在系统级错位——能力层（不懂领域视觉语义与退化输入）、数据层（噪声大、无审计闭环）、评测层（无统一细粒度基准）三重失配（§1）
- **[[quad-data-curation|QUAD]]**：四阶段蒸馏 69.25M→3.40M（20.4×），ShopBench 56.7→59.2（+2.5）；核心论点：高信噪比 + 校准的能力覆盖比数据量重要（§3、§6.3.1）
- **训练策略**：Caption Bootstrapping（领域知识注入）→ 离线课程学习 OCL（参考模型投票定难度分层；视频独立成阶段）→ [[mixed-preference-optimization|MPO]]；消融显示 CB+OCL 贡献最大，三者全开 60.1（§4、§6.3.2）
- **[[shopbench|ShopBench]]**：5,818 题 / L1-L4 四层分类 / 单图·多图·视频；每图平均实例数 13.0 为八个主流基准最高；提出 VNR/VIF 指标分解传统 Multimodal Gain（§5）
- **结果**：开源阵营第二（次于 InternVL3.5-241B 的 61.1）；代价是通用基准 72.4→66.7（MMVet 掉到 36.4），但场景相关的中文 OCRBench 88.5 基本保住（§6.1–6.2）
- **训练配置**：full FT 1 epoch、AdamW lr 2e-6（cosine + 2% warmup）、global batch 144、BF16 + ZeRO-2、max seq 8192、图像 602,112 px、视频 2fps ≤128 帧（附录 C）

## 值得追踪的实体与概念

- [[ostrakon-vl]]（模型）｜ [[shopbench]]（基准）
- [[quad-data-curation]] ｜ [[mixed-preference-optimization]] ｜ [[domain-specific-mllm]]
- [[qwen]]（基座 Qwen3-VL-8B 的提供方）
