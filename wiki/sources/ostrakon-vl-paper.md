---
type: source
title: Ostrakon-VL：面向餐饮零售的领域多模态大模型（论文）
created: 2026-09-14
updated: 2026-09-16
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

> **收录注记**：全文公式号按原文（式 1–16）：式 1–7 为 QUAD 数据清洗（§3）、式 8–11 为 OCL 难度分层（§4）、式 12 为 MPO 目标（§4.3）、式 13–16 为 ShopBench 指标 MG/ML 与 VNR/VIF（§5）。式 12 的三项子式论文只给引注不给公式（DPO 项即 [[dpo-paper]] 式 7，另两项见 [[mixed-preference-optimization]] §2.2）。

## 一句话总结

基于 Qwen3-VL-8B 全参微调的餐饮零售（FSRS）领域多模态模型「三件套」：模型 [[ostrakon-vl|Ostrakon-VL]] + 首个 FSRS 基准 [[shopbench|ShopBench]] + 数据清洗管线 [[quad-data-curation|QUAD]]（69.25M→3.40M、压缩 20.4× 反涨 2.5 分），让 8B 模型在 ShopBench（60.1）反超 30 倍大的 Qwen3-VL-235B（59.4）。

## 关键要点

- **动机**：通用 MLLM 在 FSRS 场景（监管巡检、低分辨率监控、随手拍摄）存在系统级错位——能力层（不懂领域视觉语义与退化输入）、数据层（噪声大、无审计闭环）、评测层（无统一细粒度基准）三重失配（§1）
- **[[quad-data-curation|QUAD]]**（§3，式 1–7）：四阶段蒸馏 69.25M→3.40M（20.4×），ShopBench 56.7→59.2（+2.5）；核心论点：高信噪比 + 校准的能力覆盖比数据量重要（§3、§6.3.1）。公式锚点：打分 $r_a = R_\phi(I, q, a)$（式 1）、盲答 $\bar a = G_\theta(q)$ 及其打分 $r_{\bar a} = R_\phi(I, q, \bar a)$（式 2–3）、质量过滤联合保留准则 $(I, q, a) \in D_1 \iff r_a \ge \tau \,\land\, (r_a - r_{\bar a}) \ge \tau_{\bar a}$（式 4）、基座参考回答打分 $r_{\tilde a} = R_\phi(I, q, \tilde a)$（式 5）、基座参考过滤 $(I, q, a) \in D_2 \iff (I, q, a) \in D_1 \,\land\, \Delta r \ge \tau_{\tilde a}$（式 6，$\Delta r = r_a - r_{\tilde a}$）、能力分类 $c = \mathcal{M}_\mathcal{C}(q)$（式 7）
- **训练策略**（§4，式 8–12）：Caption Bootstrapping（领域知识注入）→ 离线课程学习 OCL（参考模型投票定难度分层；视频独立成阶段）→ [[mixed-preference-optimization|MPO]]；消融显示 CB+OCL 贡献最大，三者全开 60.1（§4、§6.3.2）。公式锚点：OCL 中 K 个参考模型各自生成 $\hat a_k$ 打分（式 8）、逐模型奖励差 $\Delta r_k^{\hat a} = r_k^{\hat a} - r_a$（式 9）、难度投票 $s = \sum_{k=1}^{K}\mathbb{1}\big[\Delta r_k^{\hat a} > \tau_{cl}\big]$（式 10）、分层映射 $\mathrm{Tier}(I, q, a) = f(s)$，$f: \{0,\dots,K\} \to \{1,\dots,n\}$ 由易到难（式 11）、MPO 目标 $\mathcal{L}_\text{MPO} = w_1\mathcal{L}_\text{preference} + w_2\mathcal{L}_\text{quality} + w_3\mathcal{L}_\text{generation}$（式 12）
- **[[shopbench|ShopBench]]**（§5，式 13–16）：5,818 题 / L1-L4 四层分类 / 单图·多图·视频；每图平均实例数 13.0 为八个主流基准最高；提出 VNR/VIF 指标分解传统 Multimodal Gain——$\mathrm{MG} = |S_V|/N - |S_{-V}|/N$、$\mathrm{ML} = \max(0,\ |S_{-V}|/N - |S_T|/N)$（式 13–14），$\mathrm{VNR} = |S_V \setminus S_{-V}|\,/\,|S_V|$、$\mathrm{VIF} = |F_V \cap S_{-V}|\,/\,|F_V|$（式 15–16）
- **结果**：开源阵营第二（次于 InternVL3.5-241B 的 61.1）；代价是通用基准 72.4→66.7（MMVet 掉到 36.4），但场景相关的中文 OCRBench 88.5 基本保住（§6.1–6.2）
- **训练配置**：full FT 1 epoch、AdamW lr 2e-6（cosine + 2% warmup）、global batch 144、BF16 + ZeRO-2、max seq 8192、图像 602,112 px、视频 2fps ≤128 帧（附录 C）

## 值得追踪的实体与概念

- [[ostrakon-vl]]（模型）｜ [[shopbench]]（基准）
- [[quad-data-curation]] ｜ [[mixed-preference-optimization]] ｜ [[domain-specific-mllm]]
- [[qwen]]（基座 Qwen3-VL-8B 的提供方）
