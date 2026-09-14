---
type: entity
title: Ostrakon-VL（模型）
created: 2026-09-14
updated: 2026-09-14
tags:
  - mllm
  - domain-model
  - alibaba
sources: 1
---

# Ostrakon-VL

**定义**：淘宝闪购（Rajax Network Technology，阿里）发布的餐饮零售领域（FSRS）多模态大模型，基于 [[qwen|Qwen3-VL-8B]] 全参微调，是其论文三件套（模型 / [[shopbench|ShopBench]] 基准 / [[quad-data-curation|QUAD]] 数据管线）中的模型部分。

## 详情

### 主结果（ShopBench，[[ostrakon-vl-paper]] 表 3）

| 模型               | 参数     | 架构    | ShopFront | ShopInterior | Kitchen | MultiImg | Video | AVG      |
| ---------------- | ------ | ----- | --------- | ------------ | ------- | -------- | ----- | -------- |
| Qwen3-VL-8B（基座）  | 8B     | Dense | 59.3      | 46.9         | 54.1    | 46.2     | 50.7  | 55.3     |
| GLM-4.6V-FlashX  | 9B     | Dense | 61.9      | 47.6         | 57.0    | 56.5     | 47.1  | 57.9     |
| Qwen2.5-VL-72B   | 72B    | Dense | 60.8      | 49.5         | 57.3    | 49.3     | 50.7  | 57.3     |
| Qwen3-VL-235B    | 235B   | MoE   | 63.2      | 50.3         | 58.6    | 53.1     | 53.3  | 59.4     |
| InternVL3.5-241B | 241B   | MoE   | 63.5      | 52.7         | 62.9    | 59.7     | 54.2  | 61.1     |
| **Ostrakon-VL**  | **8B** | Dense | **65.0**  | 49.1         | 59.1    | 49.6     | 53.3  | **60.1** |

- 开源阵营第二（仅次于 InternVL3.5-241B）；闭源参考（不计名次）：Seed 1.8 69.0、Gemini3-Pro 63.6、GPT-5 58.8
- 单项：ShopFront 65.0 为所有参与排名的开源模型最高，Kitchen 59.1 第二
- 弱项：MultiImg 49.6 明显低于 GLM-4.6V-FlashX（56.5）/ InternVL3.5-241B（59.7），多图一致性是短板

### 训练策略（三段递进）

1. **Caption Bootstrapping（CB）**：用密集 FSRS 描述文案（招牌文字、菜单板、设备、餐区布局、厨房线索）注入领域知识；描述语料只过 QUAD 的质量过滤 + 语义去重两阶段
2. **离线课程学习（OCL）**：K 个参考 MLLM 投票估计样本难度、分层由易到难；视频数据在 Tier 3 后独立成阶段（避免图文视频混批拖慢训练、梯度互扰）
3. **[[mixed-preference-optimization|MPO]]**：偏好对 = 最高质量正确回答 vs 「貌似正确的错误」（高温采样暴露的不稳定样本 + 规则奖励区分）

消融（表 6，均在 FSRS-Inst 基线 56.8 之上）：单项 CB 57.4 / OCL 57.9 / MPO 57.0；两两组合 CB+OCL 59.3 最强；三合一 60.1。

### 通用能力的代价（[[ostrakon-vl-paper]] 表 4）

- 14 个通用基准平均 66.7 vs 基座 Qwen3-VL-8B 的 72.4（-5.7）
- 掉得最狠：MMVet 61.2→36.4、OCRBench 85.3→61.7
- 保住的：与场景对齐的 Chinese-OCRBench 88.5（基座 90.6）、MathVista 75.4（基座 77.2）
- 论文的定性：专化换专精，未发生灾难性遗忘

### 训练配方（附录 C）

full FT、1 epoch、AdamW lr 2×10⁻⁶（cosine，2% warmup）、global batch 144、BF16、DeepSpeed ZeRO-2、max seq 8192；图像 602,112 px 总像素上限，视频 2 fps 最多 128 帧。权重与 [[shopbench]] 承诺开源（github.com/Ostrakon-VL/Ostrakon-VL）。

## 来源

- [[ostrakon-vl-paper]]

## 相关

- [[qwen]] ｜ [[shopbench]] ｜ [[quad-data-curation]] ｜ [[mixed-preference-optimization]] ｜ [[domain-specific-mllm]]
