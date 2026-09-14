---
type: concept
title: MPO（混合偏好优化）
created: 2026-09-14
updated: 2026-09-14
tags:
  - alignment
  - preference-optimization
  - training
sources: 2
---

# MPO（Mixed Preference Optimization，混合偏好优化）

**定义**：把偏好损失（DPO）、质量损失（BCO）与生成损失（SFT）加权求和的对齐训练目标，用偏好对而非在线采样完成对齐；InternVL 系列起被广泛采用，[[ostrakon-vl]] 用作三段训练策略的收尾段。

## 详情

### 三项损失（[[ostrakon-vl-paper]] §4.3）

$$\mathcal{L} = w_\text{pref}\,\mathcal{L}_\text{DPO} + w_\text{qual}\,\mathcal{L}_\text{BCO} + w_\text{gen}\,\mathcal{L}_\text{SFT}$$

- **偏好损失（源自 DPO）**：训练模型辨别两个候选回答哪个更好
- **质量损失（源自 BCO）**：评估单个回答的内在质量（不依赖成对比较）
- **生成损失（SFT 项）**：正则项，保持流畅稳定、防止过度偏离预训练分布——缓解标准 DPO 的一个失败模式：连正确回答的概率也被无意压低

### 与 GRPO 的取舍

论文选择 MPO 而非 GRPO 的理由是**计算效率**：GRPO 需要在线组采样 + 大规模奖励计算（显存与时间开销大）；MPO 训练时无需为每个输入生成并评估多个候选。在 [[ostrakon-vl]] 的消融中 MPO 单项贡献最小（+0.2，55.3 基座上 FSRS-Inst 56.8 → 57.0），但与 CB/OCL 组合时有叠加收益（CB+MPO 58.5、OCL+MPO 58.5、三者全开 60.1）。

### 偏好对的构造（质量决定 MPO 效果）

1. **高温采样**暴露模型的不稳定行为：同一输入多次生成、时对时错的样本最有价值——偶发的正确往往是表面模式匹配而非真理解
2. 每个输入取「最高质量的正确回答」配「貌似正确的错误回答」（几乎正确但含细微错误：数错物体、轻微事实偏差）
3. **规则奖励**区分正确与貌似正确；显式对比让模型注意到此前忽略的细粒度视觉与语义线索

### 偏好数据质量的直接对照（[[skywork-vl-reward-paper]] §4.7）

同一 MPO 配方下只换偏好数据的来源 RM，差异显著（Skywork R1V2 基座 reasoner，MathVista）：

| 偏好数据来源 RM | MathVista |
|---|---|
| （基座，不微调） | 69.2 |
| Qwen2.5-VL-7B-Instruct | 71.2 |
| InternVL3-8B | 71.8 |
| **[[skywork-vl-reward|Skywork-VL-Reward]]** | **73.5** |

即 MPO 的上限由 RM / 偏好数据质量决定；R1V2 的采用链由此有了第一手证据（此前仅 [[ostrakon-vl-paper]] 转述）。

### 使用脉络

论文自述的采用链：InternVL 系列（InternVL / InternVL3 / InternVL3.5）→ Skywork R1V2、Kwai Keye-VL → [[ostrakon-vl]]（均为开源多模态模型的后对齐段）。

## 来源

- [[ostrakon-vl-paper]]
- [[skywork-vl-reward-paper]]（§4.7 换 RM 对照实验）

## 相关

- [[ostrakon-vl]] ｜ [[quad-data-curation]] ｜ [[domain-specific-mllm]] ｜ [[reward-model]]
