---
type: concept
title: 奖励模型（Reward Model）
created: 2026-09-14
updated: 2026-09-14
tags:
  - reward-model
  - alignment
  - rlhf
sources: 2
---

# 奖励模型（Reward Model, RM）

**定义**：对（输入，候选回答）输出质量分数的模型，为对齐训练（RLHF / 偏好优化）和推理时选择提供评价信号；[[skywork-vl-reward-paper]] §2 给出两轴分类法。

## 详情

### 两轴分类法（[[skywork-vl-reward-paper]] §2）

**按模型形态**：

- **判别式 RM**：把偏好预测当（标量）回归问题，输入候选回答直接输出分数或偏好概率；[[skywork-vl-reward|Skywork-VL-Reward]] 属此类
- **生成式 RM**：用 LM 头按评审 prompt 生成评判结论（LLM-as-judge 风格）而非数值；多数早期多模态 RM 走这条路
- **隐式 RM**：DPO 式重参数化——偏好学习隐含在策略模型内部，不需要独立的显式 RM

**按反馈目标**：

- **结果奖励（ORM）**：整条回答一个分，评价最终产出
- **过程奖励（PRM）**：对回答的中间步骤逐步打分（如「Let's Verify Step by Step」）；多模态代表 VisualPRM——过程监督 + 作测试时扩展的 critic 提升已有 VLM 的推理

### 多模态 RM 的两大短板（§1）

文本 RM 研究充分，多模态 RM 起步晚：①**跨任务泛化不足**；②**评不了 reasoner**（复杂长链推理输出）。Skywork-VL-Reward 的补法 = 广覆盖偏好数据（含专门的推理风格回答）+ 判别式标量头，在 VL-RewardBench 的 general / 幻觉 / reasoning 三类目上同时拿到可用成绩。

### 排序损失的固有性质（§3.4）

Bradley-Terry 成对损失 $-\log\sigma(s^+-s^-)$ 只学**相对序**、不校准绝对值，两个直接后果：

- 绝对分数单独无意义，只能做差值 / 阈值比较——[[quad-data-curation|QUAD]] 对 $R_\phi$ 的三处用法（质量过滤、基座参考过滤、OCL 难度投票）全部如此，阈值须验证集 + 人工审计标定
- 「等质」偏好对引入歧义，训练数据须主动剔除（Skywork 的判决过滤把与 GPT-4o「等质」判断的对全部丢弃）

### 在本 wiki 中的角色

- [[skywork-vl-reward]]：判别式 ORM 的开源代表，发布时 VL-RewardBench 榜首，被 [[ostrakon-vl]] 全管线取用
- [[quad-data-curation]]：RM 作数据清洗裁判——质量过滤、基座参考过滤、课程难度分层
- [[mixed-preference-optimization]]：RM 产偏好对驱动 MPO；数据质量直接决定上限（MathVista 同配方换 RM：71.2 / 71.8 / 73.5）

## 来源

- [[skywork-vl-reward-paper]]（分类学与排序损失性质）
- [[ostrakon-vl-paper]]（RM 作为清洗裁判的用法）

## 相关

- [[skywork-vl-reward]] ｜ [[quad-data-curation]] ｜ [[mixed-preference-optimization]] ｜ [[ostrakon-vl]]
