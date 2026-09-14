---
type: source
title: Skywork-VL Reward（论文）
created: 2026-09-14
updated: 2026-09-14
tags:
  - paper
  - reward-model
  - mllm
  - preference-data
sources: 1
raw: raw/Skywork-VL Reward.pdf
---

# Skywork-VL Reward: An Effective Reward Model for Multimodal Understanding and Reasoning

> 原文：[[raw/Skywork-VL Reward.pdf]] ｜ 类型：论文（arXiv:2505.07263）｜ 日期：2025-05 ｜ 机构：Skywork AI（昆仑万维）｜ 模型：huggingface.co/Skywork/Skywork-VL-Reward-7B

## 一句话总结

昆仑万维的开源 7B 多模态奖励模型 [[skywork-vl-reward|Skywork-VL-Reward]]：Qwen2.5-VL-7B 拆掉 LM 头换奖励头，约 19 万对三阶段清洗的多模态偏好数据训成，VL-RewardBench 73.1% 超 GPT-4o——后被 [[ostrakon-vl]] 原样取用为 [[quad-data-curation|QUAD]] 全管线的统一打分器。

## 关键要点

- **定位**：补多模态 RM 两大短板——跨任务泛化不足、评不了 reasoner 的复杂推理输出；输入（可选图，提问，候选回答）→ 输出单个无界标量，判别式 + 结果奖励 ORM（[[reward-model]] 分类学）（§1–3）
- **架构**：Qwen2.5-VL-7B-Instruct 拆掉 token 预测头，在回答末 token 的最终隐藏态上接全连接奖励头直接回归分数（§3.3）
- **损失只学相对序**：Bradley-Terry 排序损失 $-\log\sigma(s^+-s^-)$，绝对分值不校准；「等质」偏好对引入歧义，训练前主动剔除（§3.4）
- **数据即清洗产物**：约 19 万对（70% 含图），三阶段流水线（过滤 → surrogate RM 打分修订 → 推理风格回答生成，详见下）（§3.1–3.2）
- **训练**：冻结 ViT，projector / 语言骨干 / 奖励头可训；两阶段（先多模态 lr 10⁻⁵、再混纯文本 lr 10⁻⁶），各 2 epoch，AdamW（§4.1）
- **成绩**（§4.4–4.5）：VL-RewardBench 73.1 / macro 69.0，超 GPT-4o（65.8 / 62.4）与 Gemini-2.0-flash-exp（68.8 / 64.5）；**幻觉检测 80.0 全场最佳**；reasoning 61.0 ≈ 10× 大的 InternVL3-78B（64.5）；**general 66.0 是弱项**（IXC-2.5-Reward-7B 80.3）；文本 RewardBench 90.1 为同规模多模态 RM 最佳（+2.0 vs IXC-2.5），接近纯文本专用 RM（QRM-Llama3.1-8B-v2 93.1）
- **风格偏好**（case study，§4.6）：两案均答对，简洁推理版得分 5.86 / 7.53，冗长自校正 / 重复罗列版被判低分乃至 −10.36，判词直指「wait 反复出现」「重复啰嗦」——系统性惩罚 reasoner 长链路的典型文风
- **MPO 实证**（§4.7）：用本模型造偏好数据 MPO 微调 Skywork R1V2 的基座 reasoner，MathVista 69.2→73.5；对照 Qwen2.5-VL-7B / InternVL3-8B 造的数据仅 71.2 / 71.8——RM 质量直接决定 MPO 上限

### 三阶段数据清洗（§3.2）

1. **过滤**：跨源去重 + 语义相似过滤 + 判决过滤（剔除与 GPT-4o 判断矛盾或「等质」的模糊对）→ 约 20 万高置信对 → 训一个 **surrogate RM** 给全量数据打分
2. **按分修订**：chosen 被 surrogate 打低分 → GPT-4o 重生成替换；正反分差过小 → chosen 也重生成 → 保留 15 万
3. **推理风格回答**：Skywork R1V 直接生成 47.4%；InternVL 系列写图述替代视觉输入 → DeepSeek R1 生成最终推理 52.6% → 最终约 19 万对

![[skywork-vl-reward-fig1-data-distribution.png]]
> 开源部分训练数据的 17 域分布（论文图 1）：教育与摄影 Imaging 最高（10.4% / 10.3%），地理环境最低（1.2%）。

自建部分约 5 万条推理对比数据按学科分布：数学 35.4 / 物理 24.6 / 化学 20.2 / 生物 14.7 / 其他 5.1（%，表 1）。

## 值得追踪的实体与概念

- [[skywork-vl-reward]]（模型本体）
- [[reward-model]] ｜ [[mixed-preference-optimization]] ｜ [[quad-data-curation]]
- [[qwen]]（基座 Qwen2.5-VL-7B-Instruct 的提供方）
