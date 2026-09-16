---
type: entity
title: InstructGPT（模型）
created: 2026-09-16
updated: 2026-09-16
tags:
  - alignment
  - rlhf
  - openai
sources: 1
---

# InstructGPT

**定义**：OpenAI Alignment 团队用 [[rlhf|RLHF]] 三阶段微调 GPT-3 得到的对齐模型家族（1.3B / 6B / 175B，同 GPT-3 架构），是「对齐小模型胜过未对齐大模型」的定标实证载体（1.3B 版在人类评估中净胜 175B GPT-3）。

## 1. 是什么

预训练 GPT-3（Brown et al., 2020）经三步微调的产物：SFT（标注员示范）→ 6B 奖励模型（偏好排序）→ [[ppo|PPO]] 对 RM 优化。论文中 InstructGPT 默认指 **PPO-ptx** 变体（PPO 之外再混预训练梯度，缓解公共 NLP 任务回退）；不带 ptx 的「PPO」变体在人类偏好上与 PPO-ptx 相当（[[instructgpt-paper]] §3.5、§4.1）。

部署形态：训练数据来自 OpenAI API Playground 上早期 InstructGPT 版本（仅示范数据训练）收集的客户 prompt，成品又通过 Playground 对外提供（[[instructgpt-paper]] §3.2）。

> **wiki 外补注**：InstructGPT 论文（2022-03）发表约九个月后，OpenAI 发布 ChatGPT（2022-11），公开表述其训练方法即 InstructGPT 同源的 RLHF 管线，基座换为 GPT-3.5 系。候选来源：OpenAI 博文《ChatGPT: Optimizing Language Models for Dialogue》（2022-11-30）。

## 2. 配方速览（细节见 [[rlhf]] §2 与 [[instructgpt-paper]] §3）

| 组件 | 规格 | 关键细节 |
|---|---|---|
| SFT | 1.3B/6B/175B | 16 epochs；按 RM 分选模型（验证损失 1 epoch 即过拟合） |
| RM | 6B 一档 | BT 排序损失（式 1）；整 prompt 的 $\binom{K}{2}$ 对作单 batch 元素 |
| [[ppo|PPO]] | 策略同 SFT 三档 | 逐 token KL 惩罚 β=0.02；值函数从 RM 初始化；PPO-ptx：γ=27.8 |

训练数据：SFT ~13k prompt、RM 33k、PPO 31k（Table 6）；标注团队约 40 人，训练标注员互相一致率 72.6±1.5%（§3.4）。

## 3. 代表性数字

- 175B InstructGPT 对 175B GPT-3 胜率 85±3%、对 few-shot GPT-3 71±4%；1.3B 已净胜 175B GPT-3（§4.1）
- 对齐成本：175B SFT 4.9 + PPO-ptx 60 petaflops/s-days，为 GPT-3 预训练 3,640 的约 2%（§5.1）
- 真实性：闭域幻觉率 21% vs GPT-3 41%；TruthfulQA 真实且信息量约 2×（§4.2）
- 边界：被明确要求有毒时比 GPT-3 更毒；偏见指标无改善（§4.2、§5.3）

## 来源

- [[instructgpt-paper]]

## 相关

- [[rlhf]]（范式页：管线机制与路线谱系）｜ [[ppo]]（第 3 阶段 RL 引擎的机制页）｜ [[reward-model]]（第 2 阶段的 6B RM）｜ [[dpo]]（后续：折叠第 2+3 阶段）
