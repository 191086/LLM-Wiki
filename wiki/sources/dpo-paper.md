---
type: source
title: DPO：你的语言模型暗中是个奖励模型（论文）
created: 2026-09-15
updated: 2026-09-15
tags:
  - paper
  - alignment
  - rlhf
  - preference-optimization
sources: 1
raw: raw/DPO.pdf
---

# Direct Preference Optimization: Your Language Model is Secretly a Reward Model

> 原文：[[raw/DPO.pdf]] ｜ 类型：论文（arXiv:2305.18290v3，NeurIPS 2023，页脚注明）｜ 日期：2023-05 首发（v3 2024-07-29）｜ 机构：Stanford University（+ CZ Biohub）｜ 作者：Rafailov、Sharma、Mitchell（共同一作）+ Ermon、Manning、Finn

## 一句话总结

证明 RLHF 的「KL 约束奖励最大化」目标有闭式最优解，据此把奖励重参数化为策略与参考模型的对数概率比（$r = \beta \log \frac{\pi}{\pi_\text{ref}}$）——代回 Bradley-Terry 偏好模型时配分函数相消，于是「训 RM + 跑 PPO」两阶段折叠成一个对 $\pi_\theta$ 直接可微的二元交叉熵损失（DPO），在不采样、不调 RL 超参的条件下达到或超过 PPO-based RLHF。

## 关键要点

- **动机**（§1、§2）：RLHF 管线复杂且不稳——要训多个 LM（SFT / RM / 策略）、训练回路里在线采样、RL 超参难调；本文目标是「优化同一个目标，但只用一个简单的分类损失」
- **三阶段 RLHF 预备**（§3，沿用 Ziegler et al.）：①SFT 得 $\pi^{SFT}$；②偏好标注 + BT 模型拟合 RM（式 1–2，RM 常从 SFT 模型加线性头初始化，并做奖励归一化 $\mathbb{E}[r_\phi(x,y)]=0$）；③KL 约束奖励最大化（式 3），实践中构造 $r(x,y) = r_\phi(x,y) - \beta(\log \pi_\theta(y|x) - \log \pi_\text{ref}(y|x))$ 用 PPO 最大化
- **核心推导**（§4）：式 3 的闭式最优解（式 4）反解出奖励（式 5：$r(x,y) = \beta \log \frac{\pi_r(y|x)}{\pi_\text{ref}(y|x)} + \beta \log Z(x)$）；BT 模型只依赖奖励之差，代入后配分函数相消（式 6），偏好概率只用 $\pi^*$ 与 $\pi_\text{ref}$ 表出——对参数化策略 $\pi_\theta$ 做极大似然即得 DPO 损失（式 7）。Plackett-Luce 排序版在附录 A.3
- **梯度解读：动态加权**（§4）：$\nabla_\theta \mathcal{L}_\text{DPO} = -\beta\,\mathbb{E}\big[\sigma(\hat{r}_\theta(x,y_l) - \hat{r}_\theta(x,y_w))\,[\nabla_\theta\log \pi_\theta(y_w|x) - \nabla_\theta\log \pi_\theta(y_l|x)]\big]$，其中 $\hat{r}_\theta = \beta \log \frac{\pi_\theta}{\pi_\text{ref}}$。升 chosen、降 rejected，但每例按「隐式奖励把顺序排得有多错」加权——排错越狠权重越大；去掉该系数的朴素概率比目标会让语言模型退化（附录 Table 3 的重复词退化样本）
- **理论保证**（§5.1）：奖励等价类（Definition 1：$r - r' = f(x)$）在 PL/BT 下诱导同一偏好分布（Lemma 1）、同一最优策略（Lemma 2）；Theorem 1 证明重参数化不损失表示范围（每个等价类都有一个 $r = \beta\log\frac{\pi}{\pi_\text{ref}}$ 形式的成员），Proposition 1 进一步证明该成员唯一
- **对 PPO 不稳性的诊断**（§5.2）：把 RLHF 目标写成对最优策略的 KL 投影后，PPO 需要 baseline 估计归一化项（= $\pi_\text{ref}$ 的软值函数），学值函数难、单样本 Monte-Carlo 估计方差高；DPO 的重参数化天然不需要 baseline
- **实验设置**（§6）：三任务——IMDb 情感控制（GPT-2-large + 情感分类器作 ground-truth 奖励，附录 C.1）、Reddit TL;DR 摘要（GPT-J 6B SFT，Stiennon et al. 人类偏好，TL;DR 用 $\beta=0.5$）、Anthropic-HH 单轮对话（Pythia-2.8B，无现成 SFT → 在 chosen 上 Preferred-FT 兼作参考模型）；评测用 reward-KL 前沿（有真奖励时）与 GPT-4 胜率（无真奖励时），并附人类研究校验 GPT-4 评测可靠性（§6.4，Table 2：GPT-4 与人的一致率 ≈ 人与人之间一致率）
- **结果 1：前沿严格占优**（§6.1，Figure 2 左）：22 组超参扫描中，DPO 的 reward-KL 前沿严格支配 PPO——甚至优于能用真奖励的 PPO-GT
- **结果 2：摘要与对话**（§6.2）：TL;DR 对人类参考摘要的 GPT-4 胜率 DPO ≈61%（temp 0）> PPO 57%（其最优温度），且对采样温度远比 PPO 稳健（PPO 高温退化到基座 GPT-J 水平）；人类研究中 DPO（temp 0.25）对 PPO（temp 0）胜率 58%。Anthropic-HH 上 DPO 是唯一「计算高效且胜过数据集 chosen 回答」的方法，与算力昂贵的 Best of 128 相当；Best of N 在 N≈64–128 饱和（附录 Figure 4）。OOD 泛化（Table 1）：CNN/DailyMail 上 DPO 胜率 0.36/0.31（temp 0/0.25）vs PPO 0.26/0.23
- **实现极简**（Appendix B）：损失十余行 PyTorch（两个 logps 进、logsigmoid 出）；默认 $\beta=0.1$、batch 64、RMSprop lr 1e-6、150 步线性 warmup。$\pi_\text{ref} = \pi^{SFT}$（可得时），否则在 chosen 上做 MLE 近似以缓解分布漂移（§4 DPO outline）
- **自陈局限**（§7）：OOD 泛化与自标注利用未充分研究；reward 过度优化在 DPO 中如何显形未解（Figure 2 右后段的轻微回落是否为其实例存疑）；实验规模 ≤6B；GPT-4 评测受 prompt 措辞影响

![[dpo-fig2-reward-kl-frontier-tldr-winrate.png]]
> Figure 2（§5.2 / §6.1–6.2）：左——IMDb 情感任务的 reward-KL 前沿，DPO（黄）整体在 PPO / PPO-GT / Unlikelihood / Preferred-FT 之上；右——TL;DR 对人类参考摘要的 GPT-4 胜率随采样温度的变化，DPO ≈61% 起步且平稳，PPO 在高温塌到基座水平。

## 值得追踪的实体与概念

- [[dpo]]（算法本体：完整推导、数值与梯度走查、关键性质）
- [[reward-model]]（BT 损失与 RM 分类学——本文式 1–3 是其理论起点；隐式 RM 分类项即本文标题）
- [[rlhf]]（三阶段管线：本文 §3 的预备综述对象）
- [[mixed-preference-optimization]]（DPO 作为偏好项的实战用法）｜ [[grpo]]（在线 RL 路线对照：式 3 的 KL 约束同源）
