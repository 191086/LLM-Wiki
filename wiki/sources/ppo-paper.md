---
type: source
title: PPO：近端策略优化算法（论文）
created: 2026-09-16
updated: 2026-09-16
tags:
  - paper
  - reinforcement-learning
  - policy-gradient
sources: 1
raw: raw/ppo.pdf
---

# Proximal Policy Optimization Algorithms

> 原文：[[raw/ppo.pdf]] ｜ 类型：论文（arXiv:1707.06347**v2**，2017-08-28，12 页）｜ 机构：OpenAI ｜ 作者：John Schulman、Filip Wolski、Prafulla Dhariwal、Alec Radford、Oleg Klimov

> **收录注记**：库内这份是 arXiv **v2**（2017-08-28），全文公式号、图号、表号均按 v2。原文排印有一处已知笔误：式 10 / 式 11 末项指数 $T-t+1$ 与逐项指标不符（按 $\delta_t$ 的指数为 0 逐项推应为 $T-1-t$；代入 $\lambda=1$ 展开并与式 10 递缩求和核对可确认），wiki 侧一律按修正后的标准式照录并显式标注，见 [[ppo]] §3。

## 一句话总结

把 [[trpo|TRPO]] 的「置信域」思想化装进一个一阶可优的**截断替代目标**——对策略与旧策略的概率比 $r_t$ 做 clip、取 min 得到悲观下界——使同一批采样数据能安全复用多个 epoch 的小批量更新：在 MuJoCo 上超过 TRPO / CEM / A2C 等对手、Atari 上样本效率远超 A2C 而与 ACER 相当，实现却只需在 vanilla policy gradient 上改几行；数年后它成为 LLM 对齐（RLHF 第 3 阶段）的标准 RL 引擎（[[instructgpt-paper]] §3.5）。

## 关键要点

- **动机：三种前驱各有短板**（§1）：deep Q-learning（带函数逼近）在连续控制上失效且理论不明；vanilla policy gradient（策略梯度，直接对期望回报关于策略参数求梯度的一阶方法）数据效率差、鲁棒性差；[[trpo|TRPO]]（置信域策略优化，带 KL 约束的策略更新）数据效率好但复杂——二阶近似 + 共轭梯度，且与 dropout、策略 / 值函数参数共享等架构不兼容。目标：TRPO 级的数据效率与可靠性能，只用一阶优化
- **朴素目标的缺陷**（§2.1，式 1–2）：策略梯度估计 $\hat{g} = \hat{\mathbb{E}}_t[\nabla_\theta \log \pi_\theta(a_t|s_t)\hat{A}_t]$ 对应目标 $L^{PG}$，但 $L^{PG}$ 只在「对当前策略的一次更新」意义下成立——同一批轨迹上做多步优化没有正当性，实测导致破坏性的大策略更新（§6.1 中类似或差于「无 clip 无惩罚」设定）
- **TRPO 的约束与困境**（§2.2，式 3–5）：最大化替代目标、约束平均 KL $\le\delta$（硬约束，二阶近似 + 共轭梯度求解）；理论其实支持惩罚形式（式 5），但单一 $\beta$ 难以跨问题、甚至跨训练阶段通用——本文实验亦证实「固定 $\beta$ + SGD」不够
- **核心贡献：截断替代目标**（§3，式 6–7）：$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\big[\min\big(r_t(\theta)\hat{A}_t,\ \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\big)\big]$，$r_t = \pi_\theta/\pi_{\theta_{old}}$、$\epsilon=0.2$。min 语义 = 悲观下界：比率越界只有在让目标**变好**时才被截掉（去掉继续移动的激励），让目标**变差**时保留（梯度照常往回拉）；在 $\theta_{old}$ 处与 $L^{CPI}$ 一阶等价
- **自适应 KL 惩罚变体**（§4，式 8）：目标加 $-\beta\,\mathrm{KL}[\pi_{\theta_{old}}, \pi_\theta]$，每轮按 $\hat{\mathbb{E}}_t[\mathrm{KL}]$ 与目标值 $d_{targ}$ 的偏差乘性调 $\beta$（$d < d_{targ}/1.5 \Rightarrow \beta/2$；$d > 1.5\,d_{targ} \Rightarrow 2\beta$）；实测表现劣于 clip 版，作为重要 baseline 收录
- **完整算法**（§5，式 9–12、Algorithm 1）：组合目标 $L^{CLIP+VF+S} = L^{CLIP} - c_1 L^{VF} + c_2 S$（值函数平方误差 + 熵奖励）；优势用**截断 GAE**（式 11–12，$\lambda=1$ 时退化为有限时域估计式 10）；Algorithm 1：N 个并行 actor 各采 T 步 → 在 NT 样本上做 K epochs 小批量 SGD → $\theta_{old} \leftarrow \theta$。MuJoCo 基准超参：T=2048、10 epochs、minibatch 64、$\gamma=0.99$、$\lambda=0.95$（Table 3）
- **消融：clip 胜出**（§6.1，Table 1）：7 个 MuJoCo 任务 × 3 seeds × 1M 步，归一化分（随机策略 = 0、最优 = 1、21 runs 平均）：**无 clip 无惩罚 −0.39**（half cheetah 崩到差于随机策略拖累均值）；clip $\epsilon=0.1/0.2/0.3$ → 0.76 / **0.82** / 0.70（对 ε 相当稳健）；自适应 KL ≤ 0.74；固定 KL ≤ 0.72。log 空间 clip 无增益
- **MuJoCo 对比**（§6.2，Figure 3）：PPO（clip 版）几乎在全部环境胜过 TRPO、CEM、自适应步长 vanilla PG、A2C、A2C+置信域
- **Atari 对比**（§6.4，Table 2）：49 游戏 × 3 trials，按全程平均每集奖励 PPO 胜 **30** / ACER 18 / A2C 1；按末 100 集奖励 PPO 19 / ACER **28** / A2C 1——样本复杂度显著优于 A2C、与 ACER 相当但实现简单得多（§1、§7）
- **结论定位**（§7）：置信域方法的稳定性 + vanilla PG 的简单性（改几行代码）；兼容策略 / 值函数共享参数的架构（TRPO 不行）

![[ppo-fig2-interpolation-lower-bound.png]]
> Figure 2（§3）：沿一次 PPO 更新方向在 $\theta_{old}$ 与更新后参数间插值（Hopper-v1 首次更新）。$L^{CLIP}$（红）是 $L^{CPI}$（橙）的下界——更新过头的代价被截掉；更新后策略与初始策略的 KL ≈ 0.02，恰为 $L^{CLIP}$ 的最大值点——「截断目标把更新自然推到这个 proximal 位置」的直观证据。

## 值得追踪的实体与概念

- [[ppo]]（概念页：机制四分支走查、完整算法、实验证据与工程陷阱）
- [[trpo]]（前身概念页：置信域约束与单调改进下界的原版机制，PPO 的一阶化对象）
- [[rlhf]]（应用范式：PPO 是其第 3 阶段的 RL 引擎）｜ [[instructgpt]]（LLM 对齐应用实例：PPO-ptx）
- [[grpo]]（后续：去 critic 的变体路线）｜ [[dpo]]（后续：绕开 RL 的两阶段折叠，reward-KL 前沿对照 PPO）
