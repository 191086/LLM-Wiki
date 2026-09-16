---
type: source
title: TRPO：置信域策略优化（论文）
created: 2026-09-16
updated: 2026-09-16
tags:
  - paper
  - reinforcement-learning
  - policy-gradient
sources: 1
raw: raw/Trust Region Policy Optimization.pdf
---

# Trust Region Policy Optimization

> 原文：[[raw/Trust Region Policy Optimization.pdf]] ｜ 类型：论文（arXiv:1502.05477**v5**，2017-04-20；ICML 2015）｜ 机构：UC Berkeley（EECS）｜ 作者：John Schulman、Sergey Levine、Philipp Moritz、Michael I. Jordan、Pieter Abbeel（16 页：正文 §1–9 + 附录 A–F）

> **收录注记**：全文公式号按 arXiv v5（式 1–18 + 附录式 19–57）。原文自身不一致一处：正文 §8.1 称 Walker 状态空间 18 维，附录 E Table 2 却印 20（Swimmer 10 / Hopper 12 与正文一致），wiki 侧叙述取正文值 18 并在此注记。Table 1 中对手行名为 **UCC-I**（Guo et al. 2014，MCTS + 监督训练）。

## 一句话总结

给「用神经网络表示的策略」的每次更新一个**可证明不改坏性能的下界**：证明最大化「旧策略替代目标 − KL 置信域约束」的解序列单调改进（Theorem 1：$\eta(\tilde\pi) \ge L_\pi(\tilde\pi) - C\,D^{\max}_{\mathrm{KL}}$），再把它落成可实现的近似三连——惩罚改硬约束、max KL 改平均 KL、精确解改共轭梯度 + 回溯线搜索——在 MuJoCo 运动控制上包揽全部任务前二、从原始图像玩 Atari 七局取得合理分数；它把自然梯度与策略迭代统一为特例，也是 [[ppo|PPO]] 的直接前身（PPO 把它的置信域约束一阶化为截断目标，[[ppo-paper]] §1）。

## 关键要点

- **动机：三族方法都缺「步长的正当性」**（§1）：策略迭代（值函数逼近下无收敛保证）、策略梯度（数据效率差，每批采样只更新一次就丢）、免梯度方法 CEM/CMA（把回报当黑盒，样本复杂度随参数量爆炸）。目标：对非线性策略（数万参数）给出**单调改进**的迭代过程
- **性能差分解**（§2，式 1–2，Kakade & Langford 2002）：$\eta(\tilde\pi) = \eta(\pi) + \mathbb{E}_{\tau\sim\tilde\pi}\big[\sum_t \gamma^t A_\pi(s_t,a_t)\big]$——任意改进都可写成「新策略轨迹上旧策略优势」的折扣和；由此构造替代目标 $L_\pi$（式 3，旧策略访问频率 $\rho_\pi$ 加权新策略优势），与 $\eta$ 在 $\pi$ 处一阶匹配（式 4）
- **单调改进下界**（§3，式 5–10，Theorem 1、Algorithm 1）：CPI 混合策略下界（式 6，误差 $\frac{2\epsilon\gamma}{(1-\gamma)^2}\alpha^2$）推广到一般随机策略 → **Theorem 1**（式 8）：$\eta(\tilde\pi) \ge L_\pi(\tilde\pi) - \frac{4\epsilon\gamma}{(1-\gamma)^2}\alpha^2$，$\epsilon = \max_{s,a}|A_\pi(s,a)|$、$\alpha = D^{\max}_{\mathrm{TV}}$；KL 形式（式 9，用 $D_{\mathrm{TV}}^2 \le D_{\mathrm{KL}}$）。以 $M = L - C\,D^{\max}_{\mathrm{KL}}$ 为锚的迭代是 minorization–maximization（MM）算法——每步最大化 $\eta$ 的下界，性能不降（式 10、Algorithm 1）
- **从惩罚到置信域**（§4，式 11–12）：理论给定的惩罚系数 $C$ 量级巨大（走查见 [[trpo]] §4），惩罚形式的步长会被罚得极小 → 改为**硬约束** $D^{\max}_{\mathrm{KL}} \le \delta$；逐状态 max 约束因状态数巨大不可解，启发式换成**平均 KL** $\bar D_{\mathrm{KL}} \le \delta$（式 12）
- **两套采样方案**（§5，式 13–16，Figure 1）：**single path**——常规轨迹采样，$Q$ 用折扣回报估计；**vine**——采一批「rollout set」状态、每状态采 $K$ 个动作、自归一化重要性采样（式 16）+ 共同随机数（common random numbers，同一随机数序列喂给不同候选策略以降方差）；同量 $Q$ 样本下 vine 对替代目标的估计方差更低，代价是环境必须可重置（仿真专属）
- **实用算法**（§6、附录 C）：约简后的约束问题（附录式 55）用「目标一阶近似 + KL 二阶近似」解——共轭梯度解 Fisher-vector product 方程（$k=10$ 次迭代已够、可用 10% 数据子采样，代价与算梯度同量级）+ 回溯线搜索同时保证目标改进与约束满足；「没有线搜索，算法偶尔灾难性退化」（附录 C）
- **统一视角**（§7，式 17–18）：自然策略梯度 = 式 12 的「目标线性 + KL 二次」特例（但用固定步长 $1/\lambda$，不逐步强制约束——大问题上显著更差）；$\ell_2$ 约束策略梯度 = 式 12 的欧氏距离特例
- **MuJoCo 实验**（§8.1，Figure 4，Table 2）：cartpole / swimmer / hopper / walker 四任务，$\delta = 0.01$ **全部实验通用**（Table 2、Table 3）；策略网络最小 364 参数（swimmer）、最大 8206（walker）；vine 与 single path TRPO 包揽**全部任务前二**；natural gradient（固定惩罚 $\lambda$）只赢两个简单任务、hopper/walker 学不出前进；max KL 变体学得更慢（说明平均 KL 约束效果 ≈ 理论 max KL）；免梯度的 CEM/CMA 在大问题上失效
- **Atari 实验**（§8.2，Table 1，Table 3）：七游戏、原始图像输入、33,500 参数 CNN（两层 conv 16 通道 stride 2 + 一层全连接 20 单元）、500 迭代 ≈30 小时（16 核）；vine 在 Q*bert 7732.5 超 Deep Q-Learning 1952 约 4×、Pong 20.9 近满分，但 Beam Rider 859.5 远低于 DQN 4092——论文自评「只在部分游戏超越先前方法，但一致取得合理分数」；此前该设定只有 DQN 与 UCC-I 有结果，TRPO 的意义在**通用性**（同一套方法直接从图像学）
- **结论定位**（§9）：首个对通用神经网络策略、极简奖励从头学全部运动控制（游 / 走 / 跳）的通用策略搜索方法；分析统一策略梯度与策略迭代两族——「都是某个置信域目标的特例」

![[trpo-fig5-atari-learning-curves.png]]
> Figure 5（附录 F）：Atari 七局学习曲线（cost = 负奖励，历史原因）。breakout 上 single path 在 −13 附近停滞、vine 持续下降到 −35 以下；qbert / seaquest / beam rider 上 vine 同样领先——vine 的高方差优势（每状态多次 rollout）在像素输入任务上依然成立。

## 值得追踪的实体与概念

- [[trpo]]（概念页：理论骨架、α-coupling 证明思想、两轮手工走查、采样方案与 PPO 对照）
- [[ppo]]（直接后继：置信域约束的一阶化）｜ [[ppo-paper]]（后继论文：截断替代目标）
- [[rlhf]]（应用谱系：PPO 成为对齐第 3 阶段的 RL 引擎，TRPO 是该引擎的理论前身）
