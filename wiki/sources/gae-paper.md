---
type: source
title: GAE：广义优势估计（论文）
created: 2026-09-20
updated: 2026-09-20
tags:
  - paper
  - reinforcement-learning
  - policy-gradient
sources: 1
raw: raw/High-Dimensional Continuous Control Using Generalized Advantage Estimation.md
---

# High-Dimensional Continuous Control Using Generalized Advantage Estimation

> 原文：[[raw/High-Dimensional Continuous Control Using Generalized Advantage Estimation.md]] ｜ 类型：论文（arXiv:1506.02438**v6**，ICLR 2016 camera-ready，2015-06 首版）｜ 机构：UC Berkeley（EECS）｜ 作者：John Schulman、Philipp Moritz、Sergey Levine、Michael I. Jordan、Pieter Abbeel（14 页：正文 §1–7 + 附录 A–B）

> **收录注记**：raw 为 arXiv HTML 的 Web Clipper 裁剪版，**公式号在转换中全部丢失**；本页公式号按 v6 PDF 恢复为式 1–31，以正文三处显式引用（式 6 = $g^\gamma$、式 16 = GAE 主公式、式 21 = 塑形折扣和）与算法伪代码两处引用（式 30 = 值函数 QP、式 31 = TRPO 更新）为锚，逐式与 PDF 核对无误。HTML 版 §6.2.2 质心终止阈值印作「8.m / 2.m」，系丢前导零的转换伪影——PDF 原文为 **0.8 m（biped）/ 0.2 m（quadruped）**。三张图（3dwalker / cartpole-grid / standup-circle）随裁剪自动落入 raw/assets，文件名保留原名。

## 一句话总结

策略梯度的高方差可以用值函数压下来，但要付出偏差的代价——GAE 把这个取舍变成**两个可调参数**：$\hat A_t^{\mathrm{GAE}(\gamma,\lambda)} = \sum_l (\gamma\lambda)^l \delta_t^V$（式 16），即 TD 残差的指数折扣加权和，$\lambda=0$ 是低方差高偏差的单步 TD、$\lambda=1$ 是高方差但对值函数误差免疫的蒙特卡洛回报减 baseline，中间值连续插值；配上**值函数的置信域训练**（式 29–30，与 TRPO 同一套共轭梯度机制），在 MuJoCo 3D 双足 / 四足运动与起身任务上以纯神经网络策略（原始运动学→关节力矩、完全 model-free）刷新当时纪录——这是 [[ppo|PPO]] 家族优势估计器的源头文献（PPO 用的截断 GAE 即式 16 的有限时域版，[[ppo-paper]] §5）。

## 关键要点

- **动机：两挑战分工（§1，摘要）**：① 样本复杂度——策略梯度估计的方差随时域增长（动作的效果与前后动作混杂）；② 非平稳数据下的稳定改进。GAE 攻 ①：值函数大幅降方差、容忍可控偏差；② 交给置信域优化（策略与值函数都用）。作者明言：高方差只是费样本，**偏差更险**——无限样本下偏差仍可导致不收敛或收敛到连局部最优都不是的差解
- **公式先前已有、分析是新的（§1 贡献列表）**：公式形在 Kimura & Kobayashi 1998、Wawrzyński 2009 的在线 actor-critic 里已出现；本文给出适用于在线与 batch（含置信域算法）的一般分析 + 塑形解释。另两个贡献：值函数置信域训练法、组合后在 3D 运动上有效
- **策略梯度的六种 $\Psi_t$（§2，式 1）**：$g = \mathbb{E}[\sum_t \Psi_t \nabla_\theta\log\pi]$，$\Psi_t$ 可取整条回报 / 后续回报 / 减 baseline 回报 / $Q^\pi$ / $A^\pi$ / TD 残差；优势函数「几乎方差最低」，因为它度量动作比策略平均水平好坏——梯度方向恰是「推高好于平均的动作」
- **问题不设折扣、$\gamma$ 是算法参数（§2）**：目标是不折扣总回报；折扣版问题可作为特例（折扣吸收进奖励函数）。$\gamma<1$ 用于降方差、引入偏差——与 MDP 问题设定里的折扣语义刻意区分
- **$\gamma$-just 框架（§2，Def 1 + Prop 1，附录 B 证明）**：$\hat A_t$ 是 $\gamma$-just 的，当它代进式 6 估计 $g^\gamma$ 时不引入偏差；**充分条件**（Prop 1）：$\hat A_t = Q_t - b_t$，$Q_t$ 是 $\gamma$-折扣 Q 的无偏估计、$b_t$ 只依赖 $a_t$ 之前的状态动作——**任何 baseline 都不引起偏差**（证明核心：$\mathbb{E}[\nabla\log\pi(a_t|s_t)\mid\text{过去}] = 0$）。四个 $\gamma$-just 例：折扣经验回报、$Q^{\pi,\gamma}$、$A^{\pi,\gamma}$、$r_t + \gamma V^{\pi,\gamma}(s_{t+1}) - V^{\pi,\gamma}(s_t)$
- **TD 残差与 k 步链（§3，式 10–15）**：$\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s)$ 在 $V = V^{\pi,\gamma}$ 时是 $A^{\pi,\gamma}$ 的无偏估计（式 10）；k 步和 $\hat A^{(k)}$ 递缩相消 =「k 步回报 − baseline」（式 14），$k\to\infty$ 得经验回报 − baseline（式 15），偏差随 $k$ 增大而减小（$\gamma^k V(s_{t+k})$ 被折扣压掉、$-V(s_t)$ 不影响偏差）
- **GAE 主公式（§3，式 16）**：k 步估计的指数加权平均 $(1-\lambda)\sum_k \lambda^{k-1}\hat A^{(k)}$ 收拢为 $\sum_l (\gamma\lambda)^l \delta_{t+l}^V$；与 TD($\lambda$) 构造同构但目标不同（估优势而非价值）。特例：**GAE($\gamma$,0) = $\delta_t$**（式 17，低方差、$V$ 不准则偏）、**GAE($\gamma$,1) = 折扣回报 − $V(s_t)$**（式 18，任意 $V$ 都 $\gamma$-just、高方差）；$0<\lambda<1$ 折中
- **γ 与 λ 分工不对称（§3 末）**：$\gamma$ 决定价值函数的 scale、$\gamma<1$ **无论 V 多准都引入偏差**；$\lambda<1$ **只在 V 不准时**引入偏差。故经验最优的 λ 远低于最优的 γ——λ 的偏差代价小得多
- **塑形视角（§4，式 20–26）**：取 $\Phi = V$ 的塑形奖励 $\tilde r = r + \gamma\Phi(s') - \Phi(s)$（式 20）恰为 $\delta^V$，其 $\gamma\lambda$-折扣和恰为 GAE（式 25）；**响应函数** $\chi(l;s,a) = \mathbb{E}[r_{t+l}|s,a] - \mathbb{E}[r_{t+l}|s]$（式 26）把优势按时间分解——$\gamma$ 折扣 ≈ 丢弃 $l \gg 1/(1-\gamma)$ 的响应项，塑形 + 更陡折扣 $\gamma\lambda$ = 先压缩响应时程、再截断长延迟噪声。与 Ng et al. 1999 的差别：他们证折扣目标下塑形不改最优策略，本文目标是**不折扣**的、$\gamma$ 是算法参数
- **值函数的置信域训练（§5，式 28–30）**：目标为 Monte-Carlo 回归（式 28，$\hat V = \sum\gamma^l r$）；置信域约束 $\frac{1}{N}\sum\frac{\|V_\phi - V_{\phi_{old}}\|^2}{2\sigma^2} \le \epsilon$（式 29）等价于「把 V 解释为条件 Gaussian 时的平均 KL 约束」；线性化 + Gauss-Newton 矩阵 $H = \frac{1}{N}\sum j_n j_n^\top$（Fisher / $\sigma^2$）二次近似 → 共轭梯度 + 步长重标定（式 30）——与策略侧 TRPO 更新（式 31）同一套数值机制。脚注 2：也试过 TD($\lambda$) 式目标 $\hat V^\lambda = V_{\phi_{old}} + \sum(\gamma\lambda)^l\delta$，与 Monte-Carlo 无差别
- **完整算法与更新顺序（§6.1，式 31 + 伪代码）**：采样 N 步 → 用**当前** $V_{\phi_i}$ 算 $\delta$ 与 $\hat A$ → TRPO 更新 $\theta$（式 31）→ 置信域更新 $\phi$。顺序敏感：若先更新值函数且过拟合到 $\delta \equiv 0$，策略梯度估计直接归零
- **实验设置（§6.2）**：cart-pole（线性策略 + 20 单元单隐层值网络）+ 三个 3D 任务（策略与值函数同构：100/50/25 tanh 三隐层）；MuJoCo；humanoid 33 维状态 / 10 执行器，quadruped 29 / 8；奖励（按出现序）：双足 $v_\mathrm{fwd} - 10^{-5}\|u\|^2 - 10^{-5}\|f_\mathrm{impact}\|^2 + 0.2$、四足 $v_\mathrm{fwd} - 10^{-6}\|u\|^2 - 10^{-3}\|f_\mathrm{impact}\|^2 + 0.05$、起身 $-(h_\mathrm{head}-1.5)^2 - 10^{-5}\|u\|^2$；常数偏置防「自杀式终止回合」；质心低于 0.8 m（双足）/ 0.2 m（四足）终止
- **实验结果（§6.3）**：**中间值最优**——cart-pole（21 seeds）：$\gamma\in[0.96,0.99]$、$\lambda\in[0.92,0.99]$；3D 双足（9 seeds、1000 迭代、每 trial 约 2 小时/16 核）：$\gamma\in[0.99,0.995]$、$\lambda\in[0.96,0.99]$，学到快速平滑稳定的步态；实时等效 $0.01\,\mathrm{s/步} \times 50000\,\mathrm{步/批} \times 1000\,\mathrm{批} = 5.8$ 天——真机并行学习可信可行；四足与起身（5 seeds、约 4 小时/32 核、固定 $\gamma=0.995$）：四足 $\lambda=0.96$ 胜 $\lambda=0$ 与 No VF，起身上值函数总有帮助、$\lambda=0.96$ 与 $\lambda=1$ 相当。No VF = 时间相关 baseline（按批内各时刻平均回报，不依赖状态）——值函数对照
- **FAQ 两条（附录 A）**：A.1 compatible features 与本文正交——它只关心优势在 $\nabla\log\pi$ 张成子空间上的投影（自然梯度的最小二乘实现），不指导如何利用时间结构；A.2 **为什么用 V 不用 Q**——状态价值输入维度低更好学；且 $\lambda$ 插值只在 V 路线可用，参数化 Q 的高偏差估计（如 $\hat A = Q(s,a) - V(s)$）预期同样过大
- **开放问题（§7）**：$\gamma,\lambda$ 的自适应 / 自动调整；值函数估计误差与策略梯度误差的关系（该选什么值函数误差度量——Bellman error 候选）；策略 / 值函数共享架构

![[3dwalker.png]]
> Figure 1（§6.2）：上排——3D 运动任务的机器人模型（双足 humanoid、四足 quadruped）；下排——学到的步态连续帧。策略从原始运动学直接输出关节力矩，无手工策略表示。

![[standup-circle.png]]
> Figure 4（§6.3.3）：(a) 四足行走学习曲线（$\lambda=0.96$ 与值函数胜出）；(b) 3D 起身学习曲线；(c) 起身过程连续帧——双足从仰卧到站起。

## 值得追踪的实体与概念

- [[gae]]（概念页：γ-just 框架与推导、λ 偏差插值的数值走查、塑形视角、与 PPO / TRPO / GRPO 的接线）
- [[ppo]]（直接后继：截断 GAE 即式 16 的固定段版，[[ppo]] §3.2）｜ [[ppo-paper]]（后继论文）
- [[trpo]]（同组前作：本文策略更新引擎，式 31；其 §5 的 MC 回报 Q 估计正是 GAE 要替换的高方差方案）｜ [[trpo-paper]]
- [[rlhf]]（应用谱系：PPO 引擎的优势信号源；bandit 设定下 GAE($\gamma$,1) 退化为「回报 − baseline」）｜ [[grpo]]（去 critic 路线：组均值 baseline 替代学到的 V——附录 A.2「为什么用 V」论证的另一面）
