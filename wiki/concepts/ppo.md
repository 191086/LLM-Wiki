---
type: concept
title: PPO（近端策略优化）
created: 2026-09-16
updated: 2026-09-20
tags:
  - reinforcement-learning
  - policy-gradient
  - rlhf
sources: 2
---

# PPO（Proximal Policy Optimization，近端策略优化）

**定义**：OpenAI 提出的策略梯度（policy gradient，对「期望回报」直接求策略参数梯度做随机梯度上升的强化学习方法族）算法——把新 / 旧策略的动作概率比截断（clip）在 $[1-\epsilon,\ 1+\epsilon]$ 并与未截断项取 min，构成真策略性能的**悲观下界式替代目标**（surrogate objective，用一批固定采样数据构造的可微代理目标），让「采一批数据 → 在其上做多轮（epoch，同一批数据反复过）小批量更新」成为安全操作；以纯一阶优化取得置信域方法 TRPO 级的数据效率与稳定性，实现却简单得多。LLM 时代成为 [[rlhf]] 第 3 阶段的标准 RL 引擎（InstructGPT 即以 PPO-ptx 变体对齐 GPT-3，[[instructgpt-paper]] §3.5），也是 [[grpo]]、[[dpo]] 等后续对齐路线的共同对照基线（[[ppo-paper]] 摘要、§1）。

## 1. 要解决什么问题

用神经网络逼近策略做强化学习，2017 年时的三条主路各有硬伤（[[ppo-paper]] §1）：

- **deep Q-learning**（学「动作价值」再取最大的值方法）：在连续动作空间的基础控制基准上失效，且理论性质不明；
- **vanilla policy gradient**（朴素策略梯度）：数据效率差、鲁棒性差——每批采样数据只做**一次**梯度更新就丢弃；
- **[[trpo|TRPO]]**（Trust Region Policy Optimization，置信域策略优化）：数据效率好，但复杂——要对目标做线性近似、对约束做二次近似再用共轭梯度求解（二阶信息），且与 dropout、策略 / 值函数**参数共享**等架构不兼容。

数据效率要求样本复用，而朴素目标恰好不容许复用：策略梯度估计 $\hat{g} = \hat{\mathbb{E}}_t[\nabla_\theta \log \pi_\theta(a_t|s_t)\hat{A}_t]$ 对应的目标 $L^{PG}$（[[ppo-paper]] 式 2）只在「对产生数据的那个策略做一步更新」意义下成立——同一批轨迹上多步优化没有正当性，实测导致**破坏性的大策略更新**（[[ppo-paper]] §2.1）。TRPO 用 KL 散度（Kullback–Leibler divergence，两概率分布差异的度量）硬约束限制每步更新幅度来保安全（式 3–4），但二阶求解贵、且「固定惩罚系数 + SGD」的廉价替代被实验证明不够（§2.2）。PPO 的问题就是：**能不能设计一个目标函数，让朴素的 SGD 天然迈不出大步？**

## 2. 机制：截断替代目标（式 6–7）

记新旧策略对同一「状态–动作」的概率比（概率比 = 新策略给该动作的概率 ÷ 旧策略给的，衡量这一步更新把该动作概率改了多少倍）：

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}, \qquad r_t(\theta_{old}) = 1$$

TRPO 最大化的替代目标 $L^{CPI}$（式 6，CPI = conservative policy iteration，该目标最早由 Kakade & Langford 2002 提出）：

$$L^{CPI}(\theta) = \hat{\mathbb{E}}_t\big[r_t(\theta)\,\hat{A}_t\big] \tag{式 6}$$

其中 $\hat{A}_t$ 是**优势**（advantage）估计：动作 $a_t$ 比该状态的平均水平（值函数 $V(s_t)$，即从此状态出发的期望回报估计，兼作 baseline）好多少；优势为正 → 推高该动作概率，为负 → 压低。不加约束地最大化 $L^{CPI}$ 同样会一步迈太大，PPO 把它改成（式 7，$\epsilon$ 典型取 0.2）：

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\Big[\min\big(r_t(\theta)\hat{A}_t,\ \ \mathrm{clip}(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon)\,\hat{A}_t\big)\Big] \tag{式 7}$$

min 取「未截断」与「截断」两者中**较差**者，使整体成为未截断目标的**下界（悲观界）**。语义可以一句话读完（[[ppo-paper]] §3）：**比率越界只有在让目标变好时才被截掉（去掉继续移动的激励），让目标变差时保留（梯度照常往回拉）**。在 $\theta_{old}$ 处两者一阶等价，$\theta$ 离开 $\theta_{old}$ 后才分歧。

![[ppo-fig1-clip-surrogate-single-term.png]]
> Figure 1（[[ppo-paper]] §3）：单项 $L^{CLIP}$ 随概率比 $r$ 的形状。左：优势为正——$r$ 越过 $1+\epsilon$ 后目标变平（不再奖励继续推高）；右：优势为负——$r$ 低于 $1-\epsilon$ 后目标变平（不再奖励继续压低）。红圈 = 优化起点 $r=1$。

### 手工走查：四个分支的数值代入

设 $\epsilon = 0.2$（截断区间 $[0.8,\ 1.2]$），逐项看 $L^{CLIP}$ 的单项 $\min(r\hat{A},\ \mathrm{clip}(r)\hat{A})$ 在两个越界方向上的行为（每个数值都可心算复核）：

| 情形 | $r$ 出界方向 | 未截断 $r\hat{A}$ | 截断 $\mathrm{clip}(r)\hat{A}$ | $\min$ | 后果 |
| --- | --- | --- | --- | --- | --- |
| $\hat{A}=+1$ | $r=1.3$（推高过头，越界**变好**） | $1.3$ | $1.2$ | $1.2$ | $r$ 再涨目标不动 → **不再激励** |
| $\hat{A}=+1$ | $r=0.7$（不升反降，越界**变差**） | $0.7$ | $0.8$ | $0.7$ | 惩罚生效 → 梯度往回拉 |
| $\hat{A}=-1$ | $r=0.7$（压低过头，越界**变好**） | $-0.7$ | $-0.8$ | $-0.8$ | $r$ 再跌目标不动 → **不再激励** |
| $\hat{A}=-1$ | $r=1.3$（不降反升，越界**变差**） | $-1.3$ | $-1.2$ | $-1.3$ | 未截断项（更差者）入选 → 强推回 |

两个要点：①「变好的越界」一律被截（第一、三行）——目标在边界外**平坦**，梯度为零，策略失去继续把 $r$ 推远的理由；这就是为什么在**同一批数据上做 K 个 epoch** 也不会跑飞——每次更新最多把各样本的 $r$ 挪到边界，之后 incentive 自动关闸。②「变差的越界」一律保留（第二、四行）——min 拿较差者，把策略往回拉的梯度完整保留。悲观下界的名字由此而来：逐项 $\min(a,b)\le a$，$L^{CLIP}$ 处处不超过未截断目标。

## 3. 完整算法（§5：式 9–12、Algorithm 1）

### 3.1 组合目标与 actor-critic 架构

实践中优势用学出来的值函数 $V(s)$ 做方差缩减（见 §3.2），策略与值函数常共享网络参数（TRPO 做不到、PPO 可以——这是 §1 的卖点之一），因此对**组合目标**做上升（式 9）：

$$L_t^{CLIP+VF+S}(\theta) = \hat{\mathbb{E}}_t\big[L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S[\pi_\theta](s_t)\big] \tag{式 9}$$

$L_t^{VF}$ 是值函数的平方误差 $(V_\theta(s_t) - V_t^{targ})^2$；$S$ 是**熵奖励**（entropy bonus，对动作分布熵的加项，分布越「不确定」奖励越高——防止策略过早塌缩到确定性输出、保证探索），$c_1, c_2$ 为系数。这种「策略（Actor）+ 值函数（Critic）」双头的架构称 actor-critic。

### 3.2 优势估计：截断 GAE（式 10–12）

PPO 按固定长度 $T$ 的轨迹段采样（$T$ 远小于整条回合），优势估计不能看超出 $T$ 的未来。用的是**截断版 GAE**（Generalized Advantage Estimation，广义优势估计——Schulman et al. 2015 提出的优势估计器，用 $\lambda$ 在「偏差」与「方差」间插值；推导、$\gamma/\lambda$ 分工与数值走查见 [[gae]]，原文 [[gae-paper]]；[[ppo-paper]] 引 [Sch+15a]）。先定义 TD 残差（式 12）：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \tag{式 12}$$

（$\gamma$ 为折扣因子。）截断 GAE 把一段内的残差按 $(\gamma\lambda)^l$ 衰减加权（式 11，按原文意图整理为标准求和式）：

$$\hat{A}_t = \sum_{l=t}^{T-1} (\gamma\lambda)^{\,l-t}\,\delta_{l} \;\;=\;\; \delta_t + (\gamma\lambda)\,\delta_{t+1} + \cdots + (\gamma\lambda)^{T-1-t}\,\delta_{T-1} \tag{式 11（修正）}$$

> **排印笔误注记**：原文式 11 末项印作 $(\gamma\lambda)^{T-t+1}\delta_{T-1}$、式 10 同位印作 $\gamma^{T-t+1}r_{T-1}$，与逐项指标矛盾——首项 $\delta_t$ 指数为 0、$\delta_{t+1}$ 为 1，则 $\delta_{T-1}$ 应为 $T-1-t$。独立验证：令 $\lambda=1$ 展开求和，$\sum_{l=t}^{T-1}\gamma^{l-t}\delta_l$ 中的值函数项逐级递缩相消，恰余 $-V(s_t) + \sum_{l=t}^{T-1}\gamma^{l-t}r_l + \gamma^{T-t}V(s_T)$，与式 10（修正后）的有限时域估计**严格一致**——原文指数差了 2，判为排印笔误。本页按修正后的标准式照录。

### 3.3 伪代码与超参参照

```python
for iteration in 1, 2, …:
    for actor in 1, 2, …, N:                 # N 个并行环境各自采样
        用 π_θold 跑 T 个时间步，得 {(s_t, a_t, r_t)}
        计算优势估计 Â_1 … Â_T                # 截断 GAE（§3.2）
    在全部 NT 个样本上，以 minibatch 尺寸 M ≤ NT
      做 K epochs 小批量（Adam）上升，最大化 L^{CLIP+VF+S}
    θ_old ← θ                                # 锚点前移，开始下一轮采样
```

MuJoCo 主基准的超参（[[ppo-paper]] Table 3）：T=2048、K=10 epochs、minibatch 64、Adam 3×10⁻⁴、$\gamma=0.99$、$\lambda=0.95$——「K=10 epochs 复用同一批数据」正是 vanilla PG（K=1）做不到的事。

## 4. 变体：自适应 KL 惩罚（§4）

截断之外的正统替代：目标加 KL 惩罚项（式 8）：

$$L^{KLPEN}(\theta) = \hat{\mathbb{E}}_t\big[r_t(\theta)\hat{A}_t - \beta\,\mathrm{KL}\big[\pi_{\theta_{old}}(\cdot|s_t), \pi_\theta(\cdot|s_t)\big]\big] \tag{式 8}$$

并把 $\beta$ **自适应化**——每轮算 $\hat{\mathbb{E}}_t[\mathrm{KL}]$，与目标值 $d_{targ}$ 比较：低于 $d_{targ}/1.5$ 则 $\beta \leftarrow \beta/2$、高于 $1.5\,d_{targ}$ 则 $\beta \leftarrow 2\beta$（乘性调整，1.5 / 2 为启发式、不敏感；[[ppo-paper]] §4）。实验结论：KL 惩罚版**劣于** clip 版（见 §5 消融），论文保留它作为重要 baseline。后续路线里 [[grpo]] 把 KL 惩罚项直接放进目标（固定系数），正是这一支的延续。

## 5. 实验证据（§6）

- **消融：clip 胜出且对 ε 稳健**（§6.1，Table 1）：7 个 MuJoCo 任务 × 3 seeds × 1M 步，分数按「随机策略 = 0、最优 = 1」归一化后平均（21 runs）：无 clip 无惩罚 **−0.39**（half cheetah 崩坏，差于随机策略，拖负均值）；clip $\epsilon=0.1/0.2/0.3$ → 0.76 / **0.82** / 0.70；自适应 KL ≤ 0.74；固定 KL ≤ 0.72。log 空间 clip 无增益。§2 的机制主张被数字坐实：截掉「变好的越界」正是防崩坏的药方
- **MuJoCo 对比**（§6.2，Figure 3）：PPO（clip 版）几乎在全部环境胜过 TRPO、CEM（交叉熵方法）、自适应步长 vanilla PG、A2C、A2C+置信域
- **Atari 对比**（§6.4，Table 2）：49 游戏 × 3 trials，按「全程平均每集奖励」计 PPO 胜 **30** / ACER 18 / A2C 1；按「末 100 集平均」计 PPO 19 / ACER **28** / A2C 1——样本复杂度显著优于 A2C、与 ACER 相当，但算法简单得多
- **机制直观证据**（§3，Figure 2）：沿一次真实 PPO 更新方向插值，$L^{CLIP}$ 恒为 $L^{CPI}$ 的下界；更新后策略与初始策略的 KL ≈ 0.02，恰是 $L^{CLIP}$ 的最大值点——截断目标把更新自然推到「proximal」（邻近 $\theta_{old}$）的位置

![[ppo-fig2-interpolation-lower-bound.png]]
> Figure 2（[[ppo-paper]] §3）：插值路径上四个目标的对比。$L^{CLIP}$（红）封顶于 $L^{CPI}$（橙）下方；KL（蓝）单调增——红线的峰值位置就是被截断目标选中的更新步长。

## 6. 关键性质与陷阱

- **多 epoch 复用是数据效率的来源，clip 是复用的保险**：vanilla PG 每批数据一次更新；PPO 同批 K=10 epochs——采样贵的环境（尤其 LLM 生成）里这是几十倍的样本成本差
- **悲观下界 ≠ 约束参数距离**：$L^{CLIP}$ 相对**固定**的 $\theta_{old}$ 定义，约束的是替代目标的形状，不是 $\theta$ 与 $\theta_{old}$ 的参数距离——K epochs 内部 KL 仍可能爬升，Figure 2 展示的只是一次迭代后的插值。**[wiki 外一般性知识]** 此外 PPO 的实测性能对大量原论文未提的工程细节敏感（优势按 minibatch 归一化、值函数 clip、KL 早停等）；候选来源：Engstrom et al. 2020《Implementation Matters in Deep RL》（arXiv:2005.12729）、Huang et al. 2022《The 37 Implementation Details of PPO》（ICLR 2022 Blog Track）
- **超参 $\epsilon$ 相当稳健**（0.1 / 0.2 / 0.3 全部 ≥0.70，[[ppo-paper]] Table 1），这本身是相对 TRPO（$\delta$ 难选）与固定 KL 惩罚（$\beta$ 难选）的工程优势
- **advantage 估计的截断**：固定长度 T 采样 + 截断 GAE（§3.2）保证估计不看段外未来——代价是偏差，由 $\lambda$ 调节（$\lambda=1$ 退化为 Monte-Carlo 式有限时域回报；$\gamma/\lambda$ 的不对称分工与数值演示见 [[gae]] §4–5）

## 7. 在本 wiki 的语境

- **[[rlhf]] 的第 3 阶段引擎**：RM 打分当奖励、bandit 环境（一句 prompt 采一条回答即终局结算，无跨步决策），PPO 的「采样–优化交替 + clip 保险」结构原样沿用；工程化细节（逐 token KL 惩罚、值函数从 RM 初始化、PPO-ptx 混预训练梯度）见 [[rlhf]] §2 与 [[instructgpt-paper]]
- **[[grpo]] 的对照面**：GRPO 的改动全部针对 PPO 的成本结构——去掉 critic（组内相对优势替代值函数 baseline），保留 clip 与 KL 惩罚的骨架（该页 §2 注记）
- **[[dpo]] 的绕行对象**：DPO 把 RLHF 两阶段折叠为监督损失、彻底不跑 PPO；reward-KL 前沿实验上支配 PPO，PPO 训不稳的结构原因（$\pi_\text{ref}$ 软值函数难估计）见 [[dpo-paper]] §5.2、[[dpo]] §1
- **[[trpo]] 的直接后继**：把「线性替代目标 + KL 置信域约束 + 共轭梯度」的整套求解器替换成一个截断目标——置信域思想的一阶化移植，机制对照见 [[trpo]] §8（其 §4 还数值演示了 TRPO 理论惩罚系数为何不可用）
- 评价信号两翼：PPO 消费的奖励可以来自学习式 RM（[[reward-model]]）或程序化规则（[[rule-based-reward]]）

## 来源

- [[ppo-paper]]（§1 动机与三前驱、§2 背景、§3 截断目标与式 6–7、§4 自适应 KL、§5 算法与式 9–12、§6 消融与对比实验、Algorithm 1、Table 1–3；Figure 1–2）

## 相关

- [[trpo]]（置信域前身：Theorem 1 下界 + KL 约束 + 共轭梯度）｜ [[gae]]（优势估计器本尊：式 16 的推导与 γ/λ 语义）｜ [[rlhf]]（应用范式：第 3 阶段的 RL 引擎）｜ [[instructgpt]]（应用实例）｜ [[grpo]]（去 critic 变体）｜ [[dpo]]（免 RL 折叠路线）｜ [[reward-model]] ｜ [[rule-based-reward]]
