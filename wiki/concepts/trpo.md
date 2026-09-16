---
type: concept
title: TRPO（置信域策略优化）
created: 2026-09-16
updated: 2026-09-16
tags:
  - reinforcement-learning
  - policy-gradient
  - rlhf
sources: 1
---

# TRPO（Trust Region Policy Optimization，置信域策略优化）

**定义**：Schulman et al.（UC Berkeley，ICML 2015）提出的策略优化算法——每步在「旧策略的替代目标（surrogate，用旧策略访问频率加权新策略优势的期望，式 3）」之上施加**置信域约束**（trust region，参数空间中围绕当前策略、信任更新的一小块区域，用新旧策略的 KL 散度度量，$\bar D_{\mathrm{KL}} \le \delta$），并以 **Theorem 1** 证明由此得到的策略序列**单调改进**（$\eta(\tilde\pi) \ge L_\pi(\tilde\pi) - C\,D^{\max}_{\mathrm{KL}}$）。理论把自然策略梯度与策略迭代统一为特例；实践上是 [[ppo|PPO]] 的直接前身——PPO 把它的「约束 + 二阶求解」一阶化为截断目标（[[ppo-paper]] §1–2），后者即 [[rlhf]] 第 3 阶段的 RL 引擎，因此 TRPO 是「保证不把策略改坏」这条对齐引擎思想线的源头（[[trpo-paper]] 摘要、§1）。

## 1. 要解决什么问题

用神经网络逼近策略做强化学习，当时的三大方法族都没有回答「每步更新能迈多大」（[[trpo-paper]] §1）：

- **策略迭代**（policy iteration，学值函数再贪心改进策略）：有精确值函数时保证改进，带函数逼近时无保证；
- **策略梯度**（对期望回报直接求梯度）：数据效率差——朴素目标只在「对产生数据的那个策略做一次更新」意义下成立，多步优化没有正当性，实测导致破坏性的大更新（[[ppo-paper]] §2.1 的诊断与本文一致）；
- **免梯度黑盒优化**（CEM / CMA，把回报当黑盒）：样本复杂度随参数量爆炸，大问题上失效。

步长太大，一次坏更新就可能把学到的策略毁掉；太小则采样成本不可接受。TRPO 的回答：**不用参数距离、而用策略分布的距离（KL 散度，Kullback–Leibler divergence，两概率分布差异的度量）限定更新幅度，并且给这个更新一个「不会改坏性能」的可证明下界**——这就是「置信域」：只在当前策略附近、被下界担保的区域里信任替代目标（[[trpo-paper]] §1、§3–4）。

## 2. 理论骨架：从性能差分解到单调改进下界

### 2.1 性能差分解与替代目标（式 1–4）

一切从 Kakade & Langford (2002) 的**性能差分解**开始（式 1）：新策略的回报变化可以**精确**写成「新策略轨迹上**旧策略**优势函数（advantage，$A_\pi(s,a) = Q_\pi(s,a) - V_\pi(s)$，动作比该状态平均水平好多少；$V$ 为期望回报、$Q$ 为取该动作后的期望回报）的折扣和」：

$$\eta(\tilde\pi) = \eta(\pi) + \mathbb{E}_{\tau\sim\tilde\pi}\Big[\sum_{t=0}^{\infty} \gamma^t A_\pi(s_t, a_t)\Big]$$

（$\eta(\pi)$ 为期望折扣回报，$\gamma$ 为折扣因子；期望对新策略 $\tilde\pi$ 产生的轨迹取。）读法：三个成分各司其职——$\tau\sim\tilde\pi$ 管轨迹（「走哪条路」由新策略定），$A_\pi$ 管评分（以旧策略的价值函数衡量该动作比平均水平多赚多少），$\gamma^t$ 与回报同折扣。

**为何是严格等式**（[[trpo-paper]] 附录 A，式 20–24）：把优势按定义拆开 $A_\pi(s,a) = \mathbb{E}\big[r + \gamma V_\pi(s')\big] - V_\pi(s)$，在 $\tilde\pi$ 轨迹上取期望后：

$$\mathbb{E}\Big[\sum_t \gamma^t A_\pi(s_t,a_t)\Big] = \eta(\tilde\pi) + \sum_{t\ge0}\gamma^{t+1}\,\mathbb{E}\big[V_\pi(s_{t+1})\big] - \sum_{t\ge0}\gamma^{t}\,\mathbb{E}\big[V_\pi(s_t)\big]$$

后两个和是**同一测度**下的同一列随机变量（$s_{t+1}$ 就是 $\tilde\pi$ 轨迹在 $t{+}1$ 时刻的状态），指标平移后逐项相消，只剩 $\eta(\tilde\pi) - \mathbb{E}[V_\pi(s_0)] = \eta(\tilde\pi) - \eta(\pi)$。注意：相消只要求 $V$ 项同测度，**不要求 $V$ 属于谁**——把 $V_\pi$ 换成任意函数 $\bar V$，和式都收敛到 $\eta(\text{行为策略}) - \mathbb{E}[\bar V(s_0)]$（性能差引理的一般骨架，势函数塑形同族）。退化情形互为印证：评分与轨迹出自同一策略时（$A_{\tilde\pi}$ 配 $\tilde\pi$ 轨迹、或 $A_\pi$ 配 $\pi$ 轨迹），每状态优势期望为零、总和 $0$，恰为两边之差。特例直觉：$\tilde\pi$ 仅在 $t=0$ 不同于 $\pi$、其后恢复 $\pi$，则式子退化为「改进 = 起点所改动作的优势」（一步策略改进定理）；CPI 允许每步分歧，改进累积而「分歧两次以上」贡献 $O(\alpha^2)$——Theorem 1 误差项的来源。

把轨迹期望按状态展开、换用旧策略的（未归一化折扣）访问频率 $\rho_\pi$ 加权，得**替代目标**（式 3）：

$$L_\pi(\tilde\pi) = \eta(\pi) + \sum_s \rho_\pi(s) \sum_a \tilde\pi(a|s)\, A_\pi(s,a)$$

$L_\pi$ 与 $\eta$ 在 $\tilde\pi = \pi$ 处值相等、梯度匹配（式 4）——它是 $\eta$ 的一阶近似。但**一阶近似只在 $\pi$ 的极小邻域可信**： $L_\pi(\tilde\pi) \ge 0$ 的逐状态改进并不能保证 $\eta(\tilde\pi) \ge \eta(\pi)$（摘要自陈「some states for which the expected advantage is negative」的情形）。Theorem 1 补的正是这一步。

### 2.2 CPI 下界与 Theorem 1（式 5–9）

Kakade & Langford 对**混合策略**（新策略与旧策略按 $\alpha$ 概率混合，式 5）给出下界（式 6）：$\eta(\pi_{new}) \ge L_{\pi_{old}}(\pi_{new}) - \frac{2\epsilon\gamma}{(1-\gamma)^2}\alpha^2$。TRPO 把它推广到**任意随机策略**（Theorem 1，式 8）：

$$\eta(\tilde\pi) \ge L_\pi(\tilde\pi) - \frac{4\epsilon\gamma}{(1-\gamma)^2}\,\alpha^2, \qquad \epsilon = \max_{s,a}|A_\pi(s,a)|,\ \ \alpha = D^{\max}_{\mathrm{TV}}(\pi, \tilde\pi)$$

（$D_{\mathrm{TV}}$ 为全变差距离，total variation distance，$\frac{1}{2}\sum_a|p_a - q_a|$；注意此处 $\epsilon$ 是优势的绝对值上界，与 PPO 的截断系数 $\epsilon$ 无关。）再用 $D_{\mathrm{TV}}^2 \le D_{\mathrm{KL}}$ 换成 KL 形式（式 9）：

$$\eta(\tilde\pi) \ge L_\pi(\tilde\pi) - C\, D^{\max}_{\mathrm{KL}}(\pi, \tilde\pi), \qquad C = \frac{4\epsilon\gamma}{(1-\gamma)^2}$$

**这就是单调改进的钥匙**（式 10）：记 $M(\tilde\pi) = L_\pi(\tilde\pi) - C\,D^{\max}_{\mathrm{KL}}(\pi,\tilde\pi)$，则 $\eta(\pi) = M(\pi)$ 且 $M(\tilde\pi) \le \eta(\tilde\pi)$——$M$ 是 $\eta$ 在 $\pi$ 处**相切的下界**（minorization–maximization，MM 算法，一类「最大化目标函数局部下界」的迭代方法，期望最大化 EM 是其特例）。于是 $\eta(\pi_{i+1}) \ge M(\pi_{i+1}) \ge M(\pi_i) = \eta(\pi_i)$，最大化 $M$ 的迭代永不退步（Algorithm 1：带 KL 惩罚的策略迭代）。

### 2.3 微走查：α-coupling——误差为什么是 $O(\alpha^2)$

下界的证明（附录 A）核心是 **α-coupling**（Definition 1）：让两个策略**共享随机数**采样，使得二者采出不同动作的概率 $\le \alpha$。取 $\pi = (0.8, 0.2)$、$\tilde\pi = (0.7, 0.3)$（某状态上两动作概率），则 $\alpha = D_{\mathrm{TV}} = 0.1$：

- **共享均匀随机数** $u \sim U[0,1)$：$\pi$ 采动作 0 当且仅当 $u < 0.8$、$\tilde\pi$ 当且仅当 $u < 0.7$ → 二者分歧当且仅当 $u \in [0.7, 0.8)$，概率恰为 $0.1 = \alpha$ ✓
- 前提检查：优势函数必须在 $\pi$ 下零均值（$\mathbb{E}_{a\sim\pi}[A_\pi(s,a)] = V_\pi - V_\pi = 0$），故不能随便取 $A = (1, -1)$（均值 0.6，非法）；合法示例 $A = (-0.25,\ 1)$：$0.8\times(-0.25) + 0.2\times 1 = 0$ ✓
- 该状态下 $\tilde\pi$ 对 $\pi$ 的期望优势（式 25）：$\bar A = \sum_a \tilde\pi(a) A(a) = 0.7\times(-0.25) + 0.3\times 1 = 0.125$；经 coupling 视角算：$\alpha \times (A(1) - A(0)) = 0.1 \times 1.25 = 0.125$ ✓ 两路一致
- **Lemma 2**（式 28）：$|\bar A| \le 2\alpha\epsilon = 2 \times 0.1 \times 1 = 0.2 \ge 0.125$ ✓

直觉：$L_\pi$ 只记了轨迹上**第一次**策略分歧的优势，之后两条轨迹各走各的、信息丢失——误差全由「两次及以上分歧」贡献，而「分歧两次」的概率是 $O(\alpha^2)$。这就是下界里 $\alpha^2$ 项的来源（[[trpo-paper]] 附录 A 开篇的自述）。

## 3. 从理论到可实现：三次近似（§4–§6，附录 C）

Theorem 1 直接实现不可行，TRPO 连做三次近似（§7 的三条自总结）：

1. **惩罚改硬约束**（式 11）：Algorithm 1 用的是惩罚形式 $L - C\,D^{\max}_{\mathrm{KL}}$，但理论给定的 $C$ 量级极大（走查见 §4）——惩罚项吞掉一切、步长被罚到极小。改为**硬约束**：$\max_\theta L_{\theta_{old}}(\theta)\ \ \text{s.t.}\ D^{\max}_{\mathrm{KL}}(\theta_{old},\theta) \le \delta$（式 11）
2. **max KL 改平均 KL**（式 12）：逐状态约束的数量等于状态数，神经网络策略不可解；启发式换成**平均 KL** $\bar D_{\mathrm{KL}}(\theta_1,\theta_2) = \mathbb{E}_{s\sim\rho}\big[D_{\mathrm{KL}}\big]$。实验的 max KL 消融（§6.1）支持这一近似：平均 KL 版效果与理论版相似、只是稍慢
3. **精确解改共轭梯度 + 线搜索**（附录 C，式 55）：对目标做线性近似、对 $\bar D_{\mathrm{KL}}$ 做二次近似（其 Hessian 是 **Fisher 信息矩阵**（FIM，KL 散度的二阶泰勒系数——度量「分布参数变一点、分布本身动多少」的曲率矩阵））；**共轭梯度**（conjugate gradient，只依赖「矩阵 × 向量」的迭代法解线性方程，不需显式存矩阵）解 Fisher-vector product，$k=10$ 次迭代已够、且可用 10% 数据子采样——整个求解只比算梯度贵一点；最后**回溯线搜索**（backtracking line search，步长按指数收缩直到同时满足「目标改进 + 约束成立」）兜底：「没有它，算法偶尔发生灾难性的性能退化」（附录 C）

两个统一视角（§7，式 17–18）：把目标线性化 + KL 二次化一步走到底就得到**自然策略梯度**（natural policy gradient，Kakade 2002——用 Fisher 逆矩阵转正梯度方向的经典更新 $\theta_{new} = \theta_{old} + \tfrac{1}{\lambda} A(\theta_{old})^{-1}\nabla_\theta L$，步长 $\tfrac{1}{\lambda}$ 当超参、不逐步强制约束）；换 $\ell_2$ 约束就退化回普通策略梯度。TRPO 与自然梯度的差别只在「**每步真的满足约束**」——实验显示这个差别在大问题上显著（§6.1）。

## 4. 手工走查：2 状态 MDP 上验证式 1 与 Theorem 1

以下全部数字经本机精确复算（2×2 线性方程组直接解，无近似）。MDP：状态 $\{s_0, s_1\}$（$s_0$ 为起点，$s_1$ 吸收、奖励 0）；$s_0$ 上动作 $a_0 \to (s_0: 0.5,\ s_1: 0.5)$ 奖励 0，$a_1 \to (s_0: 0.2,\ s_1: 0.8)$ **奖励 1**；$\gamma = 0.9$。旧策略 $\pi = (0.5, 0.5)$、新策略 $\tilde\pi = (0.2, 0.8)$（$s_0$ 上；$s_1$ 上动作无影响）。

**第一步：算出精确量。** $\eta(\pi) = V_\pi(s_0) = 0.7299$；$Q_\pi(s_0,\cdot) = (0.3285,\ 1.1314)$，$A_\pi(s_0,\cdot) = (-0.4015,\ +0.4015)$（在 $\pi$ 下零均值 ✓）。$\tilde\pi$ 把概率挪给 $a_1$（即拿即时奖励但更快离开 $s_0$）：$\eta(\tilde\pi) = 1.0444$，真实改进 $\Delta\eta = 0.3145$。

**第二步：验证性能差分解（式 1）。** 新策略轨迹上旧优势的折扣和 $= \tilde\rho(s_0)\, g = 1.3055 \times 0.2409 = 0.3145$，其中 $g = 0.2\times(-0.4015) + 0.8\times 0.4015 = 0.2409$ 为 $s_0$ 上的期望优势（$s_1$ 上为 0）——与 $\Delta\eta = 0.3145$ **严格相等** ✓。恒等式不是近似。

**第三步：算替代目标，看它高估。** $L_\pi(\tilde\pi) = 0.7299 + \rho_\pi(s_0)\, g = 0.7299 + 1.4599 \times 0.2409 = 1.0816 > \eta(\tilde\pi) = 1.0444$，**高估 0.0372**。原因一眼可见：$L$ 用**旧**策略的访问频率 $\rho_\pi$ 加权，而 $\pi$ 在优势集中的 $s_0$ 上停留更久（$\rho_\pi(s_0) = 1.4599$ vs $\tilde\rho(s_0) = 1.3055$）——新策略恰好在「优势大的状态」上待得更少时，替代目标就吹牛。这正是需要下界修正项的原因。

**第四步：验证 Theorem 1（式 8/9）。** $\epsilon = 0.4015$，$C = 4\epsilon\gamma/(1-\gamma)^2 = 144.53$；$\alpha = D_{\mathrm{TV}} = 0.3$，$C\alpha^2 = 13.01$。界：$\eta(\tilde\pi) \ge 1.0816 - 13.01 = -11.93$，实际 $1.0444$ ✓ 成立——但**松弛得离谱**（实际误差 0.0372，界允许到 13）。结论：下界的价值在**方向担保**（单调改进、绝不改坏），不在数值紧致。

**第五步：验证「理论惩罚不可用」（§4 的主张）。** 取典型 $\gamma = 0.99$、$\epsilon = 1$：$C = 4 \times 0.99 / 0.01^2 = 39600$；哪怕极小的 $\alpha = 0.05$ 的一步，惩罚 $C\alpha^2 = 99$——相对通常 $O(1)$–$O(10)$ 量级的回报，任何有意义的更新都会被这个惩罚项否决。这就把 §4「用理论系数则步长极小 → 改硬约束」的主张变成了数字。

## 5. 采样方案：single path vs vine（§5，Figure 1）

理论到采样之间需要用数据估计 $Q_\pi$ 与目标/约束（式 13–14）。TRPO 设计了两套方案：

| | single path（单路径） | vine（藤式） |
| --- | --- | --- |
| 采法 | 按 $\rho_0$ 起点正常滚轨迹 | 采一批「rollout set」状态 $s_{i}$，每状态另采 $K$ 个动作 |
| $Q$ 估计 | 整条轨迹折扣回报 | 短 rollout + 自归一化重要性采样（式 16） |
| 方差 | 较高 | 同量 $Q$ 样本下显著更低（共同随机数 + 同状态多动作对照） |
| 环境要求 | 无重置需求，可上实物 | **可重置到任意状态**——仿真专属 |

（重要性采样：用易采的分布 $q$ 代采难采的分布，再按 $\pi/q$ 加权修正；自归一化 = 权重归一成和为 1，降偏差。共同随机数：不同候选动作共用同一随机数序列，把「环境随机性」从对比中消掉。）vine 每状态要做多次 rollout，代价是采样次数更多、且**只能用于可重置环境**（论文原文：single path「can be directly implemented on a physical system」，vine 限于可重置仿真）。Atari 实验两者都跑：vine 每迭代 400K 模拟步 vs single path 100K（Table 3）。

![[trpo-fig1-single-path-vs-vine.png]]
> Figure 1（[[trpo-paper]] §5）：左——single path，轨迹自然经过的每个「状态–动作」对都进目标；右——vine，在 rollout set 的状态 $s_n$ 上分叉采 $a_1, a_2$ 各自短 rollout（虚线），两条 rollout 用共同随机数（CRN）对齐环境随机性。

## 6. 实验证据（§8）

### 6.1 MuJoCo 运动控制（Figure 4，Table 2）

四个任务（cartpole 倒立摆基线、swimmer 游泳、hopper 单脚跳、walker 行走），全神经网络策略（最小 364 参数、最大 8206 参数，Table 2），**$\delta = 0.01$ 全部实验通用**。对手按消融目的分三类：natural gradient（固定惩罚 $\lambda$ 替代 KL 约束——检验「约束 vs 惩罚」）、empirical FIM（Fisher 用梯度协方差估计替代解析式）、max KL（约束用 max KL 替代平均 KL——检验 §3 的第二次近似）；外加免梯度的 CEM / CMA。结果（Figure 4，五次运行平均）：

- **vine 与 single path TRPO 包揽全部任务前二**（「yielding the best and second solutions」）——vine 多数任务略优
- natural gradient 只在两个简单任务（cartpole / swimmer）表现好，**hopper / walker 学不出前进增益**——「每步满足约束」与「固定步长」的差别在大问题上显著
- max KL 变体学得稍慢（max KL 约束更苛刻）——平均 KL 近似的效果与理论版相似，近似站得住
- CEM / CMA 在大问题上失效——免梯度方法的样本复杂度随参数量爆炸
- Figure 4 图例中另有 RWR 曲线（正文未讨论其结果）

![[trpo-fig4-mujoco-learning-curves.png]]
> Figure 4（[[trpo-paper]] §8.1）：四任务学习曲线（五次运行平均，误差棒）。蓝（vine）与绿（single path）在 hopper / walker 上明显领先；natural gradient（橙）在两个简单任务上追平、在 hopper/walker 上停滞于「站得住但不前进」的分数（hopper/walker 的 −1 分 = 原地不动不摔倒）。

### 6.2 Atari 七局：从原始图像学（Table 1，Table 3）

七款 Atari 游戏（与 DQN [Mnih et al. 2013]、UCC-I [Guo et al. 2014] 同一设定）、策略为 33,500 参数的 CNN（两层 16 通道卷积 + 一层 20 单元全连接）、500 迭代 ≈30 小时（16 核计算机）。性能对比（Table 1 转录）：

| | B. Rider | Breakout | Enduro | Pong | Q*bert | Seaquest | S. Invaders |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random | 354 | 1.2 | 0 | −20.4 | 157 | 110 | 179 |
| Human | 7456 | 31.0 | 368 | −3.0 | 18900 | 28010 | 3690 |
| Deep Q-Learning | 4092 | 168.0 | 470 | 20.0 | 1952 | 1705 | 581 |
| UCC-I | 5702 | 380 | 741 | 21 | 20025 | 2995 | 692 |
| **TRPO single path** | 1425.2 | 10.8 | 534.6 | 20.9 | 1973.5 | 1908.6 | 568.4 |
| **TRPO vine** | 859.5 | 34.2 | 430.8 | 20.9 | 7732.5 | 788.4 | 450.2 |

读数：vine 在 Q*bert 上超 DQN 约 4×（7732.5 vs 1952）、Pong 近满分（20.9，满分 21）；但 Beam Rider（859.5 vs 4092）与 Seaquest（788.4 vs 1705）明显落后。论文自评诚实：「只在部分游戏超越先前方法，但**一致取得合理分数**」。这一节的意义不在刷榜——当时该设定只有 DQN 与 UCC-I 两个先例——而在**通用性**：同一套方法、同一组超参（$\delta=0.01$），既训运动控制又从原始像素学游戏。

## 7. 关键性质与陷阱

- **单调改进担保的是方向、不是速度**：走查（§4）显示下界可以松弛两个数量级——它保证「按 $M$ 优化不退步」，不保证「改进多快」。实践价值在防灾难性更新，效率要靠 $\delta$ 的选取
- **$\delta = 0.01$ 意外地通用**：全部实验（四任务 + 七游戏，Table 2 / Table 3）共用一个置信域半径，没有任何逐任务调参——「对超参不敏感」的最早实证，也是后来 PPO 声称 $\epsilon$ 稳健（0.1/0.2/0.3 全部 ≥0.70，[[ppo]] §5）的叙事先声
- **vine 的可重置要求是硬门槛**：论文自己写明 vine 限于可重置仿真、single path 才能上实物——后继方法的实践全部落在 single path 一系（PPO 的 N 个并行 actor 各采 T 步即此形态，[[ppo-paper]] Algorithm 1）
- **与参数共享不兼容**：Fisher 矩阵只对策略参数定义，策略 / 值函数共享底座的架构用不了——PPO 列为对 TRPO 的头号工程批评（[[ppo-paper]] §1，[[ppo]] §1）
- **优势估计是 Monte-Carlo 折扣回报**：vine 每批 500–2500 个 $Q$ 值（Table 2）；后来同组的 GAE（截断优势估计器）才是 PPO 采用的估计方式（[[ppo]] §3.2）
- **理论忽略优势估计误差**：Theorem 1 假设 $A_\pi$ 精确已知；论文在 §7 明确此近似未处理（Kakade & Langford 的原始推导处理过）

## 8. 与 PPO 对照：置信域的一阶化（本 wiki 谱系的关键一站）

| | TRPO | PPO |
| --- | --- | --- |
| 防「步子过大」的机制 | 显式 KL 约束（每步解约束优化问题） | clip 截断目标的悲观下界形状（机制走查见 [[ppo]] §2） |
| 求解器 | 目标线性化 + Fisher 二次约束 + 共轭梯度 + 线搜索（附录 C） | 普通 SGD / Adam，多 epoch 小批量 |
| 二阶信息 | 需要（Fisher-vector product，$k$ 次 CG） | 完全不需要 |
| 策略 / 值函数共享参数 | 不兼容 | 兼容（[[ppo-paper]] §1 头号卖点） |
| 同批数据的用法 | 单次约束求解内使用（CG 迭代 + 线搜索） | K epochs 反复小批量复用（clip 即复用的保险） |
| 关键超参 | $\delta$（置信域半径，实测 0.01 通吃） | $\epsilon$（截断半宽，实测 0.2 稳健） |

叙事链：TRPO 证明了「约束分布距离 → 单调改进」，但求解器（Fisher + CG）与架构限制（不能共享参数）使其复杂；PPO 保留「越界要受罚」的思想、把它**编译进目标函数的形状**（min + clip），整个置信域变成几行一阶代码。PPO 的自适应 KL 惩罚变体（[[ppo]] §4）则是 TRPO 惩罚形式（Algorithm 1）的直接后裔。两年后 PPO 成为 [[rlhf]] 第 3 阶段的引擎（[[instructgpt-paper]] §3.5），下游 [[grpo]]（去 critic）、[[dpo]]（免 RL 折叠）皆以它为共同基线——TRPO 是这条谱系的理论源头。

## 9. 在本 wiki 的语境

- **[[ppo]]**：一阶化后继；两页对照读——本页 §2（下界与约束）对应其 §2（截断目标的悲观下界），本页 §8 是对照表
- **[[rlhf]] / [[instructgpt]]**：应用范式侧——RLHF 的 RL 引擎谱系（TRPO → PPO → 工程变体 PPO-ptx）的理论起点
- **[[grpo]] / [[dpo]]**：谱系下游；两者的「近端」约束思想（clip / 参考模型 KL）皆可上溯至本页的置信域

## 来源

- [[trpo-paper]]（§1 动机、§2 式 1–4 性能差分解、§3 Theorem 1 与 Algorithm 1、§4 式 11–12 置信域、§5 式 13–16 采样与 Figure 1、§6 附录 C 实用算法、§7 式 17–18 统一视角、§8 Table 1–3 与 Figure 4–5、附录 A 证明）

## 相关

- [[ppo]]（一阶化后继）｜ [[rlhf]]（应用范式）｜ [[grpo]]（去 critic 下游）｜ [[dpo]]（免 RL 下游）
