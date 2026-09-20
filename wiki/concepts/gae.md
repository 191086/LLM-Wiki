---
type: concept
title: GAE（广义优势估计）
created: 2026-09-20
updated: 2026-09-20
tags:
  - reinforcement-learning
  - policy-gradient
  - rlhf
sources: 3
---

# GAE（Generalized Advantage Estimation，广义优势估计）

**定义**：Schulman et al.（UC Berkeley，ICLR 2016）提出的**优势函数估计器**——把 $k$ 步优势估计按 $(1-\lambda)\lambda^{k-1}$ 指数加权平均，收拢为 TD 残差的折扣和 $\hat A_t = \sum_l (\gamma\lambda)^l \delta^V_{t+l}$（式 16）；用 $\gamma,\lambda$ 两个参数在「单步 TD（低方差、值函数不准则偏）」与「蒙特卡洛回报减 baseline（对值函数误差免疫、高方差）」之间连续插值，是 [[trpo|TRPO]] / [[ppo|PPO]] 家族的标准优势估计器——PPO 用的截断 GAE 即其固定时域版（[[ppo]] §3.2），因而也是 [[rlhf]] 第 3 阶段 RL 引擎的信号源（[[gae-paper]] 摘要、§1、§3）。

## 1. 要解决什么问题

策略梯度对期望回报直接求导、天然兼容神经网络，但有两个老大难（[[gae-paper]] §1）：

- **方差随时域增长**：动作的效果与前后所有动作的效果混杂（confounding），梯度估计的方差随时长恶化——费样本；
- **偏差更险**：actor-critic 方法（actor 学策略、critic 学值函数，用值函数替代经验回报来降方差）确实方差低，但引入偏差——**无限样本下偏差仍可导致不收敛、或收敛到连局部最优都不是的差解**。

背后是信用分配问题（credit assignment：奖励迟到时，怎么把功劳 / 过失记到早先的动作上；行为学文献称 distal reward）。值函数是解药——它让「这个动作好不好」在迟到奖励到来之前就可估计。GAE 的问题意识：**用值函数降方差时，偏差能不能变成一个旋钮？** 答案是两个旋钮：$\gamma$ 与 $\lambda$（[[gae-paper]] §1–3）。

## 2. 理论底座：优势、γ-just 与「baseline 不偏差」

### 2.1 策略梯度的六种写法（式 1）

策略梯度都可写成 $g = \mathbb{E}\big[\sum_t \Psi_t \nabla_\theta\log\pi_\theta(a_t|s_t)\big]$（式 1），差别全在权重项 $\Psi_t$ 用什么「评分」（[[gae-paper]] §2）：

| $\Psi_t$ | 名称 | 特点 |
| --- | --- | --- |
| $\sum_{t'=0}^{\infty} r_{t'}$ | 整条轨迹回报 | 方差最大 |
| $\sum_{t'=t}^{\infty} r_{t'}$ | 后续回报 | 弃掉 $a_t$ 之前无关项 |
| 上式 $-\,b(s_t)$ | baseline 版 | 方差降、仍无偏 |
| $Q^{\pi}(s_t,a_t)$ | 动作价值 | 须知道 $Q$ |
| $A^{\pi}(s_t,a_t)$ | **优势** | 方差几乎最低 |
| $r_t + \gamma V^{\pi}(s_{t+1}) - V^{\pi}(s_t)$ | TD 残差 | 单步、依赖 $V$ |

（$V^\pi$：从状态出发的期望回报；$Q^\pi$：取该动作后的期望回报；优势 $A^\pi = Q^\pi - V^\pi$：动作比该状态的平均水平好多少。）优势之所以「几乎方差最低」：梯度方向恰好是「推高好于策略平均的动作、压低差于平均的」——$\Psi_t > 0$ 当且仅当该方向推高 $\pi(a_t|s_t)$。但优势函数未知、须估计，本文全部内容就是**怎么估**（[[gae-paper]] §2）。

### 2.2 γ 是算法参数，不是问题设定

本文刻意在**不折扣**问题上工作（目标 $\sum_t r_t$ 有限），把 $\gamma$ 当**方差缩减参数**引入：估计 $g^\gamma = \mathbb{E}\big[\sum_t A^{\pi,\gamma}(s_t,a_t)\,\nabla_\theta\log\pi_\theta(a_t|s_t)\big]$（式 6），$A^{\pi,\gamma}$ 用折扣价值函数定义。折扣问题本身可作为特例（折扣吸收进奖励函数）（[[gae-paper]] §2）。

### 2.3 γ-just：什么估计器不引入偏差（Def 1 + Prop 1）

估计器 $\hat A_t$（可以是整条轨迹的函数）是 **$\gamma$-just** 的，若把它代进式 6 估计 $g^\gamma$ 不引入偏差（Def 1）。充分条件（Prop 1，附录 B 证明）：

$$\hat A_t = \underbrace{Q_t(s_{t:\infty}, a_{t:\infty})}_{\gamma\text{-折扣 Q 的无偏估计}} - \underbrace{b_t(s_{0:t}, a_{0:t-1})}_{\text{只依赖 }a_t\text{ 之前}}$$

证明的引擎只有一行事实：**score 函数零均值**——$\mathbb{E}_{a\sim\pi}\big[\nabla_\theta\log\pi(a|s)\big] = \sum_a \nabla_\theta\pi(a|s) = \nabla_\theta 1 = 0$。对过去可测的 $b_t$ 取条件期望即得 $\mathbb{E}[\nabla\log\pi \cdot b_t] = 0$——**任何 baseline（哪怕错得离谱）都不引入偏差**；$Q_t$ 无偏的部分则逐条件期望传下去。四个现成的 $\gamma$-just 估计器：折扣经验回报 $\sum_l \gamma^l r_{t+l}$、$Q^{\pi,\gamma}$、$A^{\pi,\gamma}$、以及**正确** $V$ 下的 TD 残差（[[gae-paper]] §2、附录 B）。

## 3. 推导：TD 残差的指数折扣和（式 16）

### 3.1 从 δ 到 k 步估计（式 10–15）

设 $V$ 为近似值函数，定义 TD 残差（temporal-difference residual：单步「预测修正」量）$\delta^V_t = r_t + \gamma V(s_{t+1}) - V(s_t)$。若 $V = V^{\pi,\gamma}$ 精确，则 $\mathbb{E}_{s_{t+1}}[\delta^V_t] = A^{\pi,\gamma}(s_t,a_t)$——TD 残差是优势的**无偏**估计（式 10）；$V$ 不准时它有偏。把 $k$ 个 $\delta$ 折扣相加（式 11–13 是 $k=1,2,3$ 的展开，式 14 一般式）：

$$\hat A_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta^V_{t+l} = -V(s_t) + r_t + \gamma r_{t+1} + \cdots + \gamma^{k-1} r_{t+k-1} + \gamma^{k} V(s_{t+k}) \tag{式 14}$$

中间的 $V$ 项**递缩相消**（telescoping：相邻两项的 $+\gamma^l V(s_{t+l})$ 与 $-\gamma^l V(s_{t+l})$ 抵消），只剩「k 步经验回报 − baseline $V(s_t)$ + 尾部 $\gamma^k V(s_{t+k})$」。$k\to\infty$ 时尾部被折扣压没（式 15）：

$$\hat A_t^{(\infty)} = \sum_{l=0}^{\infty} \gamma^l \delta^V_{t+l} = -V(s_t) + \sum_{l=0}^{\infty} \gamma^l r_{t+l} \tag{式 15}$$

即蒙特卡洛回报（Monte-Carlo：直接把采样到的实际奖励按折扣加起来当回报用，不借助任何学到的函数）减 baseline——高方差（求和项多）但由 §2.3 知它对任意 $V$ 都 $\gamma$-just。$k$ 越大偏差越小、方差越大：**k 本身就是偏差–方差旋钮**。

### 3.2 指数加权收拢（式 16）

GAE 不选某个 $k$，而是对全部 $k$ 步估计做指数加权平均——近期估计权重大：

$$\hat A_t^{\mathrm{GAE}(\gamma,\lambda)} := (1-\lambda)\Big(\hat A_t^{(1)} + \lambda\hat A_t^{(2)} + \lambda^2 \hat A_t^{(3)} + \cdots\Big) = \sum_{l=0}^{\infty} (\gamma\lambda)^l\, \delta^V_{t+l} \tag{式 16}$$

推导只需对每个 $\delta^V_{t+l}$ 数它被哪些 $k$ 步估计包含：凡 $k > l$ 的 $\hat A^{(k)}$ 都含它，指数权重合计 $(1-\lambda)\sum_{k=l+1}^{\infty}\lambda^{k-1} = \lambda^l$；连同 $\delta^V_{t+l}$ 自身带的 $\gamma^l$ 折扣，净权重恰为 $(\gamma\lambda)^l$——**等价于把折扣换成更陡的 $\gamma\lambda$ 后对 TD 残差直接求和**。构造与 TD($\lambda$)（Sutton 经典的价值估计方法：对 n 步回报做同样的 $\lambda$ 指数加权，借资格迹 eligibility traces——按「近期访问的状态多记功劳」指数衰减的记账向量——实现在线计算）同构，但估计对象从价值函数换成了优势函数（[[gae-paper]] §3）。

### 3.3 两个端点（式 17–18）

$$\mathrm{GAE}(\gamma,0):\quad \hat A_t = \delta^V_t = r_t + \gamma V(s_{t+1}) - V(s_t) \tag{式 17}$$

$$\mathrm{GAE}(\gamma,1):\quad \hat A_t = \sum_{l=0}^{\infty} \gamma^l \delta^V_{t+l} = \sum_{l=0}^{\infty} \gamma^l r_{t+l} - V(s_t) \tag{式 18}$$

- **$\lambda = 0$**：单步 TD。$V = V^{\pi,\gamma}$ 时才 $\gamma$-just，否则偏——但方差通常低得多；
- **$\lambda = 1$**：经验回报 − baseline。**无论 $V$ 多不准都 $\gamma$-just**（§2.3 的 baseline 性质），但求和项多、方差高。

$0<\lambda<1$ 在两点之间连续插值，$\lambda$ 就是那个旋钮（[[gae-paper]] §3）。

## 4. γ 与 λ 的不对称分工

两个参数都在调偏差–方差，但**偏差来源不同**（[[gae-paper]] §3 末）：

| | $\gamma$ | $\lambda$ |
| --- | --- | --- |
| 主要作用 | 定价值函数 $V^{\pi,\gamma}$ 的 scale（不依赖 λ） | 截断 TD 残差链（有效时域 $\approx 1/(1-\gamma\lambda)$） |
| 引入偏差的条件 | $\gamma<1$ **无论 $V$ 多准都偏**（估计的是 $g^\gamma \ne g$） | $\lambda<1$ **只在 $V$ 不准时偏** |
| 经验最优值 | 较高（cart-pole 0.96–0.99、双足 0.99–0.995） | 较低（0.92–0.99） |

最优 λ 远低于最优 γ 的解释：对合理准确的 $V$，λ 引入的偏差远小于 γ（λ=1 端点对 $V$ 误差免疫，λ 稍降只吃二阶代价）。这一不对称是 GAE 实用性的核心：**放心把 λ 压低买方差，γ 保持接近 1**。

## 5. 数值示例：λ 如何抹平值函数误差（本机复算）

以下全部数字经本机精确复算（闭式期望 + 20 万次 Monte-Carlo）。**MDP**：$s_0$ 上两动作——$a_0$ 以奖励 0 转入吸收态 $X$（此后每步奖励 1），$a_1$ 以奖励 0 转入吸收态 $Y$（此后奖励 0）；$\gamma=0.9$；策略为 softmax，$\pi=(0.5,0.5)$。真值：$V^{\pi,\gamma}(X)=\frac{1}{1-\gamma}=10$，$V(s_0)=4.5$，优势 $A(a_0)=+4.5,\ A(a_1)=-4.5$，真梯度 $g^\gamma=(2.25,-2.25)$。

**故意用错的值函数**：$\hat V(X)=8$（低估 20%）、$\hat V(s_0)=4$、$\hat V(Y)=0$。各 $\lambda$ 的梯度估计：

| $\lambda$ | $\hat A(a_0)$ | $\hat A(a_1)$ | $\hat g_x$ | 占真值比例 |
| --- | --- | --- | --- | --- |
| $0$ | $3.2$ | $-4$ | $1.800$ | $0.800$ |
| $0.5$ | $3.364$ | $-4$ | $1.841$ | $0.818$ |
| $0.95$ | $4.379$ | $-4$ | $2.095$ | $0.931$ |
| $1$ | $5.000$ | $-4$ | $2.250$ | $1.000$ |

三个读数（每行都可手算复核）：

- **$\lambda=0$ 的偏差精确等于 $V$ 的低估比例**：$\hat A(a_0)=0+0.9\times 8-4=3.2$，而真优势 4.5 里 $V(s_0)$ 是纯 baseline（§2.3，不偏差），信号全由 $\gamma\hat V(X)=7.2$ 扛——低估 20% 则信号打八折。**下一状态价值的误差乘性进入、baseline 的误差完全消掉**；
- **$\lambda=1$ 在 $V$ 错 20% 时仍精确无偏**：$\hat A(a_0)=\sum\gamma^l r - 4 = 9-4=5$，$\mathbb{E}[5\cdot\nabla\log\pi(a_0)] + \mathbb{E}[(-4)\cdot\nabla\log\pi(a_1)] = g^\gamma$ 严格成立（式 18 的 $\gamma$-just 性数值兑现）；
- **中间 λ 的偏差单调内插**（0.800→0.818→0.931→1.000）：$\lambda$ 确实在连续调节。

**换正确的 $V$**：$\delta_0(a_0) = 0+9-4.5 = 4.5 = A(a_0)$（式 10 兑现），且 $X$ 内后续 $\delta_l = 1+0.9\times 10-10 = 0$——**$V$ 精确时 TD 残差链在首步之后全为零，GAE 对任意 $\lambda$ 都等于真优势**（§6 将解释这正是「响应函数被完全压缩」）。

**三条恒等式逐条验证**（$\lambda=0.5$、$a_0$ 轨道）：指数加权定义与式 16 闭式相等（两侧 $3.3636363636$）；式 14 的递缩相消展开对 $k=1,2,3,6$ 逐一相等；塑形奖励 $\tilde r_l = r_l + \gamma\Phi(s_{l+1})-\Phi(s_l)$（$\Phi=\hat V$）逐项等于 $\delta_l$（$3.2, 0.2, 0.2, \ldots$）。

**方差端**（把 $X$ 内奖励改为 iid Bernoulli(0.5)，其余同，Monte-Carlo 20 万条、用正确 $V$——此时各 $\lambda$ 均无偏，均值都 $\approx 2.25$）：

| $\lambda$ | $0$ | $0.5$ | $0.95$ | $1$ |
| --- | --- | --- | --- | --- |
| $\hat A(a_0)$ 样本方差 | $0.000$ | $0.064$ | $0.678$ | $1.068$ |

$\lambda=0$ 只看一步、方差为零；$\lambda\to1$ 要吞下整条轨迹的随机性——**偏差降、方差升**，两表合起来就是 GAE 的全部主张。

## 6. 塑形视角：先压缩响应时程，再截断长延迟噪声（式 20–26）

**奖励塑形**（reward shaping，Ng et al. 1999）：给奖励加势能差——

$$\tilde r(s,a,s') = r(s,a,s') + \gamma\Phi(s') - \Phi(s) \tag{式 20}$$

折扣和相消后 $\sum_l \gamma^l \tilde r = \sum_l \gamma^l r - \Phi(s_t)$（式 21）——$Q,\tilde V$ 整体平移、**优势不变**（式 22–24）。取 $\Phi = V$ 时塑形奖励恰是 TD 残差 $\tilde r = \delta^V$，于是

$$\sum_{l=0}^{\infty} (\gamma\lambda)^l \tilde r_{t+l} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta^V_{t+l} = \hat A_t^{\mathrm{GAE}(\gamma,\lambda)} \tag{式 25}$$

**GAE = 塑形后的奖励以更陡折扣 $\gamma\lambda$ 求和**。为什么这有用？用**响应函数**（response function）量化信用分配的时间跨度（式 26）：

$$\chi(l; s_t, a_t) = \mathbb{E}[r_{t+l} \mid s_t, a_t] - \mathbb{E}[r_{t+l} \mid s_t] \tag{式 26}$$

（动作对 $l$ 步之后奖励的平均影响减去不动作时的基准——$A^{\pi,\gamma} = \sum_l \gamma^l \chi(l)$ 把优势按时间分解。）两个折扣现在有了分工（[[gae-paper]] §4）：

- $\gamma < 1$ ≈ 丢弃 $l \gg 1/(1-\gamma)$ 的响应项——若动作的影响「几步后被遗忘」（$\chi$ 快衰减），这个近似就便宜；
- $\Phi = V^{\pi,\gamma}$ 精确时，塑形把响应完全压到 $l=0$（$\mathbb{E}[\tilde r_{t+l}|s_t,a_t] = \mathbb{E}[\tilde r_{t+l}|s_t] = 0$ 对 $l>0$，§5 数值示例里「后续 $\delta\equiv 0$」正是它）；$V$ 近似时**部分**压缩——残余的长延迟项是纯噪声，用 $\gamma\lambda$ 截掉（忽略 $l \gg 1/(1-\gamma\lambda)$ 的 $\delta$ 项）。

与 Ng et al. 的差别：他们证的是折扣目标下塑形不改变最优策略；本文目标是**不折扣**的、$\gamma$ 是算法参数——塑形 + 双折扣是方差缩减工具而非问题变换（[[gae-paper]] §4）。

## 7. 值函数的训练与更新顺序（式 28–30）

GAE 依赖 $V$，$V$ 怎么训？非线性逼近器下最简单是 Monte-Carlo 回归 $\min_\phi \sum_n \|V_\phi(s_n) - \hat V_n\|^2$（式 28，目标 $\hat V_n = \sum_l \gamma^l r$，即 TD(1)）。本文的增强：**值函数也用置信域**（trust region：限制每步更新幅度、防过拟合最新一批数据的优化区域概念，[[trpo]] 的核心机制）——约束新旧值函数的平均平方差除以 $2\sigma^2$（$\sigma^2$ 为旧值函数的残差方差）不超过 $\epsilon$（式 29）；该约束等价于「把 $V$ 解释为条件 Gaussian（均值 $V_\phi$、方差 $\sigma^2$）时的平均 KL 约束」。线性化目标 + Gauss-Newton 矩阵 $H = \frac{1}{N}\sum_n j_n j_n^\top$（$j_n = \nabla_\phi V_\phi(s_n)$；它是目标 Hessian 的近似、差一个 $\sigma^2$ 因子恰是 Fisher 信息矩阵——「分布参数动一点、分布本身动多少」的曲率）二次近似约束，共轭梯度（只依赖矩阵 × 向量乘积的线性方程迭代解法）出方向、按约束重标定步长（式 30）——**与策略侧 TRPO 更新（式 31）同一套数值机器**。脚注 2：也试过 TD($\lambda$) 式目标 $\hat V^\lambda = V_{\phi_{old}} + \sum_l (\gamma\lambda)^l\delta$，与 Monte-Carlo 无差别（[[gae-paper]] §5）。

完整算法（§6.1 伪代码）：采样 N 步 → 用**当前** $V_{\phi_i}$ 算 $\delta$、$\hat A$ → TRPO 更新 $\theta$（式 31）→ 置信域更新 $\phi$。顺序有讲究：若先更新值函数且它过拟合到 $\delta \equiv 0$，策略梯度估计直接归零——所以策略更新必须用更新前的值函数（[[gae-paper]] §6.1）。

## 8. 实验证据：中间值最优

- **cart-pole 网格扫描（21 seeds，Figure 2）**：$\gamma \times \lambda$ 双向扫描，**最优在中间值**——$\gamma\in[0.96,0.99]$、$\lambda\in[0.92,0.99]$；两端（$\lambda=0$ 纯 TD、$\lambda=1$ 纯 MC）与 No VF（时间相关 baseline：按批内各时刻平均回报、不依赖状态）都更差（[[gae-paper]] §6.3.1）
- **3D 双足行走（9 seeds，1000 迭代）**：最优 $\gamma\in[0.99,0.995]$、$\lambda\in[0.96,0.99]$；每 trial 约 2 小时 / 16 核；学到快速平滑稳定的步态。实时等效：$0.01\,\text{s/步} \times 50000\,\text{步/批} \times 1000\,\text{批} = 5.8$ 天——真机并行学习可信可行（§6.3.2）
- **四足与起身（各 5 seeds，固定 $\gamma=0.995$）**：四足 $\lambda=0.96$ 胜 $\lambda=0$ 与 No VF；起身上值函数总有帮助、$\lambda=0.96$ 与 $\lambda=1$ 相当（§6.3.3）
- **设置**：策略与值函数同构（100/50/25 tanh 三隐层、$>10^4$ 参数），原始运动学（humanoid 33 维状态 / 10 执行器）直接输出关节力矩，完全 model-free；MuJoCo 仿真（§1、§6.2）

![[cartpole-grid.png]]
> Figure 2（[[gae-paper]] §6.3.1）：左——$\gamma=0.99$ 固定、扫 $\lambda$ 的学习曲线，中间值（0.92–0.98）改进最快；右——20 迭代后性能随 $\gamma,\lambda$ 的网格，白 = 高奖励，最优落在双参数的中间带。偏差–方差权衡不是修辞，是可测的地形。

## 9. 关键性质与陷阱

- **「λ=1 免疫值函数误差」的机理是 score 零均值**：$\mathbb{E}_{a\sim\pi}[\nabla\log\pi] = 0$ 使得任何只依赖过去的 baseline 在期望里消失（§2.3）——这是全部 actor-critic baseline 理论的支点；代价是方差。$\lambda<1$ 的偏差**只**来自 $\delta$ 里的 $V(s_{t+1})$（它不是 baseline、与 $a_t$ 相关，§5 的乘性八折即其数值形象）
- **为什么用 $V$ 不用 $Q$（附录 A.2）**：① 状态价值输入不含动作、维度低更好学；② $\lambda$ 插值只在 $V$ 路线上可用——参数化 $Q$ 的优势估计（$\hat A = Q(s,a)-V(s)$ 类）是高偏差一族，本文实测一步估计（$\lambda=0$）偏差大到位列最差。**去 critic 路线（[[grpo]]）把学到的 $V$ 换成组内均值，是这个论证空间的第三个角**
- **compatible features 与本文正交（附录 A.1）**：Konda & Tsitsiklis 的理论只说策略梯度取决于优势在 $\nabla\log\pi$ 张成子空间上的投影（自然梯度的最小二乘实现），不指导如何利用时间结构——GAE 任何 $\gamma$-just 估计器都能喂进那个最小二乘
- **截断版**：PPO 按固定长度 $T$ 的段采样，GAE 的求和截至段尾（[[ppo]] §3.2，式 11 的排印笔误考订见该页）——有限时域是 PPO 侧的实现形态，机制不变
- **bandit 退化**：单步终局的环境（如 RLHF 的「一句 prompt 一条回答即结算」）里没有 $t+1$ 之后的项，GAE 对任何 $\lambda$ 都退化为 $\delta_0 = r - V(s_0)$——时序机制不激活，PPO 系 RLHF 实现里的优势即「奖励 − 值函数」，[[grpo]] 的组均值是同一位置的免学习替代
- **值函数置信域训练本身是贡献**：对数千参数的网络值函数，「每批数据上带 KL 型约束的回归」比朴素回归稳（§5）——后来 PPO 简化为值函数平方损失（[[ppo]] §3.1），置信域版留在这篇

## 10. 在本 wiki 的语境

- **[[ppo]] §3.2**：PPO 的优势估计即截断 GAE（式 11 修正版）；本页补齐它的推导与 $\gamma,\lambda$ 语义——PPO 页只当黑盒引用 [Sch+15a]
- **[[trpo]] §5、§7**：TRPO 原文的优势 / Q 估计是 Monte-Carlo 折扣回报（single path）与短 rollout 重要性采样（vine）——高方差方案；GAE 是同组的下一站，与 TRPO（式 31）组合成完整算法。TRPO → GAE → PPO 的谱系：约束怎么走、信号怎么估
- **[[rlhf]] §2**：第 3 阶段 PPO 引擎的优势信号源；bandit 设定下退化为「奖励 − 值函数」（本页 §9）
- **[[grpo]]**：去掉学到的 critic、用组内相对回报当优势——本页 §9「为什么用 V 不用 Q」论证的另一面：连 V 也不学，代价是回到 MC 式方差、换来免值函数的工程简洁
- **[[dpo]]**：绕开整个 RL 管线（无优势估计问题），对照面

## 来源

- [[gae-paper]]（§1 动机与贡献、§2 式 1–8 与 Def 1 / Prop 1、§3 式 9–19 推导与 γ/λ 分工、§4 式 20–26 塑形与响应函数、§5 式 28–30 值函数置信域、§6.1 算法与式 31、§6.2–6.3 实验设置与结果、§7 开放问题、附录 A–B）
- [[ppo-paper]]（§5 截断 GAE 的使用形态与超参 $\lambda=0.95$、式 11–12）
- [[trpo-paper]]（§5 single path / vine 的 MC 回报估计——GAE 所替换的对象；附录 C 共轭梯度机制——式 30 复用）

## 相关

- [[ppo]]（截断 GAE 的使用方）｜ [[trpo]]（策略更新引擎）｜ [[rlhf]]（应用范式）｜ [[grpo]]（去 critic 下游）｜ [[dpo]]（免 RL 下游）
