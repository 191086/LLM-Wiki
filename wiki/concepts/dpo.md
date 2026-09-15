---
type: concept
title: DPO（Direct Preference Optimization，直接偏好优化）
created: 2026-09-15
updated: 2026-09-15
tags:
  - alignment
  - preference-optimization
  - rlhf
sources: 2
---

# DPO（Direct Preference Optimization，直接偏好优化）

**定义**：把 RLHF 的「先训奖励模型、再强化学习」两阶段折叠成单个对策略模型直接可微的偏好损失的算法——核心是证明奖励可以从策略与参考模型的对数概率比中恢复（重参数化），让 RM「隐式地」活在策略内部。

> 本页 §1–§4 的机制推导为 wiki 外一般性知识；候选来源：Rafailov et al. 2023《Direct Preference Optimization: Your Language Model is Secretly a Reward Model》（arXiv:2305.18290）。库内锚定见 §5 与「来源」节。

## 1. 要解决什么问题：RLHF 的两阶段之痛

经典 RLHF 三阶段：SFT → 训练 RM（Bradley-Terry 排序损失，见 [[reward-model]] §4）→ 用 RL（如 PPO）最大化 RM 打分、同时以 KL 惩罚拴住策略不跑远。痛点集中在第三阶段：在线采样成本高、训练不稳（奖励黑客、策略崩塌）、流程复杂（RM 与策略两套模型、两套优化器）。DPO 的问题是：能否不训显式 RM、不上 RL，直接用偏好对做监督学习？

## 2. 重参数化：奖励藏在策略里

RLHF 的 RL 目标（最大化奖励 + KL 正则）有闭式最优解：

$$\pi^*(y \mid x) = \frac{1}{Z(x)}\, \pi_\text{ref}(y \mid x)\, \exp\!\Big(\frac{1}{\beta}\, r(x, y)\Big)$$

其中 $Z(x) = \sum_y \pi_\text{ref}(y \mid x) \exp\!\big(r(x, y)/\beta\big)$ 是配分函数，与 $y$ 无关。把这行反解出 $r$（两边取对数、移项）：

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_\text{ref}(y \mid x)} + \beta \log Z(x)$$

奖励被表示成了「策略与参考模型的对数概率比」。最后一步：把这个 $r$ 代回 Bradley-Terry 偏好概率 $P(y^+ \succ y^- \mid x) = \sigma(r^+ - r^-)$——配分函数 $\beta \log Z(x)$ 只依赖 $x$，在做差时**相消**。偏好概率于是只依赖概率比，显式 RM 彻底消失。

## 3. DPO 损失

把 §2 的 $r$ 代入 BT 损失（[[reward-model]] §4），对策略 $\pi_\theta$ 直接最小化：

$$\mathcal{L}_\text{DPO} = -\,\mathbb{E}_{(x,\, y^+,\, y^-)}\ \log \sigma\!\Big(\beta \log \frac{\pi_\theta(y^+ \mid x)}{\pi_\text{ref}(y^+ \mid x)} - \beta \log \frac{\pi_\theta(y^- \mid x)}{\pi_\text{ref}(y^- \mid x)}\Big)$$

形式上就是 BT 损失把打分器换成 $s = \beta \log \frac{\pi_\theta}{\pi_\text{ref}}$——纯监督学习：一遍前向、一遍反向，没有在线采样。数值小例（取 $\beta = 1$，记 $h = \log \frac{\pi_\theta(y\mid x)}{\pi_\text{ref}(y\mid x)}$）：

| log 概率比 | 含义 | $\mathcal{L}_\text{DPO}$ |
| --- | --- | --- |
| $h^+ = 1.0$，$h^- = -0.5$ | chosen 概率升、rejected 降 | $-\log \sigma(1.5) = 0.20$ |
| $h^+ = 5.0$，$h^- = 3.5$ | **两条都大涨**，差值不变 | $-\log \sigma(1.5) = 0.20$（完全无感） |

第二行是 DPO 最常被诟病的性质：只看差值（平移不变，承 [[reward-model]] §4 的 BT 三后果）——被拒回答的绝对概率暴涨也不管；[[mixed-preference-optimization]] 混入 BCO 质量项补的正是这块。

## 4. 关键性质

- **参考模型 $\pi_\text{ref}$ 的角色**：锚。损失优化的是「相对参考的概率比」，天然约束策略不偏离预训练分布太远——RLHF 里 KL 正则的活儿由结构自带。$\pi_\text{ref}$ 通常取 SFT 后的策略并全程冻结。
- **$\beta$ 的作用**：隐式奖励 $r = \beta \log \frac{\pi_\theta}{\pi_\text{ref}}$ 中，$\beta$ 越小、同样的隐式奖励变化要求的概率比变化越大（$e^{r/\beta}$），即策略被允许偏离得更远；实践中常取 0.1 量级，是最重要的超参之一。
- **隐式 RM**：$r(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$ 可随时从训好的策略里读出——[[reward-model]] §1 分类法里的「隐式 RM」即此，DPO 论文标题「Your Language Model is Secretly a Reward Model」说的也是这件事。
- **失败模式**：只压差值可能连带压低两条回答的绝对概率（包括正确那条）——[[mixed-preference-optimization]] 的 SFT 生成项就是对这个失败的缓解。

## 5. 在本 wiki 的语境

- [[ostrakon-vl-paper]] §4.3：MPO 的偏好损失「derived from DPO (Rafailov et al., 2023)」
- [[reward-model]] §1：「隐式 RM——DPO 式重参数化」分类项
- [[mixed-preference-optimization]] §2：$\mathcal{L}_\text{DPO}$ 作为 MPO 三项之一

## 来源

- [[ostrakon-vl-paper]]（MPO 偏好损失溯源 DPO 的原文语境）
- [[skywork-vl-reward-paper]]（§2 隐式 RM 分类法）
- （机制推导为 wiki 外知识，候选来源见页首）

## 相关

- [[reward-model]]（BT 损失与 RM 分类学——DPO 推导的起点）
- [[mixed-preference-optimization]]（DPO 作为组成项的实战用法）
- [[grpo]]（在线 RL 路线的对照）
