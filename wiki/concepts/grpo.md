---
type: concept
title: GRPO（组相对策略优化）
created: 2026-09-15
updated: 2026-09-16
tags:
  - reinforcement-learning
  - alignment
  - training
sources: 2
---

# GRPO（Group Relative Policy Optimization，组相对策略优化）

**定义**：DeepSeekMath（arXiv:2402.03300，[[vision-r1-paper]] 引文 [38]）提出的无 critic 强化学习算法——同一提示采样一组完成，用组内奖励的均值 / 标准差当基线算相对优势；因 DeepSeek-R1（引文 [17]）用它做规则奖励训练而成为「R1 式」后训练的标准底座。

## 1. 要解决的问题

[[ppo|PPO]] 需要一个与策略同规模的价值网络（critic）来估计基线，对 LLM / LVLM 而言显存与训练开销都很重，价值模型本身也难训好。GRPO 的观察：对答案客观可验证的任务（数学 / 代码 / 定位），监督只落在最终结果上，同一问题的 N 个采样**互相比较**即可提供基线——组就是 critic，且组内相对比较天然不需要人工偏好标注（[[vision-r1-paper]] §3.1）。

## 2. 机制

对样本 $q$，旧策略 $\pi_{\theta_{old}}$ 采 $N$ 个完成 $\{o_1,\dots,o_N\}$，奖励函数 $f_{reward}$ 打分 $\{r_1,\dots,r_N\}$。

**组内相对优势**（[[vision-r1-paper]] Eq.1）：

$$A_i = \frac{r_i - \mathrm{mean}(\{r_j\}_{j=1}^{N})}{\mathrm{std}(\{r_j\}_{j=1}^{N})} \tag{Eq.1}$$

**目标函数**（[[vision-r1-paper]] Eq.2；比率项乘优势 + 对冻结参考模型 $\pi_{ref}$ 的 KL 惩罚，$\beta$ 为系数）：

$$\mathcal{J}_{GRPO}(\theta) = \frac{1}{N}\sum_{i=1}^{N}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i - \beta\,\mathbb{KL}\big(\pi_\theta(o_i|q)\,\big\|\,\pi_{ref}(o_i|q)\big)\right) \tag{Eq.2}$$

> **[wiki 外一般性知识]** 原始 GRPO 在比率项上还有 [[ppo]] 式 clip 截断（机制见 [[ppo]] §2），上式是 Vision-R1 引用的简化写法。候选来源：DeepSeekMath 论文（arXiv:2402.03300）。

对冻结参考模型 $\pi_{ref}$ 的 KL 惩罚与经典 [[rlhf]] 的约束项同源（[[dpo-paper]] 式 3）——同一条「KL 约束奖励最大化」目标，PPO / GRPO 在线解它，[[dpo]] 用重参数化离线解它。

### 手工走查：一组 4 个完成的优势计算

设某定位样本组内 4 个完成的总奖励为 $\{1.0,\ 0.86,\ 0.6,\ 0.0\}$（量级参照 [[vision-r1-paper]] Figure 2 示意的单路奖励值），std 取组内总体标准差（除 $N$）：

- $\mathrm{mean} = 0.615$，$\mathrm{std} = 0.383$
- $A_1 = (1.0-0.615)/0.383 \approx \mathbf{+1.01}$（组内最优，强推）
- $A_2 = (0.86-0.615)/0.383 \approx \mathbf{+0.64}$（次优，中推）
- $A_3 = (0.6-0.615)/0.383 \approx \mathbf{-0.04}$（≈均值，几乎无信号）
- $A_4 = (0.0-0.615)/0.383 \approx \mathbf{-1.61}$（最差，强压）

两个直接推论：**① 信号强弱由组内分化决定**——整组奖励都挤在 0.6 附近时优势趋零，这正是定位任务上「高 IoU 难拿满 → 奖励趋同」的困境，[[rule-based-reward|Vision-R1 的渐进收紧]]就是解它（[[vision-r1-paper]] §3.3）；**② 组内必须有成败分化**，全对 / 全错的组贡献不了梯度。

## 3. 关键性质与代价

- **无 critic**：不需要价值网络，省掉一份与策略同规模的显存与训练；相对优势直接从组统计量得到（[[vision-r1-paper]] §3.1）
- **无需偏好数据**：基线 = 组均值，比较在采样之间发生——这是「规则奖励 + GRPO」能同时免 RM、免人工偏好对的结构原因
- **代价 = 在线组采样**：每个输入要生成 $N$ 个完成再逐个打分，推理侧开销大——[[mixed-preference-optimization|MPO]] 阵营弃它选离线偏好对的理由（效率论证见该页「与 GRPO 的取舍」）
- **只适结果可验证的任务**：奖励必须能程序化算出（精确匹配 / IoU / 单测通过）；主观开放任务仍需 [[reward-model|学习式 RM]] 或偏好数据

## 4. 本 wiki 中的实例

- [[vision-r1]]：把 GRPO 搬到多模态定位，奖励 = 格式 + 召回 + 精度三路规则（[[rule-based-reward]]）；β=0.2、lr 1e-6、49K 样本 1 epoch 即大幅提升（[[vision-r1-paper]] §4.1）
- [[mixed-preference-optimization]]：同任务域的对照组——离线偏好对齐路线，为何弃 GRPO（在线采样开销）+ MPO 单项贡献最小的实测
- 评价信号从哪来的路线全景：[[rule-based-reward]]（规则）vs [[reward-model]]（学习的 RM）vs 人工偏好数据

## 来源

- [[vision-r1-paper]]（§3.1 preliminaries、Eq.1–2、§4.1 训练配置；DeepSeekMath / DeepSeek-R1 经其引文 [38][17] 锚定）
- [[dpo-paper]]（式 3：KL 约束目标与参考模型的同源性）

## 相关

- [[ppo]]（clip 与 KL 骨架的来源、critic 成本的对照面）｜ [[rlhf]]（KL 约束目标的共同源头）｜ [[rule-based-reward]] ｜ [[reward-model]] ｜ [[mixed-preference-optimization]] ｜ [[vision-r1]] ｜ [[dpo]]
