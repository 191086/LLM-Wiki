---
type: concept
title: DPO（Direct Preference Optimization，直接偏好优化）
created: 2026-09-15
updated: 2026-09-15
tags:
  - alignment
  - preference-optimization
  - rlhf
sources: 3
---

# DPO（Direct Preference Optimization，直接偏好优化）

**定义**：把 [[rlhf]] 的「先训奖励模型、再强化学习」两阶段折叠成单个对策略模型直接可微的偏好损失的算法（Rafailov et al., NeurIPS 2023）——核心是证明奖励可以从策略与参考模型的对数概率比中恢复（重参数化），让 RM「隐式地」活在策略内部，即论文标题「Your Language Model is Secretly a Reward Model」（[[dpo-paper]] §5.1）。

## 1. 要解决什么问题：RLHF 的两阶段之痛

经典 RLHF 三阶段（[[dpo-paper]] §3，机制详见 [[rlhf]] §2）：SFT → 训练 RM（Bradley-Terry 排序损失，见 [[reward-model]] §4）→ 用 RL（如 PPO）最大化 RM 打分、同时以 KL 惩罚拴住策略不跑远。痛点集中在第三阶段（[[dpo-paper]] §1）：在线采样成本高（训练回路里不断从 LM 采完成）、训练不稳（PPO 需要学值函数当 baseline，方差高，见 §5）、流程复杂（RM 与策略两套模型、两套优化器、RL 超参难调）。DPO 的问题是：能否不训显式 RM、不上 RL，直接用偏好对做监督学习？答案成立的前提是一个重参数化观察（§2）。

## 2. 重参数化：奖励藏在策略里

RLHF 的 RL 目标（最大化奖励 + KL 正则，[[dpo-paper]] 式 3）有闭式最优解（[[dpo-paper]] 式 4，附录 A.1 给完整推导）：

$$\pi^*(y \mid x) = \frac{1}{Z(x)}\, \pi_\text{ref}(y \mid x)\, \exp\!\Big(\frac{1}{\beta}\, r(x, y)\Big)$$

其中 $Z(x) = \sum_y \pi_\text{ref}(y \mid x) \exp\!\big(r(x, y)/\beta\big)$ 是**配分函数**（对全部候选回答求和的归一化常数，依赖 $x$ 但与单个 $y$ 无关），$\pi_\text{ref}$ 是冻结的参考模型（通常 = SFT 模型）。直接用式 4 不现实——估计 $Z(x)$ 本身很难；但这行可以**反解**出 $r$（两边取对数、移项，[[dpo-paper]] 式 5）：

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_\text{ref}(y \mid x)} + \beta \log Z(x)$$

奖励被表示成了「策略与参考模型的对数概率比」。最后一步：把这个 $r$ 代回 Bradley-Terry 偏好概率 $P(y^+ \succ y^- \mid x) = \sigma(r^+ - r^-)$——$\beta \log Z(x)$ 只依赖 $x$，在做差时**相消**（[[dpo-paper]] 式 6）。偏好概率于是只依赖概率比，显式 RM 彻底消失。

这个替换不损失一般性：论文 §5.1 定义「奖励等价类」——两个奖励函数若只差一个 $f(x)$（只依赖 prompt 的函数）则等价，并证明（Lemma 1、2）等价类内所有成员诱导**相同的偏好分布与相同的最优策略**；Theorem 1 进一步证明每个等价类都有（且 Proposition 1：唯一有）一个形如 $r = \beta \log \frac{\pi}{\pi_\text{ref}}$ 的代表成员。所以「只能表示这类奖励」不是约束——BT/Plackett-Luce 可表达的奖励，重参数化全能覆盖。

## 3. DPO 损失

把 §2 的 $r$ 代入 BT 损失（[[reward-model]] §4），对策略 $\pi_\theta$ 直接最小化（[[dpo-paper]] 式 7）：

$$\mathcal{L}_\text{DPO} = -\,\mathbb{E}_{(x,\, y^+,\, y^-)}\ \log \sigma\!\Big(\beta \log \frac{\pi_\theta(y^+ \mid x)}{\pi_\text{ref}(y^+ \mid x)} - \beta \log \frac{\pi_\theta(y^- \mid x)}{\pi_\text{ref}(y^- \mid x)}\Big)$$

形式上就是 BT 损失把打分器换成 $s = \beta \log \frac{\pi_\theta}{\pi_\text{ref}}$——纯监督学习：一遍前向、一遍反向，没有在线采样。实现十余行 PyTorch（论文附录 B 给出参考实现：两个序列的 log-prob 差过 `logsigmoid`）。数值小例（取 $\beta = 1$，记 $h = \log \frac{\pi_\theta(y\mid x)}{\pi_\text{ref}(y\mid x)}$）：

| log 概率比 | 含义 | $\mathcal{L}_\text{DPO}$ |
| --- | --- | --- |
| $h^+ = 1.0$，$h^- = -0.5$ | chosen 概率升、rejected 降 | $-\log \sigma(1.5) = 0.20$ |
| $h^+ = 5.0$，$h^- = 3.5$ | **两条都大涨**，差值不变 | $-\log \sigma(1.5) = 0.20$（完全无感） |

第二行是 DPO 最常被诟病的性质：只看差值（平移不变，承 [[reward-model]] §4 的 BT 三后果）——被拒回答的绝对概率暴涨也不管；[[mixed-preference-optimization]] 混入 BCO 质量项补的正是这块。

## 4. 梯度：出错样本加权更大

对机制更重要的视角是梯度（[[dpo-paper]] §4「What does the DPO update do?」）。记隐式奖励 $\hat{r}_\theta(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$，则：

$$\nabla_\theta \mathcal{L}_\text{DPO} = -\beta\;\mathbb{E}\Big[\ \underbrace{\sigma\big(\hat{r}_\theta(x, y^-) - \hat{r}_\theta(x, y^+)\big)}_{\text{隐式奖励排错越狠，权重越大}}\ \Big[\ \underbrace{\nabla_\theta \log \pi_\theta(y^+ \mid x)}_{\text{升 chosen}} - \underbrace{\nabla_\theta \log \pi_\theta(y^- \mid x)}_{\text{降 rejected}}\ \Big]\Big]$$

三件事叠加：①梯度方向恒为「升 $y^+$、降 $y^-$」；②每例的步长被 $\sigma(\hat{r}^- - \hat{r}^+)$ 动态加权——隐式奖励把顺序**排对且拉开**后权重趋零（不再加力），**排错**时权重过 0.5（重点纠正）；③整体幅度由 $\beta$ 缩放。论文强调这个加权不可去：去掉系数的朴素概率比目标（unlikelihood 式）会让语言模型退化成重复词垃圾输出（[[dpo-paper]] §4 + 附录 Table 3）。

**手工走查**（$\beta = 0.1$，论文默认值）：比较「排错」与「排对且拉开」两种情形的权重。

| 情形 | $h^+$ | $h^-$ | $\hat{r}^+ = \beta h^+$ | $\hat{r}^- = \beta h^-$ | 权重 $\sigma(\hat{r}^- - \hat{r}^+)$ |
| --- | --- | --- | --- | --- | --- |
| 刚排错 | $-0.2$ | $+0.3$ | $-0.02$ | $+0.03$ | $\sigma(0.05) = 0.513$（重点纠正） |
| 排对且拉开 | $+2.0$ | $-2.0$ | $+0.2$ | $-0.2$ | $\sigma(-0.4) = 0.401$（已收敛，减力） |

训练早期策略 ≈ 参考模型（$h^\pm \approx 0$），权重恒 0.5——梯度均匀铺开；随训练推进，已排对的对自动降权，梯度集中到仍排错的对上。这与 [[reward-model]] §4.2 RM 侧「梯度随 margin 饱和」是同一个行为在策略侧的镜像。

## 5. 关键性质

- **参考模型 $\pi_\text{ref}$ 的角色**：锚。损失优化的是「相对参考的概率比」，天然约束策略不偏离预训练分布太远——RLHF 里 KL 正则的活儿由结构自带。$\pi_\text{ref}$ 通常取 SFT 后的策略并全程冻结；SFT 模型不可得时，论文的做法是在偏好数据的 chosen 回答上做 MLE 近似，缓解分布漂移（[[dpo-paper]] §4 DPO outline）。
- **$\beta$ 的作用**：隐式奖励 $r = \beta \log \frac{\pi_\theta}{\pi_\text{ref}}$ 中，$\beta$ 越小、同样的隐式奖励变化要求的概率比变化越大（$e^{r/\beta}$），即策略被允许偏离得更远；实践中常取 0.1 量级（论文默认 $\beta=0.1$，TL;DR 摘要任务用 0.5，[[dpo-paper]] 附录 B），是最重要的超参之一。
- **隐式 RM**：$r(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$ 可随时从训好的策略里读出——[[reward-model]] §1 分类法里的「隐式 RM」即此。
- **失败模式**：只压差值可能连带压低两条回答的绝对概率（包括正确那条）——[[mixed-preference-optimization]] 的 SFT 生成项就是对这个失败的缓解。注意这是后续文献的实证观察而非本文内容；本文 §7 自陈把「reward 过度优化在 DPO 中如何显形」列为开放问题（候选来源：Pal et al. 2024《Fixing Failure Modes of Preference Optimisation with DPO-Positive》，arXiv:2402.13228）。

## 6. 实验证据（[[dpo-paper]] §6）

三个开放文本生成任务、模型 ≤6B：IMDb 情感控制（GPT-2-large，有情感分类器当真奖励）、Reddit TL;DR 摘要（GPT-J 6B）、Anthropic-HH 单轮对话（Pythia-2.8B）。

- **reward-KL 前沿严格占优**（§6.1）：22 组超参扫描中，DPO 在同等 KL 偏离下拿到最高奖励，前沿严格支配 PPO——甚至优于能用真奖励的 PPO-GT（Figure 2 左，见 [[dpo-paper]]）
- **摘要**（§6.2）：对人类参考摘要的 GPT-4 胜率 DPO ≈61%（temp 0）> PPO 57%（其最优温度），且对采样温度远比 PPO 稳健；人类研究中 DPO 对 PPO 胜率 58%
- **对话**（§6.2）：Anthropic-HH 上 DPO 是唯一「计算高效且胜过数据集 chosen 回答」的方法，与昂贵的 Best of 128 基线相当
- **OOD 泛化**（Table 1）：TL;DR 训出的策略在 CNN/DailyMail 上 DPO 胜率 0.36 vs PPO 0.26
- **对比基线的教训**：Preferred-FT（只在 chosen 上做 SFT）几乎不涨——单边最大似然不足以学偏好；unlikelihood 退化；Best of N 在 N≈64–128 饱和

## 7. 在本 wiki 的语境

![[dpo-fig1-rlhf-vs-dpo-pipeline.png]]
> Figure 1（[[dpo-paper]] §1）：RLHF（左）= 偏好数据极大似然训 RM + RM 与策略采样的强化学习闭环；DPO（右）= 偏好数据直接极大似然到最终 LM。

- [[rlhf]] §2–3：DPO 所折叠的两阶段管线的完整描述与谱系定位
- [[ostrakon-vl-paper]] §4.3：MPO 的偏好损失「derived from DPO (Rafailov et al., 2023)」
- [[reward-model]] §1：「隐式 RM——DPO 式重参数化」分类项
- [[mixed-preference-optimization]] §2：$\mathcal{L}_\text{DPO}$ 作为 MPO 三项之一
- [[grpo]]：在线 RL 路线对照——共享同一个 KL 约束目标（式 3），一个离线免采样、一个在线组采样

## 来源

- [[dpo-paper]]（§3 三阶段预备与式 1–3、§4 式 4–7 与梯度、§5 理论、§6 实验、附录 B 实现）
- [[ostrakon-vl-paper]]（MPO 偏好损失溯源 DPO 的原文语境）
- [[skywork-vl-reward-paper]]（§2 隐式 RM 分类法）

## 相关

- [[rlhf]]（被折叠的两阶段管线本体）
- [[reward-model]]（BT 损失与 RM 分类学——DPO 推导的起点）
- [[mixed-preference-optimization]]（DPO 作为组成项的实战用法）
- [[grpo]]（在线 RL 路线的对照）
