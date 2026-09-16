---
type: concept
title: RLHF（基于人类反馈的强化学习）
created: 2026-09-15
updated: 2026-09-15
tags:
  - alignment
  - rlhf
  - reinforcement-learning
sources: 2
---

# RLHF（Reinforcement Learning from Human Feedback，基于人类反馈的强化学习）

**定义**：用人类偏好反馈训练语言模型使其行为对齐人类意图的范式——先以监督微调（SFT，Supervised Fine-Tuning，在人类示范上做标准的下一词元交叉熵）热身，再从偏好对训练奖励模型，最后用强化学习最大化奖励同时以 KL 惩罚约束策略不偏离；自 InstructGPT / ChatGPT 起成为对话 LM 对齐的主流框架（[[dpo-paper]] §3 沿用 Ziegler et al. 的管线综述；InstructGPT 为其引文 [28]，Ouyang et al. 2022，arXiv:2203.02155）。

## 1. 要解决什么问题

预训练 LM 从人类写的海量文本里学到能力，但「选哪些行为」无法靠模仿解决——数据里混着好坏各种意图，模型该输出「高质量回答」而非「平均水平的网络文本」（[[dpo-paper]] §1）。人类**判断相对好坏**远比**写出示范**或**打绝对分**容易，所以反馈以偏好对（$y^+ \succ y^-$）的形式收集，再由算法把偏好变成可优化的信号——这是整个范式的信息论起点（[[reward-model]] §4.3 的推论）。

## 2. 三阶段管线（[[dpo-paper]] §3）

1. **SFT**：在下游任务的高质量数据上监督微调预训练 LM，得 $\pi^{SFT}$——后续一切以它为起点。
2. **奖励建模**：用 $\pi^{SFT}$ 对每个 prompt 采两个回答，人类标注偏好 $y_w \succ y_l$；假设偏好由潜奖励 $r^*(x,y)$ 经 **Bradley-Terry 模型**（偏好概率 = 分差过 sigmoid 的概率模型，[[reward-model]] §4.1）生成，以负对数似然拟合 $r_\phi$（[[dpo-paper]] 式 1–2）。工程细节：$r_\phi$ 常从 $\pi^{SFT}$ 加线性标量头初始化，并做奖励归一化 $\mathbb{E}[r_\phi(x,y)]=0$ 以降方差（[[dpo-paper]] §3）。
3. **RL 微调**：解 KL 约束的奖励最大化（[[dpo-paper]] 式 3）：

$$\max_{\pi_\theta}\ \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(y|x)} \big[r_\phi(x, y)\big] - \beta\, \mathbb{D}_\mathrm{KL}\big(\pi_\theta(y|x)\ \|\ \pi_\text{ref}(y|x)\big)$$

$\beta$ 控制偏离参考策略 $\pi_\text{ref}$（= $\pi^{SFT}$）的代价：没有这一项，策略会涌向 RM 打分虚高但实际糟糕的回答（reward hacking，[[rule-based-reward]] §4），并塌缩成单一高分回答、丧失多样性（[[dpo-paper]] §3）。实践中把约束并入奖励构造 $r(x,y) = r_\phi(x,y) - \beta\big(\log \pi_\theta(y|x) - \log \pi_\text{ref}(y|x)\big)$，用 PPO 最大化。

### 手工走查：形化奖励的逐样本符号 ≠ 约束的逐样本含义

把 KL 约束并入奖励后，**单个回答**的形化奖励可能比原始分还高——但期望上约束恰好成立。设候选 $\{y_A, y_B\}$，$\pi_\text{ref} = (0.8, 0.2)$，策略 $\pi_\theta = (0.5, 0.5)$，RM 打分 $r_\phi = (1.0, 0.9)$，$\beta = 0.1$：

- 形化奖励：$r(y_A) = 1.0 - 0.1\ln\frac{0.5}{0.8} = 1.0 + 0.047 = 1.047$；$r(y_B) = 0.9 - 0.1\ln\frac{0.5}{0.2} = 0.9 - 0.092 = 0.808$
- 注意 $y_A$：策略把它的概率**调低**了（0.8→0.5），形化奖励反而**加**了 0.047——逐样本看惩罚方向是反的
- 但取期望：$\mathbb{E}_{y\sim\pi_\theta}[r] = 0.5 \times 1.047 + 0.5 \times 0.808 = 0.928$，而 $\mathbb{E}[r_\phi] - \beta\,\mathrm{KL}(\pi_\theta\,\|\,\pi_\text{ref}) = 0.95 - 0.1 \times 0.223 = 0.928$——**严格相等**

即形化奖励在 $\pi_\theta$ 采样下恰好把式 3 的目标变成无约束最大化；KL 惩罚只在期望层面成立，逐样本不可读。两个推论：①必须**在线**从 $\pi_\theta$ 采样才能评估这个目标——这正是 DPO（离线、不采样）要折叠掉的环节，也是 [[grpo]] 在线组采样成本的结构来源；②逐样本奖励信号噪声大、方差高，PPO 要再学一个值函数当 baseline 才能训稳（[[dpo-paper]] §5.2）。

## 3. 为什么难训（[[dpo-paper]] §1、§5.2）

- **要养多个 LM**：SFT、RM、策略、（PPO 的）值函数——两三套参数同时在训
- **训练回路里在线采样**：每步都要从策略生成回答、过 RM 打分，推理开销大
- **值函数/baseline 难学**：KL 约束的最优解里有个归一化项（$\pi_\text{ref}$ 的软值函数，配分函数 $\log Z(x)$），学它难、单样本估计方差高——PPO 不稳的结构原因（[[dpo-paper]] §5.2；DPO 的重参数化则让该项彻底消失，见 [[dpo]] §2）
- **RM 可被利用**：策略会找 RM 的漏洞而不是真好（reward hacking）——绝对分经 BT 训练本就不可辨识，只能差值使用（[[reward-model]] §4.3）

## 4. 本库中的路线谱系

| 路线 | 页面 | 对管线的改造 | 采样时机 |
| --- | --- | --- | --- |
| 显式 RM + PPO（经典 RLHF） | 本页 | 原样三阶段 | 在线（训练回路） |
| DPO | [[dpo]] | 重参数化折叠 2+3 阶段为一个损失 | 离线（无采样） |
| MPO | [[mixed-preference-optimization]] | DPO 偏好项 + BCO 质量项 + SFT 生成项 | 离线（偏好对预先备好） |
| 规则奖励 + GRPO | [[rule-based-reward]] + [[grpo]] | 免 RM：程序化规则直接打分；免 critic：组内相对优势 | 在线（每组采 N 个） |

两条离线路线（DPO / MPO）与两条在线路线（PPO / GRPO）共享同一个 KL 约束目标（式 3 / [[dpo]] §2 的闭式解）；区别在评价信号来源（学习式 RM vs 规则）与更新时机（离线偏好对 vs 在线采样）——全景对照见 [[rule-based-reward]] §4 与 [[overview]]「评价信号谱系」。

## 5. 在本 wiki 的语境

- [[dpo]]：本文 §2–3 管线的折叠（本库主锚来源即 DPO 论文）
- [[reward-model]]：第 2 阶段的产物，两轴分类法与 BT 损失
- [[mixed-preference-optimization]] / [[grpo]] / [[rule-based-reward]]：谱系各分支的机制与实战
- [[vision-r1-paper]] §2.2：偏好标注 + RM 路线的成本批判（为什么视觉定位任务可以整体绕开）

## 来源

- [[dpo-paper]]（§1 动机、§3 三阶段与式 1–3、§5.2 PPO 不稳性诊断、引文 [28] InstructGPT）
- [[vision-r1-paper]]（§2.2 偏好路线成本对照）

## 相关

- [[dpo]] ｜ [[reward-model]] ｜ [[grpo]] ｜ [[mixed-preference-optimization]] ｜ [[rule-based-reward]]
