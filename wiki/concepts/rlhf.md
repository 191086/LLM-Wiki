---
type: concept
title: RLHF（基于人类反馈的强化学习）
created: 2026-09-15
updated: 2026-09-16
tags:
  - alignment
  - rlhf
  - reinforcement-learning
sources: 3
---

# RLHF（Reinforcement Learning from Human Feedback，基于人类反馈的强化学习）

**定义**：用人类偏好反馈训练语言模型使其行为对齐用户意图的范式——先以监督微调（SFT，Supervised Fine-Tuning，在人类示范上做标准的下一词元交叉熵）热身，再从人类偏好排序训练奖励模型（RM，Reward Model），最后用强化学习（实践中是 PPO，Proximal Policy Optimization，近端策略优化）最大化奖励、同时以 KL 惩罚约束策略不偏离 SFT 起点；InstructGPT 以「1.3B 对齐模型胜过 175B 未对齐 GPT-3」把它从窄任务实验变成对话 LM 对齐的主流框架（[[instructgpt-paper]] §3.1、§4.1）。

## 1. 要解决什么问题

预训练 LM 从人类写的海量文本里学到能力，但预训练目标（下一词预测）与「跟随用户指令、有帮助且无害」是两个目标——论文称之为行为**错位**（misaligned）：模型会编造事实、输出有毒内容、无视指令（[[instructgpt-paper]] §1）。「对齐」在实践中用 **3H** 操作化：helpful（帮用户解决任务）、honest（不编造不误导，可测的是真实性子集）、harmless（不造成伤害）（[[instructgpt-paper]] §3.6）。注意它对齐的是「标注员 + 研究者指南 + API 客户」这群人表达出的偏好，不是抽象的「人类价值观」（[[instructgpt-paper]] §5.2）。

这条路线的信息论起点：人类**判断相对好坏**远比**写出示范**或**打绝对分**容易，所以反馈以偏好排序的形式收集，再由算法把偏好变成可优化的信号（[[reward-model]] §4.3 的推论）。InstructGPT 把这点做成两级工程化：标注员不标「哪个好」而是一次给 K=4–9 个输出**排序**，一个排序展开出 $\binom{K}{2}$ 个偏好对（[[instructgpt-paper]] §3.5）。

## 2. 三阶段管线（InstructGPT 定标版；[[instructgpt-paper]] §3.1、Figure 2）

![[instructgpt-rlhf-pipeline.png]]
> 三步管线（[[instructgpt-paper]] Figure 2）：①示范数据训 SFT；②排序数据训 RM；③PPO 对 RM 优化策略。②③可迭代——对当前最优策略继续采排序数据。

1. **SFT**：标注员对 prompt 撰写示范（~13k 训练 prompt），监督微调 GPT-3 得 $\pi^{SFT}$——后续一切以它为起点。工程细节：训 16 epochs，验证损失 1 epoch 后就过拟合，但 RM 分与人类偏好评分仍随 epochs 提升，所以**按 RM 分选模型**、不按验证损失（[[instructgpt-paper]] §3.5、C.1）。
2. **奖励建模**：同一 prompt 采 K 个输出，标注员从优到劣排序，展开成 $\binom{K}{2}$ 个偏好对 $y_w \succ y_l$；假设偏好由潜奖励 $r^*(x,y)$ 经 **Bradley-Terry 模型**（偏好概率 = 分差过 sigmoid 的概率模型，[[reward-model]] §4.1）生成，以负对数似然拟合 $r_\theta$——即 [[instructgpt-paper]] 式 1：

$$\mathrm{loss}(\theta) = -\frac{1}{\binom{K}{2}}\,\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\Big[\log\sigma\big(r_\theta(x,y_w) - r_\theta(x,y_l)\big)\Big]$$

三个工程细节：只用 **6B RM**（175B RM 训练不稳定，不宜作 PPO 值函数初始化，且算力贵，附录 C.2）；同一 prompt 的全部 $\binom{K}{2}$ 对打包为**单个 batch 元素**（各对高度相关，拆成独立样本则单 epoch 即过拟合，还把每回答的前向次数从 $\binom{K}{2}$ 降到 1）；训练前用 bias 把示范数据均分归 0——损失平移不变、绝对分不可辨识，必须人为定锚（[[instructgpt-paper]] §3.5；平移不变性见 [[reward-model]] §4.3）。

3. **RL 微调**：bandit 环境（一句 prompt 采一条回答即终局结算，无多步状态），PPO 最大化 RM 打分；同时每个 token 加对 $\pi^{SFT}$ 的 KL 惩罚（β=0.02），缓解对 RM 的过度优化（reward hacking，[[rule-based-reward]] §4）；值函数从 RM 初始化（[[instructgpt-paper]] §3.5、C.4）。**PPO-ptx** 变体在目标里再混入预训练数据的对数似然梯度（系数 γ=27.8、预训练样本量 8× 于 RL episodes，附录 C.4），用于缴「对齐税」（见 §3.3；式 2 原文见 [[instructgpt-paper]] §3.5）。

阶段 3 解的形式化目标即 KL 约束的奖励最大化（[[dpo-paper]] 式 3）：

$$\max_{\pi_\theta}\ \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(y|x)} \big[r_\phi(x, y)\big] - \beta\, \mathbb{D}_\mathrm{KL}\big(\pi_\theta(y|x)\ \|\ \pi_\text{ref}(y|x)\big)$$

实践中把约束并入奖励构造 $r(x,y) = r_\phi(x,y) - \beta\big(\log \pi_\theta(y|x) - \log \pi_\text{ref}(y|x)\big)$（InstructGPT 式 2 是它的逐 token 工程版，$\pi_\text{ref} = \pi^{SFT}$）用 PPO 最大化。$\beta$ 控制偏离参考策略的代价：没有这一项，策略会涌向 RM 打分虚高但实际糟糕的回答，并塌缩成单一高分回答、丧失多样性（[[dpo-paper]] §3）。

### 手工走查：形化奖励的逐样本符号 ≠ 约束的逐样本含义

把 KL 约束并入奖励后，**单个回答**的形化奖励可能比原始分还高——但期望上约束恰好成立。设候选 $\{y_A, y_B\}$，$\pi_\text{ref} = (0.8, 0.2)$，策略 $\pi_\theta = (0.5, 0.5)$，RM 打分 $r_\phi = (1.0, 0.9)$，$\beta = 0.1$：

- 形化奖励：$r(y_A) = 1.0 - 0.1\ln\frac{0.5}{0.8} = 1.0 + 0.047 = 1.047$；$r(y_B) = 0.9 - 0.1\ln\frac{0.5}{0.2} = 0.9 - 0.092 = 0.808$
- 注意 $y_A$：策略把它的概率**调低**了（0.8→0.5），形化奖励反而**加**了 0.047——逐样本看惩罚方向是反的
- 但取期望：$\mathbb{E}_{y\sim\pi_\theta}[r] = 0.5 \times 1.047 + 0.5 \times 0.808 = 0.928$，而 $\mathbb{E}[r_\phi] - \beta\,\mathrm{KL}(\pi_\theta\,\|\,\pi_\text{ref}) = 0.95 - 0.1 \times 0.223 = 0.928$——**严格相等**

即形化奖励在 $\pi_\theta$ 采样下恰好把约束目标变成无约束最大化；KL 惩罚只在期望层面成立，逐样本不可读。两个推论：①必须**在线**从 $\pi_\theta$ 采样才能评估这个目标——这正是 DPO（离线、不采样）要折叠掉的环节，也是 [[grpo]] 在线组采样成本的结构来源；②逐样本奖励信号噪声大、方差高，PPO 要再学一个值函数当 baseline 才能训稳（InstructGPT 直接从 RM 初始化值函数，[[instructgpt-paper]] §3.5；不稳性的结构诊断见 [[dpo-paper]] §5.2）。

## 3. 实证：对齐有效且便宜（[[instructgpt-paper]] §4–5）

### 3.1 小模型赢大模型

方法阶梯 GPT < GPT(prompted) < SFT < PPO ≈ PPO-ptx 在 1.3B/6B/175B 三个规模一致：175B InstructGPT 对 175B GPT-3 胜率 **85±3%**、对 few-shot GPT-3 **71±4%**；**1.3B InstructGPT 已净胜 175B GPT-3**（约 100× 参数差）（§4.1，Figure 1）。held-out 标注员（不产训练数据）给出与训练标注员相同的偏好排序——结果不是过拟合训练标注员；RM 的 5 折跨标注组精度 69.6±0.9%（组内 72.4±0.4%），偏好信号本身可跨人泛化（§4.1，Figure 3）。

![[instructgpt-winrate-heldout.png]]
> 胜率四象限（[[instructgpt-paper]] Figure 3）：左列 GPT 分布 prompt、右列 InstructGPT 分布 prompt；上排 held-out 标注员、下排训练标注员。四象限阶梯形态一致——对齐收益不依赖特定 prompt 分布，也不依赖特定标注员群体。

### 3.2 真实性、毒性、偏见

- **真实性**：TruthfulQA 上真实且信息量大的回答约为 GPT-3 的 2×，且是默认行为（无需指令「说真话」）；闭域任务（摘要、闭域 QA）幻觉率 21% vs GPT-3 41%（§4.2）
- **毒性**：RealToxicityPrompts 上「尊重」指令下毒性输出 −25%，但无指令时优势消失；**被明确要求有毒时比 GPT-3 更毒**——「跟随指令」本身是双刃剑，也是 §5.3 自陈的最大局限（§4.2、§5.3）
- **偏见**：Winogender / CrowS-Pairs 无改善（§4.2）

### 3.3 对齐税：混预训练梯度，而非收紧约束

纯 PPO 在 SQuADv2、DROP、HellaSwag、WMT15 Fr→En 上回退——对齐以牺牲下游能力为代价即「对齐税」（alignment tax）。**PPO-ptx**（混预训练对数似然梯度）基本修复：HellaSwag 反超 GPT-3，DROP/SQuADv2/翻译仍落后但大幅收窄。对照组实验：改为加大 KL 系数，验证奖励大跌且修不回 DROP/SQuAD——缴税的正确姿势是**混回预训练分布**，不是收紧对 RL 的约束（§4.2，Figure 29/33/34）。

### 3.4 成本账与泛化边界

对齐算力：175B SFT 4.9 + 175B PPO-ptx 60 petaflops/s-days，对比 GPT-3 预训练 3,640——**约 2% 的开销换来超过 100× 参数增长的偏好收益**，「当下投资对齐比做大模型更划算」（§5.1）。分布外泛化：能跟随非英语指令、做代码问答与摘要（训练分布占比极小），但残余错误清晰：假前提照单全收、简单问题过度对冲（标注指南奖励「认知谦逊」被 RM 放大）、多显式约束性能下降（§4.3）。

## 4. 为什么难训

- **要养多个 LM**：SFT、RM、策略、（PPO 的）值函数——InstructGPT 里是三档策略 + 6B RM + 6B 值函数多套参数（[[instructgpt-paper]] §3.5）
- **训练回路里在线采样**：每步都要从策略生成回答、过 RM 打分（bandit 回路）
- **值函数/baseline 难学**：KL 约束最优解里有归一化项（$\pi_\text{ref}$ 的软值函数，配分函数 $\log Z(x)$），学它难、单样本估计方差高——PPO 不稳的结构原因（[[dpo-paper]] §5.2；DPO 的重参数化让该项彻底消失，见 [[dpo]] §2）；实例：InstructGPT 发现 175B RM 训练不稳定，值函数只能从 6B RM 初始化（附录 C.2）
- **RM 可被利用**：策略会找 RM 的漏洞而不是真好（reward hacking，[[rule-based-reward]] §4）——绝对分经 BT 训练本就不可辨识，只能差值使用（[[reward-model]] §4.3）；逐 token KL 惩罚（β=0.02）就是对此的第一道闸（[[instructgpt-paper]] §3.5）

## 5. 本库中的路线谱系

| 路线 | 页面 | 对管线的改造 | 采样时机 |
| --- | --- | --- | --- |
| 显式 RM + PPO（经典 RLHF） | 本页 | 原样三阶段（[[instructgpt-paper]]） | 在线（训练回路） |
| DPO | [[dpo]] | 重参数化折叠 2+3 阶段为一个损失 | 离线（无采样） |
| MPO | [[mixed-preference-optimization]] | DPO 偏好项 + BCO 质量项 + SFT 生成项 | 离线（偏好对预先备好） |
| 规则奖励 + GRPO | [[rule-based-reward]] + [[grpo]] | 免 RM：程序化规则直接打分；免 critic：组内相对优势 | 在线（每组采 N 个） |

两条离线路线（DPO / MPO）与两条在线路线（PPO / GRPO）共享同一个 KL 约束目标（式 3 / [[dpo]] §2 的闭式解）；区别在评价信号来源（学习式 RM vs 规则）与更新时机（离线偏好对 vs 在线采样）——全景对照见 [[rule-based-reward]] §4 与 [[overview]]「评价信号谱系」。

## 6. 在本 wiki 的语境

- [[instructgpt]]：范式定标的模型实例（三档规模、配方速览）；本页主锚来源即其论文
- [[dpo]]：本文 §2 管线的折叠（[[dpo-paper]]）；[[reward-model]]：第 2 阶段的产物，两轴分类法与 BT 损失
- [[mixed-preference-optimization]] / [[grpo]] / [[rule-based-reward]]：谱系各分支的机制与实战
- [[vision-r1-paper]] §2.2：偏好标注 + RM 路线的成本批判（为什么视觉定位任务可以整体绕开）

## 来源

- [[instructgpt-paper]]（§1 动机与 3H、§3 三步方法与式 1–2、§4 主结果与对齐税、§5 讨论；Figure 2 管线图）
- [[dpo-paper]]（式 3 的 KL 约束目标形式化、§5.2 PPO 不稳性诊断）
- [[vision-r1-paper]]（§2.2 偏好路线成本对照）

## 相关

- [[instructgpt]] ｜ [[dpo]] ｜ [[reward-model]] ｜ [[grpo]] ｜ [[mixed-preference-optimization]] ｜ [[rule-based-reward]]
