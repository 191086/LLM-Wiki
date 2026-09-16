---
type: concept
title: 奖励模型（Reward Model）
created: 2026-09-14
updated: 2026-09-16
tags:
  - reward-model
  - alignment
  - rlhf
sources: 5
---

# 奖励模型（Reward Model, RM）

**定义**：对（输入，候选回答）输出质量分数的模型，为对齐训练（[[rlhf]] / 偏好优化）和推理时选择提供评价信号；[[skywork-vl-reward-paper]] §2 给出两轴分类法。

## 1. 两轴分类法

**按模型形态**分三类，区别在「分数从哪来」。**判别式 RM** 把偏好预测当（标量）回归问题：输入候选回答，网络直接输出一个分数（或偏好概率）——[[skywork-vl-reward|Skywork-VL-Reward]] 属此类（拆掉 LM 头、接奖励头回归）。**生成式 RM** 反过来用 LM 头：按评审 prompt 生成一段评判结论（LLM-as-judge 风格），分数隐含在生成文本里而非数值头输出——多数早期多模态 RM 走这条路。**隐式 RM** 不需要独立的打分器：偏好学习通过 [[dpo]] 式重参数化隐含在策略模型内部（奖励可从 $\beta \log \frac{\pi_\theta}{\pi_\text{ref}}$ 读出——[[dpo-paper]] §5.1，Theorem 1 证明该表示不损失一般性；推导见 [[dpo]] §2）。

**按反馈目标**分两类，区别在「给什么打分」。**结果奖励（ORM，Outcome Reward Model）**对整条回答给一个分，只评价最终产出。**过程奖励（PRM，Process Reward Model）**对回答的中间步骤逐步打分——文本侧的代表是「Let's Verify Step by Step」：

> wiki 外补注：Lightman et al. 2023（OpenAI）：用 GPT-4 逐步标注数学解题过程的对错，训 PRM 并在测试时按步搜索，证明过程监督优于结果监督。多模态侧的代表 VisualPRM——过程监督 + 作测试时扩展的 critic 提升已有 VLM 的推理（[[skywork-vl-reward-paper]] §2）。

## 2. 系外成员：规则奖励（无 RM 路线）

两轴分类的都是「学出来的打分器」；[[rule-based-reward|规则奖励]]干脆不要打分器——用程序化规则（格式校验 / 精确匹配 / 任务指标）直接从结果算奖励：零训练、可审计、绝对分有意义，代价是只适用于结果可程序化验证的任务。DeepSeek-R1 式训练（[[grpo]]）与 [[vision-r1]] 走这条路线；与学习式 RM 的逐项对照见 [[rule-based-reward]] §4。

## 3. 多模态 RM 的两大短板

文本 RM 研究充分，多模态 RM 起步晚，短板有二（[[skywork-vl-reward-paper]] §1）：①**跨任务泛化不足**；②**评不了 reasoner**（复杂长链推理输出）。Skywork-VL-Reward 的补法 = 广覆盖偏好数据（含专门的推理风格回答）+ 判别式标量头，在 VL-RewardBench 的 general / 幻觉 / reasoning 三类目上同时拿到可用成绩。

## 4. Bradley-Terry 排序损失：只学相对序

### 4.1 公式

**Bradley-Terry 模型**（Bradley & Terry, 1952）：每个候选回答有一个标量强度分，$y^+$ 优于 $y^-$ 的概率只由分差过 sigmoid 给出，训练目标取观测偏好的负对数似然——RLHF 奖励建模的标配损失（[[dpo-paper]] 式 1–2）：

$$P(y^+ \succ y^- \mid x) = \sigma\!\big(s^+ - s^-\big) \quad\Rightarrow\quad \mathcal{L} = -\log \sigma\!\big(s^+ - s^-\big), \qquad s^\pm = r_\theta(x, y^\pm)$$

InstructGPT 的 6B RM 即用此损失（[[instructgpt-paper]] §3.5 式 1，除以 $\binom{K}{2}$ 作归一化），并配两个工程细节：①标注员一次给 K=4–9 个输出排序、展开出全部 $\binom{K}{2}$ 偏好对，这些对**高度相关**——拆成独立样本训则单 epoch 即过拟合，InstructGPT 把同一 prompt 的所有对打包成**单个 batch 元素**（顺带把每回答的前向次数从 $\binom{K}{2}$ 降到 1）；②训练前用 bias 把示范数据均分归 0 后再做 RL——损失平移不变、绝对分无锚，必须人为定一个零点（§4.3 推论 1 的直接应用）。

### 4.2 数值走查

取三对分数，手算损失与梯度（$d \coloneqq s^+ - s^-$，$\partial \mathcal{L}/\partial d = -\sigma(-d)$）：

| 情形 | $s^+$ | $s^-$ | $d$ | $\mathcal{L} = -\log\sigma(d)$ | $\partial\mathcal{L}/\partial d$ |
| --- | --- | --- | --- | --- | --- |
| 排对但接近 | 1.2 | 0.3 | 0.9 | $-\log 0.7109 = 0.341$ | $-0.289$ |
| **排错** | 0.3 | 1.2 | $-0.9$ | $-\log 0.2891 = 1.241$ | $-0.711$ |
| 排对且拉开 | 5.2 | 1.3 | 3.9 | $-\log 0.9802 = 0.020$ | $-0.020$ |

读法：梯度恒为负（梯度下降总是增大 $d$、拉开分差），但**排错时梯度最大（0.711），排对且分差拉开后趋零（0.020）**——损失只负责「把没分开的对分开」，已分开的不再加力。

### 4.3 三个直接后果

1. **平移不变**：给同一 prompt 的所有候选分加同一常数，损失不变——绝对分值在数学上不可辨识、单独看无意义，只能做差值 / 阈值比较。[[quad-data-curation|QUAD]] 对 $R_\phi$ 的三处用法（质量过滤、基座参考过滤、OCL 难度投票）全部如此，阈值须验证集 + 人工审计标定；InstructGPT 在 RL 前用 bias 把示范均分归 0，就是这个性质的规范化应用（[[instructgpt-paper]] §3.5）。
2. **梯度随 margin 饱和**：即 §4.2 第三行所见。
3. **「等质」偏好对引入歧义**：两条真等价的回答也会被强行拉出一个任意序，训练数据须主动剔除（Skywork 的判决过滤把与 GPT-4o「等质」判断的对全部丢弃）。

推论（分析）：BT 只需要「哪个更好」的比较、不需要绝对分数标注，而人类判比较远比打绝对分可靠（[[dpo-paper]] §2：相对判断比专家示范更易收集，这是偏好路线的出发点）——这是它成为 RLHF 奖励建模标配的根本原因；[[mixed-preference-optimization|MPO]] 之所以要混入 BCO 质量项，补的正是 BT 缺的绝对质量信号。

## 5. 在本 wiki 中的角色

- [[skywork-vl-reward]]：判别式 ORM 的开源代表，发布时 VL-RewardBench 榜首，被 [[ostrakon-vl]] 全管线取用
- [[quad-data-curation]]：RM 作数据清洗裁判——质量过滤、基座参考过滤、课程难度分层
- [[mixed-preference-optimization]]：RM 产偏好对驱动 MPO；数据质量直接决定上限（MathVista 同配方换 RM：71.2 / 71.8 / 73.5）
- [[instructgpt-paper]] §4.1：RM 精度的参照样本——6B RM 预测训练标注员偏好 72.4±0.4%，5 折跨标注组 69.6±0.9%，而人与人的训练标注员一致率本身只有 72.6±1.5%：RM 上限被人类标注一致性封顶

## 来源

- [[instructgpt-paper]]（§3.5 式 1 BT 损失与整提示打包、bias 归一化、6B RM 选型；§3.4/§4.1 标注一致率与 RM 精度）
- [[dpo-paper]]（式 1–2 BT 奖励建模、§5.1 隐式 RM 重参数化）
- [[skywork-vl-reward-paper]]（分类学与排序损失性质）
- [[ostrakon-vl-paper]]（RM 作为清洗裁判的用法）
- [[vision-r1-paper]]（无 RM 的规则奖励路线）

## 相关

- [[rlhf]]（RM 在三阶段管线中的位置）｜ [[instructgpt]]（范式定标模型）｜ [[skywork-vl-reward]] ｜ [[quad-data-curation]] ｜ [[mixed-preference-optimization]] ｜ [[ostrakon-vl]] ｜ [[rule-based-reward]] ｜ [[dpo]] ｜ [[grpo]]
