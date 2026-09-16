---
type: source
title: InstructGPT：用人类反馈训练语言模型跟随指令（论文）
created: 2026-09-16
updated: 2026-09-16
tags:
  - paper
  - alignment
  - rlhf
  - instruction-tuning
sources: 1
raw: raw/instructgpt.pdf
---

# Training language models to follow instructions with human feedback

> 原文：[[raw/instructgpt.pdf]] ｜ 类型：论文（arXiv:2203.02155**v1**，2022-03-04，68 页）｜ 机构：OpenAI（Alignment 团队）｜ 作者：Long Ouyang、Jeff Wu、Xu Jiang、Diogo Almeida 等（多人共同一作；Jan Leike、Ryan Lowe 为团队负责人）

> **收录注记**：库内这份是 arXiv **v1**。图号与后来被广泛引用的 v2 不同——本文 Figure 2（三步管线图）在 v2 中是 Figure 1；v1 亦无 v2 补充的「RM 验证精度随参数/数据量增长」曲线。下文所有定位一律按 v1 页码与编号。

## 一句话总结

GPT-3 的下一词预测目标与「听懂用户指令」错位（misaligned），用三阶段人类反馈微调修复——标注员示范做 SFT、偏好排序训 6B 奖励模型（RM）、PPO 对 RM 优化并加逐 token KL 惩罚拴住 SFT 策略（PPO-ptx 变体再混预训练梯度）——结果 **1.3B InstructGPT 的输出在人类评估中被偏好胜过 175B GPT-3**，真实性与毒性同步改善，对齐算力成本仅约预训练的 2%；RLHF 由此从摘要等窄任务上的小规模实验变成对话 LM 对齐的主流范式。

## 关键要点

- **问题定义**（§1）：预训练目标（网页文本下一词预测）≠「有帮助且安全地跟随指令」，故称语言建模目标 misaligned；对齐用 3H 框架操作化——helpful（解决用户任务）/ honest（不编造不误导）/ harmless（不造成伤害）（§3.6）。论文明确：对齐到的是「特定人群（标注员 + 研究者 + API 客户）表达出的偏好」，不是抽象的「人类价值观」（§5.2）
- **三步方法**（§3.1，Figure 2）：①SFT——标注员对 prompt 写示范，监督微调 GPT-3；②RM——对同一 prompt 的 K 个模型输出做排序，训奖励模型预测人类偏好；③PPO——以 RM 分为奖励微调 SFT 策略。②③可迭代：对当前最优策略继续采比较数据、训新 RM 再训新策略
- **数据规模**（§3.2，Table 6）：SFT ~13k 训练 prompt（11,295 标注员写 + 1,430 客户）、RM 33k（6,623 + 26,584）、PPO 31k（纯客户 prompt、无人工标签）；标注形式为 K=4–9 个输出从优到劣排序，每 prompt 展开出 $\binom{K}{2}$ 个偏好对——「排序一次产多对」是标注效率的关键设计
- **标注员与一致率**（§3.4）：约 40 名承包商（Upwork / ScaleAI），经筛选测试选拔、共享聊天室答疑；训练标注员互相一致率 **72.6±1.5%**，held-out 标注员 77.3±1.3%（对照 Stiennon et al. 2020 摘要任务的研究员一致率 73±4%）
- **SFT 细节**（§3.5、C.1）：16 epochs、residual dropout 0.2、cosine LR；验证损失 1 epoch 后即过拟合，但训满 16 epochs 的模型 RM 分与人类偏好评分反而更高——模型选择按 **RM 分**而非验证损失
- **RM 细节**（§3.5 式 1；C.2）：只用 **6B** RM——175B RM 训练不稳定、不适合作 PPO 值函数初始化，且算力代价大。损失为 Bradley-Terry 成对排序（式 1：$-\frac{1}{\binom{K}{2}}\mathbb{E}\big[\log\sigma(r_\theta(x,y_w)-r_\theta(x,y_l))\big]$）；两个关键工程：同一 prompt 的全部 $\binom{K}{2}$ 对打包成**单个 batch 元素**（各对高度相关，拆开则单 epoch 即过拟合；同时把每回答前向次数从 $\binom{K}{2}$ 降到 1）；训练前用 bias 把示范数据均分归 0（损失平移不变，需人为定锚，见 [[reward-model]] §4.3）
- **[[ppo|PPO]] 细节**（§3.5 式 2；C.4）：bandit 环境（一句 prompt 采一条回答即终局结算）；**逐 token KL 惩罚**（相对 SFT 模型，β=0.02）缓解 RM 过度优化；值函数从 RM 初始化。**PPO-ptx** = 式 2 再加 γ·预训练对数似然（γ=27.8，预训练样本量 8× 于 RL episodes）以缴「对齐税」；本文 InstructGPT 默认指 PPO-ptx
- **主结果：小模型赢大模型**（§4.1，Figure 1/3）：1.3B InstructGPT 输出被偏好胜过 175B GPT-3；175B InstructGPT 对 175B GPT-3 胜率 **85±3%**、对 few-shot GPT-3 **71±4%**；方法阶梯 GPT < GPT(prompted) < SFT < PPO ≈ PPO-ptx 在三个规模一致；held-out 标注员给出同样偏好（不是过拟合训练标注员）；RM 5 折跨标注组精度 69.6±0.9%（组内 72.4±0.4%）
- **公共 NLP 数据 ≠ 真实使用分布**（§4.1，Figure 5）：175B GPT-3 在 FLAN / T0++（各约 100 万例，按 RM 分选 checkpoint）微调后反而**不及 SFT 基线**；head-to-head InstructGPT 对 FLAN / T0 胜率 78±4% / 79±4%。原因：分类 + QA 只占 API 真实用量的约 18%，开放生成 + 头脑风暴占约 57%（Table 1），公共数据集覆盖不了后者
- **真实性 ↑ 毒性 ↓ 偏见 ✗**（§4.2）：TruthfulQA 上真实且信息量大的回答约为 GPT-3 的 2×（给「不确定就说 I have no comment」指令后，PPO 模型宁可真实而无信息量，GPT-3 不然）；闭域任务幻觉率 21% vs GPT-3 41%；RealToxicityPrompts 在「尊重」指令下毒性输出 −25%（无指令则优势消失；**被明确要求有毒时比 GPT-3 更毒**——「听指令」本身是双刃剑，§5.3）；Winogender / CrowS-Pairs 偏见无改善
- **对齐税与 PPO-ptx**（§4.2，Figure 29/33/34）：纯 PPO 在 SQuADv2、DROP、HellaSwag、WMT15 Fr→En 上回退；PPO-ptx 混预训练梯度基本修复（HellaSwag 反超 GPT-3），DROP/SQuADv2/翻译仍落后；对照组：加大 KL 系数导致验证奖励大跌且修不回 DROP/SQuAD——**缴对齐税靠混预训练分布，不靠收紧约束**
- **分布外泛化与残余错误**（§4.3）：能跟随非英语指令、做代码问答与摘要（训练分布中占比极小），GPT-3 做同样的事需要精细 prompting；仍会：假前提照单全收、简单问题过度对冲（标注指南奖励「认知谦逊」被 RM 放大）、多显式约束指令性能下降
- **成本账**（§5.1）：175B SFT 4.9、175B PPO-ptx 60 petaflops/s-days，对比 GPT-3 预训练 3,640——对齐总开销约 2%，收益却超过 100× 参数增长；「当下投资对齐比做大模型更划算」的原始实证
- **自陈局限**（§5.2–5.3）：对齐对象三层都不代表全体受影响者（40 名多为美国/东南亚英语使用者的标注员；研究者写的标注指南；早期 waitlist 偏向 OpenAI 自己人脉的 API 客户）；多数比较数据仅 1 人标注；模型「太听话」——用户要求有害输出时照做

![[instructgpt-winrate-vs-gpt3.png]]
> 本文 Figure 1（§4.1）：以 175B SFT 为基线的胜率随模型规模变化。GPT < GPT(prompted) < SFT < PPO ≈ PPO-ptx 的阶梯在三个规模上一致；1.3B PPO-ptx 已越过 50% 线——1.3B InstructGPT 净胜 175B SFT，摘要的直接对拍进一步给出净胜 175B GPT-3（175B 对 175B 为 85±3%）。

## 值得追踪的实体与概念

- [[instructgpt]]（模型本体：三档规模、配方速览、部署形态）
- [[ppo]]（第 3 阶段的 RL 算法本体：clip 替代目标、截断 GAE、完整算法——本文式 2 的 PPO-ptx 是其 LLM 工程变体）
- [[rlhf]]（范式页：管线机制、形化奖励走查、PPO/DPO/MPO/GRPO 路线谱系——本文是其主锚来源）
- [[reward-model]]（第 2 阶段的产物：BT 排序损失即本文式 1，平移不变性的工程应用）
- [[dpo]]（后续工作：把本文第 2+3 阶段折叠为一个分类损失）｜ [[grpo]]（免 critic 的在线 RL 后续路线）
- [[vision-r1-paper]]（免 RM、免偏好标注的对照路线）
