---
type: index
title: 索引
created: 2026-09-12
updated: 2026-09-16
---

# 索引

> 全库目录，每次 ingest 或沉淀分析后更新。查询时先读此页。

## 总览

- [[overview]] — 对整个知识域的当前综合理解（三条主线：分词 / 多模态领域化 / 架构与长上下文）

## 来源（sources/）

- [[building-a-fast-bpe-tokenizer-from-scratch|Building a Fast BPE Tokenizer from Scratch]] — 五级递进优化把 BPE 训练加速 ~230×（Jun Yu Tan，2025-11，博文）
- [[qwen-tokenization-note|Qwen 官方 Tokenization Note]] — Qwen-7B 分词器工程说明：token 体系、注入防护、词表扩展（QwenLM 官方文档）
- [[ostrakon-vl-paper|Ostrakon-VL（论文）]] — 餐饮零售领域多模态模型三件套：8B 模型反超 235B、数据压缩 20.4× 反升分（淘宝闪购，2026-01，arXiv）
- [[skywork-vl-reward-paper|Skywork-VL Reward（论文）]] — 开源 7B 多模态奖励模型：判别式 ORM + 19 万对三阶段清洗偏好数据，VL-RewardBench 73.1% 超 GPT-4o（昆仑万维，2025-05，arXiv）
- [[vision-r1-paper|Vision-R1（论文）]] — 免 RM 的视觉规则强化学习：三路准则奖励 + 渐进收紧 + GRPO，Qwen2.5-VL-7B 定位 mAP +50%、ODINW-13 反超 72B（CASIA，2025-03，arXiv）
- [[dpo-paper|DPO（论文）]] — RLHF 两阶段折叠为单个偏好损失：奖励 = β·log(π/π_ref) 重参数化（闭式解 + 配分函数相消），reward-KL 前沿严格支配 PPO、TL;DR 胜率 61% vs 57%（Stanford，2023-05，arXiv:2305.18290 / NeurIPS 2023）
- [[instructgpt-paper|InstructGPT（论文）]] — RLHF 三阶段定标之作：SFT → 6B RM → PPO-ptx 对齐 GPT-3，1.3B 胜 175B GPT-3（175B vs 175B 胜率 85±3%）、闭域幻觉减半、PPO-ptx 缴对齐税；对齐算力 ≈ 预训练 2%（OpenAI，2022-03，arXiv:2203.02155v1）
- [[ppo-paper|PPO（论文）]] — 一阶化置信域：截断概率比的悲观下界替代目标 + 多 epoch 复用采样；clip 消融 0.82 vs 无约束 −0.39，Atari 30 胜 / ACER 18 / A2C 1（OpenAI，2017-08，arXiv:1707.06347v2）
- [[trpo-paper|TRPO（论文）]] — 置信域策略优化：替代目标 − KL 约束更新的单调改进下界（Theorem 1），自然梯度 / 策略迭代统一为特例；MuJoCo 四任务包揽前二、Atari 原始图像七局（Berkeley，ICML 2015，arXiv:1502.05477v5）
- [[rope-paper|RoFormer（论文）]] — 旋转位置编码：把「内积只依赖相对位置差」解成均匀角速度旋转（式 11–16 函数方程），零参数、保范数、长程衰减、线性注意力兼容；WMT14 27.3→27.5、GLUE 三胜三负（QQP +15.2）、CAIL2019-SCM 1024 长文 66.07%（追一科技，2021-04 首版 / v5 2023-11，arXiv:2104.09864）

## 实体（entities/）

- [[jun-yu-tan|Jun Yu Tan]] — 博主（jytan.net），BPE 优化文作者，CS336 学员
- [[stanford-cs336|Stanford CS336]] — 《Language Modeling from Scratch》课程：19 讲七模块、五作业（Asgn1 即 BPE 从零实现）
- [[tinystories|TinyStories]] — 合成儿童故事基准语料（Eldan & Li 2023：<10M 参数亦可连贯）
- [[qwen|Qwen（通义千问）]] — 阿里开源模型家族（含谱系补注）；三个切面：byte-level BPE 分词器工程样本、Qwen3-VL-8B 领域微调基座、Qwen2.5-VL 同模型当 RM 基座兼 RL 靶子
- [[tiktoken]] — OpenAI 分词库（Qwen 分词器宿主）：可运行小例、cl100k 正则（所有格量词）与注入防护 API
- [[ostrakon-vl|Ostrakon-VL（模型）]] — 淘宝闪购的 FSRS 领域多模态模型（Qwen3-VL-8B 基座），ShopBench 60.1
- [[shopbench|ShopBench（基准）]] — 首个餐饮零售多模态基准：5,818 题 / L1-L4 / 单图·多图·视频；VNR/VIF 指标
- [[skywork-vl-reward|Skywork-VL-Reward（奖励模型）]] — 昆仑万维开源 7B 多模态奖励模型（Qwen2.5-VL-7B 基座）；QUAD/OCL 统一打分器，原样取用未做领域适配
- [[vision-r1|Vision-R1（方法）]] — CASIA 的视觉规则 RL 方法（Griffon-G / Qwen2.5-VL 双基座）：准则驱动奖励 + 渐进式规则收紧 + 一条样本奖励走查，human-free alignment
- [[instructgpt|InstructGPT（模型）]] — OpenAI 用 RLHF 对齐的 GPT-3 微调版（1.3B/6B/175B，默认 PPO-ptx）：RLHF 范式的定标实证载体，ChatGPT 前身

## 概念（concepts/）

- [[bpe-tokenization|BPE（字节对编码）]] — LLM 标准分词算法：训练/编码/解码全流程伪代码与手工示例、byte-level 特性、复杂度五级优化阶梯（230×）、纯分布陷阱
- [[pretokenization|预切分]] — BPE 前用正则切词块防跨词合并：GPT-2 正则逐段拆解 + 中英混合/弯引号/空白分工手工走查；中文行为见分析页
- [[special-tokens|特殊 Token]] — regular/special 平行双轨类型系统；注入攻击与防护（两开关四行为，tiktoken 抛错默认 vs Qwen 全解析默认）
- [[vocabulary-expansion|词表扩展]] — 训后追加 token：add_merges 端到端走查（六条新 merge 逐轮还原）、预切分/码点/优先级三坑与官方流程
- [[quad-data-curation|QUAD（数据清洗管线）]] — 四阶段蒸馏 69.25M→3.40M（20.4×）反升 2.5 分；质量 > 数量
- [[mixed-preference-optimization|MPO（混合偏好优化）]] — DPO+BCO+SFT 三项损失落到公式与数值对比走查；离线偏好对齐、偏好对构造与 GRPO 取舍
- [[ppo|PPO（近端策略优化）]] — clip 替代目标四分支走查、截断 GAE（式 10/11 原文排印笔误考订）、自适应 KL 变体与消融证据；RLHF 第 3 阶段的 RL 引擎、GRPO / DPO 的共同对照
- [[trpo|TRPO（置信域策略优化）]] — KL 置信域更新 + 单调改进下界：性能差分解、α-coupling 证明思想与 2 状态 MDP 数值走查（式 1 严格相等、下界松弛 ~13、理论惩罚系数 39,600 不可用）、single path / vine 采样、共轭梯度实现；PPO 一阶化的前身
- [[rlhf|RLHF（人类反馈强化学习）]] — 三阶段管线（SFT→RM→KL 约束 RL），InstructGPT 一手锚定：形化奖励期望恒等式走查（逐样本符号 ≠ 期望含义）、1.3B>175B 实证与对齐税、难训四因、PPO/DPO/MPO/GRPO 四路线谱系
- [[dpo|DPO（直接偏好优化）]] — RLHF 两阶段折叠为单个偏好损失：重参数化推导（式 4–7）+ 梯度动态加权走查（排错加权 0.513 vs 排对 0.401）；β 与 π_ref 性质、隐式 RM、实验证据
- [[reward-model|奖励模型]] — RM 两轴分类（判别/生成/隐式 × ORM/PRM）；BT 排序损失数值走查：只学相对序的后果与用途；规则奖励是其系外成员
- [[grpo|GRPO（组相对策略优化）]] — 无 critic 的 RL：组内相对优势当基线，R1 式训练底座；组内分化决定信号强弱，在线组采样是代价
- [[rule-based-reward|规则奖励]] — 免 RM / 免偏好标注的评价信号：格式 + 召回 + 精度三路程序化打分；reward hacking 与渐进收紧对策；与学习式 RM 对照
- [[domain-specific-mllm|领域专用多模态大模型]] — 通用 MLLM 三重错位、领域化配方（数据管线 + 多阶段训练 + 自建基准）与各领域成型栈；参数效率论证与代价
- [[rope|RoPE（旋转位置编码）]] — 函数方程推出的位置编码：2D 旋转 → $d/2$ 子空间块对角，正交恒等式给出相对语义；长程衰减 Abel 界、线性注意力兼容、七方案对照表；LLaMA 系默认（wiki 外）

## 分析（analyses/）

- [[pretokenization-cjk-and-mixed-text|预切分如何处理中文与中英混合文本]] — `\p{L}` 不分文字系统：汉字串整块、混合文本同 chunk；主流 tokenizer 均无 CJK 专门规则；L 暴增实验（显式句对可复现，$L^2$ 放大 ≈12.8×）
