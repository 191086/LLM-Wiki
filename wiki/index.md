---
type: index
title: 索引
created: 2026-09-12
updated: 2026-09-15
---

# 索引

> 全库目录，每次 ingest 或沉淀分析后更新。查询时先读此页。

## 总览

- [[overview]] — 对整个知识域的当前综合理解（两条主线：分词 / 多模态领域化）

## 来源（sources/）

- [[building-a-fast-bpe-tokenizer-from-scratch|Building a Fast BPE Tokenizer from Scratch]] — 五级递进优化把 BPE 训练加速 ~230×（Jun Yu Tan，2025-11，博文）
- [[qwen-tokenization-note|Qwen 官方 Tokenization Note]] — Qwen-7B 分词器工程说明：token 体系、注入防护、词表扩展（QwenLM 官方文档）
- [[ostrakon-vl-paper|Ostrakon-VL（论文）]] — 餐饮零售领域多模态模型三件套：8B 模型反超 235B、数据压缩 20.4× 反升分（淘宝闪购，2026-01，arXiv）
- [[skywork-vl-reward-paper|Skywork-VL Reward（论文）]] — 开源 7B 多模态奖励模型：判别式 ORM + 19 万对三阶段清洗偏好数据，VL-RewardBench 73.1% 超 GPT-4o（昆仑万维，2025-05，arXiv）
- [[vision-r1-paper|Vision-R1（论文）]] — 免 RM 的视觉规则强化学习：三路准则奖励 + 渐进收紧 + GRPO，Qwen2.5-VL-7B 定位 mAP +50%、ODINW-13 反超 72B（CASIA，2025-03，arXiv）

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

## 概念（concepts/）

- [[bpe-tokenization|BPE（字节对编码）]] — LLM 标准分词算法：训练/编码/解码全流程伪代码与手工示例、byte-level 特性、复杂度五级优化阶梯（230×）、纯分布陷阱
- [[pretokenization|预切分]] — BPE 前用正则切词块防跨词合并：GPT-2 正则逐段拆解 + 中英混合/弯引号/空白分工手工走查；中文行为见分析页
- [[special-tokens|特殊 Token]] — regular/special 平行双轨类型系统；注入攻击与防护（两开关四行为，tiktoken 抛错默认 vs Qwen 全解析默认）
- [[vocabulary-expansion|词表扩展]] — 训后追加 token：add_merges 端到端走查（六条新 merge 逐轮还原）、预切分/码点/优先级三坑与官方流程
- [[quad-data-curation|QUAD（数据清洗管线）]] — 四阶段蒸馏 69.25M→3.40M（20.4×）反升 2.5 分；质量 > 数量
- [[mixed-preference-optimization|MPO（混合偏好优化）]] — DPO+BCO+SFT 三项损失落到公式与数值对比走查；离线偏好对齐、偏好对构造与 GRPO 取舍
- [[dpo|DPO（直接偏好优化）]] — RLHF 两阶段折叠为单个偏好损失：奖励可从策略/参考模型 log 概率比恢复（重参数化推导）；机制为 wiki 外知识标注
- [[reward-model|奖励模型]] — RM 两轴分类（判别/生成/隐式 × ORM/PRM）；BT 排序损失数值走查：只学相对序的后果与用途；规则奖励是其系外成员
- [[grpo|GRPO（组相对策略优化）]] — 无 critic 的 RL：组内相对优势当基线，R1 式训练底座；组内分化决定信号强弱，在线组采样是代价
- [[rule-based-reward|规则奖励]] — 免 RM / 免偏好标注的评价信号：格式 + 召回 + 精度三路程序化打分；reward hacking 与渐进收紧对策；与学习式 RM 对照
- [[domain-specific-mllm|领域专用多模态大模型]] — 通用 MLLM 三重错位、领域化配方（数据管线 + 多阶段训练 + 自建基准）与各领域成型栈；参数效率论证与代价

## 分析（analyses/）

- [[pretokenization-cjk-and-mixed-text|预切分如何处理中文与中英混合文本]] — `\p{L}` 不分文字系统：汉字串整块、混合文本同 chunk；主流 tokenizer 均无 CJK 专门规则；L 暴增实验（显式句对可复现，$L^2$ 放大 ≈12.8×）
