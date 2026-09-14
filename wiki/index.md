---
type: index
title: 索引
created: 2026-09-12
updated: 2026-09-14
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

## 实体（entities/）

- [[jun-yu-tan|Jun Yu Tan]] — 博主（jytan.net），BPE 优化文作者，CS336 学员
- [[stanford-cs336|Stanford CS336]] — 《Language Modeling from Scratch》课程
- [[tinystories|TinyStories]] — 儿童故事数据集，常用基准语料
- [[qwen|Qwen（通义千问）]] — 阿里开源模型家族；两个切面：byte-level BPE 分词器工程样本、Qwen3-VL-8B 领域微调基座
- [[tiktoken]] — OpenAI 分词库；Qwen 分词器与 cl100k/o200k 编码的宿主
- [[ostrakon-vl|Ostrakon-VL（模型）]] — 淘宝闪购的 FSRS 领域多模态模型（Qwen3-VL-8B 基座），ShopBench 60.1
- [[shopbench|ShopBench（基准）]] — 首个餐饮零售多模态基准：5,818 题 / L1-L4 / 单图·多图·视频；VNR/VIF 指标
- [[skywork-vl-reward|Skywork-VL-Reward（奖励模型）]] — 昆仑万维开源 7B 多模态奖励模型（Qwen2.5-VL-7B 基座）；QUAD/OCL 统一打分器，原样取用未做领域适配

## 概念（concepts/）

- [[bpe-tokenization|BPE（字节对编码）]] — LLM 标准分词算法；训练复杂度、优化路径与纯分布特性
- [[pretokenization|预切分]] — BPE 前用正则切词块，防止跨词合并；中文行为见分析页
- [[special-tokens|特殊 Token]] — 功能型 token 类型系统；表面形式注入与防护
- [[vocabulary-expansion|词表扩展]] — 训后追加 token：中间 merge 链、约束与坑
- [[quad-data-curation|QUAD（数据清洗管线）]] — 四阶段蒸馏 69.25M→3.40M（20.4×）反升 2.5 分；质量 > 数量
- [[mixed-preference-optimization|MPO（混合偏好优化）]] — DPO+BCO+SFT 三项损失；偏好对构造与 GRPO 取舍
- [[reward-model|奖励模型]] — RM 两轴分类（判别/生成/隐式 × ORM/PRM）；排序损失只学相对序的后果与用途
- [[domain-specific-mllm|领域专用多模态大模型]] — 通用 MLLM 的三重错位与各领域成型栈；参数效率论证与代价

## 分析（analyses/）

- [[pretokenization-cjk-and-mixed-text|预切分如何处理中文与中英混合文本]] — `\p{L}` 不分文字系统：汉字串整块、混合文本同 chunk；主流 tokenizer 均无 CJK 专门规则（实验验证）
