---
type: source
title: Building a Fast BPE Tokenizer from Scratch
created: 2026-09-12
updated: 2026-09-12
tags:
  - tokenization
  - bpe
  - algorithm-optimization
sources: 1
raw: "raw/Building a Fast BPE Tokenizer from Scratch.md"
---

# Building a Fast BPE Tokenizer from Scratch

> 原文：[[raw/Building a Fast BPE Tokenizer from Scratch]] ｜ 类型：技术博文 ｜ 作者：[[jun-yu-tan]] ｜ 发表：2025-11-20

## 一句话总结

从朴素实现出发，用五级递进优化（增量更新 → 并行预切分 → 倒排索引 → 堆 + 惰性删除）把 [[bpe-tokenization|BPE]] 训练加速约 **230×**（21MB 语料、词表 5000：1341s → 5.8s），每一级都给出复杂度推导与实测交叉点。实现基于 [[stanford-cs336]] Assignment 1，基准语料为 [[tinystories]]。

## 关键要点

### 算法与瓶颈

- BPE：从 256 个单字节出发，迭代合并最高频相邻对直至目标词表；先经 [[pretokenization|预切分]] 切成词块
- 朴素实现总复杂度 O(m×W×L)，核心低效：每次 merge 都**从头重算全部 pair 频次**、**扫描全部 W 个词**找受影响者
- 关键观察：一次 merge (A,B)→AB 只影响含该 pair 的词，其余 pair 频次不变——朴素做法存在大量重复劳动

### 五级优化阶梯

| 版本 | 优化手段 | 单次 merge 复杂度 |
| --- | --- | --- |
| V1 | 朴素实现 | O(W×L) |
| V2 | 增量更新 pair 频次 | O(W×L)（常数改善 5–15×） |
| V3 | + 并行预切分（按文档边界切块，multiprocessing） | 同 V2（预切分 O(n/p)） |
| V4 | + 倒排索引 pair→词集合 | O(P + A×L) |
| V5 | + 最大堆 + 惰性删除 + 堆压缩 | O(A×L·logP) |

- **V4 的漂亮上界**：每个词每次被 merge 至少缩短 1，初始长 L₀ 的词至多被更新 L₀−1 次，故总更新量 Σ(A) ≤ W×L，与 merge 次数无关
- **V5 的技巧**：用 `ReversedBytes` 反转字节比较，把 max-heap 语义套进 Python `heapq` 的 min-heap；惰性删除留下的陈旧堆项靠「堆大小 > 3× 有效 pair 数」阈值触发重建压缩——不压缩时内存膨胀 2×，拖垮缓存性能反被 V4 追平
- **V5 总复杂度** O(W×L²·logP)，表达式中不含 m——词表越大优势越大

### 何时用哪个

- 实测交叉点 vocab≈1300：低于此 V4 略胜（堆的常数开销未摊销），高于此 V5 一致胜出；生产词表 32K–100K，堆是标配
- 小词表（<2000）：倒排索引是主要收益；大语料：并行预切分恒有收益（500MB 8 核：~30s → ~5s）

### 经验教训

1. **先 profile 再优化**：预想堆总是赢，实测小词表时开销反而占优，只有基准测试暴露了交叉点
2. **内存即性能**：惰性删除的 2× 内存膨胀曾让 V5 慢于 V4，堆压缩后才反超
3. **同阶复杂度不代表同速**：V2 与 V1 同为 O(m×W×L)，实测快 5–15×，常数因子藏着重大的实际差异

## 值得追踪的实体与概念

- 概念：[[bpe-tokenization]]、[[pretokenization]]
- 实体：[[jun-yu-tan]]、[[tinystories]]、[[stanford-cs336]]
- 后续来源候选：HuggingFace `tokenizers`（Rust 实现，文中称比纯 Python 快 10–100×）；Zouhar et al. 2023 的 BPE 复杂度形式化分析（堆优化同思路）
