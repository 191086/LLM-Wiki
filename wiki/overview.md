---
type: overview
title: 总览
created: 2026-09-12
updated: 2026-09-12
sources: 1
---

# 总览

> 本页是全局综合页，反映「读过的所有来源叠加之后」的整体理解。每次 ingest 后更新。

**当前状态**：已收录 1 份来源，主题聚焦 **LLM 分词（tokenization）与 BPE 训练优化**。

## 当前主线

- LLM 基础设施中的分词层：[[bpe-tokenization|BPE]] 是现代 LLM 的标准分词算法；其训练本身是一个可系统性优化的算法工程问题（[[building-a-fast-bpe-tokenizer-from-scratch]] 给出了五级优化路径）

## 关键结论

- 朴素 BPE 训练为 O(m×W×L)；增量更新、并行预切分、倒排索引、堆+惰性删除四项累计约 **230×** 加速
- 收益随词表规模变化：vocab≈1300 是倒排索引方案与堆方案的分水岭；生产词表（32K–100K）必用堆
- 常数因子不可忽视：同阶复杂度的实现可差 5–15×；惰性删除不配堆压缩会因内存膨胀变慢

## 未解决的疑问

- HuggingFace `tokenizers`（Rust）与纯 Python 实现在工程上的具体差异？文中仅提及其快 10–100×
- BPE 在中文语料上的行为：多字节汉字（「你」= 3 字节起步）如何影响合并路径与最终词表效率？

## 相关来源

- [[building-a-fast-bpe-tokenizer-from-scratch]]
