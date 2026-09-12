---
type: concept
title: BPE（Byte-Pair Encoding，字节对编码）
created: 2026-09-12
updated: 2026-09-12
tags:
  - tokenization
  - algorithm
sources: 1
---

# BPE（字节对编码）

**定义**：一种子词分词算法——从基础词表（现代实现为 256 个字节值）出发，迭代合并语料中出现频率最高的相邻符号对，直至达到目标词表规模。是当前 LLM 的标准分词方法。

## 详情

### 算法流程

1. [[pretokenization|预切分]]：用正则把语料切成词块，统计各词块频次
2. 统计所有相邻符号对的频次（按词块频次加权）
3. 合并最高频对（并列时按字节串字典序取较大者），新符号加入词表
4. 重复 m 次（m = vocab_size − 256 − 特殊 token 数）

### 历史脉络

- **1994**：Philip Gage 提出作为数据压缩算法
- **2016**：Sennrich et al. 引入神经机器翻译，用 `</w>` 词尾标记处理词边界
- **GPT-2 以降**：byte-level BPE + 预切分正则成为 LLM 事实标准

### byte-level 变体

- 基础词表恒为 256（每字节一个 token），任意 Unicode 文本都可表示 → **无 UNK 问题**
- 多字节字符先拆后合：如「你」(U+4F60) 编码为 `E4 BD A0`，初始是三个 token，BPE 会在训练中学着合并
- 词边界改由预切分正则保证，不再需要 `</w>` 标记；空格通常附着在下一个词前（` the`）

### 训练复杂度

朴素实现 O(m×W×L)；经增量更新、并行预切分、倒排索引、堆+惰性删除四项优化，可到 **O(W×L²·logP) 且与 m 无关**，累计实测约 230× 加速（详见 [[building-a-fast-bpe-tokenizer-from-scratch]]）。

## 来源

- [[building-a-fast-bpe-tokenizer-from-scratch]]

## 相关

- [[pretokenization]]
- [[tinystories]]（常用 BPE 基准语料）
