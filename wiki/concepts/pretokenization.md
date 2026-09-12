---
type: concept
title: 预切分（Pretokenization）
created: 2026-09-12
updated: 2026-09-12
tags:
  - tokenization
  - regex
sources: 1
---

# 预切分（Pretokenization）

**定义**：在运行 [[bpe-tokenization|BPE]] 之前，用正则把原始文本切成「词块」（通常是单词、数字、标点串），BPE 只在词块内部做合并，不跨越边界。

## 为什么需要

- 不预切分的话，BPE 可能学到跨词边界的合并（如 "the" 词尾的 `e` + 空格 + "time" 开头的 `t`），产生与语言单元不对齐的 token
- 保证词作为完整单元参与合并；空格通常附着在**下一个**词前（` the` 而非 `the `）
- 英文缩写（`'s`、`'ll` 等）单独成块

### GPT-2 预切分正则

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

| 片段 | 匹配 | 例 |
| --- | --- | --- |
| `'(?:[sdmt]\|ll\|ve\|re)` | 英文缩写 | `'s`、`'ll` |
| ` ?\p{L}+` | 可选空格 + 字母 | ` the`、`hello` |
| ` ?\p{N}+` | 可选空格 + 数字 | ` 42` |
| ` ?[^\s\p{L}\p{N}]+` | 可选空格 + 标点 | ` ...` |
| `\s+(?!\S)\|\s+` | 空白串 | 换行、尾部空格 |

### 与 Sennrich 方案的对比

- Sennrich et al. 2016：字符级 + `</w>` 词尾标记区分词边界
- GPT-2 起主流：**正则预切分 + byte-level**，不需要特殊标记

## 工程注记

预切分是可并行的独立阶段（按文档边界切块、多进程处理后合并词频），与串行的 merge 循环解耦——这是 BPE 训练优化的第一刀切分点（[[building-a-fast-bpe-tokenizer-from-scratch]]）。

## 来源

- [[building-a-fast-bpe-tokenizer-from-scratch]]

## 相关

- [[bpe-tokenization]]
