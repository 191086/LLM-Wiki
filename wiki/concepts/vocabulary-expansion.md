---
type: concept
title: 词表扩展（Vocabulary Expansion）
created: 2026-09-12
updated: 2026-09-12
tags:
  - tokenization
  - bpe
  - fine-tuning
sources: 1
---

# 词表扩展（Vocabulary Expansion）

**定义**：在已训好的 [[bpe-tokenization|BPE]] 词表上追加新 token 的工程流程——不能直接把词塞进词表，因为编码时需要完整的**中间 merge 链**。

## 详情

### 流程（[[qwen]] 官方 `add_merges.py`）

1. 准备 `词<TAB>频次` 的纯文本文件（频次用于计算 BPE）
2. 取基础词表文件（151,643 regular + 208 control，扩展起始 index 151,851）
3. 跑 `add_merges.py` 学新 merge，产出 `qwen_extra.tiktoken`（base64 字节串 → 新 index）
4. 加载时传 `extra_vocab_file`，**并微调模型**——新 token 对预训练模型没有学过含义，不微调不可用

### 约束与坑

- **预切分约束**：跨 [[pretokenization]] 边界的词加不进去（`夸张的 比喻手法` 会被切成 `['夸张的', ' 比喻手法']`，整词无法成为 token）
- **码点边界风险**：有限数据上学出的 merge 可能跨 Unicode 码点（如 `一只` 的字节 `b'\xe4\xb8\x80\xe5\x8f\xaa'` 中 `b'\x80\xe5'` 先合并）——对已知 token 无碍，对未知词可能产生预训练模型难以理解的异常切分。稳妥做法：把涉及的所有码点以更高频一并加入
- **合并路径受既有优先级支配**：`是|一|只|猫` 的路径是 `是一|只|猫 → 是一|只猫 → 是一只猫`，因为 `是一` 已是 token 且优先级高于 `一只`——想加的词不一定按你预期的方式拼出来
- 官方判断：Qwen「已覆盖绝大多数中文词」，通常只加中文词即可

## 来源

- [[qwen-tokenization-note]]

## 相关

- [[bpe-tokenization]]
- [[pretokenization]]
- [[special-tokens]]
- [[qwen]]
