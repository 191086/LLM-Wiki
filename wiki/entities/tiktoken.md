---
type: entity
title: tiktoken
created: 2026-09-12
updated: 2026-09-12
tags:
  - library
  - tokenization
sources: 1
---

# tiktoken

**定义**：OpenAI 开源的分词库（Rust 实现 + Python 绑定），承载 byte-level [[bpe-tokenization|BPE]] 及 `cl100k_base`、`o200k_base` 等编码；[[qwen]] 的分词器即基于它实现。

## 详情

- 编码定义在 `tiktoken_ext/openai_public.py`：cl100k 的预切分正则为 `[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+|…`（数字最多三位一组，所有格量词 `++` 防回溯）——2026-09-12 从源码核验，用于 [[pretokenization-cjk-and-mixed-text]]
- `allowed_special` / `disallowed_special` 参数控制输入文本中特殊 token 表面形式的处理方式，是 [[special-tokens|注入防护]]的 API 出处（[[qwen-tokenization-note]] 引用了其 `core.py` 文档）
- 对照：HuggingFace `tokenizers`（Rust）走的是另一套实现，[[building-a-fast-bpe-tokenizer-from-scratch]] 称其比纯 Python 快 10–100×

## 来源

- [[qwen-tokenization-note]]

## 相关

- [[qwen]]
- [[bpe-tokenization]]
- [[special-tokens]]
- [[pretokenization]]
