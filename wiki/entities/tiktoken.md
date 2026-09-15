---
type: entity
title: tiktoken
created: 2026-09-12
updated: 2026-09-15
tags:
  - library
  - tokenization
sources: 1
---

# tiktoken

**定义**：OpenAI 开源的分词库（Rust 实现 + Python 绑定），承载 byte-level [[bpe-tokenization|BPE]] 及 `cl100k_base`、`o200k_base` 等编码；[[qwen]] 的分词器即基于它实现。

## 基本用法（cl100k 词表，本机实测）

```python
>>> import tiktoken
>>> enc = tiktoken.get_encoding("cl100k_base")
>>> enc.encode("Hello, 使用GPT-4！")
[9906, 11, 86758, 38, 2898, 12, 19, 6447]
>>> enc.decode(ids)
'Hello, 使用GPT-4！'                        # roundtrip 无损
>>> enc.encode_ordinary('print("<|endoftext|>")')   # 表面形式按普通文本编码
[1374, 9836, 91, 8862, 728, 428, 91, 83698]
```

`encode` 对特殊 token 表面形式的处理由 `allowed_special` / `disallowed_special` 两个开关控制，是 [[special-tokens|注入防护]]的 API 出处（[[qwen-tokenization-note]] 引用了其 `core.py` 文档）；四种行为与封装默认值差异的对照见 [[special-tokens]] §4。

## cl100k 预切分正则（2026-09-15 从源码核验）

```python
'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s'
```

与 GPT-2 正则（[[pretokenization]] §2）同族但有代际差异：数字最多三位一组（`\p{N}{1,3}`）、单个标点可黏到后词前（`[^\r\n\p{L}\p{N}]?+` 的可选前缀，如 `"word` 整体成块）。曾用于 [[pretokenization-cjk-and-mixed-text]] 的对照实验。

量词清一色是 `++`、`?+`、`{1,3}+`——**所有格量词**（possessive quantifier）：匹配成功后**拒绝回溯**。普通贪心量词在后文匹配失败时会吐出已吃字符重试，交替分支多时最坏可到指数级（灾难性回溯）；所有格版本一次匹配定死、不参与回溯——分词是热路径，这是必要的工程选择。代价是语义损失：所有格排除了「让出字符换全局匹配」的可能，写正则时须确保各分支无需让步（此正则的分支顺序即为此设计）。

## 对照：HuggingFace tokenizers

HuggingFace `tokenizers`（Rust）走的是另一套实现，[[building-a-fast-bpe-tokenizer-from-scratch]] 称其比纯 Python 快 10–100×。

## 来源

- [[qwen-tokenization-note]]

## 相关

- [[qwen]]
- [[bpe-tokenization]]
- [[special-tokens]]
- [[pretokenization]]
