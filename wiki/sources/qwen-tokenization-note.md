---
type: source
title: Qwen 官方 Tokenization Note
created: 2026-09-12
updated: 2026-09-12
tags:
  - tokenization
  - bpe
  - qwen
sources: 1
raw: "raw/Qwentokenization_note.md at main.md"
---

# Qwen 官方 Tokenization Note

> 原文：[[raw/Qwentokenization_note.md at main.md]] ｜ 类型：官方工程文档 ｜ 作者：[[qwen]] 团队 ｜ [GitHub](https://github.com/QwenLM/Qwen/blob/main/tokenization_note.md)

## 一句话总结

[[qwen]]-7B 分词器的官方说明：基于 [[tiktoken]] 的 byte-level [[bpe-tokenization|BPE]]，讲清 token 类型体系（regular bytes / special str）、[[special-tokens|特殊 token]] 与注入攻击防护、[[vocabulary-expansion|词表扩展]]的正确姿势与坑——核心主题是「BPE 纯粹按分布工作，没有任何 Unicode / 语言学知识」。

## 关键要点

### Token 体系

- **Regular tokens**（`bytes` 型）：从 UTF-8 字节序列学出的 BPE token；**special tokens**（`str` 型）：系统注入的功能符号（`<|endoftext|>` 等），理论上不出现在输入文本中
- 词表：**151,643 regular + 208 control**，扩展起始 index 151,851；`<|extra_0|>`…`<|extra_204|>` 留给用户
- byte-level 保证无 UNK，但生截断处可能出现不完整字节序列 → decode 出替换符 `�`（可用 `errors="ignore"`）

### Special tokens 语义

- Qwen-7B 用 `<|endoftext|>`；Chat 版另有 `<|im_start|>`、`<|im_end|>`
- `bos`/`eos`/`unk`/`mask`/`sep` 对预训练模型**不适用**；`pad` 理论上任意已知 token 均可（模型从不计算它），实践取 `<|endoftext|>`
- ⚠️ 不要把 `<|endoftext|>` 当 `eos` 用，除非确认「句末 = 文档末」在你的场景成立

### 注入攻击防护

- 默认行为**已改为**把输入文本中特殊 token 的表面形式解析成特殊 token（迁就社区习惯，但不安全）
- 防护：`allowed_special=set()` 把表面形式当普通文本；`disallowed_special=(...)` 直接抛错；细粒度控制传集合

### 词表扩展（详见 [[vocabulary-expansion]]）

- 不能直接加词——BPE 需要**中间 merge 链**；官方 `add_merges.py` 按词频文件学新 merge
- **预切分约束**：跨预切分边界的词加不进去（示例：`夸张的 比喻手法` 会被切成 `['夸张的', ' 比喻手法']`）
- 新 token 需**微调**模型才能生效

### Caveats：纯分布、无语言学知识

- 有限数据上学的 merge 可能**跨 Unicode 码点边界**（如 `b'\x80\xe5'` 先合并）；稳妥做法是把涉及的码点也以更高频加入
- 合并路径受既有 token 优先级支配：`是|一|只|猫 → 是一|只|猫 → 是一|只猫 → 是一只猫`（`是一` 已是 token 且优先级更高）
- **同一词在不同上下文切分不同**：`"Panda"` → `P|anda`，`" Panda"` → 整块，`" Pandas"` → ` Pand|as`

## 值得追踪的实体与概念

- 实体：[[qwen]]、[[tiktoken]]
- 概念：[[special-tokens]]、[[vocabulary-expansion]]、[[bpe-tokenization]]、[[pretokenization]]
- 后续来源候选：SentencePiece 官方文档（码点级 + byte fallback 的对照路线）；tiktoken `core.py`（`allowed_special` 语义出处）
