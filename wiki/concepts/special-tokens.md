---
type: concept
title: 特殊 Token（Special / Control Tokens）
created: 2026-09-12
updated: 2026-09-12
tags:
  - tokenization
  - security
sources: 1
---

# 特殊 Token（Special / Control Tokens）

**定义**：不对应任何输入文本、只由系统在文本处理后才注入的功能 token；其人类可读的表面形式（如 `<|endoftext|>`）仅为引用方便，与 regular token（`bytes` 型）是两个类型系统。

## 详情

### 语义边界（以 [[qwen]] 为例）

- `<|endoftext|>`：文档结束（Qwen-7B）；Chat 版另有 `<|im_start|>` / `<|im_end|>` 标记对话轮次
- `bos` / `eos` / `unk` / `mask` / `sep` 概念对预训练模型**不适用**——模型没为它们学过含义，乱设可能引入未定义行为
- `pad` 是唯一例外的「无语义」特殊 token：模型理论上从不计算它，可用任意已知 token 充当
- ⚠️ `<|endoftext|>` ≠ `eos`：句末与文档末（可含多句）不是一回事

### 注入攻击（special token injection）

若输入文本里恰好出现表面形式（如代码 `print("<|endoftext|>")`），被解析成特殊 token 即攻击者获得了注入控制符的能力（提示注入的一个具体通道）。

- Qwen 的默认行为**曾为安全**（表面形式按普通文本处理），后为迁就社区习惯**改为默认解析**——安全的默认被不安全的惯例打败
- 防护三档（[[tiktoken]] API）：
  - `allowed_special=set()`：所有表面形式按普通文本处理（防注入）
  - `allowed_special={...}`：白名单集合，其余按普通文本
  - `disallowed_special=(...)`：遇指定表面形式直接抛 `ValueError`

## 来源

- [[qwen-tokenization-note]]

## 相关

- [[qwen]]
- [[tiktoken]]
- [[bpe-tokenization]]
- [[vocabulary-expansion]]
