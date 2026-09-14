---
type: entity
title: Qwen（通义千问）
created: 2026-09-12
updated: 2026-09-14
tags:
  - llm
  - alibaba
sources: 2
---

# Qwen（通义千问）

**定义**：阿里巴巴（QwenLM）推出的开源大模型家族；本库从两个切面关注它——分词器工程实现，以及作为领域模型的微调基座。

## 详情

### 切面一：分词器（Qwen-7B 时代）

- 本库收录的官方文档以 Qwen-7B / Qwen-7B-Chat 为对象：分词器为基于 [[tiktoken]] 的 byte-level [[bpe-tokenization|BPE]]，词表 151,643 regular + 208 control tokens（[[qwen-tokenization-note]]）
- Chat 版引入 `<|im_start|>` / `<|im_end|>` 对话控制符（见 [[special-tokens]]）
- 官方称词表「已覆盖绝大多数中文词」；中文预切分无专门规则、汉字串整块保留，实验核验见 [[pretokenization-cjk-and-mixed-text]]（2026-09-12 比对 Qwen2.5 `tokenizer.json` 的真实正则，与 note 描述一致，仅数字改为逐位切分）

### 切面二：领域模型的基座（Qwen3-VL 时代）

- 多模态侧的 Qwen3-VL 系列被 [[ostrakon-vl]]（淘宝闪购的餐饮零售领域模型）选为基座：Qwen3-VL-8B 全参微调后，域内基准 ShopBench 从 55.3 升到 60.1，反超 Qwen3-VL-235B 的 59.4——同一论文同时给出了「通用大模型域内不敌 1/30 参数领域模型」的证据（[[ostrakon-vl-paper]]）
- 论文也记录了 Qwen3-VL-235B 在 FSRS 场景的典型失败（AI 生成视频误判为真实、厨房卫生属性误读），作为领域适配必要性的论据

## 来源

- [[qwen-tokenization-note]]
- [[ostrakon-vl-paper]]

## 相关

- [[tiktoken]]
- [[bpe-tokenization]]
- [[special-tokens]]
- [[ostrakon-vl]]
- [[domain-specific-mllm]]
