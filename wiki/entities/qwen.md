---
type: entity
title: Qwen（通义千问）
created: 2026-09-12
updated: 2026-09-15
tags:
  - llm
  - alibaba
sources: 3
---

# Qwen（通义千问）

**定义**：阿里巴巴（QwenLM）推出的开源大模型家族；本库从三个切面关注它——分词器工程实现、领域模型的微调基座、多模态后训练的实验对象。

## 家族谱系（背景）

> wiki 外补注（候选来源：Qwen 官方 GitHub QwenLM/Qwen 与官方博客）：Qwen 家族自 2023 年起开源迭代。文本主线五代：Qwen（2023）→ Qwen1.5（2024 初）→ Qwen2 → Qwen2.5（2024）→ Qwen3（2025），多规模、含 MoE 变体；多模态支线：Qwen-VL → Qwen2-VL → Qwen2.5-VL（2025-01）→ Qwen3-VL。本库涉及的型号横跨三代——Qwen-7B（分词器文档对象）、Qwen2.5-VL-7B/72B（RM 基座与 RL 靶子）、Qwen3-VL-8B/235B（领域模型基座与被反超的对照），正好构成「同一家族不同代际被本库三份来源各自取用」的样本。

## 详情

### 切面一：分词器（Qwen-7B 时代）

- 本库收录的官方文档以 Qwen-7B / Qwen-7B-Chat 为对象：分词器为基于 [[tiktoken]] 的 byte-level [[bpe-tokenization|BPE]]，词表 151,643 regular + 208 control tokens（[[qwen-tokenization-note]]）
- Chat 版引入 `<|im_start|>` / `<|im_end|>` 对话控制符（见 [[special-tokens]]）
- 官方称词表「已覆盖绝大多数中文词」；中文预切分无专门规则、汉字串整块保留，实验核验见 [[pretokenization-cjk-and-mixed-text]]（2026-09-12 比对 Qwen2.5 `tokenizer.json` 的真实正则，与 note 描述一致，仅数字改为逐位切分）

### 切面二：领域模型的基座（Qwen3-VL 时代）

- 多模态侧的 Qwen3-VL 系列被 [[ostrakon-vl]]（淘宝闪购的餐饮零售领域模型）选为基座：Qwen3-VL-8B 全参微调后，域内基准 ShopBench 从 55.3 升到 60.1，反超 Qwen3-VL-235B 的 59.4——同一论文同时给出了「通用大模型域内不敌 1/30 参数领域模型」的证据（[[ostrakon-vl-paper]]）
- 论文也记录了 Qwen3-VL-235B 在 FSRS 场景的典型失败（AI 生成视频误判为真实、厨房卫生属性误读），作为领域适配必要性的论据

### 切面三：Qwen2.5-VL——RM 的基座，也是 RL 的靶子（2026-09-15 新增）

- 同一个 Qwen2.5-VL-7B 在本库两份来源里扮演相反角色：[[skywork-vl-reward]] 拿它当**打分器基座**，[[vision-r1]] 拿它当**被 RL 强化的策略**（[[vision-r1-paper]]）
- 其定位能力是通用底座的短板：COCO 检测 mAP 仅 17.7，远低于定位特化的 Griffon-G-7B（40.2）；[[vision-r1]] 用规则奖励 GRPO 把它拉到 26.6（≈+50%）、ODINW-13 37.0→46.0，反超自家大 10 倍的 Qwen2.5-VL-72B（43.1）（[[vision-r1-paper]] §4.2）
- 「感知宽而不精」的又一证据：动态分辨率、任务全面，dense detection 上输给固定高分辨率 + 定位特训的专家路线（[[vision-r1-paper]] §2.1、§4.1–4.2）

## 来源

- [[qwen-tokenization-note]]
- [[ostrakon-vl-paper]]
- [[vision-r1-paper]]（Qwen2.5-VL-7B 作为 RL 实验对象）

## 相关

- [[tiktoken]]
- [[bpe-tokenization]]
- [[special-tokens]]
- [[ostrakon-vl]]
- [[domain-specific-mllm]]
- [[vision-r1]] ｜ [[skywork-vl-reward]]
