---
type: entity
title: TinyStories
created: 2026-09-12
updated: 2026-09-15
tags:
  - dataset
sources: 1
---

# TinyStories

**定义**：[HuggingFace 数据集](https://huggingface.co/datasets/roneneldan/TinyStories)，由词汇简单的儿童故事组成的合成语料。

## 背景：它为什么存在

> wiki 外补注（候选来源：Eldan & Li 2023《TinyStories: How Small Can Language Models Be and Still Speak Coherent English?》，arXiv:2305.07759）：微软研究院用 GPT-3.5 / GPT-4 按受控提示词批量生成短篇儿童故事——只用 3–4 岁幼儿能懂的词汇、简单叙事结构。论文的论点：在这个语料上训练，**总参数低于 1000 万**（小至约 1M）的模型也能生成连贯、多样、语法正确的英语故事，而常规 LLM 需要十亿级参数——幼儿级词汇与简单情节大幅削去了「世界知识 + 推理」的负担。评估也另辟蹊径：用 GPT-4 给生成故事按语法 / 创造性 / 一致性 / 连贯性打分，绕开标准基准对弱模型不适用的问题。

## 在本库的用法

- 特点：小到可以快速迭代，又大到能暴露性能瓶颈——适合作为算法基准语料
- [[building-a-fast-bpe-tokenizer-from-scratch]] 中用作 [[bpe-tokenization|BPE]] 训练基准：验证集切片 1–21.5MB（唯一词 W ≈ 4.6K–13.1K），全量约 500MB 时朴素实现需数小时
- 测试机：Apple M3 Max / 36GB / Python 3.12 / 4 workers

## 来源

- [[building-a-fast-bpe-tokenizer-from-scratch]]

## 相关

- [[bpe-tokenization]] ｜ [[stanford-cs336]]
