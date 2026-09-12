---
type: entity
title: TinyStories
created: 2026-09-12
updated: 2026-09-12
tags:
  - dataset
sources: 1
---

# TinyStories

**定义**：[HuggingFace 数据集](https://huggingface.co/datasets/roneneldan/TinyStories)，由词汇简单的儿童故事组成。

## 详情

- 特点：小到可以快速迭代，又大到能暴露性能瓶颈——适合作为算法基准语料
- 在 [[building-a-fast-bpe-tokenizer-from-scratch]] 中用作 [[bpe-tokenization|BPE]] 训练基准：验证集切片 1–21.5MB（唯一词 W ≈ 4.6K–13.1K），全量约 500MB 时朴素实现需数小时
- 测试机：Apple M3 Max / 36GB / Python 3.12 / 4 workers

## 来源

- [[building-a-fast-bpe-tokenizer-from-scratch]]

## 相关

- [[bpe-tokenization]]
