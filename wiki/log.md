---
type: log
title: 日志
created: 2026-09-12
updated: 2026-09-12
---

# 日志

> Append-only 操作日志，新条目追加到文件末尾。
> 条目格式：`## [YYYY-MM-DD] <ingest|query|lint|init> | <标题>`
> 查看最近条目：`grep "^## \[" wiki/log.md | tail -5`

## [2026-09-12] init | 初始化 LLM-Wiki

- 按 Karpathy 的 llm-wiki 模式建立三层架构：`raw/`（不可变来源）、`wiki/`（LLM 维护）、`AGENTS.md`（规范层）
- 创建 wiki 骨架：index / log / overview 与 sources、entities、concepts、analyses 四类目录
- 初始化 git 仓库
