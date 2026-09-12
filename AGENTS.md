# AGENTS.md — LLM-Wiki 规范（Schema）

> 本文件是这个知识库的「规范层」：告诉 LLM 本 wiki 如何组织、遵守什么约定、执行哪些工作流。
> 由用户和 LLM 在使用中共同演进——发现更好的做法就更新这里。
> 模式来源：Karpathy《LLM Wiki》 <https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f>

## 核心理念

传统 RAG 每次查询都从原始文档重新检索、重新拼装答案，知识不积累。本知识库相反：LLM **增量地构建和维护一个持久的 wiki**——收录新来源时读取、提炼、整合进现有页面，更新实体页、修订综述、标注新旧矛盾。知识「编译一次、持续保鲜」，而不是每次重新推导。

**分工：用户负责选材、提问、把握方向；LLM 负责摘要、交叉引用、归档、一致性维护等全部簿记工作。**

## 三层架构

| 层 | 位置 | 谁写 | 说明 |
|---|---|---|---|
| 原始来源 | `raw/` | 仅用户 | 不可变，LLM 只读。事实的源头 |
| 知识库 | `wiki/` | 仅 LLM | 全部由 LLM 生成与维护，用户只读浏览 |
| 规范 | `AGENTS.md` | 用户 + LLM | 本文件 |

**硬性规则：`raw/` 中的文件一律不得修改或删除。**

## 目录结构

```
LLM-Wiki/
├── AGENTS.md            # 本文件（规范层）
├── raw/                 # 原始来源（用户放入，LLM 只读）
│   └── assets/          #   图片与附件（Obsidian 附件目录）
└── wiki/                # LLM 生成与维护的知识库
    ├── index.md         #   内容目录（按类别编目）
    ├── log.md           #   操作日志（append-only）
    ├── overview.md      #   全局综合页
    ├── sources/         #   来源摘要页（每个来源一页）
    ├── entities/        #   实体页（人物 / 组织 / 产品 / 项目…）
    ├── concepts/        #   概念页（理论 / 方法 / 模式…）
    └── analyses/        #   分析页（对比、综合，来自 query 的沉淀）
```

## 约定

- **语言**：正文默认中文；文件名用英文 kebab-case（如 `attention-is-all-you-need.md`）；中文标题写在 frontmatter `title` 字段（Obsidian 可装 Front Matter Title 插件，让图谱节点显示中文）。
- **链接**：一律用 Obsidian `[[wikilink]]`。建立交叉引用是 LLM 的核心职责，每个新页面至少与已有页面相连，避免孤儿页。
- **引用**：事实性内容标注来源，链到对应的 `[[sources/…]]` 页（来源页再指向 `raw/` 原文件）。
- **矛盾处理**：新来源与旧内容冲突时，不悄悄删掉旧说法——保留双方并显式标注矛盾及各自来源，由用户裁决。
- **frontmatter**（所有页面）：
  ```yaml
  ---
  type: source | entity | concept | analysis | index | log | overview
  title: 中文标题
  created: YYYY-MM-DD
  updated: YYYY-MM-DD
  tags: []
  sources: 0    # 支撑本页的来源数（source 页固定为 1）
  ---
  ```
- **日志格式**：新条目追加到 `wiki/log.md` 末尾，首行固定为 `## [YYYY-MM-DD] <ingest|query|lint|init> | <标题>`，保证 `grep "^## \[" wiki/log.md | tail -5` 可解析。

## 页面模板

### 来源页 `wiki/sources/<slug>.md`

```markdown
---
type: source
title: <来源标题>
created: …
updated: …
tags: []
sources: 1
raw: raw/<原文件名>
---

# <标题>

> 原文：[[raw/<原文件名>]] ｜ 类型：论文 / 文章 / 播客 / 视频… ｜ 日期：

## 一句话总结

## 关键要点

- …

## 值得追踪的实体与概念

- [[…]]
```

### 实体 / 概念页 `wiki/entities|concepts/<slug>.md`

```markdown
---
type: entity    # 或 concept
title: <中文名>
created: …
updated: …
tags: []
sources: N
---

# <名称>

**定义**：一句话。

## 详情

（综合自各来源的内容，关键论断按来源标注）

## 来源

- [[sources/…]]

## 相关

- [[…]]
```

### 分析页 `wiki/analyses/<slug>.md`

结构自由（对比表、综述、时间线均可），但同样需要 frontmatter、来源引用和相关页面链接。

## 工作流

### Ingest（收录）

1. 用户把来源文件放入 `raw/`（网页可用 Obsidian Web Clipper 存成 markdown，图片自动落 `raw/assets/`）。
2. LLM 通读来源，先与用户讨论关键收获和拟创建/更新的页面清单。
3. 写 `wiki/sources/<slug>.md` 摘要页。
4. 逐个创建或更新相关的 entities / concepts 页面，补充双向交叉引用。
5. 更新 `wiki/overview.md` 的综合理解。
6. 更新 `wiki/index.md` 目录。
7. 在 `wiki/log.md` 末尾追加 ingest 条目。

一次收录通常触及 5–15 个页面。默认逐份收录、保持人在环上；用户明确要求时才批量。

### Query（查询）

1. 先读 `wiki/index.md` 定位相关页面，再读页面本身。
2. 综合回答，所有关键论断带 `[[sources/…]]` 引用。
3. 回答若有沉淀价值（对比、分析、新发现），存为 `wiki/analyses/<slug>.md`，并更新 index 与 log。

回答形式可按需变化：markdown 页、对比表格、Marp 幻灯、图表等。

### Lint（体检，定期执行）

检查并修复：

- 页面之间的矛盾
- 被新来源推翻或已过时的论断
- 孤儿页面（无入链）
- 被多处提到但没有独立页面的概念
- 缺失的交叉引用
- 数据空白（值得用 web 搜索补齐的）

产出体检报告，并建议新的问题与新的候选来源。完成后在 log 记录。

## index.md 与 log.md

- **index.md**：全库目录，按类别列出每个页面 + 一行摘要。每次 ingest / 沉淀分析后更新。查询的入口，先读它。
- **log.md**：append-only 时间线，天然记录了这个 wiki 从想法到现状的演化路径。

## 用户偏好（随使用演进）

（初始为空。用户表达操作偏好时记录于此，例如「回答一律标注来源」「批量收录时不逐条确认」。）

## Obsidian 设置（一次性）

- ✅ Attachment folder path 已设为 `raw/assets`（2026-09-12）。
- 建议给命令「Download attachments for current file」绑快捷键，方便把网页图片本地化到 assets。
- 推荐插件：Front Matter Title（中文标题）、Dataview（按 frontmatter 建动态目录）、Web Clipper。
- 用图谱视图（Graph View）检查结构：孤立即缺上下文，密集团簇即核心主题。

## 扩展（按需）

- 规模变大、wiki 检索变慢时，可引入 [qmd](https://github.com/willccbb/qmd) 做本地 markdown 混合检索（BM25+向量）。
- 需要演示时可用 Marp 把任意分析页转成幻灯片。
