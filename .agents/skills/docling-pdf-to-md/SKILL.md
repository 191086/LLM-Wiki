---
name: docling-pdf-to-md
description: 用本机 docling 将 PDF（论文/文档）转换为 Markdown，图片导出为独立 PNG 文件按引用嵌入而非 base64 内嵌。收录（ingest）raw/ 中的 PDF、用户提到「转 markdown / 解析 PDF / 提取论文文本 / PDF 图片导出」、或 docling 时使用。
---

# docling PDF → Markdown

本机 docling 2.126 装在 uv 工具环境（解释器 `~/.local/share/uv/tools/docling/bin/python`，脚本 shebang 已指定）。直接调用现成脚本，不要现场重写转换代码：

```bash
.agents/skills/docling-pdf-to-md/scripts/convert_pdf.py <pdf路径> \
    [--out DIR] [--scale 2.0] [--obsidian-assets raw/assets]
```

## 参数与行为

- **输出**：`<out>/<stem>.md` + `<out>/<stem>_artifacts/*.png`。默认 `--out /tmp/docling/<stem>`——转换结果是 ingest 的**工作材料，不进 wiki/**；读它来写 `wiki/sources/` 页，图要嵌入 wiki 页面时才走 `--obsidian-assets`。
- **`--scale`**：图片清晰度，默认 2.0（≈144dpi）。图偏糊就升到 4.0（≈288dpi）。图片是按 bbox 从页面渲染图裁剪的，**不是** PDF 原始位图，只有提高 scale 一条路。
- **`--obsidian-assets <dir>`**：复制 PNG 到该目录（通常 `raw/assets`，vault 附件目录），并把 md 中引用改写为 `![[name.png]]`。PNG 按内容哈希命名，重跑不会重复拷贝。
- 一篇论文约 2 分钟；AI 模型已缓存于 `~/.cache/huggingface`，离线可用。

## 本 vault 的硬约束

- `raw/` 只读：PDF 原件不动；仅 `raw/assets/` 允许新增附件（Obsidian 附件目录本就落此处）。
- vault 在 iCloud 同步：运行前后文件可能被外部增删，拷贝/改写前先确认目标存在。
- **图片按需入 vault（2026-09-14 用户指示）**：只有被 wiki 页面实际嵌入（`![[...]]`）的图才拷入 `raw/assets`，从 artifacts 里手动挑选并起描述性文件名；**不要**用 `--obsidian-assets` 把全部 artifacts 拷进去——未被引用的图一律留在 /tmp 工作区。
- 图表多、版式复杂的论文优先考虑视觉直读（见 memory: vision-first-ingestion）；docling 适合以文字为主的 PDF。
