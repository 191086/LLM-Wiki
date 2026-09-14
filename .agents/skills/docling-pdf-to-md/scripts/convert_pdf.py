#!/Users/wrj/.local/share/uv/tools/docling/bin/python
"""PDF → Markdown via docling，图片导出为独立 PNG（referenced），不内嵌 base64。

用法:
  convert_pdf.py <pdf> [--out DIR] [--scale 2.0] [--obsidian-assets DIR]

输出:
  DIR/<stem>.md            markdown，图片以路径引用
  DIR/<stem>_artifacts/    导出的 PNG（内容哈希命名）

选项:
  --out DIR             输出目录，默认 /tmp/docling/<stem>
  --scale N             图片清晰度：1.0≈72dpi(默认糊) 2.0≈144dpi 4.0≈288dpi
  --obsidian-assets DIR 复制 PNG 到 DIR（如 raw/assets）并把 md 引用
                        改写为 Obsidian ![[name.png]] 嵌入
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--obsidian-assets", type=Path, default=None)
    args = ap.parse_args()

    pdf = args.pdf.expanduser().resolve()
    if not pdf.exists():
        sys.exit(f"not found: {pdf}")
    out = (args.out or Path("/tmp/docling") / pdf.stem).resolve()
    out.mkdir(parents=True, exist_ok=True)

    po = PdfPipelineOptions()
    po.images_scale = args.scale
    po.generate_picture_images = True

    result = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=po)}
    ).convert(pdf)

    md_path = out / f"{pdf.stem}.md"
    result.document.save_as_markdown(
        filename=md_path, image_mode=ImageRefMode.REFERENCED
    )

    pngs = sorted((out / f"{pdf.stem}_artifacts").glob("*.png"))

    if args.obsidian_assets:
        dest = args.obsidian_assets.resolve()
        dest.mkdir(parents=True, exist_ok=True)
        for p in pngs:
            if not (dest / p.name).exists():  # 哈希命名，重跑不重复拷贝
                shutil.copy2(p, dest / p.name)
        md = md_path.read_text(encoding="utf-8")
        md = re.sub(r"!\[[^\]]*\]\([^)]*artifacts/([^)]+)\)", r"![[\1]]", md)
        md_path.write_text(md, encoding="utf-8")

    total_kb = sum(p.stat().st_size for p in pngs) // 1024
    print(f"md: {md_path} ({md_path.stat().st_size // 1024}KB)")
    print(f"images: {len(pngs)} png, {total_kb}KB total")
    if args.obsidian_assets:
        print(f"assets: copied to {args.obsidian_assets} (refs rewritten to ![[...]])")


if __name__ == "__main__":
    main()
