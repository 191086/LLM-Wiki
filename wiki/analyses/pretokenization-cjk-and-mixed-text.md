---
type: analysis
title: 预切分如何处理中文与中英混合文本
created: 2026-09-12
updated: 2026-09-15
tags:
  - tokenization
  - cjk
  - pretokenization
sources: 2
---

# 预切分如何处理中文与中英混合文本

> 起因：query「pretokenization 怎么处理中文？中英混合呢？」。方法：本地实验——Python `regex` 模块跑真实正则；GPT-2 原文见 [[pretokenization]] §2，cl100k 取自本机 tiktoken `openai_public.py`（2026-09-15 复验），Qwen2.5 取自其 `tokenizer.json`（2026-09-12 核验）。本文是 [[pretokenization]] §5（中文与中英混合）的实证展开。

## 1. 一句话结论

**主流 tokenizer 对中文几乎「不设防」**：`\p{L}` 匹配的是 Unicode 字母类（不区分文字系统），连续汉字串被切成**一整块**，直到标点 / 空白 / 数字才断开。预切分**不做中文分词**，「词」的边界完全交给 [[bpe-tokenization|BPE]] 从频率统计里学。中英混合时若无空格，汉字与拉丁字母同属 `\p{L}`，会进**同一个 chunk**。

## 2. GPT-2 正则下的实际切分（2026-09-15 复验）

对中文起作用的分支是 ` ?\p{L}+`——`\p{L}` 覆盖汉字（`\p{Lo}`）、假名、谚文、拉丁字母等全部字母类别：

| 输入 | 切分结果 | 说明 |
| --- | --- | --- |
| `今天天气不错，我们去公园散步吧。` | `今天天气不错`｜`，`｜`我们去公园散步吧`｜`。` | 整句汉字一块，中文标点单独走标点分支 |
| `使用GPT-4训练tokenizer` | `使用GPT`｜`-`｜`4`｜`训练tokenizer` | **无空格时跨文字系统成一块**；字母/数字必断 |
| `使用 GPT-4 训练 tokenizer` | `使用`｜` GPT`｜`-`｜`4`｜` 训练`｜` tokenizer` | 有空格则空格前置，与英文规则一致 |
| `2024年发布了o1模型` | `2024`｜`年发布了o`｜`1`｜`模型` | `o1` 被字母/数字边界切开；`年发布了o` 是混合块 |
| `I love you我爱你123test` | `I`｜` love`｜` you我爱你`｜`123`｜`test` | ` you我爱你` 一块，随后数字、字母各自成块 |

同一正则在混合句上的逐块分支标注走查见 [[pretokenization]] §3。

## 3. 现代 tokenizer 对照

| tokenizer | 数字 | 标点与词 | CJK 专门规则 |
| --- | --- | --- | --- |
| GPT-2 | ` ？\p{N}+` 整串 | 标点独立成块 | **无** |
| cl100k（GPT-3.5/4） | `\p{N}{1,3}` 最多三位一组 | 单个标点可黏到后词前 | **无** |
| Qwen2.5 | `\p{N}` 逐位切 | 同 cl100k 式黏前缀 | **无** |

（cl100k 行 2026-09-15 本机复验：`今天天气不错，参数量约300B` → `今天天气不错`｜`，参数量约`｜`300`｜`B`。Qwen2.5 行沿用 09-12 核验：同句 → `今天天气不错`｜`，参数量约`｜`3`｜`0`｜`0`｜`B`。）

- 三者对中文的处理一致：`\p{L}+` 整块保留，**都没有**分词或按字切分
- [[qwen-tokenization-note|Qwen 官方 note]] 佐证：byte-level BPE、词表 151,643 regular、「has most of the Chinese words」，示例 `夸张的 比喻手法` → `['夸张的', ' 比喻手法']`；多字词如 `我是一只猫` 是正常 BPE merge 的产物（走查见 [[vocabulary-expansion]] §2）
- ⚠️ 网络检索曾出现「Qwen 把汉字按单字预切分」的说法，与 tokenizer.json 及官方 note 不符（单字切块将使多字 token 无法形成），已排除

## 4. 对 BPE 训练的影响：L 暴增（可复现实验）

[[building-a-fast-bpe-tokenizer-from-scratch]] 的复杂度记号里，L 是「词块字节长」，V5 总复杂度 O(W×L²·logP) 含 $L^2$ 项。预切分对英文的好性质（词对齐）依赖**空格分词**；中文没有空格，chunk 以标点为界，L 显著更大。实验用句对（内容对译，GPT-2 正则，2026-09-15 本机复算）：

- 中文：`今天天气不错，我们去公园散步吧。街角新开的咖啡店招牌很显眼，路过的人都会多看两眼。`
- English：`The weather is nice today, so let's take a walk in the park. The new coffee shop on the corner has an eye-catching sign that makes passersby look twice.`

| 语料 | chunk 数 | 总字节 | 平均 chunk 字节 | 最长 chunk |
| --- | --- | --- | --- | --- |
| 中文 | 8 | 123 | **15.4B** | **39B** |
| English | 35 | 152 | 4.3B | 10B |

两点读法：

- 中文平均 chunk 是英文的 **3.6 倍**（15.4 / 4.3），$L^2$ 项按 $(15.4/4.3)^2 \approx 12.8$ 倍放大（同 W、P 下）——英文基准语料（[[tinystories]]，英文儿童故事）会**低估**中文语料的 BPE 训练成本
- 总字节中文反而更少（123 vs 152）：问题不在信息量，在**切分粒度**——没有空格边界，初始 pair 全部挤在长 chunk 里

（2026-09-12 版本曾给「英文 21 chunk / 4.1B vs 中文 6 chunk / 14.5B」，未附实验句子、不可复现，已由上述显式句对实验替换。）

## 5. token 层面的后果（中文 token 税）

- 一个汉字 = 3 个 UTF-8 字节（`你` = `E4 BD A0`，[[building-a-fast-bpe-tokenizer-from-scratch]]）。英文中心的词表对常见汉字覆盖差，容易碎成多 token；同义内容中文 token 数通常多于英文
- 更有甚者，token 可以**截断一个字符**：Qwen 的 token 51461 = `b' \xe6\xa0'`，只含「根」三个字节中的两个，单独解码只得 `�`（[[qwen-tokenization-note]]）
- 中文 token 是 BPE 学出的**高频字组**（1–3 字为主），与语言学词边界不一定对齐。混合 chunk 内部 BPE 也**可以**学出跨文字系统的 merge（如 `使用G` + `PT`），取决于训练语料频率，通常不如纯块内组合高频

## 6. 为什么不做分词（解读，非来源原话）

> 本节为分析性解读。预切分正则被刻意设计得**语言无关、极简**：引入分词器（如 jieba）会带来重型依赖、特定分词标准的偏置与 OOV 问题，且训练/推理必须严格一致。把「什么是中文词」留给 BPE 频率统计是更稳健的工程选择——代价即 §4 的 chunk 变长与 §5 的词边界不对齐。

## 7. 候选来源（待收录）

- [Chinese vs. English Tokens across six LLMs](https://markhuang.ai/blog/chinese-token-myth) — 中文 token 效率实证
- [To Merge or Not to Merge: Chinese Tokenization Pitfalls](https://digitalorientalist.com/2025/02/04/to-merge-or-not-to-merge-the-pitfalls-of-chinese-tokenization-in-general-purpose-lls/) — 通用 LLM 对中文切分的坑

## 来源

- [[building-a-fast-bpe-tokenizer-from-scratch]]（GPT-2 正则、byte-level 背景、复杂度记号 W/L/P）
- [[qwen-tokenization-note]]（Qwen 官方说明：byte-level BPE、词表规模、CJK 串整块示例）
- 正则原文：tiktoken `openai_public.py`（2026-09-15 本机复验）、Qwen2.5 `tokenizer.json`（2026-09-12 核验）

## 相关

- [[pretokenization]]（机制页：正则分支拆解与走查；本文是其 §5 的实证展开）
- [[bpe-tokenization]]（§8：L 所在的复杂度阶梯）
- [[vocabulary-expansion]]（多字词经 merge 链成形的走查）
