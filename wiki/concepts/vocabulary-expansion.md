---
type: concept
title: 词表扩展（Vocabulary Expansion）
created: 2026-09-12
updated: 2026-09-15
tags:
  - tokenization
  - bpe
  - fine-tuning
sources: 1
---

# 词表扩展（Vocabulary Expansion）

**定义**：在已训好的 [[bpe-tokenization|BPE]] 词表上追加新 token 的工程流程——不能直接把词塞进词表，因为编码走的是**中间 merge 链**而非词典查找；正确姿势是提供「词 + 频次」让分词器**续跑 BPE 学出新 merge**，再微调模型。

## 1. 为什么不能直接把词塞进词表

编码一个字符串 = 把它的字节序列按既有 merge 的优先级逐步合并（[[bpe-tokenization]] §5）。一个字符串想成为**单个** token，必须有 merge 链把它从字节一路拼出来；直接把 `是一只猫` 的整串字节写进 vocab 而没有链，编码器**永远轮不到查它**——它只沿着既有链走，产出一串旧 token。

模型侧还有第二道坎：即使链补齐、分词器能输出新 id，这个 id 的 embedding 对预训练模型是未训练的随机向量，必须微调才有意义（[[qwen-tokenization-note]]）。

## 2. 端到端走查：给 Qwen 加「我是一只猫」族词

以下全程取自 [[qwen-tokenization-note]] 的官方示例（词频文件、工具警告、调试日志、产物文件均为原文；字节与 base64 换算为本机核算）。

### 2.1 第 0 步：预切分资格检查

准备词频文件 `qwen_extra_vocab.txt`（词 `<TAB>` 频次）：

```
我是一只猫    20
你是一只猫    10
他是一只猫    5
一只    200
一只猫    100
夸张的 比喻手法    20
```

`add_merges.py` 先做两道过滤，实际日志：

- `夸张的 比喻手法` 含空格，预切分成 `['夸张的', ' 比喻手法']` 两块（走查见 [[pretokenization]] §3.4）→ `WARNING - ... cannot be added to vocabulary`。**跨预切分边界的词物理上加不进去**；
- `一只` 已是既有 token（`b'\xe4\xb8\x80\xe5\x8f\xaa'`）→ `skipping`，重复添加无意义；
- 剩 4 个词进入扩展（`number of existing merges: 151643`）。

### 2.2 字节展开与既有 merge 压缩

每个词先转 UTF-8 字节，再套用全部 151,643 条既有 merge 压缩到最简符号序列（频次数据结构即「符号序列 → 频次」，同 [[bpe-tokenization]] §4.1 步骤 1）：

| 词（频次） | 压缩后的符号序列 | 说明 |
| --- | --- | --- |
| `一只猫`（100） | `一只｜猫` | `(一,只)→一只` 是既有 merge，`(只,猫)` 不是 |
| `我是一只猫`（20） | `我｜是一｜只｜猫` | **不是** `我｜是｜一只｜猫`：`(是,一)→是一` 优先级高于 `(一,只)→一只`，编码时先套 |
| `你是一只猫`（10） | `你｜是一｜只｜猫` | 同上 |
| `他是一只猫`（5） | `他｜是一｜只｜猫` | 同上 |

### 2.3 续跑 BPE：六条新 merge 的诞生

对压缩后的符号统计相邻 pair 频次、迭代合并最高频——与训练 BPE 完全同构（[[bpe-tokenization]] §4），只是起点从字节换成了「既有 merge 压缩后的符号」。官方日志逐轮还原：

| 轮 | 选中的 pair | 新 token | 频次 | 计数来源 |
| --- | --- | --- | --- | --- |
| 1 | `(一只, 猫)` | `一只猫` | 100 | 独立词 `一只猫`×100 |
| 2 | `(只, 猫)` | `只猫` | 35 | `我/你/他是一只猫` 中的 `只｜猫`（20+10+5）；独立词已在轮 1 合并、不再贡献 |
| 3 | `(是一, 只猫)` | `是一只猫` | 35 | 轮 2 之后 `我｜是一｜只猫` 中的相邻 pair |
| 4 | `(我, 是一只猫)` | `我是一只猫` | 20 | 轮 3 之后 |
| 5 | `(你, 是一只猫)` | `你是一只猫` | 10 | 同上 |
| 6 | `(他, 是一只猫)` | `他是一只猫` | 5 | 同上 |

两点值得停下来看：

- **轮 2 与轮 3 候选 pair 频次同为 35**，日志实际先选 `(只,猫)`。并列时的选择规则官方文档未交代，此处以日志为准。
- **拼装路径被既有优先级支配**：`是｜一｜只｜猫` 实际走 `是一｜只｜猫 → 是一｜只猫 → 是一只猫`，而不是经过 `一只`——因为 `(是,一)` 的 merge 学得更早、优先级更高。**你加的词不一定按你预期的方式拼出来**；顺带的，「`只猫` 这种语言学上莫名其妙的 token 也被学进词表」是纯分布 BPE 的正常副产品（[[qwen-tokenization-note]]，亦见 [[bpe-tokenization]] §7）。

### 2.4 产物与验证

`qwen_extra.tiktoken` 每行是 `base64(字节串) → 新 index`，从 151,851 起连续分配（base64 解码即得汉字，本机核算无误）：

| base64 行 | 解码 | index |
| --- | --- | --- |
| `5LiA5Y+q54yr` | `一只猫` | 151851 |
| `5Y+q54yr` | `只猫` | 151852 |
| `5piv5LiA5Y+q54yr` | `是一只猫` | 151853 |
| `5oiR5piv5LiA5Y+q54yr` | `我是一只猫` | 151854 |
| `5L2g5piv5LiA5Y+q54yr` | `你是一只猫` | 151855 |
| `5LuW5piv5LiA5Y+q54yr` | `他是一只猫` | 151856 |

加载验证（官方原文）：`AutoTokenizer.from_pretrained("Qwen/Qwen-7B", extra_vocab_file="qwen_extra.tiktoken")` 后 `len(tokenizer)` = 151,857（= 151,851 + 6），且

```python
>>> tokenizer("我是一只猫")
{'input_ids': [151854], ...}    # 单 token 达成
```

分词器侧至此完工；模型侧仍需微调（§1 第二道坎）。

## 3. 官方四步流程（操作清单）

1. 准备 `qwen_extra_vocab.txt`：每行 `词<TAB>频次`——**频次参与 BPE 计算**，直接决定哪些新 merge 被学到、以什么顺序学到（§2.3 的计数即来自它）；
2. 准备基础词表 `qwen.tiktoken`，确定新 token 起始 index（默认 151,851；也可覆盖不活跃的 control token，但需改 tokenizer 代码，见 [[special-tokens]] §1 的布局）；
3. 跑 `add_merges.py qwen.tiktoken qwen_extra.tiktoken qwen_extra_vocab.txt` 学新 merge（纯 Python 实现，加大量词时慢）；
4. 加载时传 `extra_vocab_file`（需 2023-10-08 之后的 tokenizer 代码，否则手工把产物内容拼接到 `qwen.tiktoken` 末尾），**并微调模型**。

## 4. 三个坑

### 4.1 预切分约束

跨 [[pretokenization]] 边界的词加不进去——工具会警告但不报错中止，漏看日志就会以为加成功了（§2.1 已演示）。

### 4.2 码点边界风险

Qwen 分词器直接在 UTF-8 字节序列上操作（对比 SentencePiece 的码点级 + byte fallback 兜底）。有限频次数据上学出的新 merge 可能**跨 Unicode 码点**：如 `一只` 的字节 `b'\xe4\xb8\x80\xe5\x8f\xaa'` 中 `b'\x80\xe5'` 可能先合并——`一`（`e4 b8 80`）与 `只`（`e5 8f aa`）各被拦腰截断。对已知 token 无碍（反正最终能拼出整词），但**对未知词**会产生预训练模型从未见过的异常切分。稳妥做法：把涉及的所有码点也写进词频文件，且频次**高于**相关词频次之和，保证码点级 merge 优先学到（[[qwen-tokenization-note]]）。

### 4.3 拼装路径不受你控制

既有 merge 的优先级支配新词的拼装路径（§2.3 已演示），推论：想加的词不一定按你预期的方式拼出来。官方的务实判断：Qwen「已覆盖绝大多数中文词」，通常只加中文词即可（[[qwen-tokenization-note]]）。

## 来源

- [[qwen-tokenization-note]]

## 相关

- [[bpe-tokenization]]（§4 训练、§5 编码、§7 纯分布性质——本页全部机制的上游）
- [[pretokenization]]（§4.1 的边界约束来源）
- [[special-tokens]]（词表布局与扩展起始 index 的背景）
- [[qwen]]
