---
type: concept
title: 特殊 Token（Special / Control Tokens）
created: 2026-09-12
updated: 2026-09-15
tags:
  - tokenization
  - security
sources: 1
---

# 特殊 Token（Special / Control Tokens）

**定义**：不对应任何输入文本、只由系统在文本处理后才注入的功能 token（文档结束、对话轮次标记等）；其人类可读的表面形式（如 `<|endoftext|>`）仅为引用方便，与 regular token（`bytes` 型）分属词表中平行的两套类型系统。

## 1. 类型系统：regular 与 special 的平行双轨

|  | regular token | special（control）token |
| --- | --- | --- |
| 数据类型 | `bytes` | `str` |
| 来源 | BPE 从语料字节序列学出 | 开发者显式定义 |
| 产生途径 | 编码任意文本 | **只**由系统代码注入，编码普通文本永远不产生 |
| 与 merge 的关系 | 由 merge 链拼出 | 不经预切分、不参与任何 merge（[[bpe-tokenization]] §4.1 步骤 5） |

词表布局上 special 段固定排在 regular 段之后：Qwen 词表 151,643 个 regular 之后是 208 个 control（`<|endoftext|>` 即其中第一个，id 151,643），`<|extra_0|>`…`<|extra_204|>` 预留给用户扩展（起始 index 151,851）（[[qwen-tokenization-note]]）。

「编码普通文本永远不产生 special token」正是注入攻击的成因：文本里出现表面形式时，**解析与否**成为一个必须显式决定的安全选项（§3）。

## 2. 语义边界：含义是「训出来的」，不是名字给的

一个 token 对模型意味着什么，完全由**预训练时它的用法**决定。`<|endoftext|>` 之所以表示「文档结束」，是因为 Qwen 预训练语料用它分隔文档，模型学到了与之关联的行为（[[qwen-tokenization-note]]）。以此逐项看 Qwen 的边界：

- `<|endoftext|>`：文档结束。Chat 版另有 `<|im_start|>` / `<|im_end|>` 标记对话轮次——同理，它们在对话训练中承载了轮次语义。
- `bos` / `eos` / `unk` / `mask` / `sep` 对预训练模型**不适用**：模型没为这些符号学过任何含义，乱设可能引入未定义行为。名字里的联想（「beginning of sequence」）不构成语义，官方明确警告（[[qwen-tokenization-note]]）。
- ⚠️ `<|endoftext|>` ≠ `eos`：句末与文档末（可含多句）不是一回事，除非确认你的场景中二者重合。
- `pad` 是唯一被设计为**无语义**的例外：它只用来把变长序列补齐成矩形 batch，模型从不计算它，因此任意已知 token 都可充当，实践取 `<|endoftext|>`（[[qwen-tokenization-note]]）。

> 机制补注（wiki 外）：「从不计算」靠 attention mask 实现——补齐位置的 mask 置 0，注意力与损失都跳过这些位置，故 pad 的具体取值不影响输出。候选来源：Transformer 原论文 §3.2。

## 3. 注入攻击：表面形式的二义性

> 提示注入（prompt injection）：攻击者在输入文本中埋入内容、诱导模型执行非预期行为的攻击类别；special token injection 是它的一个具体通道——注入的不是文字指令，而是控制符本身。

若用户输入里恰好出现某 control token 的表面形式——例如代码片段 `print("<|endoftext|>")`——而分词器把它**解析**成特殊 token，攻击者就凭一段文本触发了本应只由系统代码触发的操作（[[qwen-tokenization-note]]）。

Qwen 分词器的默认行为经历过一次反转：旧默认把表面形式**按普通文本处理**（安全；特殊 token 由开发者在 tokenize 之后自行注入）；后为迁就社区使用习惯，**改为默认解析全部已知特殊 token**——安全的默认输给了不安全的惯例（[[qwen-tokenization-note]]）。新默认等价于 `allowed_special="all", disallowed_special=()`（§4）。

生态里还存在第三种立场（本机 tiktoken 实测，cl100k 词表）：tiktoken **本体**的 `encode` 默认 `allowed_special=set(), disallowed_special="all"`——遇任何特殊 token 表面形式**直接抛错**，强迫开发者显式表态。也就是说「默认怎么处理表面形式」没有行业共识：抛错（tiktoken 本体）、普通文本（Qwen 旧默认）、全部解析（Qwen 新默认）三种都存在于生产代码中，跨库迁移时必须显式传参而非依赖默认。

## 4. 防护：两个开关、四种行为

语义上只有两个独立开关（[[tiktoken]] API，Qwen 封装透传）：

- `allowed_special`：哪些表面形式**解析**为特殊 token；
- `disallowed_special`：哪些表面形式遇着**抛错**；
- 两个集合都不含的表面形式 → 按普通文本编码。

对输入 `print("<|endoftext|>")` 的四种行为（Qwen 词表的 id 出自 [[qwen-tokenization-note]]，cl100k 的 id 为本机 tiktoken 实测）：

| 调用 | 行为 | 输出 |
| --- | --- | --- |
| `allowed_special=set(), disallowed_special=()` | 全部按普通文本（**防注入首选**） | 8 个普通 token（Qwen：`[1350, 9639, 91, 8691, 723, 427, 91, 82598]`；cl100k 实测 roundtrip 无损） |
| `allowed_special={"<|endoftext|>"}`（Qwen 封装，disallowed 默认空） | 白名单内解析为特殊 token，其余普通 | Qwen 官方示例：`…<|endoftext|>` 混排文本中 `<|endoftext|>` → 151643，`<|extra_0|>` 得普通 id |
| `disallowed_special=("<|endoftext|>",)` | 遇指定表面形式抛 `ValueError` | `ValueError: Encountered text corresponding to disallowed special token ...` |
| `allowed_special="all", disallowed_special=()`（Qwen 新默认） | 全部解析 | Qwen：`[1350, 445, 151643, 899]`，4 个 token |

注意默认值随封装而变：Qwen 封装的 `disallowed_special` 默认为空（所以只传 `allowed_special=set()` 即得「全按普通文本」），tiktoken 本体默认 `disallowed_special="all"`（不显式传 `disallowed_special=()` 会变成抛错）。可移植的写法是两个参数都显式传。

开发者侧的正确姿势：编码用户输入时按第一行配置；特殊 token 一律在 tokenize **之后**由代码显式拼接 id——这也是 Qwen 旧默认（「表面形式按普通文本、注入交给开发者」）的设计意图。

## 来源

- [[qwen-tokenization-note]]

## 相关

- [[qwen]] ｜ [[tiktoken]]
- [[bpe-tokenization]]（§4.1 步骤 5：特殊 token 入表的位置）
- [[vocabulary-expansion]]（扩展 token 接在 special 段预留位之后）
