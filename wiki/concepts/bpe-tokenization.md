---
type: concept
title: BPE（Byte-Pair Encoding，字节对编码）
created: 2026-09-12
updated: 2026-09-14
tags:
  - tokenization
  - algorithm
sources: 2
---

# BPE（字节对编码）

**定义**：一种子词分词算法——从 256 个字节值构成的基础词表出发，迭代合并语料中频次最高的相邻符号对，直至词表达到目标规模。训练产物是一张**词表**加一条**有序 merge 规则链**，编码时按 merge 先后顺序（优先级）套用到输入上。当前几乎所有 LLM（GPT / Llama / Qwen…）的标准分词方法。

## 1. 要解决什么问题：词表的三难

| 方案          | 词表           | 问题                                               |
| ----------- | ------------ | ------------------------------------------------ |
| word-level  | 每个词一个 token  | 词形爆炸（英语百万级词），无法表示未登录词（UNK），无形态泛化（cat / cats 两码事） |
| char-level  | 每个字符一个 token | 无 OOV，但序列过长、单 token 语义太碎，浪费上下文窗口                 |
| **subword** | 高频串合并出的子词单元  | 高频词保持完整、低频词拆成有意义的部件；**词表规模可控 + 开放词表**，两头兼顾       |

BPE 是 subword 路线的贪心代表：不引入任何语言学规则，纯靠语料频率统计「长出」子词单元。

## 2. 历史脉络

- **1994**：Philip Gage 提出，作为通用**数据压缩**算法——「用新符号替换语料中最常见的字节对」。
- **2016**：Sennrich et al.（ACL）引入神经机器翻译解决 OOV；在字符序列上跑 BPE，用 `</w>` 词尾标记表达词边界。
- **GPT-2 以降**：改为 **byte-level BPE**（基础词表 = 256 字节）+ **预切分正则**（词边界交给正则而非 `</w>`），成为 LLM 事实标准。Qwen-7B 分词器即基于 [[tiktoken]] 的 byte-level BPE（[[qwen-tokenization-note]]）。

## 3. 算法总览：训练与编码两个阶段

BPE 严格分为两件事，初学最易混淆：

| 阶段         | 输入              | 输出                  | 核心操作                |
| ---------- | --------------- | ------------------- | ------------------- |
| **训练**（学习） | 语料 + 目标词表规模 $V$ | 词表 + **有序** merge 链 | 反复合并当前最高频 pair      |
| **编码**（推理） | 任意文本            | token id 序列         | 按训练时学到的 merge 优先级套用 |

训练学到的 merge 链**顺序即优先级**：先学的 merge 意味着更常见的组合，编码时优先套用。两条链路用同一套预切分正则（[[pretokenization]]）——这是硬约束，**跨预切分边界的字符串永远不可能成为一个 token**。

## 4. 训练算法

### 4.1 步骤

1. **预切分**：用正则把语料切成词块，统计各词块频次，得到「词块 → 频次」计数表（去重后按频次加权处理，不必真的重复存词）。
2. **统计相邻符号对频次**：对每个词块枚举相邻 pair，按词块频次加权求和。初始符号即 256 个字节值。
3. **合并最高频对** $(A, B) \to AB$：新符号写入词表，merge 记入链尾；并列时取字节串字典序较大者（CS336 作业实现的约定，[[building-a-fast-bpe-tokenizer-from-scratch]]）。
4. **重复** 2–3 直至执行 $m$ 次 merge：

$$m = V - 256 - \text{特殊 token 数}$$

5. **特殊 token 入表**：循环结束后追加到词表末尾（占据最高 id）。它们不经预切分、不参与任何 merge，是与 regular token 平行的另一套类型系统（[[special-tokens]]）。Qwen 词表即此布局：151,643 个 regular 之后才是 208 个 control token（[[qwen-tokenization-note]]）。

### 4.2 伪代码

```python
def train(corpus: str, vocab_size: int, special_tokens: list[str]):
    words = Counter(pretokenize(corpus))          # 词块 → 频次
    vocab  = {i: bytes([i]) for i in range(256)}  # 基础词表：256 字节
    merges: list[tuple[bytes, bytes]] = []

    for _ in range(vocab_size - 256 - len(special_tokens)):
        pairs = defaultdict(int)
        for word, freq in words.items():          # 加权统计（词内相邻 pair）
            for pair in zip(word, word[1:]):
                pairs[pair] += freq
        best = max(pairs, key=lambda p: (pairs[p], p))  # 频次并列 → 字典序较大
        merges.append(best)
        words = {replace(word, best): f for word, f in words.items()}
        vocab[len(vocab)] = best[0] + best[1]     # merge token 紧随字节之后

    for s in special_tokens:                      # 特殊 token 补占最高 id
        vocab[len(vocab)] = s.encode("utf-8")     # 不参与 merge（步骤 5）

    return vocab, merges
```

### 4.3 手工示例

语料预切分后四个词块：`low`×5、`lowest`×2、`newest`×6、`widest`×3（ASCII，每字母即一字节 token）。

| 轮 | 最高频 pair（加权计数） | 合并 | 词块变化 |
| --- | --- | --- | --- |
| 1 | `(e,s)`=11 与 `(s,t)`=11 **并列** → 字典序取 `st` | `st` | newest→`n e w e st`，widest→`w i d e st`，lowest→`l o w e st` |
| 2 | `(e,st)`=11 | `est` | `n e w est`、`w i d est`、`l o w est` |
| 3 | `(w,est)`=8 | `west` | `n e west`、`l o west`、`w i d est` |

几轮之后 `est`（跨词后缀）、`west` 这类单元自动浮现——**没有任何形态学规则，纯频率使然**。继续跑会依次合并出 `lo`→`low` 等。

## 5. 编码与解码（推理）算法

对输入文本：过同一预切分正则 → 每个词块转 UTF-8 字节序列 → **反复合并当前优先级最高（训练中最早学到）的 pair**，直到词块内不存在任何在 merge 链中的 pair。

```python
def encode(text: str, merges: list, ranks: dict) -> list[int]:
    ids = []
    for word in pretokenize(text):
        sym = list(word.encode("utf-8"))
        while len(sym) >= 2:
            pair = min(zip(sym, sym[1:]), key=lambda p: ranks.get(p, float("inf")))
            if pair not in ranks:                # 词内已无可套用的 merge
                break
            sym = replace(sym, pair)
        ids += [vocab[b] for b in sym]
    return ids
```

解码是编码的逆过程，但**不需要逆推任何 merge**——一次 merge $(A,B)\to AB$ 只是把相邻符号替换成二者的字节拼接，不改变字节的内容与顺序，所以任意时刻符号序列的字节拼接恒等于原词块的字节序列。于是解码只剩两步：每个 token id 查回字节串，全序列拼接后**整体**做一次 UTF-8 解码：

```python
def decode(ids: list[int], vocab: dict[int, bytes]) -> str:   # vocab 取 id → 字节方向
    return b"".join(vocab[i] for i in ids).decode("utf-8")
```

一个错误处理细节：若 `ids` 来自流式输出的**中间截断**，字节流末尾可能是半个多字节字符（token 可以截断码点，见 §6.3），严格解码会抛 `UnicodeDecodeError`。生产实现（[[tiktoken]] 宿主下的 Qwen 分词器即如此）用 `errors="ignore"` 丢弃不完整尾巴，或 `errors="replace"` 显式渲染为 `�`（[[qwen-tokenization-note]]）。

「每轮取 rank 最小的 pair」用优先级堆实现为 $O(L \log L)$（$L$ 为词块字节数）；朴素地每轮全扫则是 $O(L^2)$。编码是**确定的**：同一文本在同一版 merge 链下永远得到同一 id 序列。解码是线性的 $O(N)$（$N$ 为总字节数），byte-level 下 round-trip 无损——`decode(encode(text))` 精确还原原始字节。

## 6. byte-level 变体

### 6.1 基础词表 = 256 字节 → 无 UNK

任意 Unicode 文本都能编码成字节序列再切分，**天然开放词表**。代价是多字节字符初始被拆散，靠训练中的 merge 学着合拢。

### 6.2 多字节字符：先拆后合

「你」（U+4F60）UTF-8 编码为 `E4 BD A0`——训练初期是 3 个 token，出现频次够高后 merge 成整 token（[[qwen-tokenization-note]]）。高频汉字最终多为单 token；低频字符可能永远停留在字节碎片状态。

### 6.3 token 可以截断一个字符

BPE 不知道码点边界，merge 可以落在字符中间：Qwen 词表中 token 51461 = `b' \xe6\xa0'`，只含「根」（`E6 A0 B9`）三个字节中的前两个。单独解码它只得替换符 `�`（U+FFFD），与后继 token 拼接解码才是完整字符；对流式输出可用 `errors="ignore"` 吞掉不完整尾巴（[[qwen-tokenization-note]]）。

### 6.4 空格前置与词边界

GPT 系预切分惯例把空格附到下一个词前（` the` 是一个 token）——所以「行首的 the」和「句中的 the」切分不同（见 §7）。词边界完全由预切分正则保证，取代了 Sennrich 的 `</w>` 标记。

## 7. 核心性质：纯分布、无语言学知识

BPE 只看频率分布，不知道哪些字节序列构成合法码点、字符或词（[[qwen-tokenization-note]]）。三条直接后果：

1. **同一词在不同上下文切分不同**。`"Panda"` → `P|anda`，`" Panda"` → 整块，`" Pandas"` → ` Pand|as`——三者的 merge 路径只是各自恰好在训练数据中更常见，不存在「词典查到了」这回事。
2. **合并路径被既有 token 优先级支配**。`是|一|只|猫` 的实际合并顺序是 `是一|只|猫 → 是一|只猫 → 是一只猫`——因为 `是一` 已是 token 且优先级更高。推论：想给词表加新词**不能直接塞**，必须补出整条中间 merge 链并微调模型（详见 [[vocabulary-expansion]]）。
3. **merge 可能跨 Unicode 码点边界**。有限数据上学出的 `b'\x80\xe5'` 这类 token 对未知词产生异常切分；稳妥做法是把涉及的码点以更高频补进语料重训（[[qwen-tokenization-note]]）。

训练侧的对应事实：merge 决策只依赖「此刻全库 pair 频次」，贪心且不可回退——早期的局部最优选择锁定后续全部路径。

## 8. 训练复杂度与工程优化

（本节全部出自 [[building-a-fast-bpe-tokenizer-from-scratch]]：[[jun-yu-tan]] 基于 [[stanford-cs336]] Assignment 1 的实测，21MB [[tinystories]] 语料、词表 5000。）

### 8.1 朴素实现及其低效

记 $W$ = 去重词块种数，$L$ = 平均词块字节数，$m$ = merge 次数，$P$ = 全库不同 pair 种数，$A$ = 一次 merge 受影响的词块数。

朴素实现每轮 merge 都**从头重算全部 pair 频次** + **扫描全部 $W$ 个词块找受影响者**，单轮 $O(W \times L)$，总计：

$$O(m \times W \times L)$$

实测 1341s——线性于 $m$，词表越大越不可用。

### 8.2 关键观察：merge 的局部性

一次 merge $(A,B) \to AB$ **只影响含该 pair 的词块**，其余 pair 频次一概不变。这是后续所有优化的出发点。

### 8.3 五级优化阶梯

| 版本  | 优化手段                           | 单次 merge 复杂度                 | 总复杂度                     |
| --- | ------------------------------ | ---------------------------- | ------------------------ |
| V1  | 朴素实现                           | $O(W \times L)$              | $O(m \times W \times L)$ |
| V2  | + 增量更新 pair 频次                 | $O(W \times L)$（常数改善 5–15×）  | 同 V1                     |
| V3  | + 并行预切分（按文档切块，multiprocessing） | 同 V2                         | 预切分 $O(n/p)$             |
| V4  | + 倒排索引 pair → 词块集合             | $O(P + A \times L)$          | $\le O(W \times L^2)$    |
| V5  | + 最大堆 + 惰性删除 + 堆压缩             | $O(A \times L \cdot \log P)$ | $O(W \times L^2 \log P)$ |

- **V2**：merge 后只更新受影响词块的 pair 计数。渐近阶不变但常数大降——同阶复杂度不代表同速。
- **V4 的漂亮上界**：每次 merge 使词块至少缩短 1，初始长 $L_0$ 的词块至多被更新 $L_0 - 1$ 次，故总更新量 $\sum A \le W \times L$，**与 $m$ 无关**。
- **V5**：pair 频次入最大堆，直接取 max；用 `ReversedBytes` 反转字节比较把 max-heap 语义塞进 Python `heapq` 的 min-heap。惰性删除留下陈旧堆项，靠「堆大小 > 3× 有效 pair 数」阈值触发重建压缩——不压缩时内存膨胀 2×、拖垮缓存性能，实测反被 V4 追平。

### 8.4 实测与经验教训

- **总计 1341s → 5.8s（约 230×）**；V4/V5 交叉点在词表 ≈1300：更低时 V4 略胜（堆常数未摊销），更高时 V5 一致胜出。生产词表 32K–100K，堆是标配。
- 小词表（<2000）主要收益来自倒排索引；大语料则并行预切分恒有收益（500MB、8 核：~30s → ~5s）。
- 三条教训：**先 profile 再优化**（预想堆总赢，实测小词表反而亏，基准测试才暴露交叉点）；**内存即性能**（2× 膨胀曾让 V5 慢于 V4）；**同阶复杂度不代表同速**（V2 与 V1 同为 $O(m \times W \times L)$，实测快 5–15×）。

## 9. 变体对照：WordPiece / Unigram / SentencePiece

> 本节为 wiki 外一般性知识，尚无对应来源页；候选来源 SentencePiece 官方文档已列于 [[qwen-tokenization-note]]。

| 算法 | 代表 | 与 BPE 的差异 |
| --- | --- | --- |
| WordPiece | BERT | 同为贪心合并，但准则从「频次最高」改为**似然增益最大**（近似取 $\frac{\text{freq}(AB)}{\text{freq}(A)\,\text{freq}(B)}$ 最高者） |
| Unigram | T5、Llama 系（经 SentencePiece） | 方向相反：从大词表出发，迭代**删去**对语料似然损害最小的 token；切分是概率式的（可采样），非确定性贪心 |
| SentencePiece | 同上 | 实现层：直接以码点为单元 + byte fallback 兜底未知字符，空格显式化为 `▁`，与 byte-level BPE 是两条对照路线 |

## 10. 实践要点速查

- 预切分正则决定 BPE 的作用域边界：跨 chunk 的字符串**永远**成不了 token；中文/混合文本行为见 [[pretokenization]] 与 [[pretokenization-cjk-and-mixed-text]]。
- 词表规模主流 32K–100K；训练耗时对 $m$（即词表大小）敏感，工程实现见 §8。
- 词表扩展必须补中间 merge 链 + 微调，且受预切分边界约束 → [[vocabulary-expansion]]。
- 特殊 token 是词表里与 regular token 平行的另一套类型系统，有注入风险 → [[special-tokens]]。
- 参考实现：[[tiktoken]]（Qwen 分词器宿主）；教学实现 [[stanford-cs336]] Assignment 1。

## 来源

- [[building-a-fast-bpe-tokenizer-from-scratch]]（算法流程、tie-break 约定、复杂度阶梯与全部实测数字）
- [[qwen-tokenization-note]]（byte-level 特性、截断字符、纯分布特性与实例）

## 相关

- [[pretokenization]]（上游：决定 merge 的作用域）
- [[pretokenization-cjk-and-mixed-text]]（中文与混合文本的预切分实验）
- [[vocabulary-expansion]]（下游：训后加 token 为何必须走 merge 链）
- [[special-tokens]]（词表中与 regular token 并行的功能符号体系）
- [[tiktoken]] ｜ [[qwen]] ｜ [[tinystories]]（常用 BPE 基准语料）｜ [[stanford-cs336]]
