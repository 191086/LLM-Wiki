---
type: source
title: RoFormer：旋转位置编码（论文）
created: 2026-09-16
updated: 2026-09-16
tags:
  - paper
  - transformer
  - positional-encoding
sources: 1
raw: raw/RoPE.pdf
---

# RoFormer: Enhanced Transformer with Rotary Position Embedding

> 原文：[[raw/RoPE.pdf]] ｜ 类型：论文（arXiv:2104.09864**v5**，2021-04 首版 / 2023-11-08 本版，14 页）｜ 机构：追一科技（Zhuiyi Technology，深圳）｜ 作者：Jianlin Su（苏剑林）、Yu Lu、Shengfeng Pan、Ahmed Murtadha、Bo Wen、Yunfeng Liu

> **收录注记**：库内这份是 arXiv **v5**（2023-11-08），全文公式号、图号、表号均按 v5。视觉直读核验：14 页渲染通读 + 5 张表与式 15 / 式 34 局部放大转录。考订四处：①式 32 第一行印作 $q = f_q(\boldsymbol{x}_m, 0) = \boldsymbol{W}_q\boldsymbol{x}_n$，右端下标 $n$ 应为 $m$（与式 1 的 $\boldsymbol{q}_m = f_q(\boldsymbol{x}_m, m)$ 及第二行 $\boldsymbol{k}$ 行的写法对照可判）；②$\theta_i$ 定义在文中两处指标平移——式 15 用 $\theta_i = 10000^{-2(i-1)/d}$（$i$ 从 1 起），§3.3 与 §3.4.3 用 $\theta_i = 10000^{-2i/d}$（跟随 Vaswani 的 0 起），相差一个指标平移、频率族等价；③Table 2 题注把 GLUE 误排作 "GLEU"；④§4.5.4 称 RoFormer-1024「净胜 WoBERT 绝对 1.5%」，按 Table 5 算术对 WoBERT 实为 +1.69（test）/ +2.00（validation），1.5 恰为 RoFormer 自身 512→1024 的增益（68.29→69.79）——正文表述与表内算术不符，wiki 侧按表算术照录并注。**[wiki 外补注]**：该文后经同行评审刊于 Neurocomputing（2024）。

## 一句话总结

把「位置编码该长什么样」从设计题变成**求解题**：要求注意力的 query–key 内积只依赖相对位置差 $m-n$（式 11），二维下解这个函数方程得到唯一解——用与绝对位置成正比的旋转矩阵左乘 query / key（RoPE，式 12–16）。绝对式实现、相对式语义，零额外参数、正交保范数、内积随距离长程衰减、可接入线性注意力；WMT14 En-De 27.3→27.5、GLUE 三胜三负、中文长文档 CAIL2019-SCM 推理 1024 较 512 提升 1.5 分、领先 WoBERT（§4）。**[wiki 外]** LLaMA 以降几乎全部开源 LLM 的默认位置编码（候选来源：arXiv:2302.13971）。

## 关键要点

- **动机与既有方案盘点**（§2）：self-attention 本身位置无关——$\boldsymbol{q}_m = f_q(\boldsymbol{x}_m, m)$ 等三路投影（式 1）+ softmax 加权（式 2）对输入顺序置换等变，位置信息必须外挂。两族方案：**绝对式**＝输入端加位置向量 $\boldsymbol{W}_t(\boldsymbol{x}_i + \boldsymbol{p}_i)$（式 3），$\boldsymbol{p}$ 可训练（BERT 等，上限 $L$）或正弦生成（Vaswani，式 4）；**相对式**＝改内积展开式的各项——Shaw 加可训练相对嵌入（clip 相对距离，式 5）、Transformer-XL 四项分解换相对正弦项与位置无关 $\boldsymbol{u},\boldsymbol{v}$（式 6–7）、T5 折叠为纯可训练 bias $\boldsymbol{b}_{i,j}$（式 8）、DeBERTa 只留两个 content–position 交叉项（式 10）。共同点：都把位置信息「**加**」进 context 表示，依赖 softmax 内积展开后逐项打补丁——线性注意力无从套用（§1）
- **形式化：内积只编码相对位置**（§3.1，式 11）：$\langle f_q(\boldsymbol{x}_m, m),\ f_k(\boldsymbol{x}_n, n)\rangle = g(\boldsymbol{x}_m, \boldsymbol{x}_n, m-n)$，附空位初始条件（式 22）——不问「在哪」、只问「隔多远」
- **二维解＝旋转**（§3.2.1，式 12–13）：$f_q = (\boldsymbol{W}_q\boldsymbol{x}_m)e^{\mathrm{i}m\theta}$，实数形式即 $2\times2$ 旋转矩阵。求解过程（§3.4.1，式 20–33）：复数分解为模长 × 辐角后，约束强迫模长与位置无关（式 26a/27）、辐角修正项是位置的等差函数 $\phi(m) = m\theta + \gamma$（式 28–30）——**均匀角速度旋转不是众多设计之一，而是该约束的解**，$\theta$ 是唯一自由度（$\gamma$ 由与式 3 对齐取 0，式 31–33）
- **一般形式＝$d/2$ 个不同角速度的 2D 子空间**（§3.2.2，式 14–16）：$\boldsymbol{R}^d_{\Theta,m}$ 为块对角旋转阵，$\theta_i = 10000^{-2(i-1)/d}$（沿用 Vaswani base，式 15）；关键一步来自正交性：$(\boldsymbol{R}_m\boldsymbol{W}_q\boldsymbol{x}_m)^\top(\boldsymbol{R}_n\boldsymbol{W}_k\boldsymbol{x}_n) = \boldsymbol{x}_m^\top\boldsymbol{W}_q^\top\boldsymbol{R}_{\Theta,n-m}^d\boldsymbol{W}_k\boldsymbol{x}_n$（式 16，$\boldsymbol{R}_m^\top\boldsymbol{R}_n = \boldsymbol{R}_{n-m}$）——相对性「天然融入」，无需改写内积展开式
- **高效实现**（§3.4.2，式 34）：利用稀疏性，旋转 = 两组**逐元素乘**（$\boldsymbol{x} \otimes \cos(m\Theta) + \boldsymbol{x}_{\text{rot}} \otimes \sin(m\Theta)$，$\boldsymbol{x}_{\text{rot}} = (-x_2, x_1, -x_4, x_3, \ldots)$），$O(d)$、逐位置独立可并行；正交性同时保证编码过程数值稳定（§3.2.2）
- **长程衰减**（§3.3、§3.4.3，式 35–37、Figure 2）：内积按相邻二维分量配对写成复数求和（式 35），Abel 变换（离散分部求和）放缩出上界 $\big(\max_i|h_{i+1}-h_i|\big)\sum_i|S_{i+1}|$（式 37），其中频率部分和 $\sum|S_{i+1}|$ 随 $|m-n|$ 增大而衰减（Figure 2：距离 0→250 包络约 20 → 7–8，带振荡）——「远距离 token 连接更弱」直觉的形式化
- **线性注意力兼容**（§3.3，式 17–19）：softmax-free 注意力 $\phi(\boldsymbol{q}_m)^\top\phi(\boldsymbol{k}_n)\boldsymbol{v}_n / \sum\phi(\boldsymbol{q}_m)^\top\phi(\boldsymbol{k}_n)$（$\phi$ 非负，如 $\mathrm{elu}(x)+1$，$O(N)$ 复杂度）里，RoPE 只把旋转矩阵乘在分子两侧（式 19）、分母保持不动（防除零；分子可含负项，权重不再严格概率归一）——旧相对式方案全部做不到，因为它们的位置项依赖 softmax 内积的展开结构
- **实验五组**（§4，双云服务器 4×V100）：
	1. **翻译**（§4.1，Table 1）：WMT14 En-De（4.5M 句对，BPE 37k，fairseq），Transformer-base 27.3 → RoFormer **27.5** BLEU
	2. **英文预训练**（§4.2）：BERT-base 的正弦位置编码换 RoPE（同 batch 64 / len 512 / 100k 步 / AdamW 1e-5），MLM 损失收敛更快、终值更低（Figure 3 左）
	3. **GLUE 微调**（§4.3，Table 2）：三胜三负——MRPC **89.5** vs 88.9、STS-B **87.0** vs 85.8、QQP **86.4** vs 71.2（+15.2）；SST-2 90.7 vs 93.5、QNLI 88.0 vs 90.5、MNLI 80.2/79.8 vs 84.6/83.4（报告验证集 best-averaged，随 Devlin 惯例）
	4. **线性注意力**（§4.4）：PerFormer（$\phi(x)=\mathrm{elu}(x)+1$）+ RoPE，enwik8 字符级 12 层 / 768 维，序列 1024：收敛更快、损失更低（Figure 3 右）——式 19 的实证
	5. **中文长文**（§4.5）：词级 WoBERT（同第一作者）+ RoPE，34GB 中文语料**六阶段长度递进**预训练（512→1536 两轮拉伸，精度 65.0%→67.4%，Table 4）；CAIL2019-SCM 相似案例匹配（8,964 个三元组，文档普遍超 512 字，Table 3 对照四模型分词层级与位置编码类型）：512 截断下与 WoBERT 相当（test 68.29% vs 68.10%），**推理 1024** → 66.07% / **69.79%**——较自身 512 版 +1.5 分、对 WoBERT 表算术为 +1.69（Table 5；原文表述「净胜 WoBERT 1.5%」与表内算术不符，1.5 实为自身长度增益，考订④；预训练分布已含 1536 长度阶段，非训练外外推）
- **局限自陈**（§4.5.5）：①为何比其他位置编码收敛更快，无解释；②长程衰减性「与既有机制类似」（人人都有），为何长文表现更好，「没有给出忠实解释」

![[rope-fig3-pretraining-loss.png]]
> Figure 3（§4.2.3、§4.4.2）：左——英文 MLM 预训练损失，RoFormer（蓝）全程低于 BERT（橙）；右——PerFormer 线性注意力加 RoPE（蓝）对比不加（橙），收敛更快、终值更低。两组图共同支撑「乘性注入更易优化」的经验观察（机制未明，§4.5.5 自陈）。

## 值得追踪的实体与概念

- [[rope]]（概念页：函数方程推导、关键性质、位置编码方案对照表、后续谱系与候选来源）
- 候选来源（待收录）：Vaswani et al. 2017《Attention Is All You Need》——式 1–2 的注意力形式化与正弦频率几何（式 4）的源头锚点
