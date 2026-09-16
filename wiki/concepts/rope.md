---
type: concept
title: RoPE（旋转位置编码）
created: 2026-09-16
updated: 2026-09-16
tags:
  - transformer
  - positional-encoding
  - attention
sources: 1
---

# RoPE（Rotary Position Embedding，旋转位置编码）

**定义**：追一科技提出的 Transformer 位置编码（position encoding，给无序的注意力输入注入 token 顺序信息的机制）——不把位置向量「加」到输入表示上，而是用与绝对位置 $m$ 成正比的旋转矩阵 $\boldsymbol{R}^d_{\Theta,m}$ 左乘自注意力的 query 与 key（value 通路不动），使内积只依赖相对位置差 $n-m$（式 16）：**实现上是绝对式，语义上是相对式**。它不是众多设计之一，而是「内积只含 $m-n$」这一约束下函数方程的解（§2；[[rope-paper]] §3.1–3.2）；由此顺带拿到零额外参数、正交保范数、内积随距离长程衰减、线性注意力兼容四项性质（§4）。**[wiki 外]** LLaMA（2023）以降绝大多数开源 LLM 的默认位置编码（候选来源：arXiv:2302.13971）。

## 1. 要解决什么问题：注意力天生不认识顺序

自注意力（self-attention，序列中每个位置对其余所有位置做加权平均的机制）是集合式的：对每个 token 做 $\boldsymbol{q}_m = f_q(\boldsymbol{x}_m, m)$、$\boldsymbol{k}_n = f_k(\boldsymbol{x}_n, n)$、$\boldsymbol{v}_n = f_v(\boldsymbol{x}_n, n)$ 三路投影（式 1，$\boldsymbol{x}$ 为词嵌入），权重取 $\mathrm{softmax}(\boldsymbol{q}_m^\top\boldsymbol{k}_n/\sqrt{d})$（式 2）。把输入序列随机打乱，输出跟着同样打乱——注意力本身**置换等变**（permutation-equivariant，输入换序则输出同序重排、内容不变），顺序信息必须额外注入（[[rope-paper]] §2.1）。

2017–2021 年的两族注入方案（[[rope-paper]] §2.2–2.3）：

- **绝对式**：$\boldsymbol{W}_t(\boldsymbol{x}_i + \boldsymbol{p}_i)$（式 3），位置向量 $\boldsymbol{p}_i$ 可训练（BERT 等，序列上限 $L$、占 $L\times d$ 参数表）或按正弦函数生成（Vaswani 式 4，$\boldsymbol{p}_{i,2t} = \sin(i/10000^{2t/d})$）。位置信号只进输入端，「谁离谁近」要靠后续多层自己学出相对关系。
- **相对式**：直接改造注意力内积的展开式——Shaw et al. 给 key/value 加可训练相对嵌入 $\tilde{\boldsymbol{p}}_{r}$（$r = \mathrm{clip}(m-n, r_{\min}, r_{\max})$，式 5）；Transformer-XL 把 $\boldsymbol{q}_m^\top\boldsymbol{k}_n$ 拆成内容–内容、内容–位置、位置–内容、位置–位置四项分别改造（式 6–7）；T5 折叠为纯可训练 bias $\boldsymbol{b}_{i,j}$（式 8）；DeBERTa 只保留两个 content–position 交叉项（式 10）。

论文对两族的共同批评（§1）：它们都把位置信息「**加**」进 context 表示——本质是在 softmax 内积展开式里**逐项打补丁**，缺乏统一解释；且打补丁依赖展开结构，softmax-free 的线性注意力（见 §4.4）无从套用。RoPE 换一条路：**不动展开式，回到编码函数 $f$ 本身，施加约束直接求解**。

## 2. 目标的形式化：一个函数方程（§3.1）

要求位置编码后的 query–key 内积只以相对形式携带位置信息（式 11）：

$$\langle f_q(\boldsymbol{x}_m, m),\ f_k(\boldsymbol{x}_n, n)\rangle = g(\boldsymbol{x}_m, \boldsymbol{x}_n,\, m-n)$$

（$\langle\cdot,\cdot\rangle$ 为内积；$f_q, f_k$ 就是式 1 里待定的编码函数，$g$ 是某个双线性打分。）直觉：注意该问「**隔多远**」，不该问「**在哪**」——平移整段文本，注意力权重不应改变。另附空位初始条件 $f(\boldsymbol{x}, 0)$ 为无位置编码向量（式 22），作为解的规范化锚点。问题是：满足它的 $f_q, f_k$ 长什么样、是否唯一。

## 3. 解：从二维旋转到 $d$ 维块对角（§3.2、§3.4）

### 3.1 二维情形：唯一解是均匀角速度旋转

求解分三步（§3.4.1，式 20–33）。①把二维向量当复数，分解为模长 × 辐角（式 23）：$f_q = R_q(\boldsymbol{x}_q, m)e^{\mathrm{i}\Theta_q(\boldsymbol{x}_q, m)}$。②在式 11 里取 $m = n$（同位置自比）：模长部分解出 $R_q = \|\boldsymbol{q}\|$、$R_k = \|\boldsymbol{k}\|$，**与位置无关**（式 26a/27）——位置信息只能活在辐角里；辐角部分给出 $\Theta_q(\boldsymbol{x}_q, m) - \theta_q = \Theta_k(\boldsymbol{x}_k, m) - \theta_k =: \phi(m)$，即辐角修正项只依赖位置、与具体词嵌入无关（式 26b/28）。③在式 11 里取 $n = m+1$ 递推：$\phi(m+1) - \phi(m)$ 是与 $m$ 无关的常数（式 29）——$\phi$ 是**等差数列**，$\phi(m) = m\theta + \gamma$（式 30）。

于是（式 31–33）：

$$f_q(\boldsymbol{x}_m, m) = (\boldsymbol{W}_q\boldsymbol{x}_m)\,e^{\mathrm{i}m\theta}, \qquad f_k(\boldsymbol{x}_n, n) = (\boldsymbol{W}_k\boldsymbol{x}_n)\,e^{\mathrm{i}n\theta}$$

写成实数形式，就是用 $2\times2$ 旋转矩阵 $\begin{pmatrix}\cos m\theta & -\sin m\theta\\ \sin m\theta & \cos m\theta\end{pmatrix}$ 左乘常规投影后的向量（式 13）。$\gamma$ 取 0 只为与绝对式（式 3）对齐；**$\theta$ 是解里唯一的自由度**——「位置以均匀角速度旋转进入编码」是约束的必然，不是设计选择。

### 3.2 一般形式：$d/2$ 个不同角速度的子空间

$d$（偶数）维拆成 $d/2$ 个二维子空间，各配一个角速度（式 14–15）：

$$f_{\{q,k\}}(\boldsymbol{x}_m, m) = \boldsymbol{R}^d_{\Theta,m}\boldsymbol{W}_{\{q,k\}}\boldsymbol{x}_m, \qquad \boldsymbol{R}^d_{\Theta,m} = \mathrm{blockdiag}\big(\boldsymbol{R}(m\theta_1), \ldots, \boldsymbol{R}(m\theta_{d/2})\big), \quad \theta_i = 10000^{-2(i-1)/d}$$

（$\boldsymbol{R}(\cdot)$ 即上节的 $2\times2$ 旋转；base $=10000$ 沿用 Vaswani 式 4 的频率几何——论文 §3.3 行文用 $10000^{-2i/d}$，与式 15 相差一个指标平移、频率族等价，见来源页考订②。）相对性一行推完（式 16）：

$$\boldsymbol{q}_m^\top\boldsymbol{k}_n = (\boldsymbol{R}_m\boldsymbol{W}_q\boldsymbol{x}_m)^\top(\boldsymbol{R}_n\boldsymbol{W}_k\boldsymbol{x}_n) = \boldsymbol{x}_m^\top\boldsymbol{W}_q^\top\,\boldsymbol{R}^d_{\Theta,n-m}\,\boldsymbol{W}_k\boldsymbol{x}_n, \qquad \boldsymbol{R}_m^\top\boldsymbol{R}_n = \boldsymbol{R}_{n-m}$$

正交旋转阵的转置相消把绝对位置 $m, n$ 归并成差 $n-m$——相对语义来自**代数恒等式**，不需要像相对式那样改写展开式。频率谱上，低维子空间角速度大（波长 $\lambda_i = 2\pi/\theta_i$ 短）、高维子空间波长长：$d=4$ 时两维波长分别为 $6.28$ 与 $62{,}832$ 个位置（本机复算）——低维分辨近距、高维表达远距，正弦族「多分辨率」直觉的乘法版。

![[rope-fig1-implementation.png]]
> Figure 1（[[rope-paper]] §3.2.2）：RoPE 实现。上——二维子空间：位置 $m$ 的 query/key 旋转 $m\theta_1$ 后得到位置编码向量；下——逐位置多子空间：每个位置对 $d/2$ 个子空间各转各的角度，位置信息只改变 q/k 的方向、不改变其长度。

### 3.3 高效实现（§3.4.2，式 34）

块对角旋转矩阵极稀疏，不必显式构造。按式 34，$\boldsymbol{R}^d_{\Theta,m}\boldsymbol{x}$ = 两组逐元素乘之和：

$$\boldsymbol{R}^d_{\Theta,m}\boldsymbol{x} = \begin{pmatrix}x_1\\x_2\\x_3\\x_4\\\vdots\end{pmatrix} \otimes \begin{pmatrix}\cos m\theta_1\\\cos m\theta_1\\\cos m\theta_2\\\cos m\theta_2\\\vdots\end{pmatrix} + \begin{pmatrix}-x_2\\x_1\\-x_4\\x_3\\\vdots\end{pmatrix} \otimes \begin{pmatrix}\sin m\theta_1\\\sin m\theta_1\\\sin m\theta_2\\\sin m\theta_2\\\vdots\end{pmatrix}$$

（$\otimes$ 为逐元素乘。）每维 4 乘 2 加、$O(d)$，且逐位置独立、可并行；预计算 $\sin/\cos$ 表后推理开销可忽略。两个工程注记：

- **[wiki 外]** 配对约定：论文按相邻对 $(x_1,x_2),(x_3,x_4),\ldots$ 配子空间；LLaMA / GPT-NeoX 系实现按**半分配对**（$x_i$ 与 $x_{i+d/2}$）。两者性质等价——§3.2 的证明只依赖 $\boldsymbol{R}_m^\top\boldsymbol{R}_n = \boldsymbol{R}_{n-m}$，不依赖哪个分量与哪个分量配对，只要 query、key 用**同一**约定；但跨约定迁移权重会静默破坏位置语义。候选来源：arXiv:2302.13971（LLaMA）、arXiv:2204.06745（GPT-NeoX）
- **value 通路不旋转**：位置信息只进入注意力的权重分配，不直接改写被加权的内容向量——从式 16 与式 19 的结构直接可见

## 4. 关键性质

### 4.1 绝对式实现、相对式语义、零参数

训练与推理都只在 query/key 投影后各乘一个旋转（逐元素运算，§3.3），不新增任何参数——对比可训练绝对式的 $L\times d$ 参数表与 T5 的 bucket bias 表；相对语义由正交性恒等式免费给出（式 16）。既有方案的逐项对照见 §7 表。

### 4.2 正交性：保范数、数值稳定、不碰 value

旋转是正交变换、不改向量长度（$\boldsymbol{R}^\top\boldsymbol{R} = \boldsymbol{I}$），位置编码因此不会放大或缩小特征维度；论文点名正交性「保证编码过程稳定」（§3.2.2）。位置信息只经 query/key 进入权重分配（§3.3 末条）。

### 4.3 长程衰减（§3.3、§3.4.3）

把内积按相邻二维分量配对写成复数求和（式 35）：$\mathrm{Re}\big[\sum_i h_i\, e^{\mathrm{i}(m-n)\theta_i}\big]$，$h_i = \boldsymbol{q}_{2i:2i+1}\boldsymbol{k}^*_{2i:2i+1}$。对 $\sum_i h_i e^{\mathrm{i}(m-n)\theta_i}$ 用 **Abel 变换**（离散分部求和，把「频率因子的部分和」与「系数差分」重新配对的恒等式）得（式 36–37）：

$$\Big|\sum_i h_i e^{\mathrm{i}(m-n)\theta_i}\Big| \le \big(\max_i|h_{i+1}-h_i|\big)\cdot\sum_i|S_{i+1}|, \qquad S_j = \sum_{i=0}^{j-1} e^{\mathrm{i}(m-n)\theta_i}$$

右侧 $\sum|S_{i+1}|$ 是**非完备三角和**——相位步长 $\theta_i$ 取自几何递减序列（$10000^{-2i/d}$）时，它随相对距离 $|m-n|$ 增大而衰减（Figure 2：距离 0→250 包络约 20 → 7–8，带振荡）。两个口径提醒：①衰减的是内积的**上界**，不是内积逐点单调；②衰减性由正弦频率族给出，并非 RoPE 独有——论文自陈长文表现优于同类「无忠实解释」（§4.5.5）。

![[rope-fig2-long-term-decay.png]]
> Figure 2（[[rope-paper]] §3.4.3）：内积上界中频率部分和 $\frac{1}{d/2}\sum_i|S_i|$ 随相对距离（0→250）的衰减——整体包络下降、叠加振荡，即「远距离 token 连接更弱」的形状。

### 4.4 线性注意力兼容（§3.3，式 17–19）

softmax 注意力要对全部 $N^2$ 个位置对算内积。**线性注意力**（linear attention，用非负特征映射 $\phi$ 拆开 $\exp(\boldsymbol{q}^\top\boldsymbol{k}/\sqrt d)$ 使求和可换序、复杂度降到 $O(N)$ 的近似注意力，如 $\phi(x) = \mathrm{elu}(x)+1$，Performer 等）形如（式 18）：

$$\mathrm{Attention}(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V})_m = \frac{\sum_n \phi(\boldsymbol{q}_m)^\top\phi(\boldsymbol{k}_n)\,\boldsymbol{v}_n}{\sum_n \phi(\boldsymbol{q}_m)^\top\phi(\boldsymbol{k}_n)}$$

RoPE 把旋转矩阵乘在**分子**两侧 $\phi(\cdot)$ 的输出上（式 19），分母保持不动：防除零；分子可含负项，权重不再严格概率归一——论文论点是这样仍能表达各 value 的重要性（§3.3）。旧相对式方案全线不兼容，因为它们的位置项全部寄生在 softmax 内积展开的各项上（式 6–10），线性注意力里没有对应结构可改；RoPE 的乘性注入不依赖任何展开形式。

## 5. 实验证据（§4）

- **翻译**（§4.1，Table 1）：WMT14 En-De，Transformer-base 27.3 → RoFormer **27.5** BLEU——唯一改动是把正弦加法位置编码换成 RoPE（式 16 实现）
- **英文预训练**（§4.2，Figure 3 左）：BERT-base 同设置对比，MLM（masked language modeling，遮词重建的 BERT 式预训练目标）损失收敛更快、终值更低
- **GLUE 微调**（§4.3，Table 2；GLUE = General Language Understanding Evaluation，通用语言理解评测套件）：三胜三负，MRPC 89.5 / STS-B 87.0 / QQP 86.4 胜（QQP 差距 +15.2），SST-2 / QNLI / MNLI 负——收益不普适，作者只主张「显著优于基线的三个数据集上提升可观」（§4.3.3）
- **线性注意力**（§4.4，Figure 3 右）：PerFormer + RoPE（enwik8——Hutter Prize 的英文维基字节级压缩基准——字符级语言建模，序列 1024）收敛更快、损失更低——§4.4 的实证；「线性复杂度 + 相对位置」二者兼得的演示
- **中文长文**（§4.5）：词级 WoBERT（以词为单位的中文 BERT，本篇第一作者的前作）+ RoPE，34GB 语料六阶段长度递进预训练（Table 4：512→1536 两轮拉伸，精度 65.0%→67.4%——精度随训练长度上限单调升）；CAIL2019-SCM（中国法研杯相似案例匹配，8,964 个三元组，文档普遍超 512 字，512 截断会让多数样本吃不全）：512 处与 WoBERT 持平（test 68.29% vs 68.10%），**推理 1024** 处拉开到 66.07% / **69.79%**——较自身 512 版 +1.5 分、对 WoBERT 表算术为 +1.69（Table 5；原文表述「净胜 WoBERT 1.5%」与表内算术不符，1.5 实为自身长度增益，见来源页考订④）——长文本收益的兑现点在「能喂多长」。注意：预训练分布已含 1536 长度阶段（Table 4 阶段 2/5），1024 推理不是训练外外推

「收敛更快」在全部预训练对比中一致出现（Figure 3 左右），但机制未明——这是论文自己承认的两个未解之一（§4.5.5，另一见 §4.3）。

## 6. 局限与后续谱系

论文自陈两点（§4.5.5）：①相对其他位置编码为何收敛更快，缺解释；②长程衰减「与既有机制类似」，为何长文表现更好，没有忠实解释。

**[wiki 外]** 后续谱系（本库未锚定，候选来源如下）：①**成为默认**：LLaMA（2023）起，主流开源 LLM 全线采用 RoPE（arXiv:2302.13971；实现约定差异见 §3.3 注记）；②**外推与长度扩展**：RoPE 训练长度之外直接外推会退化（衰减 ≠ 有效外推，高频子空间未见过的相位组合），由此长出修正族——Position Interpolation 线性内插（arXiv:2306.15595）、NTK-aware 缩放、YaRN（arXiv:2309.00071）、LongRoPE（arXiv:2402.13753）等；③上下文长度是对话历史与长文档的硬约束，这条线与 [[rlhf]] 所在的后训练主线正交互补。

## 7. 位置编码方案对照

| 方案 | 注入方式 | 位置语义 | 额外参数 | 线性注意力兼容 | 出处锚点 |
| --- | --- | --- | --- | --- | --- |
| 正弦绝对（式 4） | 输入端加 | 绝对 | 0 | ✗ | [[rope-paper]] §2.2 |
| 可训练绝对（式 3，BERT） | 输入端加 | 绝对 | $L\times d$ | ✗ | §2.2 |
| Shaw 相对（式 5） | key/value 加 | 相对（clip 截断） | 相对嵌入表 ×2 | ✗ | §2.3 |
| Transformer-XL（式 6–7） | 内积四项改造 | 相对 | $\boldsymbol{u},\boldsymbol{v},\widetilde{\boldsymbol{W}}_k$ | ✗ | §2.3 |
| T5 bias（式 8） | 内积加 bias | 相对（分桶） | bucket 表 | ✗ | §2.3 |
| DeBERTa（式 10） | 内积只留交叉项 | 相对 | 相对嵌入表 | ✗ | §2.3 |
| **RoPE（式 14–16）** | **q/k 乘旋转** | **绝对式实现、相对式语义** | **0** | **✓（式 19）** | §3 |

## 来源

- [[rope-paper]]（§1 动机、§2 综述与式 1–10、§3.1 式 11、§3.2 式 12–16 与 Figure 1、§3.3 性质与式 17–19、§3.4.1 推导式 20–33、§3.4.2 式 34、§3.4.3 衰减式 35–37 与 Figure 2、§4 实验 Table 1–5 与 Figure 3、§4.5.5 局限）

## 相关

- [[rope-paper]]（来源页：逐节要点、三处考订、Figure 3）
- 候选待收录：Vaswani et al. 2017《Attention Is All You Need》（自注意力形式化与正弦频率几何的源头锚点，§1–2 多处引用）
