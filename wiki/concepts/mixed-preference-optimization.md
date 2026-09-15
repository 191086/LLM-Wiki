---
type: concept
title: MPO（混合偏好优化）
created: 2026-09-14
updated: 2026-09-15
tags:
  - alignment
  - preference-optimization
  - training
sources: 3
---

# MPO（Mixed Preference Optimization，混合偏好优化）

**定义**：把偏好损失（[[dpo|DPO]]）、质量损失（BCO）与生成损失（SFT）加权求和的对齐训练目标，用**离线偏好对**而非在线采样完成对齐；InternVL 系列起被广泛采用，[[ostrakon-vl]] 用作三段训练策略的收尾段。

## 1. 要解决什么问题：离线偏好对齐

偏好对齐有两大家路线。**在线 RL 路线**（如 [[grpo|GRPO]]）：训练中不断采样多条回答、打奖励分、按组相对优势更新策略——信号新鲜但每步都要「生成 + 评估」多个候选，显存与时间开销大。**离线路线**（MPO 为代表）：事先备好偏好对 $(x, y^+, y^-)$，训练时只做前向反向的监督式更新——省掉了在线采样，代价是偏好对的质量决定一切（§4）。

论文选择 MPO 而非 GRPO 的理由即**计算效率**（[[ostrakon-vl-paper]] §4.3）。消融中 MPO 单项贡献最小（+0.2，55.3 基座上 FSRS-Inst 56.8 → 57.0），但与 CB/OCL 组合时有叠加收益（CB+MPO 58.5、OCL+MPO 58.5、三者全开 60.1）。

对照数据点（2026-09-15 ingest）：GRPO 阵营的 [[vision-r1]] 用在线组采样 + 规则奖励，仅 49K 样本 / 1 epoch 就把 Qwen2.5-VL-7B 的 COCO 定位 mAP 提了 8.9、ODINW-13 反超其 72B——「GRPO 开销大」针对的是每样本计算成本，并不否定小数据下的收益上限；两路线在同任务同数据上的头对头仍然缺失（[[vision-r1-paper]] §4.2）。

## 2. 目标函数：三项损失各管一件事

### 2.1 加权外壳

$$\mathcal{L}_\text{MPO} = w_1\, \mathcal{L}_\text{preference} + w_2\, \mathcal{L}_\text{quality} + w_3\, \mathcal{L}_\text{generation} \tag{式 12}$$

每个 $w_*$ 控制对应项的贡献（[[ostrakon-vl-paper]] §4.3 原式）。论文对三项的定性：偏好损失「训练模型辨别两个候选哪个更好」、质量损失「评估单条回答的内在质量」、生成损失「正则项，保持流畅稳定、防止过度偏离预训练分布」。

### 2.2 三项的损失形式

> 三项的具体公式为 wiki 外一般性知识（Ostrakon 论文只引出处不给式子）。候选来源：DPO——Rafailov et al. 2023（arXiv:2305.18290），机制详见 [[dpo]]；BCO——Jung et al.《Binary Classifier Optimization for LLM Alignment》（arXiv:2404.04656，ACL 2025）；SFT——标准的下一词元交叉熵。

记 $h(y \mid x) \coloneqq \log \frac{\pi_\theta(y \mid x)}{\pi_\text{ref}(y \mid x)}$（策略相对冻结参考模型的 log 概率比，$\pi_\text{ref}$ 的角色见 [[dpo]] §4）：

$$\mathcal{L}_\text{pref} = -\,\mathbb{E}\, \log \sigma\!\big(\beta\,(h^+ - h^-)\big) \qquad \text{（源自 DPO：只看差值，学「谁更好」）}$$

$$\mathcal{L}_\text{qual} = -\,\mathbb{E}\, \log \sigma\big(\beta\, h^+\big) \;-\; \mathbb{E}\, \log \sigma\big({-\beta\, h^-}\big) \qquad \text{（源自 BCO：两条各判各的，学「绝对好不好」）}$$

$$\mathcal{L}_\text{gen} = -\,\mathbb{E}\, \log \pi_\theta(y^+ \mid x) \qquad \text{（源自 SFT：在 chosen 上做标准交叉熵）}$$

三个术语首次出现时的定位：**DPO**（Direct Preference Optimization）把偏好学习直接做在策略上、无需显式 RM（[[dpo]]）；**BCO**（Binary Classifier Optimization）把「成对比较」退化成「单条二元判断」——每一回答独立地过 sigmoid，好坏信号不再捆绑在对比里；**SFT**（Supervised Fine-Tuning，监督微调）即对目标回答的最大似然。

### 2.3 数值走查：为什么偏好项之外还要质量项

取 $\beta = 1$，比较两种情形（同 margin、不同绝对水平）：

| 情形 | $h^+$ | $h^-$ | $\mathcal{L}_\text{pref}$（只看差） | $\mathcal{L}_\text{qual}$（各判各的） |
| --- | --- | --- | --- | --- |
| A | $1.0$ | $-0.5$ | $-\log\sigma(1.5) = 0.20$ | $0.31 + 0.47 = 0.79$ |
| B | $5.0$ | $3.5$ | $-\log\sigma(1.5) = \mathbf{0.20}$（无感） | $0.007 + 3.53 = \mathbf{3.54}$ |

情形 B 中两条回答的概率都相对参考模型大涨（被拒那条涨了 $e^{3.5} \approx 33$ 倍），但差值不变——**DPO 项完全无感，BCO 项重罚被拒回答的绝对上涨**。这正是 MPO 混入质量项的动机：补上 [[reward-model]] §4 所说 BT 路线缺失的绝对质量信号。生成项则缓解 DPO 的另一失败模式：连正确回答的绝对概率也被无意压低（$h^+$ 本可被差值优化拖下去，SFT 项把它拉住）。

## 3. 偏好对的构造（质量决定 MPO 效果）

1. **高温采样**暴露模型的不稳定行为：同一输入多次生成、时对时错的样本最有价值——偶发的正确往往是表面模式匹配而非真理解
2. 每个输入取「最高质量的正确回答」配「貌似正确的错误回答」（几乎正确但含细微错误：数错物体、轻微事实偏差）——论文称 challenging preference pairs：显式对比让模型注意到此前忽略的细粒度视觉与语义线索（[[ostrakon-vl-paper]] §4.3）
3. **规则奖励**区分正确与貌似正确

## 4. 偏好数据质量的直接对照（[[skywork-vl-reward-paper]] §4.7）

同一 MPO 配方下只换偏好数据的来源 RM，差异显著（Skywork R1V2 基座 reasoner，MathVista）：

| 偏好数据来源 RM | MathVista |
|---|---|
| （基座，不微调） | 69.2 |
| Qwen2.5-VL-7B-Instruct | 71.2 |
| InternVL3-8B | 71.8 |
| **[[skywork-vl-reward|Skywork-VL-Reward]]** | **73.5** |

即 MPO 的上限由 RM / 偏好数据质量决定；R1V2 的采用链由此有了第一手证据（此前仅 [[ostrakon-vl-paper]] 转述）。

## 5. 使用脉络

论文自述的采用链（[[ostrakon-vl-paper]] §4.3）：InternVL 系列（InternVL / InternVL3 / InternVL3.5）→ Skywork R1V2、Kwai Keye-VL → [[ostrakon-vl]]（均为开源多模态模型的后对齐段）。

## 来源

- [[ostrakon-vl-paper]]（式 12 与三项定性、GRPO 对比、偏好对构造）
- [[skywork-vl-reward-paper]]（§4.7 换 RM 对照实验）
- [[vision-r1-paper]]（§4.2 GRPO 路线对照数据点）

## 相关

- [[dpo]]（偏好项的完整机制：重参数化推导与性质）
- [[reward-model]]（BT 排序损失——偏好项的理论源头）
- [[ostrakon-vl]] ｜ [[quad-data-curation]] ｜ [[domain-specific-mllm]] ｜ [[grpo]] ｜ [[rule-based-reward]] ｜ [[vision-r1]]
