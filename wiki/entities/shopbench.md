---
type: entity
title: ShopBench（基准）
created: 2026-09-14
updated: 2026-09-15
tags:
  - benchmark
  - mllm
  - fsrs
sources: 1
---

# ShopBench

**定义**：首个面向餐饮零售场景（FSRS）的公开多模态基准，随 [[ostrakon-vl]] 论文发布，覆盖单图、多图、视频三类输入，用于评测零售合规与运营决策所需的细粒度感知与推理。

## 详情

### 构成（[[ostrakon-vl-paper]] 附录 A.2）

- 共 **5,818 题**：感知 4,716 + 推理 1,102
- L2 层：细粒度感知 2,362 单实例 + 1,355 跨实例、粗感知 999
- L3 层最大的类：目标定位 846、属性识别 713、OCR 674（应对招牌 / 地址 / 菜单等零售核心）；共展开到 L4 层 **79 个任务定义**
- 五个子数据集：ShopFront / ShopInterior / Kitchen（单图三场景）+ MultiImg + Video（跨场景、按输入格式划分）
- 三种输出格式：Open-Ended（自由问答）、Format（预定义 schema）、MCQ（选择）

![[ostrakon-vl-fig2-shopbench-taxonomy.png]]
> ShopBench 的 L1-L4 四层分类树：感知（粗 / 细粒度×单实例 / 跨实例）与推理（属性 / 逻辑 / 关系）统一映射到与输入格式无关的 L4 标签空间（论文图 2）。

### 与现有基准的区隔

- **域分布独立**：GME-Qwen2VL-2B 嵌入 + t-SNE（wiki 外 gloss：把高维嵌入压到二维平面、尽量保住局部邻接关系的可视化降维方法——远邻距离不可信，「成独立簇」的判读只依赖局部不混叠）显示 ShopBench 与 MMBench / MMStar / MMVet / HallusionBench / AI2D / OCRBench / MathVista / MMMU 八个主流基准几乎不重叠（论文图 4）
- **场景最复杂**：平均每图实例数（InsPerImg）**13.0**，八个基准中最高（第二名 MMBench-EN 10.6、MMStar 9.3）——拥挤货架、SKU 多样、人-物交互密集
- **抗泄漏**：平均 Multimodal Leakage 0.02，八个基准中最低（论文表 2）

![[ostrakon-vl-fig4-benchmark-tsne.png]]
> t-SNE 投影下 ShopBench（三场景）与其他八个基准的分布对比：形成独立簇，说明其视觉-语义分布未被现有基准覆盖（论文图 4）。

### 诊断指标：VNR / VIF

论文指出传统 Multimodal Gain（MG）有盲区：语言先验强的基准上 MG 虚高（其实是题简单）；视觉极复杂的域上 MG 被「视觉干扰」压制甚至变负（题其实依赖视觉）。ShopBench 论文将其分解为：

- **VNR**（Visual Necessity Rate）：视觉输入对答对不可或缺的样本占比——真实的视觉必要性
- **VIF**（Vision-Induced Failure）：不看图能对、看图反而错的占比——视觉干扰的损害

跨四个模型（Qwen3-VL-235B/8B、InternVL3.5-8B、MiniCPM-V2.6）VNR/VIF 高度一致，说明二者由**基准本身**而非模型决定，适合作为基准设计质量的体检指标。OCRBench 的 VNR≈1.0 / VIF≈0.0 是「纯视觉任务」的理论基线；ShopBench 平均 VNR 0.58 / VIF 0.12。

## 来源

- [[ostrakon-vl-paper]]

## 相关

- [[ostrakon-vl]] ｜ [[domain-specific-mllm]]
