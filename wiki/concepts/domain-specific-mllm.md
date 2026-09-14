---
type: concept
title: 领域专用多模态大模型（Domain-Expert MLLM）
created: 2026-09-14
updated: 2026-09-14
tags:
  - mllm
  - domain-model
  - survey
sources: 1
---

# 领域专用多模态大模型（Domain-Expert MLLM）

**定义**：在通用 MLLM 基础上，用领域数据、领域训练与标准化领域评测构建的垂直方向多模态模型，及其配套的「领域模型-数据闭环-基准评测」完整栈；[[ostrakon-vl]] 是餐饮零售（FSRS）方向的第一个统一栈。

## 详情

### 为什么通用 MLLM 不够用（[[ostrakon-vl-paper]] §1 的三重错位）

通用模型的训练目标、真实世界的视觉数据分布、工业界端到端「感知→推理」系统的需求三者系统性错位。以 FSRS 为例：领域要求识别运营性线索（区分运营招牌与装饰元素）、容忍领域典型退化（橱窗眩光、低分辨率多语言文字、运动模糊、瞬时遮挡），且这些要求在通用预训练 / 对齐数据中基本缺席。

### 各领域的成型栈（论文 §2 综述）

| 领域 | 模型 / 工作 | 栈的形态 |
|---|---|---|
| 金融 | FinTMMBench、Open-FinLLMs | 时序表+新闻+图表的评测与预训练+指令微调 |
| 医疗 | BiomedCLIP、BrainGPT | 图文对齐预训练；嵌入临床工作流（3D 脑 CT 报告） |
| 生物材料 | Cephalo | 生物材料图文连接 + 下游结构设计 |
| 餐饮零售 | [[ostrakon-vl]] + [[shopbench]] + [[quad-data-curation|QUAD]] | 首个「模型+数据管线+基准」统一栈 |

### 核心方法论（以 Ostrakon-VL 为样本）

![[ostrakon-vl-fig1-framework.png]]
> 领域模型三件套的完整框架：上半为 QUAD 数据清洗管线，下半为 CB→OCL→MPO 多阶段训练（论文图 1）。

1. **数据管线先行**：合成数据 + 可审计的系统性清洗（见 [[quad-data-curation]]），而非朴素 MLLM-as-judge 过滤
2. **多阶段训练**：领域知识注入（caption）→ 课程式指令微调 → 偏好优化（见 [[mixed-preference-optimization]]）
3. **自建基准**：领域分布独立于现有基准、强调视觉证据必要性（见 [[shopbench]] 的 VNR/VIF）

### 参数效率论证与代价

- **域内**：8B 的 [[ostrakon-vl]] 在 ShopBench（60.1）反超 235B 的 Qwen3-VL（59.4）与 72B 的 Qwen2.5-VL（57.3）——通用 scaling 不足以覆盖领域精度，「领域指令 + 训练策略」是小模型的杠杆
- **域外**：通用基准 72.4→66.7 的回退是已知代价（详见 [[ostrakon-vl]] 页）；领域化本质是能力预算的重新分配而非免费增益

## 来源

- [[ostrakon-vl-paper]]

## 相关

- [[ostrakon-vl]] ｜ [[shopbench]] ｜ [[quad-data-curation]] ｜ [[mixed-preference-optimization]] ｜ [[qwen]]
