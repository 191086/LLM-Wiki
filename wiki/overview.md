---
type: overview
title: 总览
created: 2026-09-12
updated: 2026-09-15
sources: 5
---

# 总览

> 本页是全局综合页，反映「读过的所有来源叠加之后」的整体理解。每次 ingest 后更新。

**当前状态**：已收录 5 份来源，两条主线：**LLM 分词（tokenization）**（算法与训练优化、工业实现）与 **多模态大模型的领域化与后训练**（数据管线、训练策略、奖励模型、免 RM 强化学习、领域评测，以 Ostrakon-VL 与 Vision-R1 为两个样本），并沉淀了一份中文场景的实验分析。

## 当前主线

### 主线一：分词（tokenization）

- **分词算法层**：[[bpe-tokenization|BPE]] 是现代 LLM 的标准分词算法；其训练本身是可系统性优化的算法工程问题（[[building-a-fast-bpe-tokenizer-from-scratch]] 给出五级优化路径，~230×）
- **工程实践层**：以 [[qwen]] 官方实现为样本——[[special-tokens|特殊 token]] 体系与注入防护、[[vocabulary-expansion|词表扩展]]流程、byte-level 细节（[[qwen-tokenization-note]]）
- **中文视角**：[[pretokenization|预切分]]对中文不设防（`\p{L}` 不分文字系统、不做分词），chunk 变长恶化训练复杂度，token 层面存在「中文税」（[[pretokenization-cjk-and-mixed-text]]，实验验证）

### 主线二：多模态大模型的领域化（2026-09-14 新增）

- **样本**：[[ostrakon-vl]]（淘宝闪购，基于 Qwen3-VL-8B 的餐饮零售领域模型）带来完整的「模型 + 数据管线 + 基准」三件套（[[ostrakon-vl-paper]]）
- **数据质量 > 数据量**：[[quad-data-curation|QUAD]] 四阶段清洗把指令语料压缩 20.4× 反升 2.5 分——领域微调的首要工程是数据蒸馏而非数据堆积
- **训练方法论**：caption 注入领域知识 → 课程学习排序 → [[mixed-preference-optimization|MPO]] 偏好对齐的三段递进；组合效应大于单项（CB+OCL 贡献最大）
- **评测方法论**：[[shopbench|ShopBench]] 用 VNR/VIF 分解 Multimodal Gain，度量基准「真依赖视觉」的程度；领域基准应与现有基准分布独立、抗语言先验泄漏
- **奖励模型环节**：[[skywork-vl-reward]]（判别式 ORM，[[reward-model]]）作 QUAD 统一裁判；其训练语料本身也是「surrogate RM 打分回路」清洗的产物（[[skywork-vl-reward-paper]]），且偏好数据质量直接决定 MPO 上限（MathVista 同配方换 RM：71.2 / 71.8 / 73.5）
- **后训练强化学习（免 RM 路线，2026-09-15 新增）**：[[vision-r1]] 用规则奖励（格式 + 召回 + 精度三路程序化信号）+ 渐进收紧做 [[grpo|GRPO]]，绕开偏好数据与奖励模型，49K 样本就把 Qwen2.5-VL-7B 的定位 mAP 提 ≈50%——与学习式 RM（[[skywork-vl-reward]]）、偏好对齐（MPO）、RM 清洗（QUAD）构成「评价信号从哪来」的路线对照（[[vision-r1-paper]]）

## 关键结论

- 朴素 BPE 训练为 O(m×W×L)；增量更新、并行预切分、倒排索引、堆+惰性删除四项累计约 **230×** 加速
- 收益随词表规模变化：vocab≈1300 是倒转索引方案与堆方案的分水岭；生产词表（32K–100K）必用堆；同阶复杂度的常数因子可差 5–15×
- **BPE 纯按分布工作**，无 Unicode / 语言学知识：同一词上下文相关切分（`Panda` vs ` Panda`）、token 可截断多字节字符（Qwen 51461）、有限数据 merge 可能跨码点
- 特殊 token 是独立于 regular token 的类型系统（`bytes` vs `str`）；其表面形式出现在输入文本即为注入通道——Qwen 的安全默认被社区惯例翻盘，防护需显式 `allowed_special=set()`
- 词表扩展不能直接加词：需学中间 merge 链 + 微调模型，且受预切分边界约束（跨块词加不进去）
- 中文预切分：GPT-2 / cl100k / Qwen2.5 均无 CJK 专门规则，汉字串整块成 chunk、无空格混合文本跨文字系统同块
- **8B 领域专化可反超 30× 大的通用模型**（域内）：Ostrakon-VL 60.1 vs Qwen3-VL-235B 59.4；代价是通用基准 -5.7（72.4→66.7），领域化是能力预算的重新分配
- **指令数据压缩 20.4× 性能反升**：质量过滤（含视觉消融检验）+ 基座参考过滤 + 语义去重 + 能力重分布，各阶段全部可审计可复现
- MLLM-as-judge 一过滤了事在领域场景不够：朴素过滤易引入伪相关、无法审计、缺下游反馈闭环（[[quad-data-curation]]）
- **奖励模型的排序损失只学相对序**：绝对分值未经校准，只能差值 / 阈值使用，且「等质」偏好对须在训练前剔除（[[reward-model]]）
- **偏好数据质量决定 MPO 上限**：同配方只换数据来源 RM，MathVista 69.2→71.2 / 71.8 / 73.5（[[mixed-preference-optimization]]）
- reward-guided curation 贯穿两层：Skywork 用 surrogate RM 回路清洗 RM 自己的语料，Ostrakon 再拿成品 RM 清洗领域语料——「裁判有偏 → 数据有偏 → 下游清洗有偏」的传导链已现雏形
- **免 RM 的规则 RL 路线成立**：三路程序化奖励（格式 / 召回 / 精度，各自对应 LVLM 定位的一种典型失败）+ [[grpo|GRPO]]，49K 数据 / 1 epoch 让 Qwen2.5-VL-7B 定位 mAP +50%、ODINW-13 反超 10× 大的 72B（[[vision-r1-paper]]）
- **RL 后训练比 SFT 抗过拟合**：同 49K 数据，SFT 把 Qwen 的 ODINW-13 拉低（37.0→35.0）、通用 QA 掉分（GQA 58.8→53.5）；规则 RL 两头不掉甚至涨（GQA 61.0）——完成级奖励只惩罚结果，不像 token 级 SFT 监督那样硬扭全部分布（[[vision-r1-paper]] §4.2–4.3）
- **组内奖励趋同是定位任务 GRPO 的特有陷阱**：高 IoU 拿不满 → 组内差距塌缩 → 优势信号消失；解法是渐进收紧（低值清零 / 高值给满 + 阈值随训练上调），且切换时机须匹配模型能力（强模型中途收紧、弱模型不切）（[[rule-based-reward]]）
- **评价信号的谱系成形**：人工偏好数据 → 学习式 RM（[[reward-model]]，绝对分不可辨识）→ 规则奖励（可审计、有绝对意义，但仅限可验证任务）；两路线在 MPO 里汇合——规则信号当偏好对过滤器（[[mixed-preference-optimization]]）

## 未解决的疑问

- HuggingFace `tokenizers`（Rust）与纯 Python / [[tiktoken]] 在工程上的具体差异？来源 1 仅提及 10–100×
- 中文语料上端到端训练 BPE 的成本与词表效率：chunk 长 L 暴增已从正则行为推知（[[pretokenization-cjk-and-mixed-text]]），但缺实测基准
- SentencePiece（码点级 + byte fallback）与 byte-level BPE 的取舍——[[qwen-tokenization-note]] 仅一笔带过
- `<|im_start|>` / `<|im_end|>` 背后的 ChatML 模板在训练与推理时如何注入
- Ostrakon-VL / ShopBench 承诺开源但尚未放出：VNR/VIF 会不会被社区采纳为基准设计惯例？MultiImg（49.6）短板是数据不足还是方法缺陷？
- MPO vs GRPO 已有阵营对照（[[vision-r1]]：GRPO + 规则奖励在 49K 样本上大幅提升定位），但仍缺同任务同数据的头对头；GRPO 在线组采样在 LVLM 上的实际成本（时间 / 显存）也未量化——两篇论文各说各的效率故事
- Skywork-VL-Reward 的风格偏置（惩罚冗长自校正、偏好简短回答）对 QUAD 过滤与 OCL 分层的实际影响未量化——Ostrakon 论文未讨论，需要实验才能闭合（[[skywork-vl-reward]]）
- Vision-R1 只做定位类任务；规则奖励路线能否延展到需要中间推理的多模态任务（数学 VQA、OCR 推理）？论文未触及，R1 系后续工作待收录验证

## 相关来源

- [[building-a-fast-bpe-tokenizer-from-scratch]]
- [[qwen-tokenization-note]]
- [[ostrakon-vl-paper]]
- [[skywork-vl-reward-paper]]
- [[vision-r1-paper]]
