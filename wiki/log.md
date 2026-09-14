---
type: log
title: 日志
created: 2026-09-12
updated: 2026-09-12
---

# 日志

> Append-only 操作日志，新条目追加到文件末尾。
> 条目格式：`## [YYYY-MM-DD] <ingest|query|lint|init> | <标题>`
> 查看最近条目：`grep "^## \[" wiki/log.md | tail -5`

## [2026-09-12] init | 初始化 LLM-Wiki

- 按 Karpathy 的 llm-wiki 模式建立三层架构：`raw/`（不可变来源）、`wiki/`（LLM 维护）、`AGENTS.md`（规范层）
- 创建 wiki 骨架：index / log / overview 与 sources、entities、concepts、analyses 四类目录
- 初始化 git 仓库

## [2026-09-12] ingest | Building a Fast BPE Tokenizer from Scratch

- 新增来源页：[[building-a-fast-bpe-tokenizer-from-scratch]]
- 新增概念页：[[bpe-tokenization]]、[[pretokenization]]
- 新增实体页：[[jun-yu-tan]]、[[stanford-cs336]]、[[tinystories]]
- 更新 [[overview]]（首条主线与关键结论：BPE 训练五级优化 ~230×）、[[index]]

## [2026-09-12] query | 预切分如何处理中文与中英混合文本

- 新增分析页：[[pretokenization-cjk-and-mixed-text]]（本地实验 + Qwen2.5/cl100k 真实正则核验）
- 核心结论：`\p{L}` 不分文字系统，汉字串整块成 chunk、无空格混合文本跨文字系统同块；GPT-2/cl100k/Qwen2.5 均无 CJK 专门规则；中文 chunk 长度 L 暴增，恶化源文 O(W×L²·logP) 复杂度
- 排除一条网络错误说法（「Qwen 按单字预切分汉字」，与 tokenizer.json 及官方 note 矛盾）
- 更新 [[pretokenization]]（中文小节 + 交叉链接）、[[index]]；候选来源 3 条记入分析页

## [2026-09-12] ingest | Qwen 官方 Tokenization Note

- 新增来源页：[[qwen-tokenization-note]]
- 新增概念页：[[special-tokens]]、[[vocabulary-expansion]]
- 新增实体页：[[qwen]]、[[tiktoken]]
- 更新 [[bpe-tokenization]]（纯分布特性、token 截断字符，sources 2）、[[pretokenization]]（词表扩展约束，sources 2）、[[pretokenization-cjk-and-mixed-text]]（外部引用改为链入 source 页，补 token 截断字符证据）、[[overview]]（主线扩为算法/工程/中文三条）、[[index]]
- 关键收获：BPE 无语言学知识（上下文相关切分、跨码点 merge 风险）；特殊 token 注入防护与默认行为变更史；词表扩展需中间 merge 链 + 微调
- 与上轮分析衔接：note 中 `夸张的 比喻手法` 示例成为预切分约束的直接证据；「Qwen 按单字切分汉字」的网传说法维持否定

## [2026-09-14] ingest | Ostrakon-VL：面向餐饮零售的领域多模态大模型

- 新增来源页：[[ostrakon-vl-paper]]（arXiv 2601.21342，淘宝闪购/阿里）
- 新增实体页：[[ostrakon-vl]]（模型：主结果表、三段训练、通用回退、配方）、[[shopbench]]（基准：构成、域区隔、VNR/VIF）
- 新增概念页：[[quad-data-curation]]（四阶段数据蒸馏 20.4× 反升 2.5 分）、[[mixed-preference-optimization]]（DPO+BCO+SFT）、[[domain-specific-mllm]]（领域化方法论与各领域成型栈，按用户确认新增）
- 更新 [[qwen]]（第二切面：Qwen3-VL-8B 作为领域微调基座，sources 2）、[[overview]]（新增「多模态领域化」主线）、[[index]]
- 图表 5 张入 raw/assets（docling scale 4 导出：fig1 框架、fig2 分类树、fig4 t-SNE、fig5/6 重分布）；同日用户纠正图表策略——不再强制 PDF 原生位图，统一 docling 导出
- 关键收获：数据质量 > 数量（69.25M→3.40M，+2.5）；8B 领域专化域内反超 235B 通用；VNR/VIF 把 Multimodal Gain 拆成「真视觉必要」与「视觉干扰」；代价是通用基准 -5.7
- 遗留疑问记入 [[overview]]：开源未放出、MPO vs GRPO 缺独立对照、MultiImg 短板归因

## [2026-09-14] query | QUAD 的指令从何而来

- 查询 §3.2 数据合成细节并沉淀进 [[quad-data-curation]]：原始视觉流来自监管巡检/监控/随手拍；人的参与在语义锚（样题/caption）与阈值审计，非逐条标注；盲答 ā = Gθ(q) 同源
- 补附录表 8 原始池构成：Kitchen 40M 占 VQA 池过半，源头偏斜即重分布的对象
- 标注论文空白：Gθ 具体模型与 prompt 模板未公开

## [2026-09-14] query | QUAD 质量过滤的奖励模型来历

- 确认：论文未自训奖励模型，原样取用开源 Skywork-VL-Reward（arXiv 2505.07263，昆仑万维）；域适配靠阈值校准而非模型微调
- 新增实体页：[[skywork-vl-reward]]（Qwen2.5-VL-7B 基座 + 奖励头；约 19 万偏好对两阶段排序训练；VL-RewardBench 73.1% 超 GPT-4o；外部网络来源，未入 raw/）
- 补充发现：Rϕ 不止用于 QUAD 两处过滤，还用于 OCL 难度投票，是全管线唯一打分权威
- 指出未闭合缺口：通用域评委 × FSRS 域数据的错配未讨论
- 更新 [[quad-data-curation]]（首提处挂链）、[[index]]

## [2026-09-14] query | 奖励模型的输出形式

- 确认：$R_\phi$ 输出单个无界标量（LM 预测头替换为末 token 隐藏态上的奖励头，无压缩），仅相对序有意义、绝对值未校准
- 沉淀进 [[skywork-vl-reward]]：绝对分值无意义 ↔ QUAD 三处全用差值 / 阈值比较的因果关联

## [2026-09-14] query | 「基座模型」术语辨析

- 澄清：QUAD 第二阶段的「基座」= 微调前起点的 Qwen3-VL-8B 本尊；它已是阿里对齐过的成品通用模型，「基座」指未做本项目领域微调，非裸预训练
- 辨析四种模型角色防混淆：基座 / 生成器 Gθ（未透露型号）/ 奖励模型 Rϕ（Skywork-VL-Reward）/ OCL 参考 MLLM {M_k}（K 个现成模型）
- 更新 [[quad-data-curation]] 基座参考过滤条目加注

## [2026-09-14] query | SemDeDup 与 k-center 机制辨析

- 拆解去重阶段引用的两个方法：SemDeDup = 聚簇后剪「离质心最近的 ε 比例」语义重复（Abbas et al., arXiv:2303.09540）；k-center = 最远点采样挑多样性代表
- QUAD 组合 = k-means 定簇 + 簇内 k-center 式选代表；超参未披露
- 新发现（表 8 分项）：MultiImg 在去重阶段 7.5× 压缩为冗余之最，Video 未动
- 更新 [[quad-data-curation]] 新增「去重阶段的机制注解」小节

## [2026-09-14] query | 能力分类器微调方式与 19 类分类学

- 视觉直读 fig6（raw/assets）得完整 19 类清单及前后占比；正文只说 Qwen3-8B + 约 7,000 众包种子微调、仅看问题文本不看图
- 观察沉淀：分类学是通用多模态能力体系（含域外类目），非 FSRS 专属；π(c) 参考初步模型短板设定
- 更新 [[quad-data-curation]] 新增「能力分类器与 19 类能力分类学」小节

## [2026-09-14] ingest | Skywork-VL Reward（论文）

- 收录 raw/Skywork-VL Reward.pdf（arXiv:2505.07263，昆仑万维，2025-05）→ 新建 [[skywork-vl-reward-paper]]；docling 转换 + 通读，此前网络来源建的 [[skywork-vl-reward]] 实体页全部记载与原文吻合
- 新增细节：三阶段数据清洗（surrogate RM 打分回路 + GPT-4o 重生成 + R1V/DeepSeek R1 造推理回答）、RM 两轴分类学、排序损失只学相对序 ⇒「等质对」主动剔除、类目强弱（general 66.0 弱 / 幻觉 80.0 最佳）、风格偏好（惩罚冗长自校正）
- 新建概念页 [[reward-model]]（判别/生成/隐式 × ORM/PRM；排序损失固有性质；本 wiki 内 QUAD / MPO / Ostrakon 三线交汇点）
- 更新 [[skywork-vl-reward]]：来源落地 raw + 补新细节 + 两条分析（打分器递归性、风格偏置 = 域错配之外第二隐患）；MPO 段以论文 §4.7 数据为准（MathVista 69.2→73.5），网络来的 Res-38B 数字降为外部注
- 更新 [[mixed-preference-optimization]]（§4.7 换 RM 对照表）、[[quad-data-curation]]（打分器方法论同源注记）
- 图片：论文图 1（17 域训练数据分布）→ raw/assets/skywork-vl-reward-fig1-data-distribution.png；case study 截图未收（正文已全录）
- 更新 [[index]]、[[overview]]（主线二补奖励模型环节，sources 3→4，新增未解决疑问：风格偏置的量化影响）
