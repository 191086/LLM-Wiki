---
type: log
title: 日志
created: 2026-09-12
updated: 2026-09-16
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

## [2026-09-14] query | Bradley-Terry 成对损失

- 讲清数学形式：$P(y^+\succ y^-)=\sigma(s^+-s^-)$，损失 = 负对数似然 $-\log\sigma(s^+-s^-)$；平移不变 ⇒ 绝对分不可辨识；梯度 $-\sigma(-\text{margin})$ 随排对饱和
- 沉淀进 [[reward-model]]：「排序损失的固有性质」小节扩为数学形式 + 三条后果 + MPO/BCO 互补推论（分析）

## [2026-09-14] lint | 重建 BPE 概念页：结构化分节与细节扩充

- 用户反馈 [[bpe-tokenization]] 不够结构化、不够详细 → 全页重写为 10 节：词表三难定位 / 历史 / 训练-编码双阶段总览 / 训练（伪代码 + 手工示例：low×5 lowest×2 newest×6 widest×3，含 tie-break 演示）/ 编码（rank 堆实现）/ byte-level 四特性 / 纯分布三后果 / 复杂度五级阶梯（V1–V5 表 + V4 上界 ΣA≤W×L + V5 堆压缩细节 + 230× 实测）/ WordPiece-Unigram-SentencePiece 对照（标注 wiki 外一般性知识待补来源）/ 实践要点速查
- 数学公式改 $…$ / $$…$$；伪代码 Python 化；全部实测数字锚定 [[building-a-fast-bpe-tokenizer-from-scratch]]，byte-level 特性锚定 [[qwen-tokenization-note]]
- 更新 [[index]] 一行摘要

## [2026-09-14] lint | 概念页结构化模板写入规范

- 从 [[bpe-tokenization]] 重建提炼出概念页模板，写入 AGENTS.md「页面模板」节：原「实体/概念页」合并模板拆开——实体页保持简版；概念页新增结构化模板（问题定位 / 历史☆ / 核心机制＝总览表＋步骤＋伪代码＋手工走查示例 / 关键性质与陷阱 / 工程复杂度☆ / 变体近邻对照☆ / 实践速查☆，来源与相关必备）
- 附写作原则五条：编号小节可「§N」引用；论断句末锚定来源页、实测数字须来源页可查；wiki 外一般性知识显式标注；表格只放对照型内容；旧页升级属 lint
- ☆ 模块按需取舍，简单概念最小集 = 定义 / 核心机制 / 关键性质 / 来源 / 相关
- **修订（同日，用户反馈）**：固定章节太具体，不同知识点结构应不同 → 模板改为「定义 + 自由编号分节 + 来源/相关」骨架（算法/理论类各举组织方式为例），写作原则转质量导向：让人看懂是第一标准（示例 / 手工走查 / 伪代码 / 公式 / 术语首现即解释）、宁详勿简但不凑格式；[[bpe-tokenization]] 降为参考范例而非必套结构

## [2026-09-14] lint | BPE 页补解码伪代码

- 查询「BPE 编解码过程」后按用户要求补缺：§5 改题「编码与解码（推理）算法」，新增 decode 伪代码（id→字节串拼接 + 整体一次 UTF-8 解码）
- 补「解码无需逆推 merge」的机制解释（合并只替换相邻符号为其字节拼接，字节内容与顺序恒不变）+ 流式截断尾巴的 `errors="ignore"/"replace"` 处理（细节仍见 §6.3，锚定 [[qwen-tokenization-note]]）；解码复杂度 $O(N)$
- 更新 [[index]] 一行摘要（训练/编码双阶段 → 训练/编码/解码全流程）

## [2026-09-14] lint | 修复 BPE 训练伪代码：特殊 token 的写入位置

- 用户指出 `vocab += special_tokens` 有误——dict 与 `list[str]` 不可相加（TypeError），且把特殊 token 插在 merge 结果之前与主流词表布局不符 → 改为 merge 循环结束后 `vocab[len(vocab)] = s.encode("utf-8")` 追加
- §4.1 新增步骤 5 及说明：特殊 token 不经预切分、不参与 merge，占据词表末尾最高 id；锚定 Qwen 词表布局（151,643 regular 之后才是 208 control，扩展区自 151,851，[[qwen-tokenization-note]]），类型系统另见 [[special-tokens]]

## [2026-09-15] ingest | Vision-R1：免 RM 的视觉规则强化学习

- 通读 raw/Vision-R1.pdf（arXiv:2503.06749，CASIA，2025-03）：docling 转文字 + 视觉直读补齐 Eq.1–8（docling 未解出公式）与三张图；仅 Fig 2 框架图入 raw/assets（vision-r1-fig2-framework.png），其余留 /tmp 工作区
- 新建 4 页：[[vision-r1-paper]]（来源）、[[vision-r1]]（实体）、[[grpo]]（概念：组相对优势 + 4 完成手工走查 + 无 critic / 在线采样代价，clip 项标注为 wiki 外知识）、[[rule-based-reward]]（概念：三路准则奖励对应三大失败模式 + box-only 匹配 + reward hacking 与渐进收紧 + 与学习式 RM 对照表）
- 更新 3 页：[[reward-model]]（新增「系外成员：规则奖励」节）、[[mixed-preference-optimization]]（「与 GRPO 的取舍」补 GRPO 阵营对照数据点）、[[qwen]]（新增切面三：Qwen2.5-VL-7B 同一模型当 RM 基座与 RL 靶子）
- [[overview]]：主线二扩出「后训练强化学习（免 RM）」环节；关键结论 +4；MPO vs GRPO 疑问改写（有阵营对照、仍缺头对头与成本量化）；新增疑问：规则奖励能否延展到需中间推理的多模态任务
- [[index]] 同步（来源 4→5、实体 +1、概念 +2）

## [2026-09-15] query | Vision-R1 数据集构建细节

- 查询「数据集如何构建」→ 细节补足进 [[vision-r1-paper]]：难例判定（COCO 单图 >10 实例）、难 / 易样本各采 1/3（原文表述含糊已标注）、VG / REC 分池来源与数量（ODINW 5K + V3Det 4K；RefCOCO 5K + Visual Genome 多目标 5K）、五套指令模板跟随各模型 SFT 格式（与 dual format reward 的 template check 呼应）
- index 无需变动（一行摘要仍准确）

## [2026-09-15] lint | 按概念页标准全库充实：手工走查与结构化（阶段 1–5）

- 起因：用户反馈「内容不够结构化、没把知识点讲清楚」。逐页体检（18 个内容页对照 [[bpe-tokenization]] 范例）确认断层在 09-12 老批次：无编号分节、手工走查全库仅 bpe/grpo 有、术语不解释即用。分五阶段整改，每阶段经用户审阅
- **阶段 1（老批次概念页重写）**：[[pretokenization]]（+正则手工走查：中英混合 / 弯引号 / 空白分工，本机 regex 实测）、[[special-tokens]]（修正一处含混：tiktoken 本体默认 = 抛错，Qwen 封装新默认 = 全解析——两开关四行为分列，本机 cl100k 实测）、[[vocabulary-expansion]]（官方 add_merges 日志端到端走查：六条新 merge 逐轮还原、base64 产物本机核算）
- **阶段 2（中间态升级 + 新页）**：[[reward-model]]（BT 损失数值走查：三对分数手算 loss 与梯度）、[[mixed-preference-optimization]]（三项损失落到公式 + DPO/BCO 同 margin 数值对比走查；BCO 出处核补 Jung et al. arXiv:2404.04656）、[[domain-specific-mllm]]（目录页→讲解页，各领域栈换 PDF §2 原述锚定）；**新建 [[dpo]]**（重参数化推导完整机制页，机制部分为 wiki 外知识显式标注，待 DPO 论文入 raw 后补锚）
- **阶段 3（实体页升级）**：[[tiktoken]]（可运行小例 + 所有格量词 `++` 机制展开）、[[vision-r1]]（一条样本三路奖励走查，标注构造演示）、[[qwen]]（家族谱系 wiki 外补注）、[[tinystories]] / [[stanford-cs336]]（背景补注：Eldan & Li 2023 arXiv:2305.07759 / 课程页 2026-09-15 检索）、[[jun-yu-tan]] 补博文定位句
- **阶段 4（达标页小修）**：[[rule-based-reward]]（IoU/GT/VG/REC/匈牙利匹配首现 gloss + 式 6 口径精确化：分母 M 跑遍全部预测，垃圾框直接稀释 precision——论文行文与公式的张力显式标注）、[[quad-data-curation]]（编号分节 + 19 类占比改双列表格）、[[skywork-vl-reward]]（VL-RewardBench 补注：Li et al. 2024 arXiv:2411.17451）、[[ostrakon-vl]]（FSRS 展开 + Dense/MoE gloss）、[[shopbench]]（t-SNE gloss）
- **阶段 5（簿记）**：[[index]] 登记 dpo、刷新重写页摘要；修正本页 frontmatter `updated`（原停留在 09-12）
- 核验手段：本机 `uv run`（regex / tiktoken / pypdf 抽原文）+ web 检索（BCO / VL-RewardBench / TinyStories / CS336）；全部新增数字可复算

## [2026-09-15] lint | 写页自查清单写入规范（lint 收官的机制化）

- 概念页模板旁新增「写页自查清单」六条（定义三要素 / 编号分节 / 手工走查 / 术语首现即释 / 锚定与数字核验 / 表格纪律），条目源自本次全库 lint 的通病
- Ingest 工作流第 4 步加自查门：概念页全过清单才记 log——log 条目即质量签名
- Lint 工作流检查项补「概念页达标情况」：大 ingest 后对照清单扫全库
- 动因：09-12 老批次退化证明「结构化 / 讲清楚」这类形容词标准无法在写页时核验，必须翻译成可打勾的硬条件，并放进每次会话都加载的规范层

## [2026-09-15] lint | AGENTS.md 概念页小节去重整理

- 写作原则 8 条中 6 条与写页自查清单重叠（机制落地与术语解释 / 编号分节 / 锚定与数字可查 / wiki 外标注 / 表格纪律 / lint 归属）→ 原则收缩为 2 条（宁详勿简、范例指针），可核验硬条件单一源于清单，消除双头维护的漂移风险
- 概念页小节重排：总纲 → 模板 → 清单（写时执行）→ 原则（态度取向）
- 顺带消除第三处重复：行内 / 行间公式的写法约定只保留在「用户偏好」，不再于写作原则中复述
- 顶层结构与其余各节未动

## [2026-09-15] lint | 页面模板优化：实体页自包含、来源页定位、分析页骨架

- 实体页模板修订：正文解绑「## 详情」（自由命名小节——实践已分叉，tiktoken / tinystories 等升级页早已不用该骨架）；定义行改为「是什么 + 在本库语境中的角色或关联」；附写作要点三条：自包含不许薄成指针页（09-12 批次实体页退化的模板级原因）、实例优先于罗列、纪律同概念页但不强制走查
- 来源页模板：关键要点约定带原文定位（§N / 表号 / 公式号）——本次 lint 的口径回查（式 6、PDF §2 等）全靠既有 source 页的定位习惯
- 分析页模板：固化推荐骨架（起因 → 一句话结论先行 → 正文 → 候选来源），取自 [[pretokenization-cjk-and-mixed-text]] 的实践
- 概念页模板未动（09-14 建、09-15 已加清单并去重）
- 补遗（用户指出）：分析页此前只有一句话描述、没有模板块，已补齐正式模板（起因引言 / 一句话结论 / 正文 / 候选来源 / 来源 / 相关），与来源、实体、概念三类页面格式对齐

## [2026-09-15] lint | 重写分析页：可复现实验与编号分节

- 起因：用户指出「现有的分析页撰写的结构和内容并不是很好」。诊断为：小节无编号不自包含；关键数字（L 暴增的 21 chunk/4.1B vs 6 chunk/14.5B）未附实验句子、不可复现；解读性内容与实验事实混排未标注；与重写后的 [[pretokenization]] §3/§5 分工未收敛
- [[pretokenization-cjk-and-mixed-text]] 重写：七个编号小节；L 实验改用显式双语句对本机复算（中文 8 chunk / 平均 15.4B / 最长 39B vs English 35 chunk / 4.3B / 10B，句对原文入页，$L^2$ 放大 ≈12.8×），旧不可复现数字显式废弃并注明；GPT-2 表 5 行与 cl100k 对照行全部 2026-09-15 本机复验；Qwen2.5 行沿用 09-12 核验并注明（HF 当日网络不可达）；「为什么不做分词」独立成节并标注为解读；与概念页的分工写明（本文 = 概念页 §5 的实证展开）
- 同步 [[pretokenization]] §5 引用的数字（14.5B/4.1B → 15.4B/4.3B + 链接）；交叉补链 [[vocabulary-expansion]] §2

## [2026-09-15] chore | 模板抽到独立目录 templates/

- 用户要求：模板放专门目录、各自成文件。新建 `templates/source.md` / `entity.md` / `concept.md` / `analysis.md`（内容 = 原内嵌代码块原文）
- AGENTS.md 页面模板区改为指针（各小节标注「模板：templates/…」），写作要点、自查清单、写作原则仍留在规范层（保住每次会话必加载的执法点）；目录结构树补 `templates/`
- 一次性配置：`.obsidian/templates.json` → Templates 核心插件模板目录指向 `templates/`（插件本已启用），已记入 AGENTS.md「Obsidian 设置」节

## [2026-09-15] chore | AGENTS.md 纯化为通用规范

按用户四条要求清理 AGENTS.md：①删除头部「模式来源（Karpathy LLM Wiki）」行；②全文去除具体日期（概念页 / 实体页标题、自查清单、用户偏好、Obsidian 设置各处）；③模板改为 Obsidian 引用式链接（[[templates/source|source]] 等）；④不再锚定具体 wiki 页面——删 [[bpe-tokenization]] 范例链接（改为「从库中挑最成熟概念页作参考」的准则表述）、删分析页「来自 [[pretokenization-cjk-and-mixed-text]] 的实践」出处。规范层从此与具体库内容解耦，日期与出处只留在 log
- 续（同日）：按用户要求把写作准则收拢到「页面模板」小节末尾的统一「写作原则」（七条，全页型生效）；概念页小节只留自查清单，实体页「写作要点」、概念页「写作原则」并入统一块，四页型小节各留一句定位

## [2026-09-15] ingest | Direct Preference Optimization（DPO 论文，arXiv:2305.18290）

- raw/ 收录 Direct Preference Optimization.pdf，按库内惯例入库前改短名 DPO.pdf；docling 全文转换 + PDF 原页视觉核验全部公式（docling 对本篇公式全丢，式 1–7 与梯度式逐条对图校验）
- 新建 [[dpo-paper]] 来源页：三阶段 RLHF 预备（式 1–3）、重参数化推导（式 4–7）、梯度动态加权、§5 理论（等价类 / Theorem 1 / PPO 不稳诊断）、三任务实验与人类研究、附录 B 实现默认值（β=0.1、RMSprop 1e-6）；嵌 Figure 2
- 重写 [[dpo]] 概念页：撤销「wiki 外知识」标注、逐式锚定原文；新增 §4 梯度一节（手工走查：排错加权 σ(0.05)=0.513 vs 排对 σ(−0.4)=0.401，本机复算）与 §6 实验证据；嵌 Figure 1；失败模式显式标注为 wiki 外后续文献
- 新建 [[rlhf]] 概念页（此前全库引用 RLHF 却无独立页面）：三阶段管线、形化奖励走查（E_π[shaped] = E[r_φ] − βKL 严格相等 0.928，本机复算；逐样本符号可反向）、难训四因、PPO/DPO/MPO/GRPO 路线谱系表
- 更新关联页：[[reward-model]]（BT 损失锚定式 1–2、隐式 RM 锚定 §5.1、来源 3→4）；[[mixed-preference-optimization]]（偏好项改库内来源，BCO 仍 wiki 外，来源 3→4）；[[grpo]]（KL 惩罚与 RLHF 同源，来源 1→2）；[[rule-based-reward]]（RLHF/DPO 接线）
- 登记：overview（来源 5→6、「偏好对齐的理论基底」入主线二、关键结论 +2、开放问题 +1）；index（新来源行、rlhf 新页行、dpo 摘要刷新）；图 2 张按需入 raw/assets（dpo-fig1 管线对比、dpo-fig2 前沿与胜率）

## [2026-09-16] ingest | Training LMs to Follow Instructions with Human Feedback（InstructGPT 论文，arXiv:2203.02155v1）

- raw/ 收录论文全文 PDF（68 页，arXiv v1，2022-03-04），入库前按惯例改短名 instructgpt.pdf；docling 全文转换（公式未解码，式 1 RM 损失与式 2 PPO-ptx 目标经 PDF 原页渲染视觉核验）；收录注记：v1 图号与广泛引用的 v2 不同（v1 Figure 2 = 三步管线图），且无 v2 的 RM scaling 分析
- 新建 [[instructgpt-paper]] 来源页：3H 对齐操作化、三步方法与数据规模（SFT 13k / RM 33k / PPO 31k prompt，K=4–9 排序展开 C(K,2) 对）、标注员 40 人与人-人一致率 72.6%、SFT 按 RM 分选模、6B RM 与整提示打包训练、bias 归一化、PPO 逐 token KL（β=0.02）与 PPO-ptx（γ=27.8）、主结果（1.3B>175B、85±3%）、FLAN/T0 对照（真实使用分布 ≠ 公共基准）、真实性/毒性/偏见、对齐税、成本账 ≈2%；嵌 Figure 1 胜率图
- 新建 [[instructgpt]] 实体页（模型本体：三档规模、配方速览表、代表性数字、部署形态；「ChatGPT 前身」按规范标注为 wiki 外补注 + 候选来源）
- 重写 [[rlhf]] 概念页：主锚从 [[dpo-paper]] 二手引文升级为 [[instructgpt-paper]] 一手来源——定义行重写、§1 补 misaligned/3H 框架、§2 管线补 InstructGPT 工程细节（RM 分选模、6B RM 选型、式 1 与打包训练、逐 token KL、PPO-ptx）、新增 §3 实证节（小模型赢大模型、held-out 标注员泛化、真实性/毒性/偏见、对齐税、成本账、泛化边界）；形化奖励走查原样保留（0.928 期望恒等本机复核一致）并补值函数初始化锚点；难训四因补 175B RM 不稳定实例；路线谱系表与各节编号保持（dpo.md 的 [[rlhf]] §2 入链不断）；来源 2→3；嵌管线图与 held-out 胜率图
- 更新 [[reward-model]]：§4.1 补 InstructGPT 式 1 锚定 + 整提示打包 + bias 归一化（平移不变的规范化应用，§4.3 推论 1 同步补引）；§5 补 RM 精度参照（72.4% / 69.6% vs 人-人 72.6%，上限被人类一致性封顶）；来源 4→5
- 登记：overview（来源 6→7、主线二 +「RLHF 范式的一手锚点」、关键结论 +3、开放问题 +1）；index（新来源/实体行、rlhf 摘要刷新）；图 3 张按需入 raw/assets（instructgpt-rlhf-pipeline、instructgpt-winrate-vs-gpt3、instructgpt-winrate-heldout）

## [2026-09-16] ingest | Proximal Policy Optimization Algorithms（PPO 论文，arXiv:1707.06347v2）

- raw/ 收录论文 PDF（12 页，arXiv v2，2017-08-28），入库前按惯例改短名 ppo.pdf；docling 全文转换（公式未解码），式 1–12 与 Algorithm 1 全部经 PDF 原页渲染视觉核验；考订出原文式 10/11 末项指数排印笔误（$T-t+1$ 应为 $T-1-t$；令 λ=1 展开求和与式 10 递缩相消严格一致以验证），wiki 按修正后的标准式照录并显式标注
- 新建 [[ppo-paper]] 来源页：§1 三前驱动机、§3 截断替代目标（式 6–7）、§4 自适应 KL（式 8 与 β 乘性规则）、§5 完整算法与截断 GAE（式 9–12）、§6 消融（Table 1：clip ε=0.2 拿 0.82 vs 无约束 −0.39）/ MuJoCo 对比 / Atari（Table 2：30/18/1）；嵌 Figure 2
- 新建 [[ppo]] 概念页（对照自查清单逐条过）：clip 四分支手工走查（Â>0 / Â<0 × 越界变好 / 变差，数值代入）、$L^{PG} \to L^{CPI} \to L^{CLIP}$ 演进、式 9 组合目标与截断 GAE（含笔误注记与验证）、Algorithm 1 伪代码、自适应 KL 变体、实验证据、§6「clip 不直接约束参数距离」wiki 外补注（候选来源：Engstrom 2020、Huang 2022）；嵌 Figure 1
- 更新关联页：[[rlhf]]（定义行 / §2 / §4 / 谱系表 PPO 接线，§6 语境补条）、[[grpo]]（§1 critic 对照接线、clip 注记接 [[ppo]] §2）、[[dpo]]（§1 PPO 不稳表述补精确锚点）、[[instructgpt]]（配方表）、[[instructgpt-paper]] / [[dpo-paper]]（首提处接线）
- 登记：overview（来源 7→8、主线二 +「RL 引擎本尊」、关键结论 +2、开放问题 +1）；index（新来源 / 概念行）；图 2 张按需入 raw/assets（ppo-fig1-clip-surrogate-single-term、ppo-fig2-interpolation-lower-bound）

## [2026-09-16] ingest | Trust Region Policy Optimization（TRPO 论文，arXiv:1502.05477v5）

- raw/ 收录论文 PDF（16 页，arXiv v5，2017-04-20，ICML 2015，UC Berkeley），视觉直读全文（16 页 3× 渲染通读 + 公式区局部放大）；Table 1 数字整页通读曾误读（TRPO 行多个数字），6–10× 局部放大后以与 DQN/Human 公开值交叉印证的读数为准，wiki 侧全部按放大版转录；考订原文一处自不一致：§8.1 称 Walker 状态维数 18、附录 E Table 2 印 20（已在来源页注记）
- 新建 [[trpo-paper]] 来源页：§1 三族动机、§2 性能差分解（式 1–4）、§3 单调改进下界（Theorem 1 式 8–9、Algorithm 1 MM 视角）、§4 惩罚→硬约束（式 11–12）、§5 single path / vine 采样（式 13–16）、§6+附录 C 实用算法（Fisher-vector product、CG k=10、10% 子采样、回溯线搜索）、§7 自然梯度统一（式 17–18）、§8 MuJoCo（Table 2）与 Atari（Table 1 转录、Table 3）；嵌 Figure 5
- 新建 [[trpo]] 概念页（对照自查清单逐条过）：理论骨架三节（性能差分解 → CPI/Theorem 1 → MM）、α-coupling 微走查（π=(0.8,0.2)/π̃=(0.7,0.3)、共享随机数分歧概率 0.1、Ā=0.125=α·(A₁−A₀) 两路一致、Lemma 2 验证；优势 π 下零均值前提显式演示）、三次近似、2 状态 MDP 数值走查（式 1 分解 0.314459 严格相等、替代目标高估 0.0372、下界松弛 ~13 vs 实际误差、理论惩罚系数 γ=0.99 达 39,600 → 硬约束的数字理由；全部本机复算）、single path vs vine 对照表、MuJoCo/Atari 实验证据（Table 1 六行转录）、TRPO/PPO 六行对照表；嵌 Figure 1、Figure 4
- 更新关联页：[[ppo]]（§1 TRPO 接线、§7 补「TRPO 直接后继」条、相关行）、[[ppo-paper]]（一句话总结与要点 TRPO 接线、值得追踪补行）、[[rlhf]]（§6 补引擎谱系条）
- 登记：overview（来源 8→9、当前状态改四重锚定、主线二 +「RL 引擎的理论前身」、关键结论 +1、开放问题 +1、相关来源 +1）；index（新来源 / 概念行）；图 3 张按需入 raw/assets（trpo-fig1-single-path-vs-vine、trpo-fig4-mujoco-learning-curves、trpo-fig5-atari-learning-curves）

## [2026-09-16] query | TRPO 式 1（性能差分解）答疑

- 用户反馈 [[trpo]] §2.1 式 1 不可懂；答疑拆解三层：①轨迹测度（$\tau\sim\tilde\pi$）与评分标准（$A_\pi$）的分工——新策略走路、旧策略打分；②严格等式的来源 = telescoping：$A$ 拆成 $r+\gamma V_\pi(s')-V_\pi(s)$ 后相邻 $V$ 项同测度平移相消（附录 A 式 20–24）；③一般骨架——$V$ 换任意 $\bar V$ 仍收敛到 $\eta(\text{行为策略}) - \mathbb{E}[\bar V(s_0)]$（势函数塑形同族），并给退化情形对照（评分与轨迹同策略 → 总和 0）与一步改进特例（→ 一步策略改进定理，CPI 的 $O(\alpha^2)$ 由此起）
- 澄清块并入 [[trpo]] §2.1（式 1 之后、式 3 之前）；小体量分析点就近并入、不另开 analyses 页

## [2026-09-16] ingest | RoFormer: Enhanced Transformer with Rotary Position Embedding（RoPE 论文，arXiv:2104.09864v5）

- raw/ 收录 RoPE.pdf（14 页，arXiv v5，2023-11-08，追一科技，Su Jianlin 一作）；视觉直读全文（14 页渲染通读 + 5 张表与式 15 / 式 34 高倍局部放大转录核验）；docling 导出 3 图，按需全取入 raw/assets
- 考订四处：式 32 第一行印作 $\boldsymbol{W}_q\boldsymbol{x}_n$，下标应为 $\boldsymbol{x}_m$（与式 1 矛盾）；$\theta_i$ 定义两处指标平移（式 15 的 $10000^{-2(i-1)/d}$ vs §3.3/§3.4.3 的 $10000^{-2i/d}$，等价）；Table 2 题注 "GLEU" 误排；§4.5.4 「净胜 WoBERT 1.5%」与 Table 5 算术不符（对 WoBERT 实为 +1.69，1.5 恰为自身 512→1024 增益 68.29→69.79）
- 新建 [[rope-paper]] 来源页：§2 两族综述（式 1–10）、§3.1 函数方程（式 11）、§3.2 旋转解（式 12–16）、§3.3 性质与线性注意力（式 17–19）、§3.4 理论（推导式 20–33 / 稀疏实现式 34 / Abel 衰减界式 35–37）、§4 五组实验（Table 1–5、Figure 3）、§4.5.5 局限自陈；嵌 Figure 3
- 新建 [[rope]] 概念页（对照自查清单逐条过）：§1–§8 编号分节；双手工走查本机 numpy 复算（2D：$\boldsymbol{q}^\top\boldsymbol{R}_3\boldsymbol{k} = (1,2)\cdot(-1,3) = 5$ 严格相等、平移不变 $(2,5)$ 仍 5.0、保范 $\sqrt5$；$d=4$：$(5,12)$ 与 $(100,107)$ 同差 7 内积逐位相等 $4.4047918119$）；术语首现 gloss；§7 后续谱系标注 wiki 外 + 候选来源（LLaMA / PI / YaRN / LongRoPE）；§8 七方案对照表；嵌 Figure 1、2
- 定位：库内首个架构层页面，overview 开主线三「Transformer 架构与长上下文」；无既有页面需接线（全库此前无 transformer / 注意力类内容，grep 核实）
- 登记：overview（来源 9→10、当前状态改三条主线、主线三小节、关键结论 +1、开放疑问 +2、相关来源 +1）；index（总览行刷新、新来源 / 概念行）；图 3 张按需入 raw/assets（rope-fig1-implementation、rope-fig2-long-term-decay、rope-fig3-pretraining-loss）

## [2026-09-16] lint | 删除手工走查硬性要求（用户指示）：规范修订与 rope 页瘦身

- 规范层四处去「走查」：AGENTS.md 写页自查清单删「至少一个手工走查」条（原第 3 条，余条重排为 5 条）；lint 扫库项「揪出无走查」改「揪出无实例」；写作原则实例枚举去「走查」、「概念页从严」句去「手工走查为硬性要求」；templates/concept.md 机制详解示例改「数值示例」——走查降为可选手段，不再是记 log 的门槛
- [[rope]] 删 §4 手工走查整节，后续章节重排（关键性质 §4 / 实验 §5 / 局限与后续 §6 / 方案对照 §7），页内交叉引用同步；走查验证数字（2D 内积 5.0、$d=4$ 平移不变 4.4047918119）留档于上方 ingest 条目
- 摘要去「双走查」表述并改节号：[[overview]]（主线三、关键结论）、[[index]]（rope 行）、[[rope-paper]]（值得追踪行）
