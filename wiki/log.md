---
type: log
title: 日志
created: 2026-09-12
updated: 2026-09-15
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
