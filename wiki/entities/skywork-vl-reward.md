---
type: entity
title: Skywork-VL-Reward（奖励模型）
created: 2026-09-14
updated: 2026-09-14
tags:
  - reward-model
  - mllm
  - data-curation
sources: 2
---

# Skywork-VL-Reward

**定义**：昆仑万维 Skywork 团队 2025 年 5 月发布的开源多模态奖励模型（7B，Qwen2.5-VL-7B-Instruct 基座 + 奖励头），发布时 VL-RewardBench 榜首；在 [[ostrakon-vl]] 论文中被**原样取用**为 [[quad-data-curation|QUAD]] 管线与课程学习的统一打分器 $R_\phi$，未做任何领域适配。

## 详情

### 模型本体（[[skywork-vl-reward-paper]]）

- **形态**：判别式 + 结果奖励（ORM），见 [[reward-model]] 分类学；输入（可选图，提问，候选回答）→ 输出单个**无界标量**
- **架构**：把 Qwen 的 token 预测头整个拆掉，在回答末 token 位置的最终隐藏态上接全连接奖励头直接回归分数，无 sigmoid / softmax 压缩
- **排序损失只学相对序**：$-\log\sigma(s^+-s^-)$，绝对分值**未经校准**、单独看无意义；论文因此**主动剔除「等质」偏好对**（判决过滤）——这正是 Ostrakon 三处用法全为差值 / 阈值比较、且阈值必须验证集 + 人工审计标定的机制根源
- **偏好数据**：约 19 万对（约 70% 含图），拼装自 LLaVA-Critic-113k、Skywork-Reward-Preference-80K-v0.2（纯文本）、RLAIF-V-Dataset，外加约 5 万条人工标注的自建推理对比数据（数学 35.4 / 物理 24.6 / 化学 20.2 / 生物 14.7 / 其他 5.1，%）
- **训练**：视觉编码器（ViT）冻结以保住预训练视觉能力，projector / 语言骨干 / 奖励头可训；两阶段成对排序损失，先多模态偏好（lr 10⁻⁵）再混纯文本偏好（lr 10⁻⁶），各 2 epoch，AdamW
- **成绩**（VL-RewardBench overall 73.1 / macro 69.0，超 GPT-4o 65.8/62.4 与 Gemini-2.0-flash-exp 68.8/64.5）：
  - 幻觉检测 **80.0 全场最佳**（GPT-4o 67.6、IXC-2.5-Reward-7B 65.3）
  - reasoning 61.0，与 10× 参数的 InternVL3-78B（64.5）相当
  - general 66.0 为**明显弱项**，落后 IXC-2.5-Reward-7B（80.3）逾 14 分
  - 文本 RewardBench 90.1：同规模多模态 RM 最佳（+2.0 vs IXC-2.5），接近纯文本专用 RM（QRM-Llama3.1-8B-v2 93.1）
- **风格偏好（case study）**：两案均答对，简洁推理版得 5.86 / 7.53，冗长自校正版分差显著乃至 −10.36，判词直指「wait 反复出现」「重复啰嗦」——它系统性惩罚 reasoner 长链路的典型文风

### 训练数据本身是三阶段清洗的产物（[[skywork-vl-reward-paper]] §3.2）

1. **过滤**：跨源去重 + 语义相似过滤 + 判决过滤（丢弃与 GPT-4o 判断矛盾或「等质」的模糊对）→ 约 20 万高置信对 → 训一个 **surrogate RM** 给全量数据打分
2. **按分修订**：chosen 被 surrogate 打低分 → GPT-4o 重生成替换；正反分差过小 → chosen 也重生成 → 保留 15 万
3. **推理风格回答**：Skywork R1V 直接生成 47.4%；InternVL 系列先写图述替代视觉输入 → DeepSeek R1 生成最终推理 52.6% → 最终约 19 万对

**递归性（分析，非论文原话）**：Ostrakon 拿这个「surrogate RM 打分回路清洗出来」的模型当 QUAD 全管线唯一裁判——reward-guided curation 是两家共享的方法论，一个造裁判、一个用裁判；「裁判有偏 → 训练数据有偏 → 下游清洗有偏」的传导链在现实中已经接通。

### 在 Ostrakon-VL 中的角色（[[ostrakon-vl-paper]]）

论文只一句「Skywork-VL-Reward in our implementation」带过，当作现成评分器：

- [[quad-data-curation|QUAD]] 质量过滤：打 $r_a$；视觉消融检验再给盲答打 $r_{\bar{a}}$，用差值隔离视觉贡献
- QUAD 基座参考过滤：给基座参考回答打 $r_{\tilde{a}}$，以奖励差 $\Delta r$ 估学习增益
- 离线课程学习 OCL：给 K 个参考 MLLM 的候选回答打 $r_{\hat{a}_k}$，投票估难度分层
- 可复现性的固定件：给定 $\{M_k\}$、$R_\phi$、阈值、$f$，管线全程确定（§4.2）

即它是**全管线唯一的打分权威**，三处过滤 / 分层共用同一个冻结 checkpoint。

### 为什么是它（分析，非论文原话）

- 7B 开源权重 → 约 9,500 万候选样本 × 每条 2–3 次打分，成本可控（闭源 API 不现实）
- 公开固定 checkpoint → 契合论文「可审计、可复现」的管线主张
- 发布时 VL-RewardBench 榜首 → 判别力有公信力
- Qwen 家族基座，与整条 Qwen 技术栈（生成器生态、微调基座）同源
- **隐患一（域错配）**：通用域评委给 FSRS 域数据打分，论文未讨论评委自身的域错配，仅靠阈值校准吸收；本篇 general 类目的弱项（66.0，落后 IXC-2.5 逾 14 分）为此再添一个注脚
- **隐患二（风格偏置）**：本篇 case study 显示它惩罚冗长自校正、偏好简短回答——用它给 FSRS 数据打分 / 排序，可能在 QUAD 阈值过滤与 OCL 难度分层上引入「偏好简短回答」的偏置。域错配之外的第二个未闭合缺口（[[skywork-vl-reward-paper]] §4.6 的风格偏好 × [[ostrakon-vl-paper]] 的取用方式）

### 与 MPO 的实证关联（[[skywork-vl-reward-paper]] §4.7）

- 用本模型构造偏好数据，MPO 微调 VLM reasoner（Skywork R1V2 的基座）：MathVista 69.2→**73.5**；用 Qwen2.5-VL-7B / InternVL3-8B 造的数据仅 71.2 / 71.8——RM 质量直接决定 [[mixed-preference-optimization|MPO]] 上限
- 同门 [[ostrakon-vl-paper|Ostrakon 论文]]引 Skywork R1V2 作为 MPO 的范例之一——Ostrakon「用它的模型筛数据、用它推崇的方法训练」两边都沾
- （外部网络来源，未入 raw/：后续 Skywork-VL-Res-38B，MMMU 70.1）

## 来源

- [[skywork-vl-reward-paper]]（模型本体、数据、训练、成绩）
- [[ostrakon-vl-paper]]（取用方式与三处用途）
- 外部网络来源（2026-09-14 核实）：[arXiv:2505.07263](https://arxiv.org/abs/2505.07263) ｜ [HuggingFace Skywork/Skywork-VL-Reward-7B](https://huggingface.co/Skywork/Skywork-VL-Reward-7B)

## 相关

- [[reward-model]] ｜ [[quad-data-curation]] ｜ [[ostrakon-vl]] ｜ [[qwen]] ｜ [[mixed-preference-optimization]]
