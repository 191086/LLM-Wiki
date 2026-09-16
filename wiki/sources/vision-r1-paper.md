---
type: source
title: Vision-R1：免 RM 的视觉规则强化学习（论文）
created: 2026-09-15
updated: 2026-09-16
tags:
  - paper
  - mllm
  - reinforcement-learning
  - reward
  - object-localization
sources: 1
raw: raw/Vision-R1.pdf
---

# Vision-R1: Evolving Human-Free Alignment in Large Vision-Language Models via Vision-Guided Reinforcement Learning

> 原文：[[raw/Vision-R1.pdf]] ｜ 类型：论文（arXiv:2503.06749v1）｜ 日期：2025-03-23 ｜ 机构：中科院自动化所（CASIA）· 国科大 · 鹏城实验室 · 武汉人工智能研究院 ｜ 代码：github.com/jefferyZhan/Griffon/tree/master/Vision-R1

> **收录注记**：全文公式号按原文（§3.1 Eq.1–2、§3.2 Eq.3–7、§3.3 Eq.8）。一处口径出入：式 6 行文称「全部 valid 预测的平均 IoU」，但公式求和带指示子、分母是**全部**预测数 $M$——wiki 侧按公式口径记录（见下方要点与 [[rule-based-reward]] 口径注）。

## 一句话总结

把 DeepSeek-R1 式规则强化学习（[[grpo|GRPO]]）搬到多模态物体定位：用「视觉准则驱动奖励」（格式 + 召回 + 精度三路规则信号，直接从预测框与 GT 算出）替代奖励模型与人工偏好数据，配合「渐进式规则收紧」防 reward hacking，仅 49K 定位指令数据一轮训练就把 Qwen2.5-VL-7B 的 COCO mAP 提升 ≈50%（17.7→26.6）、ODINW-13 反超 10 倍大的 Qwen2.5-VL-72B（46.0 vs 43.1）。

## 关键要点

- **问题定位**：LVLM 后训练的偏好优化路线（RLHF / DPO / [[mixed-preference-optimization|MPO]]）依赖昂贵的人工偏好标注 + 健壮 RM；而视觉定位任务答案客观、标注精确——「现成的精确指令数据本身已内嵌人类偏好」，规则奖励可直接取用，即标题的 human-free alignment（§1、§2.2）
- **GRPO 底座**（§3.1，Eq.1–2）：组内相对优势 $A_i = \big(r_i - \mathrm{mean}(\{r_j\}_{j=1}^{N})\big) / \mathrm{std}(\{r_j\}_{j=1}^{N})$（Eq.1）；目标 $\mathcal{J}_{GRPO}(\theta) = \frac{1}{N}\sum_{i=1}^{N}\big(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i - \beta\,\mathrm{KL}(\pi_\theta(o_i|q)\,\|\,\pi_{ref}(o_i|q))\big)$（Eq.2，引 DeepSeekMath 的简化写法，比率项无 clip；机制展开见 [[grpo]] §2）
- **LVLM 定位三大失败模式**（§3.2，Figure 2）：①多实例长序列格式错误（模板 + 数值内容）；②有效预测数不足（漏检、召回低）；③小目标 / 难目标框不准
- **[[rule-based-reward|准则驱动奖励函数]]**（§3.2，Eq.3–7）：先做 **box-only 简化匈牙利匹配**（LVLM 输出确定性类别标签、无类别概率，类别项帮助小：42.1 vs 41.9 mAP，§4.3 Table 3），每个预测实例得坐标、类别与 IoU（Eq.3：$\{P_m\}_{m=1}^{M} = \mathrm{extract\_match}(o)$，IoU 超过阈值 $\xi_0$ 才算 valid）；三路奖励求和（Eq.7：$r = r_{DF} + r_{recall} + r_{prec}$）——dual format：模板 $f_{tem}$ ∧ 内容 $f_{cont}$ 全过才给 1（Eq.4）；recall：有效预测数 / GT 数（Eq.5：$r_{recall} = \#\text{Valid}/\#\text{GT}$）；precision（Eq.6）：$r_{prec} = \frac{1}{M}\sum_{m=1}^{M}\mathbb{1}[\mathrm{IoU}_m \ge \xi_0]\cdot \mathrm{IoU}_m$，分母 $M$ 跑遍**全部**预测（口径注见收录注记与 [[rule-based-reward]]）
- **[[rule-based-reward|渐进式规则收紧]]**（§3.3，Eq.8）：differentiation 分段映射 $f(x) = 1$（$x \ge \xi_2$）、$0$（$x < \xi_1$）、$x$（否则）——拉开组内奖励差距，逐实例作用于 precision、整条作用于 recall；+ staged progression（阈值 0.5/0.5/0.75 → 0.75/0.75/0.9）持续加压，防 reward hacking（§3.3）
- **结果**（§4.2，Table 1–2）：COCO mAP Griffon-G-7B 40.2→42.0（+1.8）、Qwen2.5-VL-7B 17.7→26.6（+8.9）；ODINW-13 平均 46.3 / 46.0，双双超过 Qwen2.5-VL-72B（43.1）与 Gemini 1.5 Pro（36.7）；OOD 四集 +7.1 / +4.8，在 BoggleBoards、MountainDewCommercial 上反超专家模型 GroundingDINO
- **vs SFT 同数据对照**（§4.2–4.3，Table 6、11）：SFT 把 Qwen2.5-VL 的 ODINW-13 从 37.0 拉低到 35.0（有限数据过拟合），通用 QA 也明显掉（GQA 58.8→53.5）；Vision-R1 两头都不掉，GQA 还升到 61.0（定位提升外溢到物体感知类常识任务）
- **STEP 消融**（§4.3 Table 5 + 附录 Table 10）：强模型（Griffon-G）STEP=1/2 最优（42.1），不收紧（STEP=1）甚至低于基线（39.9 < 40.2——AR100 最高 56.7 但低质框泛滥成假阳性）；弱模型（Qwen2.5-VL）反而 STEP=1 最优（26.6 vs 23.3）——收紧时机须匹配模型现有能力
- **数据构建**（§4.1 + 附录 §1，Table 7）：49K = 30K 检测（COCO）+ 9K VG（ODINW 5K + V3Det 4K）+ 10K REC（RefCOCO 5K + Visual Genome 多目标指代 5K）；不新标数据，从既有精标定位指令集中筛选，原则 = 多样性 + 挑战性；COCO 难例按单图 >10 实例判定、自难 / 易样本各采 1/3（原文表述含糊）；每类内约 50% 高挑战（类别与实例更多）+ 一定比例负样本
- **训练配置**（§4.1 + 附录）：指令模板严格沿用各模型 SFT 阶段格式（Qwen2.5-VL 用 JSON 坐标、Griffon-G 用纯文本），未覆盖任务按相近格式改造（Table 8）——模板即 dual format reward 的校验对象；Open-R1 多模态框架（R1-V），β=0.2，lr 1e-6，1 epoch

## 值得追踪的实体与概念

- [[vision-r1]]（方法 / 模型）｜ [[grpo]]（底层算法）
- [[rule-based-reward]]（本文两大贡献的概念化：准则驱动奖励 + 渐进收紧）
- [[reward-model]]（被绕开的部件，路线对照）｜ [[mixed-preference-optimization]]（偏好路线代表，其出处论文即本文引文 [45]）
- [[qwen]]（Qwen2.5-VL-7B：实验基座之一，也是 [[skywork-vl-reward]] 的基座）
