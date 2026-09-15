---
type: entity
title: Vision-R1（视觉规则强化学习方法）
created: 2026-09-15
updated: 2026-09-15
tags:
  - mllm
  - reinforcement-learning
  - object-localization
sources: 1
---

# Vision-R1

**定义**：中科院自动化所等提出的视觉引导 R1 式强化学习方法——用视觉准则驱动的规则奖励（无 RM、无人工偏好数据）+ 渐进式规则收紧，以 [[grpo|GRPO]] 强化 LVLM 的物体定位能力。

## 1. 定位与思想

名字对齐 DeepSeek-R1（R1 式 = 规则奖励 + GRPO），但任务不是长链推理而是**物体定位**——把检测 / visual grounding（VG，视觉定位：给类别或描述找框）/ REC（referring expression comprehension，指代表达理解：给一句自然语言描述找对应框）统一为检测式输出。「human-free alignment」指无需人工偏好标注与 RM 训练（[[vision-r1-paper]] §1）。针对 LVLM 定位三大失败模式（§3.2，Figure 2）：①多实例长序列格式错误；②漏检（召回低）；③小/难目标框不准——三路奖励一一对应。

## 2. 两大组件

①**准则驱动奖励函数**：dual format（模板 ∧ 数值内容校验）+ recall（有效预测数 / GT 数）+ precision（有效预测的 IoU 质量），文本坐标还原成框、与 GT 匹配后按规则直接算分，机制详见 [[rule-based-reward]] §2。②**渐进式规则收紧**：differentiation（分段映射拉开奖励差距）+ staged progression（阈值随训练抬升），防组内奖励趋同与 reward hacking，详见 [[rule-based-reward]] §3。

## 3. 一条样本的奖励走查（构造演示，按论文式 4–8）

设一张图有两个 GT 目标（一只猫、一只狗），初阶阈值 $(\xi_0,\xi_1,\xi_2)=(0.5,0.5,0.75)$，模型（采样自 GRPO 组内的一条完成 $o_i$）输出 3 个预测框：

| 预测 | 内容 | 与匹配 GT 的 IoU | 有效？ |
|---|---|---|---|
| P1 | cat | 0.86 | ✓（≥ξ₀） |
| P2 | dog | 0.42 | ✗ |
| P3 | chair | 无匹配（计 0） | ✗ |

逐路计算：

- **dual format**：模板与坐标数值（界内、小数位）校验全过 → $r_{DF}=1$
- **recall**：有效预测 1 / GT 2 = 0.5；收紧映射 $f$ 按「整完成」施加：$f(0.5)=0.5$（未达 $\xi_2=0.75$ 给不满、未低于 $\xi_1=0.5$ 不清零，走线性段）→ 0.5
- **precision**：收紧映射按实例施加于每个 IoU：$f(0.86)=1$（≥ξ₂ 给满）、$f(0.42)=0$（<ξ₁ 清零）、$f(0)=0$；按式 6 除以预测总数 $M=3$ → $1/3 \approx 0.33$。（论文行文称「有效预测的平均 IoU」，但式 6 分母 $M$ 跑遍全部预测、带指示子——未匹配的 P3 计 0 仍占分母，**垃圾框因此被直接稀释**。）
- **总奖励** $= 1 + 0.5 + 0.33 = 1.83$（上限 3）

这条 1.83 随后进入 GRPO 的组内归一化算优势（见 [[grpo]] §2 走查）：组内各完成的总奖励做差 → 优势 → 更新策略。P2 那个 0.42 的框不受鼓励（清零）但也只损失自身，P3 则拉低整条完成的 precision——三路规则就是这样把「报得全」（recall）与「报得准」（precision）制衡起来的。

## 4. 训练配方

以 [[qwen|Qwen2.5-VL-7B]] 与 Griffon-G-7B（一弱一强两种定位水平）为基座，49K 定位指令数据（约 50% 难例：30K 检测 COCO + 9K VG + 10K REC，不新标数据），Open-R1 多模态框架，β=0.2、lr 1e-6、1 epoch（[[vision-r1-paper]] §4.1）。指令模板严格沿用各模型 SFT 阶段格式——模板即 dual format 奖励的校验对象。

## 5. 成绩

Qwen2.5-VL-7B COCO mAP 17.7→26.6（≈+50%）、ODINW-13 37.0→46.0 反超自家 72B（43.1）；Griffon-G-7B 40.2→42.0 / 43.8→46.3；通用 QA 不掉而 SFT 对照明显掉（GQA 58.8→53.5 vs Vision-R1 升到 61.0——定位提升外溢到物体感知类任务）（[[vision-r1-paper]] §4.2–4.3）。

## 6. 家族

Griffon 系（Griffon / Griffon-G / Griffon v2）是同一团队的定位特化 LVLM 谱系；Vision-R1 是该谱系从 SFT 走向 RL 后训练的延展，也在更强的 Qwen2.5-VL 上验证通用性（[[vision-r1-paper]] §2.1、§4.1）。

## 来源

- [[vision-r1-paper]]

## 相关

- [[grpo]] ｜ [[rule-based-reward]] ｜ [[qwen]] ｜ [[reward-model]] ｜ [[mixed-preference-optimization]]
