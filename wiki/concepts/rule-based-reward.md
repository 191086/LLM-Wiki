---
type: concept
title: 规则奖励（Rule-based Reward）
created: 2026-09-15
updated: 2026-09-15
tags:
  - reinforcement-learning
  - reward
  - object-localization
sources: 1
---

# 规则奖励（Rule-based Reward，准则驱动奖励）

**定义**：不用学习的奖励模型、不用人工偏好标注，直接用程序化规则（格式校验、精确匹配、任务指标）从「最终结果」计算奖励的评价信号路线；因 DeepSeek-R1 在数学 / 代码上的验证器奖励走红，[[vision-r1]] 把它推广到视觉定位任务（[[vision-r1-paper]] §1–2.2）。

## 1. 要解决的问题

偏好优化路线（[[rlhf]] / [[dpo]] / [[mixed-preference-optimization|MPO]]）有两重成本：高质量偏好数据靠人工标注（千条级也昂贵），稳健 RM 难训且可被钻空子。而视觉定位任务的标注本身就是精确框——**现成指令数据里已内嵌人类偏好**，缺的只是把标注变成奖励的程序（[[vision-r1-paper]] §1）。

## 2. 机制：Vision-R1 的准则驱动奖励函数

![[vision-r1-fig2-framework.png]]

把所有定位任务（检测 / VG〔visual grounding，给类别名找框〕/ REC〔referring expression comprehension，给一句自然语言描述找对应框〕）统一为检测式输出：模型吐「文本坐标 + 类别」，先还原成框、与 GT（ground truth，人工标注的真值框）做匹配，再按三路规则打分求和（[[vision-r1-paper]] §3.2，Eq.4–7）。框的重合度用 **IoU**（Intersection over Union，交并比）度量——交集面积 ÷ 并集面积，完全重合为 1、不相交为 0。小算例：预测框恰好把 GT 完全包住且自身面积是 GT 的 2 倍，则交集 = GT 面积、并集 = 预测框面积，$IoU = 1/2$。三路规则：

| 奖励 | 定义 | 对应的失败模式 |
|---|---|---|
| dual format | $reward_{DF}(o_i)=1$ 当且仅当模板校验 $f_{tem}$ 与数值内容校验 $f_{cont}$（坐标在界内、小数点位置正确）都通过，否则 0 | ①多实例长序列的模板 / 内容格式错误 |
| recall | $reward_{real}(o_i)=\dfrac{num(\text{Valid Predictions})}{num(GT)}$，与 GT 匹配且 IoU≥ξ₀ 才算有效预测 | ②漏检——LVLM 倾向少报、只报有把握的 |
| precision | $reward_{prec}(o_i)=\dfrac{\sum_{m=1}^{M}\left[(IoU'_m\ge\xi_0)\cdot IoU'_m\right]}{M}$，$M$ 为该完成的**全部**预测数 | ③小目标 / 难目标框不准 |

总奖励 = 三者之和（Eq.7）：召回鼓励「报得全」、精度鼓励「报得准」，互为制衡。消融（Griffon-G，COCO）：只留 precision 时框质量升（AP75 45.1）但召回掉（AR100 49.6 < 基线 52.2），加上 recall 后 mAP 42.1 反超基线 40.2（§4.3 Table 4）。

**口径注（式 6）**：论文行文把 precision 说成「valid predictions 的平均 IoU」，但式 6 的求和带指示子 $(IoU'_m \ge \xi_0)$、分母 $M$ 跑遍**全部**预测——未过阈值 / 未匹配的预测计 0 仍占分母，垃圾框因此被直接稀释。本页按公式口径表述，[[vision-r1]] §3 的走查亦按此计算。

**匹配用 box-only 简化匈牙利匹配**（匈牙利算法：多预测 ↔ 多 GT 的最优一一分配算法，代价取框间距离，检测器训练的标配环节；box-only 即代价只看框、不含类别）：检测专家模型按「框损失 + 类别概率」做分配，而 LVLM 直接输出确定性类别、无类别概率，类别项意义不大——box-only vs box+label 消融 42.1 vs 41.9 mAP（§3.2、§4.3 Table 3）。

## 3. 陷阱：reward hacking 与渐进式规则收紧

定位任务高 IoU 极难拿满 → 组内完成奖励趋同 → [[grpo|GRPO]] 优势信号消失（见其 §2 走查推论①）；且模型会「刷分」：产出大量低质框抬 recall。实测：Griffon-G 不收紧时 AR100 最高（56.7）但低质框沦为假阳性，mAP 39.9 反而低于基线 40.2（[[vision-r1-paper]] §3.3、§4.3 Table 5）。两条对策：

- **Differentiation**（Eq.8）：分段映射 $f(x)=\begin{cases}1 & x\ge\xi_2\\ 0 & x<\xi_1\\ x & \text{otherwise}\end{cases}$——低值清零、高值给满、中间线性，人为拉开奖励差距；precision 按实例施加、recall 按整完成施加
- **Staged progression**：阈值随训练收紧，初阶 $(\xi_0,\xi_1,\xi_2)=(0.5,0.5,0.75)$ → 进阶 $(0.75,0.75,0.9)$（ξ₂ 留在 0.9 而非 1，因完美框不存在）；切换时机 STEP 是超参，**须匹配模型能力**——强模型（Griffon-G）STEP=1/2 最优（42.1），弱模型（Qwen2.5-VL）够不到严格标准，不切换反而最优（STEP=1 得 26.6 vs STEP=1/2 得 23.3）（§3.3、§4.3 Table 5、附录 Table 10）

## 4. 与学习式 RM 路线的对照

| | 规则奖励 | 学习式 RM（[[reward-model]]） |
|---|---|---|
| 信号来源 | 程序规则 + GT 标注 | 从人类偏好数据学出的打分网络 |
| 数据需求 | 现成精确标注指令数据即可 | 千条级人工偏好对（多模态更贵，如 [[skywork-vl-reward|Skywork]] 的 19 万对） |
| 绝对分值 | 有意义（任务指标直接可比、可审计） | 不可辨识（BT 排序损失只学相对序） |
| 适用边界 | 仅限结果可程序化验证的任务 | 可评主观 / 开放任务，但可被 hack、有风格偏置 |
| 训练开销 | 零训练，在线计算 | 须先训 RM 或依赖现成 RM |

两路线并不互斥：[[mixed-preference-optimization|MPO]] 构造偏好对时也用规则奖励区分「正确」与「貌似正确」的回答——规则信号在偏好路线里当**数据过滤器**，在 R1 路线里直接当**奖励本身**。

## 5. 本 wiki 的实例与对照

- [[vision-r1]]：本页机制的全套落地（定位任务、49K 数据、GRPO 底座）
- [[quad-data-curation]]：反方向对照——QUAD 用学习式 RM 当清洗裁判；若任务可规则验证（如定位），规则信号可免掉这层 RM 依赖
- [[reward-model]]：学习式路线的系统分类（判别 / 生成 / 隐式 × ORM / PRM），规则奖励是其「系外成员」

## 来源

- [[vision-r1-paper]]

## 相关

- [[grpo]] ｜ [[reward-model]] ｜ [[vision-r1]] ｜ [[mixed-preference-optimization]] ｜ [[quad-data-curation]]
