# knowledge_base 经验律的理论解释映射

> 日期：2026-10-20 ｜ scope：把 `../../Science of AI Textbook/knowledge_base/`（以下简称 KB）的主要经验律逐条映射到 01–03 的理论对象（$A/I/O/F$、核谱、有效步长、熵地板），标注**已解释 / 部分解释 / 未解释**。
> 定位：KB 是"待解释现象"，不是理论来源（README §5）。本文是理论对现象的覆盖审计；逐题的定量裁决由 `../verification/theory_answerer.py` 的实测提供。
> 记号：$\eta\lambda_j T$ = 可学窗口；$\lambda_{\max}$、条件数 $\kappa$、有效秩均指**初始化核**；"EoS" = Edge of Stability（$\eta\lambda_{\max}\gtrsim2$）。

---

## 0. 总结论

KB 的六族经验律在理论框架下**大多数是 $O$（优化滞后）的相图切面**，少数是 $A$（表达）与 $F$（测量噪声），几乎没有一条是"参数个数"的效应。逐项审计：

| 现象类型 | KB 中的代表律 | 理论项 | 解释状态 |
|---|---|---|---|
| 优化器×lr 主导、甜点表、3e-5 灾难 | spiral C01/C02、tabular C06、xor K01/K03、bigram B-OPT | $O$：$\eta_{\rm eff}\lambda_jT$ 可学窗口 | **机制已解释**；逐题定量=答题器检验 |
| steps 单调有益（回归） | univar/mvar C06/C01 | $O$：无噪声核/lazy 区 $R_{\rm te}(T)$ 单调（02 §1.5） | **已解释（lazy 区是定理）** |
| steps 非单调（CE 族长训反弹） | xor K02、bigram B-BUD | CE 非二次；无平滑时收敛到未平滑 MLE → KL 上升（`bigram-estimation.md` §4）；EoS | **部分解释**（bigram 有定量机制；xor 定性） |
| 更深/更宽更好，高 lr 翻转 | univar C03、mvar C06/C02 | $O$：深/宽 → $\lambda_{\max}\uparrow$、谱含更多可学成分；高 lr 翻转 = 先撞 EoS | **部分解释**（EoS 给机制；精确边界 open） |
| LN/残差效应及交互 | univar C04/C05、mvar C04/C05、tabular C04 | $O$：LN/残差改善初始化核条件数；窄网条件数差更依赖 LN | **部分解释**（方向可推导；"LN=谱变换"的精确形式是 03 §9 开放问题 7） |
| spiral 深度瓶颈 | spiral C03（d=1 硬瓶颈，d≥3 解决） | $A$：深度分离/线性区域数（03 §1.4） | **已解释（方向）；定量 $A$ 未逐题算** |
| 容量不足风险（w16 必差等） | mvar cheatsheet、xor K04 | $A$+核谱（小 $w$ → 核秩/谱宽不足） | **部分解释** |
| 数据集难度 3–5× | univar C08、spiral C05、bigram B-DIFF-01 | 目标谱 $\{a_j\}$ 与核的对齐；bigram：$H(V,\alpha)$+地板 | **已解释**（bigram 已定量，见 §3） |
| 刀刃题/种子噪声 | univar C07、spiral C08、tabular C08、xor K06、bigram B-NOISE | $F$：配对差 + $2$–$3\,\mathrm{se}$ 判据（01 §9） | **已解释** |
| 跨 V 排序一致 92.9%（bigram） | B-DIFF-01 | 地板只抬常数，不改 $O$ 的排序 | **已解释**（与 $O$ 主导一致） |
| Adam=AdamW@wd=0 逐位同 | mvar C07 | 同一算法 | **平凡解释** |

**一句话**：KB 的"语境支配"在理论上 = $O$ 项的方差远大于 $A/I$ 项的方差；"Q 可预测排序" = 初始化核的谱对齐差异。这与旧项目 `a_priori/SCIENCE.md` 的观察一致，但现在每一项都有可计算对象。

---

## 1. 横切律的理论对象

### 1.1 "3e-5 系统性灾难" = 可学窗口为空

KB 在 univar/mvar/tabular/xor 都发现 lr=3e-5 是强负项（tabular：赢家仅 1/33）。理论翻译：

$$
\text{可学窗口：}\ \eta\lambda_jT\gtrsim1
\quad\Rightarrow\quad
\eta=3\times10^{-5},\ T\sim10^2\text{–}10^3\ \text{需要}\ \lambda_j\gtrsim30\text{–}300 .
$$

答题器实测初始化核 $\lambda_{\max}$ 分布（`theory_answerer.jsonl` 的 `lambda_max` 与 `eta_lambda_max` 字段）可直接检验：$\eta\lambda_{\max}T\ll1$ 的候选应当全部欠训。**注意 Adam 系的有效步长不按此缩放**（逐坐标归一化使每步位移 $\approx\eta$，与梯度尺度弱相关），所以"3e-5 灾难"在 Adam 题上的机制不同：是**绝对位移预算** $\eta T$ 太小（参数相对位移 ~$\eta T/\|\theta_0\|$），不是谱窗口。两种机制都能在答题器的 `param_rel_disp` 字段上看到。

### 1.2 优化器甜点表 = 有效步长 × 谱的匹配

KB 甜点：Adam/RMSprop ≈1e-3（bigram 可到 1e-2）、Adagrad ≈1e-2、SGD ≈1e-3–1e-2。理论翻译：

- SGD：有效收缩率 $(1-\eta\lambda_j)^T$；甜点在 $\eta\lambda_{\max}$ 接近 EoS 但不越过。
- Adam：位移 $\approx\eta$ 每坐标，甜点由"$\eta T$ 足够走完 $\|\theta_*-\theta_0\|$ 的典型尺度"决定；$\eta$ 过大则在最优点附近振荡/越过（CE 族长训反弹的另一候选机制）。
- Adagrad：累积使有效步长 $\sim\eta/\sqrt{\sum g^2}$ 随时间衰减，所以需要更大的初始 $\eta$（KB：Adagrad@3e-4 垫底、3e-3 反超成最优）。

**状态**：方向与排序机制清楚；Adam/Adagrad 的有限 $T$ 精确预测仍是 03 §9 的开放问题 8。答题器把优化器当作"对线性化模型的精确迭代"来模拟，绕开了解析困难——这是数值答案，不是定理。

### 1.3 刀刃题判据 = $F$

KB 各族一致的规则（margin ≥ 2–3× 种子噪声才可判）就是 01 §9 的配对差判据 $\lvert\Delta\rvert > 2.26\,\mathrm{se}_{\rm paired}$。KB 的数字（univar arch 间隔中位 0.024、spiral 相对 std 14%、tabular 33% 组 margin<0.01、bigram spread 0.026、xor 5-seed rel-std 5%）全是 $F$ 的实测。**已解释，且这是 KB 与理论一致性最好的部分。**

---

## 2. 分族映射要点

### 2.1 univar / mvar（无噪声回归）

- **steps 单调**（225:7、116/116）：02 §1.5 已证 lazy/核区无噪声时总体 $R_{\rm te}(T)$ 单调降；7/116 的反例应是 $F$ 内的涨落或 rich 区候选。**已解释（lazy 区为定理）。**
- **深/宽/LN/残差**（C03/C04/C05，mvar C02–C06）：全部归入 $O$ 的核谱/条件数效应 + 高 lr 的 EoS 翻转。可检验预测：在 `architecture_only` 题上，答题器（含真实优化器）应复现大部分方向；失败处即特征学习项。
- **silu 垫底**（mvar C03）：激活 → 核谱（光滑度/高频含量）与梯度尺度；方向可解释，定量 open。
- **wd=0 时 Adam≡AdamW 48/48 逐位同**：算法同一，平凡。

### 2.2 spiral / tabular / xor（确定性分类）

- **spiral d=1 硬瓶颈**（C03）：深度分离（$A$）；d≥3 后 $A\approx0$、胜负回到 $O$。**已解释方向。**
- **spiral C01（opt,lr 差 $2\times10^4$ ≫ 架构差 $10^2$–$10^3$）**：$O$ 方差 ≫ $A$ 方差的直接证据。
- **tabular "浅网更好"vs lr=3e-4 翻转**（C03/R3）：$A$ 对 smooth_additive 本来就小 → 浅网 + 快优化胜；低 lr 时浅网 $\lambda_{\max}$ 小、先欠训，深网梯度尺度大反而快 → 符号随 lr 翻转。这与"深度收益是 $O$ 不是 $A$"（04 §3.2 读数）一致。
- **xor 容量**（K04：w48→16 差 3.3–3.6×）：二阶交互对 2 层宽网 $A$ 小，但 w=16 时核谱太窄 → $O$（或有效容量）受限；KB 自报"更宽仅 41.4%"的题面统计与受控冲突，理论支持受控结论（题面统计混杂 lr）。
- **CE 有限 $T$ 无闭式**：以上全部是结构解释 + 答题器数值裁决，不是定理。

### 2.3 bigram（随机目标）

`bigram-estimation.md` 已把该族的 $H/I/F$ 定量化。对 KB 条目的对应：

| KB 条目 | 理论解释 |
|---|---|
| B-DIFF-01：V24/32/48 中位 CE 3.09/3.40/3.79，$H$≈2.79/3.02/3.40 | $H(V,\alpha)$ 由生成器精确可算；超额中位 ≈0.3–0.4 与 $O$ 主导一致；跨 V 排序一致 92.9% = 地板只加常数 |
| B-OPT-02/04：甜点表、SGD 全档差 | $O$；查表模型给凸情形参照；SGD 的有效步长 × 谱窗口不足 |
| B-BUD：lr≤1e-4 步数单调好；Adam+1e-3/3e-3 时 256>1024 为 89/90 | 低 lr：纯欠训，$O$ 单调降；中高 lr 长训反弹：**候选机制 = 收敛到欠平滑解**（wd 小/无 → logits 冲向未平滑 MLE，test KL 上升，见 `bigram-estimation.md` §4 的 U 形观察）或 EoS 振荡；可用"查表 + 同优化器"消融判别 |
| B-ARCH：width 强、depth 弱、heads 反转 | 推测为条件数/$O$ 效应；未逐条推导，标 **未解释（有假设）** |
| B-NOISE：margin≥0.10 才判 | $F$；注意我们测得**测试集抽样 sd≈0.013**（共用测试集会抵消），KB 的 0.026 是 3-seed spread（含初始化/ minibatch 噪声），两者不同口径 |

---

## 3. KB 里理论目前不能解释的（诚实清单）

1. **所有"高 lr 翻转"的精确边界**：EoS 只给 $\eta\lambda_{\max}\approx2$ 的阈值，翻转的具体位置依赖有限宽 Hessian 演化——open。
2. **LN/残差的定量谱效应**：方向可推（条件数），但"哪个位置、几个 LN"的细规则（如 tabular last>all0 47/48）没有推导。
3. **激活排序**（gelu≈silu>leaky>relu 等）：只有核光滑度的定性说法。
4. **bigram 的 heads/宽度细规则**（B-ARCH-04 等）。
5. **mvar 的宽度×优化器翻转**（Adam@1e-3 宽度好、RMSprop@1e-3 宽度差）：指向"预条件×谱"的交互，是 Adam 理论缺口的直接实例。

这些应进入 03 §9 的开放问题清单；不要用新的经验律去补。

---

## 4. 裁决工具

`../verification/theory_answerer.py` 对每题每候选：初始化 Jacobian → 切线性化模型 → **原封不动跑官方协议**（同 seed 的初始化与 minibatch 序列、同优化器配置、同步数）→ 预测 test 指标。若候选真 lazy，预测≈实测；偏差 = 特征学习项。结果与分族准确率见 `../verification/theory_answerer_result.md`。
