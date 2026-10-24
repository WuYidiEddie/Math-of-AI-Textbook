# Math of AI Textbook

> 建立日期：2026-10-19 ｜ Lead
> 定位：把"训练模型"从**实验科学式的规律归纳**中拿出来，当成一个**纯数学问题**处理。
> 母项目：`../Science of AI Textbook/`（v1.5 题库、knowledge_base、experiments；本目录只把它们当 testbed，不当作理论来源）。
> 公式渲染：本目录统一用 `$...$`（行内）与 `$$...$$`（行间）；若查看器不渲染，请在 Markdown 预览中启用 KaTeX/MathJax。

---

## 0. 为什么有这个目录

前一阶段（`../theory/`、`.team/`、`knowledge_base/`）做的是：在大量合成任务上跑受控实验，从 156 条条件律里归纳"什么架构/优化器在什么语境下更好"。

这条路作为**工程产品**是有效的（域内规则约 82%），但作为 **Science** 有三个结构性缺陷：

1. **归纳对象是采样器，不是学习本身**。题库的候选与数据都由固定采样器生成；纯物理先验在 400 题上只到 56.3%，低于无 KB 的 LLM（68.4%）——说明"规律"的大部分来自元学习，而不是可迁移的物理。
2. **OFAT 设计有盲区**。固定一个因素做扫描，就永远学不到该因素的绝对水平（`a_priori/SCIENCE.md §4.1`）；而语境恰好是最大效应来源。
3. **规律不能回答"为什么"**。156 条律给的是符号与区间；当问题超出网格（depth≥6、无 LN、新族）时，只能外推，不能推导。

数学路线的主张不同：

> **给定目标函数 $f$、表达式类/参数化 $\mathcal G$、算法 $\mathcal A$（GD，$T$ 步，lr，初始化），
> 验证误差 $R_{\rm te}$ 能算到什么程度、能界到什么程度、哪些性质决定它。**

这不需要"从例子里归纳"；它需要的是定理、反例、以及可计算的判据。

---

## 1. 核心问题

用一个参数化表达式 $g_\theta$（$p$ 个参数）拟合从确定性函数 $f$ 采样的 $n$ 个点，参数 $p>n$，用 GD 走 $T$ 步，问：

$$
R_{\rm te}(\theta_T)=\mathbb E_{x\sim P_{\rm te}}\big[\ell(g_{\theta_T}(x),f(x))\big]
$$

**如何只用 $f$ 与 $\mathcal G$（以及 $\mathcal A,n,T$）的性质判断它的大小？** 特别是验证集上的大小。

本目录的回答分三层：

| 层 | 内容 | 状态 |
|---|---|---|
| **已解决** | 线性/特征模型、核方法、随机特征、NTK/lazy 区：有限步 GD 的 $R_{\rm te}$ 有**精确闭式**；由核谱、目标谱系数、$n,T,\eta$ 决定 | 定理 |
| **有严格界** | 算法稳定性/PAC-Bayes/压缩、局部 Rademacher、逼近论率、minimax 下界、n-width/optimal recovery | 界（常松，但方向正确） |
| **开放** | 有限宽非线性网络的特征学习区、分类 CE 的有限 $T$ 精确误差、深层网络的隐式偏置 | 研究前沿 |

**一句话**：在 $p>n$ 的设定里，"过参数化"本身不是变量；真正的变量是
**近似误差 $A$ + 识别/隐式偏置误差 $I$ + 优化滞后 $O(T)$ + 估计/验证波动 $F$**。
本目录要做的，是逐项给出这四项与 $f,\mathcal G,n,T$ 的关系。

---

## 2. 核心判据（先给结论）

把目标 $f$ 投影到表达式 $\mathcal G$ 在初始化处诱导的核 $K$ 的特征基上：

$$
f=\sum_j a_j\varphi_j,\qquad K(x,x')=\sum_j\lambda_j\varphi_j(x)\varphi_j(x').
$$

则（核/lazy 区，MSE，5 个确定性族无标签噪声，从零初始化；bigram 见 04 的熵地板例外）：

**精确形式**（总体核、无噪声）：

$$
R_{\rm te}(T)=\sum_j(1-\eta\lambda_j)^{2T}a_j^2 .
$$

**两尺度近似**（把模式分成"已学到"与"没学到"）：

$$
\boxed{\quad
R_{\rm te}(T)\quad\approx\quad
\underbrace{\sum_{j:\ \eta\lambda_jT\ll 1}a_j^2}_{\text{还没学到的目标分量（}O\text{）}}
\quad+\quad
\underbrace{\sum_{j\in\text{kernel tail}}a_j^2}_{\text{学不到的分量（}A\text{）}}
\quad+\quad
\underbrace{F}_{\text{验证/种子噪声}}
\quad}
$$

- **可学窗口**：$T$ 步内能学到的模式是 $\eta\lambda_jT\gtrsim1$ 的那些；小 $\lambda_j$（高频/难方向）需要更多步。这就是 **spectral bias**，也是"预算 $T$"的精确含义。
- **表达式的性质**：$\{\lambda_j\}$（由 Jacobian/NTK 决定，初始化即可算）+ 核的 tail（决定 $A$）。
- **函数的性质**：$\{a_j\}$（目标在核基上的能量分布）+ 目标是否落在核空间里（决定 $A$）。
- **验证误差**不是训练误差：$p>n$ 时训练误差可以到 0，但它对 $R_{\rm te}$ 没有信息；上式中只剩"目标能量中未被学到的部分"。

严格版本、有限 $n$ 修正、非核区、以及"如何检验这个公式"见后续章节。

---

## 2.5 答题器成绩（持续更新）

理论答题器（`verification/theory_answerer.py`）：每个候选 = 初始化切线性化模型 + 官方原协议复刻，**零拟合**（没有任何系数从题库学出）。

**out-of-sample 纪律**：题库按 qid 哈希固定切成 dev/test 两半（`theory_answerer_eval.py`）；失败分析只在 dev 半区做，**headline 数字是 test 半区准确率**。预测器若在观察失败后改动，改后的 test 半区数字仍是 OOS，但要在备注里声明改动的诱因。

| 日期 | 预测器版本 | 全题库（500 题） | **test 半区（258 题）** | dev 半区（242 题） | 备注 |
|---|---|---|---|---|---|
| 2026-10-21 | v1 惰性线性化（seed 0；48 题为 3-seed 均值） | 83.2%（416/500） | **85.7%**（221/258） | 80.6%（195/242） | 基线：随机 33.3%、max-lr 启发式 53.0%、KB 经验律自报 74–88%、标签天花板 98.5% |

外部题库（真·OOS，未参与任何开发）：

| 日期 | 题库 | top-1 | 备注 |
|---|---|---|---|
| 2026-10-21 | dataflip_v1_strict（49 题，strict 翻转，标签噪声≈0） | **49.0%**（24/49） | spiral 10/10、mvar 9/10、tabular **0/9**（系统性反转）；≈ 实测 10% 截断基线（44–53%）；分析见 `verification/dataflip_result.md` |

分族（全题库）：spiral 92.0% / tabular 86.0% / univar 85.0% / mvar 85.0% / bigram 76.0% / xor 76.0%；分题型：optimizer_only 88.1% > mixed 82.7% > architecture_only 78.8%。已证伪的改进方向：预测侧多种子（M0，48 题试点 argmin 翻转 0/48，见 `verification/m0_multiseed_result.md`）。

复现：`python verification/theory_answerer.py --skip-done`（可续跑）+ `python verification/theory_answerer_eval.py`。

---

## 3. 文档结构

| 文件 | 内容 |
|---|---|
| [`textbook/`](textbook/) | **给人类读者的教程（Markdown + LaTeX 数学）**：`0.0`–`0.6` 是前置知识（谱线代、Jacobian 与梯度流、统计噪声与熵、训练实操词汇、核与 NTK、逼近论与信息极限，含 30 题自测与依赖映射），`00`–`09` 是正文（**训练=选函数（平方损失的高斯似然来历 → 条件均值 → 噪声地板；GD 从 0 出发收敛到最小范数插值解 → 零空间地板；在真题 q_11b94d 上把 $A$/$F$/$O(T)$/$I$ 逐项测出来）**、四分解、**可精确算的核情形（含从平方损失到闭式递推的逐步推导）**、预算/lr/曲线、**泛化与验证（含 double descent 期望公式的推导与配对比较）**、**函数结构与架构（$A$ 的正交投影闭式、$n^{-s/d}$ 的五行推导与格点计数、真实族上分离 $A$ 与 $O(T)$、惰性区的最小范数权重与 GP 后验方差恒等式）**、分类/bigram、四道真题、理论边界、答题器实现）。建议先读 `textbook/README.md`，不确定基础够不够就先做 `textbook/0.0-prereq-map.md` §2 |
| [`01-formulation.md`](01-formulation.md) | 问题的严格表述；"拟合好坏"的五层含义；四类误差 $A/I/O/E/F$ 的分解；从 $f,\mathcal G$ 的性质出发的判定流程（判据/决策树）；验证集的两种含义（总体风险 vs 有限验证估计） |
| [`02-exact-regimes.md`](02-exact-regimes.md) | 可精确求解的核心：过参数化线性回归、核/随机特征、有限步 GD 的谱滤波与有限 $n$ 精确风险、early stopping≈ridge（同尺度，非精确）、NTK/lazy、深度线性网 plateau；$p>n$ 的零空间地板；double descent 的条件与**无标签噪声时的失效**；n-width/optimal recovery 下界 |
| [`03-frontier.md`](03-frontier.md) | 不可精确求解的前沿：逼近论（Barron/Sobolev/ReLU 区域/深度分离）、隐式偏置、算法稳定性、特征学习（lazy/rich、mean-field、μP）、分类 margin 与 CE；哪些是定理、哪些是 open problem |
| [`04-benchmark-testbed.md`](04-benchmark-testbed.md) | 用 v1.5 题库做 testbed：六个族的 $f$/$\mathcal G$ 性质、理论预测的 dominant term、可计算 pipeline（初始化 NTK + 目标谱 + 滤波）、与旧 empirical law 的关系、可证伪检验 |
| [`05-formalization-and-coverage.md`](05-formalization-and-coverage.md) | **编码自查、机器验证、Lean 状态与"理论够不够解 v1.5"的诚实评估**；含两道代表题（`q_11b94d` / `q_aef977`）的 lazy 探针结果 |
| [`verification/`](verification/) | `verify_symbolic.py`（SymPy 精确验证 10 条代数骨架）+ `lazy_probe.py`（真实 v1.5 候选的 NTK 变化探针）+ `four_decomposition.py`（在真题 q_11b94d 上逐项测量 $A$/$F$/$O(T)$/$I$）+ `exact_case_derivation.py` / `budget_curve_derivation.py` / `generalization_stats.py`（正文 02/03/04 章的推导链核验）+ `architecture_approximation.py`（正文 05 章：$A$ 的正交投影闭式、$n^{-s/d}$ 的格点计数、真实族上分离 $A$ 与 $O(T)$、惰性区的最小范数权重与 GP 后验方差恒等式）+ `zero_space_init.py`（正文 02 章 §6.4–6.5：零空间投影的守恒是不是被强制的——逐优化器实测 drift、初始化税、AdamW 在 v1.5 真题上的漂移）+ `second_order_pilot.py`（M4 试点·四次式路线：二阶切模型在官方 lr 下不稳定，不是可靠升级）+ `moving_linearization_pilot.py`（M4 试点·核修正路线：移动线性化，凸且稳定，$S=64$ 把 4 道试点题全翻对；$S=$ 步数时逐位复现真实训练） |
| [`lean/`](lean/) | Lean 4 形式化规格（本机无工具链，未编译；不放入伪装的 `.lean`） |
| [`notes/`](notes/) | 支撑底稿：文献综述、公式推导、题库设定统计（来自团队，原始引用路径在文中） |
| [`related-work/`](related-work/) | 相关论文存档（32 篇 PDF，2026-10-23 检索）：同类纲领、谱正则化母体文献、EoS 严格化进展；结论见目录内 README |

**阅读顺序**：随手读 → [`textbook/README.md`](textbook/README.md)（人话版）；完整数学 → README → 01 → 02 → 03 → 04 → 05。若只关心"能不能解题"，直接读 `textbook/07-case-studies.md` + 05 §4。

---

## 4. 方法论纪律

1. **定理优先**：每个断言标明 exact / bound / asymptotic / heuristic 四档之一；不把"在网格上成立"写成"定律"。
2. **p>n 不是论点**：有效容量由核的有效秩、函数流形的维数（去掉对称性/规范化自由度）、以及算法隐式偏置决定，不由参数个数决定。
3. **除 bigram 外没有标签噪声**（5 个确定性族；bigram 有不可约熵，见 04）：经典 benign-overfitting 的"噪声方差"故事在这里不适用；必须区分"标签噪声"与"采样/插值 aliasing"（精确标签不消除设计采样项）。
4. **benchmark 是检验，不是来源**：先用数学预测一个候选的 $R_{\rm te}(T)$，再与实测比；差距是理论缺口（尤其 feature-learning 项），不是新规律。
5. **实验只做证伪**：任何新实验都应针对一个可证伪的定理/预测设计，而不是继续扫网格找规律。
6. **承认边界**：一般非线性有限宽网络没有精确理论；本目录的目标不是假装解决它，而是把"能解决的部分"与"开放部分"划清。

---

## 5. 与旧项目的关系

- `../Science of AI Textbook/theory/UNIVERSAL-LAWS.md`、`knowledge_base/`、`a_priori/` 的经验结果**不删除**，但降级为：
  - **testbed 的测量**（v1.5 题目的实际行为）；
  - **启发式先验**（当数学理论未覆盖时使用，低置信）；
  - **待解释现象**（例如"语境支配"应被推导为 $U$ 的跨度 vs $G(U)$ 的跨度；"Q 可预测排序"应被推导为初始核的谱对齐）。
- 旧知识库里的 156 条律，在本目录的视角下大多是**相图的经验切面**：真正要推导的是相边界（n-width、谱边、优化时间、EoS、噪声阈值），而不是逐条律。
