# 数学怎么帮我们理解 AI 训练：一本给人类读者的教程

> 位置：`Math of AI Textbook/textbook/`（原 `tutorial/`，2026-10-22 更名）
> 读者：有基本线性代数/概率直觉、但不一定学过学习理论的 AI 研究者和工程师。**不确定自己够不够格？先做 [`0.0-prereq-map.md`](0.0-prereq-map.md) §2 的自测题（20 分钟）。**
> 写法：**纯 Markdown**；公式用 `$...$`（行内）与 `$$...$$`（行间），与项目根 `README` 的约定一致。若查看器不渲染，请在 Markdown 预览中启用 KaTeX/MathJax。

---

## 0. 这本教程想解决什么问题

你训练一个小模型：目标是一个确定性的函数 $f$，你从它上面采了 $n$ 个点，用一个参数化表达式 $g_\theta$（$p$ 个参数，$p>n$）去拟合，用梯度下降走 $T$ 步。训练误差可以降到接近 0，但**验证集上到底好不好？为什么 A 配置好、B 配置差？**

常见回答是经验规律："更深更好/更宽更好/大 lr 不行/小 lr 欠拟合"。但这些规律互相矛盾，换个任务就翻车。

这本教程换一个问法：

> 给定目标函数 $f$、表达式类 $\mathcal G$、算法 $\mathcal A$（GD，$T$ 步，lr，初始化），
> **仅用 $f$ 和 $\mathcal G$ 的性质，能不能算出/界定验证误差？**

答案分三层：

| 层次 | 能做到什么 | 对应文档 |
|---|---|---|
| 已解决 | 线性/核/惰性（lazy）区：验证误差有闭式，可在训练前算 | 02 |
| 有严格界 | 逼近论、$n$-width 信息下界、稳定性/泛化界 | 04、05 |
| 开放 | 有限宽非线性特征学习、Adam 有限步、分类 CE、bigram 网络的有限 $T$（bigram 的地板/估计/噪声已可算，见 06 §5.3） | 06、08 |

---

## 1. 五个核心想法（先记住这五句）

1. **训练不是"把参数调到最优"，而是"在无数个能拟合数据的函数里，算法选了哪一个"。**
2. **验证误差 $R_{\rm te}\approx A+I+O(T)+F$**：
   - $A$（approximation）：表达式类根本表达不了的部分；
   - $I$（identification）：算法在数据上选错了函数（隐式偏置 + 有限样本 aliasing）；
   - $O(T)$：$T$ 步还没走到；
   - $F$：验证集/种子的测量噪声。
3. **$p>n$ 本身不是变量。** 有效的"容量"是核的有效秩/函数流形的维数，不是参数个数。
4. **在核/惰性区，"$T$ 步后好不好" = 目标函数的能量有多少落在可学窗口 $\eta\lambda_jT\gtrsim1$ 里。** 这是 spectral bias 的精确版本。
5. **本设定里 5 个族没有标签噪声**（univar/mvar/spiral/xor/tabular），所以经典"过拟合噪声"故事不是主体；bigram 例外，它有不可约熵地板。

---

## 2. 阅读路线

### 路线 P：先补前置（`0.0`–`0.6`，1.5–4.5 小时）

如果 $\sum_j(1-\eta\lambda_j)^{2T}a_j^2$ 这个公式让你卡住，或者你不知道 `LN=FTTF`、`Adagrad@3e-5`、$H(Y\mid X)=3.196$ nats 在说什么，先走这条：

1. `0.0-prereq-map.md`：自测题 + 六章前置的依赖映射 + 四条补课路线（P0 全补 / P1 最省 / P2 工程补数学 / P3 数学补工程）+ 统一符号表
2. `0.1-linear-algebra.md`：谱定理、零空间/行空间、SVD、条件数、有效秩 —— **正文 02 的承重墙**
3. `0.2-calculus-and-gradients.md`：Jacobian、JVP/VJP、泰勒线性化、梯度流、EoS、plateau ODE
4. `0.3-probability-and-statistics.md`：标准误、配对 t、多重比较、熵/KL、$(V-1)/(2N)$ 的推导
5. `0.4-training-in-practice.md`：读懂题面每个字段、有效步长表、参数量公式、**瞎猜基线**
6. `0.5-kernels-and-ntk.md`：核、Mercer 谱、RKHS、NTK、lazy 判据、谱偏置、隐式偏置
7. `0.6-approximation-and-limits.md`：维度诅咒、Barron、深度分离、$n$-width、double descent 机制、四档纪律

每章都标了"承重位置"（对应正文哪一章哪一节），可以按需跳读。**时间只够一章就读 `0.5`**——它是 02/09 两章的地基。

### 路线 A：只想快速会判断（1 小时）

> 卡住了就回前置：`02` 的公式 → `0.1`+`0.5`；`07` 的题面字段 → `0.4`；`04` 的显著性 → `0.3`。

1. `00-cheatsheet.md`：一页纸判据
2. `01-what-training-is.md`：问题重述与四项分解
3. `02-exact-case.md`：唯一能精确算的情形
4. `07-case-studies.md`：四道真实题的判读

### 路线 B：想真正理解理论（半天，前置补齐后）

> 建议先走完路线 P 的 `0.1`/`0.2`/`0.5`，否则第 3、6、7 步会很吃力。

1. `00-cheatsheet.md`
2. `01-what-training-is.md`
3. `02-exact-case.md`（前置：`0.1` §3–§7、`0.5` §4–§6）
4. `03-budget-and-curves.md`：$T$、lr、学习曲线、plateau、EoS（前置：`0.2` §6–§8、`0.4` §4）
5. `04-generalization-and-validation.md`：$p>n$、double descent、验证集噪声（前置：`0.3` §3–§5、`0.6` §8–§9）
6. `05-function-and-architecture.md`：目标结构、逼近论、$n$-width、深度/宽度（前置：`0.6` §2–§6）
7. `06-classification-and-bigram.md`：CE、margin、熵地板（前置：`0.3` §7–§9、`0.4` §7–§8）
8. `07-case-studies.md`（前置：`0.4` 全章）
9. `09-how-to-build-answerer.md`：怎么把理论变成自动答题器（可选，偏工程；前置：`0.2` §4–§5、`0.5` §8–§9、§12）
10. `08-limits-and-next.md`：理论不能做什么（前置：`0.6` §11 的四档纪律）

### 路线 C：只想知道"这套东西够不够解题"

`00-cheatsheet.md` → `07-case-studies.md` → `08-limits-and-next.md`。想自己动手复现答题器再加 `09-how-to-build-answerer.md`（那一章需要 `0.2` §4 的 JVP/VJP 和 `0.4` §10 的 PyTorch 实操）。

---

## 3. 文档清单

### 前置（`0.0`–`0.6`）：读懂正文需要的工具

| 文件 | 回答的问题 |
|---|---|
| `0.0-prereq-map.md` | 我缺什么？30 题自测、六章依赖映射、四条补课路线、统一符号表 |
| `0.1-linear-algebra.md` | 特征分解、零空间、SVD、条件数、有效秩：$\sum_j(1-\eta\lambda_j)^{2T}a_j^2$ 和 $\lVert P_{\mathrm{null}}\theta^{*}\rVert^{2}$ 从哪来 |
| `0.2-calculus-and-gradients.md` | Jacobian、前向/反向 AD、泰勒线性化、梯度流、EoS 判据 $\eta\lambda_{\max}>2$、深网 plateau 的 $a^{2-2/L}$ |
| `0.3-probability-and-statistics.md` | $\mathrm{SE}=s/\sqrt k$、配对 $t$ 与 2.262、$\sqrt{2\ln K}$ 复用偏差、$\mathrm{CE}=H+\mathrm{KL}$ 给出地板、$(V-1)/(2N)$ 的推导与适用域 |
| `0.4-training-in-practice.md` | `d4 w256 leaky LN=FTTF AdamW/3e-5` 每个字段是什么、参数量公式（对 8 个真实候选验算通过）、有效步长表、瞎猜基线 |
| `0.5-kernels-and-ntk.md` | 核与特征映射、Mercer 谱、RKHS 范数与"学不到"、NTK $=JJ^\top$、lazy 判据、谱偏置、min-norm/max-margin |
| `0.6-approximation-and-limits.md` | $A$ 与 $n$-width 两个地板、维度诅咒 $n^{-s/d}$、Barron/深度分离、double descent 的机制与失效条件、四档纪律 |

### 正文（`00`–`09`）

| 文件 | 回答的问题 |
|---|---|
| `00-cheatsheet.md` | 一页纸：四项分解、谱滤波、判定流程、常见坑 |
| `01-what-training-is.md` | 训练到底在做什么？为什么训练误差为 0 不代表验证好？ |
| `02-exact-case.md` | 线性/核表达式：验证误差怎么精确算？零空间地板、可学窗口 |
| `03-budget-and-curves.md` | 为什么 $T$ 和 lr 决定短预算成败？plateau、EoS、early stopping |
| `04-generalization-and-validation.md` | $p>n$ 为什么不一定坏？double descent 什么时候适用？验证集怎么比才可信？ |
| `05-function-and-architecture.md` | 目标函数的结构如何决定深度/宽度？$n$-width 和信息极限 |
| `06-classification-and-bigram.md` | 分类和语言模型为什么不能套回归公式？margin 与熵地板 |
| `07-case-studies.md` | 用四道 v1.5 真题演示"先算再比" |
| `08-limits-and-next.md` | 哪些问题数学还没解决？怎么把理论和实测结合？ |
| `09-how-to-build-answerer.md` | 怎么搭理论答题器：切线性化替身、复刻官方协议、在线 JVP/VJP；500 题 83.2% 的实现细节 |

---

## 4. 约定

- 公式用 `$...$` / `$$...$$`（LaTeX）；代码、字段名、文件名、章节引用用 `等宽反引号`，例如 `R_te ≈ A + I + O(T) + F` 在正文里写作 $R_{\rm te}\approx A+I+O(T)+F$。
- $\approx$ 表示"在给定假设下的近似/主项"；$=$ 表示定义或严格等式。
- 所有结论都会标注它是 **严格定理 / 上界 / 渐近 / 启发式** 中的哪一类——这是本项目最重要的纪律（四档的判据与例子见 `0.6-approximation-and-limits.md` §11）。
- **符号**：前置章（`0.0`–`0.6`）与正文共用同一套符号，统一表在 `0.0-prereq-map.md` §5。有一处刻意的差别——前置章把 Gram 矩阵写成 $A_{\rm mat}$，因为正文用 $A$ 同时表示"近似误差项"和 $\Phi^T\Phi/n$，推导时容易混。读回正文时记得换回来。
- **编号**：带小数点的 `0.0`–`0.6` 是前置章，两位数的 `00`–`09` 是正文。上一级目录还有另一套 `01-formulation.md`…`05-formalization-and-coverage.md`（完整数学版），三套编号互不相同。
- **数值都验过**：前置章里的每个可运行代码块都在本机（`numpy 2.2.6`）实跑过，实测输出直接印在代码块下方，不写"预期约为"。
- 想看完整推导和文献，对应到上一级的 `01-formulation.md` … `05-formalization-and-coverage.md`，以及 `notes/`。

---

## 5. 一句话

**数学不能替你训练模型，但它能告诉你：现在这个验证误差是"表达不了"、"选错了"、"没走到"、还是"测不准"；以及换什么最可能有用。** 这本教程就是把这四句话讲清楚。
