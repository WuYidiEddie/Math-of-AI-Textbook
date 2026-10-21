# 01 · 把"拟合好坏"写成数学问题

> 日期：2026-10-19 ｜ Math of AI Textbook
> 本章目标：给出问题的严格表述、误差分解、以及"只用 $f$ 与 $\mathcal G$ 的性质判断验证集好坏"的判定流程。
> 阅读前提：微积分、线性代数、概率论；不需要机器学习背景。

---

## 0. 本章结论

1. **"拟合好坏"不是"训练误差小不小"，也不是"参数多不多"。** 在 $p>n$ 时训练误差可以做到 0，但它对验证误差没有信息。真正要问的是：
   $$
   R_{\rm te}(\theta_T)=\mathbb E_{x\sim P}\big[\ell(g_{\theta_T}(x),f(x))\big]
   $$
   的大小。

2. **对 $R_{\rm te}$ 做四分解**（在核/lazy 区可严格可加；一般情形是分析模板）：
   $$
   R_{\rm te}\;\approx\;\underbrace{A}_{\text{表达式类表达不了}}+\underbrace{I}_{\text{算法在数据上选错了函数}}+\underbrace{O(T)}_{\text{$T$ 步没走到}}+\underbrace{F}_{\text{验证/种子噪声}} .
   $$
   $A$ 由 $f$ 到 $\mathcal G$ 的距离决定；$I$ 由**隐式偏置 + 有限样本 aliasing** 决定；$O$ 由**核谱 + $T,\eta$** 决定；$F$ 由验证集大小与种子数决定。

3. **判据（核/lazy 区，MSE，无标签噪声）**：把 $f$ 投影到表达式在初始化处诱导的核 $K$ 的特征基上，
   $$
   f=\sum_j a_j\varphi_j,\qquad K(x,x')=\sum_j\lambda_j\varphi_j(x)\varphi_j(x'),
   $$
   则
   $$
   \boxed{\;R_{\rm te}(T)\approx \sum_{j:\ \eta\lambda_jT\ll1}a_j^2\;+\;\sum_{j\in\text{kernel tail}}a_j^2\;+\;F\;}
   $$
   **"拟合好坏"= 目标 $f$ 的能量分布 $\{a_j^2\}$ 与表达式的可学窗口 $\{\eta\lambda_jT\gtrsim1\}$ 的重叠程度**。这就是 spectral alignment（谱对齐）判据。

4. **$p>n$ 不是决定量。** 有效的"容量"是核的有效秩/函数流形的维数（去掉对称性、规范化等不改变函数的自由度），不是参数个数。$p$ 只通过核谱、条件数、隐式偏置进入；在核/lazy 区，$p>n$ 只保证 $K_{\rm tr}$ 可逆（插值可达），**不保证泛化**。

5. **除 bigram 外，本设定没有标签噪声**（v1.5 的 univar/mvar 是符号表达式精确求值，spiral/xor/tabular 是确定性规则；bigram 是固定随机 bigram law，有不可约熵 $H(Y|X)>0$，是例外）。所以对 5 个确定性族，经典 double descent / benign overfitting 的"噪声方差"故事**不是主体**。无噪声时：
   - 核/lazy 区的总体 $R_{\rm te}(T)$ 对 $T$ 单调下降，最终停在地板 $A+I$；这里**没有标签噪声方差项**，但有限 $n$ 的**设计采样项 $I$** 仍在（精确标签不消除它；见 02 §2.2b）；
   - 真正可能"训久反而变差"的来源是：有限样本 aliasing、特征学习、分类边界复杂度、以及大 lr 的 Edge-of-Stability——这些都不是"拟合了标签噪声"。
   - 这一条直接改变了"如何判断验证集好坏"：先判断主导项是 $A,I,O$ 还是 $F$，再谈"过拟合"。

---

## 1. 严格设定

### 1.1 数据与目标

- 输入空间 $\mathcal X\subseteq\mathbb R^d$，数据分布 $P$（训练与测试同分布）。
- **目标函数** $f:\mathcal X\to\mathcal Y$ 是**确定性**的（v1.5 中：一元/多元回归是符号表达式的精确求值；分类是规则标签；**bigram 例外**：固定随机 bigram law，有不可约熵）。
- 训练集 $S=(x_1,\dots,x_n)\sim P^{\otimes n}$，无噪声标签 $y_i=f(x_i)$。
- 测试/验证集 $S_{\rm te}=(x_1^{\rm te},\dots,x_{n_{\rm te}}^{\rm te})\sim P^{\otimes n_{\rm te}}$，独立于 $S$。

### 1.2 表达式（模型）

- 表达式类：$\mathcal G=\{g_\theta:\theta\in\Theta\subseteq\mathbb R^p\}$，$p\gg n$。
- 元数化：$g_\theta(x)$ 对 $\theta$ 可微；记初始化 $\theta_0\sim\pi$。
- 例子：
  - 线性/多项式表达式：$g_\theta(x)=\sum_{k=1}^{p}\theta_k\phi_k(x)$（固定特征基 $\phi_k$）。
  - 神经网络：$g_\theta$ 是若干层线性映射与逐点非线性的复合；$p$ 是全部权重/偏置。
- **架构**（深度、宽度、激活、LayerNorm、残差、参数化方式）决定了函数类 $\mathcal G$ 以及它在初始化处的几何。

### 1.3 算法

- 损失 $\ell$：回归用平方损失，分类用交叉熵。
- 经验风险 $\hat R_n(\theta)=\frac1n\sum_{i=1}^n\ell(g_\theta(x_i),f(x_i))$。
- GD/SGD：$\theta_{t+1}=\theta_t-\eta\nabla\hat R_n(\theta_t)$（或小批量版本），步长 $\eta$，$T$ 步，输出 $\theta_T$。
- 随机性来源：初始化 $\theta_0$、小批量采样、数据采样。

### 1.4 要预测的量

**总体验证误差**
$$
R_{\rm te}(\theta_T)=\mathbb E_{x\sim P}\big[\ell(g_{\theta_T}(x),f(x))\big].
$$

**题库实际测量的量**（固定测试集、10 个种子）：
$$
\hat R_{\rm te}(\theta_T)=\frac1{n_{\rm te}}\sum_{i=1}^{n_{\rm te}}\ell(g_{\theta_T}(x_i^{\rm te}),f(x_i^{\rm te})),
\qquad
\bar R=\frac1{10}\sum_{s=1}^{10}\hat R_{\rm te}(\theta_T^{(s)}).
$$

理论要预测的是 $R_{\rm te}$ 或 $\mathbb E[\bar R]$；$\bar R$ 只是它的一个带噪估计。**把"验证集上的值"与"总体风险"分开，是本章的第一条纪律。**

---

## 2. "拟合好坏"的五个层次

判断一个表达式拟合得好不好，必须依次问五个问题（任何一个答"不好"，后面的问题就没有意义）：

| 层 | 问题 | 数学对象 | 主导时该改什么 |
|---|---|---|---|
| **L0 表达** | 表达式类里有没有一个函数接近 $f$？ | 逼近误差 $A=\inf_{g\in\mathcal G}\|f-g\|^2$；n-width 下界 | 换函数类/特征/深度宽度；加先验 |
| **L1 识别** | 即使有，$n$ 个样本能把 $f$ 与类里其他函数区分开吗？ | 信息极限（n-width/optimal recovery）；设计点几何 | 加样本、改采样分布、加正则化 |
| **L2 选择** | 在所有能拟合数据的函数里，算法会选哪一个？它离 $f$ 多远？ | 隐式偏置 + 有限样本 aliasing：$I=\|f-g_{X,\infty}\|^2$ | 改参数化/初始化/正则化；改核 |
| **L3 优化** | $T$ 步之内，算法走得到那个函数吗？ | 优化误差 $O(T)=\|g_{X,\infty}-g_{\theta_T}\|^2$ | 改 lr/优化器/T/条件数/归一化 |
| **L4 测量** | 验证集上的估计可信吗？候选间差异超出噪声吗？ | $F$：浓度、配对差、10 seeds | 加测试点/种子；报告置信区间 |

**关键**：$p>n$ 只直接影响 L1/L2 的"可辨识性"，不直接决定 L0/L3/L4。把"参数多于样本"当成"会过拟合"是层次错误。

---

## 3. 四类误差的分解

设 $g_{\mathcal G}^*$ 是 $\mathcal G$ 中对 $f$ 的最优逼近（infimum 可达时），$g_{X,\infty}=\lim_{T\to\infty}g_{\theta_T}$ 是算法在样本 $X=(x_1,\dots,x_n)$ 上的极限输出（若存在；例如最小范数插值）。

定义：

$$
\begin{aligned}
A&=\|f-g_{\mathcal G}^*\|_{L^2(P)}^2 &&\text{（approximation，表达式类的能力）},\\
I&=\|g_{\mathcal G}^*-g_{X,\infty}\|_{L^2(P)}^2 &&\text{（identification：算法极限与最优逼近的距离；含隐式偏置+aliasing）},\\
O(T)&=\|g_{X,\infty}-g_{\theta_T}\|_{L^2(P)}^2 &&\text{（optimization：有限 }T\text{ 的滞后）},\\
F&=\text{验证集估计的波动} &&\text{（finite test / seeds）}.
\end{aligned}
$$

由三角不等式：
$$
\|f-g_{\theta_T}\|\le \|f-g_{\mathcal G}^*\|+\|g_{\mathcal G}^*-g_{X,\infty}\|+\|g_{X,\infty}-g_{\theta_T}\|
\;\Longrightarrow\;
R_{\rm te}(\theta_T)\le 3\big(A+I+O(T)\big).
$$

在核/lazy 区（见 02），这三项**可以精确计算且近似可加**；在一般非线性网络里，这个分解是**分析模板**（三项会交互，不能假设可加），但每一项仍可单独给界/估计。

### 3.1 为什么不是"偏差-方差"

经典偏差-方差分解把误差写成"模型偏差² + 估计方差"，其中方差来自**标签噪声**。本设定（除 bigram 外）$y=f(x)$ 无噪声，方差项的结构不同：

- 有限 $n$ 的"方差"主要来自**样本位置/初始化**（哪 $n$ 个点、哪个随机种子），而不是标签噪声；
- 在核/lazy 区无噪声时，$I$ 是**插值/aliasing 误差**（由核的 power function 与 $f$ 的 RKHS 范数控制），不是噪声方差；
- 因此 $p>n$ 时"更多参数 → 更少偏差但更多方差"的经典叙事**不自动成立**。更多参数可能只改变核谱与隐式偏置，既不加偏差也不加方差；也可能因条件数变差而增加 $O$。

### 3.2 $A,I,O$ 的严格版本（核/lazy 区）

记核 $K$（由初始化 Jacobian 诱导），RKHS $\mathcal H_K$，$P_{\mathcal H}$ 为到 $\mathcal H_K$ 的投影。则：

- $A=\|(I-P_{\mathcal H})f\|^2$（核 tail；若 $f\in\mathcal H_K$ 则 $A=0$）。
- $I=\|f-s_X\|^2$，其中 $s_X$ 是核最小范数插值（$s_X(x)=K(x,X)K(X,X)^{-1}y$）。
- $O(T)=\|s_X-g_{\theta_T}\|^2$，而 $g_{\theta_T}$ 是 $s_X$ 的谱滤波版本（02 §3）。
- $F$ 与 $n_{\rm te}$、种子数、以及 $\ell$ 的方差有关（§7）。

对于核插值，$I$ 有一个**只用 $f$ 的 RKHS 范数与设计点几何**的严格上界（power function）：
$$
|f(x)-s_X(x)|\le P_X(x)\,\|f\|_{\mathcal H_K},
\qquad
P_X(x)^2=K(x,x)-K(x,X)K(X,X)^{-1}K(X,x).
$$
于是
$$
I\le \|f\|_{\mathcal H_K}^2\;\mathbb E_{x\sim P}\big[P_X(x)^2\big].
$$
这是本目录最重要的"a priori 判据"之一：**验证误差可以由 $f$ 的范数 × 核/设计的 power function 来控制**，不需要训练、不需要标签噪声模型。

---

## 4. 信息极限：$n$ 个样本最多能告诉你多少

即使表达式类无限大、优化完美，$n$ 个样本也携带有限信息。纯数学的刻画是 **n-width / optimal recovery**。

给定函数类 $\mathcal F$（例如 Sobolev 球 $W^{s,2}$、RKHS 单位球）与 $n$ 个点上的值，最佳恢复误差
$$
E_n(\mathcal F)=\inf_{\text{恢复算法 }R}\ \sup_{f\in\mathcal F}\ \|f-R(f(x_1),\dots,f(x_n))\|_{L^2(P)}
$$
满足经典下界/上界（例如 $d$ 维 $s$ 阶 Sobolev：$E_n\asymp n^{-s/d}$；RKHS 由特征值衰减决定）。

**推论**：

1. $R_{\rm te}$ **不可能低于** n-width 给出的信息极限；任何"更好的表达式"都不能突破它。
2. 若实测/理论误差远高于 n-width，则问题不在信息，而在 $A$、$I$ 或 $O$：表达式选错、隐式偏置不好、或没优化到位。
3. 若误差已接近 n-width，则表达式/算法已经"接近最优"，再换架构的收益有上限。
4. **$p\gg n$ 不增加信息**；它只改变函数类的几何与算法选择哪一 interpolant。这就是为什么"参数多"既可能无害也可能有害：取决于它把 $I$ 往哪个方向推。

> 注：n-width 是"类内最坏情形"的极限。对**单个已知的 $f$**，最佳恢复误差可以远小于 n-width；这时应使用 $f$ 的具体性质（光滑度、谱系数、结构），而不是最坏情形率。本目录同时使用两种视角：worst-case（保证）与 target-specific（预测）。

---

## 5. 表达式侧：哪些性质进入哪一项

| 表达式的性质 | 数学对象 | 主要进入 | 说明 |
|---|---|---|---|
| 函数类/表达能力 | $\mathcal G$、逼近率、线性区域数、深度分离 | $A$ | 深度/宽度/激活决定能表达多复杂的 $f$ |
| 初始化 Jacobian / NTK | $J=\partial_\theta g_{\theta_0}$，$K=JJ^\top$ | $A,I,O$ | 核谱 $\{\lambda_j\}$ 决定可学窗口与隐式偏置 |
| 核的有效秩/条件数 | $\mathrm{tr}(K)^2/\mathrm{tr}(K^2)$、$\lambda_{\max}/\lambda_{\min}$ | $I,O$ | 有效秩小→估计/aliasing 小；条件数大→优化慢 |
| 参数化/初始化尺度 | NTK 缩放、$\mu$P、lazy/rich 判据 | $I,O$ | 小初始化→lazy（核可描述）；大初始化→特征学习（无闭式） |
| 隐式偏置 | min-norm / max-margin / flatness | $I$ | 算法在插值解中选哪个；由参数化与算法决定 |
| 对称性/规范化 | 参数空间的规范自由度 | "有效 $p$" | 很多参数方向不改变函数；参数个数高估容量 |
| 归一化/残差/深度 | 条件数、PL 常数、$\lambda_{\max}$ | $O$ | 主要影响可训练性，而非表达能力 |
| 宽度 | 随机特征数、NTK 极限 | $A,O$ | 宽→核极限；对 $n$ 固定时有效容量未必增加 |

**要点**：架构不是"容量旋钮"，而是**同时改变核谱、隐式偏置、条件数**的三合一对象。判断拟合好坏时，不能只问"参数够不够"，要问"核谱与目标的谱对不对齐、算法选的函数离目标多远、T 步够不够"。

---

## 6. 函数侧：哪些性质进入哪一项

| 目标 $f$ 的性质 | 数学对象 | 主要进入 | 说明 |
|---|---|---|---|
| 光滑度/频率含量 | Sobolev/Besov 指数、Fourier 衰减 | $A,O$ | 高频分量对应小 $\lambda_j$，学得慢（spectral bias） |
| 在核基上的能量分布 | $a_j=\langle f,\varphi_j\rangle$ | $A,O$ | $a_j^2$ 与可学窗口的重叠决定验证误差 |
| 是否在 RKHS 内 / 范数大小 | $\|f\|_{\mathcal H_K}$、source condition | $A,I$ | 范数控制 power function 上界；不在核内→tail |
| 结构（加性/低阶交互/复合/分段） | Barron 范数、交互阶、深度分离 | $A$ | 决定需要多深的表达式、样本复杂度是否随维度爆炸 |
| 有效维度/内在维度 | 谱的有效秩、active subspace | $A,E$ | 低有效维→高维问题可学 |
| 分类边界复杂度/ margin | 边界曲率、margin 分布、类重叠 | $A,I$ | CE 没有闭式；margin 是分类版的"谱对齐" |
| 尺度/条件数 | $\|f\|_{L^2}$、动态范围 | $F$ | 影响验证噪声与数值 |

**核心量**：目标在模型核基上的"谱能量" $a_j^2$ 与核特征值 $\lambda_j$ 的联合分布。**不需要知道 $f$ 的显式公式**；只需要在训练的 $n$ 个点/一个网格上求值，就能近似 $\{(a_j,\lambda_j)\}$（核矩阵特征分解）。

---

## 7. 算法侧：$T,\eta$、优化器、批大小

### 7.1 有限步 = 谱滤波

在核/lazy 区，从零初始化、全批量 GD、MSE 下，预测的精确形式是（02 §3）：
$$
g_{\theta_T}(x)=\sum_j\big[1-(1-\eta\lambda_j)^T\big]\,a_j\,\varphi_j(x),
$$
因此
$$
R_{\rm te}(T)=\sum_j(1-\eta\lambda_j)^{2T}a_j^2
\qquad(\text{总体核，无噪声}).
$$
**可学窗口**：$\eta\lambda_jT\gtrsim1$；时间常数 $\tau_j=1/(\eta\lambda_j)$。
- $T$ 小、$\eta$ 小：只有大 $\lambda_j$（低频/易学方向）被学到；
- 小 $\lambda_j$（高频/难方向）需要 $T\gtrsim1/(\eta\lambda_j)$；
- 这就是"一定步数后拟合好坏"的精确回答（在核/lazy 区）。

### 7.2 优化器与有效步长

- SGD：$\eta_{\rm eff}\approx\eta\cdot|\bar g|$（依赖梯度尺度）；
- Adam/RMSprop：逐坐标归一化，每步位移 $\approx\eta$（与梯度尺度弱相关）；
- Adagrad：位移 $\propto\sqrt T$。
这些"有效步长"的差异会改变上式中的 $\eta$（或等价地改变可学窗口）。严格理论里，Adam 不是固定核滤波，闭式公式不适用；但可以把 Adam 视为**时变预条件**，用数值模拟（在初始化 Jacobian 上）预测，而不必真训练。

### 7.3 批大小/随机性

小批量噪声 $\propto\eta\sigma_g^2/B$；它带来 $F$ 与轻微正则化，但在无标签噪声的确定性任务上不是主导项。批大小主要影响稳定性与泛化界的常数。

### 7.4 Edge of Stability

当 $\eta\lambda_{\max}(\nabla^2\hat R)\gtrsim2$，GD 进入非单调/振荡区（EoS）。此时：
- 损失曲线出现抖动/回升，但不等于验证误差变差；
- 架构的"方向翻转"（宽度/深度偏好随 lr 变化）可由 EoS 相位解释；
- 形式化：$\lambda_{\max}\approx2/\eta$ 是稳定边界；越过它后 $T$ 步输出对 $\eta$ 极度敏感，理论预测退化为相位判断而非精确值。

---

## 8. 判定流程（核心）

给定 $f,\mathcal G,\mathcal A,n,T$，按以下顺序判断（不是先看 $p$ vs $n$）：

```
Step 1  信息极限：n-width / optimal recovery 下界
        → 若目标类已知：误差不可能低于 E_n；记录下界 L_info。
Step 2  近似项 A：f 到 G / 核空间的 tail
        → 在训练点上算核矩阵与目标投影；A ≈ 核 tail 能量。
        → 若 A 已经 > 可接受误差：换表达式类/特征，结束。
Step 3  识别项 I：算法极限 s_X 与 f 的距离
        → 核/lazy：s_X = 最小范数核插值；I = ||f-s_X||²，可用 power function 上界。
        → 检查隐式偏置是否与 f 对齐（min-norm/max-margin）。
        → 若 I 主导：改核/正则化/参数化/初始化。
Step 4  优化项 O(T)：T 步是否走到 s_X
        → 谱滤波公式（核/lazy）或 PL/Saxe/EoS 估计。
        → 若 O 主导：加 T、调 lr/优化器、改条件数（LN/残差/初始化）。
Step 5  验证项 F：n_te、种子数、配对差
        → 计算 se；若候选差 < 2–3 se：题目是刀刃，理论预测不可分辨。
Step 6  比较 A,I,O,F 的量级，报告主导项与误差上界/预测值。
```

**两条纪律**：

1. 不要用训练误差代替验证误差；$p>n$ 时训练误差≈0 没有信息。
2. 不要用"参数多"解释一切；先算 $A,I,O$ 的相对大小。

---

## 9. 验证集的两种含义

### 9.1 作为"总体风险的估计"

给定 $\theta_T$，验证误差 $\hat R_{\rm te}$ 是 $R_{\rm te}$ 的无偏估计（测试点 iid 时）。其标准误：
$$
\mathrm{se}_{\rm test}=\sqrt{\mathrm{Var}\big(\ell(g_{\theta_T}(x),f(x))\big)/n_{\rm te}}.
$$
题库 $n_{\rm te}=256$（回归）或 200（bigram）等；不可忽略。

### 9.2 作为"候选比较"的测量

题库是**同一数据集、同一测试集、同一 10 seeds** 下比较 3 个候选。此时应看**配对差**：
$$
\Delta=\bar R_A-\bar R_B,\qquad
\mathrm{se}_\Delta=\frac{s_{d}}{\sqrt{10}},\quad d_s=\hat R_A^{(s)}-\hat R_B^{(s)}.
$$
测试集噪声在配对差中**抵消**（同一批测试点），这是配对设计的好处。判定：
$$
|\Delta|>t_{9,0.975}\,\mathrm{se}_\Delta\approx2.26\,\mathrm{se}_\Delta
\;\Longrightarrow\;
\text{在 5\% 水平显著}.
$$
若候选的理论预测差小于这个阈值，题目本身就是"刀刃"，任何理论的预测都不应被当成确定答案。

### 9.3 验证集复用偏差

若用同一验证集反复挑选模型/超参，会系统性乐观：
$$
\mathbb E[\max_{k=1}^{K}\hat R_k]-\min_k R_k\approx \sigma\sqrt{2\log K}
\quad(\text{独立高斯近似}).
$$
`v1.5` 的候选指标是离线预计算好的，题库层面没有复用偏差；但**用验证集做二次筛选**（如挑阈值/挑规则）会重新引入它。

---

## 10. 最小可解示例：线性表达式 + $p>n$ + GD

这是"表达式拟合函数"的最简数学模型，所有量可手算。

设 $\phi(x)\in\mathbb R^p$ 是固定特征，$g_\theta(x)=\theta^\top\phi(x)$。记 $\Phi\in\mathbb R^{n\times p}$ 为训练特征矩阵，$y=\Phi\theta^*$（目标在特征空间内，无噪声），测试点特征 $\phi(x)$。

**GD 精确递推**（平方损失、步长 $\eta$）：
$$
\theta_{t+1}-\theta^*=(I-\eta\Phi^\top\Phi)(\theta_t-\theta^*).
$$
因此
$$
\theta_T-\theta^*=(I-\eta\Phi^\top\Phi)^T(\theta_0-\theta^*).
$$
$\Phi^\top\Phi$ 的零空间（$p>n$ 时维数至少 $p-n$）上特征值为 0，对应分量**永不改变**：
$$
\text{不可约地板}=\|P_{\rm null}(\theta_0-\theta^*)\|^2.
$$
在测试分布下（各向同性）：
$$
R_{\rm te}(T)=\big\|(I-\eta\Phi^\top\Phi)^T(\theta_0-\theta^*)\big\|^2.
$$

**读法**：

- 若 $\theta^*$ 恰好没有零空间分量（目标在行空间内），且 $T\to\infty$，则 $R_{\rm te}\to0$——**无论 $p$ 多大**；
- 若 $\theta^*$ 有零空间分量（目标需要"样本没告诉你的方向"），则即使训练误差为 0、$T\to\infty$，验证误差也不为 0——这是 **$p>n$ 的真正风险**：不是"参数多"，而是"目标在不可辨识子空间里有能量"；
- 初始化 $\theta_0$ 也进入地板；从零初始化时地板 = $\|P_{\rm null}\theta^*\|^2$。

这个例子说明：**判断拟合好坏，要比较的是目标在"可辨识子空间"与"不可辨识子空间"的能量，而不是参数个数与样本数的比值。** 把特征换成核特征/神经网络 Jacobian，这个结论的形式不变（02 §3）。

---

## 11. 与"经验规律"的关系

旧项目从 12,537 次实验里归纳了"语境/宽度/深度/LN 的条件律"。在本章的框架下：

- 这些律大多是 $A,I,O$ 中某一项占主导时，**相图边界**的经验切面；
- 它们不是"定理"，因为：(i) 依赖采样器与网格；(ii) 没有给出 $A,I,O$ 的分解；(iii) 外推到新语境时无法判断是否仍处同一相；
- 数学理论的任务是**推导相边界**（n-width、谱边、优化时间、EoS），而不是拟合律的系数；
- 经验律仍有两个合法用途：**testbed 的测量**（检验理论预测）与**理论未覆盖时的弱先验**（明确标注低置信）。

---

## 12. 本章小结

1. "拟合好坏" = 总体验证误差 $R_{\rm te}$，不是训练误差、参数个数、或验证集均值本身。
2. 四分解：$A$（表达不了）+ $I$（选错函数）+ $O(T)$（没走到）+ $F$（测量噪声）。
3. 核/lazy 区：$R_{\rm te}(T)=\sum_j(1-\eta\lambda_j)^{2T}a_j^2$；"好坏"由**目标谱能量与可学窗口的重叠**决定。
4. 除 bigram（有不可约熵）外，无标签噪声时经典过拟合/double descent 不是主体；风险来自 aliasing、特征学习、边界复杂度、EoS。
5. 判定流程：信息极限 → $A$ → $I$ → $O$ → $F$；先找主导项，再谈改进。
6. 下一章（02）给出可精确求解的核/线性/深度线性情形；03 讨论前沿与开放问题；04 把 v1.5 当 testbed。
