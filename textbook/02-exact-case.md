# 02 · 唯一能精确算的情形：线性/核/惰性区

> 本章回答：在什么条件下，验证误差可以**不训练就算出来**？公式长什么样？为什么"预算 $T$"等于"只学了前几个模式"？
> 本章是全书的**推导中心**：后面的核公式、谱偏置、$p>n$ 的地板、early stopping，全都是这一节的一次次推广。
> 阅读方式：§0 是推导地图，§2–§7 是**逐步推导**（每一步都说清"为什么这一步成立"），§8 之后是推广与实测。

---

## 0. 推导地图：一条链，七步

整章其实只做了一件事——把"训练"这件事化简成"若干个互不干扰的一维递推"：

| 步骤 | 做了什么 | 得到什么 | 节 |
|---|---|---|---|
| 1 | 把一个训练集写成一个矩阵 $\Phi\in\mathbb R^{n\times p}$ | 平方损失变成矩阵二次型 | §2 |
| 2 | 求平方损失的梯度 | $\nabla L(\theta)=\hat\Sigma(\theta-\theta^{\ast})$ | §3 |
| 3 | 走一步 GD | 误差**自己跟自己递推**：$\theta_{t+1}-\theta^{\ast}=(I-\eta\hat\Sigma)(\theta_t-\theta^{\ast})$ | §4 |
| 4 | 迭代 $T$ 步 | $\theta_T-\theta^{\ast}=(I-\eta\hat\Sigma)^{T}(\theta_0-\theta^{\ast})$ | §4 |
| 5 | 换到 $\hat\Sigma$ 的特征基里 | 每个方向一条一维递推 $z_{T,j}=(1-\eta\lambda_j)^{T}z_{0,j}$ | §5 |
| 6 | 找出 $\lambda_j=0$ 的方向 | 零空间：至少 $p-n$ 个方向**冻结**，永远不学 | §6 |
| 6' | 追问"冻结"是不是被强制的 | 是推论不是约束；wd / Adam / 特征移动会打破它；初始化税 | §6.4–6.5 |
| 7 | 算残留 | 两个"地板"：逼近地板 $A$ 与辨识地板 $I$ | §7 |

> **记号提醒 1**：很多教材把 $\Phi^\top\Phi/n$ 写成 $A$。本项目不用这个记号——四项分解里 $A$ 已经被"approximation（逼近误差）"占用，混用会造成"推导时把两个 $A$ 当一个"的经典错误。本章一律写 $\hat\Sigma$（Gram 矩阵）。前置章 `0.1-linear-algebra.md` 写 $A_{\rm mat}$，指的是同一个东西。
>
> **记号提醒 2**：公式 $(I-\eta\hat\Sigma)^{T}$ 里的上标 $T$ 是**步数**，不是转置！转置在本章一律写成 $\top$（如 $\Phi^\top$）。这是本章最容易看错的地方，念出来就不会错：$T$ 步以后。

---

## 1. 为什么先看线性/核

一般神经网络的训练是高度非线性的，没有闭式。但有一类情形可以严格求解：

- 表达式对参数是线性的：$g_\theta(x)=\theta^\top\varphi(x)$（固定特征）；
- 或者网络处于**惰性区（lazy regime）**：训练中特征几乎不动，网络等价于"初始化处的核方法"。

这两类情形覆盖了 v1.5 的一部分候选（尤其小 lr、短预算、宽网），也是理解其余情形的基准。

**纪律**：先查惰性判据；不满足时，核公式只是"基线"，不是精确预测。

---

## 2. 设定：把一个训练集写成一个矩阵

### 2.1 模型

固定一个特征映射 $\varphi\colon\mathcal X\to\mathbb R^{p}$，表达式对参数是线性的：

$$
g_\theta(x)=\theta^\top\varphi(x),\qquad \theta\in\mathbb R^{p}.
$$

$p$ 是"参数个数"（特征维数），$n$ 是训练点个数。

### 2.2 数据矩阵

$n$ 个训练点 $x_1,\dots,x_n$，标签 $y_i=f(x_i)$。把它们**竖着摞成矩阵**：

$$
\Phi=\begin{pmatrix}\varphi(x_1)^\top\\ \varphi(x_2)^\top\\ \vdots\\ \varphi(x_n)^\top\end{pmatrix}\in\mathbb R^{n\times p},
\qquad
y=\begin{pmatrix}y_1\\ y_2\\ \vdots\\ y_n\end{pmatrix}\in\mathbb R^{n},
\qquad
y=\Phi\theta^{\ast}\ \ (\text{无噪声时}).
$$

这样写的好处：$n$ 个点的预测可以一次算完——$\Phi\theta$ 的第 $i$ 个分量就是 $g_\theta(x_i)$。

$\Phi$ 的第 $i$ 行是**一个训练点**的特征，$\Phi$ 的列是**一个特征**在 $n$ 个点上的取值。后面说的零空间、行空间，指的都是**参数空间 $\mathbb R^p$** 里的方向（$\mathrm{null}(\Phi)$、$\mathrm{row}(\Phi)\subset\mathbb R^p$），和 $\mathbb R^n$ 里的列空间不是一回事——这一点在 §6 会用到。

### 2.3 平方损失：它是什么，为什么是它

经验风险（平方损失）：

$$
\hat R_n(\theta)=\frac1n\sum_{i=1}^{n}\big(g_\theta(x_i)-y_i\big)^2
=\frac1n\lVert\Phi\theta-y\rVert_2^2 .
$$

本章后面一律用带 $\tfrac12$ 的版本，记

$$
L(\theta):=\frac1{2n}\lVert\Phi\theta-y\rVert_2^2 .
$$

**为什么要平方？** 三个理由，第三个才是本章能写闭式的原因：

1. **统计上**：如果标签带噪声 $y=f(x)+\varepsilon$、且 $\varepsilon$ 的均值为 $0$ 且对称，那"最好的预测"是条件期望 $E[y\mid x]$。让 $\theta$ 去最小化**平方**损失，恰好收敛到条件期望（最小化绝对值损失则收敛到条件中位数）。数值检验（`verification/exact_case_derivation.py` 的 C8b）：

   ```
   true slope = 3.0 | noise = Exponential(1)-1, mean = -0.0009, median = -0.3060
   squared-loss fit : slope = 2.9798, intercept = -0.0007   <- targets E[y|x] = 3x + 0
   L1 fit (IRLS)    : slope = 2.9714, intercept = -0.3063   <- targets median[y|x] = 3x + (-0.3060)
   ```

   同一个带偏斜噪声的数据，平方损失找回**均值**（截距 $\approx 0$），绝对值损失找回**中位数**（截距 $\approx-0.31$）。所以"用平方"是一个**建模选择**，不是"平方更好"。

2. **几何上**：平方损失属于"凸 + 梯度对参数仿射"这一类（$L$ 是二次型）——正因如此，下一步梯度才能写成 $\hat\Sigma(\theta-\theta^{\ast})$ 这样一个**线性**表达式。

3. **可解性**：二次型有一阶最优条件（正规方程），可以解出闭式；L1 没有（它要解分位数回归）。

**为什么带 $\tfrac12$ 和 $\tfrac1n$？** $\tfrac1n$ 是"按样本平均"，让梯度不随 $n$ 缩放（否则 $n$ 越大步长要越小）；$\tfrac12$ 是为了让求导后不出现因子 $2$（$\frac{\mathrm d}{\mathrm d\theta}\frac12\lVert u\rVert^2=u^\top\frac{\mathrm du}{\mathrm d\theta}$）。**因子 $2$ 并进步长 $\eta$ 里**——所以本章的 $\eta$ 和别人论文里的 lr 差一个 2，读别人的公式时要留意。

本章先不带正则项（weight decay 是第 03 章的话题），也不带 minibatch（随机性会在第 09 章的答题器里处理）。

---

## 3. 第一步：平方损失的梯度

求导，两行：

$$
\nabla_\theta L(\theta)
=\frac1{2n}\cdot 2\,\Phi^\top(\Phi\theta-y)
=\frac1n\Big(\Phi^\top\Phi\,\theta-\Phi^\top y\Big)
=\hat\Sigma\,\theta-\frac1n\Phi^\top y,
\qquad \hat\Sigma:=\frac1n\Phi^\top\Phi .
$$

**无噪声时** $y=\Phi\theta^{\ast}$，于是 $\frac1n\Phi^\top y=\frac1n\Phi^\top\Phi\theta^{\ast}=\hat\Sigma\theta^{\ast}$，代进去：

$$
\boxed{\ \nabla_\theta L(\theta)=\hat\Sigma\,(\theta-\theta^{\ast})\ }
$$

> **关键一步**：梯度变成了"当前点与不动点之差"再过一个**固定的矩阵** $\hat\Sigma$。$L$ 是二次的，所以 $\nabla L$ 是**仿射**的——这是后面所有闭式的根源。
>
> **$\theta^{\ast}$ 是谁？** 它是**正规方程** $\Phi^\top\Phi\theta^{\ast}=\Phi^\top y$ 的任意一个解，也就是"训练数据能看出来的那个最优参数"。当 $p>n$ 时它**不唯一**（零空间方向随便取），但下面的递推对**任何**这样的 $\theta^{\ast}$ 都成立——这是本章少有的、可以随便选的地方。
>
> **$\hat\Sigma$ 是什么？** 训练数据的 Gram 矩阵（除以 $n$）。它**不含标签**，只含特征；它决定"训练能看出哪些方向"。

数值检验（脚本 C1）：

```
max |grad_analytic - grad_finite_difference|      = 1.797e-09
max |grad_analytic - A (theta - theta*)|           = 6.661e-16
A theta* - (1/n) Phi^T y      = 3.331e-16  (0 => theta* is a fixed point)
```

（脚本里把 $\hat\Sigma$ 记作 `A`，与本章记号对应。）

---

## 4. 第二步：GD 一步 = 一次线性代数运算

梯度下降：$\theta_{t+1}=\theta_t-\eta\nabla L(\theta_t)$。代入 §3：

$$
\theta_{t+1}=\theta_t-\eta\hat\Sigma(\theta_t-\theta^{\ast})
=\big(I-\eta\hat\Sigma\big)\theta_t+\eta\hat\Sigma\theta^{\ast}.
$$

把 $\theta^{\ast}$ 挪到左边（技巧：两边同时减 $\theta^{\ast}$，注意 $\hat\Sigma\theta^{\ast}=\hat\Sigma\theta^{\ast}$）：

$$
\theta_{t+1}-\theta^{\ast}=\big(I-\eta\hat\Sigma\big)\big(\theta_t-\theta^{\ast}\big).
$$

**读法：误差自己跟自己递推，训练数据消失了。** 递推里只有 $\hat\Sigma$（数据）和 $\eta$（步长）；$\theta^{\ast}$ 只作为"误差的参考点"出现。

迭代 $T$ 步（归纳法：$T$ 次矩阵乘就是矩阵幂）：

$$
\boxed{\ \theta_T-\theta^{\ast}=\big(I-\eta\hat\Sigma\big)^{T}\big(\theta_0-\theta^{\ast}\big)\ }
$$

数值检验（脚本 C2，$p=100$、$n=20$、$\eta=0.2$、$T=200$）：

```
max |GD run - closed form| (theta_0 = 0, T=200)   = 1.162e-14
max |GD run - closed form| (theta_0 != 0)           = 1.671e-14
```

**这就是"训练前就能算"的来源**：给定 $\hat\Sigma$ 和 $\theta^{\ast}$，$T$ 步后的参数是一个矩阵幂，可以直接算（§7 会演示它连测试误差都能预测）。

---

## 5. 第三步：换到特征基里 → 每个方向各跑各的

$\hat\Sigma$ 对称半正定（为什么：$v^\top\hat\Sigma v=\frac1n\lVert\Phi v\rVert^2\ge 0$），所以可以做谱分解（前置章 `0.1` §3）：

$$
\hat\Sigma=U\Lambda U^\top,\qquad \Lambda=\mathrm{diag}(\lambda_1,\dots,\lambda_p),\quad \lambda_1\ge\dots\ge\lambda_p\ge 0 .
$$

**换基**：令 $z_t:=U^\top(\theta_t-\theta^{\ast})$。$U$ 是正交阵，把 §4 的式子左乘 $U^\top$、右乘 $U$，矩阵 $I-\eta\hat\Sigma$ 在新基下变成 $I-\eta\Lambda$（对角）：

$$
z_{t+1}=(I-\eta\Lambda)z_t
\quad\Longrightarrow\quad
z_{t+1,j}=(1-\eta\lambda_j)\,z_{t,j}
\quad\Longrightarrow\quad
\boxed{\ z_{T,j}=(1-\eta\lambda_j)^{T}z_{0,j}\ } .
$$

**这就是全章的核心图像**：$p$ 个互不干扰的一维递推，第 $j$ 个方向的进度由标量 $(1-\eta\lambda_j)^{T}$ 决定。

由此得到三个立刻可用的读数：

1. **稳定性**：第 $j$ 个方向收敛当且仅当 $\lvert 1-\eta\lambda_j\rvert<1$，对所有 $j$ 成立即
   $$
   0<\eta<\frac{2}{\lambda_{\max}} .
   $$
   这就是"步子迈大了会炸"的精确来源（前置章 `0.2` §6 的 EoS 判据）。数值检验（脚本 C6）：

   ```
   eta =   0.1000 | eta*lambda_max =  0.962 | |1-eta*lambda_max| = 0.038 | ||theta_500|| = 2.583e-01
   eta =   0.2059 | eta*lambda_max =  1.980 | |1-eta*lambda_max| = 0.980 | ||theta_500|| = 2.583e-01
   eta =   0.2100 | eta*lambda_max =  2.020 | |1-eta*lambda_max| = 1.020 | ||theta_500|| = 5.248e+02
   eta =   0.5000 | eta*lambda_max =  4.809 | |1-eta*lambda_max| = 3.809 | ||theta_500|| = 2.670e+12   (diverges)
   PASS C6: stable iff 0 < eta < 2/lambda_max = 0.2079
   ```

2. **时间常数**：当 $\eta\lambda_j\ll 1$ 时 $(1-\eta\lambda_j)^{T}\approx e^{-\eta\lambda_jT}$，所以第 $j$ 个模式的**时间常数**是 $\tau_j=1/(\eta\lambda_j)$，**可学窗口**是 $\eta\lambda_jT\gtrsim 1$。$T$ 步之后"学没学会"，只看这一个数。

3. **零特征值不衰减**：$\lambda_j=0$ 时因子是 $(1-0)^{T}=1$——**分量原地不动**。下一节专门讲这件事。

数值检验（脚本 C3）：

```
lambda range: [-0.0000, 9.6179] | #lambda>1e-10: 20
max |U^T (theta_T - theta*) - diag((1-eta*lambda)^T) U^T (theta_0 - theta*)| = 1.937e-14
lambda_min (should be ~0): -4.032e-15 | (1-eta*lambda_min)^T = 1.000000  <- null directions do NOT decay
```

---

## 6. 零空间：至少 $p-n$ 个方向永远不学

### 6.1 零空间是同一个

$\hat\Sigma$ 的零空间和 $\Phi$ 的零空间**是同一个**。证明一行：

$$
v\in\mathrm{null}(\hat\Sigma)\ \Longrightarrow\ 0=v^\top\hat\Sigma v=\frac1n\lVert\Phi v\rVert^2\ \Longrightarrow\ \Phi v=0\ \Longrightarrow\ v\in\mathrm{null}(\Phi).
$$

（反方向显然。）所以 $\mathrm{null}(\hat\Sigma)=\mathrm{null}(\Phi)=\{\,\Phi v=0\,\}$：这些参数方向的改动**不改变任何一个训练点的预测**——数据看不见它们。

### 6.2 有多少个

秩-零化度定理（前置章 `0.1` §5.2）：

$$
\dim\mathrm{null}(\Phi)=p-\mathrm{rank}(\Phi)\ \ge\ p-n .
$$

$p>n$ 时，**至少 $p-n$ 个方向是数据看不见的**。

### 6.3 它们冻结

在零空间上 $\hat\Sigma v=0$，于是 $(I-\eta\hat\Sigma)v=v$：因子恒等于 $1$，**不是衰减到 0，而是原地不动**。于是对任意 $t$：

$$
P_{\mathrm{null}}(\theta_t)=P_{\mathrm{null}}(\theta_0).
$$

从 $\theta_0=0$ 出发，零空间分量永远是 $0$；从随机初始化出发，它永远停在初始值。**它既不会被学，也不会被忘掉。**

数值检验（脚本 C4，$p=100$、$n=20$）：

```
rank(Phi) = 20 | dim null(Phi) = 80 | p - n = 80
max |A Pnull - 0|                                  = 6.566e-16
max |Pnull (theta_T) - Pnull (theta_0)|  (frozen)   = 8.132e-15
||P_null theta*||^2 = 1.000000   ||P_row theta*||^2 = 1.000000
||P_null (theta_T - theta*)|| = 1.000000   <- the part of the target that was never learnt
||P_row (theta_T - theta*)|| = 1.237859e-08   <- the part that was learnt
max |(I - eta*Sigma)^20000 - P_null| (limit = NULL-space projector, since (1-eta*0)^T = 1) = 6.245e-12   (T_BIG=20000)
```

### 6.4 但"冻结"不是被强制的——三种能看见零空间的机制

§6.3 的"冻结"成立，但**没有任何一行优化器代码在保证它**。它只是一个推论：

$$
\nabla L(\theta)=\frac1n\Phi^\top(\Phi\theta-y)\ \in\ \mathrm{row}(\Phi),\qquad \mathrm{row}(\Phi)\perp\mathrm{null}(\Phi).
$$

梯度恒落在行空间里，所以它在零空间方向的分量恒为 $0$——**不是"不许动"，是"看不见"**。一旦引入一个**看得见**零空间的机制，守恒就破缺。这样的机制有三个：

1. **weight decay**：衰减项 $-\lambda\theta$ 加进梯度，而 $\theta$ 在零空间里有分量——零空间分量每步乘 $(1-\eta\lambda)$，**被主动缩小**；
2. **Adam 的自适应缩放**：步长是 $\hat m/(\sqrt{\hat v}+\varepsilon)$，$\hat v$ 是 $g^2$ 的滑动平均。**行空间向量的逐元素平方不一定是行空间向量**（$g=\Phi^\top c\in\mathrm{row}(\Phi)$，但 $g_i^2=\big(\sum_j\Phi_{ij}c_j\big)^2$ 一般不在行空间里），所以这一步会**漏出**行空间；
3. **特征学习**：上面的"零空间"指的是**初始化处**的 $J$。真实训练里 $J$ 在动，$\mathrm{null}(J_t)$ 是随时间变化的子空间，"冻结"退化成基线（正文 `01` §5.5 实测核余弦只有 $0.896$/$0.775$）。

实测（脚本 `verification/zero_space_init.py` S1，合成线性模型 $p=200$、$n=32$、$\theta_0$ 随机、3000 步，$\lVert P_{\mathrm{null}}\theta_0\rVert=9.199$）：

```
  optimizer                          | drift of the null-space projection
  ------------------------------------------------------------------------------------------------
  full-batch GD, no wd               | ||P_null th_T - P_null th_0|| = 6.352e-06   (rel 6.905e-07)   | ||P_null th_T|| = 9.198691   (should be ~ machine eps: exact identity)
  minibatch SGD, no wd, bs=16        | ||P_null th_T - P_null th_0|| = 4.189e-06   (rel 4.554e-07)   | ||P_null th_T|| = 9.198690   (still in row(Phi): row(Phi_B) subset row(Phi))
  full-batch GD, wd=1e-3             | ||P_null th_T - P_null th_0|| = 4.122e-01   (rel 4.482e-02)   | ||P_null th_T|| = 8.786452   (should decay: (1-lr*wd)^T = 0.955184)
  Adam, no wd                        | ||P_null th_T - P_null th_0|| = 5.405e+00   (rel 5.876e-01)   | ||P_null th_T|| = 10.936105   (elementwise 2nd moment leaks out of row(Phi))
  AdamW, wd=1e-3                     | ||P_null th_T - P_null th_0|| = 5.382e+00   (rel 5.851e-01)   | ||P_null th_T|| = 10.851935   (Adam leak + decay: (1-lr*wd)^T = 0.991040)
```

读法（**实测**）：

- **纯 GD 与 minibatch SGD 都精确守恒**（到 float32 机器精度）。minibatch 的梯度落在 $\mathrm{row}(\Phi_B)\subset\mathrm{row}(\Phi)$——随机性只在行空间里探索，够不到零空间；
- **weight decay 让投影主动缩小**：相对漂移 $4.5\%$，末态 $\lVert P_{\mathrm{null}}\theta_T\rVert/\lVert P_{\mathrm{null}}\theta_0\rVert=0.9552$，和预测 $(1-\eta\lambda)^T=0.9552$ 一致到小数点后四位；
- **Adam 漏得最多**：相对漂移 $59\%$。

**结论：零空间投影不变是"设定"（纯 GD + 特征不动）的性质，不是普适定律。** 换成 AdamW 它就破缺。

### 6.5 初始化税：训练误差看不见、测试误差洗不掉

从 $\theta_0=0$ 和从随机 $\theta_0$ 各训到插值（同一合成模型，S2），**训练误差完全相同**，但测试误差差出一块，且两条轨迹之差**精确等于**初始化的零空间分量：

```
  train MSE, theta_0 = 0       : 1.017e-12
  train MSE, theta_0 = random  : 1.616e-12   <- identical: Phi P_null theta_0 = 0
  test  MSE, theta_0 = 0       : 1.868e+02   <- the I-term only
  test  MSE, theta_0 = random  : 3.301e+02   <- I-term + the permanent init offset
  the two runs differ ONLY in the null space: max |(f_R - f_0) - Phi_te P_null theta_0| = 2.865e-05
  mean (Phi_te P_null theta_0)^2 = 8.397e+01   (the offset's own energy)
  test_R - test_0 = 1.433e+02  = offset^2 (8.397e+01) + 2<residual_0, offset> (5.935e+01)
```

因为 $\Phi P_{\mathrm{null}}=0$，初始化的零空间分量对**训练误差**毫无贡献（训练曲线完全看不见它）；但测试点上 $\Phi_{\rm te}P_{\mathrm{null}}\theta_0\neq0$，它变成一个**永久偏移**。超额的测试误差 = 偏移自身的能量 $+$ 一个相关项（偏移和行空间残差不正交，因为两者来自同一个随机几何）。**这就是 §7.2 地板公式里 $\theta_0$ 那一项的几何含义；从 $\theta_0=0$ 出发恰好把这项清零，这是"从零出发"的最优性来源。**

**v1.5 真题上的样子**（S3，`q_11b94d`，dev 半边）：每一道题都是随机初始化（`torch.manual_seed(seed); Model()`），优化器是 **AdamW**（lr $3\times10^{-5}$、wd $10^{-5}$、256 步、bs 16），所以"$\theta_0\neq0$"是默认情况，而且优化器看得见零空间。

- weight decay 的衰减因子 $(1-\text{lr}\cdot\text{wd})^{256}=0.99999992$，可忽略；
- 跑完官方 256 步后，零空间分量的实测漂移是 **2.3%（候选 A）/ 0.95%（候选 C）**——基本冻结，但**不是严格冻结**（Adam 漏出来的）；
- 惰性极限下"训练永远洗不掉的初始化部分"（初始化输出里正交于可达修正的那部分，测试均方）：**A 为 $1.5\times10^{-5}$，C 为 $4.7\times10^{-4}$**。

值得单独说的一条：候选 C 的这个初始化税（$4.7\times10^{-4}$）比它官方最终测试误差（$2.2\times10^{-4}$）**还大**——若特征不动，它会被卡在这块税上。但真实训练跑到了 $2.2\times10^{-4}$，**低于**这块税，因为特征在动（rich regime）。**惰性地板不是真地板**（和第 05 章 §5.6 是同一件事）。

> **标签**：§6.4 的机制分析是**严格定理**（在固定特征下）；§6.4 的实测表与 §6.5 的数字是**实测**；§6.5 里 v1.5 的"初始化税"是**惰性极限基线**，不是真实训练的预测——真实训练会移动特征，把它变成基线。

---

## 7. 两个"地板"必须分清（本章最容易读错的一节）

很多材料（包括本章早期版本）笼统地说"$p>n$ 有个地板 $=\lVert P_{\mathrm{null}}\theta^{\ast}\rVert^2$"。这句话**方向对，但没说清是哪个空间的地板**。实际上有两个：

### 7.1 逼近地板 $A$（函数空间）：目标在模型类里就是 0

模型类能表达的函数集合是 $\mathrm{span}\{\varphi(x)\}$。目标 $f$ 落在外面的那部分，**任何**参数都表达不出来：

$$
A=\min_{\theta}\frac1m\sum_{i=1}^{m}\big(g_\theta(x^{\rm te}_i)-f(x^{\rm te}_i)\big)^2
=\frac1m\lVert\big(I-P_{\mathrm{span}(\Phi_{\rm te})}\big)y_{\rm te}\rVert_2^2 .
$$

- 若目标确实在模型类里（$y_{\rm te}=\Phi_{\rm te}\theta^{\ast}$），则 $A=0$；
- **$A$ 与 $p>n$ 无关**——它是"表达不了"的问题，不是"参数太多"的问题。

### 7.2 辨识地板 $I$（参数空间）：GD 的解被锁在行空间里

即使 $A=0$（目标完全在模型类里），GD 从 $\theta_0$ 出发也永远待在 $\mathrm{row}(\Phi)$ 里（因为零空间分量冻结，§6.3）。所以**学到的函数**和**目标函数**仍然差一个不可辨识的成分。步长稳定（$0<\eta<2/\lambda_{\max}$）时 $T\to\infty$ 有

$$
(I-\eta\hat\Sigma)^{T}\ \longrightarrow\ P_{\mathrm{null}}(\Phi)
\quad\Longrightarrow\quad
\theta_T-\theta^{\ast}\ \longrightarrow\ P_{\mathrm{null}}(\theta_0-\theta^{\ast}).
$$

（注意这里的极限是**零空间**投影：行空间方向的因子 $(1-\eta\lambda_j)^{T}\to 0$，零空间方向的因子恒为 $1$。）

即：**行空间里学到的部分全部学到位，零空间里的部分原封不动地差着**。把它放到**测试分布**上衡量，得到真正的测试误差地板：

$$
\text{地板}=\big\lVert P_{\mathrm{null}}(\theta_0-\theta^{\ast})\big\rVert_H^2,
\qquad H:=\mathbb E_{x\sim P_{\rm te}}\big[\varphi(x)\varphi(x)^\top\big].
$$

$H$ 是"测试分布看到的特征几何"。各向同性时 $H=I$，地板就是 $\lVert P_{\mathrm{null}}(\theta_0-\theta^{\ast})\rVert_2^2$；从 $\theta_0=0$ 出发，就是 $\lVert P_{\mathrm{null}}\theta^{\ast}\rVert_2^2$——**这才是那句常被引用的话的准确版本**。

### 7.3 关键：地板的大小取决于（训练设计，测试分布）的几何，不只是 $p$ 和 $n$

同一个 $\Phi$，换测试分布，地板就变。数值检验（脚本 C5，$m=2000$）：

```
train error (1/n)||Phi theta_T - y||^2              = 1.473e-15   (target in class => 0)
test  error (1/m)||Phi_te theta_T - y_te||^2        = 0.918674
predicted floor (1/m)||Phi_te P_null theta*||^2      = 0.918674
||P_null theta*||^2 (isotropic proxy)               = 1.000000
test error when test points ARE the train points    = 0.000000   <- null direction invisible => floor 0
floor with theta_0 != 0: (1/m)||Phi_te P_null (theta_0 - theta*)||^2 = 1.737784   (vs 1.844724 in parameter space)
floor when the test design only spans the row space = 5.372e-32  (target invisible in the null direction)
```

读法：

- 训练误差真的到 $0$（$1.5\times10^{-15}$），测试误差却停在 $0.9187$——**训练误差为 0 救不了测试误差**；
- 若**验证点就是训练点**，零空间方向完全不可见，地板变成 $0$——所以"地板"是数据几何的性质，不是 $p$、$n$ 的性质；
- 初始化不为零时，地板还包含"初始化里带的零空间分量"，而且在测试几何下和参数空间里数值不同（$1.7378$ vs $1.8447$）。

### 7.4 训练前就能预测测试误差

把 §4 的闭式和 §7.2 的地板合起来，$T$ 步后的测试误差可以**不训练**算出来。数值检验（脚本 C7）：

```
T=   1  GD R_te = 1.079150   closed form = 1.079150   row-space part of ||theta_T-theta*||^2 = 0.141695
T=   5  GD R_te = 0.932007   closed form = 0.932007   row-space part of ||theta_T-theta*||^2 = 0.010026
T=  20  GD R_te = 0.918607   closed form = 0.918607   row-space part of ||theta_T-theta*||^2 = 0.000778
T=  50  GD R_te = 0.918611   closed form = 0.918611   row-space part of ||theta_T-theta*||^2 = 0.000006
T= 200  GD R_te = 0.918674   closed form = 0.918674   row-space part of ||theta_T-theta*||^2 = 0.000000
```

**行空间部分在 $T=50$ 就基本学完（$6\times10^{-6}$），测试误差却已经贴住地板 $0.9187$ 不动了**——剩下的是不可辨识的那部分，加步数、加参数都改不了。

### 7.5 一句话总结这一节

> **$p>n$ 的代价是"不可辨识"，不是"表达不了"。** 表达不了的部分是 $A$（换个特征/核能改善）；不可辨识的部分是 $I$（只能靠更多样的训练点、更好的初始化或更强的正则化来改）。训练误差归零而测试误差不降，就是撞上了 $I$。

---

## 8. 核/惰性区：把上面的搬到无穷维

### 8.1 从参数到函数

线性情形里，"方向"是 $\mathbb R^p$ 里的方向；核情形里方向变成 RKHS 里的**特征函数**。核 $K(x,x')=\langle\varphi(x),\varphi(x')\rangle_{\mathcal H}$，用核矩阵 $K_{XX}\in\mathbb R^{n\times n}$、$K_{xX}\in\mathbb R^{1\times n}$。

函数空间的 GD（平方损失、全批量、步长 $\eta$、从 $f_0=0$ 出发）：

$$
f_{t+1}=f_t-\frac{\eta}{n}K_{XX}(f_t-y),
$$

对应参数空间 GD 线性化后的同一个更新（前置章 `0.2` §5、§6）。迭代 $T$ 步：

$$
f_T=K_{xX}K_{XX}^{-1}\Big[I-\Big(I-\frac{\eta}{n}K_{XX}\Big)^{T}\Big]y .
$$

（$f_0\neq 0$ 时把 $y$ 换成 $y-f_0(X)$ 再加上 $f_0(x)$。）

### 8.2 谱滤波

对 $K_{XX}$ 做特征分解 $K_{XX}=U\Lambda U^\top$，目标在特征函数上的系数 $a_j=\langle f,v_j\rangle$。每个特征函数方向各跑一条一维递推（和 §5 一模一样，只是 $\Phi^\top\Phi/n$ 换成了 $K_{XX}/n$）：

$$
a_j(T)=(1-\eta\lambda_j)^{T}a_j(0).
$$

于是在核模型内，**总体**的测试风险是

$$
\boxed{\ R_{\rm te}(T)=\sum_{j}(1-\eta\lambda_j)^{2T}a_j^{2}\ }
$$

（严格性说明：这是"核模型内 + 目标在核空间内 + 无噪声"时的**精确**值；目标不在核空间里要加上 $A$，有标签噪声要加噪声项，见 §9。）

平方在指数上，因为**误差是平方范数**：$\lVert\sum_j \delta_j v_j\rVert^2=\sum_j \delta_j^2$（特征函数正交归一）。

### 8.3 三个读数

1. **可学窗口**：$\eta\lambda_jT\gtrsim 1$ 的模式已经被学到；远小于 $1$ 的还没学到。
2. **时间常数**：$\tau_j=1/(\eta\lambda_j)$；第 $j$ 个模式大约需要 $\tau_j$ 步。
3. **验证误差**：目标能量 $a_j^{2}$ 中，落在"还没学到的模式"上的部分，加上核 tail（永远学不到的部分）。

**"$T$ 步后拟合好不好" = 目标能量有多少落在可学窗口里。** 这就是 spectral bias 的精确版本。

### 8.4 三通道例子

假设核只有三个模式（$\eta=0.01$，$T=100$）：

| 模式 | $\lambda_j$ | $a_j^{2}$（目标能量） | $\eta\lambda_jT$ | $(1-\eta\lambda_j)^{2T}$ | 误差贡献 |
|---|---:|---:|---:|---:|---:|
| 1 | 10 | 1.00 | 10 | 0.000000 | 0.000000 |
| 2 | 1 | 0.25 | 1 | 0.133980 | 0.033495 |
| 3 | 0.01 | 0.04 | 0.01 | 0.980198 | 0.039208 |

总验证误差 $=0.072703$，**53.9% 来自"几乎没学的模式 3"**（脚本 C9 输出）。

如果目标是高频/难方向，它的能量就在小 $\lambda$ 模式上——**不是模型容量不够，是预算/学习率不够。**

### 8.5 线性与核的对应关系

| | 线性情形 | 核/惰性情形 |
|---|---|---|
| 方向 | $\mathbb R^p$ 里的向量 | RKHS 里的特征函数 |
| 矩阵 | $\hat\Sigma=\Phi^\top\Phi/n$ | $K_{XX}/n$ |
| 特征值 | $\lambda_j$ | $\lambda_j$（核谱） |
| 目标能量（每个方向） | $z_{0,j}^{2}$ | $a_j^{2}$ |
| 零空间 | $p-n$ 个冻结方向 | 核空间外永远学不到的部分（核 tail） |

---

## 9. 有限 $n$ 与噪声：地板之外还有什么

前面都是"无噪声"（5 个确定性族）。有噪声时，递推形式不变（把 $\theta^{\ast}$ 换成最小二乘解 $\hat\theta_{\rm LS}$），但四项分解的每一项都能被指出来：

$$
R_{\rm te}(T)\ \approx\
\underbrace{A}_{\text{逼近：目标在模型类外}}+
\underbrace{\sum_{j:\lambda_j>0}(1-\eta\lambda_j)^{2T}\big\langle\theta_0-\hat\theta_{\rm LS},v_j\big\rangle_H^{2}}_{O(T)\text{：行空间，}T\text{ 越大越小}}+
\underbrace{\big\lVert P_{\mathrm{null}}(\theta_0-\hat\theta_{\rm LS})\big\rVert_H^{2}+\sigma^{2}\cdot(\text{放大因子})}_{I\text{：不可辨识 + 噪声被放大}} .
$$

注意 $I$ 这一项有两块：零空间部分（$p>n$ 的固有代价，改不了）和噪声放大部分（$p>n$ 时最小范数插值会把噪声放大，$\gamma$ 越大放大越小）。

在随机高斯设计、$\gamma=p/n>1$、标签噪声 $\sigma^2$ 下，$T\to\infty$ 的**期望**超额风险有闭式：

$$
\mathbb E\,R_{\rm te}^{\rm excess}=\Big(1-\frac1\gamma\Big)\lVert\theta^{\ast}\rVert^2+\frac{\sigma^2}{\gamma-1}.
$$

- 第一项是零空间带来的偏差（$p>n$ 越厉害越大）；
- 第二项是噪声被"最小范数插值"放大（$\gamma$ 越大越小）。

数值检验（脚本 C8，测试误差对测试分布解析平均，避免 $\chi^2$ 涨落淹没对比）：

```
p=60 n=15 gamma=4.0 sigma=0.1: E R_te = 0.7622 +/- 0.0038   formula = 0.7634   z = -0.32
p=60 n=30 gamma=2.0 sigma=0.1: E R_te = 0.5136 +/- 0.0045   formula = 0.5203   z = -1.50
p=60 n=15 gamma=4.0 sigma=0.0: E R_te = 0.7488 +/- 0.0039   formula = 0.7500   z = -0.31
```

**无噪声时（$\sigma=0$），第二项消失，只剩零空间偏差**——这正是 v1.5 五个确定性族的情况：不存在"经典 overfitting 噪声"故事，风险只由 $A$、$I$、$O(T)$ 决定。（详细讨论见第 04 章。）

> **别用单个 seed 的"峰高"做定量结论**：在 $p\approx n$ 附近，期望因逆 Wishart 矩发散，但单次实现值远小于期望。v1.5 的 $p/n\sim10^1$–$10^2$ 已远离峰值。

---

## 10. 训练前算：完整流程

- Step 1  初始化模型，算 Jacobian $J=\partial g/\partial\theta$（在训练点上）
- Step 2  $K_{\mathrm{tr}}=JJ^\top/n$，$K_{\mathrm{te}}=J_{\mathrm{te}}J^\top/n$
- Step 3  对 $K_{\mathrm{tr}}$ 做特征分解（$n\times n$，$n\le 2048$ 可行）
- Step 4  用滤波公式预测 $T$ 步后的测试输出：

  $$
  f_T(x_{\mathrm{te}})=f_0+K_{\mathrm{te}}K_{\mathrm{tr}}^{-1}\left[I-(I-\eta K_{\mathrm{tr}})^{T}\right](y-f_0)
  $$

- Step 5  与目标的测试值比较，得到预测 $R_{\mathrm{te}}$
- Step 6  查惰性：训练前后 Jacobian/NTK 的相对变化是否远小于 1？
  - 是 → 预测可信；否 → 核公式只是基线

**注意**：Adam/AdamW 不是固定核滤波，闭式公式不直接适用；只能做数值模拟或相位判断。

**什么时候用哪套公式**（速查）：

| 你的模型 | 用哪套 | 前提 |
|---|---|---|
| 固定特征的线性模型 | §4 闭式 + §7 地板 | 无需训练 |
| 核回归 / 惰性区网络 | §8 谱滤波 | 核在训练中不变 |
| Adam/minibatch/非惰性 | 都不适用 | 只能数值模拟（第 09 章的答题器） |

---

## 11. 在 v1.5 上的实测：惰性往往不成立

我们对两道 univar 题的真实候选做了探针（3 seeds，256 步）：

`q_11b94d`（AdamW@3e-5）：

| 候选 | 参数位移 | NTK 相对变化 | NTK 余弦 | test MSE（初→终） |
|---|---:|---:|---:|---:|
| A | 2.58% | 11.85 | 0.794 | 0.352 → 0.00361 |
| B | 1.48% | 0.212 | 0.986 | 1.076 → 0.00455 |
| C | 0.95% | 2.92 | 0.714 | 0.330 → 0.000178 |

`q_aef977`（SGD@1e-4）：

| 候选 | 参数位移 | NTK 相对变化 | NTK 余弦 | test MSE（初→终） |
|---|---:|---:|---:|---:|
| A | 0.76% | 0.304 | 0.988 | 8.85 → 0.0267 |
| B | 0.47% | 0.108 | 0.9994 | 9.54 → 0.0156 |
| C | 1.79% | 3.16 | 0.983 | 8.97 → 0.127 |

读法：

- 参数只动了 0.5–2.6%，但 NTK 已经显著改变；
- 只有 B 这类候选接近惰性；A/C 的核公式只能当基线；
- 所以**不能用一个候选的惰性结论代表整题**，更不能假设 v1.5 都是核方法。

---

## 12. 本章的"怎么做"

1. 先判断是否能进惰性区：小 lr、短预算、宽网、初始化尺度合适 → 更可能。
2. 能进：算初始化核，用谱滤波预测；这是严格/半严格的。
3. 不能进：核公式当基线，真实误差与它的差 = 特征学习项；此时要靠诊断（03）和实测。
4. 无论哪种：**先看目标能量落在哪些 $\lambda$ 模式上**，再看 $\eta$ 和 $T$ 够不够覆盖它们。
5. 看到"训练误差 0、验证误差不降"，先算两个地板（§7）：$A$ 有没有？$I$ 有多少？再决定是换特征、加数据，还是调 $\eta$、$T$。

---

## 13. 常见坑

1. **把 $(I-\eta\hat\Sigma)^{T}$ 里的 $T$ 当成转置**。转置在本章写 $\top$。
2. **把参数空间的地板当成验证误差地板**。验证误差地板要按测试分布的 $H$ 度量（§7.2），各向同性时才等于 $\lVert P_{\mathrm{null}}\theta^{\ast}\rVert_2^2$。
3. **以为 $p>n$ 就必然有地板**。目标落在行空间里（或者零空间方向在测试分布上不可见）时地板是 $0$（§7.3 的数值）。
4. **忘记步长上限** $\eta<2/\lambda_{\max}$。超了不是"收敛慢"，是发散。
5. **把 $(1-\eta\lambda_j)^{T}$ 写成 $(1-\eta\lambda_jT)^{T}$**。前者是 $T$ 次乘同一个因子，后者是"把 $T$ 步挤成一步"，只在 $T=1$ 时相同。
6. **把 $\lambda_j$ 和 $1/\lambda_j$ 的角色搞反**：小 $\lambda$ = 难学的方向 = 时间常数大。
7. **拿 early stopping 和 ridge 当成同一个东西**。它们只在两端渐近一致（脚本 C10：同一个 $\lambda$ 上两者的滤波因子可以差 1.8 倍，也可以差到 0.01 倍）：

   ```
   T=50  lambda=0.05: GD=0.6050 ridge(1/(eta T))=0.3333 ratio=1.82 | lambda=0.5: GD=0.0052 ridge=0.8333 ratio=0.01 | lambda=5.0: GD=0.0000 ridge=0.9804 ratio=0.00
   T=200 lambda=0.05: GD=0.1340 ridge=0.6667 ratio=0.20 | lambda=0.5: GD=0.0000 ridge=0.9524 ratio=0.00 | lambda=5.0: GD=0.0000 ridge=0.9950 ratio=0.00
   ```

   正确的说法是"**$T$ 步 early stopping 与岭参数 $\lambda_{\rm eff}\approx1/(\eta T)$ 同尺度**"（$T$ 越大正则越弱），而不是"相等"。

---

## 14. 一页总结

- 平方损失 → 梯度是仿射的 → GD 变成线性递推 → 换基后每个方向各跑一条一维递推：$z_{T,j}=(1-\eta\lambda_j)^{T}z_{0,j}$。
- 零空间（至少 $p-n$ 维）上的 $\lambda_j=0$，因子恒为 $1$：**冻结，既学不到也忘不掉**。
- 两个地板要分清：$A$（表达不了，换特征能改）、$I$（不可辨识，只能改数据/初始化/正则）。"$p>n$ 的风险"是后者。
- 地板的大小取决于（训练设计，测试分布）的几何，不是只看 $p,n$。
- 核/惰性区把同一套搬到特征函数上：$R_{\rm te}(T)=\sum_j(1-\eta\lambda_j)^{2T}a_j^2$，可学窗口 $\eta\lambda_jT\gtrsim 1$。
- 先查惰性；不成立时，核公式只是基线。

**本章的公式都能在 `verification/exact_case_derivation.py` 里复现**（本机实跑输出见 `verification/exact_case_result.md`）。
