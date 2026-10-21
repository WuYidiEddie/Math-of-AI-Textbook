# 02 · 可精确求解的核心：线性、核、NTK/lazy 与深度线性网

> 日期：2026-10-19 ｜ Math of AI Textbook
> 本章回答：在什么条件下，$R_{\rm te}(\theta_T)$ 有**精确闭式**或**严格界**；公式里 $f$ 与 $\mathcal G$ 的性质分别出现在哪里。
> 纪律：每个公式标注 exact / approximate / asymptotic；不把近似写成恒等式。

---

## 0. 本章结论

1. **线性/固定特征模型**：有限步 GD 有精确递推；$p>n$ 时零空间分量永不学习，构成不可约地板。测试风险由"目标在行空间/零空间的能量 + 噪声 + 步长"精确决定。
2. **核/随机特征模型**：有限步 GD 是**谱滤波**；给定初始化核矩阵与标签，$T$ 步后的**测试预测**有精确公式（在核模型内），不需要训练。$R_{\rm te}(T)=\sum_j(1-\eta\lambda_jT)^? a_j^2$（总体、无噪声）是"预算 $T$"的精确含义。
3. **NTK/lazy 区**：神经网络在无限宽/小位移极限下**等价于**上述核模型；有限宽的误差是 $O(1/\sqrt w)$ 量级。这是"表达式性质 → 核谱"的严格桥梁。
4. **深度线性网**：Saxe 型 plateau 精确可解，给出深度/初始化 → 学习时间的标度；非线性网络只有定性版本。
5. **无标签噪声时**（5 个确定性族；bigram 例外）：核/线性区**没有标签噪声方差项**，但**设计采样项（有限 $n$）仍在**——它表现为核插值/aliasing 误差 $I$，以及 $O(\sqrt{\mathcal N(\lambda)/n})$ 量级的经验算子涨落（§2.2b）。因此不能简单说"只剩 A+O"；正确说法是"只剩 $A+$ 设计采样 $I+$ 有限 $T$ 的优化 $O$"。double descent 的"第二降段/峰值"主要由噪声方差驱动，**不能直接搬到本题**。
6. **信息下界**：n-width/optimal recovery 给出任何表达式/算法都无法突破的误差下限；若理论误差远高于它，问题在 $A,I,O$，不在"信息不够"。

---

## 1. 线性/多项式表达式：最干净的精确模型

### 1.1 设定

固定特征映射 $\phi:\mathcal X\to\mathbb R^p$，表达式
$$
g_\theta(x)=\theta^\top\phi(x),\qquad \theta\in\mathbb R^p.
$$
训练集 $\Phi=(\phi(x_1),\dots,\phi(x_n))^\top\in\mathbb R^{n\times p}$，$y=(f(x_1),\dots,f(x_n))^\top$。
平方损失
$$
\hat R_n(\theta)=\frac1{2n}\|\Phi\theta-y\|^2.
$$
GD 从 $\theta_0$ 出发，步长 $\eta$。

### 1.2 精确递推（exact）

$$
\theta_{t+1}-\theta^*=\Big(I-\frac{\eta}{n}\Phi^\top\Phi\Big)(\theta_t-\theta^*)
\quad\Longrightarrow\quad
\theta_T-\theta^*=\Big(I-\frac{\eta}{n}\Phi^\top\Phi\Big)^T(\theta_0-\theta^*)
$$
（无噪声 $y=\Phi\theta^*$ 时；有噪声时把 $\theta^*$ 换成最小二乘解，并加噪声项）。
稳定条件：$0<\eta<2n/\lambda_{\max}(\Phi^\top\Phi)$。

记 $\hat\Sigma=\Phi^\top\Phi/n$。$\hat\Sigma$ 的零空间维数至少 $p-n$。在零空间上 $(I-\eta\hat\Sigma)=I$，于是
$$
\boxed{\ \text{不可约地板}=\|P_{\rm null}(\theta_0-\theta^*)\|^2\ }.
$$
在测试分布下（记 $H=\mathbb E_{x\sim P}[\phi(x)\phi(x)^\top]$）：
$$
R_{\rm te}(T)=\big\|(I-\eta\hat\Sigma)^T(\theta_0-\theta^*)\big\|_H^2
=\sum_{j:\lambda_j>0}(1-\eta\lambda_j)^{2T}\big\langle\theta_0-\theta^*,v_j\big\rangle_H^2
+\underbrace{\|P_{\rm null}(\theta_0-\theta^*)\|_H^2}_{\text{地板}}.
$$
若 $H=I$（各向同性测试分布），$\|\cdot\|_H=\|\cdot\|_2$。

**这是"表达式拟合函数 + $p>n$ + 有限步 GD"的最干净精确答案**。读法：

- 可训练方向（行空间，$n$ 维）按 $(1-\eta\lambda_j)^T$ 指数收敛；
- 不可训练方向（零空间，$p-n$ 维）**永远不学**；
- 验证误差 = 目标在零空间的能量 + 未收敛的行空间能量；
- **参数 $p$ 变大只改变零空间与谱，不自动改变误差**。若目标恰好落在行空间，$p$ 大也无害；若目标在零空间有能量，训练误差 0 也救不了验证误差。

### 1.3 随机设计下的精确期望：double descent 及其条件

取各向同性高斯设计 $\Phi_{ij}\sim N(0,1)$，$y=\Phi\theta^*+\varepsilon$，$\varepsilon\sim N(0,\sigma^2I)$，测试点 $x\sim N(0,I_p)$。

**（a）$p<n$（过定）**：最小二乘 $\hat\theta=(\Phi^\top\Phi)^{-1}\Phi^\top y$，
$$
\mathbb E\,R_{\rm te}=\sigma^2\Big(1+\frac{p}{n-p-1}\Big)+\text{（信号项）}.
$$
**（b）$p>n$（过参数化，最小 $\ell_2$ 范数插值）**：$\hat\theta=\Phi^\top(\Phi\Phi^\top)^{-1}y$，
$$
\mathbb E\,R_{\rm te}=\sigma^2\Big(1+\frac{n}{p-n-1}\Big)+\text{（信号项）}.
$$
（当 $\theta^*=0$ 时只有噪声项；一般 $\theta^*$ 时信号项为 $\|P_{\rm null}\theta^*\|^2$ 的期望，见 §1.4。）

**double descent**：在 $p\approx n$ 处两式都发散（插值阈值）；$p>n$ 后噪声项 $\sigma^2(1+n/(p-n-1))$ 随 $p$ 增大**下降**回 $\sigma^2$——这就是"第二降段"。峰值高度由噪声 $\sigma^2$ 与设计决定。

> **阈值处的重尾**：$\mathbb E R_{\rm te}$ 在 $p=n+1$（或 $p=n-1$）时因逆 Wishart 矩发散而**期望无穷**，但单个有限样本的实现值仍然有限（典型值远小于期望）。因此**不要用单个 seed 的"峰高"做定量结论**；v1.5 的 $p/n\sim10^1$–$10^2$ 已远离峰值（见 04 §1）。

> **关键限定**：第二降段是**噪声方差**的下降。5 个确定性族 $y=f(x)$ 无噪声（$\sigma=0$），所以：
> - 噪声项恒为 0；
> - 不存在"经典 double descent 峰"；
> - $R_{\rm te}$ 由信号项（目标在行/零空间的能量）与 $O(T)$ 决定；
> - 若出现"容量越大越差"，原因不是噪声方差，而是：目标在零空间的能量、条件数变差导致 $O$、或特征学习/分类边界。

### 1.4 信号项与"良性/恶性"插值

一般 $\theta^*$ 下，最小范数插值的偏差 = 目标在**零空间**的能量。对随机 $\theta^*\sim N(0,(\|\theta^*\|^2/p)I)$，期望
$$
\mathbb E\|\hat\theta-\theta^*\|^2=\Big(1-\frac1\gamma\Big)\|\theta^*\|^2,\qquad \gamma=\frac pn>1
$$
（在测试分布各向同性时等于测试偏差）。于是**excess risk**（扣掉不可约测试噪声 $\sigma^2$）：
$$
\mathbb E\,R_{\rm te}^{\rm excess}
=\Big(1-\frac1\gamma\Big)\|\theta^*\|^2+\frac{\sigma^2}{\gamma-1}.
$$
- $\sigma=0$：excess risk $=(1-1/\gamma)\|\theta^*\|^2$，随 $\gamma$ **增大而增大**（零空间变大，偏差变大）；若 $\theta^*$ 是固定稀疏向量、零空间不含其能量，则偏差为 0。
- $\sigma>0$：噪声项 $\sigma^2/(\gamma-1)$ 随 $\gamma$ **减小**；两项竞争给出"良性/恶性"的边界。
- **Benign overfitting**（Bartlett–Long–Lugosi–Tsigler 等）给出：当协方差的有效秩相对 $n$ 较小、尾部衰减足够快时，即使 $p>n$、插值噪声，excess risk 仍可 → 0。**注意**：这些条件的核心是"噪声方向的有效秩大、信号方向的有效秩小"；在无噪声题目里，噪声项消失，条件退化为"目标与特征子空间对齐"。

### 1.5 岭正则与 early stopping

岭回归 $\hat\theta_\lambda=(\Phi^\top\Phi+n\lambda I)^{-1}\Phi^\top y$，测试风险是 $\lambda$ 的偏差-方差曲线；$\lambda\to0$ 给出最小范数极限，$\lambda$ 大给出强收缩。

GD 的 early stopping 与岭回归的关系（**谱滤波视角**）：$T$ 步 GD 的滤波因子是 $(1-\eta\lambda_j)^T$；岭的滤波因子是 $\lambda_j/(\lambda_j+\lambda)$。在小 $\eta$ 下
$$
(1-\eta\lambda_j)^T\approx e^{-\eta\lambda_jT}\approx \frac{\lambda_j}{\lambda_j+1/(\eta T)},
$$
即 **$T$ 步 early stopping 与岭参数 $\lambda_{\rm eff}\approx1/(\eta T)$ 同尺度**。**注意：这是粗略对应，不是精确等价**——GD 滤波是 $\mu$ 的多项式，ridge 滤波是有理函数，二者只在两端渐近一致；数值校准显示 $\lambda_{\rm eff}$ 与真正最优 $\lambda^*$ 可差 1.5–2.2 倍（见 `notes/exact-formulas.md` §2.7）。因此：
- 用它做**量级换算/趋势判断**可以；
- 用它做**精确数值预测**不行。
"训练步数 = 正则化强度"的正确表述是：**在 $p>n$ 插值区，$T$ 起正则化作用，最优 $T$ 由验证误差的 U 型曲线决定**。但在**无噪声核/lazy**情形，总体曲线对 $T$ 单调下降（没有 U 型），early stopping 不必要；U 型只在有噪声/有限样本 aliasing/特征学习时出现。

---

## 2. 核方法 / 随机特征：有限步 GD 的精确谱滤波

### 2.1 设定

核 $K(x,x')=\langle\phi(x),\phi(x')\rangle_{\mathcal H}$，RKHS $\mathcal H_K$。核回归目标
$$
\hat R_n(f)=\frac1{2n}\sum_{i=1}^n(f(x_i)-y_i)^2+\frac\lambda2\|f\|_{\mathcal H}^2
$$
（先看 $\lambda=0$ 的插值/最小范数情形）。记 $K_{XX}\in\mathbb R^{n\times n}$，$(K_{XX})_{ij}=K(x_i,x_j)$，$K_{xX}\in\mathbb R^{1\times n}$。

**最小范数核插值**（$T\to\infty$ 的极限）：
$$
s_X(x)=K_{xX}K_{XX}^{-1}y.
$$
**有限步 GD**（核模型内，exact；全批量，平方损失，步长 $\eta$，从 $f_0=0$ 出发）：
$$
\boxed{\;
f_T(x)=K_{xX}K_{XX}^{-1}\Big[I-\Big(I-\frac{\eta}{n}K_{XX}\Big)^T\Big]y
\;}
$$
（注意这是**函数空间**的 GD：$f_{t+1}=f_t-\frac{\eta}{n}K_{XX}(f_t-y)$，不是对系数 $\alpha$ 做普通 GD。参数空间 GD 经线性化后给出同一个函数空间更新。）
若 $f_0\neq0$，把 $y$ 换成 $y-f_0(X)$，再加 $f_0(x)$。

**测试误差**（固定测试点，exact 于核模型）：若只关心训练点，令
$$
F_t=\Big[I-\Big(I-\frac{\eta}{n}K_{XX}\Big)^T\Big]y
$$
即可（**无需 $K^{-1}$**；任意 PSD 核都成立）。对测试点 $x^{(1)},\dots,x^{(m)}$，令
$$
B_t=K_{\rm te}K_{XX}^{-1}\Big[I-\Big(I-\frac{\eta}{n}K_{XX}\Big)^T\Big]\in\mathbb R^{m\times n},
\qquad \hat f_{\rm te}(t)=B_t y .
$$
若 $y=y^0+\varepsilon$、$\mathbb E\varepsilon=0,\ \mathrm{Cov}(\varepsilon)=\sigma^2I$，则**给定 $K$ 的精确测试风险**为
$$
\boxed{\;
\mathbb E\big[R_{\rm te}(t)\big]
=\frac1m\big\|B_t\,y^0-f_{\rm te}\big\|^2+\frac{\sigma^2}{m}\|B_t\|_F^2
\;}
$$
（$f_{\rm te}$ 是 $f$ 在测试点的值向量；$\sigma=0$ 时只剩第一项，此时它是**确定性等式**）。这是"从 $f,\mathcal G$ 算验证误差"最可依赖的有限 $n$ 工具；推导与数值验证见 `notes/exact-formulas.md` §2.5、附录 A.2。成立条件：固定核、平方损失、全批量 GD；失效：特征学习/非 MSE/动量/Adam。

> **步长归一化提醒（重要）**：本目录统一用**平均损失** $\frac1{2n}\sum_i(\cdot)^2$，步长记 $\eta_M$。若某文献用**总和损失** $\frac12\sum_i(\cdot)^2$，其步长 $\eta_\Sigma$ 与本文的关系是 $\eta_\Sigma=\eta_M/n$（或把本文公式中的 $\eta$ 换成 $n\eta$）。`notes/lit-survey.md` 的 $(1-\eta n\lambda)^T$ 是总和约定；跨文档比较前必须先统一。

**总体谱形式**（$n\to\infty$、无噪声、核积分算子特征分解 $K=\sum_j\lambda_j\varphi_j\otimes\varphi_j$，$f=\sum_j a_j\varphi_j$）：
$$
R_{\rm te}(T)=\sum_j(1-\eta\lambda_j)^{2T}a_j^2
\quad(\text{连续时间用 }e^{-2\eta\lambda_jT}).
$$
> 归一化提醒：若把经验平均的 $1/n$ 放进损失，则把 $\eta$ 换成 $\eta/n$，或把 $\lambda_j$ 换成 $\hat K=K_{XX}/n$ 的特征值；公式形式不变。

**可学窗口**：
$$
\eta\lambda_jT\gtrsim1\ \Longleftrightarrow\ \text{第 }j\text{ 个模式在 }T\text{ 步内被学到}.
$$
时间常数 $\tau_j=1/(\eta\lambda_j)$。**"一定步数后"的精确含义就是：目标能量中 $\tau_j\ll T$ 的部分已学到，$\tau_j\gg T$ 的部分还没学到。**

### 2.2 Source condition 与逼近率

若 $f=\mathcal L_K^{r/2}g$（source condition；约定不同文献的指数可能差 2，见注），$\|g\|_{\mathcal H}\le R$，则 $a_j^2\le R^2\lambda_j^{r}$。结合谱滤波：
$$
R_{\rm te}(T)\le R^2\sum_j(1-\eta\lambda_j)^{2T}\lambda_j^{r}.
$$
配合容量条件 $\mathcal N(\lambda)\lesssim\lambda^{-\beta}$（$\mathcal N(\lambda)=\mathrm{tr}((K+\lambda)^{-1}K)$ 为有效维数），取 $\lambda_*\asymp n^{-1/(r+\beta)}$ 得可达率
$$
R_{\rm te}\ \lesssim\ n^{-\frac{r}{r+\beta}},
\qquad
T_*\ \asymp\ (\eta n)^{-1}\,n^{\frac{1}{r+\beta}}
$$
（最后一条是 early stopping 的最优步数；$\eta n$ 是把 $T$ 换成谱滤波尺度后的归一化）。特例：
- 核谱 $\lambda_j\asymp j^{-2}$（$\beta=1/2$）→ $n^{-2r/(2r+1)}$；
- Sobolev $W^{s,2}$（$\lambda_j\asymp j^{-2s/d}$，$\beta=d/(2s)$，$r=1$）→ $n^{-2s/(2s+d)}$。
这是"目标光滑度 $r$ × 核谱指数 $\beta$"共同决定学习曲线的严格框架（Caponnetto–De Vito；Yao–Rosasco–Caponnetto；Bordelon–Canatar–Pehlevan）。注意：这里 $r$ 的约定是 $f=\mathcal L_K^{r/2}g$；若文献写成 $f=\mathcal L_K^{r}g$，指数需相应换算。

### 2.2b 无噪声 ≠ 无 $E$：设计采样项仍在

一个容易犯的错误：因为 $y=f(x)$ 没有标签噪声，就认为估计项 $E=0$。**不对。** 精确标签只删掉"标签噪声项"；用 $n$ 个点近似总体算子 $T$ 的**设计采样误差**仍在。核岭回归（KRR）的精确分解（无噪声）：
$$
f_\lambda-f
=\underbrace{-\lambda(T+\lambda)^{-1}f}_{\text{bias（}A/O\text{）}}
+\underbrace{\lambda(T+\lambda)^{-1}(T_n-T)(T_n+\lambda)^{-1}f}_{\text{设计采样（}E\text{）}},
$$
其中 $T_n$ 是经验核算子。由集中不等式，第二项量级为
$$
\|E\|_{L^2(P)}\ \lesssim\ \sqrt{\frac{\mathcal N(\lambda)}{n}},
\qquad
\mathcal N(\lambda)=\mathrm{tr}\big((K+\lambda)^{-1}K\big).
$$
**推论**：
- 有限 $n$ 时 $E$ 以 $n^{-1/2}$ 量级存在，与 $A,O$ 一起构成 $R_{\rm te}$；
- 在**无噪声**问题里，正确的类层下界基准是 **optimal recovery / n-width**（$\sqrt{\lambda_{n+1}}$、Sobolev $n^{-s/d}$），而不是有噪声统计 minimax 的 $n^{-2s/(2s+d)}$（后者对本题偏保守）；
- 当 $A$ 与 $E$ 都不可忽略时，增加 $T$ 只能压 $O$，不能压 $A+E$。

### 2.3 核插值的 a priori 误差：power function

对 $f\in\mathcal H_K$，最小范数插值满足
$$
|f(x)-s_X(x)|\le P_X(x)\,\|f\|_{\mathcal H_K},
\qquad
P_X(x)^2=K(x,x)-K_{xX}K_{XX}^{-1}K_{Xx}.
$$
于是
$$
\boxed{\;I=\|f-s_X\|_{L^2(P)}^2\le \|f\|_{\mathcal H_K}^2\;\mathbb E_{x\sim P}\big[P_X(x)^2\big]\;}
$$
这是**仅依赖 $f$ 的 RKHS 范数与核/设计点几何**的严格上界。对 Sobolev 型核，power function 由**fill distance** $h_{X,\Omega}$ 控制：$P_X\lesssim h_X^{s}$（$s$ 为光滑度），于是 $I\lesssim \|f\|_{\mathcal H}^2 h_X^{2s}$。**这把"目标性质（范数/光滑度）+ 表达式性质（核/设计）→ 验证误差"写成了显式不等式**，不需要训练。

### 2.4 信息极限：n-width / optimal recovery

- **Kolmogorov n-width**：RKHS 单位球在 $L^2(P)$ 中的 n-width 为 $\sqrt{\lambda_{n+1}}$（$\lambda_j$ 为核积分算子特征值）。
- **Sobolev 类**：$d$ 维 $s$ 阶光滑类的 n-width $\asymp n^{-s/d}$。
- 因此任何用 $n$ 个样本的恢复算法都满足
  $$
  \inf_{\hat f}\sup_{\|f\|_{\mathcal F}\le1}\|\hat f-f\|_{L^2(P)}\ \ge\ c\,E_n(\mathcal F).
  $$
- 反过来，核插值/KRR 在适当条件下达到这些率（optimal recovery），所以**理论误差与下界的差距**就是"表达式/算法还有多少改进空间"的度量。

> 对单个已知的 $f$，n-width（最坏情形）通常过于悲观；应使用 $f$ 的具体谱系数与 RKHS 范数（target-specific bound）。

---

## 3. NTK / lazy 区：把神经网络化为核方法

### 3.1 NTK 与线性化

对表达式 $g_\theta$，在初始化 $\theta_0$ 处的 Jacobian
$$
J(x)=\frac{\partial g_{\theta_0}(x)}{\partial\theta}\in\mathbb R^{1\times p},
\qquad
K_{\rm NTK}(x,x')=J(x)J(x')^\top=\sum_{k=1}^p\partial_{\theta_k}g_{\theta_0}(x)\,\partial_{\theta_k}g_{\theta_0}(x').
$$
**线性化模型**（lazy 区）：
$$
g_{\theta_0+\Delta\theta}(x)\approx g_{\theta_0}(x)+J(x)\Delta\theta.
$$
在 MSE + 全批量 GD 下，线性化模型的训练动力学就是 §2 的核模型，取 $K=JJ^\top$（训练集上 $n\times n$）。因此 §2 的**有限步精确公式**直接适用于神经网络——**条件是**：参数位移足够小，使 $J(\theta_t)\approx J(\theta_0)$。

### 3.2 lazy 判据（何时可信）

Chizat–Oyallon–Bach / Chizat–Bach 的"lazy training"条件：训练引起的 Jacobian 相对变化小，
$$
\frac{\|J(\theta_t)-J(\theta_0)\|}{\|J(\theta_0)\|}\ll1
\quad\text{（在相关算子范数下）}.
$$
经验上：
- 初始化尺度小、输出尺度相对特征变化大 → lazy；
- 初始化尺度大、lr 大、μP 参数化 → rich（特征学习）；
- 宽度 $w$ 越大，有限宽 NTK 与无限宽 NTK 的偏差越小（通常 $O(1/\sqrt w)$）。

**有限宽修正**：把 §2 的公式用于 $K=J_0J_0^\top$ 得到"NTK 预测"；真实网络与它的差就是特征学习项。**这个差本身是可测量的**（训练前后 Jacobian 的相对变化、NTK 对齐度），因此可以给出"理论预测的置信度"。

### 3.3 表达式性质如何进入核

- **宽度**：增大宽度 → 核更接近无限宽 NTK；对固定 $n$，核的**有效秩**不会无限增大（受 $n$ 限制）；宽度主要通过核谱形状与条件数影响 $O$。
- **深度**：改变核的复合结构、谱与条件数；深度越深，NTK 的 $\lambda_{\max}$ 通常越大（更易 EoS），且特征学习的复合效应越强。
- **激活**：改变核的平滑度/谱；光滑激活的核谱衰减更快（低频偏置更强）。
- **LayerNorm/残差**：主要改变 Jacobian 的尺度/条件数（即 $\lambda_j$ 的分布），从而改变 $O$，对 $A$ 的影响是间接的。
- **参数化/初始化**：决定 NTK 缩放与 lazy/rich 区；标准参数化 vs μP 会给出不同的宽度极限。

**结论**：在 lazy 区，"表达式"通过**核矩阵 $K$** 进入公式；在 rich 区，核本身随训练变化，公式失效——这是 03 的主题。

### 3.4 Adam / 自适应优化器

Adam/RMSprop 的逐坐标预条件使更新不再是固定核滤波：
$$
\theta_{t+1}=\theta_t-\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$
在函数空间，这等价于一个**时变、坐标依赖的预条件核**；没有 §2 的闭式。可行的严格替代：
- 在初始化 Jacobian 上**数值模拟**优化器（不需要真训练网络，只需模拟参数更新/函数更新）；
- 用"有效步长"序数（lr/lr_crit）做相位判断；
- 或把 Adam 近似为对角预条件后的 SGD，再套核公式（近似，需验证）。

---

## 4. 深度线性网络：plateau 的精确原型

Saxe–McClelland–Ganguli 的深度线性网络理论给出"为什么深网训得慢/有 plateau"的精确可解模型。

考虑 $L$ 个权重矩阵的线性网络 $g=W_L\cdots W_1x$（线性激活），平方损失，连续时间梯度流（时间 $\tau=\eta t_{\rm iter}$），白化/正交输入，平衡初始化。对单个奇异模态，令该模态在每层的标量为 $w_1,\dots,w_L$，乘积 $P=\prod_{l=1}^L w_l$（该模态的输入-输出增益），平衡初始化给 $w_l^2=P^{2/L}$。梯度流给出（本目录重推并数值验证，见 `notes/exact-formulas.md` §3）：
$$
\frac{dP}{d\tau}=L\,P^{\,2-2/L}\,(s-P),
$$
其中 $s$ 是该模态的目标强度。由此：
$$
T_{\rm plateau}\ \approx\ \frac{P_0^{-(L-2)/L}}{\eta\,s\,(L-2)}
\qquad(L>2),
$$
- $L=2$：精确解是 logistic 曲线，$T\sim\log(s/P_0)$（**没有幂律 plateau**）；
- 每层初始化尺度 $\varepsilon$（$P_0\propto\varepsilon^L$）时，$T\propto\varepsilon^{-(L-2)}$：**初始化越小、越深，plateau 越长**；
- $L\to\infty$：$T\sim1/P_0$。
- 深度 $L$ 越大，学习越慢（在能等的前提下最终误差更低）。

> 口径：一般 $L$ 的 ODE 是本目录从权重梯度流重推的形式，与 Saxe 原文（2 层精确解）的记号对照见 `notes/exact-formulas.md` §3；引用一般 $L$ 时标注"重推，待核原文记号"。

这是**深度/初始化 → 学习时间**的精确原型，也是"训练远未收敛/57× 地板/τ≈140 步"这类现象的数学对应。非线性网络没有精确解，但"多尺度 plateau + 谱偏置"的定性图像通常保留（saddle-to-saddle 动力学）。

---

## 5. 分类：没有闭式，但有 margin 理论

设 $f(x)\in\{0,1\}$（或 $\{\pm1\}$），表达式输出 logits $z_\theta(x)$，交叉熵损失。与 MSE 不同：

- **无闭式谱滤波**：CE 是非二次损失，GD 动力学不对应固定核滤波；核/lazy 区可以线性化 logits，但仍需处理 logistic 非线性。
- **可分数据上的隐式偏置**：GD 在可分数据上沿 max-margin 方向收敛（Soudry et al. 2018）；logit 范数缓慢发散，训练 loss 以 $O(1/t)$ 下降。
- **margin 是分类版的"谱对齐"**：测试误差由 margin 分布与边界复杂度决定；若训练 margin 大、模型复杂度受控，泛化好。
- **Benign overfitting in classification**（Cao–Gu–Belkin 等）：在特定高维/低秩结构下，插值分类器仍可一致；条件比回归更微妙（需要 margin/几何）。
- **有限 $T$**：只有界/定性；没有类似 §2 的精确公式。

对 v1.5 的分类族（tabular/spiral/xor），理论判断应以：
1. 目标边界的复杂度（线性/分段/螺旋/二次交互）→ $A$；
2. 有限 $n$ 的边界估计误差 → $I$；
3. $T$ 步的 margin 增长/优化 → $O$；
4. margin 与验证噪声 → $F$。
为主，不能直接套回归的谱公式。

---

## 6. 一般泛化界：稳定性、PAC-Bayes、压缩

这些界在 $p>n$ 时仍然**非空**，但通常很松；它们的作用是"保证泛化存在"，不是"预测哪个候选更好"。

- **一致稳定性**（Bousquet–Elisseeff）：若算法对单样本替换的损失差 $\le\beta$，则 $\mathbb E[R-\hat R]\le2\beta$。对 SGD（Hardt–Recht–Singer）：凸情形 $\beta\lesssim L^2\eta T/n$；非凸情形是 $(\eta T)$ 的指数型界。方向：**$T$ 小、$\eta$ 小、$n$ 大 → 泛化 gap 小**。但常数大，且与数据几何无关。
  - **定量提醒**：对 v1.5 量级（$n=256,\eta=10^{-3},T=10^3$），非凸稳定性界可到 $O(10)$（见 `notes/exact-formulas.md` §4.5），完全空泛。因此它是"存在性保证"，不是"预测器"。
- **PAC-Bayes**（McAllester）：
  $$
  \mathbb E_Q[R]\le\mathbb E_Q[\hat R]+\sqrt{\frac{\mathrm{KL}(Q\|P)+\log(2n/\delta)}{2n}}.
  $$
  需要选后验 $Q$；对百万参数网络通常非空但很松。
- **压缩界**：若算法输出可由 $C$ 比特描述，则 $R\lesssim\hat R+O(\sqrt{(C+\log(1/\delta))/n})$；对 SGD 轨迹不一定可压缩到很小。
- **Local Rademacher**：在损失小的局部类上给快率；核类可给 $O(\text{有效维}/n)$ 型界。

**用法**：作为 sanity check 与"最坏情形保证"；不要用来排序候选，也不要用它解释"为什么 A 比 B 好 0.01"。

---

## 7. 一张表：哪些结论是 exact / bound / asymptotic

| 结论 | 类型 | 条件 | 能否训练前计算 |
|---|---|---|---|
| 线性 GD 递推、零空间地板 | **exact** | 线性表达式、平方损失、给定 $\eta,T,\theta_0$ | ✅（只需数据/初始化） |
| 线性随机设计 double descent 公式 | **exact（期望）** | 各向同性高斯设计、噪声、最小范数/OLS | ✅（需标签/噪声假设） |
| 核 GD 有限步预测公式 | **exact（核模型内）** | 固定核、平方损失、全批量、给定 $T,\eta$ | ✅（初始化核即可） |
| 核插值 power function 界 | **bound** | $f\in\mathcal H_K$、核插值 | ✅（需 $\|f\|_{\mathcal H}$ 或估计） |
| 总体谱滤波 $R(T)=\sum(1-\eta\lambda)^{2T}a^2$ | **exact（总体、无噪声）** | 无限数据、核固定、梯度流/小 $\eta$ | 部分（需总体核谱） |
| NTK/lazy 有限宽 | **approximate** | lazy 判据成立；宽度大 | ✅（初始化 Jacobian） |
| Source condition / minimax rates | **bound/rate** | $f$ 在 RKHS、光滑度已知 | ✅（需范数/光滑度假设） |
| n-width / optimal recovery | **lower bound** | 函数类已知 | ✅（只需类） |
| Saxe plateau | **exact（线性网）** | 深度线性、平衡初始化、单模态 | ✅（需初始/目标奇异值） |
| 稳定性/PAC-Bayes | **bound** | 一般算法 | ❌（需训练轨迹/后验） |
| double descent 的 RMT/AMP 精确曲线 | **asymptotic** | $n,p,d\to\infty$、比例固定、高斯/线性化 | 部分（需数据协方差） |
| 特征学习区的有限 $T$ 误差 | **open** | 非线性、有限宽、rich | ❌（无通用公式） |

---

## 8. a priori 误差证书：怎么用这些公式判断一个具体问题

给定 $f,\mathcal G,n,T,\eta$（以及可选的验证集），做以下步骤：

1. **算核**：在初始化处算 Jacobian（或固定特征）$\Phi$，得
   $$
   K_{\rm tr}=\frac1n\Phi\Phi^\top\in\mathbb R^{n\times n},\qquad
   K_{\rm te}=\frac1n\Phi_{\rm te}\Phi^\top.
   $$
2. **算核预测**：由 §2 公式（把 $\eta$ 归一化到 $K_{\rm tr}$ 的尺度）
   $$
   \hat f_T(X_{\rm te})=K_{\rm te}K_{\rm tr}^{-1}\Big[I-(I-\eta K_{\rm tr})^T\Big]y .
   $$
   这是**lazy 区的测试预测**；与真实 $f(X_{\rm te})$ 比较得预测误差。
3. **算三项代理**：
   - $O(T)$：把上面的 $T$ 换成 $T$ 与 $T=\infty$ 的差；
   - $I$：$T=\infty$ 的核插值误差（min-norm 极限）；
   - $A$：核空间外的目标能量（用更高容量/不同核/留出数据估计）。
4. **查 lazy**：训练/模拟若干步，测 $\|J_t-J_0\|/\|J_0\|$；若 $\ll1$，第 2 步的预测可信；若不小，预测是"核基线"，真实误差会偏离，偏差量就是特征学习项。
5. **算下界**：$n$-width/谱 tail 给出不可能低于的误差；若预测误差已接近下界，说明没有改进空间。
6. **报告**：预测值 + 主导项 + 置信度（lazy 判据）+ 验证噪声 se。

**注意**：第 2 步的 $\eta$ 必须与训练协议一致（loss 是否带 $1/n$、是否有 weight decay/momentum）。Adam 不能直接套；需要数值模拟优化器状态（在 Jacobian 上），或退化为相位判断。

---

## 9. 本章小结

1. 线性模型给出精确递推与零空间地板：**$p>n$ 的风险是"目标在不可辨识子空间有能量"，不是参数个数。**
2. 核模型给出有限步 GD 的精确谱滤波公式；"$T$ 步后"= 目标谱能量中 $\eta\lambda_jT\gtrsim1$ 的部分被学到。
3. NTK/lazy 把神经网络化为核模型；lazy 判据决定预测是否可信；有限宽是 $O(1/\sqrt w)$ 修正。
4. 深度线性网给出 plateau 的精确原型；非线性只有定性。
5. 无标签噪声时，double descent/benign overfitting 的噪声机制不适用；核/lazy 区总体误差对 $T$ 单调。
6. 稳定性/PAC-Bayes 是保证，不是预测器；分类没有闭式，用 margin 理论。
7. 下一章：当 lazy 失效、需要逼近论/隐式偏置/特征学习理论时，能说什么、不能说什么。
