# 精确公式底稿：过参数化拟合 / 核动力学 / 误差界

> 共享任务 `task-3` 交付底稿（v1）｜作者：`math-derive`｜日期：2026-10-19
> 唯一写入文件：`.team/theory-overparam/exact-formulas.md`（本文件）。未修改任何其他文件；数值验证全部用 `python -` 从 stdin 内联执行，未落盘。
> 定位：最终数学文档的**公式地基**。每条给「设定/记号 → 公式 → 推导要点 → 成立条件 → 失效情形」。凡不确定处显式标 **【待核】**。
> 证据等级标签（全文统一，与 `lit-survey.md` 的 [E]/[U]/[L]/[A]/[H]/[O] 对应）：
> **[E] 精确**（有限量等式/可数值复核）｜**[T] 经典定理**（有出处、条件明确）｜**[P] 本文推导**（推导自洽且已数值复核，但非引用定理）｜**[H] 启发式模板**（需额外假设，只能当模板用）｜**[?] 待核**（需外部文献核实）。

---

## 0. 约定、记号与「什么算 a priori」

### 0.1 全局公设（除非某节另有声明）

- 目标函数 $f:\mathcal X\to\mathbb R$ **确定性**、无标签噪声：训练标签 $y_i=f(x_i)$。测试点 $x_{\rm te}\sim P$，验证误差
  $$
  R_{\rm te}(g):=\mathbb E_{x\sim P}\big[(g(x)-f(x))^2\big]\quad(\text{回归/MSE});
  $$
  对有限验证集 $D_{\rm val}$ 用经验均值 $\widehat R_{\rm val}(g)=\frac{1}{m}\sum_{x\in D_{\rm val}}(g(x)-f(x))^2$ 估计（其统计噪声见 §7.4）。
- 表达式类/参数化 $\mathcal G=\{g_\theta:\theta\in\mathbb R^p\}$，$p$ 为可训练参数个数（候选 arch 的 `trainable_params`），$n$ 为训练点数，$\gamma:=p/n$。
- 算法：GD，步长 $\eta$，初始化 $\theta_0$（默认 $\theta_0=0$，线性/核情形），$T$ 步；损失为 MSE。
- **步长归一化有两套常用约定，全文必须说清**（这也是文献与各底稿之间最容易对不上的地方）：
  - **(S) 总和损失** $L_\Sigma(\theta)=\frac12\sum_{i=1}^n(g_\theta(x_i)-y_i)^2$，步长记 $\eta_\Sigma$；
  - **(M) 平均损失** $L_M(\theta)=\frac1{2n}\sum_{i=1}^n(\cdot)^2$，步长记 $\eta_M$。
  - 同一条 GD 轨迹满足 $\eta_M=n\,\eta_\Sigma$。本文 §1、§2.1–§2.6 的**有限 $n$ 精确式**用 (S)（Gram 矩阵形式最干净，与 `lit-survey.md` 的精确式一致）；§2.8 的**总体学习曲线模板**用 (M)，并对 (S) 显式给出换算。若从文献/综述抄公式，**先核对其 $\eta$ 归一化**。

### 0.2 记号表

| 记号 | 含义 | 维度/取值 |
|---|---|---|
| $f$ | 确定性目标函数 | — |
| $\mathcal G=\{g_\theta\}$ | 参数化/表达式类 | $p$ 个参数 |
| $n,\ m$ | 训练点数、验证(测试)点数 | 整数 |
| $p,\ \gamma=p/n$ | 参数个数、过参数化比 | $\gamma>1$ 即过参数化 |
| $\eta,\ T$ | 学习率、GD 步数 | $\eta>0$ |
| $\beta\in\mathbb R^p$ | 线性模型的真系数（$f(x)=x^\top\beta$） | — |
| $X\in\mathbb R^{n\times p}$ | 线性模型训练设计矩阵 | 行 $x_i^\top$ |
| $\varepsilon,\ \sigma^2$ | 噪声向量/噪声方差（§1 的统计模型） | $\varepsilon\sim N(0,\sigma^2 I_n)$ |
| $K$ | 核矩阵或 Gram 矩阵，$K_{ij}=K(x_i,x_j)$ | $n\times n$ 对称半正定 |
| $\mu_j$（总体）、$\lambda_i$（经验） | 核积分算子/核矩阵特征值 | $\mu_1\ge\mu_2\ge\cdots\ge0$ |
| $\psi_j,\ c_j=\langle f,\psi_j\rangle_{L^2(P)}$ | 核特征函数、目标谱系数 | — |
| $g_T(\mu)$ | 谱滤波因子（$T$ 步 GD 之后该模式被学到的比例） | $[0,1]$（稳定域内） |
| $\lambda$（ridge 参数） | $\ell_2$ 正则化强度 | $\ge0$ |
| $P_{\rm null},\,P_{\rm row}$ | 零空间/行空间正交投影 | 秩 $p-n$ / $n$ |
| $H_N$ | 训练点张成的核子空间 $\mathrm{span}\{K(x_i,\cdot)\}$ | $\dim\le n$ |
| $\|\cdot\|$ | 缺省为欧氏 2-范数；$\|\cdot\|_{L^2(P)}$ 才是验证误差的范数 | — |

### 0.3 「a priori 可计算」的三个层次（本稿的取舍标准）

1. **给定实现的精确式 [E]**：知道具体 $X$（或 $K$）、$f$ 的样本值、$\eta,T$ 时可直接算出的量（§2.3、§2.5 的矩阵公式）。这是最可依赖的部分。
2. **分布层 [T]/[P]**：只给 $P$、$f$ 的范数/谱、初始化时，对数据取期望或高概率的式子（§1、§2.8、§4）。
3. **类层 [T]**：对函数类 $\mathcal F,\mathcal G$ 的 minimax 下界/上界（§5）。这一层**不含算法**，所以 GD 只能逼近、不能突破。

**贯穿全文的一条判据**：验证误差不是 $p$ 的函数，而是 $(f$ 在 $\mathcal G$ 中的可逼近性 $+\ \mathcal G$/核的谱几何 $+\ n$ 决定的采样/估计误差 $+\ T,\eta$ 决定的优化误差$)$ 的函数。$p$ 只通过① $\mathcal G$ 的逼近能力、② 核/算子的谱、③ 数值条件数这三条渠道进入。

---

## 1. 过参数化线性回归（$p>n$，min-norm 插值）的精确风险

### 1.1 设定

统计模型（与 Hastie–Montanari–Rosset–Tibshirani 2019/2022 一致）：
$$
x_i\stackrel{iid}{\sim}N(0,\Sigma),\quad y_i=x_i^\top\beta+\varepsilon_i,\quad \varepsilon_i\stackrel{iid}{\sim}N(0,\sigma^2),\quad i=1,\dots,n,
$$
测试点 $(x,y)$ 独立同分布。预测风险（含测试噪声）$R_{\rm te}(\hat\beta)=\mathbb E\big[(y-x^\top\hat\beta)^2\big]$；**超额风险** $R_{\rm ex}(\hat\beta)=\mathbb E[(x^\top\beta-x^\top\hat\beta)^2]$，则 $R_{\rm te}=R_{\rm ex}+\sigma^2$。本节的**精确闭式只在 $\Sigma=I_p$（isotropic design）时存在**；$\Sigma\ne I$ 见 §1.8。

**min-norm（ridgeless）估计**：$p>n$ 时 $X$ 行满秩 a.s.，
$$
\hat\beta=X^\top(XX^\top)^{-1}y,\qquad X\hat\beta=y\ \text{（插值）},\qquad \hat\beta\in\mathrm{row}(X).
$$

### 1.2 分解与条件风险 [E]

记 $X^{+}=X^\top(XX^\top)^{-1}$，$P_{\rm row}=X^{+}X=X^\top(XX^\top)^{-1}X$（到行空间的正交投影，秩 $n$），$P_{\rm null}=I-P_{\rm row}$（秩 $p-n$）。则恒等式
$$
\beta-\hat\beta=\underbrace{P_{\rm null}\beta}_{\text{信号/aliasing 项}}-\underbrace{X^{+}\varepsilon}_{\text{噪声项}},\qquad
\big(P_{\rm null}\beta\big)^\top\big(X^{+}\varepsilon\big)\ \text{在 }\varepsilon\text{ 上期望为 }0 .
$$
**推导要点**：$\hat\beta=X^{+}(X\beta+\varepsilon)=P_{\rm row}\beta+X^{+}\varepsilon$；$P_{\rm row}$、$P_{\rm null}$ 是正交投影；两项分别只依赖 $\beta$ 与 $\varepsilon$，且 $\mathbb E\varepsilon=0$。

对给定 $X$，对 $\varepsilon$ 与测试点取期望（$x\sim N(0,I_p)$ 时 $\mathbb E_x(x^\top a)^2=\|a\|^2$）：
$$
\boxed{\ \mathbb E_{\varepsilon,x}\big[(x^\top\beta-x^\top\hat\beta)^2\,\big|\,X\big]=\|P_{\rm null}\beta\|^2+\sigma^2\,\mathrm{tr}\!\big((XX^\top)^{-1}\big)\ } \tag{1.1}
$$
（全风险再加 $\sigma^2$。式 (1.1) 是**有限 $n,p$ 的精确等式**，无渐近。）

### 1.3 两个 Wishart 逆的精确期望 [T]

设 $W\sim\mathcal W_n(p,I_n)$（$=XX^\top$，$p>n$）与 $W'\sim\mathcal W_p(n,I_p)$（$=X^\top X$，$n>p$）。经典逆 Wishart 一阶矩：
$$
\mathbb E\big[(XX^\top)^{-1}\big]=\frac{I_n}{p-n-1}\ (p>n+1),\qquad
\mathbb E\,\mathrm{tr}\big((XX^\top)^{-1}\big)=\frac{n}{p-n-1},\qquad
\mathbb E\,\mathrm{tr}\big((X^\top X)^{-1}\big)=\frac{p}{n-p-1}\ (n>p+1). \tag{1.2}
$$
**推导要点**：$W^{-1}$ 的密度 $\propto|W|^{-(p+n+1)/2}e^{-\mathrm{tr}W/2}$ 是指标 $(p-n-1)$ 的逆 Wishart；用 $\mathbb E[W^{-1}]$ 的已知矩公式（或用 $W^{-1}$ 与 $W$ 的 Bartlett 分解逐项积分）。条件 $p>n+1$ 不可去掉：$p=n+1$ 时 $W$ 几乎必然接近奇异，$\mathbb E[W^{-1}]$ **发散**（见 §1.5）。

### 1.4 主定理：各向同性高斯、固定 $\beta$ 的精确期望风险 [P]（推导自洽；极限形式 [T]）

由 $X$ 的各向同性，$P_{\rm null}$ 是均匀随机 $(p-n)$-维子空间上的正交投影，故
$$
\mathbb E_X[P_{\rm null}]=\frac{p-n}{p}I_p\quad\Longrightarrow\quad
\mathbb E_X\|P_{\rm null}\beta\|^2=\Big(1-\frac np\Big)\|\beta\|^2 .
$$
结合 (1.1)、(1.2)：

> **命题 1.1（精确期望风险，$p>n+1$）**
> $$
> \boxed{\ \mathbb E R_{\rm te}(\hat\beta)=\Big(1-\frac np\Big)\|\beta\|^2+\sigma^2\Big(1+\frac{n}{p-n-1}\Big)
> =\Big(1-\frac1\gamma\Big)\|\beta\|^2+\sigma^2\Big(1+\frac{1}{\gamma-1}\Big),\quad \gamma=\frac pn.\ } \tag{1.3}
> $$
> **成立条件**：$x_i$ iid $N(0,I_p)$；$p>n+1$；$\beta$ 固定（或随机但只通过 $\|\beta\|^2$ 进入）；$\varepsilon$ 独立同方差高斯。**推导要点**：$P_{\rm null}$ 的期望投影 + 逆 Wishart 一阶矩 + 交叉项为零。
> **失效**：$\Sigma\ne I$（信号项不再是 $(1-1/\gamma)\|\beta\|^2$）；异方差/相关噪声；$P_{\rm null}$ 与 $\beta$ 的方向对齐依赖（对齐或落在低维 latent 子空间时行为可完全不同，见 §1.8）；$p\le n+1$。

对应的**欠参数化（OLS，$p<n-1$）精确式**（此时无偏，无 aliasing 项）：
$$
\mathbb E R_{\rm ex}(\hat\beta_{\rm OLS})=\sigma^2\mathbb E\,\mathrm{tr}\big((X^\top X)^{-1}\big)=\frac{\sigma^2 p}{n-p-1},\qquad
\mathbb E R_{\rm te}=\frac{\sigma^2(n-1)}{n-p-1}. \tag{1.4}
$$
【成立条件：$n>p+1$，$X$ 行 iid $N(0,I_p)$；$\hat\beta_{\rm OLS}=(X^\top X)^{-1}X^\top y$。】

### 1.5 $p\approx n$（插值阈值）的行为：期望发散 vs 典型值有限 [E]+[P]

由 (1.3)、(1.4)：
- $p<n-1$：$R_{\rm ex}=\sigma^2p/(n-p-1)$ 随 $p\uparrow n-1$ 单调增，$p\to n-1$ 时 $\to\infty$；
- $p>n+1$：$R_{\rm ex}=(1-n/p)\|\beta\|^2+\sigma^2n/(p-n-1)$，$p\downarrow n+1$ 时 $\to\infty$；
- 在阈值 $p\approx n$：**期望风险是无穷大**（因为 $\mathbb E\,\mathrm{tr}((XX^\top)^{-1})$ 在 $p-n-1=0$ 处发散）。

**这不是"曲线画错了"，而是重尾**：$\mathrm{tr}((XX^\top)^{-1})$ 的大偏差概率不可忽略，**期望被极少数近奇异样本主导，中位数/典型值仍是有限的**。数值验证（§附录 A.1，$n=20,p=21$）：$\mathbb E[\mathrm{tr}((XX^\top)^{-1})]$ 理论为 $+\infty$，有限样本 MC 均值 152.96、中位数 21.22、95% 分位 204.8——**典型值约 21，均值被重尾拉到 153**。
**推论（对最终文档的说法）**：
- 若用"期望风险"画 double descent，峰值在 $\gamma=1$ 是**发散**的；实测/有限样本曲线在峰附近的数值强烈依赖 seed 与样本量，**不能用单个 seed 的峰高做定量断论**。
- 「$p>n$ 就一定更差」不成立；「$p\approx n$ 附近期望风险趋于无穷」是精确数学事实（在该模型中）。

### 1.6 比例极限下的曲线形状：double descent 的精确来源 [T]（Hastie et al. 的定性结论）+ [P]（本稿从 (1.3)(1.4) 直接取极限）

令 $n,p\to\infty$，$p/n\to\gamma\in(0,\infty)\setminus\{1\}$，$\|\beta\|,\sigma$ 固定：
$$
\gamma<1:\quad \mathbb E R_{\rm te}\to\frac{\sigma^2}{1-\gamma};\qquad
\gamma>1:\quad \mathbb E R_{\rm te}\to\frac{\sigma^2\gamma}{\gamma-1}+\Big(1-\frac1\gamma\Big)\|\beta\|^2. \tag{1.5}
$$
**推导要点**：把 (1.4) 与 (1.3) 分别取极限；注意 $\sigma^2/(1-\gamma)=\sigma^2(1+\gamma/(1-\gamma))$。
**形状（可从 (1.5) 直接验证的命题）**：
- $\gamma\uparrow1$ 与 $\gamma\downarrow1$ 两侧都 $\to+\infty$（**期望峰**）；
- $\gamma\to\infty$ 时风险 $\to\sigma^2+\|\beta\|^2$（= null risk，即"只输出 0"的风险）。原因是 $n$ 维行空间在 $p$ 维中的占比 $n/p\to0$，可捕捉的信号份额 $\to0$；
- 方差项 $\sigma^2\gamma/(\gamma-1)$ 在 $\gamma>1$ 上单调下降（"第二降段"的来源）；偏差项 $(1-1/\gamma)\|\beta\|^2$ 在 $\gamma>1$ 上单调上升（aliasing 变严重）。
- **SNR 判据**：记 $\mathrm{SNR}=\|\beta\|^2/\sigma^2$。$\mathrm{SNR}\le1$ 时 (1.5) 在 $\gamma>1$ 上全局单调下降；$\mathrm{SNR}>1$ 时在 $\gamma^*=\dfrac{\|\beta\|}{\|\beta\|-\sigma}>1$ 处有唯一局部极小，且
  $$
  \min_{\gamma>1}\ \Big[\frac{\sigma^2\gamma}{\gamma-1}+\Big(1-\frac1\gamma\Big)\|\beta\|^2\Big]=2\sigma\|\beta\|\quad(\mathrm{SNR}>1),\qquad\text{对比 null risk }=\sigma^2+\|\beta\|^2 . \tag{1.6}
  $$
  【(1.6) 由 (1.5) 求导可得；$\mathrm{SNR}<1$ 时 $\inf$ 在 $\gamma\to\infty$ 取到 null risk。该定性结论与 Hastie et al. 2019 摘要 §1.1 的 "SNR>1 有局部极小 / SNR≤1 单调下降" 一致；指数内的精确表达式以原文 Theorem 1 为准 **【待核常数表达形式】**。】
**失效情形**：以上"第二降段"在 **$\beta$ 各向同性、misspecification 为零**的模型里成立；若特征/目标有低维结构（$\beta$ 与 $\Sigma$ 的主特征对齐，Hastie et al. 的 latent-space model），风险在 $\gamma>1$ 可全局单调下降、最优在 $\gamma\to\infty$——**double descent 的形状是任务几何（task geometry）的函数，不是普适定律**。

### 1.7 ridge 的平滑版本

设 $\hat\beta_\lambda=(X^\top X+\lambda I)^{-1}X^\top y$，$\lambda>0$（$y=X\beta+\varepsilon$，$X$ 任意有限 $n,p$）。记 $M=X^\top X$，则
$$
\hat\beta_\lambda=\arg\min_\beta\ \|X\beta-y\|^2+\lambda\|\beta\|^2,\qquad
\beta-\hat\beta_\lambda=\lambda(M+\lambda I)^{-1}\beta-(M+\lambda I)^{-1}X^\top\varepsilon,
$$
$$
\boxed{\ \mathbb E_{\varepsilon,x}\big[(x^\top\beta-x^\top\hat\beta_\lambda)^2\,\big|\,X\big]
=\lambda^2\beta^\top(M+\lambda I)^{-2}\beta+\sigma^2\,\mathrm{tr}\!\big(M(M+\lambda I)^{-2}\big)\ } \tag{1.7}
$$
（全风险再加 $\sigma^2$）。**推导要点**：对 $\varepsilon$ 取期望时交叉项为零；$\mathbb E||(M+\lambda I)^{-1}X^\top\varepsilon||^2=\sigma^2\mathrm{tr}((M+\lambda I)^{-1}M(M+\lambda I)^{-1})$。
**性质（可从 (1.7) 直接验证）**：
- $\lambda\to0^+$ 时 $\lambda(M+\lambda I)^{-1}\to P_{\rm null}$（$p>n$），$M(M+\lambda I)^{-2}\to M^{+}$，故 (1.7) $\to$ (1.1) 的 ridgeless 条件风险。**min-norm 插值 = ridge 的 $\lambda\downarrow0$ 极限**；也是 GD 从 0 初始化、$T\to\infty$ 的极限（§2.6）。
- **归一化警告**：若 $p,n\to\infty$ 且 $\lambda$ **固定**，则 $\lambda$ 相对 $X^\top X$ 的特征值（量级 $O(n)$）可忽略，(1.7) 的比例极限退化为 (1.5) 的 ridgeless 曲线；要让 ridge 在比例极限下真正起作用，需 $\lambda=\Theta(n)$（等价地：在 $\frac1nX^\top X$ 的尺度上取固定正则）。**【待核：与 Hastie et al. 2019 Theorem 5（ridge 风险）的精确对照；本文只核对了 $\lambda\downarrow0$ 退化。】**
- ridge 的"最优调参"在期望风险意义下通常严格优于 ridgeless（Hastie et al. 2019 §1.1(iii)：optimally-tuned ridge dominates min-norm；CV 渐近等价于最优调参）。**本文不复述其精确风险公式**（需要 MP 律/Stieltjes 变换，属 §1.8 的待补项）。
- 有限 $n,p$ 下 $\mathbb E_X$ 的闭式（对 (1.7) 再对 $X$ 取期望）**没有初等闭式**（需要 Wishart 逆的二阶矩/超几何函数）；§1.7 只给**给定 $X$ 的精确式**。

### 1.8 失效边界与推广（写进最终文档时必须声明）

1. **$\Sigma\ne I$**：信号项变为 $\beta^\top(I-\mathbb E_X[P_{\rm row}])\beta$，$\mathbb E_X[P_{\rm row}]$ 一般不是 $cI$；无初等闭式。定向结论（Hastie et al. 2019）：若 $\beta$ 与 $\Sigma$ 主特征对齐、存在低维 latent 子空间，过参数化风险可**单调下降**、min-norm 可最优；若 $\beta$ 与 $\Sigma$ 特征"非相干"（spherical prior），行为与各向同性类似。**【待核：latent model 的精确风险公式（原文 §5.4），本文未推导。】**
2. **misspecification**（真实目标非线性/不在 $p$ 维线性类中）：$R_{\rm te}=R_{\rm appr}+R_{\rm ex}$，$R_{\rm appr}$ 是 $f$ 到类 $\mathcal G$ 的 $L^2(P)$ 距离平方；double descent 可更明显，第二降段可低于 null risk（Hastie et al. §5.3）。**这直接说明：$p$ 大不大无关，"$f$ 到 $\mathcal G$ 的距离"才是逼近项。**
3. **非高斯/相关设计**：需 universality（只需均值/协方差）或 AMP/state evolution；在比例极限下结论常保持 **[A]**，但有限样本无精确闭式。
4. **噪声非各向同性/重尾**：$\mathrm{tr}((XX^\top)^{-1})$ 的系数结构改变；期望风险的峰附近重尾更严重。
5. **$p/n\gg1$ 时**（本题 v1.5 的 $\gamma\sim10^1$–$10^2$）：由 (1.3)，风险已远离峰值，只剩 $(\mathrm{bias}\approx\|\beta\|^2,\ \mathrm{var}\approx\sigma^2)$ 的"地板"；此时**再增大 $p$ 不改善验证误差**，除非目标/特征有低维对齐结构。

### 1.9 对最终文档的可操作性

- 若候选可视为"固定特征 + 线性头"（随机特征/线性化），**验证误差在给定实现下可以精确算**（§1.2、(1.1)、(1.7)），不需要训练；$p$ 只出现在矩阵维度里。
- 若候选是完整网络、但处于 lazy/NTK 区，(1.1)-(1.7) 以 NTK 特征 $\varphi(x)=J_{\theta_0}(x)$ 套用；否则**只剩界/渐近**，不能声称精确。
- 「$p>n$ ⇒ 过拟合/更差」**不是定理**；(1.3) 说明在 $\gamma$ 很大时风险趋于 null risk $\sigma^2+\|\beta\|^2$，与 $p$ 的增长无关（在该模型里）。

---

## 2. 有限步 GD/梯度流在核回归与随机特征上的精确解

### 2.1 设定

固定特征映射 $\varphi:\mathcal X\to\mathbb R^p$（随机特征、多项式特征、或初始化处的 NTK 特征 $J_{\theta_0}(x)$），$g_\theta(x)=\varphi(x)^\top\theta$。训练数据 $\{(x_i,f(x_i))\}_{i=1}^n$，记
$$
\Phi=\begin{pmatrix}\varphi(x_1)^\top\\ \vdots\\ \varphi(x_n)^\top\end{pmatrix}\in\mathbb R^{n\times p},\quad
K=\Phi\Phi^\top\in\mathbb R^{n\times n},\quad y=f(X)=(\Phi\theta^*)\ \text{（well-specified 时）}.
$$
**函数空间视角（无需求逆，对任意 PSD 核成立）**：设 $f_t(x)=\varphi(x)^\top\theta_t$ 为 $t$ 步后的预测函数，训练预测向量 $F_t=(f_t(x_1),\dots,f_t(x_n))^\top=\Phi\theta_t\in\mathbb R^n$。MSE 的 GD（总和损失约定 (S)）在**函数空间**的更新为
$$
F_{t+1}=F_t-\eta\,K\,(F_t-y),\qquad F_0=\Phi\theta_0\quad(\text{默认 }\theta_0=0\Rightarrow F_0=0). \tag{2.1}
$$
（等价地：$\alpha_t=(I-(I-\eta K)^t)K^{-1}y$，$F_t=K\alpha_t$；对 $\theta$ 的更新为 $\theta_{t+1}=\theta_t-\eta\Phi^\top(\Phi\theta_t-y)$。）**这条公式对随机特征、固定核、lazy/NTK 线性化网络都精确成立**（前提：特征/核在训练中不变、损失为 MSE、无 weight decay）。

### 2.2 精确闭式 [E]

$$
\boxed{\ F_t=\big[I-(I-\eta K)^t\big]\,y\ } \tag{2.2}
$$
**推导要点**：令 $R_t=F_t-y$，则 $R_{t+1}=(I-\eta K)R_t$，故 $R_t=(I-\eta K)^t(y-F_0)$；$F_0=0$ 即得。**对奇异 $K$ 无需逆**（$(I-\eta K)^t$ 是矩阵多项式）；若 $K$ 奇异且 $y\notin\mathrm{range}(K)$，GD 收敛到 $P_{\mathrm{range}(K)}y$ 而非插值（见 §2.6）。
带非零初始化：$F_t=[I-(I-\eta K)^t]y+(I-\eta K)^tF_0$。

**测试点预测**：$k(x)=(K(x,x_1),\dots,K(x,x_n))^\top\in\mathbb R^n$，$f_t(x)=k(x)^\top\alpha_t$，其中
$$
\alpha_t=(I-(I-\eta K)^t)K^{-1}y\ \ (K\ \text{可逆}),\qquad f_t(x)=k(x)^\top K^{-1}\big[I-(I-\eta K)^t\big]y. \tag{2.3}
$$
**特征空间显式（随机特征）**：$\theta_t=\theta_t^{(0)}-\eta\Phi^\top\sum_{s=0}^{t-1}(I-\eta K)^{t-1-s}(\Phi\theta_0-y)$，$\theta_0=0$ 时 $\theta_t=\Phi^\top(I-(I-\eta K)^t)K^{-1}y$（$K$ 可逆时），$f_t(x)=\varphi(x)^\top\theta_t$。

### 2.3 谱滤波形式 [E]（本文核心公式）

$K=V\Lambda V^\top$，$\Lambda=\mathrm{diag}(\lambda_1,\dots,\lambda_n)$，$\lambda_i\ge0$。定义**谱滤波因子**
$$
\boxed{\ g_t(\lambda):=1-(1-\eta\lambda)^t\ }\qquad(\text{discrete GD filter}) \tag{2.4}
$$
则
$$
F_t=V\,\mathrm{diag}\big(g_t(\lambda_i)\big)V^\top y,\qquad
f_t(x)=\sum_{i=1}^n g_t(\lambda_i)\,\frac{(v_i^\top y)\,(k(x)^\top v_i)}{\lambda_i}\quad(K\ \text{可逆}) . \tag{2.5}
$$
即：**每个核主模式 $i$ 以因子 $g_t(\lambda_i)$ 被"学到"**；$g_t(0)=0$（纯零模式不动），$g_t(\lambda)\uparrow1$ 当 $t\uparrow\infty$（稳定域内）。
**推导要点**：$(I-\eta K)^t=V\mathrm{diag}((1-\eta\lambda_i)^t)V^\top$；训练残差 $(I-\eta K)^ty$ 逐模式衰减。

### 2.4 步长、稳定性、离散 vs 连续 [E]/[P]

记 $M=\max_i\lambda_i=\|K\|_2$。
- **稳定域**：$0<\eta<2/M$ 时，$|1-\eta\lambda_i|<1\ \forall i$，GD 收敛；$\eta>2/M$ 时至少一个模式 $|1-\eta\lambda_i|>1$ 发散；$\eta\lambda_i=2$ 的模式永久振荡（因子 $-1$）。
- **收敛速度**：最慢模式速率 $|1-\eta\lambda_{\min}|$（$\lambda_{\min}>0$ 时），即拟合该模式需 $t\approx \log(1/\epsilon)/(\eta\lambda_{\min})$ 步——**谱偏置（spectral bias）的精确来源**：$\lambda_i$ 越小的方向学得越慢，时间常数 $1/(\eta\lambda_i)$。
- **离散 vs 连续**（$\eta\lambda=:x\in[0,1)$）：
  $$
  (1-x)^t=e^{-tx}\,\exp\!\Big(-t\sum_{k\ge2}\frac{x^k}{k}\Big),\qquad
  \big|(1-x)^t-e^{-tx}\big|\ \le\ e^{-tx}\,\frac{t\,x^2}{2(1-x)}. \tag{2.6}
  $$
  **推导要点**：$\log(1-x)=-x-\sum_{k\ge2}x^k/k$，再用 $1-e^{-u}\le u$（$u\ge0$）与 $\sum_{k\ge2}x^k/k\le \frac{x^2}{2(1-x)}$。**成立条件**：$x\in[0,1)$；**失效**：$x\to1^-$（界仍成立但很松）、$\eta\lambda>1$ 时 $(1-x)^t$ 振荡，不能用 $e^{-tx}$ 代替。数值验证见附录 A.2（$x\in[10^{-3},0.999]$ 全部满足）。
- **连续时间极限**：$\eta\to0,\ t\to\infty,\ \eta t=\tau$ 固定时 $g_t(\lambda)\to 1-e^{-\lambda\tau}$（gradient flow 滤波）。这是"小学习率 = 连续梯度流"的精确含义；**大学习率/大步数下离散与连续的差别是 $O(t\eta^2\lambda^2)$ 量级**（由 (2.6)）。

### 2.5 有限 $n$ 的精确测试误差公式 [E]（给定实现可算，无需训练）

把 $y$ 分成"无噪声目标"与"噪声"：$y=y^0+\epsilon$，$y^0=f(X)$，$\epsilon=(\varepsilon_1,\dots,\varepsilon_n)^\top$，$\mathbb E\epsilon=0$，$\mathrm{Cov}(\epsilon)=\sigma^2I_n$（$\sigma^2=0$ 对应确定性标签）。设测试点 $x^{(1)},\dots,x^{(m)}$，$K_{\rm te}\in\mathbb R^{m\times n}$，$K_{\rm te,ji}=K(x^{(j)},x_i)$。由 (2.3)：
$$
B_t:=K_{\rm te}K^{-1}\big[I-(I-\eta K)^t\big]\in\mathbb R^{m\times n},\qquad
\hat F_{\rm te}(t)=B_t\,y .
$$
则**对噪声与测试点平均的精确风险**为
$$
\boxed{\ \mathbb E\big[R_{\rm te}(t)\big]
=\frac1m\big\|B_t\,y^0-f_{\rm te}\big\|^2+\frac{\sigma^2}{m}\|B_t\|_F^2\ } \tag{2.7}
$$
（$f_{\rm te}$ 是 $f$ 在测试点的值向量；$\sigma^2=0$ 时只剩第一项）。**若只要"给定实现的验证误差"**，把 $\sigma$ 项换成具体噪声实现即可（树莓核情形 $f_t(x)$ 由 (2.3) 直接算）。
**推导要点**：$\hat F_{\rm te}(t)=B_t y^0+B_t\epsilon$；对 $\epsilon$ 取期望时交叉项为零，$\mathbb E\|\epsilon\|^2$ 结构给出 $\sigma^2\|B_t\|_F^2$。
**条件**：$K$ 可逆（$p\ge n$ 且特征满秩时 a.s.）；奇异时用 $K^{+}$ 与 $P_{\rm range}$ 版本（极限意义下成立）。**失效**：特征在训练中变化（feature learning/lazy 条件破坏）；损失非 MSE；有 weight decay/动量；$B_t$ 数值上病态（$K$ 条件数大时需稳定分解）。数值验证见附录 A.2：$T=10$ 时 exact 0.49349 vs MC 0.49366（se 0.00038）；$T=22$ 时 exact 0.48254 vs MC 0.48270（se 0.00044）。

### 2.6 极限行为：min-norm 插值与 T→∞ [E]

若 $0<\eta<2/M$：
- $K$ 可逆：$F_t\to y$（插值），$f_t\to k(\cdot)^\top K^{-1}y$ = **最小 RKHS 范数插值**（也是 §1 的 min-norm 解的核版本）。
- $K$ 奇异：$F_t\to P_{\mathrm{range}(K)}y$（若 $y\in\mathrm{range}(K)$ 则插值到 0 训练误差）；等价地 $f_t\to$ 最小范数解。
- **等价性**：在 $p>n$ 的线性特征下，从 $\theta_0=0$ 出发的 GD 收敛到 min-norm 插值；"$T=\infty$"与"ridge 的 $\lambda\downarrow0$"、与"min-norm 惩罚"给出同一解。这是 §1 与 §2 的接缝：**§1.4 的期望风险 = §2.5 在 $T=\infty$, $\eta<2/M$ 下的期望风险**（无噪声时）。

### 2.7 early stopping 与 ridge / spectral filter 的对应

定义两种滤波器（把 $\lambda_i$ 记作模式强度 $\mu$）：
$$
\text{GD: } g_t^{\rm GD}(\mu)=1-(1-\eta\mu)^t;\qquad
\text{ridge: } g_\lambda^{\rm R}(\mu)=\frac{\mu}{\mu+\lambda}. \tag{2.8}
$$
二者都是 $[0,1]\to[0,1)$ 的谱滤波器：小 $\mu$ 被强压缩，大 $\mu$ 被保留。**精确对应关系（严格可陈述的部分）**：
- **小 $\mu$ 斜率匹配**：$\mu\to0$ 时 $g_t^{\rm GD}\approx t\eta\mu$，$g_\lambda^{\rm R}\approx\mu/\lambda$，故
  $$
  \lambda_{\rm eff}^{\rm slope}=\frac{1}{t\,\eta}; \tag{2.9}
  $$
- **半收缩点匹配**：$g_t^{\rm GD}(\mu)=1/2\iff \mu=\mu_{1/2}(t):=\dfrac{1-2^{-1/t}}{\eta}\ (\to\ln2/(\eta t))$；$g_\lambda^{\rm R}(\mu)=1/2\iff \mu=\lambda$，故
  $$
  \lambda_{\rm eff}^{\rm half}=\frac{1-2^{-1/t}}{\eta}\approx\frac{\ln 2}{\eta t}. \tag{2.10}
  $$
- **不是精确等价**：$g_t^{\rm GD}$ 是 $\mu$ 的 $t$ 次多项式，$g_\lambda^{\rm R}$ 是有理函数；两者的曲线形状不同（GD 在大 $\mu$ 处指数逼近 1，ridge 按 $\mu/(\mu+\lambda)$ 逼近 1），因此 **"early stopping = ridge" 只在下面的意义下成立：作为一参数族谱滤波器的"粗略对应"，$\lambda_{\rm eff}\asymp 1/(\eta T)$**。**【待核：Ali–Kolter–Tibshirani 2019 的连续时间精确耦合（对最小二乘路径有精确关系）只适用于连续时间/特定路径；离散 GD 与 ridge 路径的精确等价不成立。】**
- **risk 层面的数值校准**（附录 A.2，随机特征 $n=40$、噪声 $\sigma=0.2$）：最优 GD $T^*=22$ 时风险 0.4825；最优 ridge $\lambda^*=1.99$ 时风险 0.4803；此时 $\lambda_{\rm eff}^{\rm slope}=4.47$、$\lambda_{\rm eff}^{\rm half}=3.05$。即 $\lambda_{\rm eff}$ 与真最优 $\lambda^*$ 同量级（差 1.5–2.2 倍），滤波器 RMS 差 0.05。**结论：作量级换算可以，作精确预测不行。**
- **T 的最优性（启发式）**：early stopping 的偏差-方差权衡见 §2.8；对幂律谱得到 $T^*\asymp \eta^{-1}(n/\sigma^2)^{p/q}$（参数含义见 §2.8），即 $T^*$ 随 $n$ 次线性增长、与 $\eta$ 成反比；**最优 $T$ 依赖 $\sigma^2$ 与谱**，实际只能由 CV/验证集选。
**成立条件**：squared loss、固定核/特征、无 weight decay；**失效**：feature learning（核随时间变）、非 MSE、动量/Adam、batch noise。

### 2.8 总体谱学习曲线：偏差-方差模板 [H]（必须标注为启发式）

**设定**：$P$、核 $K$ 的积分算子 $T_K$ 有特征分解 $T_K\psi_j=\mu_j\psi_j$（$\psi_j$ 在 $L^2(P)$ 中标准正交），目标 $f=\sum_j c_j\psi_j$，$c_j=\langle f,\psi_j\rangle_{L^2(P)}$；训练点 $x_i\stackrel{iid}{\sim}P$，无噪声标签 $f(x_i)$（加噪时 $\varepsilon_i$ 方差 $\sigma^2$）。$g_j:=g_t(\mu_j)$ 为 GD 滤波（平均损失约定 (M)；若用总和损失 (S)，把 $\eta$ 换成 $n\eta$）。在"经验算子 ≈ 总体算子"（大 $n$、谱集中）近似下：
$$
\boxed{\ \mathbb E R_{\rm te}(t)\ \approx\ \underbrace{\sum_j c_j^2\,\big(1-g_j\big)^2}_{\text{bias}^2\ (\text{未学到的信号})}
\ +\ \underbrace{\frac{\sigma^2}{n}\sum_j g_j^2}_{\text{variance}\ (\text{噪声被滤波后残留})}\ } \tag{2.11}
$$
ridge 版本：$g_j=\mu_j/(\mu_j+\lambda)$（同式）。

**推导要点**：核梯度流逐模式 $\dot{\hat c}_j=-\eta\mu_j(\hat c_j-\hat y_j)$，$\hat y_j=c_j+O(\sigma/\sqrt n)$ 为经验投影；离散解 $\hat c_j(t)=g_j\hat y_j$；测试误差 $=\sum_j(\hat c_j-c_j)^2$，对噪声取期望得 $\sigma^2/n\cdot\sum g_j^2$。**（2.11 是 $n\to\infty$ 首阶的模板；有限 $n$ 的严格版本是 §2.5 的矩阵公式。）**

**幂律谱下的率与最优 $T$ [H]**：设
$$
\mu_j\asymp a\,j^{-p}\ (p>0),\qquad c_j^2\asymp b\,j^{-q}\ (q>1),\qquad
J:=(\eta t)^{-1/p}\ \text{（GD 的谱截断；ridge 用 }J\asymp\lambda^{-1/p}\text{）}.
$$
则（$J$ 远小于 UV 截断、$p>1/2$ 的区间）
$$
\text{bias}^2\approx \frac{b}{q-1}J^{1-q},\qquad
\text{variance}\approx C_p\,\frac{\sigma^2}{n}J\quad\Big(C_p=1+\frac{1}{2p-1}\ \text{量级}\Big),
$$
平衡 $J^{1-q}\asymp(\sigma^2/n)J$ 得
$$
\boxed{\ J^*\asymp\Big(\frac{n}{\sigma^2}\Big)^{1/q},\qquad
\mathbb E R_{\rm te}^*\ \asymp\ \Big(\frac{\sigma^2}{n}\Big)^{(q-1)/q},\qquad
T^*_{\rm GD}\asymp\frac{(J^*)^p}{\eta}\asymp\frac1\eta\Big(\frac{n}{\sigma^2}\Big)^{p/q}\ } \tag{2.12}
$$
**翻译成光滑度**：核为 $s$ 阶（$\mu_j\asymp j^{-2s/d}$，即 $p=2s/d$）、目标光滑度 $r$（$c_j^2\asymp j^{-2r/d-1}$，即 $q=2r/d+1$）时
$$
q<2p+1\ (\Leftrightarrow r<2s)\ \text{（非饱和区）}:\quad \text{rate}=n^{-\frac{q-1}{q}}=n^{-\frac{2r}{2r+d}};\qquad
q>2p+1\ (\Leftrightarrow r>2s)\ \text{（饱和区）}:\quad \text{rate}=n^{-\frac{2p}{2p+1}}=n^{-\frac{4s}{4s+d}}. \tag{2.13}
$$
（饱和区由 bias$^2\asymp\lambda^2$ 与 variance $\asymp(\sigma^2/n)\lambda^{-1/p}$ 平衡得到。）
**成立条件（模板的"适用边界"，缺一不可）**：
(i) 核固定（lazy/NTK 区或显式核），**无 feature learning**；
(ii) 训练点数 $n$ 足够大，使经验核谱集中在总体谱附近（$J^*$ 落在大 $n$ 可分辨范围内）；
(iii) 幂律谱 + source condition（$c_j$ 相对 $\mu_j$ 有界，即存在 $r$）；
(iv) 目标/噪声同分布，测试与训练同 $P$；
(v) (M) 损失约定（或按 §0.1 换算）。
**失效情形**：
- **没有 source condition**（$f$ 太粗糙、$c_j$ 衰减慢到 $q\le1$）：bias 不收敛，率退化；
- **谱有 gap/平台**（非幂律）：平衡点可能不存在，(2.12) 不适用；此时直接用 (2.11)/(2.5) 数值算；
- **有限 $n$/小 $\mu$ 的 UV 截断**：GD 滤波器 $g_j$ 在 $\mu_j\ll 1/(\eta n T)$ 上近似 $\eta nT\mu_j$，求和会截断在有效秩；精确说法必须回到 §2.5；
- **feature learning**：$\mu_j$ 随时间变，(2.11) 完全不适用；深度学习里"谱偏置"经验现象与 (2.11) 的定性一致但定量不可靠；
- **非高斯/相关噪声、label noise 与 feature 相关**：variance 项结构改变。

**【待核】**：式 (2.11) 的 $1/n$ 系数与谱分布的精确形式在 Canatar–Bordelon–Pehlevan (2021)、Bordelon et al. (2020) 的"mode-counting/自洽 $\kappa$"框架里有修正版本；本文给的是首阶模板，引用时须注明"模板级"。

### 2.9 与 §1 的一致性检查

令线性特征 $\varphi(x)=x$、$K=XX^\top$、$y^0=X\beta$、$T\to\infty$（$\eta<2/\|K\|$）、$\sigma^2=0$：由 (2.3) $f_\infty=x^\top X^\top(XX^\top)^{-1}X\beta=x^\top P_{\rm row}\beta$，代入 (2.7) 得 $\mathbb E R_{\rm te}=\mathbb E\|(I-P_{\rm row})\beta\|^2=(1-n/p)\|\beta\|^2$，与 §1.4 的 noiseless 部分一致；噪声项 $\sigma^2\|B_\infty\|_F^2/m$ 对 $x\sim N(0,I)$ 的期望给出 $\sigma^2\,\mathrm{tr}((XX^\top)^{-1})$，与 (1.1) 一致。**两条独立路线互相印证，说明 §1、§2 的公式是自洽的。**

---

## 3. 深度线性网络的 plateau 动力学（Saxe–McClelland–Ganguli 2014）

### 3.1 设定（Saxe 2014 的精确可解模型）

- 网络：$y=W^{(L)}W^{(L-1)}\cdots W^{(1)}x$，**线性激活**，$L$ 个权重矩阵（Saxe 原文的 $N_l$ 层网络有 $N_l-1$ 个权重矩阵；本节 $L$ 均指**权重矩阵个数**）。
- 损失：平方误差 $E=\frac12\sum_{\mu}\|y^\mu-\text{目标}^\mu\|^2$；训练动态：**连续时间梯度流**（原文 $t$ 以 $1/\lambda$ 为时间单位，$\lambda$ 为学习率；本稿把连续时间变量记 $\tau$，并注意 $\tau=\eta t_{\rm iter}$）。
- 数据：**白化/正交输入** $\Sigma^{11}=\sum_\mu x^\mu x^{\mu\top}=I$；输入-输出相关矩阵做 SVD：$\Sigma^{31}=\sum_\alpha s_\alpha u_\alpha v_\alpha^\top$（"任务"分解为独立模态 $\alpha$，目标强度 $s_\alpha$）。
- 初始化：**平衡初始化（balanced initialization）** $W^{(l+1)\top}W^{(l+1)}=W^{(l)}W^{(l)\top}$，或满足"各层奇异值平方沿深度相等"（$w_l^2=s^2$）；Saxe 证明该平衡在梯度流下**保持**；另有一类**正交初始化**给出深度无关的学习时间。

### 3.2 单模态 ODE 的推导 [P]（2 层情形即 Saxe Eq.(8)(10)(11) [T]）

在解耦的不变流形上，对模态 $\alpha$ 只留两个标量（或向量）$a,b$（前后层连接强度），乘积 $u=ab$ 驱动该模态的学习。**一般 $L$ 的推导**：设该模态上每层标量 $w_l$，乘积 $P=\prod_{l=1}^L w_l$（= 该模态的输入-输出增益），平衡初始化给 $w_l^2=P^{2/L}$。连续时间梯度流（时间单位 $\tau$，$\tau$ 与迭代步 $t$ 的关系见下）：
$$
\dot w_l=-\eta\,\frac{\partial E}{\partial w_l}=-\eta\,(P-s)\,\frac{P}{w_l}\quad\Longrightarrow\quad
\dot P=\sum_l\frac{P}{w_l}\dot w_l
=-\eta L\,P^{\,2-2/L}(P-s)
\ \Longrightarrow\
\boxed{\ \frac{dP}{d\tau}=L\,P^{\,2-2/L}(s-P),\quad \tau=\eta\,t_{\rm iter}\ } \tag{3.1}
$$
（本稿从权重梯度流重推并数值验证，见附录 A.3；Saxe 原文对 2 层给 $2u(s-u)$，即 $L=2$ 的特例。**一般 $L$ 的公式与原文记号对照【待核】。**）
**2 层（$L=2$）的精确解 [T]**（Saxe Eq.(10)(11)，此处 $u=P$）：
$$
\dot u=2u(s-u)\ \Longrightarrow\ u(t)=\frac{s}{1+\big(\frac{s}{u_0}-1\big)e^{-2st/\tau}},\qquad
t=\frac{\tau}{2s}\ln\frac{u_f(s-u_0)}{u_0(s-u_f)} . \tag{3.2}
$$
当 $u_0\ll s$、要求学到 $u_f=s-\epsilon$：$t\approx\frac{\tau}{2s}\ln(s/u_0)$——**对数慢，但没有 plateau**。

### 3.3 一般 $L$ 的隐式解与 plateau 时间标度 [P]

由 (3.1)：
$$
\tau(P)=\int_{P_0}^{P}\frac{dP'}{L\,P'^{\,2-2/L}(s-P')}
=\frac{1}{L}\int_{P_0}^{P}\frac{P'^{\,2/L-2}}{s-P'}\,dP' . \tag{3.3}
$$
在 plateau 区 $P\ll s$（此时 $s-P\approx s$）：
$$
\tau(P)\approx\frac{1}{Ls}\int_{P_0}^{P}P'^{\,2/L-2}dP'
=\frac{1}{Ls}\cdot\frac{P^{2/L-1}-P_0^{2/L-1}}{2/L-1}
=\frac{1}{s(L-2)}\Big(P_0^{-(L-2)/L}-P^{-(L-2)/L}\Big)\quad(L>2). \tag{3.4}
$$
故从 $P_0$ 到任意固定阈值（例如 $P=s/2$）的**plateau 时间**为
$$
\boxed{\ \tau_{\rm plat}(L>2)\ \approx\ \frac{1}{s\,(L-2)}\,P_0^{-(L-2)/L}
\ \ \Longleftrightarrow\ \
T_{\rm plat}\ \approx\ \frac{P_0^{-(L-2)/L}}{\eta\,s\,(L-2)}\ \ (\text{迭代步数})\ } \tag{3.5}
$$
**推导要点**：$P\ll s$ 时 $\dot P\approx LsP^{2-2/L}$（$P$ 很小时速度 $\sim P^{1-2/L}\to0$：**越深越慢**）；分离变量积分得 (3.4)；$L=2$ 时 (3.4) 不适用（积分变为 $\frac{1}{2s}\ln(P/P_0)$，对数），与 (3.2) 一致。
**深度/初始化标度**：若每层初始化尺度为 $\epsilon$（例如 $w_l(0)=\epsilon$），则 $P_0=\epsilon^L$，于是
$$
\boxed{\ T_{\rm plat}\ \propto\ \frac{\epsilon^{-(L-2)}}{\eta\,s\,(L-2)}\ } \tag{3.6}
$$
即 **plateau 时间随深度 $L$ 以幂指数 $L-2$ 增长、随初始化尺度以 $\epsilon^{-(L-2)}$ 增长**；对给定 $L$，$T_{\rm plat}\propto 1/(\eta s)$（与目标强度成反比——**弱模态学得最晚**）。
**数值验证（附录 A.3）**：(a) 直接对 $L=3,4,5$ 的标量权重做连续时间梯度流，到 $P\to0.999s$ 的时间与 (3.3) 的数值积分一致到 $10^{-4}$；(b) 用 (3.3) 计算到 $P=s/2$ 的时间，对 $a_0\in[10^{-2},10^{-10}]$ 拟合 $\log\tau$ vs $\log P_0$ 的斜率：$L=3$ 为 $-0.342$（理论 $-1/3$），$L=4$ 为 $-0.502$（理论 $-1/2$），$L=5$ 为 $-0.601$（理论 $-3/5$），$L=6$ 为 $-0.667$（理论 $-2/3$）；$L=2$ 斜率 $\approx0$（无 plateau）。$P_0=10^{-8}$ 时 $\tau_{\rm cross}/\tau_\infty$：$L=3$ 为 0.9982，$L=4$ 为 0.9999——(3.5) 的渐近常数是对的。

### 3.4 适用条件与失效情形（最终文档必须写）

**成立条件（缺一不可）**：
1. **线性激活**（无 ReLU/tanh）；否则层间不对易，只有近似或数值现象；
2. **连续时间梯度流**（小学习率极限）；离散 GD 在大 $\eta$ 时偏离，且可能因离散化"跳过" plateau；
3. **白化/正交输入** $\Sigma^{11}=I$（或输入相关矩阵与任务 SVD 对齐，使模态解耦）；
4. **平衡初始化**（各层奇异值平方相等）或落在 Saxe 的不变流形上；普通随机初始化只在近似意义上适用（Saxe 2014 的数值显示"合理的近似"）；
5. **平方损失 + 全 batch**；SGD 噪声、动量、weight decay 改变动力学。
**失效情形**：
- **非线性网络**：plateau→rapid transition 的现象仍常见（Saxe 用 tanh 数值验证），但**没有精确 ODE**；(3.5) 只是定性/标度模板；ReLU 网络的 plateau 由其他机制（saddle-to-saddle、激活模式切换）主导，见其他文献 **【待核/超出本稿范围】**。
- **SGD 噪声 / 小 batch**：噪声帮助逃离 plateau，实际 plateau 可远短于 (3.5)；此时 (3.5) 是**上界式的时间尺度**，不是预测。
- **非平衡初始化**：模态间竞争（Saxe Eq.(5) 的第二项）导致非单调动力学，单模态 ODE 不封闭。
- **深度极大 / 初始化极小**：$P_0^{-(L-2)/L}$ 可达天文数字，数值上等价于"永远学不会"——这正是深层线性网络训练困难的精确机制；实践中用正交/更聪明初始化（Saxe：正交初始化可做到深度无关的学习时间）规避。**【待核：Saxe 2014 §2 对 depth-independent delay 的精确条件与常数。】**
- **把 plateau 当成"验证误差平台"**：本条只描述**训练误差/模态增益**的 plateau；验证误差在 plateau 期间近似等于"未学到模态的贡献"，若目标本身不含弱模态，plateau 未必反映在验证误差上。**不要跨对象套用。**

---

## 4. 算法稳定性给出的泛化界（Bousquet–Elisseeff；Hardt–Recht–Singer）

### 4.1 定义（replace-one 均匀稳定性）

设训练集 $S=(z_1,\dots,z_n)$，$z_i=(x_i,y_i)$，$S^{(i)}$ 是只替换第 $i$ 个样本的邻接集。随机算法 $\mathcal A$ 是 $\varepsilon_{\rm stab}$-**均匀稳定**的，若
$$
\sup_{S,S^{(i)},\,z}\ \mathbb E_{\mathcal A}\big[\ell(\mathcal A(S),z)-\ell(\mathcal A(S^{(i)}),z)\big]\le\varepsilon_{\rm stab}. \tag{4.1}
$$
### 4.2 Bousquet–Elisseeff 泛化定理 [T]

若 $\ell\in[0,M]$ 且 $\mathcal A$ 是 $\varepsilon_{\rm stab}$-均匀稳定的，则对任意 $\delta\in(0,1)$，以概率 $\ge1-\delta$：
$$
\boxed{\ R(\mathcal A(S))\ \le\ \widehat R_S(\mathcal A(S))+2\varepsilon_{\rm stab}+(4n\varepsilon_{\rm stab}+M)\sqrt{\frac{\ln(1/\delta)}{2n}}\ } \tag{4.2}
$$
且期望意义下 $|\mathbb E[R-\widehat R]|\le\varepsilon_{\rm stab}$。**【T】（Bousquet–Elisseeff 2002, JMLR, Theorem 12 形式；常数/上界项以原文为准【待核】。）**
**推导要点（期望版，2–3 行）**：用 $S,S'$ 双样本技巧把 $\mathbb E[\widehat R_S(\mathcal A(S))]$ 与 $\mathbb E[R(\mathcal A(S))]$ 之差写成 $\frac1n\sum_i\mathbb E[\ell(\mathcal A(S^{(i)}),z_i')-\ell(\mathcal A(S),z_i')]$，逐项用 (4.1) 控制，得到 $\le\varepsilon_{\rm stab}$；再用 McDiarmid 集中把 $\widehat R$ 的随机性加上（$1/n$ 差有界 $+2M/n$）。

### 4.3 SGD 的稳定性：凸、Lipschitz 情形 [T]

设损失 $f(\cdot;z)$ 对每个 $z$ 是 **$\rho$-Lipschitz、凸**，SGD 步长 $\eta_t$（$t=1,\dots,T$）。则
$$
\boxed{\ \varepsilon_{\rm stab}\ \le\ \frac{2\rho^2}{n}\sum_{t=1}^{T}\eta_t\ } \tag{4.3}
$$
**【T】（Hardt–Recht–Singer 2016, Theorem 3.8 形式；条件：凸 + Lipschitz；若还 $\beta$-smooth，步长需 $\eta_t\le 2/\beta$ 的稳定性/收敛条件。常数与精确条件【待核】。）**
**推导要点**：邻接轨迹的差 $\delta_t=\mathbb E\|w_t-w_t'\|$；同一样本时梯度更新对 $\delta_t$ 不放大（凸+适当步长），以 $1/n$ 概率抽到不同样本时梯度差 $\le2\rho$；逐项累加得 $\delta_T\le(2\rho/n)\sum\eta_t$；最后 $\varepsilon_{\rm stab}\le\rho\,\delta_T$。
**强凸**情形：更新是 $(1-\eta_t\mu)$-压缩的，取 $\eta_t\asymp1/(\mu t)$ 可得 $\varepsilon_{\rm stab}=O(\rho^2/(\mu n))$，**与 $T$ 无关**（因此可训练任意久）**【待核精确常数】**。
**代入 (4.2) 的结论**：常数步长 $\eta$ 时泛化界 $\approx\widehat R+O(\rho^2\eta T/n+\sqrt{\log(1/\delta)/n})$——即 **"训练得越快（$\eta T$ 越小）泛化越好"**。这就是 Hardt et al. 的 "train faster, generalize better"。

### 4.4 SGD 的稳定性：非凸、$L$-smooth 情形 [P]（本文重推的递推界；与原文同阶）

在非凸光滑情形，邻接轨迹的差满足递推（**推导要点**：同一样本时 $\delta_{t+1}\le(1+\eta_t L)\delta_t$；以概率 $1/n$ 抽到不同样本时额外 $+2\eta_t\rho$）：
$$
\delta_{t+1}\le(1+\eta_tL)\delta_t+\frac{2\eta_t\rho}{n}\quad\Longrightarrow\quad
\varepsilon_{\rm stab}\le\rho\,\delta_T\le\frac{2\rho^2}{n}\sum_{t=1}^{T}\eta_t\prod_{s=t+1}^{T}(1+\eta_sL). \tag{4.4}
$$
常数步长时：
$$
\varepsilon_{\rm stab}\ \le\ \frac{2\rho^2}{nL}\Big[(1+\eta L)^T-1\Big]\ \approx\ \frac{2\rho^2}{nL}\,e^{\eta LT}. \tag{4.5}
$$
**【P】推导自洽；Hardt et al. 2016 Theorem 3.12 给出同阶（$T$ 的指数依赖）的结论，精确形式/步长调度条件【待核】。**
**含义**：只要 $\eta LT\ll\ln n$，界随 $n$ 收敛；若 $\eta LT$ 是 $\Theta(\ln n)$ 量级（例如 $\eta T\sim\log n/L$），则 $\varepsilon_{\rm stab}=O(\rho^2/(nL)\cdot n^{c})$，仍可随 $n$ 减小——这就是"$T$ 可以长到 $n^c$"的来源。

### 4.5 界松在哪里（定量 + 结构性）

1. **损失有界性 $M$**：深层网络/回归的 MSE 无界（权重范数在训练中可任意增长），(4.2) 的 $M$ 只能取实际观测上界，界随之膨胀；交叉熵同样无界。
2. **Lipschitz 常数 $\rho$ 随训练增长**：对 MSE，$\rho\approx\sup_x\|x\|\cdot\|w_t\|$（权重范数随时间增长），于是 $\rho^2T\eta$ 双重增长——**训练越久界越差，与"overparameterized 网络训练越久测试误差可下降"的经验相反**。
3. **最坏情况稳定性**：replace-one 稳定性要求**所有**样本、所有替换都稳定；而实际泛化只依赖数据分布下的平均行为（分布相关界、局部 Rademacher、PAC-Bayes 可以更紧，但需要额外假设）。
4. **插值/双下降区**：$\widehat R=0$ 的插值点附近，min-norm 插值的稳定性约 $O(1/n)$（故 (4.2) 给出 $O(1/n)$ 的泛化间隙），但有限 $n$ 的真实风险在 $\gamma\approx1$ 附近可以 $O(1)$（§1.5）——**界在双下降峰附近严重偏乐观**；反之在良态区它又过于悲观。
5. **不含隐式偏置**：稳定性只控制"算法对单样本扰动的敏感度"，完全不区分 GD 收敛到哪个插值解（min-norm？max-margin？）。同样的稳定性可以对应好/坏两种隐式偏置，因此**不能预测验证误差的水平**，只能给"算法类"的一致性。
6. **与本文设定（$n=256$、$T\sim10^3$）的量级**：取 $\rho\approx1$、$\eta=10^{-3}$、$T=1000$，凸界给出 $2\rho^2\eta T/n\approx7.8\times10^{-3}$，加上集中项 $\sqrt{\ln(1/\delta)/(2n)}\approx0.1$（$\delta=0.05$）——**总界 $O(0.1)$**，与验证误差可能的量级可比；但非凸界 (4.5) 取 $L\approx10$（$\eta L=0.01$，$\eta LT=10$）给出 $\frac{2}{256\cdot10}e^{10}\approx17$——**完全 vacuous**。**结论：稳定性界只能当"定性/量级"参考，不能当定量预测；最终文档中应标注为 [U] 且声明其松。**
**成立条件**：需要严格检查损失有界/Lipschitz/smooth/凸性假设；**失效**：假设不成立时界无意义（不是"仍然成立但松"，而是不适用）。

---

## 5. n-width / optimal recovery / minimax 估计率

> 本节只写**能确认的经典结果**；条件/常数不确定处标 【待核】。核心结论：**模型参数个数 $p$ 不是决定量；决定量是函数类的几何（n-width/谱/熵）与 $n$**。

### 5.1 定义

设 $F$ 是 Banach 空间 $X$ 中的函数类（如 Sobolev 球）。
- **Kolmogorov $n$-width**：$d_n(F,X):=\inf_{E:\ \dim E\le n}\ \sup_{f\in F}\ \inf_{g\in E}\|f-g\|_X$（用 $n$ 维**线性子空间**逼近的最坏误差）。
- **线性 $n$-width**：$d_n^{\rm lin}$：把 $\inf$ 限制为线性算子（投影）的版本。
- **Gelfand $n$-width**：用 $n$ 个线性泛函的信息恢复的最坏误差。
- **采样数（sampling numbers）**：$g_n(F,X):=\inf_{\{x_1,\dots,x_n\},\,R}\ \sup_{f\in F}\|f-R(f(x_1),\dots,f(x_n))\|_X$：允许 $n$ 次**函数值采样**、任意（可非线性）重建 $R$ 的最坏误差。
经典关系（对中心对称凸类）：$d_n(F,X)\le g_n(F,X)$ 的形式依空间与类而定；对 Hilbert 空间中的紧集有 $d_n\le g_n\le C d_{n/2}$ 之类的标准不等式（Pinkus 1985）。**【待核：精确定理编号/常数。】**

### 5.2 Sobolev/Besov 类的宽度：经典幂律 [T]

设 $B^s_2:=W_2^s([0,1]^d)$ 为 $L^2$ 型 Sobolev 单位球（$s>0$）。则
$$
\boxed{\ d_n\big(B^s_2,L^2([0,1]^d)\big)\ \asymp\ n^{-s/d}\ } \tag{5.1}
$$
**注释**（可直接写进最终文档的"能确认"清单）：
- **上界构造**：取 $n^{1/d}$ 个频率/基函数的方向做张量积 Fourier/小波截断，尾项 $\sum_{|k|>m}(1+|k|^2)^{-s}\lesssim m^{d-2s}\lesssim n^{-2s/d}$（$m\asymp n^{1/d}$），故宽度 $\lesssim n^{-s/d}$。
- **下界**：经典（Kolmogorov–Tikhomirov；Pinkus 1985, *n-Widths in Approximation Theory*）：对 $s>0$ 的 Sobolev 球，$d_n\gtrsim n^{-s/d}$。**（常数只依赖 $d,s$。）**
- **$d=1$ 的精确常数（可数值复核的"钉子"）**：$H_0^1([0,1])$ 单位球（半范 $\|f'\|_{L^2}$），用 Dirichlet 特征值 $\lambda_k=(k\pi)^2$，最优 $n$ 维子空间是前 $n$ 个特征函数张成的空间，**精确**地
  $$
  d_n\big(\{f:\|f'\|_{L^2}\le1\},L^2\big)=\frac{1}{(n+1)\pi}. \tag{5.2}
  $$
  **推导要点**：单位球上 $\|f\|_{L^2}^2=\sum c_k^2$，约束 $\sum(k\pi)^2c_k^2\le1$；截断到前 $n$ 个模式的尾能量被 $(n+1)^{-2}\pi^{-2}\cdot1$ 控制，且任何 $n$ 维子空间不能做得更好（特征值排序/极小极大原理）。**【T+可直接验证】**——最终文档若要给"n-width 的确切数"，用 (5.2)；一般 $d$ 只引 (5.1) 的 $\asymp$ 形式。
- Besov 类 $B^s_{p,q}$ 的宽度指数与 $p,q,s$ 有关（有 $p\ge2$ 与 $p<2$ 的不同情形）；**本稿不写未核实的 Besov 指数**。**【待核：Edmunds–Triebel / Pinkus 的 Besov 宽度表。】**

### 5.3 从 $n$ 个样本恢复 $f$：minimax 估计率 [T]

**统计模型**：$y_i=f(x_i)+\varepsilon_i$，$x_i\stackrel{iid}{\sim}{\rm Unif}([0,1]^d)$，$\mathbb E\varepsilon_i=0,\ \mathrm{Var}\le\sigma^2$；$f\in F=$ Sobolev（或 Besov）光滑度 $s$、半径 $L$ 的球。则
$$
\boxed{\ \inf_{\hat f}\ \sup_{f\in F}\ \mathbb E\|\hat f-f\|_{L^2}^2\ \asymp\ n^{-\frac{2s}{2s+d}}\ } \tag{5.3}
$$
**【T】Stone (1982), "Optimal global rates of convergence for nonparametric regression", Ann. Statist.；亦见 Tsybakov (2009) 教科书。条件：$s>0$、$d\ge1$、噪声方差有界非零、$x$ 有正密度；常数依赖 $s,d,L,\sigma$。**
**推导要点（下界，两函数 Fano/两点法）**：构造 $M\asymp n^{d/(2s+d)}$ 个在 $L^2$ 中相距 $\asymp n^{-s/(2s+d)}$、在 $L^2$ 经验度量下相距 $\asymp 1/\sqrt n$ 的函数（局部 packing），Fano 不等式给出 $\gtrsim n^{-2s/(2s+d)}$。**上界**：核/筛估计（或 KRR 取 $\lambda\asymp n^{-2s/(2s+d)}$）达到同阶。
**与 n-width 的接口**：$d_n\asymp n^{-s/d}$ 与 (5.3) 的指数关系是 $n^{-2s/(2s+d)}$（"wide"类需更多样本）；**注意 $n$ 的含义不同**（宽度里的 $n$ 是子空间维数；估计里的 $n$ 是样本数，且指数平方/含 $2s+d$ 而非 $d$）。**最终文档不可把这两个 $n$-指数混写。**
**确定性（无噪声）最优恢复 vs 统计估计的区别**：若允许无限精度函数值、要求**确定性** worst-case 恢复，Sobolev 球 $s>d/2$ 时 $g_n\asymp n^{-s/d}$（与宽度同阶）；**但 $s\le d/2$ 时 Sobolev 函数不连续、点值未定义，需先定义采样/改用 $L^2$ 平均**，此时 (5.3) 的统计模型仍良定义（在随机设计点处），因此**"从 $n$ 个函数值恢复"的确定性结论只在 $s>d/2$ 时可直接引用**。**【待核：sampling numbers 的精确条件与定理出处（Bakhvalov/Novak）。】**

### 5.4 RKHS 的谱表述：p 不是决定量 [T]/[H]

核 $K$ 的积分算子 $T_K$ 特征值 $\mu_j\asymp j^{-p}$（$p>0$；Sobolev $s$ 阶对应 $p=2s/d$）。设目标满足源条件 $f=T_K^{a}g$（$a\ge0$，$\|g\|\le R$；等价 $c_j^2\lesssim R^2\mu_j^{2a}$，即 §2.8 的 $q=2ap+1$）。则正则化估计（KRR/early stopping）在 $\lambda\asymp n^{-1/(2a+1/p)\cdot p\cdot?}$ 处的率（§2.8 的平衡推导）为
$$
\mathbb E\|\hat f-f\|^2\ \lesssim\ n^{-\frac{2\min(a,1)}{2\min(a,1)+1/p}}\quad\text{（$a<1$ 非饱和；$a>1$ 饱和）}. \tag{5.4}
$$
**等价写法**：$p=2s/d$、$a=r/(2s)$ 时 (5.4) 化为 $\min\{n^{-2r/(2r+d)},\ n^{-4s/(4s+d)}\}$——**非饱和时等于经典 minimax 率 (5.3)，饱和时被核的光滑度 $s$ 限住**。**【H】（偏差-方差平衡推导；非饱和区的上界有 Caponnetto–De Vito 2007 的严格定理支撑；饱和区指数与经典结果一致但本稿未逐条核对常数/条件【待核】。）**
**决定性变量**：$p$ 只通过谱 $\mu_j$ 的衰减指数 $p$（即"核有多光滑"）进入；**同一 $p$ 个参数可以对应完全不同的谱/有效维数**，因此"参数个数"本身不能给率。真正 a priori 可算的判据是：
$$
\text{有效维数 }\mathcal N(\lambda)=\sum_j\frac{\mu_j}{\mu_j+\lambda},\qquad
\text{源条件指数 } a,\qquad\text{噪声 }\sigma^2,\qquad n .
$$

### 5.5 与"模型类逼近误差"的关系：minimax = 逼近 + 估计 [T]（三角分解，严格）

设估计 $\hat f$ 的取值范围落在模型类 $\mathcal G$ 中（或 $\mathcal G$ 的某个 $n$-维子空间）。对任意 $f\in F$：
$$
\|\hat f-f\|^2\ \le\ 2\underbrace{\mathrm{dist}(f,\mathcal G)^2}_{\text{approximation}}\ +\ 2\underbrace{\|\hat f-g_f\|^2}_{\text{estimation over }\mathcal G}\qquad(g_f=\text{proj}_{\mathcal G}f),
$$
故
$$
\sup_{f\in F}\mathbb E\|\hat f-f\|^2\ \le\ 2\sup_{f\in F}\mathrm{dist}(f,\mathcal G)^2+2\sup_{g\in\mathcal G\cap F}\mathbb E\|\hat f-g\|^2 .
$$
**严格下界方向（简单但正确）**：若 $\mathrm{dist}(f_0,\mathcal G)\ge D$ 对某个 $f_0\in F$ 成立，则**任何**取值在 $\mathcal G$ 中的估计在 $f_0$ 处风险 $\ge D^2$（三角不等式，无需统计假设）。所以"参数化/模型类能不能表示目标"是第一道门槛；**$p$ 再大也只是把这道门槛降下去，不会突破 §5.3 的统计下界**。
**【T】式的直接推论；引用时注明它是"估计误差 + 逼近误差"的模板，不是 minimax 定理的替代。**

### 5.6 一句话结论（可直接写进最终文档）

> **模型参数 $p$ 不是误差的决定量。** 决定验证误差率的是：函数类的几何（n-width/谱 $\mu_j$）、目标相对该几何的源条件 $a$、样本数 $n$、噪声 $\sigma^2$、以及算法（GD/正则化）对谱的滤波方式。$p$ 通过"表示能力/谱/条件数"间接进入；在 $p>n$ 的插值区，$p$ 的继续增加通常只改变 aliasing/噪声分配，而不改变上述率的上限。

---

## 6. 分类的 margin / 隐式偏置（只写可严格陈述的部分）

### 6.1 可分数据 + 指数尾损失 ⇒ GD 收敛到 max-margin 方向 [T]

**设定**：$(x_i,y_i)_{i=1}^n$，$y_i\in\{-1,+1\}$，数据**线性可分**（$\exists w:\ y_i x_i^\top w>0\ \forall i$）；损失 $\ell(u)=\log(1+e^{-u})$（或任意指数尾损失 $\ell(u)\to0$ 指数快、$\ell'(u)$ 指数小）；全 batch GD：$w_{t+1}=w_t-\eta\nabla R_S(w_t)=w_t-\eta\sum_i\ell'(y_i x_i^\top w_t)y_ix_i$，任意初始化 $w_0$（或 $w_0=0$）。

> **定理（Soudry et al. 2018，JMLR；arXiv:1710.10345）**：存在 $\eta_0>0$，当 $\eta<\eta_0$ 时，$\|w_t\|\to\infty$，且
> $$
> \frac{w_t}{\|w_t\|}\ \xrightarrow{t\to\infty}\ \hat w_{\rm SVM}:=\arg\min_{w}\ \frac12\|w\|^2\quad\text{s.t. } y_ix_i^\top w\ge1\ \forall i, \tag{6.1}
> $$
> 且 $\|w_t\|=\log t+O(\log\log t)$、方向收敛速率 $O(1/\log t)$。**【T】（精确速率/条件以原文 Theorem 3 为准【待核】。）**
**推导要点**：对可分数据，$\inf_w R_S(w)=0$ 且不可达；GD 使 $R_S(w_t)\to0$、$\|w_t\|\to\infty$；沿"范数增长、方向收敛"的分解 $w_t=\|w_t\|\hat w+$ 有界残差，利用 KKT 条件证明极限方向满足 max-margin 的 KKT 方程（唯一性来自可分数据下 hard-margin SVM 解唯一）。
**成立条件**：线性可分、指数尾损失、全 batch、步长足够小、无 weight decay/正则；$t\to\infty$ 时**范数发散**（不是收敛到某个有限解）。
**失效情形**：
- 不可分数据：收敛到有限范数、对某个 $\lambda$ 的 $L^2$-正则解（"隐式正则"路径不同）；
- SGD/小 batch：隐式偏置改变（与 batch 噪声/损失形状有关，见后续文献），不一定是 max-margin **【待核】**；
- 大学习率/动量/weight decay：方向极限改变；
- **有限步 $T$**：只得到"尚未到达 max-margin"的中间解，其方向取决于初始化与 $T$——这与本文的设定（有限 $T$）直接相关：**$T$ 有限时不能直接套用 $t\to\infty$ 的 max-margin 结论**。

### 6.2 benign overfitting in classification：可严格陈述的部分 [T]/[?]

**可严格陈述的负例（不需要新定理）**：设标签 $y_i\in\{\pm1\}$ 与 $x_i$ **独立**（纯噪声，Bayes 误差 $=1/2$），$p\gg n$。任何在训练集上插值（0 训练误差）的分类器，其测试精度可任意依赖算法/数据：存在插值解在测试集上精度 $\approx1/2$（如只看零空间分量的分类器）；而 min-norm 插值在 $x_i$ 各向同性高斯、$p\gg n$ 时也趋于"随机方向"，测试精度 $\to1/2$。**因此"插值 ⇒ 泛化"在分类中同样不是定理。**
**benign overfitting 的结构性条件（Cao–Gu–Belkin 等）**：文献给出的一类充分条件是：数据 $x=\mu y+$（高维噪声项），信号方向 $\mu$ 与噪声协方差的谱满足"信号可被低维强方向承载、噪声分布在大量弱方向上"，且 $n$ 相对**有效噪声维数**足够大；此时 GD/Logistic 的插值解方向仍由信号主导，测试误差接近 Bayes 误差。**具体定理的条件/常数本稿未核对【待核】**；最终文档中只建议写：**"分类的 benign overfitting 需要 (i) 隐式偏置把信号方向放在 margin 大的位置，(ii) 噪声可被高维弱方向吸收，二者都是任务几何条件，不是 $p>n$ 的自动推论。"**
**不要写的东西**：本稿不写 "分类版 a priori 上界公式"，因为分类的 CE 损失无谱分解闭式、隐式偏置非 max-margin（有限 $T$）、且涉及不可约 Bayes 误差；套用回归模板（§1、§2）会产生错误结论。

---

## 7. 综合"定理式" a priori 误差上界模板

### 7.1 严格分解（核回归，无噪声 + 有噪声两项都写出）

记 $\mathcal H_N={\rm span}\{K(x_i,\cdot):i=1,\dots,n\}$（$\dim\le n$），$P_N$ 为 $L^2(P)$ 到 $\mathcal H_N$ 的正交投影（$P_N$ 一般不是插值算子；见下），$f_\infty$ 为 $T\to\infty$ 的 GD 极限（$=k(\cdot)^\top K^{-1}f(X)$，min-RKHS-norm 插值），$f_T=f_{\theta_T}$。由于 $f_T,f_\infty\in\mathcal H_N$：
$$
\boxed{
\begin{aligned}
R_{\rm te}(f_T)&=\ \big\|f-P_Nf\big\|_{L^2(P)}^2
&&\text{(A) 逼近/aliasing：}\mathcal H_N\text{ 之外的目标能量}\\
&\quad+\big\|P_Nf-f_\infty\big\|_{L^2(P)}^2
&&\text{(B) 表示/隐式偏置：min-norm 解与 }L^2\text{-最优解之差}\\
&\quad+2\langle P_Nf-f_\infty,\ f_\infty-f_T\rangle_{L^2(P)}+\big\|f_\infty-f_T\big\|_{L^2(P)}^2
&&\text{(C) 优化/早停：有限 }T\text{ 残差}\\
&\quad+\text{噪声项 (D)}+\sigma^2\ \text{(Dtest)}
\end{aligned}} \tag{7.1}
$$
**推导要点**：$f-P_Nf\perp_{L^2}\mathcal H_N\ni(f_T-P_Nf)$ 且 $\perp(f_\infty-P_Nf)$，故交叉项 $(A)$ 与其余项正交，只剩 $(B)$、$(C)$ 之间的内积；对 $(C)$ 用 $(\alpha+\beta)^2\le2\alpha^2+2\beta^2$ 可把内积吸收，得到**严格**的
$$
R_{\rm te}(f_T)\ \le\ \underbrace{\|f-P_Nf\|^2}_{A}+2\underbrace{\|P_Nf-f_\infty\|^2}_{B}+2\underbrace{\|f_\infty-f_T\|^2}_{C}\ +\ (D)+\sigma^2 . \tag{7.2}
$$
**关键事实（消除常见误解）**：
- **(A)** 只依赖 $f$、核与训练点位置；对**任意**算法/任意 $p$ 都存在（这是 n-width/最优恢复的领域，§5）。
- **(B)** 是**隐式偏置项**：它衡量"GD 收敛到的 min-RKHS-norm 插值"与"$\mathcal H_N$ 中 $L^2$-最优逼近"的差距。**线性 well-specified 无噪声情形 (B)=0**（因为 $f_\infty=P_Nf$，见 §2.9）；核情形一般 $\ne0$，这正是"min-norm/早停的隐式正则"进入验证误差的地方。
- **(C)** 是**优化项**：由 §2 的谱滤波精确控制：$\|f_\infty-f_T\|_{L^2(P)}^2=\sum_j(1-g_j)^2\cdot(\text{系数})^2$ 型。
- **(D)** 训练标签噪声（若 $\sigma^2>0$）：由 §2.5 的 $B_t$ 精确给出；对验证集期望为 $\frac{\sigma^2}{m}\|B_t\|_F^2$ 型。
- **(Dtest)** 测试噪声：不可约，$\sigma^2$。

**线性回归的特例（与 §1 完全对齐）**：$\varphi(x)=x$，$\mathcal H_N=\mathrm{row}(X)$，$P_N=P_{\rm row}$；$f_\infty=x^\top P_{\rm row}\beta$，故 (A)$=\|P_{\rm null}\beta\|^2$、(B)$=0$、(C)$=0$（$T=\infty$）；噪声 (D) 由 §1.4 给出。**这说明 (7.1) 不是新假设，而是把 §1、§2 的精确式按"逼近/偏置/优化/噪声"重排。**

### 7.2 四项模板 + 逐项所需 $f,\mathcal G$ 性质

$$
\boxed{\ \mathbb E R_{\rm te}\ \lesssim\ \underbrace{\mathcal A(f,\mathcal G)}_{\text{逼近/aliasing}}+\underbrace{\mathcal B(\mathcal G,\text{implicit bias})}_{\text{表示/隐式偏置}}+\underbrace{\mathcal C(\mathcal G,n,T,\eta,\text{spectrum})}_{\text{估计+优化}}+\underbrace{\mathcal D(\sigma^2,m)}_{\text{噪声/验证统计}}\ } \tag{7.3}
$$

| 项 | 精确形式（能做时） | 需要 $f$ 的什么 | 需要 $\mathcal G$/核的什么 | 分级 | 失效/说明 |
|---|---|---|---|---|---|
| $\mathcal A$ | $\mathrm{dist}_{L^2(P)}(f,\mathcal H_N)^2$（核）；线性情形 $\|P_{\rm null}\beta\|^2$ | $f$ 到模型类的 $L^2$ 距离/谱系数尾 | 模型类/核的逼近能力（n-width、谱） | **[E] 可算/[T] 可用 n-width 界** | 只在给定 $P$ 时有定义；$P$ 未知时只能用假设类 |
| $\mathcal B$ | $\|P_Nf-f_\infty\|^2$（核）；线性 well-specified $=0$ | $f$ 的 RKHS/$L^2$ 对齐结构 | 核的 RKHS 范数 vs $L^2$ 几何、min-norm 选择 | **[E] 可算（给定 $K,f(X)$）** | 对线性 well-specified 消失；对核一般非零；feature learning 下无定义 |
| $\mathcal C$ | 谱滤波公式：$\sum_j(1-g_j)^2\cdot$系数$^2$；有限 $n$ 用 §2.5 矩阵式 | $f$ 的谱系数 | $\lambda_i(K)$、$n$、$\eta$、$T$ | **[E] 有限 $n$ 精确（固定核）；[H] 大 $n$ 学习曲线/最优 $T$** | 步长需 $0<\eta<2/\|K\|$；$K$ 病态时数值/有限精度问题 |
| $\mathcal D$ | $\sigma^2/m$ 的估计噪声 + $\sigma^2$（测试噪声）；训练噪声项 $\frac{\sigma^2}{m}\|B_t\|_F^2$ | 噪声水平（若确定性标签 $\sigma=0$） | 无 | **[E]（给定噪声模型）/[T]（Hoeffding 界）** | 验证集复用/多重比较带来额外偏差；分布漂移时无效 |
| 验证统计 | $\widehat R_{\rm val}\pm O(\sqrt{\log(1/\delta)/m})$（有界损失 Hoeffding） | — | — | **[T]** | 只在同分布、独立验证点下成立；多次选模型要 Bonferroni/嵌套 CV |

### 7.3 严格 vs 启发式：给最终文档的分级

- **严格（可写"定理/命题"）**：(1.1)(1.2)(1.4)(1.7)（有限 $n,p$ 精确，条件明确）；(2.2)(2.3)(2.5)(2.6)（有限 $n$ 精确，条件明确）；(5.1)(5.2)(5.3)（经典定理）；(4.2)(4.3)（经典定理）；(6.1)（经典定理）；(7.2) 的分解不等式本身。
- **本文推导 [P]（推导自洽、已数值复核，但引用前需核对文献形式）**：(1.3) 的极限形状与 (1.6)；(2.11)-(2.13)（偏差-方差模板与率）；(3.1)(3.3)-(3.6)（Saxe 的一般 $L$ 版本，2 层部分是 [T]）；(4.4)(4.5)。
- **启发式模板 [H]（只能当"组织框架/量级估计"，不能当断言）**：(2.11) 的 $1/n$ 系数与 mode-counting 修正；§5.4 的饱和区指数；§6.2 的 benign overfitting 条件；所有"最优 $T$/最优 $\lambda$"的显式标度（依赖未知谱/噪声，实际要 CV）。
- **待核 [?]**：见附录 B。

### 7.4 验证集本身的统计（作为模板第 4 项的一部分）

若损失有界 $|(g(x)-f(x))^2|\le M$，验证集 $m$ 个独立同分布点，则对任意 $\delta$，以概率 $\ge1-\delta$：
$$
\big|\widehat R_{\rm val}(g)-R_{\rm te}(g)\big|\ \le\ \sqrt{\frac{M\ln(2/\delta)}{2m}}\quad\text{(Hoeffding)}. \tag{7.4}
$$
**【T】条件：验证点独立同分布、损失有界（回归里通常不成立！$f$ 无界时需截断/次高斯假设）。**
**对 v1.5 的可操作性**：$m$ 个验证点、10 seed 均值、以及"候选间比较"的噪声量级，应直接用 $1/\sqrt m$ / 配对差的标准误来定"最小可分辨差异"；**不要用不同 $p$ 的模型在同一个验证集上的多次选择后的最小误差作为无偏估计**（选择性偏差/winner's curse）。**[T]（标准统计）**

---

## 8. 对 v1.5 设定的直译（"公式地基"怎么变成 a priori 判据）

设定（来自题库审计，详见 `bank-audit.md`）：确定性表达式 $f$；MLP 候选，$p\sim10^4$–$10^5$，$n=256$（univar/mvar）；$T\sim10^2$–$10^3$，$\eta$ 给定；验证集 $m$ 个点。

1. **若候选处于 lazy/NTK 区**（训练中 $J_{\theta_t}\approx J_{\theta_0}$）：
   - 取 $\varphi(x)=J_{\theta_0}(x)$（$p$ 维），$K=JJ^\top$（$n\times n$），**验证误差可在训练前用 §2.5 (2.7) 精确算出**（用 $f$ 在训练/验证点的值、$\eta,T$）。
   - 若 $p>n$ 且目标落在特征张成空间内（线性 well-specified），进一步用 §1 的 (1.1)/(1.3) 给出"与 $T$ 无关的 aliasing 地板"——**这是 $p>n$ 时验证误差的下限，训练再久也不降**。
   - （A）项：$f$ 到 $\mathrm{span}\{J_{\theta_0}(x_i)\}$ 的 $L^2(P)$ 距离；用验证点/网格数值投影即可算。
2. **有限 $T$ 的作用**：§2 的谱滤波 (2.4)(2.5)：$\lambda_i$ 小的方向在 $T$ 步后仍是 $g_T(\lambda_i)\approx T\eta\lambda_i$，即**"大 $p$ 但有效秩小/谱衰减快"的模型在有限 $T$ 下等价于低秩拟合**；这是"$p$ 大但验证误差不爆炸"的精确解释（不是玄学）。
3. **early stopping / CV**：验证集选 $T$（或 $\lambda$）等价于选谱截断 $J\asymp(\eta T)^{-1/p}$（§2.8）；对给定 $f$ 的谱，最优 $T$ 可**在训练前估计量级**（用 $f$ 的谱系数 + 核谱），但这只是 [H] 级预测，最终选择仍靠验证集。
4. **不要做的事**：不要用"$p$ 越大 ⇒ 容量越大 ⇒ 验证误差越大/越小"做任何断言；不要用测试误差的单个 seed 峰值讨论 double descent（§1.5 的重尾）；不要把分类候选（bigram 的 token 预测）套回归闭式（§6.2）。
5. **报告规范**：将验证误差按 (7.3) 分项报告（即使只能给 (A)/(C) 的数值与 (B)/(D) 的界），并标注每项的分级（[E]/[T]/[P]/[H]）。

---

## 9. 最终文档可直接引用的公式清单（按依赖顺序）

| 编号 | 公式 | 条件 | 分级 |
|---|---|---|---|
| (1.1) | $\mathbb E[R_{\rm ex}\mid X]=\|P_{\rm null}\beta\|^2+\sigma^2\mathrm{tr}((XX^\top)^{-1})$ | 有限 $n,p$，$\Sigma=I$ | [E] |
| (1.2) | $\mathbb E\mathrm{tr}((XX^\top)^{-1})=n/(p-n-1)$；$\mathbb E\mathrm{tr}((X^\top X)^{-1})=p/(n-p-1)$ | $p>n+1$ / $n>p+1$ | [T] |
| (1.3) | $\mathbb E R_{\rm te}=(1-1/\gamma)\|\beta\|^2+\sigma^2(1+1/(\gamma-1))$ | $p>n+1$，$\Sigma=I$ | [P]（有限 $n$ 精确） |
| (1.4) | $\mathbb E R_{\rm te}=\sigma^2(n-1)/(n-p-1)$ | $p<n-1$ | [T] |
| (1.5) | $\gamma<1:\sigma^2/(1-\gamma)$；$\gamma>1:\sigma^2\gamma/(\gamma-1)+(1-1/\gamma)\|\beta\|^2$ | 比例极限 | [T]（Hastie et al.） |
| (1.7) | ridge 条件风险 $\lambda^2\beta^\top(M+\lambda)^{-2}\beta+\sigma^2\mathrm{tr}(M(M+\lambda)^{-2})$ | 有限 $n,p$ | [E] |
| (2.2) | $F_t=[I-(I-\eta K)^t]y$ | MSE + 固定特征/核 | [E] |
| (2.4)(2.5) | $g_t(\lambda)=1-(1-\eta\lambda)^t$；逐模式 $g_t(\lambda_i)$ | 稳定域 $0<\eta<2/\|K\|$ | [E] |
| (2.6) | 离散-连续界 $\|(1-x)^t-e^{-tx}\|\le e^{-tx}tx^2/(2(1-x))$ | $x\in[0,1)$ | [P]（已数值验证） |
| (2.7) | 有限 $n$ 测试风险 $\frac1m\|B_ty^0-f_{\rm te}\|^2+\frac{\sigma^2}{m}\|B_t\|_F^2$ | 固定核 | [E] |
| (2.8)-(2.10) | GD/ridge 滤波 + $\lambda_{\rm eff}\asymp1/(\eta T)$ | 近似对应 | [H]/[P] |
| (2.11)-(2.13) | bias/variance 谱模板、率、最优 $T$ | 幂律谱+source condition | [H] |
| (3.1)(3.5)(3.6) | $dP/d\tau=L P^{2-2/L}(s-P)$；$T_{\rm plat}\propto P_0^{-(L-2)/L}/(\eta s(L-2))$ | 线性、连续时间、平衡初始化、白化输入 | [P]（2 层 [T]） |
| (4.2)(4.3) | 稳定性泛化界；$\varepsilon_{\rm stab}\le2\rho^2\sum\eta_t/n$ | 有界/凸 Lipschitz | [T] |
| (4.4)(4.5) | 非凸递推界 $\varepsilon_{\rm stab}\lesssim\frac{2\rho^2}{nL}e^{\eta LT}$ | $L$-smooth | [P] |
| (5.1)(5.2)(5.3) | $d_n\asymp n^{-s/d}$；$d_n=1/((n+1)\pi)$（1D 精确）；minimax $n^{-2s/(2s+d)}$ | Sobolev/Besov 类 | [T] |
| (6.1) | 可分数据 GD 方向 → max-margin；$\|w_t\|=\log t+O(\log\log t)$ | 指数尾损失、小步长、全 batch | [T] |
| (7.1)(7.2)(7.3) | A+B+C+D 分解与模板 | 核回归/MSE | [E]+[H] |
| (7.4) | 验证集 Hoeffding 界 | 有界损失、独立同分布 | [T] |

---

## 附录 A：数值验证（命令 + 结果）

环境：Windows，pwsh；Python 3.10.19，NumPy 2.2.6。**全部脚本以 stdin 直接喂给 `python -`，未创建任何文件。** 复现方式：把下面的 `$code = @' ... '@` 块粘贴到 PowerShell 执行（或用任意方式把其中 Python 代码送进 `python -`）。

### A.1 逆 Wishart 迹的期望 + min-norm 插值的精确风险（§1.3、§1.4、§1.5）

```powershell
$code = @'
import numpy as np
rng = np.random.default_rng(12345)
def minnorm(X,y):
    G = X @ X.T
    return X.T @ np.linalg.solve(G, y)
# (a) E[tr((XX^T)^-1)] = n/(p-n-1), p>n+1
n,p,trials = 20,40,40000; trs=np.empty(trials)
for t in range(trials):
    X=rng.normal(size=(n,p)); trs[t]=np.trace(np.linalg.inv(X@X.T))
print("V1a E[tr((XX^T)^-1)] p=40,n=20: MC=%.6f formula=%.6f se=%.6f"
      % (trs.mean(), n/(p-n-1), trs.std(ddof=1)/np.sqrt(trials)))
# (b) exact expected test risk, p>n+1
n,p,sigma = 16,40,0.5; beta=rng.normal(size=p); trials=40000
ex=np.empty(trials)
for t in range(trials):
    X=rng.normal(size=(n,p)); eps=sigma*rng.normal(size=n)
    bh=minnorm(X, X@beta+eps); ex[t]=np.sum((beta-bh)**2)
pred=(1-n/p)*np.sum(beta**2)+sigma**2*(1+n/(p-n-1))
print("V1b test risk p>n: MC=%.6f formula=%.6f se=%.6f"
      % (ex.mean()+sigma**2, pred, ex.std(ddof=1)/np.sqrt(trials)))
# (c) underparametrised exact formula
n,p,sigma = 40,16,0.5; b=rng.normal(size=p); trials=40000; ex=np.empty(trials)
for t in range(trials):
    X=rng.normal(size=(n,p)); eps=sigma*rng.normal(size=n)
    bh=np.linalg.solve(X.T@X, X.T@(X@b+eps)); ex[t]=np.sum((b-bh)**2)
print("V1c underparam excess: MC=%.6f formula=%.6f" % (ex.mean(), sigma**2*p/(n-p-1)))
# (d) heavy tail at the interpolation threshold p=n+1
n=20; p=n+1; vals=np.empty(20000)
for t in range(len(vals)):
    X=rng.normal(size=(n,p)); vals[t]=np.trace(np.linalg.inv(X@X.T))
print("V1d p=n+1: mean=%.3f median=%.3f p95=%.3f (mean theory=+inf)"
      % (vals.mean(), np.median(vals), np.quantile(vals,0.95)))
'@
$code | python -
```

实际输出：
```
V1a E[tr((XX^T)^-1)] p=40,n=20: MC=1.052782 formula=1.052632 se=0.000563
V1b test risk p>n: MC=25.019114 formula=25.037723 se=0.021944
V1c underparam excess: MC=0.173214 formula=0.173913
V1d p=n+1: mean=152.955 median=21.224 p95=204.812 (mean theory=+inf)
```
**结论**：(1.2)(1.3)(1.4) 的有限样本公式与 MC 在标准误内一致（V1a 相对误差 $1.4\times10^{-4}$；V1b 相对误差 $7\times10^{-4}$，差 $0.85$ 个标准误；V1c 一致）；$p=n+1$ 处期望发散由"均值 153 vs 中位数 21"的重尾直接验证，支持 §1.5 的"期望发散、典型值有限"。

### A.2 核 GD 的谱滤波形式 + 有限 $n$ 精确测试风险（§2.2、§2.3、§2.5）+ early stopping vs ridge（§2.7）

```powershell
$code = @'
import numpy as np
rng = np.random.default_rng(7)
N,p,M = 40,64,200
Phi=rng.normal(size=(N,p)); Phi_te=rng.normal(size=(M,p))
wstar=rng.normal(size=p)/np.sqrt(p)
K=Phi@Phi.T; Kte=Phi_te@Phi.T; y0=Phi@wstar; y_te=Phi_te@wstar
evals,evecs=np.linalg.eigh(K); eta=0.9*2.0/evals.max(); sigma=0.2
def gd_alpha(y,T):
    a=np.zeros(N)
    for t in range(T): a=a-eta*(K@a-y)
    return a
T=50
# (a) closed form vs direct iteration
gK=np.eye(N)-np.linalg.matrix_power(np.eye(N)-eta*K,T)
print("V2b max|K alpha_T-(I-(I-eta K)^T)y| = %.3e"
      % np.abs(K@gd_alpha(y0,T)-gK@y0).max())
# (b) spectral-filter predictions
filt=1-(1-eta*evals)**T; R=evecs@np.diag(filt/evals)@evecs.T
print("V2c max|pred_spec-pred_loop| = %.3e"
      % np.abs(Kte@(R@y0)-Kte@gd_alpha(y0,T)).max())
# (c) exact finite-n test risk vs MC over training noise
B=Kte@R; bias2=np.mean((Kte@gd_alpha(y0,T)-y_te)**2)
exact=bias2+sigma**2*np.sum(B**2)/M; MC=4000; mse=np.empty(MC)
for s in range(MC):
    eps=sigma*rng.normal(size=N); mse[s]=np.mean((Kte@gd_alpha(y0+eps,T)-y_te)**2)
print("V2d noisy MSE: MC=%.6f exact=%.6f (bias %.6f var %.6f) se=%.6f"
      % (mse.mean(), exact, bias2, sigma**2*np.sum(B**2)/M, mse.std(ddof=1)/np.sqrt(MC)))
# (d) discrete-vs-continuous filter bound
x=np.linspace(1e-3,0.999,3000); t=30.0
lhs=np.abs((1-x)**t-np.exp(-t*x)); rhs=np.exp(-t*x)*t*x**2/(2*(1-x))
print("V2e bound holds on [1e-3,0.999]:", np.all(lhs<=rhs*(1+1e-9)))
# (e) early stopping vs ridge with noise
def risk(filt):
    B=Kte@(evecs@np.diag(filt/evals)@evecs.T)
    return np.mean((B@y0-y_te)**2)+sigma**2*np.sum(B**2)/M
Ts=np.arange(1,301); gd=np.array([risk(1-(1-eta*evals)**T) for T in Ts])
lams=np.logspace(-5,2,400); rd=np.array([risk(evals/(evals+l)) for l in lams])
Tb=Ts[np.argmin(gd)]; lb=lams[np.argmin(rd)]
print("V2f best GD T*=%d risk=%.4e | best ridge lam*=%.4e risk=%.4e"
      % (Tb, gd.min(), lb, rd.min()))
print("     lambda_eff=1/(eta T*)=%.3e | half-shrinkage=%.3e | lam*=%.3e"
      % (1/(eta*Tb), (1-2**(-1/Tb))/eta, lb))
'@
$code | python -
```

实际输出：
```
V2b max|K alpha_T-(I-(I-eta K)^T)y| = 9.992e-16
V2c max|pred_spec-pred_loop| = 4.441e-15
V2d noisy MSE: MC=0.488133 exact=0.487688 (bias 0.423620 var 0.064068) se=0.000451
V2e bound holds on [1e-3,0.999]: True
V2f best GD T*=22 risk=4.8254e-01 | best ridge lam*=1.9872e+00 risk=4.8027e-01
```
细项（同一脚本加打印；正文 §2.7 引用）：
```
best GD: T*=22 risk=4.8254e-01 (bias 4.3224e-01 var 5.0300e-02)
best ridge: lam*=1.9872e+00 risk=4.8027e-01 (bias 4.3249e-01 var 4.7782e-02)
lam->0: bias=4.2242e-01 var=7.1764e-02 (risk=4.9418e-01)
lambda_eff=1/(eta*T*)=4.4733e+00 ; lambda_half=(1-2^(-1/T*))/eta=3.0523e+00 ; lambda*=1.9872e+00
filter match at optima: max|gGD-gRidge|=0.1115  rms=0.0504
T= 10: exact=0.49349  MC=0.49366  se=0.00038
T= 22: exact=0.48254  MC=0.48270  se=0.00044
```
（注：`lam*=9.548e-03, risk=4.2242e-01` 是早期脚本的一个 bug 输出，**已作废**；以本文 `lam->0` 一行 + 细项为准：$\lambda^*=1.9872$、风险 0.48027。$\lambda\to0$ 时 bias 0.42242 但方差 0.07176，总风险 0.49418，故最优不在 $\lambda\to0$。）
**结论**：(2.2)(2.3) 的闭式与迭代一致到 $10^{-15}$；有限 $n$ 测试风险 (2.7) 与 MC 一致（$T=10,22$ 均在 1 个标准误内）；(2.6) 的离散-连续界在 $x\in[10^{-3},0.999]$ 全部成立；(2.9)(2.10) 的 $\lambda_{\rm eff}\asymp1/(\eta T)$ 与真最优 ridge 参数同量级（$1.5$–$2.2$ 倍），但**滤波器形状不同（RMS 差 0.05）⇒ 只能当标度对应，不是精确等价**。

### A.3 深度线性网络的 plateau 标度（§3.2、§3.3）

```powershell
$code = @'
import numpy as np
def rhs(P,L,b): return L*P**(2-2/L)*(b-P)
# (a) ODE vs direct continuous-time gradient flow on L scalar weights
def weight_flow(L,a0,b,eta=1.0,dt=1e-4,Tmax=200.0):
    w=np.full(L,a0**(1/L)); P=a0; t=0.0
    while P<0.999*b and t<Tmax:
        w=w+dt*(-eta*(P-b)*P/w); P=np.prod(w); t+=dt
    return t
def tau_cross(L,a0,b,Ptarget):
    u=np.linspace(np.log(a0),np.log(Ptarget),4000001); P=np.exp(u)
    return np.trapezoid(1.0/(L*P**(1-2/L)*(b-P)),u)
for L in [3,4,5]:
    a0=1e-2; ts=weight_flow(L,a0,1.0); to=tau_cross(L,a0,1.0,0.999)
    print("V3a L=%d: t_sim=%.4f tau_ode=%.4f ratio=%.4f" % (L,ts,to,ts/to))
# (b) plateau-time exponent
for L in [2,3,4,5,6]:
    a0s=np.logspace(-2,-10,9); taus=np.array([tau_cross(L,a,1.0,0.5) for a in a0s])
    slope=np.polyfit(np.log(a0s),np.log(taus),1)[0]
    print("V3b L=%d slope=%+.5f theory=%+.5f" % (L,slope,-(L-2)/L))
'@
$code | python -
```

实际输出：
```
V3a L=3: t_sim=6.1673 tau_ode=6.1674 ratio=1.0000
V3a L=4: t_sim=6.5228 tau_ode=6.5230 ratio=1.0000
V3a L=5: t_sim=6.6481 tau_ode=6.6482 ratio=1.0000
V3b L=2 slope=-0.08299 theory=+0.00000
V3b L=3 slope=-0.34194 theory=-0.33333
V3b L=4 slope=-0.50237 theory=-0.50000
V3b L=5 slope=-0.60080 theory=-0.60000
V3b L=6 slope=-0.66683 theory=-0.66667
```
另有精确对照（正文 §3.3）：$P_0=10^{-8}$ 时 $\tau_{\rm cross}/\tau_\infty$：$L=3$ 0.9982、$L=4$ 0.9999、$L=6$ 1.0000——(3.5) 的渐近系数正确。$L=2$ 的斜率 $-0.083$ 是有限 $a_0$ 修正（理论上 $L=2$ 为对数增长，无幂律），与 (3.2) 一致。

**结论**：(3.1) 的单模态 ODE 与真实连续时间权重动力学一致；(3.5)(3.6) 的 plateau 时间标度 $P_0^{-(L-2)/L}$（深度每加一层指数加一）在 $L\ge3$ 上被数值确认，$L=2$ 无幂律 plateau。

---

## 附录 B：待核清单（交付最终文档前必须外部核实）

1. **Bousquet–Elisseeff (2002) 定理 12 的精确形式**：(4.2) 的集中项是否为 $(4n\varepsilon_{\rm stab}+M)\sqrt{\ln(1/\delta)/(2n)}$；$[0,M]$ 有界损失下的常数。
2. **Hardt–Recht–Singer (2016) 的精确定理编号/假设**：Theorem 3.8（凸 Lipschitz 的 $\varepsilon_{\rm stab}\le\frac{2\rho^2}{n}\sum\eta_t$）与 Theorem 3.12（非凸）的步长调度条件；本稿 (4.4)(4.5) 为自推，需对照原文常数。
3. **Hastie et al. (2019/2022) Theorem 1/2/5 的精确表达式**：(1.5) 的定性结论已由原文摘要确认，但比例极限下的**精确常数项**（尤其 $\Sigma\ne I$、ridge 的 Stieltjes 闭式）本稿未逐条核；§1.7 的"固定 $\lambda$ 在比例极限下退化为 ridgeless"需对照原文 Theorem 5。
4. **逆 Wishart 期望公式**：(1.2) 的常数 $1/(p-n-1)$ 已数值验证，但**引用时需给标准出处**（Muirhead 或 von Rosen 的定理号）；$p=n+1$ 发散的表述要写清是期望（不是典型值）。
5. **Saxe 2014 一般 $L$ 的 ODE 记号**：本稿 (3.1) 用"$L$=权重矩阵个数、$s$=目标奇异值"；原文 $N_l$、$S^{31}$ 的记号与 2 层特例 (3.2) 已核对，但一般 $L$ 的原文公式（arXiv:1312.6120 §2）与"depth-independent delay 的正交初始化条件"需核对。**【本文数值只验证了自己的 ODE，不能替代原文核对。】**
6. **总体学习曲线模板 (2.11) 的有限 $n$ 修正**：Canatar–Bordelon–Pehlevan (2021) 的 mode-counting/自洽 $\kappa$ 版本；本稿的 $1/n$ variance 与 UV 截断处理是首阶模板。
7. **早停–ridge 精确关系**：Ali–Kolter–Tibshirani (2019) 的连续时间精确分析；本稿只给斜率/半收缩点两种匹配 + 数值量级验证，**不得声称"early stopping = ridge"**。
8. **Cao–Gu–Belkin benign overfitting in classification 的精确条件**（arXiv 编号/定理条件/常数）——本稿 §6.2 只给结构性描述，未核定理。
9. **Soudry et al. 2018 的精确速率**：$\|w_t\|=\log t+O(\log\log t)$、方向收敛 $O(1/\log t)$ 的精确定理编号/步长条件。
10. **n-width/Besov 的"可确认"范围**：(5.1) 的 $\asymp$ 对 Sobolev 类成立（Pinkus 1985）；**Besov $B^s_{p,q}$ 的指数表未核**，最终文档若写 Besov 必须补引用。采样数 $g_n$ 的精确条件（$s>d/2$）与定理出处（Bakhvalov/Novak）待核。
11. **饱和区率 (5.4) 的常数与阈值**：$a>1$（等价 $r>2s$）的饱和率 $n^{-4s/(4s+d)}$ 与 Caponnetto–De Vito / Steinwart 的写法对照（不同文献的 source condition 指数约定差 2 倍，**引用时必须以定义为准**）。
12. **验证集噪声/复用偏差**：(7.4) 的 Hoeffding 界要求损失有界；回归的 $f$ 无界，实际要截断或改次高斯；验证集复用（模型选择）带来的选择偏差需要额外处理（本稿只给标准提醒，未给定量修正）。

---

## 附录 C：参考文献（只列本稿实际引用到的）

1. T. Hastie, A. Montanari, S. Rosset, R. J. Tibshirani. *Surprises in High-Dimensional Ridgeless Least Squares Interpolation.* Annals of Statistics 50(2), 2022；arXiv:1903.08560. （§1；其摘要 §1.1 的定性结论本稿已对照）
2. A. M. Saxe, J. L. McClelland, S. Ganguli. *Exact solutions to the nonlinear dynamics of learning in deep linear neural networks.* ICLR 2014；arXiv:1312.6120. （§3；2 层公式 Eq.(8)(10)(11) 已对照原文）
3. O. Bousquet, A. Elisseeff. *Stability and Generalization.* JMLR 2:499–526, 2002. （§4.2；常数形式待核）
4. M. Hardt, B. Recht, Y. Singer. *Train faster, generalize better: Stability of stochastic gradient descent.* ICML 2016；arXiv:1509.01240. （§4.3–4.5；Theorem 2.2/3.8 等已对照原文前段，精确常数待核）
5. D. Soudry, E. Hoffer, M. S. Nacson, S. Gunasekar, N. Srebro. *The implicit bias of gradient descent on separable data.* JMLR 19(70), 2018；arXiv:1710.10345. （§6.1；速率待核）
6. A. B. Pinkus. *n-Widths in Approximation Theory.* Springer, 1985. （§5.1–5.2）
7. C. J. Stone. *Optimal global rates of convergence for nonparametric regression.* Annals of Statistics 10(4):1040–1053, 1982. （§5.3）
8. A. Caponnetto, E. De Vito. *Optimal rates for the regularized least-squares algorithm.* Foundations of Computational Mathematics 7(3):331–368, 2007. （§5.4；无 arXiv）
9. A. Canatar, B. Bordelon, C. Pehlevan. *Spectral bias and task-model alignment explain generalization in kernel regression and infinitely wide neural networks.* Nature Communications 12:2914, 2021；arXiv:2006.13198. （§2.8 的修正模板，待核）
10. B. Bordelon, A. Canatar, C. Pehlevan. *Spectrum dependent learning curves in kernel regression and wide neural networks.* ICML 2020；arXiv:2002.02561. （§2.8）
11. A. Ali, J. Z. Kolter, R. J. Tibshirani. *A continuous-time view of early stopping for least squares regression.* AISTATS 2019；arXiv:1810.10082. （§2.7，待核）
12. A. Tsybakov. *Introduction to Nonparametric Estimation.* Springer, 2009. （§5.3 的下界/上界标准参考）
13. Y. Cao, Q. Gu, M. Belkin. *Benign overfitting in linear classification.* NeurIPS 2021（arXiv 编号待核）. （§6.2，条件待核）
14. M. Belkin, D. Hsu, J. Xu. *Two models of double descent for weak features.* SIAM J. Math. Data Sci., 2020. （§1 背景）
15. S. Gunasekar, J. Lee, D. Soudry, N. Srebro. *Implicit bias of gradient descent on linear convolutional networks.* NeurIPS 2018（线性模型的隐式偏置=最小范数）. （§1 背景）

---

## 附录 D：与其他底稿的接口（给 Lead 的说明）

- 与 `lit-survey.md`：其 §1 的精确递推/§2 的核梯度流闭式与本稿 (2.2)(2.5) 一致；**但 $\eta$ 归一化约定不同**（其 §2 的谱式写作 $(1-\eta n\lambda_j)^T$，即总和损失步长；本稿 §2.8 用平均损失步长 $\eta_M$，换算 $\eta_M=n\eta_\Sigma$）。合并文档时请统一符号，建议在总文档开头写清。
- 与 `bank-audit.md`：其 $n=256$、$p\sim10^4$–$10^5$ 的设定对应本稿 $\gamma\sim10^1$–$10^2$，处于 §1.8 第 5 条的"远离插值峰、接近 null risk 地板"区；验证误差差异应主要来自 (A) 逼近项与 (C) 有限 $T$ 项，而不是 $p$ 本身。
- 与最终数学文档：可直接引用 §9 的公式清单（含条件与分级）；**凡标 [H] 的公式必须带上"启发式模板"字样**；凡标 [?] 的须先核附录 B。
