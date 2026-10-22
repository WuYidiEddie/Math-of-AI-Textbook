# 04 · 泛化与验证集：$p>n$ 为什么不必然坏，以及"验证差一点"到底可不可信

> 本章回答：double descent 的精确机制是什么？为什么在 v1.5 的确定性族里它多半不适用？$p>n$ 时风险到底由什么决定？10 个 seed 的验证均值该怎么比？"差 0.01"是噪声还是信号？
> 阅读方式：§2 是**逐步推导**（每一步都说清"为什么这一步成立"），§3–§4 讲什么时候不能套，§5–§7 讲验证集本身的统计学。

---

## 0. 推导地图

本章有两条线，别混：

- **§1–§4（理论线）**：$p>n$ 到底意味着什么风险，风险从哪来；
- **§5–§7（测量线）**：验证误差本身只是一个估计，它的噪声多大，怎么比。

| 步骤 | 做了什么 | 得到什么 | 节 |
|---|---|---|---|
| 1 | 分清三个量 | 训练误差 / 验证误差 / 总体风险 | §1 |
| 2 | 设定 ridgeless 插值器 | $w=X^{+}y$（SVD 上是 $VS^{+}U^\top y$） | §2.1 |
| 3 | 把"预测误差"翻译成"参数误差" | $R_{\mathrm{te}}=\lVert w-\beta\rVert^2+\sigma^2$ | §2.2 |
| 4 | 算插值器的参数误差平方 | $\mathbb E\lVert w\rVert^2=\sigma^2\operatorname{tr}(S^{-2})$ | §2.3 |
| 5 | 用 Wishart 逆矩 | $\mathbb E\operatorname{tr}((X^\top X)^{-1})=\frac{p}{n-p-1}$ | §2.4 |
| 6 | 合成一个公式 | $\mathbb E R_{\mathrm{te}}=\sigma^2\frac{\max(p,n)-1}{\lvert p-n\rvert-1}$ | §2.5 |
| 7 | 看形状 | 单峰在 $\lvert p-n\rvert=1$；右臂下降=第二降段，左臂的下降来自 $A$ | §2.6 |
| 8 | 拆三项 | 噪声地板 + 零空间偏差 + 噪声放大 | §2.7 |
| 9 | 检查适用边界 | 公式 $\propto\sigma^2$ → 无标签噪声就没有这个机制 | §3 |
| 10 | 回到验证集 | 配对差、SE、刀刃、复用偏差 | §5–§7 |

> **记号提醒 1**：本章的 $p$ 是"参数/特征个数"，$n$ 是"**训练点个数**"。v1.5 里 $n=\texttt{ds\_train\_size}$（univar/mvar 256；tabular/xor/spiral 1024；bigram 800 序列），**不是** $\texttt{samples\_seen}$（那是 $\text{steps}\times\text{bs}\in\{4096,\dots,32768\}$）。理论里说 "$p>n$" 时，$n$ 指哪一个必须写清楚，否则结论会互相打架。
>
> **记号提醒 2**：$R_{\mathrm{te}}$ 是**总体风险**（对数据分布的期望），$m_{\mathrm{te}}$ 是**验证集上的测量值**。本章反复要区分这两者：验证集给出的是 $m_{\mathrm{te}}$，它是 $R_{\mathrm{te}}$ 的一个带噪声的估计，不是 $R_{\mathrm{te}}$ 本身。
>
> **记号提醒 3**：$\sigma^2$ 是**标签噪声方差**。$\sigma=0$ 意味着"标签是 $f(x)$ 的精确求值"，也就是 v1.5 里 5 个确定性族的设定。

---

## 1. "泛化"到底在问什么

### 1.1 三个不同的量

训练集上误差小，不代表在没见过的点上误差小。泛化误差（总体风险）是：

$$
R_{\mathrm{te}}=\mathbb{E}_{x}\big[\mathrm{loss}\big(g_\theta(x),f(x)\big)\big],
\qquad x\sim \mathcal D
$$

$x$ 从测试/验证分布采样。**关键区分**：

| 量 | 定义 | 性质 |
|---|---|---|
| 训练误差 | 在固定训练点上的误差 | 可以被优化压到 0，**不含**泛化信息 |
| 验证误差 $m_{\mathrm{te}}$ | 在固定验证集上的测量值 | $R_{\mathrm{te}}$ 的**估计**，带噪声 |
| 总体风险 $R_{\mathrm{te}}$ | 对分布的真实期望 | 理论对象，**不可直接观测** |

本章的统计学部分（§5）全部围绕"从 $m_{\mathrm{te}}$ 推断 $R_{\mathrm{te}}$ 的差异"展开。

### 1.2 $p>n$ 的时候，训练误差为 0 说明不了任何事

当 $p>n$，线性模型有 $p-n$ 个以上的自由方向可以专门用来拟合噪声，于是

$$
\text{训练误差}\ \equiv\ 0 \quad\text{（插值）}
$$

但这只说明"模型足够大"，不说明"模型好"。真正决定 $R_{\mathrm{te}}$ 的是另外三件事：

1. **目标方向是否可辨识**（能不能被特征张成）；
2. **不可辨识方向上有没有目标的能量**（有 ⇒ 存在辨识地板 $I$，见第 02 章 §7.2）；
3. **有限预算下有没有走到**（$O(T)$）。

这就是本章 §2 要量化、§3 要限制适用范围的内容。

---

## 2. Double descent：把它算准

### 2.1 设定：ridgeless（最小范数）插值器

这是 double descent 结论的标准模型（Belkin et al. 2019；Hastie et al. 2019）。设定：

- 设计矩阵 $X\in\mathbb R^{n\times p}$，元素 iid $\mathcal N(0,1)$；
- 标签 $y=X\beta+\varepsilon$，$\varepsilon\sim\mathcal N(0,\sigma^2 I_n)$；
- 测试点 $x\sim\mathcal N(0,I_p)$，标签 $y_{\mathrm{te}}=\beta^\top x+\varepsilon_{\mathrm{te}}$，$\varepsilon_{\mathrm{te}}\sim\mathcal N(0,\sigma^2)$，与训练独立；
- 估计器：**最小范数插值器**（在所有满足 $Xw=y$ 的 $w$ 中取 $\lVert w\rVert$ 最小的）：

$$
w=X^{+}y,\qquad X^{+}=V S^{+}U^\top\quad (X=USV^\top)
$$

这个估计器有两个关键性质：

- 它**总是**插值训练集（$Xw=y$），所以它是"无正则化"的；
- 它的行为在 $p=n$ 处发生**定性**变化（插值阈值）。

> **为什么是它**：double descent 讨论的是"没有正则化、训练误差为 0"的模型；最小范数插值器是这类模型里最标准、信息量最大的一个。换成别的插值器，阈值位置不变，但数值会变。

### 2.2 第一步：把"预测误差"翻译成"参数误差"

在测试点上：

$$
\big(w^\top x-y_{\mathrm{te}}\big)^2=\big(x^\top(w-\beta)-\varepsilon_{\mathrm{te}}\big)^2
$$

对 $(x,\varepsilon_{\mathrm{te}})$ 取期望。交叉项为零（$\mathbb E[\varepsilon_{\mathrm{te}}]=0$ 且独立），于是

$$
\mathbb E_x\big[\big(x^\top(w-\beta)\big)^2\big]=\operatorname{tr}\big(\mathbb E[xx^\top](w-\beta)(w-\beta)^\top\big)=\lVert w-\beta\rVert^2
$$

（用了 $\mathbb E[xx^\top]=I_p$。）

$$
\boxed{\ R_{\mathrm{te}}=\lVert w-\beta\rVert^2+\sigma^2\ }
$$

这一步把一个**函数空间**的量换成了**参数空间**的量——和第 02 章 §7.2 的"辨识地板 $I$"是同一套语言。

### 2.3 第二步：纯噪声下 $\lVert w\rVert^2$ 的期望

先取 $\beta=0$（纯噪声），此时 $y=\varepsilon$，$w=VS^{+}U^\top\varepsilon$。给定 $X$（即给定 $U,S$）：

$$
\mathbb E_\varepsilon\big[\lVert w\rVert^2\big]=\mathbb E_\varepsilon\big[\lVert S^{+}U^\top\varepsilon\rVert^2\big]=\sigma^2\operatorname{tr}\big(S^{+}S^{+ \top}\big)=\sigma^2\operatorname{tr}(S^{-2})
$$

而 $S$ 是 $X$ 的奇异值，$\operatorname{tr}(S^{-2})$ 就是**小 Gram 矩阵逆的迹**：

$$
\operatorname{tr}(S^{-2})=\begin{cases}\operatorname{tr}\big((X^\top X)^{-1}\big), & p\le n\\[4pt]\operatorname{tr}\big((XX^\top)^{-1}\big), & p\ge n\end{cases}
$$

（用 $X^\top X=VS^2V^\top$ 与 $XX^\top=US^2U^\top$ 直接得到。）

> **注意这里已经出现了阈值**：$p\le n$ 时用 $X^\top X$（$p\times p$），$p\ge n$ 时用 $XX^\top$（$n\times n$）——总是取**小的那一个**。所以"插值"这件事在 $p>n$ 时并没有改变估计器的定义，但把参数误差换到了另一个子空间里。

### 2.4 第三步：Wishart 逆矩

$X^\top X\sim\mathrm{Wishart}_p(n,I_p)$。Wishart 分布逆矩阵的迹有精确期望（**精确**，不是渐近）：

$$
\mathbb E\operatorname{tr}\big((X^\top X)^{-1}\big)=\frac{p}{n-p-1}\qquad (p\le n-2)
$$

对偶地（$XX^\top\sim\mathrm{Wishart}_n(p,I_n)$）：

$$
\mathbb E\operatorname{tr}\big((XX^\top)^{-1}\big)=\frac{n}{p-n-1}\qquad (p\ge n+2)
$$

> 条件 $p\le n-2$ / $p\ge n+2$ 不是苛刻要求，而是**发散的位置**：$|p-n|\le1$ 时分母 $\le 0$，期望无穷（§2.6 会看到实测的重尾）。

### 2.5 第四步：合成一个公式

代回 §2.2、§2.3：

$$
\mathbb E R_{\mathrm{te}}=\sigma^2+\sigma^2\frac{p}{n-p-1}=\sigma^2\frac{n-1}{n-p-1}\quad (p<n)
$$

$$
\mathbb E R_{\mathrm{te}}=\sigma^2+\sigma^2\frac{n}{p-n-1}=\sigma^2\frac{p-1}{p-n-1}\quad (p>n)
$$

两支可以合成一个对称的式子（记 $M=\max(p,n)$、$m=\min(p,n)$）：

$$
\boxed{\ \mathbb E R_{\mathrm{te}}(p,n)=\sigma^2\cdot\frac{M-1}{M-m-1}=\sigma^2\frac{\max(p,n)-1}{\lvert p-n\rvert-1}\ }
$$

**实测核验**（`verification/generalization_stats.py`，C1；两侧用的是**同一个估计器**：最小范数插值器，SVD 稳定计算；测试误差对各向同性测试分布解析平均）：

```
p=  30 n= 100 (p/n= 0.30, |p-n|-1= 69): E R_te = 0.1298 +/- 0.0006   formula = 0.1291   ratio = 1.005   z = +1.14
p=  60 n= 100 (p/n= 0.60, |p-n|-1= 39): E R_te = 0.2305 +/- 0.0023   formula = 0.2285   ratio = 1.009   z = +0.89
p= 100 n=  30 (p/n= 3.33, |p-n|-1= 69): E R_te = 0.1293 +/- 0.0006   formula = 0.1291   ratio = 1.001   z = +0.31
p= 200 n=  30 (p/n= 6.67, |p-n|-1=169): E R_te = 0.1061 +/- 0.0002   formula = 0.1060   ratio = 1.001   z = +0.44
```

比值在 1.00–1.01 之间，$z$ 都在 $\pm1.2$ 以内：公式成立（有限 $n$ 下经验均值略高于公式，属有限样本修正）。

### 2.6 第五步：这条曲线的形状

**看公式本身**，比看图更可靠：

- 固定 $n$，扫 $p$：$p<n$ 时 $R_{\mathrm{te}}$ 随 $p$ **上升**；$p>n$ 时随 $p$ **下降**。峰值在 $\lvert p-n\rvert=1$。
- 固定 $p$，扫 $n$：**完全对称**（公式对 $(p,n)$ 对称），峰值同样在 $\lvert p-n\rvert=1$。
- 所以那条"脊"就是直线 $p=n$；常见资料里那张图横轴画的是样本数 $n$，有的画参数数 $p$，说的是同一件事。

**实测核验**（C1b：先打印公式曲线，再在四个点跑真实模拟）：

```
  formula, n=100, sigma=0.3:  p=1:0.0909  p=10:0.1001  p=50:0.1818  p=90:0.9900  p=98:8.9100  p=99:inf  p=101:inf  p=110:1.0900  p=200:0.1809  p=1000:0.1000
  simulated p=   1, n=100: E R_te = 0.0908 +/- 0.0001   formula = 0.0909   ratio = 0.999
  simulated p=  50, n=100: E R_te = 0.1808 +/- 0.0013   formula = 0.1818   ratio = 0.994
  simulated p= 110, n=100: E R_te = 1.1048 +/- 0.0290   formula = 1.0900   ratio = 1.014
  simulated p= 200, n=100: E R_te = 0.1808 +/- 0.0010   formula = 0.1809   ratio = 1.000
```

**这里有一个必须说清的细节**：这条曲线是**单峰**的——从左边**升**到阈值、从右边**降**。经典 double descent 图里左边那条"下降"臂，**不来自这个公式**。

- 这个公式是**纯方差**部分（$A\equiv0$ 的线性各向同性设计下，信号被精确拟合：$X^+X=I$）；
- 经典图的左臂来自**逼近误差 $A(\text{容量})$ 随模型类变大而下降**（更大的随机特征/ReLU 类能更好地表达 $f$）。方差项同时在上升，两者相加才有"先降后升"的形状；
- **右臂（"第二降段"）才是这个公式自己给出的**：$p>n$ 时噪声被摊薄，方差项下降。

> 所以"double descent"这个形状是**逼近 + 方差**两条曲线叠加的结果。只背公式会漏掉左臂；只背图会漏掉左臂的来源。

**阈值处期望发散，但单次实现有限**（C3 实测）：

```
p=41 n=40 (|p-n|-1=0): median = 2.6807 | p90 = 17.0457 | p99 = 162.6947 | p99.9 = 783.0306 | mean = 12.5034   formula = diverges
p=50 n=40 (|p-n|-1=9): median = 0.4292 | p90 = 0.7391 | p99 = 1.4228 | p99.9 = 2.1237 | mean = 0.4848   formula = 0.4900
```

$p$ 只从 41 涨到 50，**中位数**几乎不动（2.68 → 0.43），**均值**却从 12.50 掉到 0.48；但在 $p=41$ 处均值（12.50）是中位数（2.68）的 4.7 倍——**重右尾**。

> **结论（硬规矩）**：double descent 的"峰高"是期望量，而期望在阈值处是无穷；单次实现看到的是重尾分布的一次抽样，**远小于期望**。任何"我们观察到 double descent 的峰高达 X"的说法，如果不注明是哪个统计量、几个样本、什么分布，都是不可复现的。

### 2.7 第六步：把风险拆成三项

把 $\beta\ne0$ 的信号项加回来。对 $p>n$（信号项只在插值区有闭式）：

$$
\mathbb E\lVert w-\beta\rVert^2=\Big(1-\frac1\gamma\Big)\lVert\beta\rVert^2+\frac{\sigma^2 n}{p-n-1},\qquad \gamma=\frac pn
$$

**实测核验**（C2）：

```
p=60 n=15 gamma=4.0 sigma=0.1: E||theta_hat - theta*||^2 = 0.7522 +/- 0.0038   formula = 0.7534   z = -0.32
p=60 n=30 gamma=2.0 sigma=0.1: E||theta_hat - theta*||^2 = 0.5112 +/- 0.0043   formula = 0.5103   z = +0.20
p=60 n=15 gamma=4.0 sigma=0.0: E||theta_hat - theta*||^2 = 0.7531 +/- 0.0038   formula = 0.7500   z = +0.81
```

于是总体风险可以拆成**三项**，正好对应本项目的四项分解：

$$
\mathbb E R_{\mathrm{te}}\ \approx\ \underbrace{\sigma^2}_{\text{噪声地板（不可约）}}\ +\ \underbrace{\Big(1-\frac1\gamma\Big)\lVert\beta\rVert^2}_{I:\ \text{零空间偏差（辨识地板）}}\ +\ \underbrace{\frac{\sigma^2 n}{p-n-1}}_{\text{噪声放大}}
$$

- **零空间偏差**随 $p/n$ **增大**：$p$ 越大，可辨识方向占比越小（第 02 章 §6）；
- **噪声放大**随 $p/n$ **减小**：$p$ 越大，噪声被摊到更多方向上，单个方向的方差越小。

> 这正是"第二降段"的机制：**噪声放大项在退场**，代价是零空间偏差在进场。$p\to\infty$ 时模型退化成"什么都不学"的平凡预测器，风险趋向 $\sigma^2+\lVert\beta\rVert^2$（纯噪声时就是 $\sigma^2$）。

所以"$p>n$ 一定更差"和"$p>n$ 一定更好"都不对，**是两股力量在赛跑**。

**再看 $p<n$ 那一侧**（这里没有偏差项）：

当 $p<n$，$X$ 列满秩，于是 $X^+X=I$，插值器把信号**精确**拟合（$A=0$、无零空间偏差），误差**只剩噪声**：

$$
w=X^+(X\beta+\varepsilon)=\beta+X^+\varepsilon
\quad\Longrightarrow\quad
\mathbb E\lVert w-\beta\rVert^2=\frac{\sigma^2 p}{n-p-1}
$$

于是 $p<n$ 的一侧是"**容量越大、插值越狠、方差越大**"的单调上升段（§2.6 的左臂）；要看到"下降"，必须引入**逼近误差 $A$** 随容量下降的那部分。

> **把两段放在一起看**：
>
> | | 逼近项 $A$ | 辨识/方差项 | 净效果 |
> |---|---|---|---|
> | $p<n$ | 线性设计下 $\equiv0$；一般特征类下随容量**下降** | 噪声放大 $\sigma^2p/(n-p-1)$ **上升** | 经典图的"左臂下降"来自 $A$，本公式只给右半 |
> | $p>n$ | $\equiv0$ | 零空间偏差**上升** $\ (1-1/\gamma)\lVert\beta\rVert^2$，噪声放大**下降** | 右臂下降 = 第二降段 |

---

## 3. 无标签噪声时会发生什么

### 3.1 机制上，double descent 是"噪声方差"的产物

回看 §2.5 的公式：整条曲线的幅度**正比于 $\sigma^2$**。$\sigma=0$ 时，公式给出的 $R_{\mathrm{te}}$ 恒等于 $0$（信号项另算）——噪声驱动的尖峰**整支消失**。

所以：

> double descent 的第二降段不是"容量大所以好"，而是**噪声被摊薄**。没有标签噪声，就没有可摊薄的噪声。

### 3.2 v1.5 的 5 个确定性族

v1.5 的 5 个确定性族（univar/mvar/spiral/xor/tabular）标签是确定性函数/规则（`notes/bank-audit.md` §4.4），没有标签噪声。此时：

- **核/惰性区**：$R_{\mathrm{te}}(T)=\sum_j(1-\eta\lambda_j)^{2T}a_j^{2}$，对 $T$ **单调下降**（第 02 章 §8、第 03 章 §2）；
- **容量增大**：如果核/特征能更好地表达目标，$A$ 下降；不会因为"噪声方差"冲高；
- **训练误差 0 不提供验证信息**，但也不会自动导致验证爆炸。

**所以：**

> 在这些族里看到"训久了验证变差"，不要第一反应叫过拟合；
> 先查：有限样本 aliasing、分类边界复杂度、特征学习、EoS、验证噪声。

### 3.3 bigram 是例外

bigram 的目标是**随机 bigram 转移矩阵** $P(y\mid x)$，交叉熵有不可约的熵地板：

$$
H(Y\mid X)=2.281\ \text{–}\ 3.604\ \text{nats}\quad(\text{median }2.973)
$$

任何模型都不能低于这个地板；候选差异是"地板之上"的估计/优化误差。实测 $\mathrm{CE}/\text{地板}\in[1.021,1.566]$，**没有候选低于地板**。

> 注意口径：bigram **不是**"确定性目标"，把它当"无噪声"套会错（`notes/bank-audit.md` §4.3）。它的"噪声"是转移矩阵本身随机，表现为熵地板，而不是方差尖峰。

### 3.4 v1.5 离阈值有多远

把 v1.5 的 $p/n$ 摆到 §2.5 的公式旁边（`notes/bank-audit.md` §1.3）：

| 族 | $n_{\text{train}}$ | $p/n$ min | $p/n$ median | $p/n$ max | $p>n$ 的候选 |
|---|---:|---:|---:|---:|---|
| univar | 256 | 1.254 | 66.63 | 1296 | 300/300 (100.0%) |
| mvar | 256 | 2.441 | 68.00 | 1300 | 300/300 (100.0%) |
| tabular | 1024 | 0.596 | 12.33 | 325.5 | 290/300 (96.7%) |
| xor | 1024 | 0.705 | 12.81 | 327.5 | 145/150 (96.7%) |
| spiral | 1024 | 0.377 | 12.75 | 325.0 | 139/150 (92.7%) |
| bigram（序列口径） | 800 序列 | 9.870 | 85.29 | 672.7 | 300/300 (100.0%) |

即：绝大多数候选都 $p>n$（92.7%–100%），但 $p/n$ 的中位数在 $10^1$–$10^2$，**离阈值 $p=n$ 差着好几个数量级**。

按 §2.5 的公式，此时 $\lvert p-n\rvert-1$ 极大，噪声放大项 $\sigma^2n/(p-n-1)$ 早已是 $O(1/n)$ 量级——**阈值尖峰在 v1.5 里根本没有出现的位置**。

> **一句话**：v1.5 里的"$p>n$"绝大多数不是"处在 double descent 的危险区"，而是"处在第二降段很靠右的地方"。真正决定胜负的是第 02 章的四分解，不是插值阈值的噪声尖峰。

---

## 4. Benign overfitting：$p>n$ 有时真的没事

### 4.1 条件

"Benign overfitting"研究的是：在什么条件下，一个把训练数据（含噪声）插值到 0 的模型，测试误差仍接近最优？

结论不是"$p>n$ 总是 benign"，而是需要条件，例如：

- 协方差**有效秩**相对 $n$ 较小；
- 尾部特征值衰减足够快；
- 信号方向与噪声方向的有效秩分离。

### 4.2 本教程的用法

- **无标签噪声时**，这些条件退化为"目标与特征子空间对齐"：目标在可辨识方向上，零空间没有目标能量 ⇒ 辨识地板 $I=0$，$p>n$ 无害（第 02 章 §7.3、§7.4）；
- **不要用"良性过拟合"给任意 $p>n$ 的模型背书**：它是一组条件，不是一句安慰；
- 判断标准仍然是四分解：$A+I+O+F$。

---

## 5. 验证集本身有多准

### 5.1 两个噪声源

v1.5 每题：

- 测试集固定（univar/mvar 256 点；tabular/spiral/xor 2048 点；bigram 200 序列）；
- 每个候选跑 10 个 seed；
- 指标是 10-seed 均值 `mean_metric`。

噪声来源：

1. **训练随机性**（初始化、minibatch）→ 用 10 seeds 平均；这是 `std_metric` 记录的东西；
2. **测试集抽样**（固定的，不再重采样）→ 所有候选共用同一测试集，比较时用**配对差**。

> **重要口径**：v1.5 的测试集是**固定且所有候选共用**的，所以"测试集抽样"不产生 seed 间的波动，它是一个**所有候选共享的固定偏移**，在配对差里被抵消（§5.3 实测）。`std_metric` 里**没有**测试集重采样噪声，只有训练随机性（`notes/bank-audit.md` §3）。

### 5.2 10-seed 均值的标准误

$$
\mathrm{SE}(\mathrm{mean})=\mathrm{std\_metric}/\sqrt{10}
$$

v1.5 各族的 `std_metric` 中位数与对应 SE（`notes/bank-audit.md` §3.1，**设定事实**）：

| 族 | metric | 候选数 | `std_metric` 中位数 | SE(10-seed 均值) 中位数 |
|---|---|---:|---:|---:|
| univar | test_mse | 300 | 0.006463 | 0.002044 |
| mvar | test_mse | 300 | 0.03603 | 0.01139 |
| tabular | test_ce | 300 | 0.01479 | 0.004677 |
| xor | test_ce | 150 | 0.01476 | 0.004668 |
| spiral | test_ce | 150 | 0.01109 | 0.003506 |
| bigram | test_ce | 300 | 0.009824 | 0.003107 |

对照题内候选差的实际尺度（`notes/bank-audit.md` §3.2）：题内 $\lvert\Delta\mathrm{mean}\rvert$ 中位数 0.059–0.220，而 $\lvert\Delta\rvert/\text{pair\_std}$ 中位数是 3.63–8.96，**88.3%–94.0% 的候选对满足 $\lvert\Delta\rvert>1$ 个 pair_std**。所以大多数题是能判的；不能判的是少数"刀刃题"（§5.4）。

> `pair_std = sqrt((std_a²+std_b²)/2)` 是**独立近似**下的尺度，比配对 SE 保守，所以"88%–94% 的候选对差超过 1 个 pair_std"是一个偏保守的下限。

### 5.3 候选比较：用配对差

同一题、同一测试集，比较 A 和 B：

$$
\Delta=\mathrm{mean}(A)-\mathrm{mean}(B),
\qquad
\mathrm{SE}_\Delta=\frac{\mathrm{std}(\text{逐 seed 配对差})}{\sqrt{10}}
$$

判据（10 seeds，双侧 5%）：$\lvert\Delta\rvert>2.262\times\mathrm{SE}_\Delta$ 才算显著（$t_{0.975,9}=2.262$）。

**为什么要配对？** 两个原因，都要分清：

1. **测试集噪声在配对差里抵消**：同一批测试点 ⇒ 测试集抽样误差对 A、B 是**同一个**偏移，相减即消掉；
2. **共享的种子噪声在配对差里抵消**：seed $s$ 下 A 与 B 的初始化/minibatch 随机性部分相关，相减后只剩"差异本身"的噪声。

**实测核验**（C4：三个候选共用一个固定测试集，10 seeds）：

```
  candidate A (8 features, 200 steps): 10-seed mean test error = 0.01114 (sd 0.00048) | population risk = 0.01107
  candidate B (8 features, 210 steps): 10-seed mean test error = 0.01139 (sd 0.00046) | population risk = 0.01129
  candidate C (4 features, 200 steps): 10-seed mean test error = 0.27647 (sd 0.00969) | population risk = 0.28305
  A - B: diff = -0.000253 (population truth -0.000223) | SE independent = 0.000210 | SE paired = 0.000177 (ratio 1.19x) | |t| = 1.43 vs 2.262 -> knife-edge (not significant)
  A - C: diff = -0.265328 (population truth -0.271980) | SE independent = 0.003067 | SE paired = 0.003050 (ratio 1.01x) | |t| = 87.00 vs 2.262 -> SIGNIFICANT
```

A 与 B 几乎一样（真实差 $-0.00022$）：$|t|=1.43<2.262$ → **刀刃**。A 与 C 差一个量级：$|t|=87$ → **显著**。

**测试集抵消了多少？**（C4b：固定模型，300 个独立测试集）

```
  A vs B: sd(test error of A) = 0.00068 | of B = 0.00070 | sd of the paired difference on the SAME test set = 0.00036 | independent formula would give 0.00097 -> 2.72x
  A vs C: sd(test error of A) = 0.00068 | of C = 0.01774 | sd of the paired difference on the SAME test set = 0.01774 | independent formula would give 0.01776 -> 1.00x
```

- 两个候选**相近**时，配对把测试集噪声压掉 2.72 倍；
- 两个候选**差别大**时几乎不压（1.00 倍）——因为此时噪声由差别大的那个候选自己的测试集敏感度主导。

> **正确表述**：配对**总是**让测试集噪声在差里抵消；抵消的**幅度**取决于两个候选有多像。别把它讲成"配对总是显著"。

### 5.4 刀刃题

如果候选差小于 2–3 个标准误，这题就是"刀刃"：

- 任何规则/理论预测都可能翻；
- 应该标低置信，而不是硬选。

**门槛有多大？** 用各族的中位噪声水平算，$2.262\times\mathrm{SE}$ 的判别门槛是：

| 族 | SE(10-seed 均值) 中位数 | $2.262\times\mathrm{SE}$ |
|---|---:|---:|
| univar | 0.002044 | 0.0046 |
| mvar | 0.01139 | 0.0258 |
| tabular | 0.004677 | 0.0106 |
| xor | 0.004668 | 0.0106 |
| spiral | 0.003506 | 0.0079 |
| bigram | 0.003107 | 0.0070 |

也就是说：**在 univar 上，小于 0.005 的差距测不出来**（只能当噪声）；在 mvar 上要小于 0.026 才是噪声。题目里写的"差 0.01"，在 univar/xor/spiral/bigram 上确实在噪声范围内，在 mvar 上则未必。

> **量级感**：C4 里 A−B 的真实差是 $-0.00022$，10 个 seed 也测不出显著；v1.5 的题内差中位数是 $0.059$–$0.220$，比它大两个数量级。**"刀刃"是少数题的状态，不是常态**——但正是这些题最容易被写成"定律"。

---

## 6. 一个数字例子

假设某题两个候选：

- A：10-seed 均值 0.500，std 0.015
- B：10-seed 均值 0.490，std 0.015

用**独立近似**（把 A、B 当成互不相关的两组）：

- $\mathrm{SE}\approx 0.015/\sqrt{10}\approx 0.0047$；
- 差的 SE $\approx\sqrt{2}\times 0.0047\approx 0.0067$；
- $2.262\times 0.0067\approx 0.015$；
- $\lvert\Delta\rvert=0.010<0.015$ → **不显著**。

用**配对差**（同一测试点、同一 seed 编号）：

- $\mathrm{SE}_\Delta=\mathrm{std}(\text{逐 seed 差})/\sqrt{10}$，通常**小于**上面的 0.0067；
- 若逐 seed 差的标准差是 0.010，则 $\mathrm{SE}_\Delta\approx0.0032$，$2.262\times0.0032\approx0.0072$；
- $\lvert\Delta\rvert=0.010>0.0072$ → **显著**。

> 同一个数字，两种算法给出相反结论。这就是"用独立样本公式算配对差"为什么是坑：它把本该抵消的噪声留在里面，**低估显著性**。

---

## 7. 验证集的复用偏差

如果你用同一个验证集反复挑选模型/超参 $K$ 次，会系统性乐观：

$$
\mathbb E\big[\max_k\ \text{验证误差}\big]-\min_k\ \text{真实风险}\ \lesssim\ \sigma\sqrt{2\log K}
$$

其中 $\sigma$ 是单个候选的验证误差噪声标准差。注意 $\sqrt{2\log K}$ 是**上界**（不是下界），精确渐近是

$$
\sqrt{2\log K}-\frac{\log\log K+\log 4\pi}{2\sqrt{2\log K}}
$$

**实测核验**（C5：$K$ 个独立 $\mathcal N(0,1)$ 噪声）：

```
K=   3: E[max] = 0.8449 | sqrt(2 log K) = 1.4823 (upper bound, gap -0.6374) | refined asymptotic = 0.5968
K=  10: E[max] = 1.5350 | sqrt(2 log K) = 2.1460 (upper bound, gap -0.6109) | refined asymptotic = 1.3619
K=  50: E[max] = 2.2488 | sqrt(2 log K) = 2.7971 (upper bound, gap -0.5484) | refined asymptotic = 2.1009
K= 200: E[max] = 2.7459 | sqrt(2 log K) = 3.2552 (upper bound, gap -0.5093) | refined asymptotic = 2.6104
```

（$K=3$ 时精确值 0.5968 与 refined asymptotic 0.5968 几乎相同；$K$ 越大上界越松。）

**对 v1.5 的含义**：

- $K$ 越大，偏差越大；
- v1.5 的候选指标是**离线预计算**的，题库层面没有复用偏差（答案键就是 3 个候选 10-seed 均值的 argmin，`notes/bank-audit.md` §4.6）；
- 但如果你用验证集做**二次筛选**（挑阈值/挑规则/挑"看起来更合理的解读"），偏差会回来。

---

## 8. 实战：怎么判断"验证集上的好坏"

1. **先看差异是否显著**：配对差 vs $2.262\times\mathrm{SE}_\Delta$（10 seeds）。
2. **显著后再看四分解**：$A$/$I$/$O$/$F$ 哪项主导。
3. **不显著就标刀刃**：不要因为 0.01 的差距写"科学结论"。
4. **报告要带噪声**：均值 $\pm\mathrm{SE}$，或者配对差 + 置信区间。
5. **不要用训练误差替代验证误差**。
6. **区分统计显著与实际重要**：v1.5 的族内指标量纲不同（`test_mse` 与 `test_ce` 不可直接比，见 `notes/bank-audit.md` §4.11）。

---

## 9. 常见坑

| 坑 | 后果 |
|---|---|
| 用单 seed 曲线下结论 | 把噪声当信号 |
| 用独立样本公式算配对差 | 低估显著性（测试集噪声没抵消） |
| 忽略 10 seeds 的 std | 把 0.01 当真实差异 |
| 把"训练久了变差"一律叫过拟合 | 混淆 EoS/aliasing/边界复杂度 |
| 在刀刃题上写"定律" | 不可复现 |
| 用验证集反复调参后不校正 | 乐观偏差 |
| 引用 double descent 的"峰高" | 期望在阈值处发散，单次实现是重尾抽样（§2.6） |
| 把 bigram 当"确定性目标" | 忽略熵地板 $H(Y\mid X)$ |
| 谈 $p/n$ 时不声明 $n$ 是训练点数还是 samples_seen | 结论互相打架 |

---

## 10. 一页总结

- $p>n$ 不是泛化的充分条件，也不是不泛化的理由；风险由"目标在不可辨识方向有没有能量"决定。
- double descent 的定量骨架是 $\mathbb E R_{\mathrm{te}}=\sigma^2\frac{\max(p,n)-1}{\lvert p-n\rvert-1}$：单峰在 $\lvert p-n\rvert=1$，**右臂下降（第二降段）由它给出**，**左臂下降来自逼近误差 $A$ 随容量下降**（纯线性设计里 $A\equiv0$，所以左臂是上升的）；**整条曲线的幅度正比于 $\sigma^2$**，所以无标签噪声时它多半不适用（5 个确定性族；bigram 是熵地板的例外）。
- 阈值处期望**发散**、单次实现是**重尾**抽样：不要引用"峰高"。
- v1.5 的 $p/n$ 中位数在 $10^1$–$10^2$，离阈值好几个数量级，双降尖峰不在题库的活动范围里。
- 验证集是估计：10 seeds 用**配对差**比较，$\lvert\Delta\rvert>2.262\,\mathrm{SE}_\Delta$ 才算显著；配对同时消掉测试集偏移与共享种子噪声。
- 刀刃题标低置信，不硬判；报告带 $\pm\mathrm{SE}$。
