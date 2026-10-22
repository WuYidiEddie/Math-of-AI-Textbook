# 05 · 函数的结构与架构：为什么同一个网络换个任务就翻车

> 本章回答：$A$（逼近误差）到底由什么决定，**能不能在训练前就算出来**？$n$-width 是什么、怎么推、怎么数？深度和宽度各自改了什么，为什么在 v1.5 的真实族上"深更好"既成立又失效？六个族的目标画像里，哪些是理论判断、哪些是实测？
> 阅读方式：§1–§3 是**推导线**（每一步说清"这一步为什么成立"）；§4–§6 是**实测线**，全部数字来自 `verification/architecture_approximation.py`（只碰 dev 半边，输出在 `verification/architecture_approximation_result.md`）。

---

## 0. 推导地图

本章有两条线，**别混**：

- **§1–§3（理论线）**：$A$ 是什么、怎么算；信息极限怎么来的；架构通过哪些通道影响训练；
- **§4–§6（实测线）**：在 v1.5 的真实族上，深度/宽度到底怎么影响指标；六个族的目标画像长什么样。

| 步骤 | 做了什么 | 得到什么 | 节 |
|---|---|---|---|
| 1 | 把"架构"拆成三条通道 | 函数类 → $A$；核谱 → $O(T)$；隐式偏置 → $I$ | §1.2 |
| 2 | 定义 $A$，指出它为什么算不出来 | $A=\inf_{g\in G}\lVert f-g\rVert^2$，但 $G$ 是参数化的非凸无限维集合 | §2.1 |
| 3 | 换成特征空间 | $G=\mathrm{span}\{\varphi_j\}$ 时 $A$ 是**正交投影**，有闭式 | §2.2 |
| 4 | 数值实现：不要用正规方程 | 用 QR：$A=\lVert f-QQ^\top f\rVert^2$ | §2.2 |
| 5 | 三类表达式的基与谱 | 多项式 / 随机 ReLU / 轴对齐 ReLU，固定预算 $m$ | §2.3 |
| 6 | 测"结构匹配" | 加性目标偏爱加性类；复合目标抛弃加性类 | §2.4 |
| 7 | 推出信息极限的率 | $E_n\asymp n^{-s/d}$，五行推导 | §3.2 |
| 8 | 把 $B(n)$ 数清楚 | $N(B)=\#\{k:\lVert k\rVert_1\le B\}$，闭式壳计数 | §3.3 |
| 9 | 核版本 | $E_n=\sqrt{\lambda_{n+1}}$ | §3.4 |
| 10 | $A$ 的第二个度量（惰性区） | 最小范数权重 $\lVert f\rVert^2_H=\sum_j c_j^2/\lambda_j$ | §5 |
| 11 | 在真实族上分离 $A$ 与 $O(T)$ | spiral 深网赢、xor 浅网赢 | §4.2 |

> **记号提醒 1**：本章的 $A$ 一律指"逼近误差项"，不是矩阵。前置章 `0.1` 把 Gram 矩阵写成 $A_{\mathrm{mat}}$ 正是为了避免这个混淆。
>
> **记号提醒 2**：$m$ = 基函数个数（表达式类的**预算**），$n$ = 训练样本数，$T$ = 优化步数，$p$ = 参数个数，$\eta$ = 学习率，$d$ = 输入维度，$s$ = 光滑度阶。§2 用 $m$，§3 用 $n$，§5 用 $n/T$，三者不要串。
>
> **记号提醒 3**：本章只讨论**确定性目标**（v1.5 里 univar/mvar/spiral/xor/tabular 五族）。bigram 的目标是随机过程，$A$ 的定义要换成"分布逼近"，见正文 `06`。

---

## 1. 架构不是"容量旋钮"

### 1.1 回到四分解

正文 `01` 的结论是

$$
R_{\mathrm{te}}\approx A+I+O(T)+F
$$

其中 $A$ 是"表达式类里根本没有的那部分"，$I$ 是"在能拟合的函数里选错了"，$O(T)$ 是"$T$ 步还没走到"，$F$ 是"测不准"。

常见错误是把架构选择简化成"容量越大越好"。这个错误的信息论根源是：**"容量"这个词把三个不同的东西压成了一个数**。

### 1.2 架构的三条通道

一个架构改动，同时沿三条通道影响最终误差：

| 通道 | 架构改了什么 | 落到哪一项 | 能不能在训练前算 |
|---|---|---|---|
| **函数类** | 特征空间 $\mathcal S$ 变了（更深/更宽/换激活 = 换一个子空间） | $A=\lVert f-P_{\mathcal S}f\rVert^2$ | **能**（§2） |
| **核谱 / 条件数** | 可学方向数 $\#\{j:\eta\lambda_jT\gtrsim1\}$ 变了 | $O(T)$ | **能**（§5） |
| **隐式偏置** | 在无穷多个插值解里选谁（min-norm、max-margin、谱偏置） | $I$ | 部分能（§5） |

所以"深好还是浅好"这个问题本身是**病态**的：深度同时压低 $A$（能表达更多复合结构）**并**抬高 $O(T)$（更深的核谱动态范围更大、有效步长更小）。净效应是两者的差：

$$
\Delta(\text{净收益})\ \approx\ \underbrace{\Delta A}_{\text{结构收益}}\ -\ \underbrace{\Delta O(T)}_{\text{优化代价}}
$$

**谁的收益大，取决于目标函数的结构和预算。** 这就是"换个任务就翻车"的机制——不是理论自相矛盾，是两个被比较的东西换了。

### 1.3 本章要做的事：把"架构直觉"换成三个能算的量

| 量 | 含义 | 需要什么 | 节 |
|---|---|---|---|
| **类内逼近误差** | 目标离表达式类有多远 | 目标的解析式（或可采样） | §2 |
| **信息极限 $n$-width** | $n$ 个样本最多能定多准 | 维度 $d$、光滑度 $s$ | §3 |
| **惰性极限误差 + 学习窗口** | 不动特征的话，训练结果是什么 | 初始化处的 NTK | §5 |

三个量的共同点：**都不需要训练**。这就是"用数学预测训练结果"在本章能走到的最远处。

> 诚实的边界（对应前置章 `0.6` §11 的四档纪律）：这三个量都是**基线**，不是预测。它们假设了最优优化（$O(T)=0$）、惰性（不特征学习）、以及目标的可解析性。实测残差才轮到 $I$ 和 $O(T)$。

---

## 2. 逼近误差 $A$ 的第一个度量：类内正交投影

### 2.1 定义，以及它为什么算不出来

$$
A(G,f)\ :=\ \inf_{g\in G}\lVert f-g\rVert_{L^2(\mu)}^{2},
\qquad
\lVert h\rVert_{L^2}^{2}=\int_{\mathcal X}\lvert h(x)\rvert^{2}\,\mathrm d\mu(x)
$$

这里 $G$ 是**表达式类**，$\mu$ 是数据分布。

困难有两个，都不是"算力不够"：

1. **$G$ 是参数化的**：神经网络类 $\{\,g_\theta:\theta\in\mathbb R^p\,\}$ 是非凸的无限维集合，上面的下确界既没有闭式，也没有通用算法能在一般情况下求出它（这是非凸、无限维的变分问题，不是"算力不够"）；
2. **更根本**：$A$ 只回答"能不能表达"，不回答"用 $n$ 个样本能不能**找到**"。后者是 $I$（辨识）与信息极限（§3）。

### 2.2 定理：$A$ 是正交投影（严格定理）

> **定理（严格定理）** 设 $G=\mathrm{span}\{\varphi_1,\dots,\varphi_m\}\subset L^2(\mu)$ 是有限维闭子空间，$f\in L^2(\mu)$。则
> $$
> A(G,f)=\inf_{g\in G}\lVert f-g\rVert^{2}=\lVert f-P_Gf\rVert^{2}=\lVert f\rVert^{2}-\lVert P_Gf\rVert^{2},
> $$
> 且取到最小值的唯一元素是 $g^{\ast}=P_Gf$（$P_G$ 为到 $G$ 的正交投影算子）。

**证明**：令 $r=f-P_Gf$。投影的定义给出 $r\perp G$。对任意 $g\in G$，有 $P_Gf-g\in G$，于是

$$
\lVert f-g\rVert^{2}=\lVert r+(P_Gf-g)\rVert^{2}=\lVert r\rVert^{2}+\lVert P_Gf-g\rVert^{2}\ \ge\ \lVert r\rVert^{2},
$$

等号成立当且仅当 $g=P_Gf$。∎

**这个定理的意义**：$A$ 从"一个需要优化的东西"变成了"一次投影"。**没有任何优化、没有随机种子、没有训练曲线**——这正是第 01 章反复强调的"把随机的东西换成确定的东西"。

**数值实现**：不要用正规方程 $\hat a=(\Phi^\top\Phi)^{-1}\Phi^\top f$（它把 $\Phi$ 的条件数**平方**）。用 QR：$\Phi=QR$，$Q$ 列正交，则 $P_Gf=QQ^\top f$：

```python
import numpy as np

def comps(total, parts):
    """Weak compositions of `total` into `parts` parts."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in comps(total - first, parts - 1):
            yield (first,) + rest

def multi_indices(d, deg):
    """All multi-indices with |alpha|_1 <= deg (weak compositions)."""
    idx = []
    for total in range(deg + 1):
        idx.extend(comps(total, d))
    return idx

def cheb_axis(t, deg):
    """Chebyshev polynomials T_0..T_deg of t in [-1,1] by the stable recurrence."""
    polys = [np.ones(len(t)), t]
    for _ in range(2, deg + 1):
        polys.append(2.0 * t * polys[-1] - polys[-2])
    return polys

def cheb_basis(X, deg):
    """Total-degree Chebyshev basis on [0,1]^d."""
    d = X.shape[1]
    axes = [cheb_axis(2.0 * X[:, ax] - 1.0, deg) for ax in range(d)]
    cols = [np.ones(len(X))]
    for alpha in multi_indices(d, deg):
        c = np.ones(len(X))
        for ax, k in enumerate(alpha):
            c = c * axes[ax][k]
        cols.append(c)
    return np.stack(cols, axis=1)

def proj_error(Phi, f):
    """A = ||f - P_span(Phi) f||^2, via QR (never the normal equations)."""
    Q, _ = np.linalg.qr(Phi, mode="reduced")
    resid = f - Q @ (Q.T @ f)
    sse = float(resid @ resid)
    return sse, float(np.sqrt(sse / max(float(f @ f), 1e-30)))

x = np.linspace(0, 1, 4001)[:, None]
f = np.sin(2 * np.pi * x[:, 0])
print("target f(x) = sin(2 pi x) on [0,1], 4001 grid points")
for deg in (1, 3, 7, 15, 31):
    Phi = cheb_basis(x, deg)
    sse, rel = proj_error(Phi, f)
    print(f"  degree {deg:3d} | terms {Phi.shape[1]:4d} | A = {sse:.4e} | rel L2 = {rel:.4e}")
```

```
target f(x) = sin(2 pi x) on [0,1], 4001 grid points
  degree   1 | terms    3 | A = 7.8415e+02 | rel L2 = 6.2616e-01
  degree   3 | terms    5 | A = 1.7561e+01 | rel L2 = 9.3703e-02
  degree   7 | terms    9 | A = 9.8034e-05 | rel L2 = 2.2140e-04
  degree  15 | terms   17 | A = 1.7445e-19 | rel L2 = 9.3395e-12
  degree  31 | terms   33 | A = 1.1502e-27 | rel L2 = 7.5836e-16
```

读法：**7 次多项式（9 个基函数）就把一个光滑目标逼近到 $2.2\times10^{-4}$ 的相对 $L^2$ 误差**。而第 01 章测的那道真题里，三个候选网络是 4–5 层、宽 192–256，参数量 $10^5$ 量级——**它们的表达能力远远超过这个 9 维子空间，所以那题的 $A$ 不绑定**。

### 2.3 三类表达式：基、谱、预算

设表达式类的预算固定为 $m$ 个基函数。三类典型基：

| 类 | 基函数 | 结构 | 基函数个数与预算 |
|---|---|---|---|
| `poly` | $T_{\alpha}(x)=\prod_{i=1}^{d}T_{\alpha_i}(x_i)$，$T_k$ 是 Chebyshev 多项式 | 全局谱方法，总次数 $\lvert\alpha\rvert_1\le\deg$ | $\binom{d+\deg}{d}$ 个（弱组合计数） |
| `relu` | $\varphi_j(x)=\bigl(w_j^\top x+b_j\bigr)_{+}$，$w_j\sim\mathcal N(0,I)$ | 通用随机特征（一层无限宽极限） | $m$ 个（随机抽） |
| `addit` | 同上，但 $w_j=v_j e_{i_j}$，$i_j$ 随机 | 轴对齐 ⇒ 每个特征只看**一个**坐标 | $m$ 个（随机抽） |

三件事必须说清楚：

1. **加性类是通用类的子集**（集合意义）：轴对齐方向也是合法方向。但**随机抽样几乎不会抽到轴对齐方向**，所以下面比较的是"结构对齐的抽样"对"结构不对齐的抽样"，而不是"类的包含关系"。
2. **弱组合计数**：总次数不超过 $\deg$ 的多指标个数是
   $$
   \#\{\alpha\in\mathbb N^{d}:\lvert\alpha\rvert_1\le\deg\}=\binom{d+\deg}{d}
   $$
   （把 $\lvert\alpha\rvert_1$ 的"额度"分配到 $d$ 个坐标上，是标准的隔板法）。
3. **枚举方式是个陷阱**：用 `itertools.product(range(deg+1), repeat=d)` 是 $d$ 维网格，指数级；用弱组合枚举才是多项式级：

```python
import math

def comps(total, parts):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in comps(total - first, parts - 1):
            yield (first,) + rest

def multi_indices(d, deg):
    idx = []
    for total in range(deg + 1):
        idx.extend(comps(total, d))
    return idx

print("a total-degree Chebyshev class of degree `deg` in d dimensions has")
print("  C(d+deg, d) basis functions -- the budget m=120 buys:")
for d in (1, 2, 4, 8, 16):
    deg = max(k for k in range(0, 200) if math.comb(d + k, d) <= 120)
    print(f"  d = {d:2d}: affordable degree {deg:3d} -> {math.comb(d + deg, d):4d} terms"
          f"   (next degree would need {math.comb(d + deg + 1, d)} terms)")
print()
print("the enumeration must be by weak composition, not by a full grid:")
d, deg = 16, 4
print(f"  itertools.product(range({deg+1}), repeat={d}) would enumerate "
      f"{(deg+1)**d:,} tuples")
print(f"  weak composition enumeration gives {len(multi_indices(d, deg))} tuples")
```

```
a total-degree Chebyshev class of degree `deg` in d dimensions has
  C(d+deg, d) basis functions -- the budget m=120 buys:
  d =  1: affordable degree 119 ->  120 terms   (next degree would need 121 terms)
  d =  2: affordable degree  14 ->  120 terms   (next degree would need 136 terms)
  d =  4: affordable degree   4 ->   70 terms   (next degree would need 126 terms)
  d =  8: affordable degree   2 ->   45 terms   (next degree would need 165 terms)
  d = 16: affordable degree   1 ->   17 terms   (next degree would need 153 terms)

the enumeration must be by weak composition, not by a full grid:
  itertools.product(range(5), repeat=16) would enumerate 152,587,890,625 tuples
  weak composition enumeration gives 4845 tuples
```

**这一小段就是维度诅咒的全部内容**：预算 $m$ 固定时，可负担的次数从 $d=1$ 的 119 掉到 $d=16$ 的 1。

### 2.4 实测：结构匹配 > 类的大小

下面全部来自 `verification/architecture_approximation.py` 的 `S1`（dev 半边；相对 $L^2$ 误差，越小越好；三类的预算都是 $m=120$）：

```
  relative L2 error of the best approximation, m = 120

  target                                |   poly    |   relu    |  addit    |  winner
  --------------------------------------------------------------------------------------
  smooth 1-D  sin(2 pi x)              | 4.06e-15 | 6.68e-03 | 5.98e-03 | poly
  fast 1-D    sin(20 pi x)             | 4.57e-15 | 5.77e-01 | 5.54e-01 | poly
  kink 1-D    |x - 0.5|                | 6.11e-04 | 1.11e-02 | 6.77e-03 | poly
  additive 2-D sin(2pi x)+cos(2pi y)   | 7.50e-10 | 1.25e-01 | 4.54e-02 | poly
  composite 2-D sin(12 x y)            | 7.36e-04 | 3.89e-01 | 7.92e-01 | poly
  additive 4-D sum_j sin(2 pi x_j)     | 1.15e-01 | 3.61e-01 | 6.53e-02 | addit
  --------------------------------------------------------------------------------------
```

**读法一（结构匹配）**：只看两个 ReLU 类（大小、连接方式预算完全相同，只有结构不同）：

- 加性目标上，轴对齐类**赢**：2 维 $4.54\times10^{-2}$ 对 $1.25\times10^{-1}$；4 维 $6.53\times10^{-2}$ 对 $3.61\times10^{-1}$；
- 复合目标 $f(x,y)=\sin(12xy)$ 上，轴对齐类**输**：$7.92\times10^{-1}$ 对 $3.89\times10^{-1}$。

**为什么**：单变量函数的和做不出乘积（这是**启发式**层面的直觉，实测支持它）；反过来，通用类的随机方向天然可以混合坐标。

**读法二（维度诅咒藏在固定预算里）**：`poly` 的可负担次数是 119（$d=1$）、14（$d=2$）、4（$d=4$），于是它的误差从 $4\times10^{-15}$ 涨到 $1.15\times10^{-1}$；而**结构化的加性类在 4 维仍是 $6.53\times10^{-2}$**。所以高维下的关键不是"换个更大的类"，而是"换个结构更匹配的类"。

**读法三（`poly` 那一列不能过度解读）**：多项式类是**光谱类**，对任何光滑目标都能做到任意精度——代价是基函数个数随维度爆炸。表里 `poly` 赢，不是因为多项式"更强"，而是因为**在低维它便宜到近乎免费**；在 4 维它已经输了。

**读法四（收敛速度，不只是误差大小）**：1 维单目标，嵌套族（$m$ 特征类是 $240$ 特征类的子空间，所以曲线可比）：

```
  target                 |  class  |  smallest m with rel. L2 <= 1e-3
  ------------------------------------------------------------------
  smooth 1-D sin(2 pi x) |  poly  |  8
      curve: m=4:9.37e-02, m=8:2.21e-04, m=16:9.35e-12, m=32:8.63e-16, m=64:3.95e-15, m=120:4.06e-15, m=240:4.02e-15
  smooth 1-D sin(2 pi x) |  relu  |  >240
      curve: m=4:8.63e-01, m=8:5.93e-01, m=16:2.43e-01, m=32:1.62e-01, m=64:5.85e-02, m=120:2.89e-02, m=240:5.27e-03
  fast 1-D   sin(20 pi x) |  poly  |  64
      curve: m=4:9.90e-01, m=8:9.75e-01, m=16:9.16e-01, m=32:1.81e-01, m=64:3.92e-15, m=120:4.57e-15, m=240:3.00e-15
  fast 1-D   sin(20 pi x) |  relu  |  >240
      curve: m=4:9.97e-01, m=8:9.97e-01, m=16:9.93e-01, m=32:9.81e-01, m=64:7.68e-01, m=120:6.54e-01, m=240:3.18e-01
  kink 1-D   |x - 0.5|   |  poly  |  120
      curve: m=4:1.25e-01, m=8:3.91e-02, m=16:1.31e-02, m=32:4.51e-03, m=64:1.58e-03, m=120:6.11e-04, m=240:2.15e-04
  kink 1-D   |x - 0.5|   |  relu  |  >240
      curve: m=4:4.25e-01, m=8:1.97e-01, m=16:4.43e-02, m=32:3.10e-02, m=64:1.23e-02, m=120:6.86e-03, m=240:5.95e-03
  ------------------------------------------------------------------
```

- **频率的代价**：目标频率 $a$ 翻倍，光谱类需要的系数个数按 $O(a)$ 增长（$\sin(2\pi x)$ 要 8 个，$\sin(20\pi x)$ 要 64 个）；
- **折点的代价**：折点让光谱类的收敛从"几何级"退化成"代数量级"（$m$ 加倍误差只降 $0.387$ 倍，约 $m^{-1}$），而分段光滑的 ReLU 类在同样区间降得更快；
- **一个必须承认的偏差**：ReLU 类在 1 维其实可以**精确**表示 $|x-0.5|$（两个折线段就够），表里的 $6.86\times10^{-3}$ 是**随机抽样没把折点放在 0.5 附近**造成的，不是类的上限。随机特征版的误差下限约 $1/m$——这正是 §4.5 会撞到的那堵墙。

### 2.5 这个度量的两个盲区

1. **它没有价格**：$A$ 只说"离得多近"，不说"从 $n$ 个样本里找得到吗"。上表里 `poly` 的 $A$ 极小，但要在 $n=256$ 个样本上确定 119 个系数，还要看条件数与噪声放大（$I$，正文 `01` §3.4 / `04` §2）。
2. **它假设了全局优化**：投影是"无穷精度、无噪声、无限算力"下的最优。真实训练只能达到 $A+O(T)+I+F$。

---

## 3. 信息极限：$n$-width 与 $n^{-s/d}$

### 3.1 定义

$$
E_n(\mathcal F)\ :=\ \inf_{A}\ \sup_{f\in\mathcal F}\ \lVert f-A(X)f\rVert_{L^2}
$$

其中 $X$ 是 $n$ 个样本，$A$ 是任意算法（不限表达式类、不限优化器）。$E_n$ 是 Kolmogorov $n$-width：**$n$ 个样本的信息极限**。任何算法的误差都不可能系统性地小于它。

它的用处（也是本章反复要用的）：

- 如果实测误差远高于 $E_n$ ⇒ 问题不在"信息不够"，而在 $A$/$I$/$O(T)$；
- 如果实测误差接近 $E_n$ ⇒ 已经接近该信息量下的最优，换架构收益有限。

### 3.2 五行推导 $E_n\asymp n^{-s/d}$

设定：$f:[0,1]^d\to\mathbb R$，$f$ 在 Sobolev 球 $\mathcal H^s$ 内（$\mathcal H^s=\{\,\lvert f\rvert\le1,\ \sum_k\lVert k\rVert^{2s}\lvert c_k\rvert^2\le1\,\}$，其中 $c_k$ 是 Fourier 系数）。

1. **展开**：$f=\sum_{k\in\mathbb Z^{d}}c_k e^{2\pi i k\cdot x}$，$\lVert f\rVert^2=\sum_k\lvert c_k\rvert^2$（Parseval）。
2. **光滑度**：$f\in\mathcal H^s\iff\sum_k\lVert k\rVert^{2s}\lvert c_k\rvert^2\le1$。
3. **样本数定带宽**：$n$ 个样本最多定 $n$ 个系数，所以可分辨的带宽 $B$ 满足
   $$
   N(B):=\#\{k\in\mathbb Z^{d}:\lVert k\rVert\le B\}\ \asymp\ B^{d}\ \lesssim\ n
   \quad\Longrightarrow\quad B\ \lesssim\ n^{1/d}.
   $$
4. **尾部有多大**：Sobolev 约束直接给出 $\sum_{\lVert k\rVert>B}\lvert c_k\rvert^{2}\le (B+1)^{-2s}$（把 $\lvert c_k\rvert^2$ 换成 $\lvert c_k\rvert^2\lVert k\rVert^{2s}/\lVert k\rVert^{2s}$ 逐项比较）。
5. **合成**：最坏情形（把尾部能量全堆在第一壳层上）下
   $$
   E_n\asymp (B+1)^{-s}\asymp n^{-s/d}.
   $$

**标签**：**渐近**（$n\to\infty$，常数依赖 $s,d$；第 3 步的"$n$ 个样本定 $n$ 个系数"是上界方向的论证，严格版本是 Kolmogorov 宽度定理）。

> **一个必须说清的区分**：$n$-width 是"**对 $f$ 一无所知**"时的最坏情况。§2 的投影误差是"**知道 $f$**"时的最好情况。两者不矛盾：**$n$-width 是地板，投影误差是天花板**，真实可达误差在中间。v1.5 的题库里 $f$ 都是已知的解析式，所以实际离 $n$-width 很远——但这不意味着 $n$-width 没用：它告诉你"再加数据最多能好到什么程度"。

### 3.3 把 $B(n)$ 数清楚（闭式）

实操里需要的是 $B(n)=\max\{B:N(B)\le n\}$。壳层计数有闭式：

$$
\#\{k\in\mathbb Z^{d}:\lVert k\rVert_1=r\}=\sum_{j=1}^{\min(d,r)}\binom{d}{j}\,2^{j}\binom{r-1}{j-1},
$$

**推导**（启发式→精确计数）：恰好 $j$ 个坐标非零 ⇒ 选坐标 $\binom dj$ 种；每个非零坐标的符号 $2^{j}$ 种；$j$ 个正整数绝对值之和为 $r$ 的组合数是 $\binom{r-1}{j-1}$（隔板法）。三者相乘再对 $j$ 求和。

```python
import itertools, math
import numpy as np

def shell_closed(d, r):
    """#{k in Z^d : ||k||_1 = r} = sum_j C(d,j) 2^j C(r-1, j-1)."""
    if r == 0:
        return 1
    return sum(math.comb(d, j) * 2**j * math.comb(r - 1, j - 1)
               for j in range(1, min(d, r) + 1))

def shell_brute(d, r):
    return sum(1 for k in itertools.product(range(-r, r + 1), repeat=d)
               if sum(abs(v) for v in k) == r)

def N(B, d):
    """#{k in Z^d : ||k||_1 <= B}."""
    return sum(shell_closed(d, r) for r in range(B + 1))

print("closed-form shell count vs brute force:")
for d, r in ((1, 3), (2, 4), (3, 3), (2, 6)):
    a, b = shell_closed(d, r), shell_brute(d, r)
    print(f"  d = {d}, r = {r}: closed {a:5d} | brute {b:5d} | {'match' if a == b else 'MISMATCH'}")

ns = [16, 64, 256, 1024, 4096]
print()
print("  s   d | predicted -s/d | measured slope |  ratio | B(n) at n = 16..4096")
for s, d in [(2, 1), (2, 2), (2, 4), (4, 2), (1, 2)]:
    vals, bs = [], []
    for n in ns:
        B = 0
        while N(B + 1, d) <= n:
            B += 1
        bs.append(B)
        vals.append((B + 1) ** (-float(s)))
    xs = np.log(np.array(ns, dtype=float))
    ys = np.log(np.array(vals, dtype=float))
    slope = float(np.polyfit(xs, ys, 1)[0])
    print(f"  {s}   {d} |     {-s/d:+.4f}      |    {slope:+.4f}      |"
          f" {slope/(-s/d):.3f}  | {bs}")
```

```
closed-form shell count vs brute force:
  d = 1, r = 3: closed     2 | brute     2 | match
  d = 2, r = 4: closed    16 | brute    16 | match
  d = 3, r = 3: closed    38 | brute    38 | match
  d = 2, r = 6: closed    24 | brute    24 | match

  s   d | predicted -s/d | measured slope |  ratio | B(n) at n = 16..4096
  2   1 |     -2.0000      |    -2.0000      | 1.000  | [7, 31, 127, 511, 2047]
  2   2 |     -1.0000      |    -0.9752      | 0.975  | [2, 5, 10, 22, 44]
  2   4 |     -0.5000      |    -0.5340      | 1.068  | [1, 2, 3, 5, 8]
  4   2 |     -2.0000      |    -1.9505      | 0.975  | [2, 5, 10, 22, 44]
  1   2 |     -0.5000      |    -0.4876      | 0.975  | [2, 5, 10, 22, 44]
```

**读法**：实测斜率与 $-s/d$ 的比值在 $0.975$–$1.068$ 之间，偏差来自 $B(n)$ 取整（$B(n)$ 是阶梯函数，斜率拟合会抖）。

**这张表里最该记住的一行是 $s=2,d=4$**：斜率只有 $-0.53$，$n$ 翻 256 倍（16→4096）带宽才从 1 涨到 8。**这就是为什么"同样的 $n$、同样的 $s$，高维任务的误差下不去"**——不是模型不行，是格点上的自由度就不够。

### 3.4 核版本：$E_n=\sqrt{\lambda_{n+1}}$

对核方法（RKHS），信息极限有精确形式：

$$
E_n=\sqrt{\lambda_{n+1}},
$$

其中 $\lambda_1\ge\lambda_2\ge\dots$ 是核的 Mercer 特征值。含义：核在 $n$ 维之后的那条**特征方向**有多小，用 $n$ 个样本就只能把它分辨到那个程度。

- 高斯核的谱衰减**快于任何幂**，所以它的 $E_n$ 也快于任何幂（但常数依赖长度尺度）；
- 与 §3.2 的 $n^{-s/d}$ 是同一件事的两种语言：$n^{-s/d}$ 是"格点计数"给出的率，$\sqrt{\lambda_{n+1}}$ 是"同一个量的谱表达"。

数值（`S3`，1 维高斯核，长度尺度 0.15，256 个点）：

```
    n      lam_{n+1}       sqrt(lam_{n+1})
       8   2.5765e-01   5.0760e-01
      16   1.2520e-06   1.1189e-03
      32   5.2383e-15   7.2376e-08   <- numerical floor
      64   2.1868e-15   4.6764e-08   <- numerical floor
     128   3.3242e-16   1.8232e-08   <- numerical floor
     256   0.0000e+00   0.0000e+00   <- numerical floor
    (below the floor the eigenvalues are numerical noise, so the tail of this
     table says 'faster than any power', not a specific rate)
```

**读法**：从 $n=8$ 到 $n=16$，$E_n$ 掉了 $5.08\times10^{-1}\to1.12\times10^{-3}$——**两个数量级**，这正是"高斯核比任何多项式类都强"的数值证据。但从 $n=32$ 起就撞到**数值地板**（双精度下 256×256 核能分辨的方向就这么多了）。

> **这张表的诚实读法**：它证明了"$E_n$ 衰减很快"（前两行），但**没有**证明具体的衰减率（后面几行是浮点噪声，不是数学）。要讨论超多项式衰减，得用更大的点集或更稳定的特征分解——本章不做，留作开放问题。

### 3.5 什么时候 $n$-width 不是正确的地板

1. **目标已知且结构简单**：$E_n$ 是"对 $f$ 一无所知"的最坏情况；§2 的投影误差才是有意义的量。
2. **有限 $n$ 下真正绑定的是 $I$**：如果目标在**零空间**方向上有能量（$p>n$），那么 $I>0$ 且它常常比 $E_n$ 大得多（正文 `01` §3.4）。
3. **分类/CE 没有直接的 $L^2$ 版本**：CE 的下限是**熵地板** $H(Y\mid X)$，不是 $n$-width（正文 `06` §5）。

---

## 4. 深度与宽度：在真实族上把 $A$ 和 $O(T)$ 拆开

### 4.1 实验设计（以及为什么必须做两组预算）

- **族**：spiral（`q_04316e`）与 xor（`q_07d60b`），都在 dev 半边；
- **模型**：我们自己建的普通 ReLU MLP（无 LayerNorm、无残差），**只变** depth $\in\{1,2,3\}$ 与 width $\in\{16,64\}$；
- **协议**：各族自己的官方优化器/lr/步数/batch（spiral：Adagrad $3\times10^{-3}$、4096 步、bs 16；xor：SGD $3\times10^{-3}$、16384 步、bs 64），官方评测（test CE），2 个 seed 取均值；
- **第二组**：同一道题，预算改成 $1/4$ 和 $4$ 倍。

**为什么必须做第二组**：一个架构表现差有两种完全不同的原因，而**只测一个预算点分不出来**：

| 现象 | 真正的含义 | 换架构有用吗 |
|---|---|---|
| $A$ 太大 | 这个架构根本表达不了目标 | 有用（换表达式类） |
| $O(T)$ 太大 | 它能表达，但没走到 | 有用（换优化器/加预算/减深） |

所以"深好还是浅好"的结论会**随着预算翻转**——这不是理论自相矛盾，是被比较的两项变了。

### 4.2 spiral：深度赢，而且预算越大赢得越多

```
  spiral_classification (q_04316e, Adagrad lr=0.003, 4096 steps, bs=16), mean over 2 seeds
    depth |  width |   test CE
    ----------------------------------
        1 |     16 |  0.6830   (chance = 0.693)
        2 |     16 |  0.6830   (chance = 0.693)
        3 |     16 |  0.6754
        1 |     64 |  0.6848   (chance = 0.693)
        2 |     64 |  0.6620
        3 |     64 |  0.6035
    ----------------------------------
```

```
  spiral_classification (q_04316e)
    depth | width |  budget |   test CE
    ----------------------------------------
        1 |     64 |       1x    |  0.6848
        3 |     64 |       1x    |  0.6035
        1 |     64 |       4x    |  0.6771
        3 |     64 |       4x    |  0.5421
    ----------------------------------------
```

**读法（这是本章最重要的一组数）**：

- **浅网（depth 1）在螺旋上几乎不动**：预算翻 4 倍，test CE 只从 $0.6848$ 变到 $0.6771$——离随机水平 $0.693$ 也就差那么一点。**再多给预算也没用**，因为单隐层 ReLU 的决策边界是分段线性的，画不出螺旋（$A$ 绑定）。
- **深网（depth 3）真的在解这个结构**：$1\times$ 预算 $0.6035$，$4\times$ 预算 $0.5421$。
- **差距随预算拉大**（$0.0813\to0.1350$）⇒ 深网的收益是**结构性的**（$A$），不是"深网只是跑得快一点"。

### 4.3 xor：深度赢不了，预算越少越糟

```
  xor_classification (q_07d60b, SGD lr=0.003, 16384 steps, bs=64), mean over 2 seeds
    depth |  width |   test CE
    ----------------------------------
        1 |     16 |  0.2403
        2 |     16 |  0.2504
        3 |     16 |  0.6088
        1 |     64 |  0.2263
        2 |     64 |  0.1936
        3 |     64 |  0.2165
    ----------------------------------
```

```
  xor_classification (q_07d60b)
    depth | width |  budget |   test CE
    ----------------------------------------
        1 |     64 |    0.25x    |  0.6396
        3 |     64 |    0.25x    |  0.6891
        1 |     64 |       1x    |  0.2263
        3 |     64 |       1x    |  0.2165
        1 |     64 |       4x    |  0.1274
        3 |     64 |       4x    |  0.2819
    ----------------------------------------
```

**读法**：

- **¼ 预算**：所有深度都还在随机水平附近（$0.6396$ / $0.6891$，随机是 $\ln2=0.693$）；深网甚至**比随机还差**；
- **$1\times$ 预算**：三个深度基本打平（$0.2263$ / $0.2165$）；因为 xor 的边界是二阶交互，**一层宽网就能直接切出来**（$A$ 不绑定），剩下的全是 $O(T)$；
- **$4\times$ 预算**：浅网继续变好（$0.1274$），**深网反而变差**（$0.2819$）。深网在这一族上更慢（更深的核谱 ⇒ 更小的有效步长），$O(T)$ 绑定；
- **窄 + 深会炸**：width 16、depth 3 是 $0.6088$（接近随机），而 width 16、depth 1 是 $0.2403$。这是优化失稳的典型表现，不是表达能力的差别。

### 4.4 "翻车"机制，一句话

| 族 | 卡在哪一项 | 深度的净效应 |
|---|---|---|
| spiral | $A$（浅网表达不了弯曲边界） | **正**：且随预算增大而增大 |
| xor | $O(T)$（$A$ 已经够，优化没跟上） | **负**：加预算反而更差 |

同一个"深一点"的建议，在两个族上给出相反结果，**因为两族分别卡在四项分解的不同项上**。§1.2 的 $\Delta A-\Delta O(T)$ 就是这张表。

### 4.5 一个失败的设计，和它失败的原因（诚实记录）

这一章的第一版用的是**合成目标**：$\sin(3xy)$（复合）对 $\sin(3x)+\sin(3y)$（加性），配参数匹配的 1 层宽 66 与 2 层宽 14（$p=265$ 对 $267$）。想看的是"深度分离"。结果（`S2b`，2 seeds，3000 步，301×301 网格）：

```
  target   |  1 hidden layer  w=66  |  2 hidden layers w=14
  --------------------------------------------------------------------------
  comp sin(3xy)       |            3.741e-02  |            6.437e-02
  add  sin(3x)+sin(3y)|            1.543e-02  |            2.075e-02
  --------------------------------------------------------------------------

  the same numbers for the TRAINING points (did the net fit the data at all?)
  --------------------------------------------------------------------------
  comp sin(3xy)       |            3.030e-02  |            4.306e-02
  add  sin(3x)+sin(3y)|            1.818e-02  |            3.037e-02
  --------------------------------------------------------------------------

  and the RESOLUTION FLOOR of each width: the best-in-class relative L2 error
  of a random ReLU class with m = width features (the lazy limit of one layer)
  --------------------------------------------------------------------------
  comp sin(3xy)       |            3.782e-02  |            2.298e-01
  add  sin(3x)+sin(3y)|            2.299e-02  |            1.288e-01
  --------------------------------------------------------------------------

  trained grid error DIVIDED BY the 1-layer random-feature floor
  of the SAME width
  (1.0 = the trained net sits at the resolution limit of a 1-layer class of
   that width;  < 1 = it beat it, which is what depth composes for)
  --------------------------------------------------------------------------
  comp sin(3xy)       |                0.989  |                0.280
  add  sin(3x)+sin(3y)|                0.671  |                0.161
  --------------------------------------------------------------------------
```

**这个设计量不出深度分离，而且能量出它为什么量不出**：

- **没有分离**：宽的浅网在**两个**目标上都赢（复合目标 $3.74\times10^{-2}$ 对 $6.44\times10^{-2}$）。宽度效应压过了深度效应。
- **训练误差不是 0**（$1.8\times10^{-2}$–$4.3\times10^{-2}$）：因为 $m=66$ 个特征配 $n=1024$ 个点，**这个类连自己的训练数据都拟合不了**——误差被类本身钉住了。
- **比值行说明"钉住"的是什么**：宽网的**网格**误差与"同宽度 1 层随机特征类的逼近地板"之比是 $0.99$ 与 $0.67$——**训练收敛到了自己那个类的分辨率极限**；深网的比值是 $0.28$ 与 $0.16$，**远低于**同宽度的 1 层地板，因为**深度会复合**（2 层宽 14 能表达的函数比 1 层宽 14 多得多）。

> **这里有一个必须自己踩过才知道的坑**：**"地板"只对"产生它的那一类"是地板**。用 1 层随机特征类的地板去框 2 层网络是错的——深网能复合，所以它的真实地板比同宽度 1 层的低得多。§2 的类内逼近误差恰好是逐类算的，所以没有这个歧义。

**三条可迁移的教训**：

1. **分离是关于"目标"的定理，不是关于"深度"的定理**。$\sin(3xy)$ 在这个分辨率下**并不难**——宽 1 层 ReLU 的地板只有 $3.8\times10^{-2}$，所以"复合结构"根本没被逼出来。真正的深度分离结果需要**可证明落在浅层类之外的目标**（比如分段极多或刚性函数）。
2. **匹配参数量会把深度和宽度混在一起**。参数相等不等于"类相等"；要比结构，得比**类**。
3. **所以 §4.2–4.3 改用真实族**：那里真正不同的是**边界复杂度**（spiral 的 12 阶 vs xor 的 2 阶），而这恰好是"深/浅"该分胜负的地方。

### 4.6 这组实验的边界

- 只测了 depth/width 两个旋钮，没有测激活、归一化、残差、初始化；
- 用的是我们自己建的模型，不是 v1.5 的候选模型，所以绝对数值不能和题库的 `test_ce` 直接比，只能内部横比；
- 2 个 seed：族内差异（$0.68$ vs $0.60$、$0.13$ vs $0.28$）远大于种子噪声，但接近的格子（$0.2263$ vs $0.2165$）**不要**当成结论。

---

## 5. $A$ 的第二个度量：惰性区里的最小范数权重

### 5.1 记号

在初始化处把网络写成线性形式：

$$
f_\theta(x)=w^\top\varphi(x),\qquad \varphi(x)=\nabla_\theta f_\theta(x)\big|_{\theta=\theta_0}
$$

$\varphi$ 就是**初始化处的 Jacobian**（第 02 章 §7 的记号）。经验核矩阵：

$$
K=\frac1nJJ^\top\in\mathbb R^{n\times n},\qquad J_{i,:}=\varphi(x_i)^\top
$$

它的特征值 $\lambda_1\ge\dots\ge\lambda_n$ 就是 NTK 的特征值（差一个 $1/n$ 的归一化，约定与正文 `01` §5.5 一致：那里写的是 $\lambda_{\max}=34.03$、$206.2$）。

### 5.2 惰性极限的解，和它的代价

**惰性（lazy）假设**：训练中特征 $\varphi$ 基本不动，于是优化问题变成线性的：找 $w$ 使 $f_w$ 拟合 $y$。这个问题的解是**最小范数插值器**（正文 `02` §5 的结论）：

$$
\hat w=\Phi^{+}y,\qquad \Phi=\begin{bmatrix}\varphi(x_1)^\top\\ \vdots\\ \varphi(x_n)^\top\end{bmatrix}
$$

它的**代价**（也就是 §5.5 要用的量）：把 $K=V\Lambda V^\top$ 代入，

$$
\lVert\hat w\rVert^{2}=y^\top(JJ^\top)^{+}y=\sum_j\frac{b_j^{2}}{\lambda_j^{(0)}},\qquad b=V^\top y,\quad \lambda_j^{(0)}=n\lambda_j
$$

换成 $1/n$ 归一化的 $K$：

$$
\boxed{\ \lVert\hat w\rVert^{2}=\frac1n\sum_j\frac{b_j^{2}}{\lambda_j}\ }
$$

**读法**：$\lambda_j$ 越小的方向，$b_j\ne0$ 就越贵。**这正是 $A$ 在核语言里的样子**——"这个目标在这个架构里有多贵"，用**表示它所需的权重能量**来量，而不是用 $L^2$ 距离。（对照：RKHS 范数 $\lVert f\rVert_H^2=\sum_j c_j^2/\lambda_j$，是同一个式子。）

### 5.3 学习窗口：哪些方向动得了

梯度下降在方向 $j$ 上的振幅按 $(1-\eta\lambda_j)^t$ 衰减，所以 $T$ 步后能改变的方向满足

$$
\eta\lambda_jT\ \gtrsim\ 1 .
$$

对真题 `q_11b94d`（AdamW $\eta=3\times10^{-5}$、$T=256$ 步，故阈值 $1/(\eta T)=130.2$）：

```
  candidate A (c_6c95ed): d=4 w=256 p=264961
    init NTK: lambda_max 34.03 | numerical rank 214/256 in float64, 71/256 in float32 | cond(K) 3.8e+10
    learning window: lam >= 1/(lr*T) = 130.2  (0/256 directions are inside it)
  candidate C (c_d7ace4): d=5 w=192 p=187393
    init NTK: lambda_max 206.2 | numerical rank 206/256 in float64, 67/256 in float32 | cond(K) 6.09e+09
    learning window: lam >= 1/(lr*T) = 130.2  (1/256 directions are inside it)
```

**$0/256$ 与 $1/256$**：官方预算下，**几乎没有任何方向落在可学窗口里**。这就是正文 `01` 的结论"$O(T)$ 主导"的谱版本。

> **一个必须注意的算术细节**：NTK 矩阵在 float64 下数值秩 214/256，但**训练跑的是 float32**，秩只有 71/256。所以真正的核比理论上的小得多——这说明"训练实际看到的核"和"你算出来的核"可能不是一回事（正文 `01` §5.5 同样标注过这一点）。

### 5.4 平滑目标 vs 高频目标：惰性极限差了几个数量级

用同一套核，对两个目标算最小范数插值器（候选 A）：

```
    target smooth  tanh(2x)^3 + sin(2 pi x)   (variance of the target on the test points: 0.3926)
      min-norm weight norm of the interpolant  ||w_hat||^2 = 6.0096e+00
      energy of the target OUTSIDE the learning window: 100.00%
      lazy-limit (min-norm) interpolant: train MSE 2.6477e-05, test MSE 8.6610e-04
      the same two numbers, relative to the target's variance:
        lazy-limit relative test error                 2.2062e-03
        GP posterior variance of the same kernel        7.7220e-04   (ratio 2.857)
    target fast    sin(20 pi x)   (variance of the target on the test points: 0.4825)
      min-norm weight norm of the interpolant  ||w_hat||^2 = 5.5035e+03
      energy of the target OUTSIDE the learning window: 100.00%
      lazy-limit (min-norm) interpolant: train MSE 1.2809e-01, test MSE 4.5428e+00
      the same two numbers, relative to the target's variance:
        lazy-limit relative test error                 9.4142e+00
        GP posterior variance of the same kernel        7.7220e-04   (ratio 12191.338)
```

**读法**：

- **权重能量差了 916 倍**（$6.01$ 对 $5.50\times10^{3}$）：高频目标在同一个架构里**贵了三个数量级**；
- **平滑目标**：惰性极限的相对测试误差 $2.2\times10^{-3}$，很好；
- **高频目标**：惰性极限的测试 MSE 是 $4.54$，而目标自身的方差只有 $0.48$——**比直接预测零还差 9 倍**（相对误差 $9.41>1$）。

> **结论（这是本章最有用的一条）**：**在高频目标上，特征学习不是可选项。** 如果架构不改变特征（惰性），再大的权重也买不到高频；"这个任务需要更深的网络/更好的初始化"这句话的真正含义是"这个任务需要**动特征**"。

（候选 C 的同一组数：平滑目标 $\lVert\hat w\rVert^2=1.28$、窗口外能量 $54.23\%$、惰性相对误差 $1.62\times10^{-3}$；高频目标 $\lVert\hat w\rVert^2=1.30\times10^{3}$、窗口外 $99.40\%$、相对误差 $1.24$。在这两个候选之间，能看到的差别是**核谱**：C 的 $\lambda_{\max}=206.2$ 高于窗口阈值 $130.2$，所以窗口里多了**一个**方向；A 的 $34.03$ 一个都没有。两个候选的深度也不同（4 层对 5 层）、宽度也不同（256 对 192），所以**不能把差全算在深度头上**——只能说：在**谱**这个口径上，C 拿到了 A 没有的那一个可学方向。）

### 5.5 惰性极限的误差可以在训练前算出来（GP 恒等式）

**命题（严格定理，GP 设定）** 设目标 $f\sim\mathrm{GP}(0,k)$ 且观测带噪声 $\sigma^2$，训练数据 $y=f(X)+\varepsilon$。则后验均值与后验方差是

$$
\mu(x)=k_x^\top(K+\sigma^2I)^{-1}y,\qquad
\operatorname{var}(x)=k(x,x)-k_x^\top(K+\sigma^2I)^{-1}k_x
$$

且 $\mathbb E\big[\big(f(x)-\mu(x)\big)^2\big]=\operatorname{var}(x)$。

**证明**：条件高斯 $\Rightarrow$ 后验是高斯的，$f(x)\mid y\sim\mathcal N\big(\mu(x),\operatorname{var}(x)\big)$，故平方误差的期望等于后验方差。∎

**为什么值得验**：这条恒等式说明**核方法的惰性极限误差是可以在训练前算出来的**——但它是对**核自己的先验**成立的。对确定性目标（v1.5 全是确定性目标），后验方差只是一个**参照尺度**，两者的差正是"这个核不知道怎么表示的那部分目标"。

**数值核验**（`S4b`，从候选 C 的初始 NTK 里**抽 200 个 GP 目标**，用同一个核插值，再和解析后验方差比）：

```
  200 GP draws from the init NTK of candidate C (unit-diagonal kernel):
    analytic posterior variance at the test points : 1.076173e-03
    realised mean squared error of the interpolant   : 2.209590e-03
    ratio of the two MEANS (the tight check)         : 2.0532
    per-draw ratio: mean 2.0532, sd 1.3025
```

**怎么读这组数**：均值之比 $2.05$ 而不是 $1.00$，**不是**恒等式被推翻了，而是"在固定的 256 个测试点上求平均"这个统计量本身方差很大（逐次抽样的比值标准差 $1.30$）——因为光滑核的后验协方差几乎是秩 1 的，测试点高度相关，平均误差被一个公共模态支配。**恒等式是逐点的期望等式**，用均值比去检验只能给出数量级。

**一个必须记录的数值陷阱**：后验方差公式 $k(x,x)-k_x^\top(K+\sigma^2I)^{-1}k_x$ 是**两个几乎相等的数相减**。经验 NTK 在 float64 下秩只有 214/256、条件数 $10^{9}$–$10^{10}$，原始尺度下它会算出**负数**。修正办法（脚本里采用的）：把核归一化成对角为 1，再取 $\sigma^2=10^{-6}$，并把结果与"惰性极限误差 ÷ 目标方差"放在**同一个相对尺度**上比较。

### 5.6 这一节的边界

- 惰性假设**在真实训练里常常不成立**（正文 `01` §5.5：候选 A、C 的训练前后核余弦相似度只有 $0.896$、$0.775$，说明特征在动）。所以 §5.4 的数字是**基线**，不是预测；
- $A$ 与 $I$ 的分界线取决于"类"怎么定义（正文 `01` §3.6）：把类缩小，$A$ 变大约束变紧，$I$ 变小。**换架构同时动了两项**——这是本章反复出现的同一个机制。

---

## 6. 六个 v1.5 族的目标画像

### 6.1 回归族：类内逼近误差（相对 $L^2$）

`S5`，dev 半边，每族一道题：

```
  univar (q_07b3f8, d=1, n=256, metric=mse, x in [0.00, 1.00])
      poly deg 3  : best fit of the training points 6.769e-08 | best fit of the test points 5.948e-08
      poly deg 7  : best fit of the training points 6.723e-08 | best fit of the test points 5.888e-08
      relu 256    : best fit of the training points 8.113e-16 | best fit of the test points 6.742e-16
  mvar (q_0525ce, d=5, n=256, metric=mse, x in [0.00, 1.00])
      poly deg 3  : best fit of the training points 6.710e-02 | best fit of the test points 5.951e-02
      poly deg 7  : best fit of the training points 7.360e-16 | best fit of the test points 6.917e-16
      relu 256    : best fit of the training points 8.038e-16 | best fit of the test points 8.292e-16
```

**读法**：

- **univar**：3 次多项式就到 $6.8\times10^{-8}$，7 次没有再变好——**目标比 3 次多项式还简单**（$A$ 完全不绑定）；
- **mvar（$d=5$）**：3 次多项式只到 $6.7\times10^{-2}$，7 次才到 $7.4\times10^{-16}$。**同样的族，维度从 1 到 5，"便宜的表达"就没了**——这正是 §2.3 说的"固定预算下可负担次数随维度崩塌"，只不过这里的类可以自由选次数，所以只要**把预算（基函数个数）一起放大**，$A$ 仍然能压到 $10^{-16}$；
- **两族的 ReLU 类（256 个特征）都是 $10^{-16}$ 量级**：所以对回归族，$A$ 从来不是瓶颈，**瓶颈在别处**（$O(T)$、$I$、$F$）。

### 6.2 分类族：边界有多复杂

对分类族，"$A$"不是 $L^2$ 距离，而是**边界复杂度**。这里用一个便宜且诚实的代理：用多项式特征映射 + 岭回归拟合**标签**（岭参数由**训练集上的留一法**选出，**不用测试集**），逐步提高次数，看测试精度什么时候离开随机水平。

```
  spiral (q_04316e, d=2, n=1024, metric=ce, x in [-18.88, 18.87])
      label entropy 0.693 nats | class balance 0.500
      degree -> train acc / test acc / ridge: 1:0.504/0.513, 2:0.547/0.540, 3:0.521/0.526, 4:0.531/0.525, 6:0.585/0.560, 8:0.722/0.671, 12:0.951/0.955
      smallest polynomial degree that beats chance on the test set: 12
  tabular (q_036d72, d=16, n=1024, metric=ce, x in [-4.10, 3.67])
      label entropy 0.693 nats | class balance 0.492
      degree -> train acc / test acc / ridge: 1:0.807/0.760
      smallest polynomial degree that beats chance on the test set: 1
  xor (q_07d60b, d=16, n=1024, metric=ce, x in [-4.05, 4.35])
      label entropy 0.692 nats | class balance 0.521
      degree -> train acc / test acc / ridge: 1:0.521/0.512, 2:0.880/0.774
      smallest polynomial degree that beats chance on the test set: 2
```

**读法**：

- **spiral**：一次多项式完全是随机（$0.513$），要到 **12 次**才有 $0.955$ 的测试精度。**边界需要 12 阶多项式**——这就是 §4.2 里"深度是杠杆"的原因；
- **tabular**：**一次多项式就有 $0.760$**。它的边界近似线性/低阶，浅网就够（这也解释了它为什么不需要深网）；
- **xor**：一次多项式随机（$0.512$），**二次就跳到 $0.774$**。xor 是**二阶交互**——一层宽网能直接切，但一阶多项式不行。

> **注意 xor 与 spiral 的区别**：两者都需要"非线性"，但 xor 只要 2 阶、spiral 要 12 阶。**"非线性"不是二元的度量**——这正是"深好还是浅好"没有单一答案的根本原因。

### 6.3 画像表（先验判断 + 实测）

| 族 | 目标结构 | 先验判断（本章之前） | 实测（本章） | 主要瓶颈 |
|---|---|---|---|---|
| univar | 1 维光滑符号函数 | $A$ 很小 | 3 次多项式即 $6.8\times10^{-8}$ | $O(T)$ |
| mvar | 5 维光滑符号函数 | $A$ 小（以为"加性/低阶"） | **3 次不够**（$6.7\times10^{-2}$），要 7 次才到 $7.4\times10^{-16}$ | $O(T)$（$A$ 要更大预算才够） |
| tabular | 低阶/近线性边界 | $A$ 取决于边界复杂度 | **1 次多项式**即 $0.760$ 精度 | $I$/$O(T)$ |
| spiral | 弯曲边界 | 浅网 $A$ 大 | **12 次**多项式才 $0.955$；depth 3 显著优于 depth 1 | $A$（深度）$\times O(T)$ |
| xor | 二阶交互 | 2 层宽网 $A$ 已够 | **2 次**多项式即 $0.774$；深网在 $4\times$ 预算更差 | $O(T)$ |
| bigram | 固定随机 bigram law | 有熵地板 | （不在本章：目标是随机过程） | 熵地板之上的估计/优化 |

**这张表现在是"先验 + 实测"两列**：第三列是可以背下来的直觉，第四列是本机跑出来的数字。两者对不上的时候，**以第四列为准**。

> **本表最重要的一处对不上**：mvar 的先验是"加性/低阶交互，$A$ 小"，实测却是**3 次多项式完全不够**（$6.7\times10^{-2}$）。这条实测把先验改成了"$A$ 要更大的预算才够"——**"这一族是 5 维的"就足以改变结论**，这正是 §2.3 的维度诅咒在真实数据上的样子。

---

## 7. 怎么从题面读目标结构（可执行清单）

拿到一道题，按顺序问：

| 步 | 问题 | 影响哪一项 |
|---|---|---|
| 1 | 表达式/规则是什么（多项式？三角？绝对值？分段？螺旋？XOR？随机 bigram？） | 决定 §6.2 的"所需次数" |
| 2 | 名义维度 $d$ 是多少？有效维度呢？ | 固定预算下可负担的次数（§2.3） |
| 3 | 目标是**加性**、**低阶交互**、还是**高阶复合**？ | 深度能不能省（§2.4、§6.2） |
| 4 | 频率多高？有没有折点？ | 需要多少基函数（§2.4） |
| 5 | 训练点 $n$ 相对维度够不够？ | 信息极限（§3.3） |
| 6 | 标签确定吗？ | 有没有熵地板（bigram） |
| 7 | 指标是 MSE 还是 CE？ | CE 不能套回归的谱公式 |

然后估计：

- **$A$**：目标结构是否超出候选的表达类？（§2 的投影误差可以直接算）
- **$I$**：$n$ 个点是否足以定位目标？（核的零空间里有没有目标能量，正文 `01` §3.4）
- **$O$**：$T$/lr 是否足以走到？（学习窗口里有没有几个方向，§5.3）
- **$F$**：候选差是否超过噪声？（正文 `04` §5）

**顺序很重要**：先 $A$（能不能表示），再 $O(T)$（走到了没），最后 $I$（选对了没）和 $F$（测得准不准）。

---

## 8. 常见坑

| 坑 | 后果 |
|---|---|
| 把架构当容量旋钮（"更大更好"） | 忽略 $A$/$I$/$O(T)$ 联动，结论换个族就翻 |
| 只测一个预算点就下"深/浅"的结论 | 把 $O(T)$ 误读成 $A$（§4.1 的两组预算） |
| 用合成光滑目标证"深度分离" | 分离是关于目标的定理，不是关于深度的（§4.5） |
| 用 1 层随机特征类的地板去框深网 | 深网会复合，同宽度的 1 层地板不是它的地板（§4.5） |
| 匹配参数量当成"匹配了类" | 参数相等 ≠ 类相等；要比结构得比类（§4.5） |
| 把 $n$-width 当"实际可达误差" | 它是对 $f$ 一无所知的最坏情况地板（§3.5） |
| 用多项式类的误差说"多项式最强" | 低维便宜 ≠ 强；预算固定时高维就崩（§2.4） |
| 把随机 ReLU 类的误差当类上限 | 那是随机抽样的误差；类可以更好（§2.4 读法四） |
| 在原始尺度上算 GP 后验方差 | 灾难性抵消会算出**负数**（§5.5） |
| 用 float64 的核秩讨论训练 | 训练跑 float32，秩只有 71/256（§5.3） |
| 把惰性极限当成预测 | 真实训练里特征在动（§5.6） |
| 拿 2 个 seed 的小差别当结论 | 接近的格子不可比（§4.6） |
| 把分类族套回归的 $L^2$ 公式 | 边界复杂度是另一套度量（§6.2） |

---

## 9. 一页总结

- **架构不是容量旋钮**：它同时改 $A$（函数类）、$O(T)$（核谱与学习窗口）、$I$（隐式偏置）。净效应是 $\Delta A-\Delta O(T)$，所以同一个建议在不同族上会翻车（spiral 深赢、xor 深输，§4.2–4.4）。
- **$A$ 可以算**：固定类时 $A$ 是**正交投影**（严格定理），用 QR 一次算完，零优化、零随机性（§2.2）。
- **固定预算下，"对味的类"比"更大的类"重要**：加性目标偏爱加性类、复合目标抛弃加性类（§2.4）。
- **信息极限是 $n^{-s/d}$**：从 Fourier 展开 + Sobolev 球 + 格点计数推出来，指数可精确核验（§3.2–3.3）。$d=4$、$s=2$ 时斜率只有 $-0.53$——高维难在**自由度**，不在模型。
- **核语言下 $A$ 是"价格"**：$\lVert\hat w\rVert^2=\frac1n\sum_j b_j^2/\lambda_j$（§5.2）；高频目标的权重能量比平滑目标贵 916 倍，惰性极限的误差**比预测零还差 9 倍**——**特征学习在高频目标上不是可选项**（§5.4）。
- **惰性极限的误差可在训练前算**（GP 后验方差恒等式），但只对核自己的先验成立（§5.5）。
- **六个族的目标画像**（§6.3）：univar/mvar 的 $A$ 不绑定；tabular 一次多项式够；xor 二次够（$O(T)$ 绑定）；spiral 要 12 次（$A$ 绑定）。**"非线性"不是二元度量**。
- **动手指之前先问三句**：$A$ 够不够（§2）、$n$ 够不够（§3）、$T$ 够不够（§5.3）。

---

## 附：本章的脚本

```powershell
python "Math of AI Textbook\verification\architecture_approximation.py"          # 全量（约 15 分钟）
python "Math of AI Textbook\verification\architecture_approximation.py" --quick  # 冒烟版
```

输出：`verification/architecture_approximation_result.md`（人读）与 `.json`（机读）。
本脚本只读 v1.5 的数据集与候选目录，**只碰 dev 半边的题**（`md5(qid)%2==0`），不写 v1.5 树。
