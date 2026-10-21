# 过参数化拟合 + 有限步 GD + 验证集：a priori 数学理论综述

> 共享任务 `task-1` 交付底稿（v2，按 Lead 方向修正重写）｜ 作者：lit-survey ｜ 日期：2026-10-19
> 问题（纯数学版）：给定确定性目标函数 $f$、表达式类/参数化 $\mathcal G=\{g_\theta:\theta\in\mathbb R^p\}$、算法 $\mathcal A_T$（GD，步长 $\eta$，初始化 $\theta_0$，$T$ 步）、样本数 $n$ 与数据分布 $P$，**仅用 $f$ 与 $\mathcal G$ 的性质**，能证明/计算关于验证误差的什么结论？
> 约定：中文为主，英文术语保留；公式 LaTeX。arXiv 编号于 2026-10-19 经 `export.arxiv.org` API 逐条核对；无 arXiv 的经典文献标注期刊/专著。**本版不再做 v1.5 经验映射与系数拟合**；task-1 的十个主题在 §13 给出对应索引，仍全覆盖。
> 定理分级标签：**[E]** exact（有限量等式/精确定理）｜**[U]** upper bound｜**[L]** lower bound｜**[A]** asymptotic（$n,p,d\to\infty$ 极限定理）｜**[H]** heuristic/经验规律｜**[O]** open。

## 0. 形式化与"a priori 结论"的三个层次

**设定。** 数据 $S=\{(x_i,f(x_i))\}_{i=1}^n$，$x_i\stackrel{iid}{\sim}P$；**无标签噪声**（$y_i=f(x_i)$ 精确）。$\mathcal G$ 的参数化 $g_\theta$ 有 $p$ 个参数，$p>n$；$\theta_0$ 由给定初始化分布（或固定初值）抽取；$\mathcal A_T$：$\theta_{t+1}=\theta_t-\eta\nabla_\theta\frac1{2n}\sum_i(g_\theta(x_i)-f(x_i))^2$（SGD/动量/Adam 另注）。验证误差
$$
R_{\rm te}(g_T)=\mathbb E_{x\sim P}\big[(g_T(x)-f(x))^2\big],\qquad g_T=g_{\theta_T},
$$
是一个**随机变量**：它只对"数据抽样 + 初始化"随机，不对标签随机。

**"a priori" 的三个层次（本综述的核心组织原则）。**
1. **给定实现（data-dependent exact）**：已知具体 $S$ 与 $\theta_0$、$f$、$\eta,T$ 时，能写出 $R_{\rm te}$ 的**精确表达式**（不需要真的训练）。§1、§2、§3 的闭式属于这一层。
2. **分布层（expectation/高概率）**：只给 $P$、初始化分布、$f$ 的性质，对数据/初始化取期望或给高概率界（§4、§5、§6、§8）。这里才开始出现"a priori 判据"。
3. **类层（worst-case/最优性）**：对函数类 $\mathcal F$ 与 $\mathcal G$ 给 minimax 上/下界（§6、§7）。此时结论是**任何算法（含 GD）都不能突破**的信息下界，或某算法达到下界。

**两条"不要混淆"的界。** 训练误差 $R_{\rm tr}\to0$（插值）不提供 $R_{\rm te}$ 信息：$p>n$ 时插值随时可达；验证误差由 **A（approximation/aliasing）+ E（有限 $n$ 的采样）+ O（有限 $T$ 的优化）** 决定。**无噪声 ≠ 无 E**：精确标签只删去"标签噪声方差"项；有限 $n$ 个设计点造成的算子采样误差（empirical vs population operator）仍然存在并在 $n\to\infty$ 时消失——这一点决定了下面"无噪声版"与"统计学习版"的率不同（§6、§7）。

**本文件能给出的 exact 结论的完整清单**（其余 regime 只有界/渐近/开放）：线性/多项式过参数化回归的有限步 GD（§1）、核/随机特征的有限 $n$ 精确预测（§2，给定实现）、NTK/lazy 的无限宽闭式（§3）、深度线性网的精确动力学（§3），以及 §6 的 n-width/最优恢复精确定理与 §8 的谱-ridge 等价。

**四个必须在全文中保持区分的东西**：① 训练误差 vs 验证误差；② **模型类容量**（$\mathcal G$ 能逼近什么，决定 A 的 minimax 侧）vs **算法隐式偏置**（GD 在插值解中选哪一个，决定 $T\to\infty$ 的落点）；③ **kernel/lazy 区**（NTK 固定 ⇒ 精确公式可用）vs **feature-learning 区**（NTK 演化 ⇒ 只有界/渐近/开放问题）；④ 回归 MSE（有闭式谱分解）vs 分类 CE（无闭式；隐式偏置为 max-margin；可能含不可约熵）。

**记号表（后文统一使用）。**

| 记号 | 含义 |
|---|---|
| $f$ | 确定性目标函数；$y_i=f(x_i)$（无标签噪声） |
| $\mathcal G=\{g_\theta\}$ | 表达式类/参数化，$p$ 个可训练参数 |
| $P$ | 输入分布；验证误差对 $x\sim P$ 取期望 |
| $n,\ \gamma=p/n$ | 样本数、过参数化比 |
| $T_K$ | 核/特征积分算子：$T_K\phi_j=\lambda_j\phi_j$ |
| $a_j=\langle f,\phi_j\rangle$ | 目标的核谱系数（$f$ 与核已知即 a priori 可算） |
| $R,r$ | source 条件：$f=T_K^{r/2}g$，$\|g\|_{\mathcal H}\le R$ |
| $\beta$ | 容量条件指数：$\mathcal N(\lambda)\lesssim\lambda^{-\beta}$；幂律谱 $\lambda_j\asymp j^{-\alpha}$ 时 $\beta=1/\alpha$ |
| $\mathcal N(\lambda)$ | 有效维数 $\mathrm{tr}\big((T_K+\lambda)^{-1}T_K\big)$ |
| $K_{\rm tr},k_{\rm te}$ | 训练 Gram 矩阵 / 测试-训练核向量 |
| $J$ | 初始化处 Jacobian（NTK 特征矩阵），$K=JJ^\top$ |
| $\eta,T$ | GD 步长与步数；$\lambda_{\rm eff}\asymp1/(\eta nT)$ 为对应 ridge 参数 |
| $A/E/O/F$ | approximation / estimation / optimization / fluctuation 四类误差 |
| [E]/[U]/[L]/[A]/[H]/[O] | 定理分级：exact / upper / lower / asymptotic / heuristic / open |

（每个公式/定理前的方括号标签即其分级；无标签者为所在节的默认级。）

---

# 第一部分：可精确求解的 regime

## 1. [E] 线性/多项式过参数化回归：有限步 GD 的精确误差公式

**设定。** 特征映射 $\varphi:\mathbb R^d\to\mathbb R^p$（多项式、随机特征、固定核展开都属此类），$g_\theta(x)=\varphi(x)^\top\theta$，$f\in{\rm span}\{\varphi_1,\dots,\varphi_p\}$，即 $f=\varphi^\top\theta^*$（若 $f\notin{\rm span}$，先扣掉 A：$\inf_{\theta}\|f-\varphi^\top\theta\|_{L^2(P)}^2$）。令 $X\in\mathbb R^{n\times p}$ 为训练特征矩阵，$y=X\theta^*$，$H=\mathbb E_P[\varphi\varphi^\top]$。

**精确递推（对给定 $S,\theta_0$ 成立，无需训练即可算出 $T$ 步结果）。** 对 MSE 的 GD：
$$
\theta_T-\theta^*=(I-\eta X^\top X)^T(\theta_0-\theta^*),\qquad
R_{\rm te}(T)=(\theta_T-\theta^*)^\top H(\theta_T-\theta^*).
$$
$H=I$（标准正交特征/各向同性测试）时对角化：$R_{\rm te}(T)=\sum_{i=1}^{p}(1-\eta s_i^2)^{2T}(v_i^\top(\theta_0-\theta^*))^2$，其中 $v_i$ 为 $X^\top X\in\mathbb R^{p\times p}$ 的特征向量，$s_i^2$ 为其特征值（$s_i=0$ 补零到 $p$ 维）；稳定域 $0<\eta<2/s_{\max}^2$。**$p>n$ 的结构性结论**：$\dim\ker X^\top X\ge p-n$，这些方向上 $I-\eta X^\top X$ 的特征值为 1，$T$ 再大也不收缩，故
$$
R_{\rm te}(T)\ \ge\ R_{\rm alias}:=\|P_{\ker X^\top X}(\theta_0-\theta^*)\|_H^2,
$$
其中 $P$ 为欧氏正交投影，$\|v\|_H^2:=v^\top Hv$；$T\to\infty$ 时 $R_{\rm te}\to R_{\rm alias}$（$\theta_0=0$ 即最小 $\ell_2$ 范数插值的误差）。**这是"$p>n$、有限步、无噪声"下最干净的精确答案**：可训练部分按 $(1-\eta s_i^2)^T$ 收敛，零空间/aliasing 部分原样保留；$T$ 只影响前者。

**分布层的精确闭式（白化随机特征）。** 若特征满足 $\mathbb E_P[\varphi(x)\varphi(x)^\top]=I_p$（各向同性/白化；标准正态特征或白化核特征均可归一化到此）、$p>n$，对 $X$ 取期望（$T=\infty$、无噪声）：
$$
\mathbb E_X R_{\rm te}=\Big(1-\frac np\Big)\|\theta^*\|^2=\Big(1-\frac1\gamma\Big)\|\theta^*\|^2,\qquad\gamma=p/n .
$$
**推论**：$p>n$ 本身不给出泛化；未对齐时插值地板就是目标能量的 $1-1/\gamma$ 份。$T$ 有限时上式变为谱加权和（权重 $(1-\eta s_i^2)^{2T}$ 依赖 $X$ 的实现）；对 $X$ 取期望需要随机矩阵谱分布，属 §2 的比例渐近/RMT 范畴。多项式的 $A$ 可直接判定：若 $f$ 是 $\mathcal G$ 中次数 $\le k$ 的多项式，则 $A=0$，误差纯由 O（有限 $T$ 截断）与 aliasing 决定。

**代表文献**：Hastie, Montanari, Rosset, Tibshirani 2019, arXiv:1903.08560（ridgeless 插值与比例渐近）；Gunasekar, Lee, Soudry, Srebro 2018, arXiv:1806.00468（线性模型的隐式偏置=最小范数）；Belkin, Hsu, Xu 2020, arXiv:1903.07571（弱特征双下降的精确风险）；Soudry et al. 2018, arXiv:1710.10345（§9 会用）。

**在本题中的用途**：这是唯一"**每个候选的验证误差都能在训练前算出来**"的完整 regime；$R_{\rm alias}$ 给出与 $T$ 无关的地板，$T$ 依赖性完全由 $(I-\eta X^\top X)^T$ 给出。

## 2. [E→U/A] 核回归 / 随机特征：有限 $n$ 精确公式与谱速率

**有限 $n$、有限 $T$ 的精确预测（核梯度流）。** 设 $\mathcal H_K$ 为核 $K$ 的 RKHS（$K$ 可取 $\mathcal G$ 在初始化的 NTK，见 §3），$K_{\rm tr}=[K(x_i,x_j)]_{n\times n}$，$k_{\rm te}=[K(x_{\rm te},x_i)]$，$f_{\rm tr}=[f(x_i)]$。以损失 $\frac12\sum_i(g(x_i)-y_i)^2$ 做 GD 的核梯度流有闭式：
$$
g_T(x_{\rm te})=k_{\rm te}^\top K_{\rm tr}^{-1}\big[I-(I-\eta K_{\rm tr})^T\big]f_{\rm tr},
$$
$$
R_{\rm te}(T)=\mathbb E_{x_{\rm te}}\Big[\big(f(x_{\rm te})-g_T(x_{\rm te})\big)^2\Big]
$$
（把期望换成测试点平均即得"给定实现的精确验证误差"；$T\to\infty$ 收敛到最小 RKHS 范数插值 $k_{\rm te}^\top K_{\rm tr}^{-1}f_{\rm tr}$）。**这是可数值计算的精确等式**：$K_{\rm tr}$ 为 $n\times n$（本题 $n\le1024$），只需 $f$ 的样本值与核。

**总体谱展开（$n\to\infty$ 理想化；把"采样"换成算子）。** 设 $T_K$ 为 $K$ 的积分算子，$T_K\phi_j=\lambda_j\phi_j$，$a_j=\langle f,\phi_j\rangle$，则损失 $\frac1{2n}\sum(\cdot)^2$ 的核梯度流给出
$$
R_{\rm te}(T)=\sum_j(1-\eta n\lambda_j)^{2T}a_j^2 ,
$$
对每个固定 $j$ 精确（无限数据/总体算子）；有限 $n$ 的修正是经验算子 $T_{K,n}$ 与 $T_K$ 之差，量级由有效维数控制（下一段）。该式把"有限步"完全化为**谱滤波**：$g_T(\lambda)=1-(1-\eta n\lambda)^T$ 是低通滤波，$\lambda_j\gtrsim(\eta nT)^{-1}$ 的模式被学到、更小的模式在 $T$ 步后仍未被学到。

**a priori 速率（源条件 + 容量条件）。** 定义源指标 $r\ge0$：$f=T_K^{r/2}g$，$\|g\|_{\mathcal H}\le R$（等价 $a_j^2\le R^2\lambda_j^r$）；有效维数 $\mathcal N(\lambda)={\rm tr}\big((T_K+\lambda)^{-1}T_K\big)=\sum_j\lambda_j/(\lambda_j+\lambda)$，容量指标 $\beta$：$\mathcal N(\lambda)\lesssim\lambda^{-\beta}$。则
$$
\text{偏差}^2\lesssim\lambda^{r},\qquad \text{估计}^2\lesssim\frac{\mathcal N(\lambda)}{n}\lesssim\frac{\lambda^{-\beta}}{n},
$$
平衡 $\lambda_*\asymp n^{-1/(r+\beta)}$ 得**率 $n^{-r/(r+\beta)}$**。核为 $s$ 阶 Sobolev（$\lambda_j\asymp j^{-2s/d}$，$\beta=d/(2s)$）且 $f\in\mathcal H^s$（$r=1$）时化为经典 $n^{-2s/(2s+d)}$。**约定警告**：不同文献把源条件写成 $f=T^{r}g$（本式 $r$ 要乘 2），引用时以定义为准。$f\in\mathcal H_K$ 本身（$r=0$）只给出 $O(1)$ 的平凡界，**必须有 $r>0$ 才有速率**。

**供引用的精确形式（本稿约定）。**
$$
(\mathrm{S}_r)\quad f=T_K^{r/2}g,\ \|g\|_{\mathcal H}\le R\iff a_j^2\le R^2\lambda_j^{r};\qquad
(\mathrm{C}_\beta)\quad \mathcal N(\lambda)\le c\,\lambda^{-\beta},\ \beta\in[0,1];
$$
$$
\mathbb E\|g_{\lambda}-f\|^2\ \le\ C\Big(\lambda^{r}+\frac{\mathcal N(\lambda)}{n}\Big)\ \Longrightarrow\ \text{取 }\lambda_*\asymp n^{-1/(r+\beta)}\text{ 得率 }n^{-r/(r+\beta)} .
$$
$r$ 只依赖 $f$ 与核（对数斜率 $\log a_j^2$ vs $\log\lambda_j$ 的负半斜率），$\beta$ 只依赖核（$\log\mathcal N(\lambda)$ vs $\log\lambda$ 的斜率），二者都是 **a priori 可估量**；$\sigma^2=0$ 时上式仍成立（设计采样项不消失），只是常数不同（见 §14.5）。

**随机特征比例渐近 [A]。** $N$ 个随机特征、$N/n\to\gamma$ 时，测试误差有关于 $(\gamma,\lambda,\sigma^2,\text{特征协方差谱})$ 的精确渐近公式（双下降曲线）；这是"$p/n$ 常数区"能做**定量**预测的主要理论。$p/n\gg1$（本题 $\gamma\sim10^1$–$10^2$）时已远离插值峰，渐近曲线只剩"第二降段+地板"，且要求数据近似各向同性/高斯。AMP/state evolution、Gaussian equivalence、replica 方法都属这一档 **[A]**，只在该极限下成立（Montanari–Saeed 的 universality 说明只需均值/协方差；这只在比例极限内是定理）。

**代表文献**：Caponnetto & De Vito 2007（Constr. Approx.，无 arXiv，源条件+容量条件的最优率）；Yao, Rosasco, Caponnetto 2007（Constr. Approx.，无 arXiv，早停谱滤波）；Mei & Montanari 2019, arXiv:1908.05355（随机特征精确渐近）；Hastie et al. 2019, arXiv:1903.08560；Bordelon, Canatar, Pehlevan 2020, arXiv:2002.02561（谱学习曲线）。

**在本题中的用途**：有限 $n$ 精确公式（第一个框）是"用 $f$ 与 $\mathcal G$ 算出验证误差"的**唯一通用精确工具**（$\mathcal G$ 在 lazy 区时）；谱速率给出"$f$ 多光滑、核多快衰减 ⇒ 误差率"的 a priori 判据。

## 3. [E] NTK/lazy 与深度线性网络：$T$ 步的显式解

**NTK/lazy 区 [E]。** 若训练中 Jacobian 几乎不变（判据：$\|J(\theta_t)-J(\theta_0)\|\ll\|J(\theta_0)\|$），则网络输出在参数空间的线性化是精确的：$g_\theta(x)=g_{\theta_0}(x)+J_\theta(x)(\theta-\theta_0)$，于是整个问题化为 §1 的线性回归/§2 的核梯度流，**取 $\varphi(x)=J_{\theta_0}(x)$（NTK 特征）即得有限 $T$ 的精确闭式**：
$$
g_T(X_{\rm te})=g_0(X_{\rm te})+J_{\rm te}J_{\rm tr}^\top K_{\rm tr}^{-1}\big[I-(I-\eta K_{\rm tr})^T\big]\big(f(X_{\rm tr})-g_0(X_{\rm tr})\big),\quad K_{\rm tr}=J_{\rm tr}J_{\rm tr}^\top .
$$
无限宽极限下这是严格等式（NTK 收敛到确定性核且训练中不变）；有限宽 $w$ 是 $O(1/\sqrt w)$ 量级的近似（过参数化定理需要 $w\ge{\rm poly}(n)$）。**注意**：$p>n$ 只保证 $K_{\rm tr}$ 可逆（随机特征下几乎必然），不保证泛化；泛化由 $f$ 在 NTK 特征空间的对齐（A）与有限 $T$ 截断（O）决定。Adam 的逐参数预条件使"固定谱滤波"不成立，不在此列 **[O]**。

**深度线性网络的精确动力学 [E]。** $g=W_L\cdots W_1x$、白化数据、均方损失时，梯度流在每个奇异模式上解耦。$L=2$、平衡初始化 $a_k=u_kv_k$ 满足
$$
\dot a_k=2a_k(a_k^*-a_k)\ \Longrightarrow\ a_k(t)=\frac{a_k^*}{1+\big(\frac{a_k^*-a_k(0)}{a_k(0)}\big)e^{-2a_k^*t}},\qquad
\tau_k\approx\frac{1}{2a_k^*}\log\frac{a_k^*}{a_k(0)},
$$
即**小初始值 ⇒ 长 plateau 后突发下降**；更深网络是模式 $(D-1)$ 次幂的叠加，学习时间 $\tau_k$ 与初始奇异值成反比。收敛到全局最优的充分条件（近似平衡初始化、初始损失小于任意秩亏解）见 Arora et al. 2019。**这是"特征学习但仍可精确求解"的少数 regime**：非线性来自参数乘积，但模式解耦给出 $T$ 步闭式。

**lazy vs rich 判据（何时 §2/§3 的公式失效）。** 特征变化的时间尺度与输出变化的时间尺度之比由初始化尺度、lr、宽度决定：小初始化/大 lr/μP 参数化 → rich（特征学习），NTK 演化，§2 公式只近似；大初始化/NTK 参数化 → lazy。μP 下无限宽极限保持特征学习（Yang–Hu），此时**不存在与 $T$ 无关的固定核**，故闭式预测失效 **[O]**。

**lazy 判据的定量形式（可检验）[U]。** 线性化误差的二阶项为 $\frac12(\theta-\theta_0)^\top\nabla_\theta^2g(\theta_0)(\theta-\theta_0)$；lazy 区要求它远小于一阶项。等价判据：存在 $\varepsilon\ll1$ 使
$$
\sup_{t\le T}\frac{\|J(\theta_t)-J(\theta_0)\|_{\rm op}}{\|J(\theta_0)\|_{\rm op}}\le\varepsilon,
$$
此时网络输出与线性化模型的差为 $O(\varepsilon\cdot\text{输出变化})$。有限宽 NTK 定理给出充分条件：$w\ge{\rm poly}(n,T)$ 时 $\varepsilon\ll1$（Du et al. 2019；Allen-Zhu et al. 2019）。$\varepsilon\gtrsim1$ 即进入 rich 区，§2/§3 的闭式退化为近似，A 必须用特征学习后的核重估。

**代表文献**：Jacot, Gabriel, Hongler 2018, arXiv:1806.07572；Saxe, McClelland, Ganguli 2014, arXiv:1312.6120；Arora, Cohen, Golowich, Hu 2019, arXiv:1810.02281（深度线性 GD 收敛）；Chizat, Oyallon, Bach 2019, arXiv:1812.07956（lazy 判据）；Yang & Hu 2021, arXiv:2011.14522（μP/特征学习）。

**在本题中的用途**：lazy 时验证误差=可由初始化 Jacobian 精确算出的量（§2 第一框）；深度线性给出 plateau 时间尺度，是唯一能解析回答"$T$ 步后学没学到某模式"的非线性（乘积）模型。

---

# 第二部分：a priori 判据（只用 $f$ 与 $\mathcal G$ 能算什么）

## 4. [U] 目标侧的 a priori 量：范数、光滑度、谱衰减、source condition

**要算的四个量（都只依赖 $f$ 与所选核/特征空间 $\mathcal H$）。**
1. **RKHS 范数** $R=\|f\|_{\mathcal H}$：进入所有界（近似/估计的常数），且 $f\in\mathcal H$ 是"核方法能逼近 $f$"的前提。
2. **source 指标** $r(f)=\sup\{r:f\in T_K^{r/2}\mathcal H\}$：控制谱 tail 的衰减速度；$f$ 越光滑（相对核），$r$ 越大、可达率越快。等价地 $a_j^2\lesssim j^{-\alpha r}$（$\lambda_j\asymp j^{-\alpha}$）。
3. **光滑度/函数空间**：$f\in W^{s,2}$ 或 Besov $B^s_{2,2}$，与核的阶 $s_K$ 比较：$s<s_K$（欠光滑）率由 $f$ 定；$s\ge s_K$（超光滑）率由核的谱衰减定。Barron 范数 $\|f\|_{\mathcal B}$ 给出**无维度诅咒**的 $\sqrt m$ 近似率（对加性/低阶交互目标）。
4. **对齐（alignment）**：$\rho_k=\sum_{j\le k}a_j^2/\|f\|^2$ 与核的谱耦合；或 $\kappa(f,K)=\sum_j a_j^2\lambda_j^{\,?}$。直观：$f$ 的能量集中在核的前几个特征函数上 ⇒ 小 $n$ 即可学好（A 小）；能量落在 tail ⇒ 大 $n$ 也差。

**这些量如何进入验证误差（无噪声版）。** 在核/懒区，$R_{\rm te}(T)\le$ 三项之和：A（$f$ 在 $\mathcal H$ 外的分量 + $\mathcal H$ 内 tail）、E（经验算子 vs 总体算子的设计采样）、O（谱滤波截断）。$R$、$r$、$s$ 决定 A 与 E 的率；$r$ 越大、$s$ 越光滑，越接近低秩/参数化 regime。**关键判据**：$r>0$ 才有速率；$f\in\mathcal H$ 但 $r=0$（"仅平方可积"）时最坏情况率退化到 $O(1/\sqrt n)$ 甚至更差（取决于谱）。

**无噪声目标的一个特殊结构**：labels 精确 ⇒ 不存在噪声方差项，但 **$n$ 个设计点的采样（E）仍在**；在 worst-case over 函数类意义下，"无噪声数据"可用的信息下界是 §6 的 n-width/最优恢复（率 $n^{-s/d}$），而不是统计 minimax 的 $n^{-2s/(2s+d)}$。**做 a priori 结论时必须声明用哪一个基准。**

**a priori 上界链（给定 $f,\mathcal G,n,T,\eta$ 时的计算流程）[U]。**
1. 从 $f$ 的谱系数 $\{a_j\}$ 与核谱 $\{\lambda_j\}$ 估 $r$（对数斜率）；从核谱估 $\beta$；
2. 代入 §2：$\lambda_*\asymp n^{-1/(r+\beta)}$、率 $n^{-r/(r+\beta)}$、$T_*\asymp(\eta n)^{-1}n^{1/(r+\beta)}$；
3. 用 $\sum_{j>J}a_j^2$（$J$ 为 $T$ 步内可学的模式数，$\lambda_J\approx(\eta nT)^{-1}$）与 §1 的 $R_{\rm alias}$ 给有限-$n,T$ 的加性修正；
4. 若 $r$ 不可估，退到 §6 的 n-width 地板（最保守，且是无条件下界）。
这条链只用 $f$、核、$n,T,\eta$，不需要训练；第 3 步的常数依赖数据具体实现，属于"给定实现"层。

**代表文献**：Natterer 1977（Numer. Math.，无 arXiv，source condition 的起源）；Engl, Hanke, Neubauer 1996（书）；Caponnetto & De Vito 2007；Fischer & Steinwart 2020, arXiv:1702.07254（Sobolev 范数学习率）；Barron 1993（IEEE Trans. IT，无 arXiv）；Bach 2017, arXiv:1412.8690；E, Ma, Wu 2021, arXiv:1906.08039（Barron 空间）。

**在本题中的用途**：把"目标 $f$ 有多难学"压缩成少数 a priori 标量（$R,r,s$ 或 Barron 范数）；这些量可由 $f$ 的表达式与 $\mathcal G$ 的核直接估计，不需要训练、不需要样本实现。

## 5. [U/L] 表达式侧的 a priori 量：核谱、有效秩、条件数、逼近能力

**两个不同的"容量"概念，不要混。**
- **算法实际使用的函数空间**：$\mathcal G$ 在初始化处的 NTK 的 RKHS $\mathcal H_{K_0}$（lazy 区），或 §1 中特征 $\varphi$ 的线性 span。它的谱 $\{\lambda_j(K_0)\}$ 决定 A/E/O 的率，$K_0$ 由 $\theta_0$ 与结构/宽度决定。
- **模型类的逼近能力**（与算法无关）：$\mathcal G$ 作为 $p$ 参数函数族能逼近 $f$ 到多好——由 Barron 型范数、Sobolev 型逼近率 $m^{-s/d}$、或**非线性 n-width** 刻画（$\mathcal G$ 的 n-width、以及"神经网络是否优于线性方法"的定理）。$p>n$ 保证的是**在 $n$ 个点上的插值**（$\dim\mathcal G\ge n$ 时的有限样本表达力，Zhang et al. 2017），不保证 $L^2(P)$ 逼近。

**可算的谱量。** 给定 $K_0$（$n\times n$ 或总体谱）：有效秩 $\mathcal N_{\rm eff}={\rm tr}K_0/\|K_0\|_{\rm op}$ 或 ${\rm tr}(K_0)^2/{\rm tr}(K_0^2)$；条件数 $\kappa=\lambda_{\max}/\lambda_{\min}$（控制 GD 的 $T$ 需求与稳定性，见 §10）；power function $p_n(x)^2=K(x,x)-k(x)^\top K_{\rm tr}^{-1}k(x)$（逐点最优恢复误差，见 §6）；特征值衰减指数 $\alpha$（$\lambda_j\asymp j^{-\alpha}$，$\beta=1/\alpha$）。

**逼近侧的判据。** 若 $f$ 属于 $\mathcal G$ 的 Barron 空间，$\|f\|_{\mathcal B}<\infty$ ⇒ $m$ 个隐单元近似误差 $\lesssim\|f\|_{\mathcal B}/\sqrt m$，且统计/Rademacher 复杂度 $\lesssim\|f\|_{\mathcal B}/\sqrt n$，总 excess $\lesssim m^{-1/2}+n^{-1/2}$（维度无关）；对 Sobolev $W^{s,2}$ 目标，深度 ReLU 网络参数 $N$ 的近似误差 $\asymp N^{-s/d}$（Yarotsky 型），这就是维度诅咒的形式化；ReLU 网络在**非线性 n-width**意义下可超越经典线性方法（Daubechies et al. 2022）。若 $f$ 有加性/低阶交互/低维投影结构，Barron 范数不随 $d$ 爆炸，逼近侧判据给出"小 $m$ 就够"。

**三类常见 $\mathcal G$ 的 a priori 指纹（同一张表里的两种容量）。**

| $\mathcal G$ | 算法函数空间（lazy 核） | 谱 / 有效秩 / 条件数 | 模型类逼近能力（A 的天花板） |
|---|---|---|---|
| 线性/多项式特征 | 特征 Gram $H=\mathbb E[\varphi\varphi^\top]$ 的 span（$p$ 维） | $\lambda_j(H)$；$p$ 项后为 0 | $f$ 为次数 $\le k$ 的多项式则 A=0；否则 A 为该多项式族的 tail |
| 随机特征/两层宽网 | 随机特征核 / NTK | 谱由激活、宽度、初始化决定；宽 $w$ 时以 $O(1/\sqrt w)$ 收敛到极限核 | Barron：$m^{-1/2}$；Sobolev：$N^{-s/d}$ |
| 深网（lazy，固定深度） | 极限 NTK | 谱衰减与深度/激活有关 | 深度可提升逼近率（Yarotsky；Schmidt-Hieber），但 lazy 训练只能到达 NTK 空间 |

关键 a priori 判断：若 $\mathcal G$ 的逼近能力（第三列）足够、但 NTK 谱（第二列）与 $f$ 错配，则验证误差的地板是"**NTK 空间的 A**"而不是"模型类最优 A"——**lazy 训练把模型类容量浪费在了错误的函数空间上**。

**代表文献**：Barron 1993；Yarotsky 2017, arXiv:1610.01145；Daubechies, DeVore, Foucart, Hanin, Petrova 2022, arXiv:1905.02199（非线性逼近与 n-width）；Schmidt-Hieber 2020, arXiv:1708.06633（深度网络的 minimax 率）；Pinkus 1985（n-Widths in Approximation Theory，书）；Kanagawa et al. 2018, arXiv:1807.02582（power function/GP 等价）。

**在本题中的用途**：$\mathcal G$ 的两个 a priori 指纹（NTK 谱 + 逼近能力）分别对应"给定算法能走多快/多稳"（O、E）与"模型类天花板"（A）；两者都可从初始化与结构推出，不需要训练。

## 6. [L] 信息下界：n-width、optimal recovery、minimax rates

**为什么需要下界。** 只用 $f$ 与 $\mathcal G$ 的上界（§4、§5）无法判断"是否已最优"。信息下界给出**任何算法（含 GD、Adam、任意 $T$）都不能突破**的验证误差，从而把 A 与"算法 excess（E+O）"分开。

**n-width（Kolmogorov width）。** 对函数类 $\mathcal F$ 与范数 $\|\cdot\|$：
$$
d_n(\mathcal F)=\inf_{\substack{V\subset L^2(P)\\{\rm dim}V=n}}\ \sup_{f\in\mathcal F}\ \inf_{g\in V}\|f-g\| .
$$
- RKHS 单位球 $B_K$：$d_n(B_K,L^2(P))=\sqrt{\lambda_{n+1}}$（精确；最优子空间=前 $n$ 个特征函数）。
- Sobolev 球 $W^{s,2}$：$d_n\asymp n^{-s/d}$（Kolmogorov 1936；Pinkus 1985）。
- 神经网络的非线性 n-width：ReLU 网络类在特定类上可达到优于线性方法的 width（Daubechies et al. 2022）；对一般 Sobolev 类，深度网络逼近率仍是 $N^{-s/d}$，但 n-width 的最优性在部分类上成立。

**optimal recovery（逐点精确，[E]）。** 给定 $n$ 个点值 $f(x_1),\dots,f(x_n)$，在 $\|f\|_{\mathcal H}\le R$ 下对 $f(x)$ 的**worst-case 最优恢复误差**由 power function 精确给出：
$$
\inf_{\hat f}\ \sup_{\|f\|_{\mathcal H}\le R}\ |f(x)-\hat f(x)|=R\sqrt{K(x,x)-k(x)^\top K_{\rm tr}^{-1}k(x)},
$$
且最小范数插值（=GP 后验均值）达到它；这与 Gelfand $n$-width 相等（对 Hilbert 空间中的椭球，Kolmogorov = Gelfand = $\sqrt{\lambda_{n+1}}$）。**因此：只给 $n$ 个点，任何恢复算法在 worst-case 意义下的误差都不能低于此；$\mathcal G$ 的"插值能力"不改变这个信息限制。**

**统计 minimax（有噪声）[L]。** 若 $y=f(x)+\xi$，$\mathbb E\xi^2=\sigma^2$，则 Fano+metric entropy（Yang–Barron 1999）给出：$\log N(\delta)\asymp n\delta^2$ 的临界半径 $\delta_*$ 满足 $R^*\gtrsim\delta_*^2$；Sobolev 类 $\Rightarrow n^{-2s/(2s+d)}$（Stone 1982）；RKHS 源条件类 $\Rightarrow n^{-r/(r+\beta)}$（与 §2 上界同阶，故这些上界是 minimax 最优的）。**无噪声情形**的对应下界是 optimal recovery/n-width（$n^{-s/d}$ 量级），严格强于统计率；在本题（确定性 $f$、精确标签）中，**用 n-width/最优恢复作为基准才是正确的**。

**关键下界的推导骨架（正式文档可直接引用）。**
- **RKHS 椭球 n-width**：$B_K=\{f=\sum_j a_j\phi_j:\sum_j a_j^2/\lambda_j\le1\}$；取 $V_n={\rm span}\{\phi_1,\dots,\phi_n\}$ 得上界 $\sup_{B_K}\sum_{j>n}a_j^2=\lambda_{n+1}$，下界用正交补上的椭球宽度估计（Pinkus 1985）⇒ $d_n=\sqrt{\lambda_{n+1}}$。
- **Sobolev**：$\lambda_j\asymp j^{-2s/d}$ ⇒ $d_n\asymp n^{-s/d}$；统计（有噪声）率由 Fano：packing 数 $\log M(\delta)\asymp\delta^{-d/s}$，令 $\log M(\delta)\asymp n\delta^2$ 得 $\delta\asymp n^{-s/(2s+d)}$，风险 $\asymp\delta^2$。
- **optimal recovery**：最小范数插值 $\hat f=\sum_i\alpha_iK(x_i,\cdot)$，$\alpha=K_{\rm tr}^{-1}f_{\rm tr}$；对任意 $x$，
$$
f(x)-\hat f(x)=\big\langle f,\ K(x,\cdot)-k(x)^\top K_{\rm tr}^{-1}k(\cdot)\big\rangle_{\mathcal H},
$$
Cauchy–Schwarz 给出上界 $\|f\|_{\mathcal H}\,p_n(x)$，且取 $f$ 为该表示元的倍数时取等 ⇒ 逐点 worst-case 最优误差恰为 power function（Micchelli–Rivlin 1977；Wendland 2005）。
- **等价的 Gelfand 宽度**：$n$ 个线性观测下的最优恢复 worst-case 误差 = Gelfand $n$-width；对 Hilbert 椭球有 Kolmogorov = Gelfand = $\sqrt{\lambda_{n+1}}$，故"信息够不够"与"子空间维数够不够"在此情形同一。

**代表文献**：Kolmogorov 1936（Ann. of Math.）；Pinkus 1985（书）；Micchelli & Rivlin 1977（Optimal Recovery 综述）；Melkman & Micchelli 1979（SIAM J. Numer. Anal. 16:87，Hilbert 空间最优估计）；Novak 1988（书，确定性误差界）；Traub, Wasilkowski, Woźniakowski 1988（书，Information-Based Complexity）；Donoho 1994（Ann. Statist. 22:238，统计估计与最优恢复）；Yang & Barron 1999（Ann. Statist.，熵与 minimax 率）；Wendland 2005（书，power function/最优恢复）。

**在本题中的用途**：给出"给定 $n$、$f$ 的类、$\mathcal G$ 的逼近能力，验证误差至少多大"的**严格地板**；若某个候选/算法的误差远超这个地板，超出部分是算法 excess（E+O），可继续用 §2/§3/§10 分析；若接近地板，则该候选已信息论最优。

## 7. [E/U] 对齐、容量条件与可达最优性

**两条件一对齐。** 上界（§2/§4）与下界（§6）同阶的条件：① source 条件（$r$ 足够大）；② 容量条件（$\mathcal N(\lambda)\lesssim\lambda^{-\beta}$）；③ 对齐（$f$ 的能量分布与核谱匹配，$a_j^2\approx\lambda_j^r$ 型）。满足时 **KRR/早停 GD 是 minimax 最优的**（率 $n^{-r/(r+\beta)}$）；$f$ 比源条件更光滑（$r$ 更大）时可达更快率；$f$ 的能量集中在核 tail（错配）时，误差率由 tail 决定，且**任何算法**都受 §6 下界限制。

**"精确公式 vs 下界"的差 = 算法 excess。** 在有限 $n,T$ 下，§2/§3 的精确公式值总可以分解为
$$
R_{\rm te}^{\rm exact}(T)=\underbrace{R_{\rm opt.recovery}(f;\mathcal G,n)}_{\text{A + 信息下界项}}+\underbrace{\Delta_{\rm alg}(T)}_{\text{E}_{\rm design}+\text{O}_{\rm finite\ }T+\text{隐式偏置}} ,
$$
第一项由 $f$ 与 $\mathcal G$ 决定（§4–§6），第二项由算法/初始化/步长/有限步决定（§8–§10）。这个分解不是正交分解（有交叉项），但在 kernel/lazy 区可以做到精确（§2 公式直接给出两者之和）。

**错定（misspecified）情形。** $f\notin\mathcal H_K$ 时，$A=\|(I-\Pi_{\mathcal H_K})f\|^2>0$ 是不可约的；此时增大 $T$ 只能把 E+O 压小，验证误差的地板是 A。$\mathcal G$ 的逼近能力（§5 第二个概念）决定 A 的率。

**代表文献**：Caponnetto & De Vito 2007；Steinwart & Christmann 2008（书，source+capacity 条件的统一处理）；Blanchard & Mücke 2018, arXiv:1604.04054（一般谱正则化的 minimax 最优性）；Fischer & Steinwart 2020, arXiv:1702.07254。

**在本题中的用途**：这是把"验证误差"拆成"$f,\mathcal G$ 决定的不可避免部分"与"算法部分"的严格框架；是否可达最优由 $r,\beta$ 与对齐共同决定，全部可 a priori 估计。

---

# 第三部分：有限步、隐式偏置与验证集界

## 8. [E/U] Early stopping = spectral filtering / ridge 路径，与稳定性界

**精确等价（谱滤波）。** 对核/线性模型，$T$ 步 GD 的解 = 用滤波 $g_T(\lambda)=1-(1-\eta n\lambda)^T$ 做的谱正则化；小 $\eta$ 时 $g_T(\lambda)\approx1-e^{-\eta n\lambda T}$，即**有效 ridge 参数**
$$
\lambda_{\rm eff}\ \asymp\ \frac{1}{\eta nT}
$$
（$T$ 越大 ⇔ ridge 越弱）。因此"有限步"与"显式正则化"在核/lazy 区是同一回事：$T\leftrightarrow\lambda$ 的单调对应；$\eta,T$ 只通过乘积 $\eta T$ 进入（一阶），这解释了为什么 $lr$ 与 steps 在短预算下常呈"有效步长"合并现象。**该等价对 momentum 仍成立（谱多项式）；对 Adam 不成立 [O]。**

**两种滤波的对照（为什么说"同族"）。** ridge 的滤波为 $g_\lambda(\mu)=\mu/(\mu+\lambda)$，GD 的为 $g_T(\mu)=1-(1-\eta n\mu)^T\approx1-e^{-\mu/\lambda_{\rm eff}}$。二者在 $\mu\gg\lambda_{\rm eff}$ 时都 $\to1$、在 $\mu\ll\lambda_{\rm eff}$ 时都 $\approx\mu/\lambda_{\rm eff}$：
$$
\frac{\mu}{\mu+\lambda_{\rm eff}}\ \approx\ 1-e^{-\mu/\lambda_{\rm eff}}\quad(\mu/\lambda_{\rm eff}\text{ 远离 }1\text{ 时}),\qquad \lambda_{\rm eff}=\frac{1}{\eta nT},
$$
故**率相同、常数不同**；$T$ 与 $\lambda$ 的对应是"同尺度"的定性对应，不是逐点恒等（指数滤波与 ridge 滤波只在两端的渐近上一致）。

**优化/早停速率（[U]，与 §2 的 $\lambda_*\asymp n^{-1/(r+\beta)}$ 对应）。** 取 $\lambda_{\rm eff}\asymp n^{-1/(r+\beta)}$，即
$$
T_*\ \asymp\ (\eta n)^{-1}\,n^{1/(r+\beta)},
$$
可达率 $n^{-r/(r+\beta)}$；Sobolev 情形（$r=1,\ \beta=d/(2s)$）为 $T_*\asymp(\eta n)^{-1}n^{2s/(2s+d)}$、率 $n^{-2s/(2s+d)}$。**这正是 early stopping 作为正则化的经典结论**：停得越早 ≈ 正则越强；存在与 $n$ 匹配的最优 $T_*$。$T$ 过小 ⇒ bias 大（谱 tail 未学到）；$T$ 过大 ⇒ 趋向最小范数插值，误差趋于 §6 的 aliasing/optimal-recovery 地板（**无噪声时不存在"过拟合噪声"**，故 $T$ 的取舍完全由 A/E/O 的相对大小决定，而不是由噪声方差决定）。

**验证集界（稳定性，[U]）。** 对 $L$-Lipschitz、$L$-光滑损失的 SGD，Hardt–Recht–Singer 的一致稳定性给出
$$
\mathbb E\big[R_{\rm te}(g_T)-\hat R_n(g_T)\big]\ \lesssim\ \beta_{\rm stab}\ \lesssim\ \frac{L^2\eta T}{n}\quad(\text{凸情形；非凸情形为 }(\eta T)\text{ 的幂}) ,
$$
即"验证误差 − 训练误差"的上界随 $T$ 增大而增大、随 $n$ 增大而减小。在 $p>n$、$R_{\rm tr}\approx0$ 时它直接给出 $R_{\rm te}$ 的界（但常数可能很大，且不含 A/O 的精细结构）。PAC-Bayes/压缩界给另一族 **[U]**（需后验/压缩描述长度，通常要训练后信息，不属纯 a priori）。

**代表文献**：Yao, Rosasco, Caponnetto 2007；Raskutti, Wainwright, Yu 2014, arXiv:1306.3574（数据相关早停规则，达到 minimax 率）；Blanchard & Mücke 2018, arXiv:1604.04054；Bousquet & Elisseeff 2002（JMLR，无 arXiv）；Hardt, Recht, Singer 2016, arXiv:1509.01240。

**在本题中的用途**：把"$T$ 步"与"$\lambda_{\rm eff}$"严格对应，从而用 §2/§6 的 a priori 率直接预测"给定 $f,\mathcal G,n,\eta$ 时，多大 $T$ 达到什么误差量级"；稳定性界则给出不依赖模型结构的保守验证集界。

## 9. [E/A] 隐式偏置：最小范数与最大 margin，决定 $T\to\infty$ 的落点

**线性/核：最小范数（[E]）。** $\theta_0=0$、$T\to\infty$ 时 GD 收敛到最小 $\ell_2$ 范数插值（$p>n$ 时唯一：$\theta_\infty=X^\top(XX^\top)^{-1}y$；核情形为最小 RKHS 范数插值）。因此 $T=\infty$ 的验证误差 = §1 的 $R_{\rm alias}$/§6 的 optimal-recovery 误差；**有限 $T$ 的误差是谱滤波对该落点的部分逼近**，两者都精确可算。这解释了"隐式偏置"的定量含义：它决定误差曲线在 $T\to\infty$ 的端点，而 $T$ 决定沿谱方向收敛到端点的速度。

**分类 CE：最大 margin（[A]）。** 对可分数据、logistic/CE 损失、齐次网络，GD 方向收敛到最大 margin 解，且 $\|\theta_t\|$ 仅以 $\log t$ 速度增长（Soudry et al. 2018）⇒ logits 以 $\log t$ 增长、test CE 以 $1/{\rm margin}$ 的代数速度下降，但**没有有限 $T$ 的闭式**。对 $p>n$ 的插值网络，$T\to\infty$ 的 CE 继续下降而准确率不变（margin 增长）；验证 CE 的有限 $T$ 行为只有上界/渐近 **[U/A]**。

**开放（[O]）。** 深度非线性网络的隐式偏置（有限 $T$、"which interpolant"）没有一般定理；特征学习会改变偏置本身（§3 的 lazy 判据失效），因此 §1/§2 的精确端点不再适用。

**代表文献**：Soudry, Hoffer, Nacson, Gunasekar, Srebro 2018, arXiv:1710.10345；Gunasekar et al. 2018, arXiv:1806.00468；Hastie et al. 2019, arXiv:1903.08560（最小范数插值的风险）。

**在本题中的用途**：为"$T$ 步后验证误差"提供端点+逼近速度的严格解释（回归），以及 CE 的渐近方向（分类）；同时明确这是 feature-learning 区不可用的部分。

## 10. [E/U/H] 学习曲线与优化动力学：哪些能给出"$T$ 步后"的解析预测

**能给出解析 $T$ 步预测的三类。**
1. **PL 条件（[U]，二次/强凸时为 [E]）**：若 $\|\nabla L\|^2\ge2\mu(L-L^*)$ 且 $\eta\le1/L$，则
$$
L(\theta_T)-L^*\le(1-\eta\mu)^T(L(\theta_0)-L^*);
$$
对最小二乘 $L(\theta)=\frac12\|X\theta-y\|^2$，等号式由 §1 给出；对核/线性化网络，$\mu$ 由 $K$ 的最小特征值决定（$\kappa$ 大 ⇒ 慢），这直接预测"$T$ 步后还差多少"。

**PL 常数与谱。** 对最小二乘 $L(\theta)=\frac1{2n}\|X\theta-y\|^2$，$\mu=\lambda_{\min}^+(X^\top X/n)$（最小非零特征值）；条件数 $\kappa=\lambda_{\max}/\lambda_{\min}$ 给出步数尺度 $T_{1/2}\asymp\kappa\log2$，即"$T$ 步后还差多少"可写成 $\exp(-T/\kappa)$ 量级（精确常数由 §1 的谱展开给）。$p>n$ 时 $X^\top X$ 有 $p-n$ 个零特征值：$\mu$ 只取非零部分，零方向永不收敛（§1 的 $R_{\rm alias}$），故 PL 界只在可训练子空间上成立。
2. **Saxe 深度线性（[E]）**：§3 的 logistic 模式解给出每个奇异值模式的时间尺度 $\tau_k$；这是唯一能解析给出"非线性（乘积参数化）+ 有限 $T$"的模型：小奇异值模式先 plateau、后突降；多层叠加可造成非单调学习曲线（epoch-wise 双下降的机制解释）。
3. **谱滤波学习曲线（[E]/[A]）**：核/lazy 区 $R_{\rm te}(T)=\sum_j(1-\eta n\lambda_j)^{2T}a_j^2$；幂律谱 $\lambda_j\asymp j^{-\alpha}$、$a_j^2\asymp j^{-\alpha r}$ 时，$T$ 步后仍有误差 $\asymp(\eta nT)^{-(r-1/\alpha)}$（未学到的 tail），给出学习曲线的幂律指数；这是"$T$ 预算下的 loss"的解析预测。

**只有机制/经验规律，没有有限 $T$ 定理的两类。**
4. **Edge of stability（[H]）**：全批量 GD 在 $\lambda_{\max}(\nabla^2L)\approx2/\eta$ 附近运行，loss 短尺度非单调、长尺度下降。它能解释"$lr$ 跨临界值后验证曲线非单调/双峰"，但**不是**一般网络的 $T$ 步定理；Adam 也不在覆盖范围内。对二次/线性网络有部分解析结果，深层非线性为 **[O]**。
5. **SGD 噪声（[A]）**：随机修正方程/SME 把 SGD 近似为 SDE，噪声尺度 $\propto\eta\sigma_g^2/B$；只给弱收敛/扩散极限，不给精确有限 $T$ 验证误差。

**代表文献**：Karimi, Nutini, Schmidt 2016, arXiv:1608.04636（PL）；Saxe et al. 2014, arXiv:1312.6120；Arora et al. 2019, arXiv:1810.02281；Bordelon, Canatar, Pehlevan 2020, arXiv:2002.02561；Cohen et al. 2021, arXiv:2103.00065（EoS）；Li, Tai, E 2017, arXiv:1511.06251（SME）；Heckel & Yilmaz 2020, arXiv:2007.10099（epoch-wise 双下降的叠加机制）。

**在本题中的用途**：对"$T$ 步后"能给出解析预测的只有 1–3 类（线性/核/lazy/深度线性）；一般非线性网络的 $T$ 步验证误差是 **[O]**，只能给上界（§8 稳定性）或机制解释（EoS/plateau）。

---

# 第四部分：定理分级、开放问题与文献

## 11. 四个必须保持的区分（回答本题时逐条声明）

1. **训练误差 vs 验证误差**：$p>n$ 时 $R_{\rm tr}\to0$ 恒可达（插值），因此"训练到零"不是泛化证据；验证误差的 A/E/O 分解只在 kernel/lazy 区是精确可加的，其他区域只是组织概念。
2. **模型类容量 vs 算法隐式偏置**：A（以及 §6 的下界）由 $\mathcal G$ 与 $f$ 的类决定；E/O 与 $T\to\infty$ 的落点由算法决定（min-norm/max-margin）。"$p>n$ 能插值"属于容量侧的必要条件，不蕴含泛化。
3. **kernel/lazy vs feature-learning**：只有前者有"与 $T$ 无关的固定核"和 §2/§3 的闭式；后者连 a priori 函数空间都依赖训练路径，是 §12 的开放问题。
4. **回归 MSE vs 分类 CE**：MSE 有谱分解与闭式；CE 无闭式（softmax 非线性），有限 $T$ 只有界/渐近，隐式偏置为 max-margin；若标签随机，还多出不可约熵 $H(y|x)$ 的地板（本题确定性目标的 MSE 无此项）。**验证集统计**（Hoeffding/配对检验/多重比较，Hoeffding 1963；Dietterich 1998；Demšar 2006；Benjamini–Hochberg 1995）只决定"经验均值能否分辨 $\delta$"，不影响 $R_{\rm te}$ 本身的数学预测。
5. **每个结论必须声明"层次 + 分级 + 区域"**：给定实现 / 分布期望 / 类层 worst-case；[E]/[U]/[L]/[A]/[O]；lazy / feature-learning；MSE / CE。缺任一声明，"$T$ 步后验证误差"这一命题就没有确定含义——同一公式在不同层次下可以从"恒等式"变成"对最坏情况成立的上界"（§0 的三层）。

## 12. 定理分级总清单与开放问题

**12.1 [E] exact（有限 $n,T$ 等式）**：§1 线性 GD 递推与 $R_{\rm alias}$；§2 有限 $n$ 核预测公式（给定 $S,\theta_0$）；§3 NTK 无限宽闭式、深度线性模式解；§6 power-function 最优恢复公式、$\sqrt{\lambda_{n+1}}$ 的 RKHS n-width；§7 "exact = 下界 + 算法 excess" 的谱分解；§10 PL/最小二乘闭式。
**12.2 [U] upper**：§2/§8 的源条件+容量条件率 $n^{-r/(r+\beta)}$ 与 $T_*$；§4/§5 Barron/Sobolev 逼近率；§8 稳定性/PAC-Bayes/压缩界；§10 PL 收敛界；NTK 有限宽 $O(1/\sqrt w)$ 偏差界。
**12.3 [L] lower**：§6 n-width（$\sqrt{\lambda_{n+1}}$、$n^{-s/d}$）、optimal recovery、Fano/metric entropy 的 minimax 下界（$n^{-2s/(2s+d)}$、$n^{-r/(r+\beta)}$）。
**12.4 [A] asymptotic**：§2 随机特征比例渐近（Mei–Montanari）；double descent 曲线（Belkin et al.；Hastie et al.；Nakkiran et al.）；benign overfitting 阈值（Bartlett–Long–Lugosi–Tsigler 2020；Tsigler–Bartlett 2023；Muthukumar et al. 2019）；AMP/state evolution（Bayati–Montanari 2011）、Gaussian equivalence（Hu–Lu 2022；Goldt et al. 2020）、replica（Gerace et al. 2021；Zdeborová–Krzakala 2016）、universality（Montanari–Saeed 2023）；§9 最大 margin。
**12.5 [H] heuristic/empirical**：EoS（Cohen et al. 2021）；幂律 scaling laws（Kaplan et al. 2020；Hoffmann et al. 2022；Bahri et al. 2024 —— 后者给出四象限分类但仍是现象学）。
**12.6 一致收敛为何不是主工具 [U，但空泛]**：对插值类，VC/Rademacher 界的量级为 $\sqrt{p/n}$，$p>n$ 时 $\ge1$；Nagarajan–Kolter 证明即使只对 GD 实际输出（用尽隐式偏置）做双边一致收敛也仍是空泛的，且界可随 $n$ 增大而增大（arXiv:1902.04742；Zhang et al. 2017, arXiv:1611.03530）。故它不能用来回答"$T$ 步后验证误差"。
**12.7 [O] 开放问题**（按本题重要性排序）：
1. **feature-learning 区有限宽、有限 $T$ 的验证误差**：无固定核、无闭式；mean-field（Mei–Montanari–Nguyen 2018；Chizat–Bach 2018；Rotskoff–Vanden-Eijnden 2018）只在 $m\to\infty$ 的两层网络给出 PDE 极限，有限宽的精确预测没有定理。
2. **Adam/AdamW 的有限步理论**：逐参数预条件 ⇒ 无固定谱滤波；有效步长只有经验定义，缺解析 $T$ 步误差。
3. **分类 CE 的有限 $T$ 验证误差**：只有 max-margin 渐近与界，无闭式。
4. **$\mathcal G$ 的 n-width/逼近能力与核谱之间的定量联系**：给定参数化（深度/宽度/激活），其 n-width 最优性与 NTK 谱的关系只有部分结果（Daubechies et al. 2022 主要是一维/特定类）。
5. **带算法偏置的下界**：§6 的下界是类层 worst-case；"GD 类算法在给定 $T$ 下的 minimax 下界"基本空白。
6. **验证误差的波动理论**（跨初始化/数据抽样的方差）：除线性/核区可算外，深网多为经验。

## 13. 与 task-1 十个主题的对应索引

1 一致收敛与失效 → §12.6；2 偏差-方差/double descent/benign/RMT → §1、§2、§12.4；3 核/NTK/三项分解 → §2、§3、§7；4 优化动力学（PL/Saxe/EoS/SGD）→ §10；5 泛化界（稳定性/PAC-Bayes/压缩/local Rademacher）→ §8（稳定性段）、§12.2；6 精确高维渐近（AMP/GE/replica）→ §2（随机特征比例渐近段）、§12.4；7 特征学习（lazy/rich/μP/mean-field）→ §3（lazy/rich 判据段）、§12.7；8 逼近论（Barron/Sobolev/RKHS/minimax）→ §4、§5、§6；9 学习曲线/缩放律 → §10（第 1–3 类）、§12.5；10 验证集统计 → §11（第 4 条）。

## 14. 推导速记（exact 公式的来源，供正式文档引用）

**14.1 线性 GD 有限步（§1）[E]。** 损失 $L=\frac12\|X\theta-y\|^2$、$y=X\theta^*$ ⇒ $\nabla L=X^\top(X\theta-y)=X^\top X(\theta-\theta^*)$，故
$$
\theta_T-\theta^*=(I-\eta X^\top X)^T(\theta_0-\theta^*).
$$
$X^\top X$ 对称半正定，取正交 $V$、$X^\top X=V\Lambda V^\top$ 即得 §1 的谱展开；$R_{\rm te}=(\theta_T-\theta^*)^\top H(\theta_T-\theta^*)$。$\eta\ge2/s_{\max}^2$ 时存在 $|1-\eta s_i^2|\ge1$，不收敛。

**14.2 有限 $n$ 核预测（§2）[E]。** 函数空间的 GD 残差满足 $r_T=(I-\eta K_{\rm tr})^T r_0$（$r_0=f_0(X_{\rm tr})-y$）；任意 $x_{\rm te}$ 的预测由核的表示定理 $f_T(x_{\rm te})=f_0(x_{\rm te})+k_{\rm te}^\top K_{\rm tr}^{-1}\big(f_T(X_{\rm tr})-f_0(X_{\rm tr})\big)$ 给出，两式合并即 §2 第一个框。$K_{\rm tr}\succ0$（$p>n$ 的随机特征下几乎必然）保证可逆。

**14.3 谱滤波 ↔ ridge（§8）[E]。** $g_T(\mu)=1-(1-\eta n\mu)^T$；小 $\eta$ 时 $(1-z)^T\approx e^{-zT}$ 得 $g_T\approx1-e^{-\eta n\mu T}$；ridge 滤波为 $g_\lambda(\mu)=\mu/(\mu+\lambda)$。两者在 $\mu/\lambda_{\rm eff}$ 两端渐近一致，故率的指数相同、常数不同；对应关系 $\lambda_{\rm eff}\asymp1/(\eta nT)$。

**14.4 source+capacity 的平衡（§2/§8）[U]。** 偏差² $\le R^2\lambda^{r}$（source）；估计/设计项 $\le c\,\mathcal N(\lambda)/n\le c\,\lambda^{-\beta}/n$（capacity）。令 $\lambda_*=n^{-1/(r+\beta)}$ 得率 $n^{-r/(r+\beta)}$；早停 $T_*\asymp(\eta n)^{-1}n^{1/(r+\beta)}$。$r=1,\ \beta=d/(2s)\Rightarrow n^{-2s/(2s+d)}$。

**14.5 无噪声时设计项仍存在（§0/§4 的关键修正）[U]。** 无噪声 ⇒ $g_n=T_nf$，KRR 解 $f_\lambda=(T_n+\lambda)^{-1}T_nf$，于是
$$
f_\lambda-f=\underbrace{-\lambda(T+\lambda)^{-1}f}_{\text{bias (A/O)}}+\underbrace{\lambda(T+\lambda)^{-1}(T_n-T)(T_n+\lambda)^{-1}f}_{\text{设计采样 }(E:\ O(\sqrt{\mathcal N(\lambda)/n}))},
$$
$T_n\to T$ 由集中不等式控制（$\|(T_n-T)(T+\lambda)^{-1}\|\lesssim\sqrt{\mathcal N(\lambda)/n}$ 高概率）。所以 $\sigma=0$ 只删掉标签噪声项，**E 项仍以 $n^{-1/2}$ 量级存在**——这是"确定性目标 + 有限 $n$"与"无限数据谱公式"的本质差别。

**14.6 n-width / 最优恢复（§6）[E/L]。** 见 §6 的推导骨架：$d_n(B_K)=\sqrt{\lambda_{n+1}}$、$d_n(W^{s,2})\asymp n^{-s/d}$，以及 power function 的 Cauchy–Schwarz 上界与取等构造。**要点：这些是函数类层的最坏情况下界，与算法无关**。

**14.7 Saxe 解（§3/§10）[E]。** 两层线性、平衡初始化、白化数据：$\dot a_k=2a_k(a_k^*-a_k)$ ⇒ logistic 解（§3）；时间尺度 $\tau_k\propto1/a_k^*$、plateau 长度 $\propto\log(a_k^*/a_k(0))$。深度 $L$ 时为 $(D-1)$ 次幂模式的叠加；线性网络的 $R_{\rm te}(t)=\sum_k(a_k^*-a_k(t))^2$。

**14.8 PL 界（§10）[U]，二次损失时 [E]。** $\frac{d}{dt}(L-L^*)=-2\|\nabla L\|^2\le-2\mu(L-L^*)$ ⇒ $L(\theta_t)-L^*\le(1-\eta\mu)^t(L(\theta_0)-L^*)$（$\eta\le1/L$）。最小二乘时 $\mu=\lambda_{\min}^+(X^\top X/n)$，零空间分量不衰减（§1 的 $R_{\rm alias}$）。

## 15. 文献核实清单

**15.1 已核 arXiv（2026-10-19 经 `export.arxiv.org` API 核对标题/编号）**
- 1903.08560 Hastie et al. 2019｜1806.00468 Gunasekar et al. 2018｜1903.07571 Belkin, Hsu, Xu 2020｜1710.10345 Soudry et al. 2018
- 1908.05355 Mei & Montanari 2019｜2002.02561 Bordelon et al. 2020｜1806.07572 Jacot et al. 2018｜1312.6120 Saxe et al. 2014｜1810.02281 Arora et al. 2019｜1812.07956 Chizat et al. 2019｜2011.14522 Yang & Hu 2021
- 1702.07254 Fischer & Steinwart 2020｜1412.8690 Bach 2017｜1906.08039 E, Ma, Wu 2021｜1610.01145 Yarotsky 2017｜1905.02199 Daubechies et al. 2022｜1708.06633 Schmidt-Hieber 2020
- 1306.3574 Raskutti, Wainwright, Yu 2014｜1604.04054 Blanchard & Mücke 2018｜1509.01240 Hardt, Recht, Singer 2016
- 1608.04636 Karimi et al. 2016｜2103.00065 Cohen et al. 2021｜1511.06251 Li, Tai, E 2017｜2007.10099 Heckel & Yilmaz 2020
- 1807.02582 Kanagawa et al. 2018｜1902.04742 Nagarajan & Kolter 2019｜1611.03530 Zhang et al. 2017｜1811.03804 Du et al. 2019｜1811.03962 Allen-Zhu, Li, Song 2019｜1804.06561 Mei, Montanari, Nguyen 2018｜1805.09545 Chizat & Bach 2018｜1805.00915 Rotskoff & Vanden-Eijnden 2018
- [A] 类：1812.11118 Belkin et al. 2019｜1912.02292 Nakkiran et al. 2019｜1906.11300 Bartlett, Long, Lugosi, Tsigler 2020｜2009.14286 Tsigler & Bartlett 2023｜1903.09139 Muthukumar et al. 2019｜1001.3448 Bayati & Montanari 2011｜2009.07669 Hu & Lu 2022｜1909.11500 Goldt et al. 2020｜2002.09339 Gerace et al. 2021｜1511.02476 Zdeborová & Krzakala 2016｜2202.08832 Montanari & Saeed 2023｜2106.04795 Wang & Lin 2023｜2104.13628 Cao, Gu, Belkin 2021｜2202.05928 Frei, Chatterji, Bartlett 2022｜1804.05862 Zhou et al. 2019｜1802.05296 Arora et al. 2018｜1703.11008 Dziugaite & Roy 2017｜1808.00387 Liang & Rakhlin 2020｜0910.2120 Benaych-Georges & Nadakuditi 2011｜1506.02629 Dwork et al. 2015｜1502.04585 Blum & Hardt 2015｜2102.06701 Bahri et al. 2024｜2001.08361 Kaplan et al. 2020｜2203.15556 Hoffmann et al. 2022

**15.2 无 arXiv（期刊/专著；a priori/逆问题/逼近论/信息论经典）**
- Kolmogorov 1936（Ann. of Math. 37）；Pinkus 1985（n-Widths in Approximation Theory, Springer）；Micchelli & Rivlin 1977（A Survey of Optimal Recovery, Plenum）；Melkman & Micchelli 1979（SIAM J. Numer. Anal. 16:87）；Novak 1988（Springer LNM 1349）；Traub, Wasilkowski, Woźniakowski 1988（Information-Based Complexity, Academic Press）；Donoho 1994（Ann. Statist. 22:238）；Yang & Barron 1999（Ann. Statist. 27）；Wendland 2005（Scattered Data Approximation, CUP）；Natterer 1977（Numer. Math.）；Engl, Hanke, Neubauer 1996（Regularization of Inverse Problems, Kluwer）；Caponnetto & De Vito 2007（Constr. Approx. 26）；Yao, Rosasco, Caponnetto 2007（Constr. Approx. 26）；Barron 1993（IEEE Trans. IT 39）；Stone 1982（Ann. Statist. 10）；Vapnik & Chervonenkis 1971（Theory Probab. Appl. 16）；Bousquet & Elisseeff 2002（JMLR 2）；Hoeffding 1963（JASA 58）；Dietterich 1998（Neural Comput. 10）；Demšar 2006（JMLR 7）；Benjamini & Hochberg 1995（JRSS-B 57）；Steinwart & Christmann 2008（Support Vector Machines, Springer）；Boucheron, Lugosi, Massart 2013（Concentration Inequalities, OUP）。

**15.3 使用注意**：源条件写法有 $f=T^{r/2}g$（本稿，$r=1$ 对应 $f\in\mathcal H$ 的"一阶光滑"）与 $f=T^{r}g$（文献常见，指数差 2）两套约定，跨文献引用时须换算；n-width 的"$n$"是子空间维数、optimal recovery 的"$n$"是观测数，二者在本稿中同用 $n$ 是因为 ellipsoid 情形下相等（$\sqrt{\lambda_{n+1}}$），一般类不保证，需核对。任务描述中的"Bartlett–Montanari–Mensh"查无此组合，正确出处为 arXiv:1906.11300（Bartlett, Long, Lugosi, Tsigler）。

**一句话总结**：对确定性 $f$、给定 $\mathcal G$ 与算法，**能严格算出的验证误差只有两大类**——给定实现时 §1/§2 的谱滤波闭式（$A+E+O$ 全含，无需训练），以及 $n\to\infty$ 总体极限下的谱展开（$A+O$）；**能严格证明的 a priori 结论**是 §6 的信息下界与 §7 在 source+capacity+对齐条件下的最优率；**其余（feature-learning、Adam、CE 有限 $T$）没有定理，只有界、渐近或开放问题**。

## 16. 总表：理论 → 可计算量 → 预测什么 → 在本题（a priori）中的角色

| 理论/结果 | 可计算量（计算前提） | 预测什么（A/E/O/F 或严格性） | 角色与局限 |
|---|---|---|---|
| §1 线性/多项式 GD 有限步递推 [E] | $X^\top X$ 谱、$\theta_0-\theta^*$、$\eta,T$（需给定 $S$） | A（零空间 aliasing）+ O；$R_{\rm alias}$ 与 $T$ 无关 | 给定实现即精确；$p>n$ 时给出"不可训练地板"；随机特征期望版给出 $(1-1/\gamma)\lVert\theta^*\rVert^2$ |
| §1 线性 min-norm 风险 [E] | $\gamma=p/n$、$\lVert\theta^*\rVert^2$ | A+E（分布层） | 精确闭式；无噪声时第二降段地板=目标能量中未对齐部分 |
| §2 有限 $n$ 核预测公式 [E] | $K_{\rm tr}$、$k_{\rm te}$、$f$ 的样本值、$\eta,T$ | A+E+O（给定 $S$） | **核心工具**：不训练即可算任意测试点的验证误差；lazy 区适用 |
| §2/§8 源条件+容量条件率 [U] | $r(f)$、$\beta$（核谱）、$n$ | A+O 的率、$T_*$ | 给出 $n^{-r/(r+\beta)}$ 与 $T_*\asymp(\eta n)^{-1}n^{1/(r+\beta)}$；无噪声时仍成立（设计采样项仍在） |
| §2 随机特征/双下降 [A] | $\gamma$、$\lambda$、$\sigma^2$、协方差谱 | A+E+O 全曲线 | 只在比例渐近+随机数据下精确；$\gamma\gg1$ 时只剩地板；不能当有限 $n$ 预测器 |
| §3 NTK 闭式 [E]/[U] | 初始化 Jacobian $J$、$K=JJ^\top$、$f$ 值、$\eta,T$ | A+O；有限宽 $O(1/\sqrt w)$ | lazy 判据成立时精确；Adam/强特征学习失效 |
| §3 深度线性模式解 [E] | 初始/目标奇异值、深度、$T$ | O 的 $T$ 步解析值；plateau 时间尺度 | 唯一"非线性乘积参数化+有限 $T$"的精确结果 |
| §4 目标 a priori 量 [U] | $\lVert f\rVert_{\mathcal H}$、$r(f)$、$s$、Barron 范数（只需 $f$ 与核） | 控制 A/E 的率与常数 | 把"目标多难"压成标量；$r=0$ 时无速率 |
| §5 表达式 a priori 量 [U/L] | NTK 谱/有效秩/$\kappa$/power function；n-width/逼近率（只需 $\mathcal G$） | A 的天花板、O 的难易、E 的尺度 | 两种容量概念须分开：算法函数空间 vs 模型类逼近能力 |
| §6 n-width / optimal recovery [L]/[E] | $\lambda_{n+1}$、Sobolev $s$、power function | 任何算法的验证误差下界 | 严格地板：RKHS $\sqrt{\lambda_{n+1}}$、Sobolev $n^{-s/d}$；无噪声时以此为准 |
| §6 minimax（有噪声）[L] | 谱/熵、$s$ 或 $(r,\beta)$、$n$ | A+E 的下界（$n^{-2s/(2s+d)}$、$n^{-r/(r+\beta)}$） | 只对有噪声统计模型成立；本题无噪声时偏保守 |
| §7 对齐/容量条件 [E/U/L] | $a_j$ vs $\lambda_j$、$\mathcal N(\lambda)$ | 上界=下界时的最优性；excess 来源 | 判定"$f$ 与 $\mathcal G$ 是否匹配"；给出可达率与不可达原因 |
| §8 早停=谱滤波/ridge 路径 [E/U] | $\lambda_{\rm eff}\asymp1/(\eta nT)$ | 把 $T$ 映射为正则化；$T_*$ 与率 | 精确等价（核/线性）；Adam 不适用；无噪声时 $T\to\infty$ 即插值，未必更差 |
| §8 稳定性/PAC-Bayes [U] | $\eta,T,n,L$（稳定性）；后验（PAC-Bayes） | 验证误差−训练误差的上界 | 不依赖结构但常数大；唯一含 $T$ 的通用验证集界 |
| §9 隐式偏置（min-norm/max-margin）[E/A] | 初始化、$T\to\infty$ 端点；margin | 决定 $T\to\infty$ 落点与 CE 的渐近 | 回归精确；CE 仅渐近；深非线性网络开放 |
| §10 PL/谱学习曲线 [E/U] | $\mu$、$\kappa$、谱、$T$ | O 的 $T$ 步界；幂律指数 $(r-1/\alpha)$ | 核/线性精确；一般网络 PL 常数难估 |
| §10 EoS [H] | Hessian $\lambda_{\max}$、$\eta$ | 稳定性边界与非单调性（非 $T$ 步定理） | 只解释 $lr$ 临界行为；深网/Adam 开放 |
| §12.6 一致收敛 [U] | VC/Rademacher/覆盖数 | E 的经典上界 | $p>n$ 时 $\ge1$，不能回答问题；仅作反例/对照 |
| §11-4 验证集统计 | 测试集大小、配对差、种子数 | F：经验均值能否分辨 $\delta$ | 只影响"测量"，不影响 $R_{\rm te}$ 的数学值 |

