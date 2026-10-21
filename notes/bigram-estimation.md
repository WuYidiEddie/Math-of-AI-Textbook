# bigram 族的测试 CE 理论：熵地板之上的三分解

> 日期：2026-10-20 ｜ scope：v1.5 `bigram_lm` 族（100 题 / 300 候选）
> 本文把 bigram 的 test CE 写成 $H + $ 瓶颈 $A + $ 估计 $I + $ 优化 $O(T) + $ 测量 $F$，逐项给出可计算公式与机器验证。
> 验证脚本：[`../verification/bigram_estimation.py`](../verification/bigram_estimation.py)（纯 numpy，复现 v1.5 生成器）；结果：[`../verification/bigram_estimation_result.md`](../verification/bigram_estimation_result.md)。
> 结论先行：**恒等式与估计地板是 exact/asymptotic 且数值通过；优化项对"查表 softmax + GD"模型可证、对 TF/GRU 只能给下界与相位判断；d_model < V−1 的秩瓶颈给出一个小的正 $A$。**

---

## 0. 设定（事实，来自题面 synthesize 代码）

- 转移矩阵：$P_{xy}=\mathrm{softmax}(\alpha z_{xy})$，$z_{xy}\overset{\rm iid}{\sim}N(0,1)$，$V\in\{24,32,48\}$，$\alpha\in\{0.8,1.0,1.4\}$；行与行独立。
- 初始分布取平稳分布 $\pi$（幂迭代 256 次）；训练 $800$ 条长 $L+1$ 的序列，$L=\mathrm{ctx}\in\{12,16,24\}$；测试 $200$ 条。同一 $P$ 供 train/test，只有窗口不同。
- 训练 token 对数 $N=800L\in\{9600,12800,19200\}$；测试 token 数 $M=200L\in\{2400,3200,4800\}$。
- 指标：test token 平均 CE（nats）。训练有 weight decay（题面候选 wd $\in\{10^{-4},10^{-3},0\}$），minibatch 有放回。
- 该族**不是确定性目标**：$H(Y|X)=\sum_x\pi_xH(P_x)>0$ 是不可约地板（题库实测 2.281–3.604 nats）。

---

## 1. 精确恒等式（exact，逐样本成立）

对任何**固定**预测器 $\hat p(\cdot|x)$（不看测试集训练出来的），

$$
\underbrace{\frac1M\sum_{t=1}^{M}-\log \hat p(y_t|x_t)}_{\text{test CE}}
=\;H(Y|X)\;+\;\underbrace{\sum_x\pi_x\,\mathrm{KL}(P_x\Vert\hat p_x)}_{\text{总体超额 }E_{\rm pop}}\;+\;F,
$$

其中 $F$ 是测试集抽样涨落，$\mathbb E[F]=0$。这是**恒等式**（把 $\pm\frac1M\sum\log P$ 插入即得），不是渐近。

$F$ 的尺度：$M$ 个 token 的逐 token 得分 $s_t=-\log\hat p(y_t|x_t)$ 有方差 $\sigma_s^2$，链内相关给混合膨胀因子 $\kappa=1+2\sum_{k\ge1}\rho_k$，

$$
\mathrm{sd}(F)=\sqrt{\kappa\,\sigma_s^2/M}.
$$

**数值**（V=32, $\alpha$=0.8, L=16，200 次测试重采样）：恒等式残差 $-0.00027$（容差 $3\,\mathrm{se}=0.0028$）✓；$\mathrm{sd}(F)=0.0132$；$\sqrt\kappa=0.99$——**随机 bigram 链混合极快，$\kappa\approx1$，iid 公式直接用即可**。

推论（重要）：$F\approx\pm0.013$ nats（M=3200）与估计地板（§2，0.014–0.117）同量级。但三候选**共用同一测试集**，配对差里 $F$ 大部分抵消；10 seeds 均值再去掉种子噪声（`std_metric` 中位 0.0098）。

---

## 2. 估计地板 $I$：plug-in 与加 $\beta$ 平滑（asymptotic + exact 数值）

bigram 的"识别"问题就是 $V$ 个独立多项分布的估计。查表估计量 $\hat p_{xy}=(n_{xy}+\beta)/(n_x+\beta V)$（$n_{xy}$=训练集里 $(x,y)$ 计数）。

**一阶公式（asymptotic，固定 $P$ 内部点、$n_x\to\infty$）**：

$$
\mathbb E\,\mathrm{KL}(P_x\Vert\hat p_x)=\frac{V-1}{2n_x}+O(n_x^{-2})
\quad\Longrightarrow\quad
\mathbb E[E_{\rm pop}]\approx\sum_x\pi_x\frac{V-1}{2n_x}\approx\frac{V(V-1)}{2N},
$$

最后一步用 $n_x\approx N\pi_x$：$\sum_x\pi_x\cdot\frac{V-1}{2N\pi_x}=\frac{V(V-1)}{2N}$，**与 $\pi$ 的形状无关（一阶）**。该率（每 context $(V-1)/2n$）是多项分布估计的 minimax 率，add-$\tfrac12$（Krichevsky–Trofimov）一致达到；所以没有估计量能一致地低于这个地板。

**精确版（exact，给定 $n_x$ 与 $\beta$）**：逐格二项期望

$$
\mathbb E[E_{\rm pop}]=\sum_x\pi_x\sum_y\sum_{k=0}^{n_x}\binom{n_x}{k}p_{xy}^k(1-p_{xy})^{n_x-k}\;p_{xy}\log\frac{p_{xy}(n_x+\beta V)}{k+\beta},
$$

可在数秒内精确算出（脚本 `exact_excess_add_beta`）。

**数值核对**（$\beta=0.5$，32 次训练数据重采样）：

| V | $\alpha$ | L | N | 模拟均值 | 精确二项 | 一阶公式 | 一阶/精确 |
|---|---|---|---|---|---|---|---|
| 32 | 0.8 | 16 | 12800 | 0.0381±0.0003 | 0.0380 | 0.0387 | 1.02 |
| 48 | 1.4 | 12 | 9600  | 0.0865±0.0005 | 0.0869 | 0.1175 | **1.35** |
| 24 | 1.0 | 24 | 19200 | 0.0142±0.0001 | 0.0143 | 0.0144 | 1.00 |

读法：

- 一阶公式在 $V\le32$ 时误差 ≤2%；**$V{=}48,\alpha{=}1.4$ 时高估 35%**：该行有很多 $p_{xy}n_x\lesssim1$ 的"未分辨格"，二项精确值自动处理，一阶展开失效。大 $\alpha$、大 $V$、小 $N$ 时用精确二项或模拟。
- **v1.5 全族地板表**（一阶公式，$\beta$ 无关到一阶）：$V(V-1)/(2N)$ 给 0.0144（V24/L24）到 0.1175（V48/L12）nats。
- 为什么 TF/GRU 也逃不开这个地板：生成器的行是先验独立的，context 之间**没有结构可共享**（非低秩、非参数族）；跨 context 池化的期望收益是 $O(1/n_x^2)$ 阶。这是"对生成器的 target-specific 推断"（grounded heuristic），不是对任意 $P$ 的定理。

---

## 3. 逼近项 $A$：d_model < V−1 的秩瓶颈（bound/estimate）

因果 LM 预测下一 token 的最优只依赖当前 token（一阶马尔可夫）。模型把 context token $x$ 映到 logits $\ell(x)\in\mathbb R^V$，而 $\ell(x)=W_{\rm head}\,h(x)$、$h(x)\in\mathbb R^{d_{\rm model}}$，所以 $V\times V$ 的 logit 矩阵 $\Lambda$ 满足 $\mathrm{rank}(\Lambda)\le d_{\rm model}+1$。softmax 消去每行常数方向后，通用 $P$ 的 log-odds 矩阵需要秩 $V-1$。因此

$$
d_{\rm model}+1<V\;\Longrightarrow\;A=\min_{\mathrm{rank}\,\Lambda\le d+1}\sum_x\pi_x\mathrm{KL}(P_x\Vert\mathrm{softmax}\,\Lambda_{x\cdot})>0 .
$$

SVD 截断给出上界估计（heuristic，未做 KL 意义下的精确秩约束优化）：

| V | $\alpha$ | $d_{\rm model}$=32 | $d_{\rm model}\ge V$ |
|---|---|---|---|
| 48 | 0.8 | 0.0086 | ≈0 |
| 48 | 1.0 | 0.0131 | ≈0 |
| 48 | 1.4 | 0.0239 | ≈0 |
| 32 | 1.0 | ≈0（$d\ge V-1$） | ≈0 |

读法：瓶颈**只咬 $V{=}48,d{=}32$ 的候选**，量级 0.01–0.024 nats——比估计地板小、比 $O$ 小得多，但在"刀刃题"（gap < 0.02）里可能翻盘。真实数据对照（bank-audit 300 候选）：$V{=}48$ 时 $w{=}32$ 组超额中位 0.306 vs $w{\in}\{64,128\}$ 组 0.297，方向一致但被优化器/层数混杂，不能当证据，只能当**可证伪预测**（P-B4，见 §6）。

---

## 4. 优化项 $O(T)$：查表 softmax + GD 可精确分析（bound + 数值）

查表模型 $\hat p_{xy}=\mathrm{softmax}(\theta_{x\cdot})$、从 $\theta=0$ 出发、平均 token 损失 + weight decay $\frac\gamma2\|\theta\|^2$、全批量 GD：

$$
L(\theta)=\sum_x w_x\,\ell_x(\theta_{x\cdot})+\tfrac\gamma2\|\theta\|^2,\qquad
w_x=n_x/N,\quad \ell_x=-\sum_y q_{xy}\log\hat p_{xy}.
$$

- **逐 context 解耦、凸**。Hessian 块 $w_x(\mathrm{diag}\,\hat p-\hat p\hat p^\top)$，特征值 $\le w_x/2$，故光滑常数 $L_s=\max_x n_x/(2N)$（exact 线性代数）。
- **凸 GD 界（exact 定理）**：$\eta\le1/L_s$ 时 $L(\theta_T)-L(\theta_*^\gamma)\le \frac{\|\theta_*^\gamma\|^2}{2\eta T}$。
- **wd 的必要性**：$\gamma=0$ 且存在零计数格时，$\theta_{xy}\to-\infty$（MLE 在无穷远），无有限最小点，标准凸界失效；测试超额对 $T$ 呈 U 形（先降后因零格被压穿而升）。$\gamma>0$ 等价于向均匀分布的平滑，给出有限 $\theta_*^\gamma$。
- **wd 是有代价的平滑（数值）**：$\gamma=10^{-3}$ 时收敛超额 0.0705 vs add-$\tfrac12$ 地板 0.0401（+76%）；$\gamma$ 相当于强加了一个未必最优的平滑强度。
- **实测形状（V=32, $\alpha$=0.8, L=16, $\gamma=10^{-3}$, $\eta=0.04/L_s$）**：凸界在全部检查点成立（训练侧，松 30–130 倍）；总体超额单调降到收敛值；中段 log-log 斜率 $\approx1.0$（$1/T$ 相），早期有 $\sim$50 步的 soft 启动平台（从均匀出发的逃逸期），后期指数饱和。总体超额 $\approx$ 1.2–5.6× 训练侧 gap。
- **时间尺度**：$\eta=1/L_s$ 时 $\sim$128 全批量步收敛——查表模型是 bigram 里**条件最好**的模型（块对角 Hessian）；任何把 context 耦合进共享参数的网络（TF/GRU/embedding）条件数只会更差。**所以查表 GD 的收敛时间是该族优化时间的下界**（heuristic，但方向可靠）：实测候选在 256–2048 步、远小于 $1/L_s$ 的有效步长下仍显著未收敛（P5：超额中位 0.296 ≫ 地板 0.039），说明 $O$ 主导。

对 TF/GRU 的有限 $T$：没有闭式；可做的是 (i) 用查表模型给时间下界与地板；(ii) 用 02 的核工具在初始化 logits 上线性化（但 CE 非二次，且 bigram 输出维 $V$ 大，核是 $nL\times nL$ 分块——计算上可行但未做）；(iii) 标 open。

---

## 5. 与 v1.5 实测的对照（bank-audit.csv，300 候选）

| 量 | 值 |
|---|---|
| 地板 $V(V-1)/(2N)$ | min 0.0144 / 中位 0.0388 / max 0.1175 nats |
| 实测超额（mean CE − $H$） | min 0.0487 / 中位 0.296 / max 1.36 nats |
| 超额 < 地板的候选 | **0 / 300**（地板没有被违反，一致性检验通过） |
| 最小 超额/地板 | 1.36（最优候选已贴近信息极限的 1.4 倍） |
| $V{=}48$：$w{=}32$ vs $w{\ge}64$ 超额中位 | 0.306 vs 0.297（方向符合瓶颈，量小、混杂，不作证据） |

读法：

1. **大多数候选的超额是 $O$（优化）主导**：中位超额是地板的 7.6 倍。bigram 的"该选谁"在当前预算下主要不是估计问题，而是"谁在 $T$ 步内把转移表学得更多"。
2. **最优候选已接近信息极限的常数倍**（1.36×）：对这些题，再改架构的收益被地板封死，天花板厚度 ≈ 0.01–0.05 nats。
3. $F$（±0.013）与地板同量级：超额差 < 0.03 nats 的题本质上是刀刃题。

---

## 6. 可证伪预测

| # | 预测 | 检验方式 |
|---|---|---|
| P-B1 | 任何候选 mean CE $\ge H + V(V-1)/(2N) - 2\,\mathrm{se}$（se 含 $F$ 与种子） | bank-audit 全体 300 候选；当前 0 违反 ✓ |
| P-B2 | 固定题、增大 $L$（$N$ 同比），地板 $I\propto1/N$；实测最优候选的超额以 $\ge$ 该速率下降 | 题族内 $L\in\{12,16,24\}$ 对照（混杂预算，需控制） |
| P-B3 | wd 过大的候选收敛超额 $\approx$ 平滑偏置（可用 §4 查表模拟预测方向） | 比较同架构不同 wd 的候选对 |
| P-B4 | $V{=}48$ 且 $d_{\rm model}{=}32$ 的候选，在**长预算/强优化器**下比 $d\ge64$ 候选高出 $\approx0.01$–$0.024$ nats 的不可约差 | 现有数据混杂；需控制 opt/lr/T 的子集或新实验 |
| P-B5 | 查表 + add-$\tfrac12$（或等价贝叶斯）预测器达到 $H+V(V-1)/(2N)$ 量级；任何神经网络候选显著低于此值即证伪本文 | 合成复现（已做，脚本 Part 2） |

---

## 7. 适用域与明确不能做的

**能做**：熵地板 exact；估计地板 asymptotic + exact 数值；$F$ 公式与 $\kappa\approx1$；查表模型的凸优化界；秩瓶颈的估计；全族地板表与 300 候选一致性核对。

**不能做（标 open）**：

1. TF/GRU 有限 $T$ 的 test CE 闭式——参数耦合 + CE 非二次，无谱滤波公式；
2. wd 到"有效 $\beta$"的精确映射（本文只有线性化直觉 + 数值）；
3. minibatch 有放回采样对 $O$ 的精确修正（本文全批量；mini-batch 噪声进入 $F$/稳定性，不是主导项的判断是 heuristic）；
4. 秩瓶颈 $A$ 的精确值（本文是 SVD 上界估计；KL 意义下的秩约束最优未做）；
5. 序列级马尔可夫相关对训练 minibatch 的影响（$\kappa\approx1$ 是对测试侧验证的，训练侧假设同阶）。

**一句话**：bigram 的 $H + I + F$ 现在是可以"先算"的（地板 0.014–0.117 nats、$F\approx\pm0.013$），$A$ 只在 $V{=}48,d{=}32$ 时非零且小（≤0.024），而实测超额的主体是 $O$——查表模型给 $O$ 一个可计算的下界时间尺度，TF/GRU 的有限 $T$ 仍是 open problem。
