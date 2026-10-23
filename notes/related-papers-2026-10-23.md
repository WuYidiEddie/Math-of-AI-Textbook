# 相关论文普查：有没有人做过类似的研究？

> 日期：2026-10-23 ｜ 方法：arXiv API 六方向 19 条查询（候选 205 篇去重后人工筛选）+ 网页检索数学期刊综述
> 对照基准：本项目已有综述 [`lit-survey.md`](lit-survey.md)（2026-10-19，约 60 条已核实文献）
> 标注：**★ = 不在 lit-survey 中**（本次新增）；分类标签取 arXiv 主分类（`math.*` = 数学论文）

---

## 0. 结论速览

1. **"把训练当纯数学问题、逐项分解验证误差"这个纲领本身，已经有形态各异的同类工作。** 最接近的一份宣言是 2026 年 4 月的 *There Will Be a Scientific Theory of Deep Learning*（[arXiv:2604.21691](https://arxiv.org/abs/2604.21691)），它把"表征训练过程/隐藏表示/最终权重/性能的统计性质"定义为一个正在成形的科学理论——与本项目母项目（Science of AI Textbook）的定位几乎逐句对应，但它停留在"五条研究线索的综述"，没有给出本目录这种"给定 $f,\mathcal G,n,T$ 就算出 $R_{\rm te}$"的可计算判据。
2. **本项目的数学骨架（谱滤波 + source/capacity + 早停）在文献里有完整的成熟形态，只是名字叫"逆问题的谱正则化理论"**（statistical inverse learning）。Lin–Rudi–Rosasco–Cevher 2018（★ [1801.06720](https://arxiv.org/abs/1801.06720)）把 GD/ridge/PCR 统一成一族谱滤波器并在 source+capacity 下证明 minimax 最优率——正文 02/03 章的推导版内容（有效步长、$\lambda_{\rm eff}\asymp 1/(\eta nT)$、率 $n^{-r/(r+\beta)}$）在这一支文献里是定理级标准品。**这条线 lit-survey 只引了 Caponnetto–De Vito 与 Yao–Rosasco–Caponnetto 两篇 2007，2016 之后的谱算法文献是空白，本次补上。**
3. **"学习曲线 = 谱的（离散）拉普拉斯变换"这一精确图像，2026 年有新论文把它做成了严格等价定理**：Li–Cheng–Chen–Lin（★ [2609.08817](https://arxiv.org/abs/2609.08817)，math.ST）证明 RKHS 回归与序列模型对一切 Lipschitz 谱算法**风险等价**——即正文 03 章"曲线由谱与目标系数决定"的严格版本。
4. **n-width / optimal recovery 与学习的连接**已有 Chen–Long–Wu 的 IBC 对偶框架（★ [2305.05642](https://arxiv.org/abs/2305.05642)，借 information-based complexity 工具统一逼近与估计）和 Hsu 的线性方法维数下界（★ [2508.13346](https://arxiv.org/abs/2508.13346)）；这条线与 lit-survey §6 互补。
5. **Edge of Stability 的数学理论 2024–2026 年显著成熟**（高维随机分析、过参数化最小二乘收敛率、分岔理论框架），lit-survey 里仍标注为 [H]（heuristic），**该评级已过时，正文 03 章的 EoS 小节值得按下面 §2.6 更新**。
6. **没有发现**与本项目完全相同组合的工作：「a priori 可计算判据 + 四分解 $A/I/O/F$ + 用题库做零拟合证伪（theory answerer）」。单个组件全部有主；组合（尤其"拿理论预测去逐题打榜"这一步）在检索范围内未见先例。

---

## 1. 最接近的"同类研究纲领"（综述/书/manifesto）

| 文献 | 年份 | 与本项目的关系 |
|---|---|---|
| *There Will Be a Scientific Theory of Deep Learning*，[arXiv:2604.21691](https://arxiv.org/abs/2604.21691) ★ | 2026 | 纲领层面最像：主张深度学习的"科学理论"正在成形。但只综述五条线索（谱动力学、隐式偏置、scaling law 等），无可计算判据、无 testbed |
| Roberts–Yaida–Hanin, *The Principles of Deep Learning Theory*，[arXiv:2106.10165](https://arxiv.org/abs/2106.10165)，CUP 2022 ★ | 2021 | 物理式有效理论：逐层迭代方程 + 近高斯修正。给出的是"预测器"风格理论，但对象是无限宽+弱相互作用，与本项目"lazy 精确 + rich 承认开放"的分层不同 |
| DeVore–Hanin–Petrova, *Neural Network Approximation*，[arXiv:2012.14501](https://arxiv.org/abs/2012.14501)，**Acta Numerica 30**（数学顶刊综述）★ | 2020/21 | 纯逼近论侧（本项目 $A$ 项）的最权威数学综述；n-width、非线性逼近与网络的对比全在里面 |
| Berner–Grohs–Kutyniok–Petersen, *The Modern Mathematics of Deep Learning*，[arXiv:2105.04026](https://arxiv.org/abs/2105.04026)，CUP 书章 ★ | 2021 | 与本 README §0 的"三缺陷批判"同一动机（过参数化为何泛化、深度、维度诅咒、优化）；但结论是列举已知定理，不做逐题预测 |
| Jentzen 等, *Mathematical Introduction to Deep Learning*，[arXiv:2310.20360](https://arxiv.org/abs/2310.20360) ★ | 2023 | 教科书式全细节，覆盖架构与优化算法；偏"把工程对象写成严格数学"，不问"验证误差能否 a priori 算出" |
| Bach, *Learning Theory from First Principles*（MIT Press 2024，有公开书稿）★ | 2024 | 核方法/谱/泛化界的标准处理；本项目 0.5/0.6 章与其重叠，可作为前置读物引用 |

**判断**：本项目的位置 ≈ "Roberts–Yaida–Hanin 的可计算精神" × "谱正则化理论的严格性" × "用固定题库做可证伪检验"——前两者各有书级载体，第三者是检索范围内未见有人做的部分。

---

## 2. 按六个方向的核心数学论文

### 2.1 谱滤波 / 核梯度流的精确理论（正文 02、03 章）

| 论文 | 年 | 分类 | 要点 |
|---|---|---|---|
| ★ Lin, Rudi, Rosasco, Cevher, *Optimal Rates for Spectral Algorithms with Least-Squares Regression over Hilbert Spaces*，[1801.06720](https://arxiv.org/abs/1801.06720) | 2018 | stat.ML/math.FA | **最该补引的一篇**：GD/ridge/PCR 统一为谱滤波族，source+capacity 下高概率 minimax 率；正文 03 的"滤波对照"是其特例 |
| ★ Li, Cheng, Chen, Lin, *Risk Equivalence between RKHS Regression and Sequence Models for Lipschitz Spectral Algorithms*，[2609.08817](https://arxiv.org/abs/2609.08817) | 2026 | **math.ST** | RKHS 回归风险与序列模型逐模式等价 ⇒ "曲线=谱的拉普拉斯变换"的严格化 |
| ★ Cheng, Chen, Lin, *Optimal Confidence Band for Kernel Gradient Flow Estimator*，[2605.05768](https://arxiv.org/abs/2605.05768) | 2026 | **math.ST** | 核梯度流的 sup 范数风险与一致推断；超出本项目只做 $L^2$ 的范围 |
| ★ Li, Zhang, Lin, *On the Asymptotic Learning Curves of Kernel Ridge Regression under Power-law Decay*，[2309.13337](https://arxiv.org/abs/2309.13337) | 2023 | **math.ST** | 幂律谱+幂律目标系数下 KRR 学习曲线的渐近刻画——正是本项目真题族假设的分布层版本 |
| ★ Lu, Zhang, Li, Lin, *On the Saturation Effects of Spectral Algorithms in Large Dimensions*，[2503.00504](https://arxiv.org/abs/2503.00504) | 2025 | stat.ML | 饱和效应（目标过光滑时 KRR 达不到信息下界）——对应本项目"$T$ 再大也突破不了的地板"的大维版本 |
| ★ Averyanov & Celisse, *Early stopping and polynomial smoothing in regression with reproducing kernels*，[2007.06827](https://arxiv.org/abs/2007.06827) | 2020 | **math.ST** | 数据驱动的早停规则；补 Raskutti–Wainwright–Yu 2014 之后的进展 |
| ★ Blanchard & Krämer, *Convergence rates of Kernel Conjugate Gradient for random design regression*，[1607.02387](https://arxiv.org/abs/1607.02387) | 2016 | **math.ST** | CG 早停=更激进的谱滤波；本项目"算法 $\mathcal A$ 改变滤波形状"的现成定理 |
| ★ Rastogi, Helin, Mücke, *Statistical inverse learning problems with random observations*，[2312.15341](https://arxiv.org/abs/2312.15341) | 2023 | **math.ST** | 统计逆问题综述：谱正则化+随机设计——本项目 02 章的"学科母体"，建议通读 |
| ★ Talwai & Simchi-Levi, *Optimal Learning Rates for Regularized Least-Squares with a Fourier Capacity Condition*，[2204.07856](https://arxiv.org/abs/2204.07856) | 2022 | **math.ST/math.FA** | 用 Fourier 容量条件替代核特征值先验，免 misspecification 假设 |
| ★ Berthier, Bach, Gaillard, *Tight Nonparametric Convergence Rates for SGD under the Noiseless Linear Model*，[2006.08212](https://arxiv.org/abs/2006.08212) | 2020 | math.OC | **无噪声**设定（与本项目 5 个确定性族相同！）的 SGD 紧率；lit-survey 的"无噪声≠无 E"论证可直接引用它 |
| ★ Bodin & Macris, *Gradient flow in the gaussian covariate model: exact solution of learning curves and multiple descent*，[2212.06757](https://arxiv.org/abs/2212.06757)；及 [2110.11805](https://arxiv.org/abs/2110.11805) | 2021/22 | stat.ML | 高斯协变量模型下梯度流学习曲线的**精确解**（model/sample/epoch-wise descent 同框） |

### 2.2 double descent / benign overfitting 的精确渐近

lit-survey 已覆盖 Bartlett–Long–Lugosi–Tsigler、Hastie et al.、Mei–Montanari、Tsigler–Bartlett 主线。新增：

| 论文 | 年 | 分类 | 要点 |
|---|---|---|---|
| ★ Adamczak, Lecué, Shang, Strzelecka, *Benign Overfitting Property of the Minimum $\ell_q$-norm Interpolant*，[2610.24350](https://arxiv.org/abs/2610.24350) | 2026 | **math.ST/math.PR** | 最小 $\ell_q$ 范数插值（$1<q<\infty$）的高概率非渐近界；把"隐式偏置选哪个插值"参数化到 $q$ |
| ★ Medvedev, Vardi, Srebro, *Overfitting Behaviour of Gaussian Kernel Ridgeless Regression*，[2409.03891](https://arxiv.org/abs/2409.03891) | 2024 | cs.LG | 固定维数下高斯核 ridgeless 调带宽也不一致——支撑本项目"核 tail 决定 $A$、插值不救错配核"的判断 |
| ★ Rizzi, Wortsman, Loureiro, *Learning between the peaks: sharp asymptotics for KRR under power-law anisotropy*，[2608.28564](https://arxiv.org/abs/2608.28564) | 2026 | stat.ML | 幂律各向异性协方差下 $n=\Theta(d^\kappa)$ 的 sharp 渐近 |
| ★ Amari, Ba, Grosse, Li, Nitanda, Suzuki, *When Does Preconditioning Help or Hurt Generalization?*，[2006.10732](https://arxiv.org/abs/2006.10732) | 2020 | stat.ML | **预条件（自然梯度类）对泛化的精确影响**——本项目"Adam 无固定谱滤波[O]"之外，二阶方法的 partial answer |

### 2.3 信息下界：n-width / optimal recovery / IBC

lit-survey §6 的经典（Kolmogorov、Pinkus、Micchelli–Rivlin、Wendland）已齐。新增现代接口：

| 论文 | 年 | 分类 | 要点 |
|---|---|---|---|
| ★ Chen, Long, Wu, *A duality framework for analyzing random feature and two-layer neural networks*，[2305.05642](https://arxiv.org/abs/2305.05642) | 2023 | stat.ML | **用 IBC 工具**在 $\mathcal F_{p,\pi}$/Barron 空间上建立逼近与估计的对偶等价——本项目 §5 "两种容量"的严格化路径 |
| ★ Hsu, *Dimension lower bounds for linear approaches to function approximation*，[2508.13346](https://arxiv.org/abs/2508.13346) | 2025 | **math.ST** | 线性方法 $L^2$ 逼近的维数下界（Barron 93 论证的短注形式）——可当作 n-width 论证的教学版引用 |
| ★ Cohen & Migliorati, *Optimal weighted least-squares methods*，[1608.00512](https://arxiv.org/abs/1608.00512)，SMAI JCM 2017 | 2016 | math.NA | 从 $n$ 个（无噪）点值恢复函数、$m\le n$ 维子空间的最优加权最小二乘——optimal recovery 的算法化 |
| ★ Daubechies, DeVore, Foucart, Hanin, Petrova 已在 lit-survey；其 Acta Numerica 综述见 §1（[2012.14501](https://arxiv.org/abs/2012.14501)） | — | — | — |

### 2.4 lazy / NTK / 最小范数插值 / GP 等价

主线（Jacot、Chizat–Oyallon–Bach、Yang–Hu）已在 lit-survey。新增：

| 论文 | 年 | 分类 | 要点 |
|---|---|---|---|
| ★ Jin, Banerjee, Montúfar, *Learning curves for Gaussian process regression with power-law priors and targets*，[2110.12231](https://arxiv.org/abs/2110.12231) | 2021 | cs.LG | 幂律先验+幂律目标下 GP/KRR 学习曲线的幂律刻画——正文 05 章 GP 恒等式节可直接对接 |
| ★ Whittle 等, *Canonical Regularisation of Wide Feature-Learning Neural Networks*，[2605.18180](https://arxiv.org/abs/2605.18180) | 2026 | stat.ML | 特征学习区梯度流隐含的 canonical regulariser/prior——本项目开放问题 #1（feature-learning 区有限 $T$）的最新入口 |
| ★ Chen 等, *Generalization Bound of Gradient Flow through Training Trajectory and Data-dependent Kernel*，[2506.11357](https://arxiv.org/abs/2506.11357) | 2025 | cs.LG | 沿轨迹的数据依赖核给泛化界——lazy 判据失效时的中间路线 |

### 2.5 spectral bias / source+capacity（逆问题视角）

并入 §2.1（该视角的本质就是谱正则化文献）。另：

| 论文 | 年 | 要点 |
|---|---|---|
| ★ Bordelon–Canatar–Pehlevan 已在 lit-survey（2002.02561）；**谱偏置在真实数据上的失败与成功**：*Failure and success of the spectral bias prediction for KRR*，[2202.03348](https://arxiv.org/abs/2202.03348) | 2022 | 与本项目 dataflip 外部题库 49% 的"谱预测失效"现象直接相关，值得引 |

### 2.6 Edge of Stability（lit-survey 评级 [H] 已过时）

| 论文 | 年 | 分类 | 要点 |
|---|---|---|---|
| ★ Arora, Li, Panigrahi, *Understanding Gradient Descent on Edge of Stability in Deep Learning*，[2205.09745](https://arxiv.org/abs/2205.09745) | 2022 | math.OC | 4 层网络上 EoS 的严格证明（自稳定机制） |
| ★ Agarwala & Pennington, *High dimensional analysis reveals conservative sharpening and a stochastic edge of stability*，[2404.19261](https://arxiv.org/abs/2404.19261) | 2024 | **math.OC/math.ST** | 高维分析下 progressive sharpening 的精确模型 |
| ★ MacDonald 等, *Convergence Rates for GD on the Edge of Stability in Overparametrised Least Squares*，[2510.17506](https://arxiv.org/abs/2510.17506) | 2025 | math.OC | **过参数化最小二乘**（本项目核心设定）在 EoS 区的收敛率 |
| ★ Gan, *A Bifurcation Theory Framework for Gradient Descent on the Edge of Stability*，[2606.15551](https://arxiv.org/abs/2606.15551) | 2026 | cs.LG | 分岔理论框架，把 EoS 从低维推到一般维度 |
| Cohen et al. 2021（2103.00065）已在 lit-survey | 2021 | — | — |

---

## 3. 对本项目的增量价值（行动建议）

1. **正文 03 章（budget/curve）**：把"曲线=谱滤波"一节的主引用换成 Lin–Rudi–Rosasco–Cevher 2018 + Li 等 2026（[2609.08817](https://arxiv.org/abs/2609.08817)）；后者给出与本章"拉普拉斯变换"图像严格对应的风险等价定理。
2. **正文 03 章 EoS 小节**：评级从 [H] 上调——过参数化最小二乘下已有收敛率（[2510.17506](https://arxiv.org/abs/2510.17506)），高维随机分析有 sharp 模型（[2404.19261](https://arxiv.org/abs/2404.19261)）。lit-survey §10 同步改。
3. **02/03 章"无噪声仍有设计项"的论证**：补引 Berthier–Bach–Gaillard（[2006.08212](https://arxiv.org/abs/2006.08212)）——与本项目设定最贴（确定性目标）。
4. **05 章/§6 下界**：补 Chen–Long–Wu IBC 对偶（[2305.05642](https://arxiv.org/abs/2305.05642)）与 Hsu 短注（[2508.13346](https://arxiv.org/abs/2508.13346)）。
5. **逆问题文献作为"学科母体"**：Rastogi–Helin–Mücke 综述（[2312.15341](https://arxiv.org/abs/2312.15341)）建议通读；本项目许多"推导"在该传统里是标准定理，教材写法可以引用它们来压缩篇幅、把原创性集中在"题库证伪"这一步。
6. **新颖性声明的依据**：若将来要把本项目写成论文/投稿，"理论预测器在固定题库上零拟合打榜"（theory answerer + dev/test 纪律）在检索范围内无先例；但该判断基于 arXiv 六方向检索 + 网页检索，**未覆盖 NeurIPS/ICML 全库**，投稿前需再做一轮 venue 级查重。
