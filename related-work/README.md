# related-work · 相关论文存档

> 日期：2026-10-23 ｜ 检索方法：arXiv API 六方向 19 条查询（候选 205 篇去重后人工筛选）+ 网页检索数学期刊综述；逐篇核对 arXiv 编号后下载 PDF 存本目录
> 对照基准：本项目已有综述 [`../notes/lit-survey.md`](../notes/lit-survey.md)（2026-10-19，约 60 条已核实文献）；检索报告原文见 [`../notes/related-papers-2026-10-23.md`](../notes/related-papers-2026-10-23.md)
> 标注：**★ = 不在 lit-survey 中**（本次新增）；`math.*` = arXiv 数学分类（纯数学/统计期刊向论文）
> 文件名格式：`<arXiv编号>_<短标题>.pdf`，均来自 arXiv 官方 PDF。

---

## 0. 结论速览（本项目 vs 已有研究）

1. **"把训练当纯数学问题、逐项分解验证误差"的纲领已有形态各异的同类工作。** 最接近的宣言是 *There Will Be a Scientific Theory of Deep Learning*（2026-04），与本项目母项目（Science of AI Textbook）的定位几乎逐句对应——但它停留在五条研究线索的综述，没有给出本目录这种"给定 $f,\mathcal G,n,T$ 就算出 $R_{\rm te}$"的可计算判据，更没有题库证伪环节。
2. **本项目的数学骨架（谱滤波 + source/capacity + 早停）在文献里有完整成熟形态，名字叫"统计逆问题的谱正则化理论"。** Lin–Rudi–Rosasco–Cevher 2018 把 GD/ridge/PCR 统一成一族谱滤波器并证明 source+capacity 下的 minimax 最优率——正文 02/03 章（有效步长、$\lambda_{\rm eff}\asymp 1/(\eta nT)$、率 $n^{-r/(r+\beta)}$）在这一支文献里是定理级标准品。**lit-survey 只引了 Caponnetto–De Vito 与 Yao–Rosasco–Caponnetto 两篇 2007，2016 之后的谱算法文献是空白，本目录 §2.1 补上了。**
3. **"学习曲线 = 谱的（离散）拉普拉斯变换"这一精确图像已有严格等价定理**（Li–Cheng–Chen–Lin 2026，RKHS 回归与序列模型对一切 Lipschitz 谱算法风险等价）——正文 03 章图像的严格版本。
4. **n-width / optimal recovery 与学习的连接**已有 IBC 对偶框架（Chen–Long–Wu 2023）与线性方法维数下界（Hsu 2025），与 lit-survey §6 互补。
5. **Edge of Stability 的数学理论 2022–2026 显著成熟**（4 层网络严格证明、高维随机分析、过参数化最小二乘收敛率、分岔框架）；lit-survey 里仍标注为 [H]（heuristic），**该评级已过时**。
6. **没有发现与本项目完全相同组合的工作**：「a priori 可计算判据 + 四分解 $A/I/O/F$ + 用题库做零拟合证伪（theory answerer + dev/test 纪律）」。单个组件全部有主；组合在检索范围内未见先例——这是本项目真正的新颖点（但检索未覆盖 NeurIPS/ICML 全库，正式声明前需再查 venue 级文献）。

---

## 1. 同类研究纲领（综述 / 书 / manifesto）

| 本地文件 | 文献 | 与本项目的关系 |
|---|---|---|
| [2604.21691_scientific-theory-of-deep-learning.pdf](2604.21691_scientific-theory-of-deep-learning.pdf) | ★ *There Will Be a Scientific Theory of Deep Learning*（2026） | 纲领层面最像；只综述五条线索，无可计算判据、无 testbed |
| [2106.10165_principles-of-deep-learning-theory.pdf](2106.10165_principles-of-deep-learning-theory.pdf) | ★ Roberts–Yaida–Hanin, *The Principles of Deep Learning Theory*（CUP 2022） | 物理式有效理论：逐层迭代方程+近高斯修正；对象是无限宽+弱相互作用，与本项目"lazy 精确+rich 承认开放"的分层不同 |
| [2012.14501_devore-hanin-petrova_neural-network-approximation.pdf](2012.14501_devore-hanin-petrova_neural-network-approximation.pdf) | ★ DeVore–Hanin–Petrova, *Neural Network Approximation*（**Acta Numerica 30**，2021） | 纯逼近论侧（$A$ 项）最权威的数学综述；n-width、非线性逼近与网络的对比 |
| [2105.04026_modern-mathematics-of-deep-learning.pdf](2105.04026_modern-mathematics-of-deep-learning.pdf) | ★ Berner–Grohs–Kutyniok–Petersen, *The Modern Mathematics of Deep Learning*（CUP 书章） | 与本 README §0 的"三缺陷批判"同一动机；列举已知定理，不做逐题预测 |
| [2310.20360_mathematical-intro-to-deep-learning.pdf](2310.20360_mathematical-intro-to-deep-learning.pdf) | ★ Jentzen 等, *Mathematical Introduction to Deep Learning*（2023） | 教科书式全细节；偏"把工程对象写成严格数学"，不问验证误差能否 a priori 算出 |
| （无 PDF，MIT Press 2024 公开书稿） | ★ Bach, *Learning Theory from First Principles* | 核方法/谱/泛化界标准处理；本项目 0.5/0.6 章与其重叠，可作前置读物 |

**定位判断**：本项目 ≈ "Roberts–Yaida–Hanin 的可计算精神" × "谱正则化理论的严格性" × "固定题库可证伪检验"——前两者各有书级载体，第三者未见先例。

---

## 2. 按方向存档

### 2.1 谱滤波 / 核梯度流的精确理论（对应正文 02、03 章）

| 本地文件 | 文献 | 要点 |
|---|---|---|
| [1801.06720_lin-rudi-rosasco-cevher_optimal-rates-spectral-algorithms.pdf](1801.06720_lin-rudi-rosasco-cevher_optimal-rates-spectral-algorithms.pdf) | ★ Lin–Rudi–Rosasco–Cevher 2018（math.FA） | **最该补引的一篇**：GD/ridge/PCR 统一为谱滤波族，source+capacity 下高概率 minimax 率 |
| [2609.08817_risk-equivalence-rkhs-sequence-models.pdf](2609.08817_risk-equivalence-rkhs-sequence-models.pdf) | ★ Li–Cheng–Chen–Lin 2026（**math.ST**） | RKHS 回归与序列模型对 Lipschitz 谱算法风险等价 ⇒ "曲线=谱的拉普拉斯变换"严格化 |
| [2605.05768_optimal-confidence-band-kernel-gradient-flow.pdf](2605.05768_optimal-confidence-band-kernel-gradient-flow.pdf) | ★ Cheng–Chen–Lin 2026（**math.ST**） | 核梯度流的 sup 范数风险与一致置信带 |
| [2309.13337_asymptotic-learning-curves-krr-power-law.pdf](2309.13337_asymptotic-learning-curves-krr-power-law.pdf) | ★ Li–Zhang–Lin 2023（**math.ST**） | 幂律谱+幂律目标系数下 KRR 学习曲线的渐近刻画——本项目真题族假设的分布层版本 |
| [2503.00504_saturation-effects-spectral-algorithms.pdf](2503.00504_saturation-effects-spectral-algorithms.pdf) | ★ Lu–Zhang–Li–Lin 2025 | 饱和效应：目标过光滑时 KRR 达不到信息下界——"$T$ 再大也突破不了的地板"的大维版本 |
| [2007.06827_early-stopping-polynomial-smoothing-rkhs.pdf](2007.06827_early-stopping-polynomial-smoothing-rkhs.pdf) | ★ Averyanov–Celisse 2020（**math.ST**） | 数据驱动早停规则；补 Raskutti–Wainwright–Yu 2014 之后进展 |
| [1607.02387_kernel-conjugate-gradient-rates.pdf](1607.02387_kernel-conjugate-gradient-rates.pdf) | ★ Blanchard–Krämer 2016（**math.ST**） | 核 CG 早停=更激进的谱滤波；"算法改变滤波形状"的现成定理 |
| [2312.15341_statistical-inverse-learning-survey.pdf](2312.15341_statistical-inverse-learning-survey.pdf) | ★ Rastogi–Helin–Mücke 2023（**math.ST**） | 统计逆问题综述：谱正则化+随机设计——正文 02 章的"学科母体"，建议通读 |
| [2204.07856_fourier-capacity-condition.pdf](2204.07856_fourier-capacity-condition.pdf) | ★ Talwai–Simchi-Levi 2022（**math.ST/math.FA**） | Fourier 容量条件替代核特征值先验，免 misspecification 假设 |
| [2006.08212_sgd-noiseless-linear-model.pdf](2006.08212_sgd-noiseless-linear-model.pdf) | ★ Berthier–Bach–Gaillard 2020（math.OC） | **无噪声**设定（与 5 个确定性族相同）的 SGD 紧率；"无噪声≠无 E"的直接引用 |
| [2212.06757_gradient-flow-gaussian-covariate-exact.pdf](2212.06757_gradient-flow-gaussian-covariate-exact.pdf) | ★ Bodin–Macris 2022 | 高斯协变量模型下梯度流学习曲线**精确解** |
| [2110.11805_epoch-wise-descents-random-feature.pdf](2110.11805_epoch-wise-descents-random-feature.pdf) | ★ Bodin–Macris 2021 | 随机特征模型 model/sample/epoch-wise descent 同框精确解 |

### 2.2 double descent / benign overfitting 精确渐近

lit-survey 已覆盖 Bartlett–Long–Lugosi–Tsigler、Hastie et al.、Mei–Montanari、Tsigler–Bartlett 主线。新增：

| 本地文件 | 文献 | 要点 |
|---|---|---|
| [2610.24350_benign-overfitting-min-lq-norm.pdf](2610.24350_benign-overfitting-min-lq-norm.pdf) | ★ Adamczak–Lecué–Shang–Strzelecka 2026（**math.ST/math.PR**） | 最小 $\ell_q$ 范数插值（$1<q<\infty$）高概率非渐近界；"隐式偏置选哪个插值"参数化到 $q$ |
| [2409.03891_gaussian-kernel-ridgeless-bandwidth.pdf](2409.03891_gaussian-kernel-ridgeless-bandwidth.pdf) | ★ Medvedev–Vardi–Srebro 2024 | 固定维数下高斯核 ridgeless 调带宽也不一致——支撑"核 tail 决定 $A$、插值不救错配核" |
| [2608.28564_krr-power-law-anisotropy-sharp.pdf](2608.28564_krr-power-law-anisotropy-sharp.pdf) | ★ Rizzi–Wortsman–Loureiro 2026 | 幂律各向异性协方差、$n=\Theta(d^\kappa)$ 的 sharp 渐近 |
| [2006.10732_preconditioning-help-or-hurt-generalization.pdf](2006.10732_preconditioning-help-or-hurt-generalization.pdf) | ★ Amari–Ba–Grosse–Li–Nitanda–Suzuki 2020 | **预条件（自然梯度类）对泛化的精确影响**——"Adam 无固定谱滤波 [O]"之外二阶方法的 partial answer |

### 2.3 信息下界：n-width / optimal recovery / IBC（对应 lit-survey §6、正文 05 章）

| 本地文件 | 文献 | 要点 |
|---|---|---|
| [2305.05642_duality-framework-random-feature-ibc.pdf](2305.05642_duality-framework-random-feature-ibc.pdf) | ★ Chen–Long–Wu 2023 | **用 information-based complexity 工具**在 $\mathcal F_{p,\pi}$/Barron 空间建立逼近与估计的对偶等价——§5"两种容量"的严格化路径 |
| [2508.13346_dimension-lower-bounds-linear-methods.pdf](2508.13346_dimension-lower-bounds-linear-methods.pdf) | ★ Hsu 2025（**math.ST**） | 线性方法 $L^2$ 逼近维数下界（Barron 93 论证的短注版）——n-width 论证的教学版引用 |
| [1608.00512_optimal-weighted-least-squares.pdf](1608.00512_optimal-weighted-least-squares.pdf) | ★ Cohen–Migliorati 2016（SMAI JCM 2017） | $n$ 个（无噪）点值恢复函数、$m\le n$ 维子空间的最优加权最小二乘——optimal recovery 的算法化 |

### 2.4 lazy / NTK / GP 等价 / 特征学习

主线（Jacot、Chizat–Oyallon–Bach、Yang–Hu）已在 lit-survey。新增：

| 本地文件 | 文献 | 要点 |
|---|---|---|
| [2110.12231_gpr-learning-curves-power-law.pdf](2110.12231_gpr-learning-curves-power-law.pdf) | ★ Jin–Banerjee–Montúfar 2021 | 幂律先验+幂律目标下 GP/KRR 学习曲线刻画——正文 05 章 GP 恒等式节可直接对接 |
| [2605.18180_canonical-regularisation-feature-learning.pdf](2605.18180_canonical-regularisation-feature-learning.pdf) | ★ Whittle 等 2026 | 特征学习区梯度流隐含的 canonical regulariser/prior——开放问题 #1（feature-learning 有限 $T$）的最新入口 |
| [2506.11357_gradient-flow-data-dependent-kernel.pdf](2506.11357_gradient-flow-data-dependent-kernel.pdf) | ★ Chen 等 2025 | 沿轨迹的数据依赖核给泛化界——lazy 判据失效时的中间路线 |

### 2.5 谱偏置（逆问题视角并入 §2.1）

| 本地文件 | 文献 | 要点 |
|---|---|---|
| [2202.03348_spectral-bias-failure-success-krr.pdf](2202.03348_spectral-bias-failure-success-krr.pdf) | ★ *Failure and success of the spectral bias prediction for KRR* 2022 | 谱偏置预测在真实数据上的失效条件——与 dataflip 外部题库 49% 的"谱预测失效"现象直接相关 |

### 2.6 Edge of Stability（lit-survey 评级 [H] 已过时）

| 本地文件 | 文献 | 要点 |
|---|---|---|
| [2205.09745_arora-li-panigrahi-eos.pdf](2205.09745_arora-li-panigrahi-eos.pdf) | ★ Arora–Li–Panigrahi 2022（math.OC） | 4 层网络上 EoS 的严格证明（自稳定机制） |
| [2404.19261_stochastic-edge-of-stability.pdf](2404.19261_stochastic-edge-of-stability.pdf) | ★ Agarwala–Pennington 2024（**math.OC/math.ST**） | 高维分析下 progressive sharpening / 随机 EoS 的精确模型 |
| [2510.17506_eos-convergence-rates-least-squares.pdf](2510.17506_eos-convergence-rates-least-squares.pdf) | ★ MacDonald 等 2025（math.OC） | **过参数化最小二乘**（本项目核心设定）在 EoS 区的收敛率 |
| [2606.15551_bifurcation-theory-eos.pdf](2606.15551_bifurcation-theory-eos.pdf) | ★ Gan 2026 | 分岔理论框架，把 EoS 从低维推到一般维度 |

---

## 3. 对本项目的行动建议（与 notes 报告 §3 一致）

1. **正文 03 章（budget/curve）**：谱滤波主引用换成 Lin–Rudi–Rosasco–Cevher 2018 + Li 等 2026（风险等价定理与本章"拉普拉斯变换"图像严格对应）。
2. **正文 03 章 EoS 小节 + lit-survey §10**：评级从 [H] 上调——过参数化最小二乘已有收敛率，高维有 sharp 模型。
3. **02/03 章"无噪声仍有设计项"**：补引 Berthier–Bach–Gaillard 2020（设定最贴：确定性目标）。
4. **05 章 / lit-survey §6 下界**：补 Chen–Long–Wu IBC 对偶与 Hsu 短注。
5. **逆问题文献作为"学科母体"**：Rastogi–Helin–Mücke 综述建议通读；教材写法可引用谱正则化标准定理压缩篇幅，把原创性集中在"题库证伪"这一步。
6. **新颖性声明**：「理论预测器在固定题库上零拟合打榜 + dev/test 纪律」在检索范围内无先例；但检索未覆盖 NeurIPS/ICML 全库，投稿前需 venue 级查重。
