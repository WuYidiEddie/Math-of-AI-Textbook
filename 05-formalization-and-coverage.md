# 05 · 编码自查、机器验证，以及"这套理论够不够解 v1.5"

> 日期：2026-10-19 ｜ Math of AI Textbook
> 本文回答用户三点：(1) 文档编码自查；(2) 能否用 Lean 之类的形式化语言编译验证；(3) 现有理论是否足以解 v1.5 题目。
> 结论先行：**编码干净；Lean 在本机不可用（无工具链、无法联网安装），已用 SymPy 对可形式化的代数骨架做机器验证（10/10 PASS）；理论不足以解全部 v1.5 题——它只覆盖 lazy/kernel 区，而代表题实测已明显偏离 lazy。**

---

## 0. 结论摘要

| 事项 | 结果 |
|---|---|
| 文档编码 | 全部 UTF-8；无乱码、无控制字符、无 BOM（`bank-audit.csv` 的 BOM/CRLF 已修）；数学定界符配对 |
| Lean / Coq / Agda | **本机全部未安装**；WSL/Docker 不可用；GitHub TLS 被拦，无法安装 Lean/Mathlib |
| 可做的机器验证 | SymPy 精确符号验证 10 项（全部 PASS，脚本在 `verification/verify_symbolic.py`）；`notes/exact-formulas.md` 另有逆 Wishart、核 GD、风险公式、Saxe 的数值验证 |
| Lean 形式化 | 交付**形式化规格**（定理陈述 + Mathlib 依赖 + 证明策略）；**没有伪造"编译通过"** |
| 理论够解 v1.5 吗 | **不够解全部**。精确理论只覆盖 lazy/kernel 区；代表题实测已明显偏离 lazy（`q_11b94d`：NTK 相对变化 0.21–11.9、余弦 0.71–0.99；`q_aef977`：0.11–3.16、余弦 0.98–0.999）。可做的是"精确子集 + 主导项诊断 + 下界 + 混合预测"，不是 500 题全解 |

---

## 1. 编码自查

### 1.1 检查项与结果

对 `Math of AI Textbook/` 下全部文件（正文 01–05、`README`、`notes/`、`verification/`、`lean/`）做同样的扫描：

- UTF-8 严格解码（含 BOM 检测）；
- U+FFFD 替换字符、C0 控制字符、零宽/NBSP/软连字符；
- 常见 mojibake 标记（`Ã`、`â€`、`ï¼`、`锟斤拷` 等）；
- LaTeX 定界符配对（`\(`/`\)`、`\[`/`\]`、`$$`、代码围栏）；
- 行尾（CRLF/LF）。

**最终扫描**：0 个 warning（全部 UTF-8、无 BOM、无 CRLF、无 U+FFFD、定界符配对、代码围栏成对）。唯一修复过的问题是 `notes/bank-audit.csv` 的 BOM+CRLF（见 §1.2）；新增的 05/lean/verification 文件在写入后也统一为 UTF-8 + LF。

| 文件组 | 结果 |
|---|---|
| `README.md` / `01-formulation.md` / `02-exact-regimes.md` / `03-frontier.md` / `04-benchmark-testbed.md` / `05-formalization-and-coverage.md` | UTF-8，无 BOM，无乱码，定界符配对 |
| `notes/lit-survey.md` / `notes/exact-formulas.md` / `notes/bank-audit.md` / `notes/README.md` | UTF-8，无 BOM，无乱码 |
| `notes/bank-audit.csv` | 原为 UTF-8 **with BOM + CRLF**；已重写为 **UTF-8 no BOM + LF**（内容不变，1500 行候选） |
| `verification/*.md` / `verification/*.py` / `verification/*.json` / `lean/README.md` | UTF-8，无 BOM，LF |

### 1.2 唯一发现的问题

`notes/bank-audit.csv` 由 pandas 在 Windows 下写出，带 UTF-8 BOM 和 CRLF。部分工具会把 BOM 当首列名的一部分（例如列名变成 `\ufeffqid`），也有的 diff 工具会显示整文件改动。已修复；`notes/bank-audit.md` 等其它文件不受影响。

### 1.3 两个需要说明的"编码"点

1. **数学定界符已统一**：正文与 `notes/` 现在全部使用 `$...$`（行内）与 `$$...$$`（行间）。此前正文用 MathJax 的 `\(...\)`，部分 GUI 的 Markdown 渲染器不识别，会原样显示成 `(\mathcal X...)`；已批量转换，并验证代码块外无残留 `\(`/`\)`、`$` 配对（转换脚本逻辑：只改代码围栏/行内代码之外的文本）。
2. **原始 v1.5 数据文件里有 mojibake**：例如 `.../candidates/.../model.py` 的 docstring 出现 `鈥?`（原始生成时的编码痕迹），但**代码本身正常**。这不是本目录文档的问题；本目录没有修改 v1.5 原始数据。

---

## 2. 机器验证：SymPy 精确检查（10/10 PASS）

### 2.1 环境事实

```text
lean   => NOT FOUND
lake   => NOT FOUND
elan   => NOT FOUND
coq    => NOT FOUND
agda   => NOT FOUND
wsl    => 未安装 Linux 发行版
docker => NOT FOUND
github => TLS/schannel 凭据错误，无法访问
```

因此**无法在本机安装 Lean/Mathlib，也无法编译 Lean 文件**。任何"Lean 编译通过"的说法在这里都会是假的。

### 2.2 实际能机器验证的部分

用 SymPy 的精确有理数算术/符号化简，对 01/02 依赖的代数骨架逐条验证（`verification/verify_symbolic.py`，结果 `verification/verify_symbolic_result.md`）：

| # | 检查 | 结果 |
|---|---|---|
| 1 | 线性 GD 递推：$\theta_T-\theta^*=(I-\eta A)^T(\theta_0-\theta^*)$ | PASS |
| 2 | 零空间地板：$\theta^*\in\ker A,\theta_0=0\Rightarrow\theta_T=0$（残差 $=-\theta^*$） | PASS |
| 3 | 零空间分量不变：$P_{\rm null}(\theta_T-\theta^*)=P_{\rm null}(\theta_0-\theta^*)$ | PASS |
| 4 | 核 GD 滤波：$F_T=[I-(I-\eta K/n)^T]y$ | PASS |
| 5 | 有限 $n$ 风险：$\mathbb E\|B\varepsilon\|^2=\sigma^2\|B\|_F^2$ | PASS |
| 6 | early stopping 与 ridge **只是同尺度、非精确**（slope/half-shrinkage 两个 $\lambda_{\rm eff}$ 不同） | PASS（并给出 RMS=0.1001） |
| 7 | Saxe $L=2$：logistic 解满足 $\tau\dot a=2a(s-a)$ | PASS |
| 8 | Saxe 一般 $L$：链式法则 $dP/d\tau=\eta L P^{2-2/L}(s-P)$ | PASS |
| 9 | min-norm 噪声 excess risk：$\mathbb E[(x^\top\hat\beta)^2]=n/(p-n-1)$（MC 0.7432 vs 0.7273，4 se 内） | PASS |
| 10 | spectral bias：第 $j$ 个模式 $T$ 步后残差 $(1-\eta\lambda_j)^T a_j$ | PASS |

另有 `notes/exact-formulas.md` 中独立完成的数值验证：

- 逆 Wishart：$\mathbb E[\mathrm{tr}((XX^\top)^{-1})]$ 的 MC 1.052782 vs 公式 1.052632（$n/(p-n-1)$）；
- min-norm 风险：MC 25.019 vs 公式 25.0377（0.85 se）；$p=n+1$ 处 mean 152.96 / median 21.22，证实"期望发散、典型值有限"；
- 核 GD 闭式与迭代误差 $10^{-15}$；有限 $n$ 风险公式与 MC 差 <1 se；
- Saxe plateau 的 log-log 斜率：$L=3,4,5,6$ 为 −0.342/−0.502/−0.601/−0.667，理论 −1/3/−1/2/−3/5/−2/3。

### 2.3 没有被机器验证的部分（必须诚实）

- 逆 Wishart 的一般矩、比例极限下的 double descent 全曲线（只有渐近结果 + 小规模 MC）；
- n-width/optimal recovery 的一般函数类（标准定理，但未在此环境形式化）；
- 特征学习区的任何有限 $T$ 结论（没有定理可验证）；
- 分类 CE 的有限 $T$、Adam 的有限步理论；
- bigram 的熵地板/估计误差（可算，但本文未做）。

---

## 3. Lean 形式化：能做什么、不能做什么

### 3.1 结论

在本机**不能**编译 Lean。原因不是数学，而是环境：没有 Lean/Coq/Agda 工具链，也没有网络权限安装，WSL/Docker 不可用。

### 3.2 如果给出 Lean+Mathlib 环境，哪些定理可以首先形式化

**第一层：有限维代数（Mathlib 完全够用，估计 1–3 天/条）**

```lean
-- 说明：以下为形式化规格，不是已编译代码；需 Lean 4 + Mathlib。
-- 1) 线性 GD 递推
theorem linear_gd_recurrence
    {ι : Type*} [Fintype ι] [DecidableEq ι]
    (A : Matrix ι ι ℝ) (hA : A.IsSymm) (η : ℝ)
    (θ θs : ι → ℝ) (T : ℕ)
    (hstep : ∀ t, θt (t+1) = θt t - η • (A *ᵥ (θt t - θs))) :
    θt T - θs = (1 - η • A) ^ T *ᵥ (θt 0 - θs) := by
  sorry -- 归纳 + Matrix.mulVec_sub/mulVec_smul 等

-- 2) 核 GD 滤波
theorem kernel_gd_filter
    {ι : Type*} [Fintype ι] [DecidableEq ι]
    (K : Matrix ι ι ℝ) (hK : K.IsSymm) (η : ℝ) (y : ι → ℝ) (T : ℕ)
    (hF : ∀ t, F (t+1) = F t - (η / Fintype.card ι) • (K *ᵥ (F t - y))) :
    F T = (1 - (1 - (η / Fintype.card ι) • K) ^ T) *ᵥ y := by
  sorry

-- 3) 零空间不变性
theorem null_space_invariance
    {ι : Type*} [Fintype ι] [DecidableEq ι]
    (A : Matrix ι ι ℝ) (v : ι → ℝ) (η : ℝ) (T : ℕ)
    (hv : A *ᵥ v = 0) :
    (1 - η • A) ^ T *ᵥ v = v := by
  sorry

-- 4) 有限 n 风险的噪声项：E‖B ε‖² = σ²‖B‖_F²
--    需要概率空间/独立同分布；Mathlib 的 ProbabilityTheory 可做，但比 1–3 重。
```

**第二层：标准分析/概率（Mathlib 有基础，但要做实际工作）**

- Saxe $L=2$ 的 logistic 解：需要 `HasDerivAt` 的链式法则与 `Real.exp` 求导，可行但繁琐；
- 稳定性界（Bousquet–Elisseeff / Hardt–Recht–Singer）：需要算法稳定性、Rademacher 复杂度的 Mathlib 化，工作量大；
- n-width/optimal recovery：需要函数空间/紧算子理论，Mathlib 覆盖有限。

**第三层：目前没有 Mathlib 库、也不建议短期形式化**

- 逆 Wishart 矩、Marchenko–Pastur、BBP 相变、AMP/state evolution、replica；
- 有限宽非线性网络的特征学习动力学（mean-field PDE 的定量收敛）；
- 分类 CE 的有限 $T$ 误差；
- 一般函数类的 n-width。

### 3.3 建议的形式化路线

1. 先形式化第一层的 4 个定理（线性递推、核滤波、零空间、有限风险噪声项）——它们覆盖 01/02 的公式骨架；
2. 再形式化 Saxe $L=2$ 与一般 $L$ 的链式法则（`Real.exp`/`deriv`）；
3. 概率部分先做有限维高斯/独立同分布的 `E‖Bε‖²`；
4. n-width/double descent 暂不进入 Lean，保留为"文献定理 + 数值核对"；
5. 每个 Lean 定理与 `verification/verify_symbolic.py` 的检查编号一一对应，避免"形式化了但对象不是论文里的对象"。

### 3.4 形式化能保证什么、不能保证什么

- **能**发现代数推导错误（符号、矩阵方向、归一化、指数）；
- **不能**保证建模假设正确（比如"lazy 成立"不是 Lean 能替你验证的）；
- **不能**替代对定义/适用范围的审稿；形式化只验证"从假设到结论"的逻辑。

---

## 4. 这套理论够不够解 v1.5？

### 4.1 短答

**不够解全部 500 题。** 它够做三件事：

1. 在 **lazy/kernel 区**给出精确、可计算的验证误差（02 的公式）；
2. 给出 **n-width/optimal recovery 下界**与**主导项诊断**（A/I/O/F）；
3. 给出**相边界/可证伪预测**，并告诉你哪些题超出了理论。

它**不够**做：直接对任意候选（有限宽 MLP/TF/GRU + Adam + CE + 特征学习）算出 10-seed 测试指标并排名。

### 4.2 实测反例：代表题已经只有部分候选近似 lazy

用题面给出的真实候选代码与数据（`verification/lazy_probe.py`，3 seeds；256 步；bs=16），测量"训练前后 NTK 的相对变化"（随机探针 8 个，$\|K_Tv-K_0v\|/\|K_0v\|$）与 NTK 余弦。

**题 1：`q_11b94d`（univar，AdamW@3e-5，$T=256$）**

| 候选 | 参数量 | 参数相对位移 | NTK 相对变化 | NTK 余弦 | 实测 test MSE（初→终） |
|---|---:|---:|---:|---:|---:|
| A `c_6c95ed` | 264,961 | 2.58% | **11.85** | 0.794 | 0.352 → 0.00361 |
| B `c_4d0869` | 265,473 | 1.48% | **0.212** | **0.986** | 1.076 → 0.00455 |
| C `c_d7ace4` | 187,393 | 0.95% | **2.92** | 0.714 | 0.330 → 0.000178 |

**题 2：`q_aef977`（univar，SGD@1e-4，$T=256$）**

| 候选 | 参数量 | 参数相对位移 | NTK 相对变化 | NTK 余弦 | 实测 test MSE（初→终） |
|---|---:|---:|---:|---:|---:|
| A `c_bae746` | 187,009 | 0.76% | 0.304 | 0.988 | 8.85 → 0.0267 |
| B `c_de78d2` | 264,961 | 0.47% | **0.108** | **0.9994** | 9.54 → 0.0156 |
| C `c_ee0bd6` | 132,865 | 1.79% | 3.16 | 0.983 | 8.97 → 0.127 |

读法：

- 参数只动了 0.5–2.6%，但 **NTK 已经显著改变**：`q_11b94d` 的 A/C 相对变化 >1、余弦 0.71–0.79；`q_aef977` 的 C 相对变化 3.16；只有 B 在两个题上都接近 lazy（相对变化 0.11–0.21，余弦 0.986–0.9994）。
- 因此 02 的"固定核精确公式"对多数候选只是**核基线**，不是精确预测；对 B 这类候选是较好的近似。**不能用一个候选的 lazy 结论代表整题**。
- 两题的最终排序（`q_11b94d`：C<A<B；`q_aef977`：B<A<C）都与官方 10-seed 均值一致。初始 test MSE 的排序只在 `q_11b94d` 与最终一致（C<A<B）；`q_aef977` 的初始排序是 C<A<B，最终却是 B<A<C——**初始几何/核信号有信息，但不是万能的**。这与旧项目 `Q` 的"ρ≈0.43、只解释约 20% 秩方差"一致。
- 探针只有 3 seeds、单题；不能外推成"所有题都 rich"。但它足以否定"理论已经足够解全部题"。

### 4.3 按族的覆盖矩阵（当前理论）

| 族 | 目标 | 当前理论覆盖 | 缺什么 |
|---|---|---|---|
| univar | 光滑 1D 确定性 $f$ | lazy 时精确；非 lazy 时只能核基线 + 诊断 | 特征学习区的有限 $T$ 定量理论 |
| mvar | 加性/低阶交互，$d=2..8$ | 同上；Barron 界可估 $A$ | 同上 |
| tabular | 确定性边界（分段/加性/稀疏） | $A$ 可用逼近论粗估；CE 无闭式 | 分类 CE 有限 $T$ + 边界估计理论 |
| spiral | 2D 螺旋边界 | 深度分离给 $A$ 的方向；$O$ 只有定性 | 深度/优化联合的有限预算理论 |
| xor | 二阶交互 | $A$ 对 2 层宽网可判；$O$ 主导 | 主要是 lazy 判据 + 有效步长 |
| bigram | 随机 bigram law | **熵地板可算**；估计/优化只有部分 | 条件分布估计 + TF/GRU 有限 $T$ |

**结论**：理论覆盖的是"lazy/kernel 子集 + 诊断 + 下界"，不是全题库。按当前证据，univar/mvar 的部分 `architecture_only` 题最可能被精确公式覆盖；tabular/spiral/bigram 大概率需要缺失的理论或经验代理。

### 4.4 理论给出的、可用来"解题"的东西

即使不够全解，它给出四类可操作量：

1. **精确公式/核基线**：对每个候选，初始化 Jacobian → $K_{\rm tr},K_{\rm te}$ → 有限步预测；先与实测比，差值是 rich 项；
2. **lazy 判据**：训练前后 Jacobian/NTK 相对变化；决定"这套公式能不能用"；
3. **信息下界**：n-width/optimal recovery；判断一个候选是否已接近"信息论最优"；
4. **主导项诊断 + 刀刃判据**：A/I/O/F 哪项大；候选差是否超过 2–3 个配对 SE。

### 4.5 要"足够解题"，还缺什么

| 缺口 | 需要的工作 |
|---|---|
| 特征学习区有限 $T$ | 2 层 ReLU/quadratic 模型的可解理论；或经验残差建模 |
| Adam/AdamW | 时变预条件核理论；或数值模拟优化器（不训练网络，只模拟更新） |
| 分类 CE | margin 增长 + 边界复杂度的联合理论；有限 $T$ 无闭式 |
| bigram | 熵地板 + 条件分布估计 + 模型类逼近/优化 |
| 逐题计算 | 对 1500 候选算 NTK/特征（用 JVP/VJP，避免存 $J$）；实现 pipeline |
| 统计判定 | 配对差、10 seeds、刀刃题识别；预测排序的置信度 |
| 采样器 | 若目标是"题库分数"，还要建模候选/数据采样器（这是元学习，不是物理） |

**务实路线**：先用精确公式覆盖 lazy 子集，用 lazy 判据分层；对 rich/分类/bigram 用"理论给结构 + 经验律/实测做残差"的混合模型，并在数据集留出上评估。不要宣称"纯数学已经能解全部题"。

---

## 5. 下一步（按优先级）

1. **把 Lean 第一层 4 个定理真正编译**：需要一台有 Lean+Mathlib 的机器（本机不行）；这能锁死代数骨架。
2. **扩大 lazy 探针**：至少覆盖 6 族各 2–3 题、10 seeds；统计"懒比例"与"核预测误差 vs NTK 变化"的关系。
3. **实现逐题 kernel pipeline**：JVP/VJP 算 $K$，对候选做预测；与实测排序对照，按族/主导项分析失败。
4. **分类/bigram 单独建理论**：不要用回归公式硬套。
5. **把"理论 + 经验律"的混合评估写清楚**：纯理论能到多少、混合能到多少、哪些题只能测。

---

## 6. 一句话

**编码没问题；Lean 在本机无法编译（环境限制，不是数学问题），已用 SymPy 机器验证了 10 条代数骨架；理论不足以解全部 v1.5——它精确覆盖 lazy/kernel 子集并能做诊断/下界，但代表题实测已偏离 lazy，特征学习、Adam、分类 CE、bigram 仍是明确缺口。**
