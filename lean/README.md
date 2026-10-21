# lean · 形式化规格与状态

> **状态：未编译。** 本机没有 Lean/Coq/Agda 工具链，WSL/Docker 不可用，GitHub TLS 被拦（详见 `../05-formalization-and-coverage.md` §3）。
> 本目录只放"形式化规格"（定理陈述、依赖、证明策略），**不放入未编译的 `.lean` 文件冒充已验证**。

## 若你有 Lean 4 + Mathlib 环境，建议这样落地

```text
lean/
  lakefile.toml          -- require mathlib
  lean-toolchain         -- 与所用 Mathlib 匹配
  MathOfAI/Basic.lean    -- 第一层：4 个有限维代数定理
  MathOfAI/Saxe.lean     -- 第二层：Saxe L=2 与一般 L 的链式法则
  MathOfAI/Risk.lean     -- 有限维高斯噪声项 E||Bε||²=σ²||B||_F²
```

### 第一层（Mathlib 够用）

1. `linear_gd_recurrence`：$\theta_T-\theta^*=(I-\eta A)^T(\theta_0-\theta^*)$；
2. `null_space_invariance`：$Av=0\Rightarrow(I-\eta A)^Tv=v$；
3. `kernel_gd_filter`：$F_T=[I-(I-\eta K/n)^T]y$；
4. `noise_risk`：独立同分布 $\varepsilon$、协方差 $\sigma^2I$ 时 $\mathbb E\|B\varepsilon\|^2=\sigma^2\|B\|_F^2$。

证明策略：对 $T$ 归纳；用 `Matrix.mulVec_sub`、`Matrix.mulVec_smul`、`Matrix.mulVec_mulVec`、`Matrix.one_mulVec`、`Matrix.sub_mulVec`；必要处 `ext i; simp [Matrix.mulVec, Finset.sum_sub_distrib]`。风险项用独立性 + `Finset.sum_mul_sum`。

### 第二层（Mathlib 有基础，但要写）

- `saxe_two_layer`：验证 $a(t)=s/(1+\frac{s-a_0}{a_0}e^{-2st})$ 满足 $\tau\dot a=2a(s-a)$；用 `HasDerivAt`、`deriv_exp`、`deriv_const_div` 等。
- `saxe_general_chain`：纯代数链式法则 $dP/d\tau=\eta L P^{2-2/L}(s-P)$，用 `field_simp`/`ring_nf` 处理 $P^{2/L}$ 的幂。
- `stability_bound`：Bousquet–Elisseeff/Hardt–Recht–Singer 的稳定性界；需要先形式化"replace-one 稳定性"与算法，工作量大。

### 不建议短期形式化

- 逆 Wishart 矩、Marchenko–Pastur、BBP、AMP/state evolution、replica；
- n-width/optimal recovery 的一般函数类（紧算子/函数空间）；
- 有限宽非线性特征学习、Adam 有限步、分类 CE 有限 $T$。

## 与 SymPy 检查的对应

`../verification/verify_symbolic.py` 的第 1–5、7–10 项与上面第一/二层一一对应。Lean 落地时，每个 theorem 的文件注释里写对应检查编号，避免"形式化的对象不是论文里的对象"。

## 重要

- 形式化只能验证"从假设到结论"的逻辑，**不能验证建模假设**（lazy、无噪声、固定核）是否适用于某个具体候选。
- 没有编译过的 Lean 代码不能算"已验证"；本目录因此不提供 `.lean` 源文件，只提供规格。
