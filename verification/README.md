# verification · 机器验证与探针

> 本目录不是证明助手，而是"能跑、可复现"的检查。Lean/Coq 在本机不可用（见 `../05-formalization-and-coverage.md` §3）。

| 文件 | 作用 |
|---|---|
| `verify_symbolic.py` | 用 SymPy 精确验算 01/02 依赖的 10 条代数骨架（矩阵递推、核滤波、零空间、风险噪声项、early stopping 近似、Saxe、double descent、spectral bias） |
| `verify_symbolic_result.md` | 上述脚本的输出（当前 10/10 PASS） |
| `lazy_probe.py` | 在真实 v1.5 候选上测量训练前后 NTK 相对变化（是否 lazy）；支持 `q_11b94d` / `q_aef977` |
| `lazy_probe_q_11b94d.json` | `q_11b94d` 三个候选 × 3 seeds 的探针结果（AdamW@3e-5） |
| `lazy_probe_q_aef977.json` | `q_aef977` 三个候选 × 3 seeds 的探针结果（SGD@1e-4） |
| `bigram_estimation.py` | bigram 族测试 CE 理论的五段数值验证（恒等式/估计地板/查表 GD 优化项/秩瓶颈/bank-audit 对照）；理论见 `../notes/bigram-estimation.md` |
| `bigram_estimation_result.md` / `.json` | 上述脚本的输出（P1–P3 PASS；P4/P5 描述性） |
| `theory_answerer.py` | 理论答题器：初始化 Jacobian → 切线性化模型 → 跑官方原协议（同 seed/minibatch/优化器/步数）→ 预测 test 指标；bigram 走在线 JVP/VJP 变体；输出 `theory_answerer.jsonl` |
| `theory_answerer_eval.py` / `theory_answerer_result.md` | 与官方答案键对照：**全部 500 题 top-1 = 83.2%**（随机 33.3%），含分族/分题型/分优化器分解与错题表 |
| `m0_multiseed_result.md` | M0 多种子试点（48 题 × seeds 0–2）：argmin 翻转 0/48，**预测侧种子噪声不是瓶颈，多种子不值得全量跑** |
| `theory_answerer_meter.jsonl` / `laziness_meter_eval.py` / `laziness_meter_result.md` | M2/M3 惰性漂移计：κ_path（路径曲率，初始化可算）与真实 NTK 变化 Spearman ρ=0.855（n=21）；但分不开答对/答错（AUROC≈0.55） |
| `lazy_probe_generic.py` + `lazy_probe_generic_*.json` | 任意 qid 的真实短训 NTK 探针（支持 MSE/CE/大输出，FD 版 K·v） |
| `theory_answerer_dataflip.jsonl` / `dataflip_result.md` / `truncation_curve_eval.py` | dataflip_v1_strict 真·OOS 评测（49 题 49.0%）+ 预测截断曲线 |
| `exact_case_derivation.py` / `exact_case_result.md` | 逐条核验正文 02 章的推导链：∇L=Σ̂(θ−θ*)、θ_T−θ*=(I−ηΣ̂)^T(θ_0−θ*)、逐方向滤波、零空间冻结与极限投影、两个地板（逼近/辨识）、步长稳定性、闭式预测测试误差、超额风险公式、"平方损失=条件均值" |
| `budget_curve_derivation.py` / `budget_curve_result.md` | 核验正文 03 章：有效步长 $\eta T$ 等价、谱隙→指数弛豫（$\tau=1/(2\eta\lambda_2)$）、幂律谱→幂律曲线、EoS 阈值 $\eta\lambda_{\max}=2$、early stopping 与岭的滤波器对照、四个优化器的实测步长 |
| `generalization_stats.py` / `generalization_stats_result.md` | 核验正文 04 章：double descent 期望 $\mathbb E R_{\mathrm{te}}=\sigma^2\frac{\max(p,n)-1}{\lvert p-n\rvert-1}$（两侧同一估计器：最小范数插值器，SVD 稳定计算）、信号项 $(1-1/\gamma)\lVert\theta^\ast\rVert^2+\sigma^2n/(p-n-1)$、阈值处期望发散而单次实现有限（重尾）、配对 vs 独立 SE、测试集噪声在配对差里的抵消、验证集复用偏差 $\sqrt{2\log K}$ 上界 |

## 复现

在仓库根目录（`D:\research\Science of AI Textbook`）运行：

```powershell
python "Math of AI Textbook\verification\verify_symbolic.py"
python "Math of AI Textbook\verification\lazy_probe.py" --qid q_11b94d --seeds 0,1,2
python "Math of AI Textbook\verification\lazy_probe.py" --qid q_aef977 --seeds 0,1,2
python "Math of AI Textbook\verification\bigram_estimation.py"
python "Math of AI Textbook\verification\exact_case_derivation.py"
python "Math of AI Textbook\verification\budget_curve_derivation.py"
python "Math of AI Textbook\verification\generalization_stats.py"
python "Math of AI Textbook\verification\theory_answerer.py" --limit-per-family 2   # 冒烟
python "Math of AI Textbook\verification\theory_answerer.py" --skip-done            # 全量（可续跑）
python "Math of AI Textbook\verification\theory_answerer_eval.py"                   # 评估
```

依赖：Python 3.10、numpy、sympy、torch（本机已验证 torch 2.10.0+cu130；CPU 即可）。`bigram_estimation.py` 只需要 numpy，并在 `../notes/bank-audit.csv` 上做真实数据对照。

## 这些检查能/不能保证什么

- **能**：抓代数符号错误、矩阵方向/归一化错误、滤波指数错误、Saxe 链式法则错误、double descent 公式的符号/常数错误。
- **不能**：替代 Lean 的形式化证明；验证建模假设（lazy、无噪声、固定核）是否成立；验证 n-width/特征学习/CE 的文献定理。
- 探针只说明"某道题在某些 seeds 下不 lazy"，不能外推成"全题库都 rich"；它是否定"理论已足够全解"的证据，不是肯定"全题库都不可解"的证据。
- `ntk_rel_change` 是在 8 个随机探针方向上的相对变化；核病态时该比值会放大，故同时报告 `ntk_cosine`。两者一起看，不要单看比值。
