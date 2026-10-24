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
| `four_decomposition.py` / `four_decomposition_result.md` / `.json` | 核验正文 01 章：在 dev 半边真题 `q_11b94d` 上复刻官方协议，把 $A$/$F$/$O(T)$/$I$ 逐项测出来——平方损失 = 高斯似然（恒等式）、总体最优 = 条件均值（残差 = $\sigma^2$）、折角目标的 Runge 现象（$A$ 在训练误差里不可见）、$p>n$ 的 min-norm 零空间地板（3.224487 vs $p<n$ 的 0）、10 seeds 复算与官方 audit 对齐、学习曲线（256→4096 步，$O(T)$ 主导）、初始 NTK 谱 + 谱窗口 + 惰性极限**近**插值解（截断 SVD，含 $\mathrm{cond}(K)$）+ 核漂移、宽度扫描 $w\in\{96,192,384\}$ |
| `zero_space_init.py` / `zero_space_init_result.md` / `.json` | 核验正文 02 章 §6.4–6.5：零空间投影**是不是被强制的**——合成线性模型上逐优化器测 $\lVert P_{\rm null}\theta_T-P_{\rm null}\theta_0\rVert$（纯 GD / minibatch SGD 精确守恒到机器精度；weight decay 主动缩、实测比 $0.9552$ 对预测 $(1-\eta\lambda)^T=0.9552$；Adam 漏出行空间、漂移 $59\%$）；初始化税（训练误差看不见、测试误差洗不掉，且轨迹之差精确等于 $\Phi_{\rm te}P_{\rm null}\theta_0$）；v1.5 真题 `q_11b94d` 上 AdamW 的衰减因子、官方预算下的实测漂移（A 2.3% / C 0.95%）、惰性极限初始化税（A $1.5\times10^{-5}$ / C $4.7\times10^{-4}$，后者大于 C 的官方测试误差但特征学习把它跑破） |
| `second_order_pilot.py` / `second_order_pilot_result.md` / `.json` | **M4 试点（四次式路线）：二阶切模型能不能翻对非惰性错题**。把替身升到 $f\approx f_0+Jd+\tfrac12 d^\top Hd$（$H$ 用有限差分——SiLU 不支持 forward-mode 二阶导），在 4 道 dev 半边回归题上同 seed 跑"真实 / 一阶 / 二阶"。结果：**朴素二阶替身在官方 lr 下不稳定**（非凸四次式，约一半候选发散/爆）；官方 lr 下翻对 2 道错题（q_ab8774、q_0b5a9f）但也翻坏 1 道原答对题，调小 lr 没有一致改善，对强非惰性题 q_0525ce 仍失败、绝对预测比一阶差。**结论：朴素二阶替身不是可靠升级** |
| `second_order_trust_pilot.py` / `second_order_trust_pilot_result.md` / `.json` | **固定二阶展开的信赖域稳定化**：每步把 $d$ 投影回 $r=c\lVert d_1\rVert$（$c\in\{0.5,1,2\}$，$d_1$ 只来自一阶轨迹），不按测试误差逐候选选半径。$c=0.5$ 消除了 `NaN`，但三个统一半径的 top-1 都仍为 **2/4**，没有超过裸二阶；`q_8f876c` 即使位移有界且误差有限仍可爆到数百，说明问题不只是步长，而是固定初始化展开本身失真 |
| `moving_linearization_pilot.py` / `moving_linearization_pilot_result.md` / `.json` | **M4 试点（核修正路线 a，稳定版）：移动线性化**。沿轨迹每 steps/$S$ 步重算一次 Jacobian（刷新核），每段是**凸**线性模型，不会发散。$S$ 是从惰性（$S=1$）到真实训练（$S=$ 步数）的拨盘；终点用真实网络在 $\theta_{\rm final}$ 处求值（不训练真网）。**一致性证明**：$S=$ 步数时逐位复现真实训练（误差 $\sim10^{-6}$）。4 道 dev 半边题上 **$S=64$ 把全部 4 个赢家都翻对**（含一阶答错的 3 道），且绝对值逼近真实；$S$ 拨盘还顺带量出"每题需要多少特征学习"（q_8f876c 惰性 $S=1$ 即可，q_ab8774 需 $S\approx16$，q_0b5a9f/q_0525ce 需 $S\approx64$） |
| `moving_second_order_pilot.py` / `moving_second_order_pilot_result.md` / `.json` | **移动二阶展开：曲率 + 刷新展开点**。每段在当前 $\theta_s$ 重算 $J_s,H_s$，终点用真实网络求值，并与同 $S$ 的移动一阶逐项对比。裸二阶 top-1：$S=4/16/64$ 分别 **1/4、3/4、3/4**；加入"二阶非有限就回退同 $S$ 一阶"的无标签规则后分别 **2/4、4/4、4/4**，其中 $S=16$ 优于一阶的 2/4。**但这只是 4 道 dev 题、seed 0，且回退规则是看到这些结果后提出的；是后续验证候选，不计作已确认升级** |
| `moving_second_order_500.py` / `moving_second_order_s16_500.jsonl` / `moving_second_order_s16_500_result.md` | **完整 500 题验证上述冻结规则**：非 bigram 固定 $S=16$、seed 0，仅当最终二阶指标非有限才回退同 $S$ 一阶；大分类候选用在线 FD-JVP/VJP（与显式版对齐：二阶相对差 $1.6\times10^{-10}$、一阶 $2.0\times10^{-7}$），bigram 保留原一阶预测。36 次回退成功，5 题二阶/一阶均非有限。完整 495 题 **85.5%**；500 保守 **84.6%**。同 495 题比 v1 多对 10 题（+48/−38），test 多对 3 题（+23/−20）；但 258 题保守 test **85.7%，与 v1 持平**，不宣称 test 净提升 |
| `architecture_approximation.py` / `architecture_approximation_result.md` / `.json` | 核验正文 05 章：$A$ 的第一个度量（类内 $L^2$ 误差 = 正交投影，QR 实现；三类表达式 × 六类目标 + 固定预算下"可负担次数随维度崩塌" 119→14→4）、真实族上分离 $A$ 与 $O(T)$（spiral/xor 的 depth×width 网格 + $1/4$、$1\times$、$4\times$ 三档预算）、失败的合成对照及其分辨率地板诊断（S2b）、信息极限 $n^{-s/d}$（闭式壳计数 vs 暴力计数、实测斜率 vs 预测 $-s/d$、高斯核 $E_n=\sqrt{\lambda_{n+1}}$）、惰性区的 $A$（初始 NTK、学习窗口 $\eta\lambda_jT\gtrsim1$、最小范数权重 $\frac1n\sum_j b_j^2/\lambda_j$、GP 后验方差恒等式：解析 $1.076\times10^{-3}$ vs 实测 $2.210\times10^{-3}$）、六个族的目标画像（回归的类内逼近误差 + 分类的 LOO 岭次数曲线） |

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
python "Math of AI Textbook\verification\four_decomposition.py"          # 约 18 分钟（10 seeds × 3 候选 + 长预算曲线）
python "Math of AI Textbook\verification\four_decomposition.py" --quick  # 冒烟版（2 seeds、短预算）
python "Math of AI Textbook\verification\zero_space_init.py"             # 约 5 分钟（S1/S2 合成 + S3 真题 Jacobian）
python "Math of AI Textbook\verification\zero_space_init.py" --syn-only  # 冒烟版（只跑 S1/S2 合成，秒级）
python "Math of AI Textbook\verification\second_order_pilot.py"          # M4 试点（四次式路线）：约 3 分钟
python "Math of AI Textbook\verification\second_order_trust_pilot.py"    # 固定二阶 + 信赖域：约 3 分钟
python "Math of AI Textbook\verification\moving_linearization_pilot.py"  # M4 试点（核修正路线 a）：约 2 分钟
python "Math of AI Textbook\verification\moving_second_order_pilot.py"   # 移动二阶 + 非有限回退：约 6 分钟
python "Math of AI Textbook\verification\moving_second_order_500.py" --skip-done  # 完整 500 题，可续跑
python "Math of AI Textbook\verification\theory_answerer_eval.py" --jsonl "Math of AI Textbook\verification\moving_second_order_s16_500.jsonl" --out "Math of AI Textbook\verification\moving_second_order_s16_500_result.md" --method-note "非 bigram 固定 S=16 移动二阶，非有限回退同 S 一阶；大分类候选在线计算；bigram 保留原一阶预测。" --compare-to "Math of AI Textbook\verification\theory_answerer.jsonl"
python "Math of AI Textbook\verification\architecture_approximation.py"          # 约 15 分钟（含 xor 4x 预算的长训练）
python "Math of AI Textbook\verification\architecture_approximation.py" --quick  # 冒烟版
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
