# notes · 支撑底稿

> 本目录存放 `Math of AI Textbook` 的支撑材料。正文（01–05）是主文档；本目录不承担论证，只提供可复现的底稿。

| 文件 | 内容 | 状态 |
|---|---|---|
| [`lit-survey.md`](lit-survey.md) | 过参数化拟合 + 有限步 GD + 验证集的数学理论综述：精确/界/渐近分类、可计算量、文献清单 | ✅ 已复制（原始：`../../.team/theory-overparam/lit-survey.md`，410 行） |
| [`exact-formulas.md`](exact-formulas.md) | 独立推导与核对：线性 min-norm、核 GD 谱滤波与有限 $n$ 精确风险、Saxe plateau、稳定性界、n-width；含数值验证与待核清单 | ✅ 已复制（原始：`../../.team/theory-overparam/exact-formulas.md`，878 行） |
| [`bank-audit.md`](bank-audit.md) | v1.5 题库 testbed 描述：n、p/n、预算、确定性边界、代表题 | ✅ 已复制（原始：`../../.team/theory-overparam/bank-audit.md`） |
| [`bank-audit.csv`](bank-audit.csv) | 1500 候选级底稿（p/n、bigram 双口径、熵地板、代表题标记） | ✅ 已复制；已从 UTF-8 BOM + CRLF 规范为 UTF-8 no BOM + LF |
| [`bigram-estimation.md`](bigram-estimation.md) | bigram 族测试 CE 的 H+A+I+O(T)+F 分解：熵地板、plug-in 估计地板（渐近公式 + 精确二项）、查表 softmax GD 的凸优化界、秩瓶颈、300 候选对照 | ✅ 2026-10-20 新增；数值验证 `../verification/bigram_estimation.py`（P1–P3 PASS） |
| [`kb-explained.md`](kb-explained.md) | knowledge_base 六族经验律 → A/I/O/F 理论对象的逐条映射（已解释/部分解释/未解释 + 诚实清单） | ✅ 2026-10-20 新增；定量裁决依赖 `../verification/theory_answerer.py` |
| [`closed-loop-phenomena.md`](closed-loop-phenomena.md) | 「最小闭环」文档现象的理论解释：截断基线/等价算力、翻转题、RL 坍缩、KB 积累；含 dataflip 复算 | ✅ 2026-10-21 新增 |

**说明**：

- `.team/theory-overparam/` 是团队协作的原始工作目录；本目录只做"归档/引用"，避免正文依赖临时路径。
- 原始 `_bank_audit.py` 可复现 CSV 与报告：在仓库根目录运行 `python .team/theory-overparam/_bank_audit.py`。
- `lit-survey.md` 与 `exact-formulas.md` 已由团队交付并复制；两者中的公式分级（[E]/[U]/[L]/[A]/[H]）与正文的 exact/bound/asymptotic 标记对应。
- 正文与底稿的口径差异：正文统一用**平均损失**步长 $\eta_M$；`lit-survey.md` 的 $(1-\eta n\lambda)^T$ 是**总和损失**约定，换算为 $\eta_M=n\eta_\Sigma$。
- `exact-formulas.md` 附录 B 列了 12 条"待核"；正文引用时已避开不确定的常数与一般 $L$ 的 Saxe 原文记号。
