# dataflip_v1_strict 真·OOS 评测与截断曲线

> 日期：2026-10-21 ｜ 题库：`../../../bench_question_packs_20261020/dataflip_v1_strict`（49 题，strict dataset-flip，10-seed 完全分离、win_rate=1.0，标签噪声≈0）
> 数据：`theory_answerer_dataflip.jsonl`（`--pack` 模式，seed 0，带 κ_path 漂移计与 21 点预算曲线）
> 性质：**真·out-of-sample**——这批题与主 500 题库不同集、未参与任何方法开发。

## 1. 总成绩

| 指标 | 值 |
|---|---|
| **top-1** | **49.0%（24/49）** |
| 主 500 题库（对照） | 83.2% |
| 10% 截断基线（实测，王泽浩） | 44–53% |
| 随机 | 33.3% |

## 2. 分族

| 族 | top-1 |
|---|---|
| spiral | **10/10** |
| mvar | **9/10** |
| xor | 1/1 |
| bigram | 3/9 |
| univar | 1/10 |
| tabular | **0/9** |

读法：

- spiral/mvar 上理论与主库一样强（这两族的翻转仍由惰性可算的量决定）；
- **tabular 0/9 是显著低于随机的**（二项 p≈2.6%）：惰性排序被特征学习**系统性反转**——不是"理论没信号"，而是"理论信号方向被 rich 动力学颠倒"。univar 1/10 同向但不显著（p≈0.10）；
- 这给出理论边界的第一张精确地图：**翻转题里，tabular/univar 的胜者由惰性世界之外的机制决定**；spiral/mvar 的胜者仍可预测。

## 3. 截断曲线（替身模拟，零额外训练）

每题取替身在预算分数 ρ 处的预测指标做 argmin（预测侧曲线）：

| ρ | 0.05 | 0.25 | 0.50 | 0.65 | 0.75 | 1.00 |
|---|---|---|---|---|---|---|
| 准确率 | 0.388 | 0.408 | 0.429 | 0.490 | 0.490 | 0.490 |

对照实测截断基线（DataFlip）：10% → 44–53%、~90% 要到 75%。**我们的惰性预测曲线与实测截断基线在同一水平（都 ~40–50%）且都平**——直接印证了王泽浩的判断：DataFlip 的胜负在训练后期才分出，早期信号（无论实测的还是理论的）都解不了。

## 4. 科学含义

1. "10% 预算选对 79.4%（main）vs 44–53%（DataFlip）"的现象 = **排序稳定时间**的分布差异：主线题的学习曲线在早期就按最终顺序排开（谱滤波：大 λ 先学，差距早定）；flip 题是特意选出的"曲线在预算内交叉"的题。
2. 答题器在 main 上的 83.2% 与在 dataflip 上的 49% 的差，量化了"惰性理论对早定型题有效、对晚翻转题失效"的边界。
3. tabular 的系统性反转（0/9）提示：该族的 rich 修正有确定方向——理解它等于理解一类特征学习。这是 M4（二阶切模型）之后最值得做的定向分析。

## 复现

```powershell
python verification/theory_answerer.py --pack ../bench_question_packs_20261020/dataflip_v1_strict --track-curve --with-meter --out verification/theory_answerer_dataflip.jsonl
python verification/truncation_curve_eval.py --jsonl verification/theory_answerer_dataflip.jsonl
```
