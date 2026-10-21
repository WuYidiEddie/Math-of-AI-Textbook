# theory_answerer 评估结果

> 日期：2026-10-20 ｜ 脚本：`theory_answerer.py`（答题）+ `theory_answerer_eval.py`（评估）
> 方法：每个候选 = 初始化处的切线性化模型（NTK/lazy），跑**官方原协议**（同 seed 初始化、同 minibatch 序列、同优化器配置、同步数），取其测试指标作为预测；seed=0 单种子。
> 实现：回归/分类用显式 Jacobian（n×C×p，测试侧 256–512 点子采样）；bigram 输出维太大，用在线 JVP（有限差分）/VJP 版本——同一切线性模型、同一协议复刻（FD 一致性相对误差 ~1e-3）。
> 覆盖：**全部 500 题**（400 非 bigram + 100 bigram）。文件里残留 23 条首跑 skipped 记录（已补齐，不影响计数）。

**结论：500 题 top-1 = 83.2%**（随机 33.3%；"选最大 lr"启发式 53.0%；KB 经验律自报 74–88%）。纯理论管线达到 KB 区间上限附近，且**零拟合**（没有任何系数是从题库里学的）。

读法：

- 分题型：optimizer_only 88.1% > mixed 82.7% > architecture_only 78.8%——优化器差异完全在 $O$ 内被模拟捕捉；架构题剩余 ~21% 误差主要来自特征学习项（线性化外），即 03 §3 的开放问题。
- 分族：spiral 92.0% / tabular 86.0% / univar 85.0% / mvar 85.0% / **bigram 76.0%** / xor 76.0%。bigram 超过了 KB 该族自测的最佳变体（69/100）与裸规则（57%）——注意 KB 数字来自不同题集划分，只作粗略参照。
- 与实测的题内 Spearman ρ 均值 0.63–0.81：预测不仅对 argmin 对，对中间排序也有信息。
- "特征学习差中位"列 = |预测 − 实测| 的题内最大差的中位数，即各族特征学习项尺度（mvar 最大 0.35 MSE；bigram 0.065 CE；spiral 最小 0.099 CE）。
- 刀刃判据（paired se）过严只标出 3 题；改进空间：用 `question.json` 的 significance.gap。

---

- 候选记录：1500；题目（3 候选齐全）：500
- 跳过（首跑残留记录，已补齐）：{'jacobian too large': 23}
- 随机基线：33.3%；KB 自报同分布约 74–88%（pack-01..04），标签天花板 98.5%

## 总体 top-1 准确率：**83.2%**（416/500）

## 按族

| 族 | n 题 | top-1 | 平均 Spearman ρ（预测 vs 实测） | 特征学习差中位 | 刀刃题占比 |
|---|---|---|---|---|---|
| bigram_lm | 100 | 76.0% | 0.63 | 0.0648 | 0% |
| multivariate_regression | 100 | 85.0% | 0.74 | 0.3537 | 0% |
| spiral_classification | 50 | 92.0% | 0.81 | 0.0989 | 2% |
| synthetic_tabular_classification | 100 | 86.0% | 0.80 | 0.1539 | 0% |
| univariate_regression | 100 | 85.0% | 0.77 | 0.09598 | 2% |
| xor_classification | 50 | 76.0% | 0.71 | 0.2203 | 0% |

## 按题型

| qtype | n 题 | top-1 |
|---|---|---|
| architecture_only | 170 | 78.8% |
| mixed | 162 | 82.7% |
| optimizer_only | 168 | 88.1% |

## 刀刃题 vs 非刀刃题

- 非刀刃题：83.1%（497 题）
- 刀刃题：100.0%（3 题）——理论上不可分辨，预期接近随机

## 按优化器集合

| 优化器集合 | n 题 | top-1 |
|---|---|---|
| Adagrad | 47 | 87.2% |
| Adam | 40 | 77.5% |
| AdamW | 36 | 72.2% |
| SGD | 33 | 87.9% |
| Adam/RMSprop/SGD | 27 | 88.9% |
| RMSprop | 26 | 61.5% |
| Adagrad/RMSprop/SGD | 22 | 86.4% |
| Adagrad/AdamW/SGD | 22 | 100.0% |
| Adagrad/Adam | 19 | 84.2% |
| AdamW/RMSprop/SGD | 19 | 89.5% |
| Adam/SGD | 18 | 94.4% |
| Adagrad/AdamW/RMSprop | 18 | 83.3% |

## 错题（84 道）前 20

| qid | 族 | qtype | 预测 | 答案 | ρ | 特征学习差 | 刀刃 |
|---|---|---|---|---|---|---|---|
| q_479632 | xor_classification | architecture_only | A | B | 0.50 | 0.403 | False |
| q_50cd62 | univariate_regress | optimizer_only | B | C | -0.50 | 0.0707 | False |
| q_e9b02d | univariate_regress | architecture_only | C | B | -1.00 | 0.158 | False |
| q_45afc9 | univariate_regress | optimizer_only | C | B | 0.50 | 0.051 | False |
| q_ab8774 | univariate_regress | architecture_only | B | A | 0.50 | 0.361 | False |
| q_11b94d | univariate_regress | architecture_only | B | C | -0.50 | 0.00265 | False |
| q_b9e43b | univariate_regress | architecture_only | B | C | 0.50 | 0.0021 | False |
| q_f85540 | univariate_regress | mixed | C | B | 0.50 | 0.0513 | False |
| q_db4008 | univariate_regress | mixed | B | C | 0.50 | 0.12 | False |
| q_cca27c | univariate_regress | architecture_only | A | C | 0.50 | 0.00756 | False |
| q_4fb324 | univariate_regress | mixed | C | B | 0.50 | 0.139 | False |
| q_8d42ff | univariate_regress | architecture_only | A | C | 0.50 | 0.096 | False |
| q_0b5a9f | univariate_regress | architecture_only | B | A | 0.50 | 0.258 | False |
| q_81c9e2 | multivariate_regre | mixed | B | C | 0.50 | 0.278 | False |
| q_fb78d2 | univariate_regress | mixed | B | C | -1.00 | 0.017 | False |
| q_0525ce | multivariate_regre | architecture_only | C | A | -0.50 | 2.86 | False |
| q_8557f0 | multivariate_regre | optimizer_only | B | C | 0.50 | 17.7 | False |
| q_130e1e | multivariate_regre | optimizer_only | B | A | -0.50 | 0.117 | False |
| q_ce0a6b | multivariate_regre | optimizer_only | C | A | 0.50 | 0.25 | False |
| q_43006c | multivariate_regre | architecture_only | B | C | 0.50 | 0.367 | False |
