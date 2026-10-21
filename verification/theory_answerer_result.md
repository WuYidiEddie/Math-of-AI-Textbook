# theory_answerer 评估结果

> 方法：每个候选 = 初始化处的切线性化模型（NTK/lazy），跑**官方原协议**（同 seed 初始化、同 minibatch 序列、同优化器配置、同步数），取其测试指标作为预测。回归/分类用显式 Jacobian（测试侧 256–512 点子采样）；bigram 用在线 FD-JVP/VJP 变体。多 seed 预测取均值。
> **out-of-sample 纪律**：预测器零拟合（没有任何系数从题库学出）。为防止「看错题改方法」的自适应污染，题库按 qid 哈希固定切成 dev/test 两半——失败分析只在 dev 半区做，**headline 数字是 test 半区准确率**。

- 候选记录：1500；题目（3 候选齐全）：500
- 跳过：{'jacobian too large': 23}
- 随机基线：33.3%；KB 自报同分布约 74–88%（pack-01..04），标签天花板 98.5%

## 总体 top-1 准确率：**83.2%**（416/500）

### out-of-sample 切分（qid 哈希固定，失败分析只许在 dev 半区做）

- **test 半区（headline OOS 数字）：85.7%**（221/258）
- dev 半区（允许迭代）：80.6%（195/242）

## 按族

| 族 | n 题 | top-1 | 平均 Spearman ρ（预测 vs 实测） | 特征学习差中位 | 刀刃题占比 |
|---|---|---|---|---|---|
| bigram_lm | 100 | 76.0% | 0.63 | 0.0648 | 0% |
| multivariate_regression | 100 | 85.0% | 0.74 | 0.3665 | 0% |
| spiral_classification | 50 | 92.0% | 0.81 | 0.0989 | 2% |
| synthetic_tabular_classification | 100 | 86.0% | 0.80 | 0.1539 | 0% |
| univariate_regression | 100 | 85.0% | 0.77 | 0.09598 | 2% |
| xor_classification | 50 | 76.0% | 0.70 | 0.2203 | 0% |

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
| Adagrad | 112 | 91.1% |
| SGD | 105 | 89.5% |
| Adam | 99 | 80.8% |
| RMSprop | 93 | 74.2% |
| AdamW | 91 | 78.0% |

## 错题（84 道）前 20

| qid | 族 | qtype | 预测 | 答案 | ρ | 特征学习差 | 刀刃 |
|---|---|---|---|---|---|---|---|
| q_479632 | xor_classification | architecture_only | A | B | 0.50 | 0.373 | False |
| q_50cd62 | univariate_regress | optimizer_only | B | C | -0.50 | 0.055 | False |
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
| q_81c9e2 | multivariate_regre | mixed | B | C | 0.50 | 0.242 | False |
| q_fb78d2 | univariate_regress | mixed | B | C | -1.00 | 0.017 | False |
| q_0525ce | multivariate_regre | architecture_only | C | A | -0.50 | 2.86 | False |
| q_8557f0 | multivariate_regre | optimizer_only | B | C | 0.50 | 17.7 | False |
| q_130e1e | multivariate_regre | optimizer_only | B | A | -0.50 | 0.117 | False |
| q_ce0a6b | multivariate_regre | optimizer_only | C | A | 0.50 | 0.25 | False |
| q_43006c | multivariate_regre | architecture_only | B | C | 0.50 | 0.367 | False |
