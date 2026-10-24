# theory_answerer 评估结果

> 方法：非 bigram 固定 S=16 移动二阶；非有限依次回退移动一阶 S=16、v1/S=64、逐步一阶；大分类候选在线计算；bigram 保留原一阶预测。
> **out-of-sample 纪律**：预测器零拟合（没有任何系数从题库学出）。为防止「看错题改方法」的自适应污染，题库按 qid 哈希固定切成 dev/test 两半——失败分析只在 dev 半区做，**headline 数字是 test 半区准确率**。

- 候选记录：1500；题目（3 候选齐全）：500
- 候选-seed 记录来源：{'moving_second_S16': 973, 'fallback_first_S16_nonfinite_second': 36, 'moving_second_S16_online': 186, 'bigram_initialization_linearized': 348, 'fallback_first_S1_nonfinite_moving': 4, 'fallback_first_S_steps_effective_real_training': 1}
- 随机基线：33.3%；KB 自报同分布约 74–88%（pack-01..04），标签天花板 98.5%

## 总体 top-1 准确率：**85.4%**（427/500）

### out-of-sample 切分（qid 哈希固定，失败分析只许在 dev 半区做）

- **test 半区（headline OOS 数字）：86.8%**（224/258）
- dev 半区（允许迭代）：83.9%（203/242）

### 与 `theory_answerer.jsonl` 的同题配对比较

只使用两种方法都有三候选预测的题目；+ = 对照错→当前对，− = 对照对→当前错。

| 切分 | n | 对照 | 当前 | 翻转 +/− |
|---|---:|---:|---:|---:|
| 全部 | 500 | 416 | 427 | +50/−39 |
| dev | 242 | 195 | 203 | +26/−18 |
| test | 258 | 221 | 224 | +24/−21 |

## 按族

| 族 | n 题 | top-1 | 平均 Spearman ρ（预测 vs 实测） | 特征学习差中位 | 刀刃题占比 |
|---|---|---|---|---|---|
| bigram_lm | 100 | 76.0% | 0.63 | 0.0648 | 0% |
| multivariate_regression | 100 | 83.0% | 0.75 | 0.1247 | 0% |
| spiral_classification | 50 | 88.0% | 0.81 | 0.05246 | 2% |
| synthetic_tabular_classification | 100 | 93.0% | 0.86 | 0.0471 | 0% |
| univariate_regression | 100 | 84.0% | 0.73 | 0.04247 | 2% |
| xor_classification | 50 | 94.0% | 0.92 | 0.04603 | 0% |

## 按题型

| qtype | n 题 | top-1 |
|---|---|---|
| architecture_only | 170 | 81.2% |
| mixed | 162 | 85.8% |
| optimizer_only | 168 | 89.3% |

## 刀刃题 vs 非刀刃题

- 非刀刃题：85.5%（497 题）
- 刀刃题：66.7%（3 题）——理论上不可分辨，预期接近随机

## 按优化器集合

| 优化器集合 | n 题 | top-1 |
|---|---|---|
| Adagrad | 112 | 89.3% |
| SGD | 105 | 92.4% |
| Adam | 100 | 87.0% |
| RMSprop | 93 | 76.3% |
| AdamW | 90 | 80.0% |

## 错题（73 道）前 20

| qid | 族 | qtype | 预测 | 答案 | ρ | 特征学习差 | 刀刃 |
|---|---|---|---|---|---|---|---|
| q_b17d5a | multivariate_regre | architecture_only | B | C | 0.50 | 0.441 | False |
| q_4af9e9 | univariate_regress | optimizer_only | B | C | -0.50 | 0.0125 | False |
| q_0f9cad | univariate_regress | optimizer_only | A | B | 0.50 | 0.441 | False |
| q_e576c7 | univariate_regress | architecture_only | C | A | -0.50 | 0.0128 | False |
| q_6495ec | univariate_regress | mixed | B | C | 0.50 | 0.134 | False |
| q_11b94d | univariate_regress | architecture_only | B | C | -0.50 | 0.0446 | False |
| q_10a452 | univariate_regress | mixed | B | C | -0.50 | 0.408 | False |
| q_f85540 | univariate_regress | mixed | C | B | 0.50 | 0.0266 | False |
| q_0f686c | univariate_regress | architecture_only | C | B | -0.50 | 0.107 | False |
| q_fac596 | univariate_regress | mixed | A | C | -0.50 | 1.18 | False |
| q_0b1ee6 | univariate_regress | optimizer_only | C | B | -1.00 | 0.177 | False |
| q_c9b0f8 | multivariate_regre | optimizer_only | A | B | 0.50 | 0.345 | False |
| q_afc81d | multivariate_regre | architecture_only | B | A | -1.00 | 1.02 | False |
| q_a6c031 | univariate_regress | architecture_only | C | B | -0.50 | 0.198 | False |
| q_daafa0 | univariate_regress | architecture_only | A | C | 0.50 | 0.0482 | False |
| q_fb78d2 | univariate_regress | mixed | A | C | -0.50 | 0.0142 | False |
| q_919c58 | multivariate_regre | architecture_only | A | C | -0.50 | 0.463 | False |
| q_16bac8 | univariate_regress | architecture_only | B | C | -1.00 | 0.00956 | False |
| q_3d7396 | univariate_regress | optimizer_only | B | A | -0.50 | 1.66 | False |
| q_810e3e | multivariate_regre | mixed | B | C | 0.50 | 0.153 | False |
