# theory_answerer 评估结果

> 方法：dataflip_v1_strict 外部题包；非 bigram 使用与主 500 题完全相同的 S=16 移动二阶及数值回退链，bigram 保留原一阶预测。本题包只看整体与同题配对，哈希 dev/test 不用于开发或调参。
> **out-of-sample 纪律**：预测器零拟合（没有任何系数从题库学出）。为防止「看错题改方法」的自适应污染，题库按 qid 哈希固定切成 dev/test 两半——失败分析只在 dev 半区做，**headline 数字是 test 半区准确率**。

- 候选记录：147；题目（3 候选齐全）：49
- 候选-seed 记录来源：{'bigram_initialization_linearized': 27, 'moving_second_S16': 82, 'moving_second_S16_online': 38}
- 随机基线：33.3%；KB 自报同分布约 74–88%（pack-01..04），标签天花板 98.5%

## 总体 top-1 准确率：**67.3%**（33/49）

### out-of-sample 切分（qid 哈希固定，失败分析只许在 dev 半区做）

- **test 半区（headline OOS 数字）：61.5%**（16/26）
- dev 半区（允许迭代）：73.9%（17/23）

### 与 `theory_answerer_dataflip.jsonl` 的同题配对比较

只使用两种方法都有三候选预测的题目；+ = 对照错→当前对，− = 对照对→当前错。

| 切分 | n | 对照 | 当前 | 翻转 +/− |
|---|---:|---:|---:|---:|
| 全部 | 49 | 24 | 33 | +15/−6 |
| dev | 23 | 12 | 17 | +7/−2 |
| test | 26 | 12 | 16 | +8/−4 |

## 按族

| 族 | n 题 | top-1 | 平均 Spearman ρ（预测 vs 实测） | 特征学习差中位 | 刀刃题占比 |
|---|---|---|---|---|---|
| bigram_lm | 9 | 33.3% | nan | nan | nan% |
| multivariate_regression | 10 | 100.0% | nan | nan | nan% |
| spiral_classification | 10 | 50.0% | nan | nan | nan% |
| synthetic_tabular_classification | 9 | 100.0% | nan | nan | nan% |
| univariate_regression | 10 | 60.0% | nan | nan | nan% |
| xor_classification | 1 | 0.0% | nan | nan | nan% |

## 按题型

| qtype | n 题 | top-1 |
|---|---|---|
| architecture_only | 24 | 62.5% |
| mixed | 9 | 44.4% |
| optimizer_only | 16 | 87.5% |

## 刀刃题 vs 非刀刃题

- 非刀刃题：nan%（0 题）
- 刀刃题：nan%（0 题）——理论上不可分辨，预期接近随机

## 按优化器集合

| 优化器集合 | n 题 | top-1 |
|---|---|---|
| Adagrad | 18 | 44.4% |
| Adam | 12 | 75.0% |
| SGD | 10 | 90.0% |
| AdamW | 8 | 75.0% |
| RMSprop | 1 | 100.0% |

## 错题（16 道）前 20

| qid | 族 | qtype | 预测 | 答案 | ρ | 特征学习差 | 刀刃 |
|---|---|---|---|---|---|---|---|
| q_f25ed0 | bigram_lm | mixed | C | B | nan | nan | None |
| q_fa231a | bigram_lm | mixed | B | C | nan | nan | None |
| q_f6fb1f | bigram_lm | mixed | A | B | nan | nan | None |
| q_52835c | bigram_lm | optimizer_only | B | A | nan | nan | None |
| q_3a2db2 | bigram_lm | mixed | B | C | nan | nan | None |
| q_4377aa | bigram_lm | mixed | A | C | nan | nan | None |
| q_2339c1 | spiral_classificat | architecture_only | B | C | nan | nan | None |
| q_3ae513 | spiral_classificat | architecture_only | B | A | nan | nan | None |
| q_28d0c9 | spiral_classificat | architecture_only | A | B | nan | nan | None |
| q_21e5e4 | spiral_classificat | architecture_only | A | B | nan | nan | None |
| q_0614c2 | spiral_classificat | architecture_only | B | A | nan | nan | None |
| q_11ad04 | univariate_regress | architecture_only | C | A | nan | nan | None |
| q_1640d6 | univariate_regress | architecture_only | A | B | nan | nan | None |
| q_90df63 | univariate_regress | architecture_only | A | B | nan | nan | None |
| q_2a2913 | univariate_regress | architecture_only | B | C | nan | nan | None |
| q_160ea3 | xor_classification | optimizer_only | B | C | nan | nan | None |
