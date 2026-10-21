# v1.5 题库 testbed 描述（过参数化设定审计）

> **定位声明：本文件仅作 testbed 描述，不作理论来源。**
> 它只回答「题目把数学设定固定成了什么」（目标函数、n、test 大小、模型类、p/n、预算、seeds、指标），
> 并给出每族可直接套用理论的代表题。**不做参数量→效果方向的归纳，不给出经验规律结论。**
>
> - 数据：`a_priori/qtable/questions.csv`（500 题）、`a_priori/qtable/candidates.csv`（1500 候选）
> - 机器可读底稿：[`bank-audit.csv`](bank-audit.csv)（1500 行候选级，字段见 §5）
> - 复现脚本：[`_bank_audit.py`](_bank_audit.py)；环境：Python 3.10.19 / pandas 2.3.3 / numpy 2.2.6
> - 复现命令（仓库根目录）：`python .team/theory-overparam/_bank_audit.py`

---

## 0. 基本事实与可得文件

| 项 | 值 | 说明 |
|---|---|---|
| questions | 500 | 每题 3 个候选（`n_choices` 全为 3） |
| candidates | 1500 | 每题恰好 3 行 |
| families | univar / mvar / tabular / bigram 各 100 题；spiral / xor 各 50 题 | 共 6 族 |
| qtype | architecture_only 170 / mixed 162 / optimizer_only 168 | 决定题内哪些量固定（§1.4） |
| metric | univar、mvar: `test_mse`；tabular、xor、spiral、bigram: `test_ce` | CE 为 nats（PyTorch `F.cross_entropy` 默认均值） |
| n_seeds | 10（seed 0..9） | 每候选独立训练 10 次；`mean_metric`=10-seed 均值，`std_metric`=10-seed 标准差；`failed_seeds` 全 0 |
| 答案键 | **500/500 = 实测 10-seed 均值 argmin** | 答案键是经验标签（不是解析/理论标签） |
| budget_total | 4096 / 8192 / 16384 / 32768 | = `training_steps × batch_size` = `samples_seen`；题内所有候选共用（500/500 题 steps、bs 题内恒定） |

**原题题面（prompt.txt）全部可得**：v1.5 数据包内有 500 个 `prompt.txt`，500/500 题均可定位，题面含数据卡（目标表达式/规则、split 大小、point seed）、Sample budget、Evaluation metric（10 seeds / mean）以及三个候选的完整模型/优化器代码。
路径模板：

```
v1.5/v15-500q-6d4ae9b/data/datasets/<family_dir>/<ds>/questions/<run_dir>/<qid>/prompt.txt
family → family_dir: univar=univariate_regression, mvar=multivariate_regression,
                     tabular=synthetic_tabular_classification, xor=xor_classification,
                     spiral=spiral_classification, bigram=bigram_lm
```

数据实例与随机性：500 题 = 500 个独立的数据实例（univar/mvar 有各自的 point seed；bigram 有 sequence seed + table seed；tabular/xor 有 point seed + 4096 行 calibration seed；spiral 有 point seed）。训练时 minibatch 从固定训练点中**有放回**抽样（题面原文 "with replacement"）。

---

## 1. 题目设定统计

### 1.1 各族数据与预算设定（题级常量）

| fam | n_q | n_train | n_test | metric | budget_total | steps 取值 | bs 取值 | model_type（候选数） |
|---|---|---|---|---|---|---|---|---|
| univar | 100 | 256 | 256 | test_mse | 4096–32768 | 256/512/1024/2048 | 16/32/64 | mlp (300) |
| mvar | 100 | 256 | 256 | test_mse | 4096–32768 | 256/512/1024/2048 | 16/32/64 | mlp (300) |
| tabular | 100 | 1024 | 2048 | test_ce | 4096–32768 | 256/512/1024/2048 | 16/32/64 | mlp (300) |
| xor | 50 | 1024 | 2048 | test_ce | 4096–32768 | 256/512/1024/2048 | 16/32/64 | mlp (150) |
| spiral | 50 | 1024 | 2048 | test_ce | 4096–32768 | 256/512/1024/2048 | 16/32/64 | mlp (150) |
| bigram | 100 | 800 序列 | 200 序列 | test_ce | 4096–32768 | 256/512/1024/2048 | 16/32/64 | transformer_lm (217) / gru_lm (83) |

题级分布：budget_total 每个取值在 100 题族各 25 题、在 spiral/xor 各 13/13/12/12 题；steps 题数 {256: 255, 512: 115, 1024: 91, 2048: 39}；bs 题数 {16: 250, 32: 168, 64: 82}。9 种 (steps,bs) 组合及题数：256×16:126、256×32:85、256×64:44、512×16:41、512×32:36、512×64:38、1024×16:44、1024×32:47、2048×16:39。

### 1.2 trainable_params 分布（候选级）

| fam | n | min | p25 | median | p75 | max | max/min |
|---|---|---|---|---|---|---|---|
| univar | 300 | 321 | 3,265 | 17,057 | 74,689 | 331,777 | 1034 |
| mvar | 300 | 625 | 3,457 | 17,409 | 67,361 | 332,801 | 532.5 |
| tabular | 300 | 610 | 3,254 | 12,626 | 68,610 | 333,314 | 546.4 |
| xor | 150 | 722 | 4,498 | 13,122 | 62,882 | 335,362 | 464.5 |
| spiral | 150 | 386 | 2,714 | 13,058 | 67,394 | 332,802 | 862.2 |
| bigram | 300 | 7,896 | 41,076 | 68,232 | 140,848 | 538,136 | 68.15 |

### 1.3 p/n 与 p>n 覆盖（p = trainable_params，n = ds_train_size）

| fam | n_train | p/n min | p/n median | p/n p90 | p/n max | 候选 p>n | 三候选全 p>n 的题数 | 存在 p≤n 的题数 |
|---|---|---|---|---|---|---|---|---|
| univar | 256 | 1.254 | 66.63 | 601.4 | 1296 | 300/300 (100.0%) | 100/100 | 0 |
| mvar | 256 | 2.441 | 68.00 | 782.0 | 1300 | 300/300 (100.0%) | 100/100 | 0 |
| tabular | 1024 | 0.596 | 12.33 | 184.1 | 325.5 | 290/300 (96.7%) | 95/100 | 5 |
| xor | 1024 | 0.705 | 12.81 | 113.2 | 327.5 | 145/150 (96.7%) | 48/50 | 2 |
| spiral | 1024 | 0.377 | 12.75 | 146.1 | 325.0 | 139/150 (92.7%) | 46/50 | 4 |
| bigram | 800 序列 | 9.870 | 85.29 | 345.6 | 672.7 | 300/300 (100.0%) | 100/100 | 0 |

全体 1500 候选：p>n 比例 = **1474/1500 = 98.27%**；三候选全 p>n 的题 = **489/500**；p/n 全局 min=0.377、median=46.48、max=1300。
→ 这 11 道「存在 p≤n 候选」的题全部在 spiral/tabular/xor（p/n 最小值 0.377 在 spiral），univar/mvar/bigram 三族在序列口径下 100% p>n。

### 1.4 qtype 结构：题内哪些量固定/变化

| qtype | 题数 | opt 不同数/题 | lr 不同数/题 | steps/bs 题内不同数 | 参数量不同数/题 | model_type 不同数/题 |
|---|---|---|---|---|---|---|
| architecture_only | 170 | {1:170} | {1:170} | 1 | {1:2, 2:38, 3:130} | {1:161, 2:9} |
| mixed | 162 | {1:9, 2:68, 3:85} | {1:5, 2:82, 3:75} | 1 | {1:2, 2:29, 3:131} | {1:149, 2:13} |
| optimizer_only | 168 | {1:3, 2:76, 3:89} | {1:7, 2:81, 3:80} | 1 | {1:168} | {1:168} |

各族 qtype 构成（题数）：

| fam | architecture_only | mixed | optimizer_only |
|---|---|---|---|
| univar | 34 | 33 | 33 |
| mvar | 34 | 33 | 33 |
| tabular | 34 | 33 | 33 |
| xor | 17 | 16 | 17 |
| spiral | 17 | 16 | 17 |
| bigram | 34 | 31 | 35 |

解读（口径，不是结论）：`architecture_only` 题内三候选的 opt/lr/steps/bs 完全一致，只有模型架构（及 p）变化；`optimizer_only` 题内模型完全一致（p 相同），只变 opt/lr；`mixed` 两者都变。qs 里区分三者，便于把理论套到受控子集。

### 1.5 模型类与优化器空间（候选级覆盖）

- **MLP**（1200 候选，mvar/spiral/tabular/univar/xor）：depth ∈ {1,2,3,4,5}；width ∈ {16,24,32,48,64,96,128,192,256}；act ∈ {silu 325, leaky_relu 296, relu 293, gelu 286}；residual ∈ {False 606, True 594}；layer_norm 为逐层 True/False 开关（62 种组合）；in_dim ∈ {1,2,3,4,5,8,16}；out_dim ∈ {1（回归）, 2（分类）}。
- **LM**（300 候选，bigram）：transformer_lm 217 / gru_lm 83；num_layers ∈ {1,2,3,4}；d_model ∈ {32,64,128}；d_ff ∈ {64,128,256}；heads ∈ {2,4}；vocab ∈ {24,32,48}；ctx_len ∈ {12,16,24}。
- **优化器**（候选级）：Adagrad 342、SGD 315、Adam 299、AdamW 279、RMSprop 265。
- **lr**：3e-5 328、1e-4 305、3e-4 316、1e-3 264、3e-3 287。
- **weight decay**：0.0 397、1e-5 366、1e-4 363、1e-3 374。
- **momentum**（部分优化器）：0.9 723、0.0 170、无该字段 607。

### 1.6 目标函数与标签噪声设定（确定性边界）

| fam | metric | n_train | n_test | 目标 f 性质 | 输入采样 | 标签/噪声 |
|---|---|---|---|---|---|---|
| univar | test_mse | 256 | 256 | 符号表达式**精确求值** y=f(x) | x~Uniform[0,1]，固定 point_seed 物化 train/test | 确定性回归，无观测/标签噪声 |
| mvar | test_mse | 256 | 256 | 符号表达式**精确求值** y=f(x) | x~Uniform[0,1]^d（d=2..8），固定 point_seed | 确定性回归，无观测/标签噪声 |
| tabular | test_ce | 1024 | 2048 | latent score s(x) 过阈值；确定性规则（smooth_additive / sparse_interaction / piecewise_boundary） | x~N(0,I_d)（d=2..16） | 确定性二分类，无标签噪声 |
| xor | test_ce | 1024 | 2048 | 活跃坐标符号一致性（XOR）；确定性规则标签，其余坐标为干扰 | x~N(0,I_d)（d=4..16） | 确定性二分类，无标签噪声 |
| spiral | test_ce | 1024 | 2048 | 两条阿基米德螺旋臂决定类别；确定性规则标签、类平衡 | 沿螺旋臂采样连续点 | 确定性二分类，无标签噪声 |
| bigram | test_ce | 800 序列 | 200 序列 | **固定 bigram 转移矩阵 P(y\|x) 的随机下一 token** | 按 P(y\|x) 采样 token 序列 | **随机目标（非确定性），不可约熵 > 0** |

分类族的 `ds_calibration` 字段是**阈值/类平衡校准**，不是标签噪声：

| fam | calibration distribution | calibration size | realized_positive_rate min/median/max | 题数 |
|---|---|---|---|---|
| spiral | none | 0 | — | 0（题面写明 classes exactly balanced, no threshold to calibrate） |
| xor | standard_normal | 4096 | 0.4836 / 0.4997 / 0.5166 | 50 |
| tabular | standard_normal | 4096 | 0.4749 / 0.4997 / 0.5234 | 100 |

**bigram 反例/边界（重要）**：bigram 的目标是一个固定的随机 bigram law，不是确定性函数。
train 是 800 条序列、每条长 ctx+1（ctx∈{12,16,24}），产生 `ctx` 个 next-token 标签；test 是 200 条序列。
所以 n 有三种口径：**800 序列**（p/n 中位数 85.29、100% p>n）/ **800×ctx = 9600–19200 个 token 标签**（p/n 中位数 5.47、94.0% 候选 p>n、94/100 题三候选全 p>n）/ 窗口重叠与马尔可夫依赖使语义上的有效独立样本量介于两者之间。
不可约 CE 地板由 P(y|x) 的熵给出：`H(Y|X) = Σ_x π_x H(P(·|x))`，可由 CSV 的 `ds_table_seed`、`ds_vocab_size`、`ds_alpha` 精确复算。

| 量 | min | median | max |
|---|---|---|---|
| H(Y\|X) 不可约 CE（nats） | 2.281 | 2.973 | 3.604 |
| 实测 test_ce（候选级） | 2.366 | 3.328 | 3.971 |
| 实测 CE / 地板 | 1.021 | 1.093 | 1.566 |

候选 CE 全部 > 地板（0 个候选低于 H(Y|X)）；即 bigram 的指标在数学上存在正的下界，与「确定性目标 + 零噪声」的设定不同。

---

## 2. 代表题（每族 1–2 道，完整可套用）

选取规则：univar/mvar 从 `architecture_only` 子集（三候选 opt/lr/steps/bs 全同，仅架构/p 变化）中取 max trainable_params 最大的 2 道；tabular/xor/spiral/bigram 各取 1 道。8 题均为 500 题之一，完整题面可从 §0 的路径模板按 qid 查到。

### 2.1 univar_r1 — `q_11b94d`

- qtype=`architecture_only`；n_train=256，n_test=256；metric=`test_mse`；budget=4096（steps=256, bs=16）；10 seeds。
- 目标：`y = tanh(2*x)**3 + sin(2*pi*x)`（x~Uniform[0,1]，精确求值，无噪声）。
- 官方答案 **C**；实测 argmin(mean_metric) = **C**（一致）。
- 题面：`v1.5/v15-500q-6d4ae9b/data/datasets/univariate_regression/sym_4d5411/questions/run_1q_3c_644fe3/q_11b94d/prompt.txt`

| letter | 模型配置 | params | opt / lr / wd / mom | steps×bs | mean_metric | std_metric | p/n |
|---|---|---|---|---|---|---|---|
| A | depth=4, width=256, act=leaky_relu, layer_norm=FTTF | 264,961 | AdamW / 3e-5 / 1e-5 / 0.9 | 256×16 | 0.003564 | 0.002734 | 1035 |
| B | depth=4, width=256, act=relu, residual=True, layer_norm=TFTT | 265,473 | AdamW / 3e-5 / 1e-5 / 0.9 | 256×16 | 0.003646 | 0.002678 | 1037 |
| C | depth=5, width=192, act=leaky_relu, layer_norm=FTTTT | 187,393 | AdamW / 3e-5 / 1e-5 / 0.9 | 256×16 | 0.0002226 | 8.237e-5 | 732 |

目标方差（`a_priori/univar_dataset_scale.json`）：σ²_test=0.21082 → 最优 MSE 对应 R²≈0.9989，最差 MSE 对应 R²≈0.9827。

### 2.2 univar_r2 — `q_aef977`

- qtype=`architecture_only`；n_train=256，n_test=256；metric=`test_mse`；budget=4096（steps=256, bs=16）；10 seeds。
- 目标：`y = x**3 - 0.5*x + 3`（x~Uniform[0,1]，精确求值，无噪声）。
- 官方答案 **B**；实测 argmin = **B**（一致）。
- 题面：`v1.5/v15-500q-6d4ae9b/data/datasets/univariate_regression/sym_801402/questions/run_1q_3c_202709/q_aef977/prompt.txt`

| letter | 模型配置 | params | opt / lr / wd / mom | steps×bs | mean_metric | std_metric | p/n |
|---|---|---|---|---|---|---|---|
| A | depth=5, width=192, act=gelu, layer_norm=FTTFT | 187,009 | SGD / 1e-4 / 1e-5 / 0 | 256×16 | 0.02862 | 0.003062 | 730.5 |
| B | depth=4, width=256, act=relu, residual=True, layer_norm=FTTF | 264,961 | SGD / 1e-4 / 1e-5 / 0 | 256×16 | 0.01495 | 0.003272 | 1035 |
| C | depth=2, width=256, act=leaky_relu, layer_norm=TF | 132,865 | SGD / 1e-4 / 1e-5 / 0 | 256×16 | 0.1557 | 0.1094 | 519 |

目标方差：σ²_test=0.022254 → 最优 MSE 对应 R²≈0.328，最差 MSE 对应 R²≈−5.997（小 σ² 放大 MSE 的跨题尺度差异，比较时需注意口径）。

### 2.3 mvar_r1 — `q_094c35`

- qtype=`architecture_only`；n_train=256，n_test=256；metric=`test_mse`；budget=4096（steps=256, bs=16）；10 seeds；input_dim=8。
- 目标：`y = x0**6 + cos(2*pi*x0)*sin(2*pi*x2) + cos(2*pi*x4)*sin(2*pi*x0) + abs(cos(2*pi*x7)) + cos(2*pi*x2**3) + cos(2*pi*x1) + tanh(2*x5) + x3 + 0.5*x4 + x6 + 3`（x~Uniform[0,1]^8，精确求值，无噪声）。
- 官方答案 **C**；实测 argmin = **C**（一致）。
- 题面：`v1.5/v15-500q-6d4ae9b/data/datasets/multivariate_regression/mvar_d66160/questions/run_1q_3c_b7f197/q_094c35/prompt.txt`

| letter | 模型配置 | params | opt / lr / wd / mom | steps×bs | mean_metric | std_metric | p/n |
|---|---|---|---|---|---|---|---|
| A | depth=4, width=256, act=silu, layer_norm=TFTF | 266,753 | Adam / 1e-4 / 1e-5 / 0.9 | 256×16 | 1.441 | 0.04552 | 1042 |
| B | depth=4, width=256, act=silu, residual=True, layer_norm=FFFT | 266,241 | Adam / 1e-4 / 1e-5 / 0.9 | 256×16 | 1.460 | 0.05601 | 1040 |
| C | depth=5, width=256, act=relu, residual=True, layer_norm=FTFTF | 332,545 | Adam / 1e-4 / 1e-5 / 0.9 | 256×16 | 1.275 | 0.08415 | 1299 |

### 2.4 mvar_r2 — `q_fe5575`

- qtype=`architecture_only`；n_train=256，n_test=256；metric=`test_mse`；budget=16384（steps=1024, bs=16）；10 seeds；input_dim=2。
- 目标：`y = tanh(2*x0**2)**3 + x0*x1**2 + abs(cos(2*pi*sin(2*pi*x1)))`（x~Uniform[0,1]^2，精确求值，无噪声）。
- 官方答案 **C**；实测 argmin = **C**（一致）。
- 题面：`v1.5/v15-500q-6d4ae9b/data/datasets/multivariate_regression/mvar_11b438/questions/run_1q_3c_4e21c4/q_fe5575/prompt.txt`

| letter | 模型配置 | params | opt / lr / wd / mom | steps×bs | mean_metric | std_metric | p/n |
|---|---|---|---|---|---|---|---|
| A | depth=4, width=256, act=gelu, layer_norm=TFFF | 264,705 | AdamW / 3e-5 / 1e-4 / 0.9 | 1024×16 | 0.1137 | 0.003367 | 1034 |
| B | depth=5, width=256, act=gelu, residual=True | 329,985 | AdamW / 3e-5 / 1e-4 / 0.9 | 1024×16 | 0.1055 | 0.002944 | 1289 |
| C | depth=5, width=256, act=leaky_relu, layer_norm=FTFTT | 331,521 | AdamW / 3e-5 / 1e-4 / 0.9 | 1024×16 | 0.09518 | 0.00481 | 1295 |

### 2.5 tabular_r1 — `q_12016f`

- qtype=`mixed`（三候选的 opt/lr 不同，steps/bs 相同）；n_train=1024，n_test=2048；metric=`test_ce`；budget=16384（steps=512, bs=32）；10 seeds；input_dim=2。
- 规则：`rule_family=smooth_additive`，active features x0,x1；latent score `s(x)=0.625*(sin(x0)+0.25*x0^2)+0.875*(sin(x1)+0.25*x1^2)`，标签 `y=1{s(x)>0.25}`（确定性，无标签噪声；阈值 0.25 由 4096 行校准集定在 s(x) 中位数附近，校准集上正类比例 0.519）。
- 官方答案 **B**；实测 argmin = **B**（一致）。
- 题面：`v1.5/v15-500q-6d4ae9b/data/datasets/synthetic_tabular_classification/stabcls_011e2b/questions/run_1q_3c_8a1801/q_12016f/prompt.txt`

| letter | 模型配置 | params | opt / lr / wd / mom | steps×bs | mean_metric | std_metric | p/n |
|---|---|---|---|---|---|---|---|
| A | depth=2, width=16, act=gelu, layer_norm=TT | 690 | Adagrad / 3e-5 / 1e-5 / — | 512×32 | 0.7221 | 0.0486 | 0.674 |
| B | depth=2, width=16, act=gelu, residual=True | 626 | AdamW / 3e-3 / 1e-3 / 0.9 | 512×32 | 0.03298 | 0.005283 | 0.611 |
| C | depth=4, width=16, act=relu, layer_norm=TFFF | 1202 | SGD / 3e-5 / 0 / 0 | 512×32 | 0.6993 | 0.006588 | 1.174 |

### 2.6 xor_r1 — `q_09e52b`

- qtype=`architecture_only`；n_train=1024，n_test=2048；metric=`test_ce`；budget=8192（steps=256, bs=32）；10 seeds；input_dim=16。
- 规则：`rule_family=xor`，active features x4,x10；latent score `s(x) = -x4*x10`，标签 `y=1{s(x)>0}`（确定性，无标签噪声；无关坐标为干扰；阈值 0 由 4096 行校准集定在中位数附近，校准集正类比例 0.5129）。
- 官方答案 **B**；实测 argmin = **B**（一致）。
- 题面：`v1.5/v15-500q-6d4ae9b/data/datasets/xor_classification/xorcls_00f53b/questions/run_1q_3c_4fdefe/q_09e52b/prompt.txt`

| letter | 模型配置 | params | opt / lr / wd / mom | steps×bs | mean_metric | std_metric | p/n |
|---|---|---|---|---|---|---|---|
| A | depth=5, width=192, act=leaky_relu, residual=True, layer_norm=FTFFT | 189,698 | Adam / 3e-4 / 1e-3 / 0.9 | 256×32 | 0.4189 | 0.06276 | 185.3 |
| B | depth=4, width=256, act=gelu, residual=True, layer_norm=FFTF | 268,546 | Adam / 3e-4 / 1e-3 / 0.9 | 256×32 | 0.2659 | 0.0184 | 262.3 |
| C | depth=4, width=256, act=leaky_relu, residual=True, layer_norm=TTFF | 269,058 | Adam / 3e-4 / 1e-3 / 0.9 | 256×32 | 0.3806 | 0.04019 | 262.8 |

### 2.7 spiral_r1 — `q_144ad6`

- qtype=`mixed`（opt/lr 不同）；n_train=1024，n_test=2048；metric=`test_ce`；budget=32768（steps=512, bs=64）；10 seeds；input_dim=2。
- 规则：`rule_family=spiral`，两条缠绕 Archimedean 螺旋臂，turns=2；标签=点来自哪条臂（确定性，无标签噪声；两类精确平衡，calibration 为 none）。
- 官方答案 **A**；实测 argmin = **A**（一致）。
- 题面：`v1.5/v15-500q-6d4ae9b/data/datasets/spiral_classification/spiralcls_0c5eaa/questions/run_1q_3c_f35f53/q_144ad6/prompt.txt`

| letter | 模型配置 | params | opt / lr / wd / mom | steps×bs | mean_metric | std_metric | p/n |
|---|---|---|---|---|---|---|---|
| A | depth=4, width=192, act=leaky_relu, residual=True, layer_norm=FFTF | 149,570 | Adagrad / 3e-3 / 1e-4 / — | 512×64 | 0.1639 | 0.04194 | 146.1 |
| B | depth=2, width=256, act=gelu, residual=True, layer_norm=TT | 133,890 | Adagrad / 3e-5 / 0 / — | 512×64 | 0.6727 | 0.06739 | 130.8 |
| C | depth=3, width=192, act=leaky_relu | 112,130 | Adagrad / 3e-5 / 1e-5 / — | 512×64 | 0.6769 | 0.001337 | 109.5 |

### 2.8 bigram_r1 — `q_235e7e`

- qtype=`architecture_only`；n_train=800 序列（=800×16=12800 个 next-token 标签），n_test=200 序列（=3200 token 标签）；metric=`test_ce`；budget=8192（steps=512, bs=16）；10 seeds。
- 目标：固定 bigram 转移矩阵 P(y|x)，vocab=32，ctx=16，alpha=0.8，table_seed=310777；不可约 CE 地板 H(Y|X)=**3.196 nats**。
- 官方答案 **B**；实测 argmin = **B**（一致）。
- 题面：`v1.5/v15-500q-6d4ae9b/data/datasets/bigram_lm/bg_03a220/questions/run_1q_3c_e4192f/q_235e7e/prompt.txt`

| letter | 模型配置 | params | opt / lr / wd / mom | steps×bs | mean_metric | std_metric | p/800 序列 | p/12800 token 标签 |
|---|---|---|---|---|---|---|---|---|
| A | transformer: layers=4, d_model=32, d_ff=256, heads=4 | 86,688 | SGD / 1e-3 / 0 / 0 | 512×16 | 3.582 | 0.01633 | 108.4 | 6.77 |
| B | gru: layers=2, d_model=64, residual=False | 54,048 | SGD / 1e-3 / 0 / 0 | 512×16 | 3.470 | 0.003044 | 67.56 | 4.22 |
| C | transformer: layers=3, d_model=64, d_ff=64, heads=4 | 80,800 | SGD / 1e-3 / 0 / 0 | 512×16 | 3.579 | 0.01266 | 101.0 | 6.31 |

注：实测 CE/地板 ≈ 1.121 / 1.086 / 1.120（A/B/C）。

### 2.9 代表题汇总

| rep | qid | fam | qtype | n_train | n_test | metric | budget | 官方答案 | argmin(实测) 一致? |
|---|---|---|---|---|---|---|---|---|---|
| univar_r1 | q_11b94d | univar | architecture_only | 256 | 256 | test_mse | 4096 | C | YES |
| univar_r2 | q_aef977 | univar | architecture_only | 256 | 256 | test_mse | 4096 | B | YES |
| mvar_r1 | q_094c35 | mvar | architecture_only | 256 | 256 | test_mse | 4096 | C | YES |
| mvar_r2 | q_fe5575 | mvar | architecture_only | 256 | 256 | test_mse | 16384 | C | YES |
| tabular_r1 | q_12016f | tabular | mixed | 1024 | 2048 | test_ce | 16384 | B | YES |
| xor_r1 | q_09e52b | xor | architecture_only | 1024 | 2048 | test_ce | 8192 | B | YES |
| spiral_r1 | q_144ad6 | spiral | mixed | 1024 | 2048 | test_ce | 32768 | A | YES |
| bigram_r1 | q_235e7e | bigram | architecture_only | 800 序列 | 200 序列 | test_ce | 8192 | B | YES |

---

## 3. 验证集/种子测量误差量级（设定事实）

`mean_metric` 是 10 个 seed 的均值；`std_metric` 是 10 个 seed 的样本标准差。**同一个固定 test split 被所有候选共用**，所以这里没有 test-set 重采样噪声：`std_metric` 来自初始化/minibatch 随机性。10-seed 均值的标准误 ≈ std/√10；两候选均值差的噪声尺度（若独立）≈ √2·std。

### 3.1 std_metric 分布（候选级，按族）

| fam | metric | n | mean_metric median | std_median | std_p90 | std_max | std/\|mean_metric\| median | SE(10-seed mean) median |
|---|---|---|---|---|---|---|---|---|
| univar | test_mse | 300 | 0.04564 | 0.006463 | 0.07896 | 0.2893 | 0.3312 | 0.002044 |
| mvar | test_mse | 300 | 0.6217 | 0.03603 | 0.1173 | 0.3681 | 0.07296 | 0.01139 |
| tabular | test_ce | 300 | 0.3595 | 0.01479 | 0.04093 | 0.1404 | 0.05686 | 0.004677 |
| xor | test_ce | 150 | 0.4929 | 0.01476 | 0.04074 | 0.1288 | 0.04418 | 0.004668 |
| spiral | test_ce | 150 | 0.6196 | 0.01109 | 0.05965 | 0.3058 | 0.03247 | 0.003506 |
| bigram | test_ce | 300 | 3.328 | 0.009824 | 0.01989 | 0.04538 | 0.003049 | 0.003107 |

按 metric 汇总：`test_ce`（900 候选）std_metric median 0.01237 / p90 0.03473 / max 0.3058；`test_mse`（600 候选）median 0.02127 / p90 0.1083 / max 0.3681。

### 3.2 题内两候选 10-seed 均值差 |Δmean_metric| 的量级（事实尺度）

| fam | metric | 题内 \|Δmean\| median | 题内 \|Δmean\| p90 | std_metric median（=1 个测量误差单位） | \|Δ\|/pair_std median | \|Δ\|>1 pair_std 的候选对 |
|---|---|---|---|---|---|---|
| univar | test_mse | 0.0591 | 0.458 | 0.006463 | 3.63 | 88.3% |
| mvar | test_mse | 0.2202 | 0.6568 | 0.03603 | 4.15 | 91.0% |
| tabular | test_ce | 0.1361 | 0.4979 | 0.01479 | 6.84 | 92.7% |
| xor | test_ce | 0.1281 | 0.5416 | 0.01476 | 6.11 | 89.3% |
| spiral | test_ce | 0.1077 | 0.5145 | 0.01109 | 5.45 | 91.3% |
| bigram | test_ce | 0.0947 | 0.3790 | 0.009824 | 8.96 | 94.0% |

（`pair_std = sqrt((std_a²+std_b²)/2)`，逐候选对计算。上表只是把「1 个 std_metric」与 |Δ| 的尺度并列，**不构成方向性结论**。）

---

## 4. 口径边界与注意事项（套理论时必读）

1. **p/n 的 n 是固定训练点数量，不是训练预算**。`n = ds_train_size`（univar/mvar 256；tabular/xor/spiral 1024；bigram 800 序列），而 `samples_seen = budget_total = steps×bs ∈ {4096, 8192, 16384, 32768}`。所以同一道题可以同时是「p>n」和「p < samples_seen」：例如 univar 最小 p=321 已经 > n=256，但远小于最小 budget=4096。若要套「p>n 有限步 GD」，需先明确理论里的 n 指训练点数还是迭代里见过的样本数。
2. **bigram 的 n 口径必须声明**：800 条序列（每条产生 ctx=12/16/24 个 next-token 标签）→ 序列口径 100% p>n（p/n median 85.29）；token 标签口径 n=800×ctx=9600–19200 → 94.0% 候选 p>n（p/n median 5.47）；有效独立样本量因窗口重叠/马尔可夫依赖介于两者之间。
3. **bigram 不是「确定性函数」**：目标是固定随机 bigram 转移矩阵 P(y|x)，CE 有不可约地板 H(Y|X)∈[2.281, 3.604] nats（median 2.973），实测 CE/地板∈[1.021,1.566]；没有候选低于地板。把「确定性目标 + 无标签噪声」套到 bigram 上会错。
4. **其余 5 族是确定性目标**：univar/mvar 的 y 是符号表达式精确求值（题面原文 "y is its exact evaluation"）；spiral/xor/tabular 的标签是 x 的确定性规则（题面给出规则族、活跃坐标、权重、阈值）。分类族的 `ds_calibration`（4096 行）只用于把阈值放在 s(x) 中位数附近、使类平衡（realized positive rate 0.475–0.523），不是标签噪声。
5. **test 大小各族不同**：univar/mvar 256；tabular/xor/spiral 2048；bigram 200 序列（=2400–4800 token 标签）。test split 固定且所有候选共用；因此「验证集误差」的有限样本噪声不由每个候选重新抽样，10-seed 差异只来自训练随机性。
6. **答案键是经验标签**：500/500 题的 `answer` 等于 3 个候选 10-seed 均值的最小值（argmin）。另外 `sig_gap` 恰好等于 winner 与 runner-up 的 mean_metric 之差（500/500 精确相等），`sig_winrate` 全为 1.0；`sig_margin` 的定义未在 CSV 中给出，本报告不使用。
7. **`qtype` 决定题内受控变量**：`architecture_only`（170 题，三候选 opt/lr/steps/bs 全同，只变架构/p；其中 161 题三候选同一 model_type）；`optimizer_only`（168 题，三候选模型完全相同、p 相同，只变 opt/lr）；`mixed`（162 题，两者都变）。若理论只想比较模型类/尺度，用 `architecture_only` 子集最干净。
8. **跨题不可当作同一 f 的重复实验**：500 题 = 500 个不同数据实例（不同 point/table/sequence seed，不同表达式或规则参数）。跨题汇总会混合不同的 f 和实例随机性；本报告不做跨题方向性归纳。
9. **模型类跨族不同**：bigram 是 transformer/GRU LM（p 含 embedding/head），其余是 MLP（p 含输入/输出投影）；「p」在不同族不是同一类容量度量，跨族比较 p 需谨慎。
10. **MLP 的 p 还受非参数量超参影响**：depth 1–5、width 16–256、act 4 种、residual、逐层 layer_norm 一起变化；`architecture_only` 题中「p 更大」不等于「只是参数更多」，也伴随这些结构差异。
11. **指标口径**：`test_mse`（univar/mvar）与 `test_ce`（其余，nats）不可直接比大小；univar 有附带的目标方差文件 `a_priori/univar_dataset_scale.json`（σ²_test），可做 R²=1−MSE/σ² 归一（mvar 没有对应文件）。
12. **minibatch 有放回抽样**：题面写明每步从固定训练点中有放回抽 batch，因此 n 个训练点被反复使用；`samples_seen/n` 对 univar/mvar 为 16–128，对分类族为 4–32，对 bigram（序列口径）为 5.12–40.96。
13. **题面全部可得**：500 个 prompt.txt 覆盖全部题（路径模板见 §0），含数据卡与三候选完整代码；`questions.csv` 的 `ds_*` 字段（如 `ds_expression`、`ds_rule_family`、`ds_active_features`、`ds_rule_weights`、`ds_spiral_turns`、`ds_calibration`）与题面数据卡一致，可用于批量重建题面信息。
14. **本报告删除的分析**：按要求不包含「参数量 vs 效果」相关性/方向性分析，也不包含任何「大 p 更好/更差」的结论。

---

## 5. `bank-audit.csv` 字段说明（1500 行 = 每候选一行）

| 字段 | 含义 |
|---|---|
| `qid, fam, ds, qtype` | 题号 / 族 / 数据集实例 id / qtype |
| `letter, model_type, d, w, dff, heads, vocab, ctx_len, in_dim, out_dim, act, res` | 候选信件与模型原始字段（MLP: d=depth, w=width；LM: d=num_layers, w=d_model） |
| `trainable_params, opt, lr, wd, mom, steps, bs, samples` | 参数量与训练配置（`samples = steps×bs = samples_seen`） |
| `metric, mean_metric, std_metric, n_seeds` | 指标名、10-seed 均值、10-seed 标准差、seed 数 |
| `n_train, n_test, budget_total` | 题的 train/test 大小与总预算（题级常量） |
| `answer, is_answer` | 官方答案键；本行是否答案候选 |
| `p_over_n, pn_gt1, q_all3_p_gt_n` | p/n（n=ds_train_size）；该候选是否 p>n；该题三候选是否全 p>n |
| `n_train_tokens, p_over_n_tokens, pn_tok_gt1` | 仅 bigram：800×ctx token 标签口径的 n、p/n、p>n |
| `bigram_H_cond, bigram_CE_over_floor` | 仅 bigram：不可约 CE 地板 H(Y\|X)（nats）、mean_metric/H(Y\|X) |
| `rep` | 代表题标记（`univar_r1` 等；非代表题为空） |

复现：`python .team/theory-overparam/_bank_audit.py` 会重新生成 `bank-audit.csv` 并打印本文件所有汇总表（markdown）。
