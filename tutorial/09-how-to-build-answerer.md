# 09 · 动手搭一个理论答题器：从初始化到预测

> 本章回答：08 章说的"答题器 500 题 83.2%"到底是怎么搭出来的？你自己怎么复现一个？
> 读者：会写 PyTorch、读过 01–03 章即可。公式用 LaTeX 数学写（行内 `$…$`、独立公式 `$$…$$`），需要支持数学渲染的阅读器。
> 对应代码：`../verification/theory_answerer.py`（约 400 行，含评估）。

---

## 1. 目标

题库每题给三个候选（模型+优化器+预算），问谁的测试指标最好。我们要：

> 不训练真网络，只根据"初始化 + 题面配置"预测谁会赢。

关键观察（02 章）：如果训练中网络的 Jacobian 几乎不动（惰性/lazy 区），那网络的学习过程**等价于**它的"一阶泰勒替身"的学习过程，而替身的动力学是可以精确模拟的。

---

## 2. 核心思想：切线性化替身

真实网络 $f(x;\theta)$ 在初始化 $\theta_0$ 处做一阶展开：

$$
f(x;\theta)\approx f(x;\theta_0)+J(x)\cdot(\theta-\theta_0)
$$

$$
J(x)=\left.\frac{\partial f(x;\theta)}{\partial\theta}\right|_{\theta=\theta_0}
\qquad\text{（Jacobian，输出维}\times\text{参数维）}
$$

替身 $f_{\mathrm{lin}}$ 对 $\theta$ 是**线性的**——不管原网络有几层、什么激活。于是：

- 回归（MSE）：替身的损失是 $\theta$ 的**二次函数**；
- 分类/LM（CE）：替身的损失是 $\theta$ 的**凸函数**（线性 logits + CE 是凸的）。

凸意味着：没有局部极小值陷阱、动力学完全由 $J$（等价地，初始化核 $K=JJ^\top$）和优化器决定。**整个训练过程可以在不看真实网络的情况下一步步算出来。**

如果真实候选确实是惰性的，替身的测试分数 $\approx$ 真实分数；如果不惰性，两者的差就是"特征学习项"——错题本身成了理论边界的测量。

---

## 3. 五步建造流程

### Step 0：读题面

从题面/spec 拿到：训练/测试数据（直接加载 `train.pt`/`test.pt`）、`steps`、`batch_size`、优化器类型与 lr/wd/betas/momentum、loss（MSE 或 CE）。

### Step 1：同 seed 初始化

```python
torch.manual_seed(seed)          # 和官方 runner 用同一个 seed
model = Model()                  # 消耗掉相同数量的随机数
theta0 = flatten(model.parameters())   # 摊平成一个长向量
f0_train = model(x_train)        # 初始输出（替身公式里的第一项）
```

为什么要管 seed：官方训练是 `manual_seed(seed) → 初始化 → 逐步抽 minibatch`，随机数流是接着初始化往后走的。同一个 seed 才能复刻同一条 minibatch 序列。

### Step 2：拿 Jacobian（两种办法）

**办法 A（显式，回归/分类题用）**：输出维小（回归 1、二分类 2、$n\leq 1024$），直接把 $J$ 存下来：

```python
from torch.func import jacrev
J = jacrev(lambda th: f(th, x_train).reshape(-1))(theta0)  # (n·C, p)
J = J.reshape(n, C, p)
```

内存账：univar 是 $256\times 26\text{万}$ ≈ 260MB，可以；分类个别候选 $p=33\text{万}$ 时是 $2048\times 33\text{万}$ ≈ 2.7GB，要么放宽内存上限，要么把测试侧子采样到 256 个点。

**办法 B（在线，bigram 用）**：LM 的输出是 $(B,L,V)$，$J$ 有几亿到几十亿个元素，存不下。不存，每步现算：

```python
# J·d 用有限差分（一次前向）：
s = 1e-3 * norm(theta0) / norm(d)
Jd = (f(theta0 + s*d, x_batch) - f(theta0, x_batch)) / s

# Jᵀ·g 用反向模式（一次反向，精确）：
out = f_params_dict(theta0, x_batch)
grad_theta = autograd.grad((out * g).sum(), params)
```

为什么混用：GRU 这类算子没有前向模式 AD（`torch.func.jvp` 会报错），所以 $J\cdot d$ 走有限差分；$J^\top\cdot g$ 是标量 $\langle g,f\rangle$ 的普通梯度，反向模式天然支持。每个候选做一次一致性校验：$\langle gp,\,J\cdot d\rangle$ 应等于 $\langle J^\top gp,\,d\rangle$（实测相对误差 $\sim 10^{-3}$，是有限差分的精度）。

### Step 3：在替身上复刻官方训练

```python
theta = theta0.clone().requires_grad_(True)
opt = 按 spec 建的优化器([theta])     # AdamW/SGD/Adagrad/RMSprop，参数照抄

for step in range(steps):
    idx = torch.randint(0, n, (bs,))     # 接着初始化后的随机数流 → 和官方同一批序列
    d = theta.detach() - theta0
    pred = f0_train[idx] + J[idx] @ d    # 替身在 minibatch 上的输出（办法A）
                                         # 办法B：pred = f0 + Jd（在线算）
    g_pred = 损失对 pred 的梯度            # MSE: 2(pred−y)/numel；CE: (softmax−onehot)/bs
    theta.grad = J[idx]ᵀ @ g_pred        # 链式法则：替身对 θ 的梯度
    opt.step()
```

两个容易错的点：

1. **优化器建在 $\theta$ 上，不是建在 $d$ 上**。weight decay 作用在 $\theta$（真实参数值），不是位移。建错对象，wd 的含义就变了。
2. **randint 的调用次数和顺序要和官方一致**。每步一次、形状 (bs,)，不多不少，随机数流才对齐。

### Step 4：读分数，argmin

```python
pred_test = f0_test + J_test @ (theta_T − theta0)
预测指标 = MSE 或 CE(pred_test, y_test)
本题答案 = 预测指标最小的候选
```

测试侧也可以用固定子采样（256–512 个点）省内存——指标是均值，子采样只加一点噪声，不影响三选一的绝大多数判断。

---

## 4. 这套东西"理论"在哪里

容易被质疑的点："你不就是又训了一遍吗？"区别在于：

- 替身是**初始化决定的线性/凸模型**。它的整条学习曲线由初始化核 $K$、目标 $y$、优化器配置决定——不再依赖真实网络的非线性结构。2–3 章的所有核公式（谱滤波、可学窗口 $\eta\lambda_j T$、凸收敛界）描述的就是它的动力学。
- 所以它回答的是一个**有条件的精确问题**："如果这个候选留在惰性区，它会得多少分？"答错的地方，就是惰性假设失效、特征学习真实发生的地方。
- 它**零拟合**：没有任何系数是从题库答案里学出来的。对照之下，旧经验律（74–88%）是从同分布实验里拟合的。

一句话：**真实训练 = 理论预测（替身）+ 特征学习残差。答题器把第一项算出来，残差就是错的那些题。**

---

## 5. 实测成绩与边界（2026-10-20）

- 全部 500 题 top-1 = 83.2%（随机 33.3%，"选最大 lr" 53.0%，KB 经验律自报 74–88%）
- optimizer_only 88.1% > mixed 82.7% > architecture_only 78.8%
- 分族：spiral 92.0 / tabular 86.0 / univar 85.0 / mvar 85.0 / bigram 76.0 / xor 76.0

边界（诚实清单）：

- **单 seed**：官方答案是 10-seed 均值；我们只算 seed 0。刀刃题上会有运气成分。
- **惰性假设失效处必错**：architecture_only 错的 $\sim 20\%$ 主要是特征学习。
- 测试侧子采样和有限差分引入小误差（$\sim 10^{-3}$ 量级），远小于候选间典型差距。
- 预测的是"lazy 世界"的分数；要把残差变成理论，需要特征学习的有限 $T$ 理论——仍是 open problem。

---

## 6. 最小代码骨架（伪代码）

```python
def predict(model_cls, spec, train, test, seed=0):
    torch.manual_seed(seed)
    model = model_cls()
    th0 = flatten(model.parameters())
    f0x = model(train.x); Jx = jacrev(model)(th0, train.x)   # 小输出才显式存
    f0v, Jv = model(test.x), jacrev(model)(th0, test.x[:512])

    th = th0.clone().requires_grad_(True)
    opt = build_optimizer_from_spec([th], spec.optimizer)
    for _ in range(spec.steps):
        idx = torch.randint(0, len(train.x), (spec.bs,))     # 与官方同流
        d = th.detach() - th0
        pred = f0x[idx] + Jx[idx] @ d
        th.grad = Jx[idx].T @ dloss_dpred(pred, train.y[idx], spec.loss)
        opt.step()
    return metric(f0v + Jv @ (th.detach() - th0), test.y, spec.loss)

def answer(question):
    return argmin over candidates of predict(candidate)
```

完整可运行版（含 bigram 在线路径、续跑、评估）在 `../verification/theory_answerer.py` 和 `theory_answerer_eval.py`。

---

## 7. 一页总结

- 替身 $=f(x;\theta_0)+J(\theta-\theta_0)$：线性化后 MSE 变二次、CE 变凸。
- 同 seed 初始化 + 同 minibatch 序列 + 同优化器 → 精确复刻"lazy 世界"的训练。
- 小输出显式存 $J$；大输出（LM）在线 FD-JVP + VJP。
- 读替身的测试分数 → argmin → 500 题 83.2%，零拟合。
- 答错的题 = 特征学习项的测量，标出理论边界，不藏。
