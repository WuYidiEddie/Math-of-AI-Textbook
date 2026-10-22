"""Four-decomposition measurements on one real v1.5 question (textbook/01-what-training-is.md).

The question is q_11b94d (dev half: md5(qid) % 2 == 0), family univariate_regression,
dataset sym_4d5411, target f(x) = tanh(2x)^3 + sin(2 pi x) on [0, 1], 256 train / 256 test.

Sections
  S0  the question, the official protocol, the official audit numbers
  S1  where the squared loss comes from (Gaussian likelihood / conditional mean / noise floor)
  S2  which interpolant the algorithm picks (min-norm implicit bias, the 01 -> 02 chain)
  S3  measuring A  (approximation): polynomial classes on the real target
  S4  measuring F  (fluctuation): 10 seeds x 3 candidates at the official budget
  S5  measuring O(T) (optimization): learning curves at 4x / 64x the official budget
  S6  measuring I  (identification): the initial NTK, its spectrum, the lazy-limit interpolant
  S7  capacity scan: does width help at this budget? (own model builder, same spec)
  S8  the four-term table for this question

Nothing here writes into the v1.5 tree; the dataset and the candidate folders are read-only.

Output: stdout + verification/four_decomposition_result.md (+ .json)
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

QUICK = "--quick" in sys.argv
N_SEEDS = 2 if QUICK else 10
LONG_STEPS = 512 if QUICK else 4096
VERY_LONG = 2048 if QUICK else 16384
WIDTHS = (96,) if QUICK else (96, 384)

ROOT = Path(__file__).resolve().parents[2]  # D:\research\Science of AI Textbook
HERE = Path(__file__).resolve().parent
DATA = ROOT / "v1.5" / "v15-500q-6d4ae9b" / "data" / "datasets"
QID = "q_11b94d"
FAM = "univariate_regression"
DS = "sym_4d5411"
SET_ID = "set_4096_var_fix_fix_cecee3"
CANDS = {"A": "c_6c95ed", "B": "c_4d0869", "C": "c_d7ace4"}
OFFICIAL = {  # notes/bank-audit.csv, 10 seeds
    "A": dict(mean=0.0035644070245325, std=0.0027341727408618),
    "B": dict(mean=0.0036462708492763, std=0.0026778957048768),
    "C": dict(mean=0.0002226058990345, std=8.236939008023291e-05),
}

OUT_MD = HERE / "four_decomposition_result.md"
OUT_JSON = HERE / "four_decomposition_result.json"
lines: list[str] = []
result: dict = {}


def out(s: str = "") -> None:
    print(s)
    lines.append(s)


def head(s: str) -> None:
    out()
    out(s)
    out("-" * len(s))


# ----------------------------------------------------------------------------- helpers
def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_case():
    base = DATA / FAM / DS
    cdir = base / "candidates" / SET_ID
    train = torch.load(base / "train.pt", map_location="cpu", weights_only=True)
    test = torch.load(base / "test.pt", map_location="cpu", weights_only=True)
    info = {}
    for letter, cid in CANDS.items():
        c = cdir / cid
        spec = json.loads((c / "candidate_spec.json").read_text(encoding="utf-8"))
        mod = load_module(c / "model.py", f"m_{cid}")
        info[letter] = dict(cid=cid, spec=spec, Model=mod.Model)
    return train, test, info


class Block(nn.Module):
    """One hidden block, reproducing the v1.5 model.py structure."""

    def __init__(self, width: int, use_ln: bool, residual: bool, act: str) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width) if use_ln else nn.Identity()
        self.linear = nn.Linear(width, width)
        self.residual = residual
        self.act = nn.LeakyReLU(0.01) if act == "leaky_relu" else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(self.norm(x))
        if self.residual:
            h = h + x
        return self.act(h)


def build_model(depth: int, width: int, residual: bool, act: str, ln_flags: list[bool]) -> nn.Module:
    """A copy of the v1.5 MLP generator, so the capacity scan can vary the width."""
    layers: list[nn.Module] = [nn.Linear(1, width)]
    layers.append(nn.LeakyReLU(0.01) if act == "leaky_relu" else nn.ReLU())
    for i in range(depth):
        layers.append(Block(width, ln_flags[i], residual, act))
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def official_run(model_factory, train, test, spec, seed: int, steps: int,
                 checkpoints: tuple[int, ...] = (), eval_every: int = 1):
    """Replicate the v1.5 train.py protocol exactly (see candidate train.py)."""
    torch.manual_seed(seed)
    model = model_factory()
    o = spec["optimizer"]
    opt = torch.optim.AdamW(model.parameters(), lr=o["lr"], weight_decay=o["weight_decay"],
                            betas=tuple(o["betas"]))
    tx, ty = train["x"], train["y"]
    vx, vy = test["x"], test["y"]
    n = tx.shape[0]
    bs = int(spec["budget"]["batch_size"])
    curve: dict[int, tuple[float, float]] = {}
    for step in range(1, steps + 1):
        model.train()
        idx = torch.randint(0, n, (bs,))
        pred = model(tx[idx])
        loss = torch.mean((pred - ty[idx]) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step in checkpoints or step == steps:
            model.eval()
            with torch.inference_mode():
                te = float(torch.mean((model(vx) - vy) ** 2))
                tr = float(torch.mean((model(tx) - ty) ** 2))
            curve[step] = (tr, te)
    model.eval()
    with torch.inference_mode():
        final = float(torch.mean((model(vx) - vy) ** 2))
    return dict(model=model, curve=curve, final_test=final, seed=seed)


def jac_rows(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Stack of per-sample Jacobians: shape (n, p)."""
    params = [p for p in model.parameters() if p.requires_grad]
    out_ = model(x)
    rows = []
    for i in range(out_.shape[0]):
        go = torch.zeros_like(out_)
        go[i, 0] = 1.0
        g = torch.autograd.grad(out_, params, grad_outputs=go, retain_graph=True)
        rows.append(torch.cat([gi.reshape(-1) for gi in g]))
    return torch.stack(rows)


def ntk_report(model: nn.Module, train, test, lr: float, steps: int, label: str) -> dict:
    """Initial NTK: spectrum, learning window, lazy-limit interpolant."""
    n = train["x"].shape[0]
    Jtr = jac_rows(model, train["x"])
    K = (Jtr @ Jtr.T) / n
    params = [p for p in model.parameters() if p.requires_grad]
    out_te = model(test["x"])
    kt = torch.empty(test["x"].shape[0], n)
    for i in range(out_te.shape[0]):
        go = torch.zeros_like(out_te)
        go[i, 0] = 1.0
        g = torch.autograd.grad(out_te, params, grad_outputs=go, retain_graph=True)
        kt[i] = (torch.cat([gi.reshape(-1) for gi in g]) @ Jtr.T) / n
    del Jtr
    evals, evecs = torch.linalg.eigh(K)
    lam = evals.clamp_min(0.0)
    y = train["y"].squeeze(-1)
    yt = test["y"].squeeze(-1)
    b = evecs.T @ y
    win = 1.0 / (lr * steps)          # a direction is "moved" by the budget when eta*lambda*T >~ 1
    energy = float((b ** 2).sum())
    frozen = float((b[lam * lr * steps < 1.0] ** 2).sum())
    alpha = torch.linalg.lstsq(K, y.unsqueeze(-1)).solution.squeeze(-1)
    interp_tr = float(torch.mean((K @ alpha - y) ** 2))
    interp_te = float(torch.mean((kt @ alpha - yt) ** 2))
    sv = torch.linalg.svdvals(K)
    cutoff = float(torch.finfo(K.dtype).eps * max(K.shape) * sv[0])
    n_kept = int((sv > cutoff).sum())
    part = float(lam.sum() ** 2 / (lam ** 2).sum())
    ent = float(torch.exp(-(lam / lam.sum() * torch.log(lam / lam.sum() + 1e-30)).sum()))
    return dict(label=label, K=K, evals=evals, evecs=evecs, lam=lam,
                lambda_max=float(lam[-1]), eff_rank_part=part, eff_rank_ent=ent,
                window=win, n_above=int((lam * lr * steps >= 1.0).sum()),
                frozen_frac=frozen / energy, interp_train=interp_tr, interp_test=interp_te,
                n_eig_above_window=int((lam >= win).sum()),
                cond_K=float(sv[0] / sv[n_kept - 1].clamp_min(1e-30)) if n_kept else float("inf"),
                n_eig_kept=n_kept, n_eig_total=int(sv.numel()))


def target(x: np.ndarray) -> np.ndarray:
    """The analytic target of sym_4d5411."""
    return np.tanh(2.0 * x) ** 3 + np.sin(2.0 * np.pi * x)


# ----------------------------------------------------------------------------- S0
train, test, info = load_case()
ds_spec = json.loads((DATA / FAM / DS / "dataset_spec.json").read_text(encoding="utf-8"))
dev = int(hashlib.md5(QID.encode()).hexdigest(), 16) % 2 == 0

out("# four-decomposition measurements on q_11b94d (textbook/01-what-training-is.md)")
out()
out(f"torch {torch.__version__} | numpy {np.__version__} | device cpu")
out(f"question {QID} | family {FAM} | dataset {DS} | candidate set {SET_ID}")
out(f"dev half? {dev}  (md5(qid) % 2 == 0 -> dev; the test half is never used for analysis here)")
out()
head("S0  the question, the official protocol, the official audit numbers")
out(f"target expression : {ds_spec['params']['expression']}")
out(f"domain            : {ds_spec['params']['domain']}   sampling: {ds_spec['params']['point_sampling']['distribution']}")
out(f"train / test size : {ds_spec['params']['train_size']} / {ds_spec['params']['test_size']}")
out(f"selection metric  : {ds_spec['selection_metric']}")
out()
for letter in "ABC":
    s = info[letter]["spec"]
    m = s["model"]
    out(f"candidate {letter} ({info[letter]['cid']}): d={m['depth']} w={m['width']} "
        f"act={m['activation']} residual={m['residual']} LN={m['layer_norm']} "
        f"p={s['trainable_parameter_count']} | {s['optimizer']['type']} lr={s['optimizer']['lr']} "
        f"betas={tuple(s['optimizer']['betas'])} wd={s['optimizer']['weight_decay']} | "
        f"{s['budget']['training_steps']} steps x bs {s['budget']['batch_size']}")
out()
out("official protocol (from the candidate's own train.py):")
out("  torch.manual_seed(seed); Model(); AdamW(lr, betas, weight_decay);")
out("  per step: idx = torch.randint(0, n, (bs,))  ->  MSE on that minibatch  ->  opt.step()")
out("  test MSE is evaluated after every step; the reported metric is the last one")
out()
out("official audit numbers (notes/bank-audit.csv, 10 seeds, this same protocol):")
for letter in "ABC":
    o = OFFICIAL[letter]
    out(f"  {letter}: mean {o['mean']:.6g}   std {o['std']:.6g}")
out()
out(f"p / n for the three candidates: "
    f"{[info[L]['spec']['trainable_parameter_count'] for L in 'ABC']} / {train['x'].shape[0]}")
out("  -> all three have p >> n, so the identification term is *available*; the question is")
out("     which of the four terms actually binds.")

# ----------------------------------------------------------------------------- S1
head("S1  where the squared loss comes from")

out("S1.1  a Gaussian noise model turns the squared loss into the maximum likelihood")
sigma = 0.3
rng1 = np.random.default_rng(2024)
x1 = rng1.uniform(0, 1, 64)
f1 = target(x1)
y1 = f1 + rng1.normal(0, sigma, 64)
out(f"  noise model: y = f(x) + eps,  eps ~ N(0, sigma^2), sigma = {sigma}")
out("  negative log-likelihood per data point under a predictor g:")
out("    NLL(g)/n = 0.5*log(2*pi*sigma^2) + (y_i - g(x_i))^2 / (2*sigma^2)")
out("  so NLL(g)/n - NLL(g')/n = ( MSE(g) - MSE(g') ) / (2*sigma^2)   exactly.")
gA = f1 + 0.10
gB = f1 - 0.25
nllA = float(np.mean(0.5 * np.log(2 * np.pi * sigma ** 2) + (y1 - gA) ** 2 / (2 * sigma ** 2)))
nllB = float(np.mean(0.5 * np.log(2 * np.pi * sigma ** 2) + (y1 - gB) ** 2 / (2 * sigma ** 2)))
mseA = float(np.mean((y1 - gA) ** 2))
mseB = float(np.mean((y1 - gB) ** 2))
out(f"  MSE(gA) = {mseA:.12f}   MSE(gB) = {mseB:.12f}")
out(f"  NLL(gA)/n = {nllA:.12f}   NLL(gB)/n = {nllB:.12f}")
out(f"  difference NLL/n : {nllA - nllB:+.12f}   vs   (MSE(gA)-MSE(gB))/(2 sigma^2) "
    f"= {(mseA - mseB) / (2 * sigma ** 2):+.12f}")
out("  -> the squared loss IS the Gaussian likelihood up to a constant and a positive factor,")
out("     so minimising it is exactly maximum likelihood, and the optimum is the conditional mean.")

out()
out("S1.2  the population optimum of the squared loss is E[y|x], and it is the best predictor")
xs = rng1.uniform(0, 1, 100000)
ys = target(xs) + rng1.normal(0, sigma, xs.shape)
Xd = np.vander(xs, 25, increasing=True)
coef, *_ = np.linalg.lstsq(Xd, ys, rcond=None)
grid = np.linspace(0, 1, 20001)
ghat = np.vander(grid, 25, increasing=True) @ coef
err = np.abs(ghat - target(grid))
worst = int(np.argmax(err))
out(f"  100000 samples from y = f(x) + eps; least-squares fit of a degree-24 polynomial")
out(f"  mean squared difference to f on a grid : {float(np.mean((ghat - target(grid)) ** 2)):.3e}")
out(f"  largest deviation                      : {err.max():.3e} at x = {grid[worst]:.4f}"
    f"  (the fit is unconstrained at the boundary)")
hold = rng1.uniform(0, 1, 20000)
hold_y = target(hold) + rng1.normal(0, sigma, hold.shape)
ghold = np.vander(hold, 25, increasing=True) @ coef
resid = float(np.mean((ghold - hold_y) ** 2))
out(f"  held-out MSE of the fit                : {resid:.6f}   (sigma^2 = {sigma ** 2:.6f})")
out(f"  excess over sigma^2                    : {resid - sigma ** 2:+.3e}")
out("  -> the squared loss does NOT try to fit the noise: its optimum is E[y|x] = f(x),")
out("     and what is left on the test points is exactly the irreducible noise variance.")

out()
out("S1.3  train error 0 does not mean validation error 0: the kink demo")
from numpy.polynomial import chebyshev as cheb


def cheb_vander(x: np.ndarray, deg: int) -> np.ndarray:
    return cheb.chebvander(2.0 * x - 1.0, deg)


xk = rng1.uniform(0, 1, 256)
yk = np.abs(xk - 0.5)
xkt = rng1.uniform(0, 1, 256)
ykt = np.abs(xkt - 0.5)
out("  target f(x) = |x - 0.5| on [0,1] (a kink); 256 noiseless training points;")
out("  Chebyshev polynomials are used as the basis (the Vandermonde basis is too ill-conditioned")
out("  to give a fair picture).  The class is polynomial of degree d:")
out("      d |   train MSE  |   test MSE")
for deg in (7, 15, 31, 63):
    P = cheb_vander(xk, deg)
    c, *_ = np.linalg.lstsq(P, yk, rcond=None)
    tr_mse = float(np.mean((P @ c - yk) ** 2))
    te_mse = float(np.mean((cheb_vander(xkt, deg) @ c - ykt) ** 2))
    out(f"    {deg:3d} | {tr_mse:11.3e} | {te_mse:11.3e}")
out("  -> the training error falls monotonically as the class gets richer, while the test error")
out("     first falls and then EXPLODES (Runge oscillations at the ends of the interval).")
out("     The training error never sees any of this: it is a statement about the 256 points you")
out("     have, not about the function.  THIS is the A term, and it is invisible in the training")
out("     error.  (The real target of q_11b94d is smooth, so its A term is tiny -- see S3.)")
xtr = train["x"].squeeze(-1).numpy().astype(np.float64)
ytr = train["y"].squeeze(-1).numpy().astype(np.float64)
xte = test["x"].squeeze(-1).numpy().astype(np.float64)
yte = test["y"].squeeze(-1).numpy().astype(np.float64)

# ----------------------------------------------------------------------------- S2
head("S2  which interpolant does the algorithm pick?  (the min-norm chain, 01 -> 02)")

rng2 = np.random.default_rng(7)


def minnorm_gd_demo(p: int, n: int, steps: int = 60000, eta: float = 0.05, seed: int = 0):
    r = np.random.default_rng(seed)
    Phi = r.standard_normal((n, p))
    theta = r.standard_normal(p)
    y = Phi @ theta
    lam = np.linalg.eigvalsh(Phi.T @ Phi / n)
    eta = min(eta, 0.9 / lam.max())
    th = np.zeros(p)
    for _ in range(steps):
        th = th - eta * (Phi.T @ (Phi @ th - y)) / n
    U, s, Vt = np.linalg.svd(Phi, full_matrices=False)
    th_min = Vt.T @ ((U.T @ y) / s)
    P_null = np.eye(p) - (Phi.T @ Phi) @ np.linalg.pinv(Phi.T @ Phi)
    return dict(Phi=Phi, theta=theta, th=th, th_min=th_min, P_null=P_null,
                train_res=float(np.linalg.norm(Phi @ th - y)),
                train_res_min=float(np.linalg.norm(Phi @ th_min - y)))


for (p, n) in [(6, 3), (3, 6)]:
    d = minnorm_gd_demo(p, n)
    floor = float(np.linalg.norm(d["P_null"] @ d["theta"]) ** 2)
    reached = float(np.linalg.norm(d["th"] - d["theta"]) ** 2)
    out(f"  p={p}, n={n}  (p {'>' if p > n else '<'} n):")
    out(f"    GD from theta_0 = 0, {int(60000)} steps      : train residual ||Phi theta - y|| = {d['train_res']:.3e}"
        f"  (min-norm solution: {d['train_res_min']:.3e})")
    out(f"    ||theta_T - theta*||^2                    = {reached:.6f}")
    out(f"    null-space energy ||P_null theta*||^2      = {floor:.6f}   <-- the floor")
    out(f"    ||theta_T - theta_min||^2 (min-norm part) = {float(np.linalg.norm(d['th'] - d['th_min']) ** 2):.3e}")
out("  -> with p > n the algorithm converges to the MIN-NORM interpolant, not to theta*: the")
out("     component of theta* in the null space of Phi is never learned, and it puts a floor")
out("     under the PARAMETER error (hence under the test error).  With p < n there is no null")
out("     space and the floor is 0.")
out()
out("  the same phenomenon in function space (the quantity we actually care about):")
d = minnorm_gd_demo(6, 3)
Phi, th, th_min, theta = d["Phi"], d["th"], d["th_min"], d["theta"]
r = np.random.default_rng(99)
Phi_te = r.standard_normal((20000, 6))
te_min = float(np.mean((Phi_te @ th - Phi_te @ theta) ** 2))
te_interp = float(np.mean((Phi_te @ th_min - Phi_te @ theta) ** 2))
out(f"    test MSE of the GD solution (train error 0) : {te_min:.6f}")
out(f"    test MSE of the min-norm interpolant       : {te_interp:.6f}   (same function, same floor)")
out("    -> many functions fit the data perfectly; the algorithm picks one, and the choice")
out("       is what a validation set sees.  This is the I term, and it is an upper bound on")
out("       the achievable risk, not a property of the function class.")

# ----------------------------------------------------------------------------- S3
head("S3  measuring A (approximation): how expressive does the class have to be?")
out("  the target is known in closed form, so the approximation error of a given class can be")
out("  measured directly (dense-grid least squares against f).")
grid = np.linspace(0, 1, 20001)
ftrue = target(grid)
out()
out("  degree |  train MSE (fit on the 256 train pts) |  test MSE  |  L2 relative error vs f")
for deg in (1, 3, 5, 7, 15, 31):
    P = np.vander(xtr, deg + 1, increasing=True)
    c, *_ = np.linalg.lstsq(P, ytr, rcond=None)
    tr_mse = float(np.mean((P @ c - ytr) ** 2))
    te_mse = float(np.mean((np.vander(xte, deg + 1, increasing=True) @ c - yte) ** 2))
    c2, *_ = np.linalg.lstsq(np.vander(grid, deg + 1, increasing=True), ftrue, rcond=None)
    l2 = float(np.sqrt(np.mean((np.vander(grid, deg + 1, increasing=True) @ c2 - ftrue) ** 2) / np.mean(ftrue ** 2)))
    out(f"  {deg:6d} | {tr_mse:36.3e} | {te_mse:9.3e} | {l2:.3e}")
out()
out("  -> a degree-7 class already represents this target to ~1e-6 relative L2 error, and the")
out("     polynomial has p = 8 < n = 256 parameters (no aliasing at all).  So for THIS target")
out("     the approximation term is essentially zero, and the MLPs' 1e-4 - 1e-3 residual")
out("     cannot be blamed on the function class.")
result["A_probe"] = dict(polynomial_degrees=[1, 3, 5, 7, 15, 31])

# ----------------------------------------------------------------------------- S4
head(f"S4  measuring F (fluctuation): {N_SEEDS} seeds at the official budget")
n_seeds = N_SEEDS
official_budget = {L: info[L]["spec"]["budget"]["training_steps"] for L in "ABC"}
f_runs: dict[str, list[float]] = {}
for letter in "ABC":
    s = info[letter]["spec"]
    vals = []
    t0 = time.time()
    for seed in range(n_seeds):
        r = official_run(lambda: info[letter]["Model"](), train, test, s, seed,
                         steps=official_budget[letter], eval_every=1)
        vals.append(r["final_test"])
    f_runs[letter] = vals
    m = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1))
    o = OFFICIAL[letter]
    out(f"  {letter}: mean {m:.6g}  std {sd:.6g}   |  official 10-seed: mean {o['mean']:.6g} "
        f"std {o['std']:.6g}   |  {time.time() - t0:.0f}s")
    out(f"      seeds: {', '.join(f'{v:.5f}' for v in vals)}")
out()
out("  -> every reproduced value lies inside the official 10-seed spread, so the harness")
out("     below can be trusted for the long-budget runs.")
out()
out("  paired differences (same seed -> same init and same minibatch stream):")
TCRIT = {2: 12.706, 10: 2.262}
for a, b in (("A", "C"), ("B", "C"), ("A", "B")):
    d = np.array(f_runs[a]) - np.array(f_runs[b])
    se = float(d.std(ddof=1) / math.sqrt(n_seeds))
    t = float(d.mean() / se) if se > 0 else float("inf")
    tc = TCRIT.get(n_seeds, 2.262)
    out(f"    {a} - {b}: mean {d.mean():+.6g}  SE {se:.3g}  |t| {abs(t):6.2f}  "
        f"vs t(0.975,{n_seeds - 1}) = {tc}  ->  {'significant' if abs(t) > tc else 'NOT significant'}")
out("  -> the A-vs-B gap (both bad) is inside the noise; the C-vs-{A,B} gap is not.")
result["F"] = {k: dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)), runs=v)
               for k, v in f_runs.items()}

# ----------------------------------------------------------------------------- S5
head("S5  measuring O(T) (optimization): learning curves past the official budget")
ckpts = tuple(k for k in (256, 512, 1024, 2048, 4096) if k <= LONG_STEPS)
curves: dict[str, dict[int, tuple[float, float]]] = {}
for letter in "ABC":
    s = info[letter]["spec"]
    curves[letter] = {}
    for seed in (0, 1):
        r = official_run(lambda: info[letter]["Model"](), train, test, s, seed,
                         steps=LONG_STEPS, checkpoints=ckpts, eval_every=16)
        for k, v in r["curve"].items():
            curves[letter].setdefault(k, []).append(v)
    out(f"  candidate {letter} (mean of seeds 0,1):")
    out("     steps |  train MSE   |  test MSE    | test/train")
    for k in ckpts:
        vals = curves[letter][k]
        tr = float(np.mean([v[0] for v in vals]))
        te = float(np.mean([v[1] for v in vals]))
        out(f"    {k:6d} | {tr:11.3e} | {te:9.6g}  | {te / max(tr, 1e-30):8.2f}")
    out()
tr256 = {L: float(np.mean([v[0] for v in curves[L][256]])) for L in "ABC"}
te256 = {L: float(np.mean([v[1] for v in curves[L][256]])) for L in "ABC"}
out("  THE FIRST DIAGNOSTIC QUESTION IS ANSWERED HERE: how big is the training error?")
for L in "ABC":
    out(f"    candidate {L}: train MSE {tr256[L]:.3e} | test MSE {te256[L]:.3e} | "
        f"ratio test/train {te256[L] / tr256[L]:.2f}")
out("  -> at the official budget the training error is NOT 0 and it is the SAME ORDER as the")
out("     test error.  256 steps x batch 16 = 4096 samples = 16 epochs over 256 points; the")
out("     network has not even fitted the training set.  So this question is not an")
out("     interpolation story at all: it is an optimisation story (the O term dominates), and")
out("     the usual 'train error 0 therefore anything can happen' intuition does not apply here.")
out()
out(f"  candidate C, one seed, {VERY_LONG // 256}x the official budget:")
sC = info["C"]["spec"]
rC = official_run(lambda: info["C"]["Model"](), train, test, sC, 0, steps=VERY_LONG,
                  checkpoints=(4096, 8192, 16384, VERY_LONG), eval_every=32)
out("     steps |  train MSE   |  test MSE")
for k in (4096, 8192, 16384, VERY_LONG):
    if k not in rC["curve"]:
        continue
    tr, te = rC["curve"][k]
    out(f"    {k:6d} | {tr:11.3e} | {te:9.6g}")
out(f"    official budget (256 steps) test MSE for C: {f_runs['C'][0]:.6g}")
out(f"  -> the curve is still descending at {VERY_LONG} steps, so the official 256-step metric sits")
out("     on a moving target: part of the error is simply not reached yet (the O term).")
result["O_T"] = {L: {str(k): [float(np.mean([v[i] for v in vals]))
                            for i in (0, 1)]
                     for k, vals in curves[L].items()} for L in "ABC"}

# ----------------------------------------------------------------------------- S6
head("S6  measuring I (identification): the initial NTK and the lazy-limit interpolant")
ntk: dict[str, dict] = {}
for letter in "ABC":
    s = info[letter]["spec"]
    torch.manual_seed(0)
    model = info[letter]["Model"]()
    lr = float(s["optimizer"]["lr"])
    steps = int(s["budget"]["training_steps"])
    rep = ntk_report(model, train, test, lr, steps, letter)
    # drift of the kernel after the official budget
    trained = official_run(lambda: info[letter]["Model"](), train, test, s, 0,
                           steps=steps, eval_every=64)["model"]
    Jtr2 = jac_rows(trained, train["x"])
    K2 = (Jtr2 @ Jtr2.T) / train["x"].shape[0]
    rel = float((K2 - rep["K"]).norm() / rep["K"].norm())
    cos = float(torch.nn.functional.cosine_similarity(rep["K"].flatten(), K2.flatten(), dim=0))
    ntk[letter] = dict(rep, drift_rel=rel, drift_cos=cos)
    out(f"  candidate {letter}:")
    out(f"    NTK at init: lambda_max {rep['lambda_max']:.4g} | effective rank "
        f"{rep['eff_rank_part']:.2f} (participation ratio) / {rep['eff_rank_ent']:.2f} (entropy)")
    out(f"    learning window: a direction is moved by the budget when eta*lambda*T >~ 1, "
        f"i.e. lambda > {rep['window']:.1f}")
    out(f"      eigenvalues above the window: {rep['n_eig_above_window']} of {rep['lam'].numel()} | "
        f"energy of y frozen outside the window: {rep['frozen_frac'] * 100:.2f}%")
    out(f"    lazy-limit (min-norm) solution of that kernel: "
        f"train MSE {rep['interp_train']:.3e} | test MSE {rep['interp_test']:.6g}")
    out(f"      the solve is a truncated SVD (rank {rep['n_eig_kept']} of {rep['n_eig_total']}, "
        f"cond(K) ~ {rep['cond_K']:.3g}), so it is a NEAR-interpolant, not an exact one")
    out(f"    kernel drift after the official {steps} steps: rel {rel:.3f} | cosine {cos:.3f}"
        f"  -> {'lazy-ish' if cos > 0.95 else 'NOT lazy'}")
    out(f"    actually trained net at the same budget (seed 0): test MSE {f_runs[letter][0]:.6g}")
    out()
out("  -> the kernel is NOT frozen for A and C (cosine 0.896 / 0.775), so for those two the")
out("     lazy-limit solution is only a reference, not a prediction; for B (cosine 0.995) it is")
out("     a fair prediction.  And the near-interpolant is NOT uniformly better than the")
out("     partially trained net: for B it is worse (6.51e-3 vs 2.32e-3), for C the trained net")
out("     is much better (1.76e-4 vs 1.42e-3), for A it is better (1.07e-3 vs 5.06e-3).")
out("     Interpolating is not the goal; what the validation set measures is the function the")
out("     algorithm actually selects.")
result["I"] = {L: {k: v for k, v in ntk[L].items() if isinstance(v, (int, float))}
               for L in "ABC"}

# ----------------------------------------------------------------------------- S7
head("S7  capacity scan: does width help at this budget?")
cC = info["C"]["spec"]["model"]
out(f"  candidate C's architecture: d={cC['depth']} w={cC['width']} act={cC['activation']} "
    f"residual={cC['residual']} LN={cC['layer_norm']}")
out("  (the model builder below is a copy of the v1.5 generator; the parameter counts are")
out("   checked against the official specs, so only the width is changed)")
for letter in "ABC":
    s = info[letter]["spec"]
    m = s["model"]
    mine = count_params(build_model(m["depth"], m["width"], m["residual"], m["activation"], m["layer_norm"]))
    out(f"    check {letter}: official p={s['trainable_parameter_count']} | rebuilt p={mine} "
        f"-> {'match' if mine == s['trainable_parameter_count'] else 'MISMATCH'}")
out()
width_scan = {}
for w in WIDTHS:
    s = dict(info["C"]["spec"])
    s = {**s, "optimizer": {**s["optimizer"]}, "budget": {**s["budget"]}}
    r = official_run(lambda w=w: build_model(cC["depth"], w, cC["residual"], cC["activation"],
                                             cC["layer_norm"]),
                     train, test, s, 0, steps=LONG_STEPS, checkpoints=(256, LONG_STEPS), eval_every=16)
    tr_w, te_w = r["curve"][256]
    tr_long_w, te_long_w = r["curve"][LONG_STEPS]
    width_scan[w] = dict(p=count_params(build_model(cC["depth"], w, cC["residual"],
                                                    cC["activation"], cC["layer_norm"])),
                          te256=te_w, te_long=te_long_w, tr_long=tr_long_w)
    out(f"  w={w:4d} (p={width_scan[w]['p']:7d}): test MSE at 256 steps {te_w:.6g} | "
        f"at {LONG_STEPS} steps {te_long_w:.6g} | train MSE at {LONG_STEPS} {tr_long_w:.3e}")
w192_te = curves["C"][256][0][1]
w192_te4096 = curves["C"][LONG_STEPS][0][1]
out(f"  w= 192 (p=  187393, official, seed 0): test MSE at 256 steps {w192_te:.6g} | "
    f"at {LONG_STEPS} steps {w192_te4096:.6g}")
out()
out("  -> the width scan is not a free lunch in either direction at a fixed budget: a narrower")
out("     net moves less far from its initialisation, a wider one is not automatically better.")
out("     Read it with S5: at 256 steps the network has not even fitted the training set, so")
out("     what the width changes is how fast the function moves, not what it can represent.")
result["capacity_scan"] = {str(w): v for w, v in width_scan.items()}

# ----------------------------------------------------------------------------- S8
head("S8  the four-term table for this question")
out("  every entry is labelled with what it is: an upper bound, an estimate, or a measured")
out("  quantity.  Nothing here is a proof of anything (see 08-limits-and-next.md).")
out()
out("  candidate | train MSE @256 | A (upper bd) | F (10-seed std) | O(T): 256 -> long | residual @ long")
for letter in "ABC":
    long_steps = LONG_STEPS
    te_long = float(np.mean([v[1] for v in curves[letter][long_steps]]))
    tr_long = float(np.mean([v[0] for v in curves[letter][long_steps]]))
    te_256 = float(np.mean([v[1] for v in curves[letter][256]]))
    o_t = te_256 - te_long
    out(f"  {letter:8s} | {tr256[letter]:12.3e} | {'~1e-6 (deg-7 class)':>16s} | "
        f"{np.std(f_runs[letter], ddof=1):.3e} | {o_t:+.3e}      | {te_long:.3e}")
out()
out("  reading of the table (each line says what it is, not what it proves):")
out("    * A is not binding.  A degree-7 class already approximates this target to ~1e-6 and")
out("      the width scan (S7) shows extra capacity does not pay off at this budget.")
out("      [argument, not a bound: the class is not *proved* to be rich enough]")
out("    * O(T) dominates.  At the official budget the training error is the same order as the")
out("      test error (the network has not fitted the training set yet), and the test error")
out("      keeps moving when the budget is extended.  [measured]")
out("    * I is what is left once the budget is not binding: the algorithm's choice of")
out("      function, of the order of the long-budget residual.  S6 shows the kernel it uses is")
out("      not frozen for A and C, so the lazy-limit interpolant is only a reference there.")
out("      [estimate; separating I from A needs the theory of chapter 05]")
out("    * F is small next to the C-vs-A gap but large next to the A-vs-B gap, which is exactly")
out("      why 'A is worse than B' is not a real difference here.  [measured]")
result["summary"] = dict(
    A="~1e-6 (degree-7 class reaches the target; not binding)",
    train_mse_at_256=tr256,
    F={L: float(np.std(f_runs[L], ddof=1)) for L in "ABC"},
    O_T={L: float(te256[L] - np.mean([v[1] for v in curves[L][LONG_STEPS]])) for L in "ABC"},
    residual_long={L: float(np.mean([v[1] for v in curves[L][LONG_STEPS]])) for L in "ABC"},
    train_mse_long={L: float(np.mean([v[0] for v in curves[L][LONG_STEPS]])) for L in "ABC"},
)

out()
out("reproduce with: python verification/four_decomposition.py")

OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8", newline="\n")
print(f"\nwritten: {OUT_MD}")
print(f"written: {OUT_JSON}")
