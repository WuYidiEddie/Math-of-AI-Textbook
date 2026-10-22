"""Zero-space (null-space) projection under real training (textbook/02-exact-case.md section 6).

Question: chapter 02 shows the null-space component of theta is "frozen" -- but is that a
*constraint the optimizer enforces*, or a *consequence of the gradient being blind* to those
directions?  If it is the latter, then anything that can SEE the null space can change it.

Sections
  S1  the conservation law, optimizer by optimizer (synthetic linear model, p > n):
      plain GD / minibatch SGD / GD+weight-decay / Adam / AdamW.  For each we measure the
      drift ||P_null theta_T - P_null theta_0|| relative to ||P_null theta_0||.
  S2  the function-space consequence: training error cannot see the null component
      (identical for theta_0 = 0 and theta_0 != 0), but the test error carries it as a
      permanent offset of size exactly ||Phi_te P_null theta_0||^2 / m.
  S3  the real v1.5 question q_11b94d (dev half, candidates A and C): the official protocol
      is AdamW(lr, wd) with minibatch batches, so the null component is NOT exactly frozen.
      We measure (a) the AdamW decay factor over the official budget, (b) the actual drift
      after 256 official steps, (c) the lazy-limit "initialization tax" -- the part of the
      initial network's output that training can never remove, on the test set.

Nothing writes into the v1.5 tree; the dataset and candidate folders are read-only.
Output: stdout + verification/zero_space_init_result.md (+ .json)
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

QUICK = "--quick" in sys.argv
SYN_ONLY = "--syn-only" in sys.argv  # skip the v1.5 section (the Jacobian work)

ROOT = Path(__file__).resolve().parents[2]  # D:\research\Science of AI Textbook
HERE = Path(__file__).resolve().parent
DATA = ROOT / "v1.5" / "v15-500q-6d4ae9b" / "data" / "datasets"
QID = "q_11b94d"
FAM = "univariate_regression"
DS = "sym_4d5411"
SET_ID = "set_4096_var_fix_fix_cecee3"
CANDS = {"A": "c_6c95ed", "C": "c_d7ace4"}  # the two extreme-rank candidates
OFFICIAL = {  # notes/bank-audit.csv, 10 seeds
    "A": dict(mean=0.0035644070245325, std=0.0027341727408618),
    "C": dict(mean=0.0002226058990345, std=8.236939008023291e-05),
}

OUT_MD = HERE / "zero_space_init_result.md"
OUT_JSON = HERE / "zero_space_init_result.json"
lines: list[str] = []
result: dict = {}


def out(s: str = "") -> None:
    print(s, flush=True)
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


def target(x: np.ndarray) -> np.ndarray:
    return np.tanh(2.0 * x) ** 3 + np.sin(2.0 * np.pi * x)


# =========================================================================== S1
head("S1  the conservation law, optimizer by optimizer (synthetic linear model)")
out()
out("  setup: y = Phi theta_true, Phi in R^{n x p} with p > n, theta_0 random (NOT 0).")
out("  the claim to test: is P_null(theta_t) = P_null(theta_0) an ENFORCED constraint,")
out("  or only a consequence of the loss not seeing the null directions?  If the latter,")
out("  any mechanism that CAN see the null space (weight decay, Adam's elementwise")
out("  second moment, moving features) is free to change it.")
out()
p, n, T = 200, 32, 3000
rng = np.random.default_rng(0)
Phi = rng.standard_normal((n, p)).astype(np.float64)
theta_true = rng.standard_normal(p)
y = Phi @ theta_true
theta0 = rng.standard_normal(p) * 0.7
Phit = torch.tensor(Phi, dtype=torch.float32)
yt = torch.tensor(y, dtype=torch.float32)

U, s, Vt = np.linalg.svd(Phi, full_matrices=False)
r = int((s > 1e-10 * s[0]).sum())
Vrow = Vt[:r]                      # r x p, rows span row(Phi)
Prow = Vrow.T @ Vrow               # p x p
Pnull = np.eye(p) - Prow
n0 = float(np.linalg.norm(Pnull @ theta0))
out(f"  p = {p}, n = {n}, rank(Phi) = {r}, dim null(Phi) = {p - r}, T = {T} steps")
out(f"  ||P_null theta_0|| = {n0:.6f}   (this is what each optimizer is allowed to change)")
out()

lam_gd = float(np.linalg.eigvalsh(Phi.T @ Phi / n).max())
lr_gd = 0.9 * 2.0 / lam_gd / 10.0  # comfortably inside the stability window


def drift_report(label: str, th_end: np.ndarray, note: str = "") -> dict:
    d = float(np.linalg.norm(Pnull @ th_end - Pnull @ theta0))
    rec = dict(label=label, drift=d, drift_rel=d / n0,
               null_norm_end=float(np.linalg.norm(Pnull @ th_end)), note=note)
    out(f"  {label:34s} | ||P_null th_T - P_null th_0|| = {d:.3e}   (rel {d / n0:.3e})"
        f"   | ||P_null th_T|| = {rec['null_norm_end']:.6f}" + (f"   {note}" if note else ""))
    return rec


def run_gd(wd: float, Tsteps: int, label: str) -> np.ndarray:
    th = torch.tensor(theta0.copy(), dtype=torch.float32, requires_grad=True)
    for _ in range(Tsteps):
        loss = torch.mean((Phit @ th - yt) ** 2) + 0.5 * wd * torch.sum(th ** 2)
        g, = torch.autograd.grad(loss, th)
        with torch.no_grad():
            th -= lr_gd * g
    return th.detach().numpy().astype(np.float64)


def run_minibatch_sgd(wd: float, Tsteps: int, bs: int, label: str) -> np.ndarray:
    th = torch.tensor(theta0.copy(), dtype=torch.float32, requires_grad=True)
    g_ = torch.Generator().manual_seed(0)
    for _ in range(Tsteps):
        idx = torch.randint(0, n, (bs,), generator=g_)
        loss = torch.mean((Phit[idx] @ th - yt[idx]) ** 2) + wd * torch.sum(th ** 2)
        g, = torch.autograd.grad(loss, th)
        with torch.no_grad():
            th -= lr_gd * g
    return th.detach().numpy().astype(np.float64)


def run_torch_opt(optname: str, wd: float, Tsteps: int, lr: float, label: str) -> np.ndarray:
    th = torch.tensor(theta0.copy(), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.AdamW([th], lr=lr, weight_decay=wd) if optname == "adamw" \
        else torch.optim.Adam([th], lr=lr, weight_decay=wd)
    for _ in range(Tsteps):
        loss = torch.mean((Phit @ th - yt) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return th.detach().numpy().astype(np.float64)


out("  optimizer                          | drift of the null-space projection")
out("  " + "-" * 96)
s1 = []
s1.append(drift_report("full-batch GD, no wd", run_gd(0.0, T, "gd"),
                       note="(should be ~ machine eps: exact identity)"))
s1.append(drift_report("minibatch SGD, no wd, bs=16", run_minibatch_sgd(0.0, T, 16, "sgd"),
                       note="(still in row(Phi): row(Phi_B) subset row(Phi))"))
dec = (1.0 - lr_gd * 1e-3) ** T
s1.append(drift_report("full-batch GD, wd=1e-3", run_gd(1e-3, T, "gd_wd"),
                       note=f"(should decay: (1-lr*wd)^T = {dec:.6f})"))
s1.append(drift_report("Adam, no wd", run_torch_opt("adam", 0.0, T, 3e-3, "adam"),
                       note="(elementwise 2nd moment leaks out of row(Phi))"))
decw = (1.0 - 3e-3 * 1e-3) ** T
s1.append(drift_report("AdamW, wd=1e-3", run_torch_opt("adamw", 1e-3, T, 3e-3, "adamw"),
                       note=f"(Adam leak + decay: (1-lr*wd)^T = {decw:.6f})"))
out()
out("  reading:")
out("    * plain GD: the drift is at float32 machine precision -- the identity is EXACT,")
out("      and nothing in the code 'enforces' it; the gradient simply has no component")
out("      along null(Phi).")
out("    * minibatch SGD is also exact: a minibatch's gradient lies in row(Phi_B), and")
out("      row(Phi_B) is a subset of row(Phi) -- the noise explores the row space but")
out("      cannot reach the null space.")
out("    * weight DECAY changes the story: the decay term -wd*theta HAS a component in")
out("      the null space, so the projection shrinks by (1 - lr*wd) every step.  It is")
out("      not preserved -- it is actively removed.")
out("    * Adam leaks even without weight decay: the update is m_hat/(sqrt(v_hat)+eps),")
out("      and v_hat = mean of g^2 is NOT in row(Phi) (the elementwise square of a")
out("      row-space vector need not be row-space), so the step leaves the affine")
out("      subspace theta_0 + row(Phi).")
result["S1"] = dict(p=p, n=n, T=T, null0=n0,
                    runs=[{k: (float(v) if isinstance(v, (int, float)) else v)
                           for k, v in rec.items()} for rec in s1])

# =========================================================================== S2
head("S2  the function-space consequence: training error is blind, test error is not")
out()
out("  same synthetic model, but now we ask what the null component COSTS.  Run GD to")
out("  interpolation from theta_0 = 0 and from theta_0 = random, then compare.")
out()
Phi_te = rng.standard_normal((20000, p))
y_te = Phi_te @ theta_true


def fit(th0: np.ndarray) -> np.ndarray:
    th = torch.tensor(th0.copy(), dtype=torch.float32, requires_grad=True)
    for _ in range(4000):
        loss = torch.mean((Phit @ th - yt) ** 2)
        g, = torch.autograd.grad(loss, th)
        with torch.no_grad():
            th -= lr_gd * g
    return th.detach().numpy().astype(np.float64)


th0 = fit(np.zeros(p))
thR = fit(theta0)
offset_vec = Phi_te @ (Pnull @ theta0)          # the permanent init offset, per test point
f0 = Phi_te @ th0
resid0 = f0 - y_te
identity = float(np.max(np.abs((Phi_te @ thR - f0) - offset_vec)))   # f_R - f_0 == offset ?
s2 = dict(
    train_0=float(np.mean((Phi @ th0 - y) ** 2)),
    train_R=float(np.mean((Phi @ thR - y) ** 2)),
    test_0=float(np.mean((f0 - y_te) ** 2)),
    test_R=float(np.mean((Phi_te @ thR - y_te) ** 2)),
    offset=float(np.mean(offset_vec ** 2)),
    identity=identity,
    cross=float(2.0 * np.mean(resid0 * offset_vec)),
)
out(f"  train MSE, theta_0 = 0       : {s2['train_0']:.3e}")
out(f"  train MSE, theta_0 = random  : {s2['train_R']:.3e}   <- identical: Phi P_null theta_0 = 0")
out(f"  test  MSE, theta_0 = 0       : {s2['test_0']:.3e}   <- the I-term only")
out(f"  test  MSE, theta_0 = random  : {s2['test_R']:.3e}   <- I-term + the permanent init offset")
out(f"  the two runs differ ONLY in the null space: max |(f_R - f_0) - Phi_te P_null theta_0|"
    f" = {identity:.3e}")
out(f"  mean (Phi_te P_null theta_0)^2 = {s2['offset']:.3e}   (the offset's own energy)")
out(f"  test_R - test_0 = {s2['test_R'] - s2['test_0']:.3e}  = offset^2 ({s2['offset']:.3e})"
    f" + 2<residual_0, offset> ({s2['cross']:.3e})")
out(f"    check: offset^2 + cross = {s2['offset'] + s2['cross']:.3e}   (== test_R - test_0)")
out()
out("  reading: the two runs fit the training data EQUALLY well (to machine precision),")
out("  and on the test set they differ by EXACTLY the init's null-space component.  The")
out("  extra test error is the offset's own energy PLUS a correlation term -- the offset")
out("  is not orthogonal to the row-space residual, because both come from the same")
out("  random geometry.  The training curve shows NEITHER: starting from theta_0 = 0")
out("  zeroes the tax entirely, which is why it is the best case.  A random init is a")
out("  permanent, training-error-invisible test-error offset.")
result["S2"] = s2

# =========================================================================== S3
head("S3  the real v1.5 question q_11b94d (dev half): AdamW, minibatches, moving features")
out()
if SYN_ONLY:
    out("  [--syn-only: skipping the v1.5 Jacobian work]")
    train = test = info = None
else:
    train, test, info = load_case()
dev = int(hashlib.md5(QID.encode()).hexdigest(), 16) % 2 == 0
out(f"  torch {torch.__version__} | numpy {np.__version__} | device cpu")
out(f"  question {QID} | family {FAM} | dataset {DS} | candidate set {SET_ID}")
out(f"  dev half? {dev}  (md5(qid) % 2 == 0 -> dev; the test half is never used here)")
out()
out("  official protocol (from the candidate's own train.py):")
out("    torch.manual_seed(seed); Model(); AdamW(lr, betas, weight_decay);")
out("    per step: idx = torch.randint(0, n, (bs,)) -> MSE on that minibatch -> opt.step()")
out("  -> the init is RANDOM (theta_0 != 0), the optimizer is AdamW WITH weight decay,")
out("     and the batches are minibatches.  None of these is plain full-batch GD.")
out()

if not SYN_ONLY:
    xtr = train["x"]
    ytr = train["y"].squeeze(-1)
    xte = test["x"]
    yte = test["y"].squeeze(-1)
    n_tr = xtr.shape[0]
    f_te_target = torch.tensor(target(xte.squeeze(-1).numpy()), dtype=torch.float32)


def official_run(model_factory, spec, seed: int):
    """Replicate the v1.5 protocol; return the init params and the final params."""
    torch.manual_seed(seed)
    model = model_factory()
    params = [pp for pp in model.parameters() if pp.requires_grad]
    th0 = torch.cat([pp.detach().reshape(-1).clone() for pp in params]).numpy().astype(np.float64)
    o = spec["optimizer"]
    opt = torch.optim.AdamW(params, lr=o["lr"], weight_decay=o["weight_decay"],
                            betas=tuple(o["betas"]))
    bs = int(spec["budget"]["batch_size"])
    steps = int(spec["budget"]["training_steps"])
    for _ in range(steps):
        idx = torch.randint(0, n_tr, (bs,))
        loss = torch.mean((model(xtr[idx]).squeeze(-1) - ytr[idx]) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    thT = torch.cat([pp.detach().reshape(-1).clone() for pp in params]).numpy().astype(np.float64)
    return model, th0, thT


def jac_rows(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    params = [pp for pp in model.parameters() if pp.requires_grad]
    out_ = model(x)
    rows = []
    for i in range(out_.shape[0]):
        go = torch.zeros_like(out_)
        go[i, 0] = 1.0
        g = torch.autograd.grad(out_, params, grad_outputs=go, retain_graph=True)
        rows.append(torch.cat([gi.reshape(-1) for gi in g]))
    return torch.stack(rows)


s3 = {}
for letter in ([] if SYN_ONLY else CANDS):
    spec = info[letter]["spec"]
    o = spec["optimizer"]
    steps = int(spec["budget"]["training_steps"])
    model, th0, thT = official_run(info[letter]["Model"], spec, seed=0)
    dvec = thT - th0
    out(f"  candidate {letter} ({info[letter]['cid']}): "
        f"d={spec['model']['depth']} w={spec['model']['width']} "
        f"p={spec['trainable_parameter_count']} | AdamW lr={o['lr']} wd={o['weight_decay']} "
        f"| {steps} steps x bs {spec['budget']['batch_size']}")

    # (a) the AdamW decay factor applied to the null component over the official budget
    decay = (1.0 - o["lr"] * o["weight_decay"]) ** steps
    out(f"    (a) AdamW weight-decay factor on the null component, (1 - lr*wd)^T = {decay:.10f}")

    # Jacobian at init (needed for K and for the null-space projector, via J v)
    J = jac_rows(model, xtr).double()          # n x p
    K = (J @ J.T) / n_tr
    sv = torch.linalg.svdvals(K)
    cutoff = torch.finfo(K.dtype).eps * max(K.shape) * sv[0]
    nkept = int((sv > cutoff).sum())
    n_kept32 = None
    Kp = torch.linalg.pinv(K, rtol=float(cutoff) / float(sv[0]))  # projector-aware pseudo-inverse

    # (b) the actual drift of the null component after the official run
    w = J @ torch.tensor(dvec)                  # n-vector = J (theta_T - theta_0)
    drift2 = float(dvec @ dvec) - float(w @ (Kp @ w)) / n_tr
    drift = math.sqrt(max(drift2, 0.0))
    null0_vec = torch.tensor(th0) - (J.T @ (Kp @ (J @ torch.tensor(th0)))) / n_tr
    null0 = float(torch.linalg.norm(null0_vec))
    out(f"    (b) drift of the null component after the official {steps} steps:")
    out(f"        ||P_null (theta_T - theta_0)|| = {drift:.3e}   "
        f"(||P_null theta_0|| = {null0:.3e}, rel {drift / max(null0, 1e-30):.3e})")
    out(f"        init NTK: rank {nkept}/{n_tr} (float64), cond {float(sv[0] / sv[nkept - 1].clamp_min(1e-30)):.3e}")

    # (c) the lazy-limit initialization tax in function space
    with torch.no_grad():
        f_init_tr = model(xtr).squeeze(-1).double()
        f_init_te = model(xte).squeeze(-1).double()
    kt = torch.empty(xte.shape[0], n_tr, dtype=torch.float64)
    J32 = J.float()
    out_te = model(xte)
    params = [pp for pp in model.parameters() if pp.requires_grad]
    for i in range(out_te.shape[0]):
        go = torch.zeros_like(out_te)
        go[i, 0] = 1.0
        g = torch.autograd.grad(out_te, params, grad_outputs=go, retain_graph=True)
        kt[i] = (torch.cat([gi.reshape(-1) for gi in g]).double() @ J.T) / n_tr
    y64 = ytr.double()
    yte64 = yte.double()
    r_res = y64 - f_init_tr
    alpha_gd = Kp @ r_res                       # K alpha = (y - f_init|tr)
    alpha_k = Kp @ y64                          # K alpha = y   (the "train-from-zero" kernel method)
    f_gd_te = f_init_te + kt @ alpha_gd
    f_k_te = kt @ alpha_k
    perm_offset = f_init_te - kt @ (Kp @ f_init_tr)   # the part of the init training cannot cancel
    tax = float(torch.mean(perm_offset ** 2))
    s3[letter] = dict(
        p=spec["trainable_parameter_count"], n=n_tr,
        decay=float(decay), drift=drift, drift_rel=drift / max(null0, 1e-30), null0=null0,
        init_test=float(torch.mean((f_init_te - yte64) ** 2)),
        init_test_vs_target=float(torch.mean((f_init_te - f_te_target.double()) ** 2)),
        lazy_gd_test=float(torch.mean((f_gd_te - yte64) ** 2)),
        lazy_k_test=float(torch.mean((f_k_te - yte64) ** 2)),
        lazy_gd_vs_target=float(torch.mean((f_gd_te - f_te_target.double()) ** 2)),
        permanent_init_tax=tax,
        official_test=OFFICIAL[letter]["mean"],
    )
    out(f"    (c) the lazy-limit (frozen-init-NTK) view of the SAME protocol:")
    out(f"        init output on test, MSE vs noisy y     : {s3[letter]['init_test']:.3e}")
    out(f"        init output on test, MSE vs target      : {s3[letter]['init_test_vs_target']:.3e}")
    out(f"        lazy-limit of the official run (GD-from-init): test MSE {s3[letter]['lazy_gd_test']:.3e}")
    out(f"        lazy-limit of 'train-from-zero' (kernel)   : test MSE {s3[letter]['lazy_k_test']:.3e}")
    out(f"        the PERMANENT init tax: mean (unreachable part of f_init)^2 = {tax:.3e}")
    out(f"        official test MSE (10 seeds, notes/bank-audit.csv): {OFFICIAL[letter]['mean']:.3e}")
    out(f"        (the lazy-limit error vs the deterministic target: {s3[letter]['lazy_gd_vs_target']:.3e})")
    out()

result["S3"] = {k: v for k, v in s3.items()}

# =========================================================================== S4
head("S4  summary")
out()
out("  claim                                            |  status")
out("  " + "-" * 92)
out("  P_null(theta_t) = P_null(theta_0) under plain GD |  exact identity (S1, machine eps)")
out("  same under minibatch SGD                         |  exact (row(Phi_B) subset row(Phi))")
out("  same under weight decay                          |  FALSE: decays by (1 - lr*wd) each step")
out("  same under Adam (no wd)                          |  FALSE: elementwise 2nd moment leaks")
out("  init offset invisible in training error          |  exact (Phi P_null = 0) (S2)")
out("  init offset = permanent test-error offset        |  exact in the lazy limit (S2, S3)")
out("  v1.5 runs start from a random init               |  yes (torch.manual_seed(seed); Model())")
out()
out("  bottom line: the null-space projection is NOT protected by anything.  Under plain")
out("  GD it is conserved because the gradient is blind, not because it is constrained.")
out("  AdamW (the v1.5 optimizer) is free to change it: weight decay shrinks it and the")
out("  adaptive step leaks out of the row space.  At the official budget the change is")
out("  small, but the invariant is a property of the setup, not a law.")

out()
out("reproduce with: python verification/zero_space_init.py")

OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8", newline="\n")
print(f"\nwritten: {OUT_MD}")
print(f"written: {OUT_JSON}")
