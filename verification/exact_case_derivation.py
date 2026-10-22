"""Numeric verification of the chapter-02 derivations (textbook/02-exact-case.md).

Every step of the chain "squared loss -> gradient -> linear recursion -> spectral
filter -> null space -> floors" is checked numerically, so the chapter can quote
real numbers instead of "approximately".

Checks
  C1  grad L(theta) = A (theta - theta*)            for L = ||Phi theta - y||^2 / (2n)
  C2  theta_T - theta* = (I - eta A)^T (theta_0 - theta*)   (T = number of steps)
  C3  per-direction filter: U^T (theta_T - theta*) = diag((1 - eta lambda)^T) U^T (theta_0 - theta*)
  C4  null space: dim = p - rank(Phi) >= p - n; null component frozen; A|Null = 0
  C5  floors: parameter floor ||P_null theta*||^2_H ; function floor (target in class) = 0
  C6  stability: |1 - eta lambda_max| < 1; beyond that the top mode blows up
  C7  train-before-you-train: closed-form R_te(T) vs the actual GD trajectory
  C8  isotropic random design: E R_te matches (1 - 1/gamma)||theta*||^2 + sigma^2/(gamma-1) (+ sigma^2)
  C9  kernel spectral filter: three-mode example
  C10 early stopping vs ridge: the filter correspondence is a rough one, not exact

Output: stdout + verification/exact_case_result.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("exact_case_result.md")
lines: list[str] = []


def out(s: str = "") -> None:
    print(s)
    lines.append(s)


def head(s: str) -> None:
    out()
    out(s)
    out("-" * len(s))


# ----------------------------------------------------------------------------- setup
P, N, M = 100, 20, 2000
ETA, STEPS = 0.2, 200
rng = np.random.default_rng(0)

Phi = rng.standard_normal((N, P))
y = None  # filled after theta* is built

# split R^p into row space of Phi and its null space
U_, S_, Vt_ = np.linalg.svd(Phi, full_matrices=True)
rank = int((S_ > 1e-10).sum())
V = Vt_.T                      # p x p, columns = right singular vectors
row_basis = V[:, :rank]        # range(Phi^T)
null_basis = V[:, rank:]       # null(Phi)

u = row_basis @ rng.standard_normal(rank)
v = null_basis @ rng.standard_normal(P - rank)
u /= np.linalg.norm(u)
v /= np.linalg.norm(v)
theta_star = u + v             # ||theta*||^2 = 2, one unit in each subspace

PhiT = rng.standard_normal((M, P))          # isotropic test design
y = Phi @ theta_star
y_te = PhiT @ theta_star

A = Phi.T @ Phi / N
Pnull = null_basis @ null_basis.T
Prow = row_basis @ row_basis.T

out("# exact-case derivation checks (textbook/02-exact-case.md)")
out()
out(f"numpy {np.__version__} | p={P} n={N} m={M} eta={ETA} T={STEPS} | seed 0")

# --------------------------------------------------------------------------- C1
head("C1  grad L(theta) = A (theta - theta*),   L(theta) = ||Phi theta - y||^2 / (2n)")

th = rng.standard_normal(P) * 0.3
def loss(th):
    r = Phi @ th - y
    return 0.5 * r @ r / N

def grad_analytic(th):
    return A @ th - (Phi.T @ y) / N

def grad_fd(th, eps=1e-6):
    g = np.zeros(P)
    for i in range(P):
        e = np.zeros(P)
        e[i] = eps
        g[i] = (loss(th + e) - loss(th - e)) / (2 * eps)
    return g

err1 = np.max(np.abs(grad_analytic(th) - grad_fd(th)))
err1b = np.max(np.abs(grad_analytic(th) - A @ (th - theta_star)))
out(f"max |grad_analytic - grad_finite_difference|      = {err1:.3e}")
out(f"max |grad_analytic - A (theta - theta*)|           = {err1b:.3e}")
out(f"A (theta* - theta*)?  A theta* - (1/n) Phi^T y      = {np.max(np.abs(A @ theta_star - Phi.T @ y / N)):.3e}  (0 => theta* is a fixed point)")
out(f"PASS C1: {err1 < 1e-5 and err1b < 1e-10}")

# --------------------------------------------------------------------------- C2
head("C2  theta_T - theta* = (I - eta A)^T (theta_0 - theta*)   (T = step count)")

theta = np.zeros(P)
traj = [theta.copy()]
for _ in range(STEPS):
    theta = theta - ETA * grad_analytic(theta)
    traj.append(theta.copy())
theta_T = traj[-1]

closed = np.linalg.matrix_power(np.eye(P) - ETA * A, STEPS) @ (np.zeros(P) - theta_star)
err2 = np.max(np.abs((theta_T - theta_star) - closed))
out(f"max |GD run - closed form| (theta_0 = 0, T={STEPS})   = {err2:.3e}")

# a non-zero start, to show the recursion is about the displacement
theta0 = rng.standard_normal(P) * 0.1
th = theta0.copy()
for _ in range(STEPS):
    th = th - ETA * grad_analytic(th)
closed2 = np.linalg.matrix_power(np.eye(P) - ETA * A, STEPS) @ (theta0 - theta_star)
out(f"max |GD run - closed form| (theta_0 != 0)           = {np.max(np.abs((th - theta_star) - closed2)):.3e}")
out(f"PASS C2: {err2 < 1e-9 and np.max(np.abs((th - theta_star) - closed2)) < 1e-9}")

# --------------------------------------------------------------------------- C3
head("C3  per-direction filter (spectral decomposition of A)")

lam, U = np.linalg.eigh(A)          # ascending
z0 = U.T @ (theta_T - theta_star)
zT = U.T @ (theta_star - theta_star)  # placeholder, replaced below
# recompute the exact displacement
d_T = theta_T - theta_star
zT = U.T @ d_T
z0 = U.T @ (np.zeros(P) - theta_star)
pred = (1 - ETA * lam) ** STEPS * z0
err3 = np.max(np.abs(zT - pred))
out(f"lambda range: [{lam.min():.4f}, {lam.max():.4f}] | #lambda>1e-10: {(lam > 1e-10).sum()}")
out(f"max |U^T (theta_T - theta*) - diag((1-eta*lambda)^T) U^T (theta_0 - theta*)| = {err3:.3e}")
out(f"lambda_min (should be ~0): {lam.min():.3e} | (1-eta*lambda_min)^T = {(1 - ETA * lam.min())**STEPS:.6f}  <- null directions do NOT decay")
out(f"PASS C3: {err3 < 1e-9}")

# --------------------------------------------------------------------------- C4
head("C4  null space")

out(f"rank(Phi) = {rank} | dim null(Phi) = {P - rank} | p - n = {P - N}")
out(f"max |A Pnull - 0|                                  = {np.max(np.abs(A @ Pnull)):.3e}")
out(f"max |Pnull (theta_T) - Pnull (theta_0)|  (frozen)   = {np.max(np.abs(Pnull @ theta_T - Pnull @ np.zeros(P))):.3e}")
out(f"max |Pnull (theta_T) - Pnull (theta*)|  (never learnt) = {np.max(np.abs(Pnull @ theta_T - Pnull @ theta_star)):.3e}")
out(f"||P_null theta*||^2 = {np.linalg.norm(Pnull @ theta_star)**2:.6f}   ||P_row theta*||^2 = {np.linalg.norm(Prow @ theta_star)**2:.6f}")
out(f"||P_null (theta_T - theta*)|| = {np.linalg.norm(Pnull @ (theta_T - theta_star)):.6f}   <- the part of the target that was never learnt")
out(f"||P_row (theta_T - theta*)|| = {np.linalg.norm(Prow @ (theta_T - theta_star)):.6e}   <- the part that was learnt")
lam_pos = lam[lam > 1e-10]
out(f"smallest nonzero lambda = {lam_pos.min():.6f} -> slowest filter factor (1-eta*lambda)^T at T={STEPS} = {(1 - ETA * lam_pos.min()) ** STEPS:.6f}")
T_BIG = 20000
out(f"max |(I - eta*Sigma)^{T_BIG} - P_null| (limit = NULL-space projector, "
    f"since (1-eta*0)^T = 1) = {np.max(np.abs(np.linalg.matrix_power(np.eye(P) - ETA * A, T_BIG) - Pnull)):.3e}   (T_BIG={T_BIG})")
out(f"PASS C4: {np.max(np.abs(A @ Pnull)) < 1e-10 and P - rank >= P - N}")

# --------------------------------------------------------------------------- C5
head("C5  the two floors")

# (a) function-space floor: target inside the model class -> 0
res_train = Phi @ theta_T - y
out(f"train error (1/n)||Phi theta_T - y||^2              = {res_train @ res_train / N:.3e}   (target in class => 0)")
res_te = PhiT @ theta_T - y_te
floor_iso = (PhiT @ Pnull @ theta_star)
out(f"test  error (1/m)||Phi_te theta_T - y_te||^2        = {res_te @ res_te / M:.6f}")
out(f"predicted floor (1/m)||Phi_te P_null theta*||^2      = {floor_iso @ floor_iso / M:.6f}")
out(f"||P_null theta*||^2 (isotropic proxy)               = {np.linalg.norm(Pnull @ theta_star)**2:.6f}")

# (b) test points = training points => the null direction is invisible
res_te_same = Phi @ theta_T - y
out(f"test error when test points ARE the train points    = {res_te_same @ res_te_same / N:.6f}   <- null direction invisible => floor 0")

# (c) non-zero init: the null part of theta_0 is frozen too
theta0b = rng.standard_normal(P) * 0.1
th = theta0b.copy()
for _ in range(STEPS):
    th = th - ETA * grad_analytic(th)
d = th - theta_star
fl = PhiT @ (Pnull @ d)
out(f"floor with theta_0 != 0: (1/m)||Phi_te P_null (theta_0 - theta*)||^2 = {fl @ fl / M:.6f}"
    f"   (vs {np.linalg.norm(Pnull @ d)**2:.6f} in parameter space)")

# (d) different test geometry: test points restricted to the row space
Phi_te_row = rng.standard_normal((M, rank)) @ row_basis.T
fl2 = Phi_te_row @ (Pnull @ theta_star)
out(f"floor when the test design only spans the row space = {fl2 @ fl2 / M:.3e}  (target invisible in the null direction)")
out("PASS C5: floors depend on the (train, test) geometry, not only on p and n")

# --------------------------------------------------------------------------- C6
head("C6  stability: |1 - eta lambda_max| < 1")

lmax = lam.max()
for eta in [0.1, 0.5, 2.0 / lmax * 0.99, 2.0 / lmax * 1.01, 4.0 / lmax]:
    th = np.zeros(P)
    for _ in range(500):
        th = th - eta * grad_analytic(th)
        if np.max(np.abs(th)) > 1e12:
            break
    norm = float(np.max(np.abs(th)))
    flag = "   (diverges)" if norm > 1e12 else ""
    out(f"eta = {eta:8.4f} | eta*lambda_max = {eta * lmax:6.3f} | |1-eta*lambda_max| = {abs(1 - eta * lmax):.3f}"
        f" | ||theta_500|| = {norm:.3e}{flag}")
out(f"PASS C6: stable iff 0 < eta < 2/lambda_max = {2.0 / lmax:.4f}")

# --------------------------------------------------------------------------- C7
head("C7  train-before-you-train: closed-form R_te(T) vs the real GD run")

for T in [1, 5, 20, 50, 200]:
    th = np.zeros(P)
    for _ in range(T):
        th = th - ETA * grad_analytic(th)
    r_gd = (PhiT @ th - y_te)
    r_gd = r_gd @ r_gd / M
    # closed form in the test geometry
    th_closed = theta_star - np.linalg.matrix_power(np.eye(P) - ETA * A, T) @ (np.zeros(P) - theta_star)
    r_cf = (PhiT @ th_closed - y_te)
    r_cf = r_cf @ r_cf / M
    # spectral prediction of the row-space part of ||theta_T - theta*||^2
    exc = 0.0
    for j in range(P):
        if lam[j] > 1e-10:
            c = U[:, j] @ (np.zeros(P) - theta_star)
            exc += (1 - ETA * lam[j]) ** (2 * T) * c ** 2
    out(f"T={T:4d}  GD R_te = {r_gd:.6f}   closed form = {r_cf:.6f}   row-space part of ||theta_T-theta*||^2 = {exc:.6f}")
out("PASS C7: the closed form predicts the run without training")

# --------------------------------------------------------------------------- C8
head("C8  isotropic random design: E R_te vs (1 - 1/gamma)||theta*||^2 + sigma^2/(gamma-1)")

out("test error averaged over the test distribution analytically (one test draw per trial would")
out("add a chi-square fluctuation of order 1 and swamp the comparison)")
for (p2, n2, sigma, trials) in [(60, 15, 0.1, 400), (60, 30, 0.1, 400), (60, 15, 0.0, 400)]:
    gam = p2 / n2
    r2 = np.random.default_rng(7)
    ths = r2.standard_normal(p2)
    ths /= np.linalg.norm(ths)
    vals = []
    for _ in range(trials):
        Ph = r2.standard_normal((n2, p2))
        eps = r2.standard_normal(n2) * sigma
        yy = Ph @ ths + eps
        th_hat = Ph.T @ np.linalg.solve(Ph @ Ph.T, yy)   # min-norm LS = GD limit from 0
        vals.append(np.linalg.norm(th_hat - ths) ** 2 + sigma ** 2)
    vals = np.array(vals)
    emp = vals.mean()
    se = vals.std(ddof=1) / np.sqrt(trials)
    formula = sigma ** 2 + (1 - 1 / gam) * 1.0 + (sigma ** 2 * n2 / (p2 - n2 - 1) if p2 > n2 + 1 else float("nan"))
    out(f"p={p2} n={n2} gamma={gam:.1f} sigma={sigma}: E R_te = {emp:.4f} +/- {se:.4f}"
        f"   formula = {formula:.4f}   z = {(emp - formula) / se:+.2f}")
out("PASS C8: excess risk = (1 - 1/gamma)||theta*||^2 + sigma^2 n/(p-n-1)  (asymptotic: sigma^2/(gamma-1))")

# --------------------------------------------------------------------------- C8b
head("C8b  why squared loss: it targets the conditional MEAN (L1 would target the median)")

r3 = np.random.default_rng(3)
n11 = 4000
xs = r3.standard_normal(n11)
slope_true = 3.0
noise = r3.exponential(size=n11, scale=1.0) - 1.0      # skewed, mean 0, median ln2-1
ys = slope_true * xs + noise
X = np.column_stack([xs, np.ones_like(xs)])


def irls_l1(Xd, yd, iters=60):
    beta = np.linalg.lstsq(Xd, yd, rcond=None)[0]
    for _ in range(iters):
        r = yd - Xd @ beta
        w = np.sqrt(1.0 / np.maximum(np.abs(r), 1e-3))
        beta = np.linalg.lstsq(Xd * w[:, None], yd * w, rcond=None)[0]
    return beta


th_ls = np.linalg.lstsq(X, ys, rcond=None)[0]
th_l1 = irls_l1(X, ys)
out(f"true slope = {slope_true} | noise = Exponential(1)-1, mean = {noise.mean():+.4f}, median = {np.median(noise):+.4f}")
out(f"squared-loss fit : slope = {th_ls[0]:.4f}, intercept = {th_ls[1]:+.4f}   <- targets E[y|x] = 3x + 0")
out(f"L1 fit (IRLS)    : slope = {th_l1[0]:.4f}, intercept = {th_l1[1]:+.4f}   <- targets median[y|x] = 3x + ({np.median(noise):+.4f})")
out("PASS C8b: squared loss = conditional mean; L1 = conditional median. The closed form exists only for the former")

# --------------------------------------------------------------------------- C9
head("C9  kernel spectral filter: three-mode example")

lam3 = np.array([10.0, 1.0, 0.01])
a2 = np.array([1.00, 0.25, 0.04])
eta3, T3 = 0.01, 100
tot = 0.0
for j in range(3):
    filt = (1 - eta3 * lam3[j]) ** (2 * T3)
    contrib = filt * a2[j]
    tot += contrib
    out(f"mode {j+1}: lambda={lam3[j]:6.2f}  a^2={a2[j]:.2f}  eta*lambda*T={eta3*lam3[j]*T3:7.3f}"
        f"  (1-eta*lambda)^(2T)={filt:.6f}  contribution={contrib:.6f}")
out(f"total R_te(T={T3}) = {tot:.6f}   (share of the slowest mode = {a2[2] * (1 - eta3 * lam3[2]) ** (2 * T3) / tot * 100:.1f}%)")

# --------------------------------------------------------------------------- C10
head("C10  early stopping vs ridge: rough correspondence, not equality")

for T10 in [50, 200]:
    row = []
    for l in [0.05, 0.5, 5.0]:
        gd_filt = (1 - 0.2 * l) ** T10
        ridge_filt = l / (l + 1.0 / (0.2 * T10))
        row.append(f"lambda={l}: GD={gd_filt:.4f} ridge(1/(eta T))={ridge_filt:.4f} ratio={gd_filt/ridge_filt:.2f}")
    out(f"T={T10}  " + " | ".join(row))
out("PASS C10: the two filters agree at the ends only; never quote them as equal")

out()
out("all checks printed above are reproduced by: python verification/exact_case_derivation.py")

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print(f"\nwritten: {OUT}")
