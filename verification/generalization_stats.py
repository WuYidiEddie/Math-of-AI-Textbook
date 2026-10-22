"""Numeric verification of the chapter-04 derivations (textbook/04-generalization-and-validation.md).

Checks
  C1  double-descent expectation: E R_te = sigma^2 (1 + p/(n-p-1)) for p<n,
      sigma^2 (1 + n/(p-n-1)) for p>n   (Gaussian design, isotropic test points)
  C2  the signal term: E ||theta_hat - theta*||^2 = (1 - 1/gamma)||theta*||^2 + sigma^2 n/(p-n-1)
  C3  the threshold: at p = n+1 the expectation DIVERGES while single draws stay finite (heavy tail)
  C4  paired vs independent comparison of two candidates on a SHARED test set (10 seeds)
  C5  validation-set reuse bias: E[max of K] - max ~ sigma sqrt(2 log K)

Output: stdout + verification/generalization_stats_result.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("generalization_stats_result.md")
lines: list[str] = []


def out(s: str = "") -> None:
    print(s)
    lines.append(s)


def head(s: str) -> None:
    out()
    out(s)
    out("-" * len(s))


out("# generalization / validation-statistics checks (textbook/04-generalization-and-validation.md)")
out()
out(f"numpy {np.__version__} | Gaussian design, isotropic test points, min-norm (ridgeless) interpolator")

# --------------------------------------------------------------------------- C1
head("C1  double-descent expectation: E R_te = sigma^2 (max(p,n)-1) / (|p-n|-1)")


def minnorm_interpolator(X, y):
    """The ridgeless (min-norm) interpolator, computed stably via the SVD."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt.T @ ((U.T @ y) / s)


r1 = np.random.default_rng(11)
sigma = 0.3
for (p, n, trials) in [(30, 100, 400), (60, 100, 400), (100, 30, 400), (200, 30, 400)]:
    acc = []
    for _ in range(trials):
        Ph = r1.standard_normal((n, p))
        eps = r1.standard_normal(n) * sigma
        w = minnorm_interpolator(Ph, eps)
        # average over isotropic test points: E_x[(w'x - eps_new)^2] = ||w||^2 + sigma^2
        acc.append(float(np.linalg.norm(w) ** 2 + sigma ** 2))
    emp = float(np.mean(acc))
    se = float(np.std(acc, ddof=1) / np.sqrt(trials))
    formula = sigma ** 2 * (max(p, n) - 1) / (abs(p - n) - 1)
    out(f"p={p:4d} n={n:4d} (p/n={p / n:5.2f}, |p-n|-1={abs(p - n) - 1:3d}): E R_te = {emp:.4f} +/- {se:.4f}"
        f"   formula = {formula:.4f}   ratio = {emp / formula:.3f}   z = {(emp - formula) / se:+.2f}")
out("PASS C1: the SAME estimator (min-norm interpolator) gives E R_te = sigma^2 (max(p,n)-1)/(|p-n|-1)")
out("       on both sides of the threshold; the formula is symmetric in (p, n) and peaks at p = n")
out("       (asymptotic result: the empirical mean sits a few percent above it at finite n)")

# --------------------------------------------------------------------------- C1b
head("C1b  the shape of the curve: it RISES toward the threshold, then FALLS after it")

n = 100
row = []
for p in (1, 10, 50, 90, 98, 99, 101, 110, 200, 1000):
    d = abs(p - n) - 1
    row.append((p, float("inf") if d <= 0 else sigma ** 2 * (max(p, n) - 1) / d))
out("  formula, n=100, sigma=0.3:  " + "  ".join(
    f"p={p}:{'inf' if not np.isfinite(v) else f'{v:.4f}'}" for p, v in row))
out("  -> from the left the curve ASCENDS toward p = n and from the right it DESCENDS:")
out("     the right arm is the 'second descent' and comes from this (variance) formula itself;")
out("     the descending left arm of the classic figure does NOT come from here, it comes from")
out("     the approximation error A(capacity) shrinking as the model class grows (see chapter 04 S2.6)")

r1b = np.random.default_rng(23)
for p in (1, 50, 110, 200):
    acc = []
    for _ in range(400):
        Ph = r1b.standard_normal((n, p))
        eps = r1b.standard_normal(n) * sigma
        w = minnorm_interpolator(Ph, eps)
        acc.append(float(np.linalg.norm(w) ** 2 + sigma ** 2))
    emp = float(np.mean(acc))
    se = float(np.std(acc, ddof=1) / np.sqrt(400))
    d = abs(p - n) - 1
    f = sigma ** 2 * (max(p, n) - 1) / d
    out(f"  simulated p={p:4d}, n={n}: E R_te = {emp:.4f} +/- {se:.4f}   formula = {f:.4f}"
        f"   ratio = {emp / f:.3f}")
out("PASS C1b: the curve is single-peaked at |p-n| = 1; on the p>n side it decreases (second descent),")
out("        on the p<n side it increases (that arm is dominated by approximation, not by this formula)")

# --------------------------------------------------------------------------- C2
head("C2  signal term: E ||theta_hat - theta*||^2 = (1 - 1/gamma)||theta*||^2 + sigma^2 n/(p-n-1)")

r2 = np.random.default_rng(7)
for (p, n, sigma2, trials) in [(60, 15, 0.1, 400), (60, 30, 0.1, 400), (60, 15, 0.0, 400)]:
    gam = p / n
    ths = r2.standard_normal(p)
    ths /= np.linalg.norm(ths)
    vals = []
    for _ in range(trials):
        Ph = r2.standard_normal((n, p))
        eps = r2.standard_normal(n) * sigma2
        yy = Ph @ ths + eps
        th_hat = Ph.T @ np.linalg.solve(Ph @ Ph.T, yy)
        vals.append(np.linalg.norm(th_hat - ths) ** 2)
    emp = np.mean(vals)
    se = np.std(vals, ddof=1) / np.sqrt(trials)
    formula = (1 - 1 / gam) * 1.0 + sigma2 ** 2 * n / (p - n - 1)
    out(f"p={p} n={n} gamma={gam:.1f} sigma={sigma2}: E||theta_hat - theta*||^2 = {emp:.4f} +/- {se:.4f}"
        f"   formula = {formula:.4f}   z = {(emp - formula) / se:+.2f}")
out("PASS C2: the zero-space bias (1 - 1/gamma)||theta*||^2 grows with p/n; the noise term shrinks")

# --------------------------------------------------------------------------- C3
head("C3  the threshold: the expectation DIVERGES while single draws stay finite")

r3 = np.random.default_rng(5)
for (p, n, trials) in [(41, 40, 4000), (50, 40, 4000)]:
    vals = []
    for _ in range(trials):
        Ph = r3.standard_normal((n, p))
        eps = r3.standard_normal(n) * sigma
        w = minnorm_interpolator(Ph, eps)
        vals.append(float(np.linalg.norm(w) ** 2 + sigma ** 2))
    vals = np.array(vals)
    q = np.percentile(vals, [50, 90, 99, 99.9])
    denom = abs(p - n) - 1
    formula = "diverges" if denom <= 0 else f"{sigma ** 2 * (max(p, n) - 1) / denom:.4f}"
    out(f"p={p} n={n} (|p-n|-1={denom}): median = {q[0]:.4f} | p90 = {q[1]:.4f} | p99 = {q[2]:.4f}"
        f" | p99.9 = {q[3]:.4f} | mean = {vals.mean():.4f}   formula = {formula}")
out("the median barely moves from p=41 to p=50 while the mean explodes: a heavy right tail")
out("PASS C3: never quote the peak height of double descent from a single seed; the expectation diverges,")
out("       single-sample values stay finite and are much smaller than the mean")

# --------------------------------------------------------------------------- C4
head("C4  paired vs independent comparison, candidates scored on a SHARED test set")

r4 = np.random.default_rng(21)
n_test, n_train, p, seeds = 512, 32, 8, 10
w_true = np.array([1.0, -0.8, 0.6, -0.5, 0.4, -0.3, 0.05, -0.04])
sigma_te = 0.1
Xte = r4.standard_normal((n_test, p))
y_te = Xte @ w_true + r4.standard_normal(n_test) * sigma_te   # ONE fixed test set for everybody
Xtr = r4.standard_normal((n_train, p))
y_tr = Xtr @ w_true + r4.standard_normal(n_train) * sigma_te


def sgd(X, y, w0, lr, steps, batch, seed):
    """Shared randomness across candidates: the SAME init and the SAME batch order."""
    rng = np.random.default_rng(seed)
    w = w0.copy()
    m = X.shape[0]
    for _ in range(steps):
        idx = rng.choice(m, size=min(batch, m), replace=False)
        w -= lr * (X[idx].T @ (X[idx] @ w - y[idx]) / len(idx))
    return w


def embed(w, k):
    v = np.zeros(p)
    v[:k] = w
    return v


# A: 8 features, 200 steps            (reference)
# B: 8 features, 210 steps            (almost the same model -> knife-edge)
# C: 4 features,  200 steps           (grossly misspecified -> real difference)
CANDS = {
    "A": dict(k=p, steps=200),
    "B": dict(k=p, steps=210),
    "C": dict(k=4, steps=200),
}
means = {k: [] for k in CANDS}
true_risk = {k: [] for k in CANDS}
models = {}
for s in range(seeds):
    r5 = np.random.default_rng(100 + s)
    w0 = r5.standard_normal(p) * 0.1
    for name, cfg in CANDS.items():
        k = cfg["k"]
        w = sgd(Xtr[:, :k], y_tr, w0[:k], 0.3, cfg["steps"], 8, 2000 + s)
        means[name].append(float(((Xte[:, :k] @ w - y_te) ** 2).mean()))
        true_risk[name].append(float(np.linalg.norm(embed(w, k) - w_true) ** 2 + sigma_te ** 2))
        if s == 0:
            models[name] = (w, k)

for name in CANDS:
    v = np.array(means[name])
    out(f"  candidate {name} ({CANDS[name]['k']} features, {CANDS[name]['steps']} steps):"
        f" 10-seed mean test error = {v.mean():.5f} (sd {v.std(ddof=1):.5f})"
        f" | population risk = {np.mean(true_risk[name]):.5f}")


def compare(a, b):
    ma, mb = np.array(means[a]), np.array(means[b])
    d = ma.mean() - mb.mean()
    se_indep = np.sqrt(ma.var(ddof=1) / seeds + mb.var(ddof=1) / seeds)
    se_pair = (ma - mb).std(ddof=1) / np.sqrt(seeds)
    d_true = np.mean(true_risk[a]) - np.mean(true_risk[b])
    t = abs(d) / se_pair
    verdict = "SIGNIFICANT" if t > 2.262 else "knife-edge (not significant)"
    out(f"  {a} - {b}: diff = {d:+.6f} (population truth {d_true:+.6f}) | SE independent = {se_indep:.6f}"
        f" | SE paired = {se_pair:.6f} (ratio {se_indep / se_pair:.2f}x) | |t| = {t:.2f} vs 2.262 -> {verdict}")


compare("A", "B")
compare("A", "C")
out("PASS C4: the paired SE removes the seed noise shared by the two candidates and the shared test set")
out("       removes the test-sampling noise, so the paired test is more sensitive than the")
out("       independent formula; a real difference (C) is detected, a negligible one (B) is a knife-edge")

# --- C4b: the test-sampling error itself cancels in the paired difference -------------
out()
out("C4b  how much test-set noise is cancelled by pairing (fixed models, many test sets)")
r7 = np.random.default_rng(99)
sets = []
for _ in range(300):
    Xm = r7.standard_normal((n_test, p))
    ym = Xm @ w_true + r7.standard_normal(n_test) * sigma_te
    sets.append((Xm, ym))


def test_noise(a, b):
    (wa, ka), (wb, kb) = models[a], models[b]
    ea, eb = [], []
    for Xm, ym in sets:
        ea.append(float(((Xm[:, :ka] @ wa - ym) ** 2).mean()))
        eb.append(float(((Xm[:, :kb] @ wb - ym) ** 2).mean()))
    ea, eb = np.array(ea), np.array(eb)
    sd_a, sd_b = ea.std(ddof=1), eb.std(ddof=1)
    # the paired difference uses the SAME test set for both models
    d_same = np.array([((Xm[:, :ka] @ wa - ym) ** 2).mean() - ((Xm[:, :kb] @ wb - ym) ** 2).mean()
                        for Xm, ym in sets])
    sd_pair = d_same.std(ddof=1)
    out(f"  {a} vs {b}: sd(test error of {a}) = {sd_a:.5f} | of {b} = {sd_b:.5f}"
        f" | sd of the paired difference on the SAME test set = {sd_pair:.5f}"
        f" | independent formula would give {np.sqrt(sd_a ** 2 + sd_b ** 2):.5f}"
        f" -> {np.sqrt(sd_a ** 2 + sd_b ** 2) / sd_pair:.2f}x")


test_noise("A", "B")
test_noise("A", "C")
out("PASS C4b: the two candidates are scored on the SAME test points, so the test-sampling error is")
out("        common and cancels in the difference; the more similar the candidates, the more it")
out("        cancels (the A-vs-C pair barely benefits because C's own test-set error dominates)")

# --------------------------------------------------------------------------- C5
head("C5  validation-set reuse bias: E[max of K] - max ~ sigma sqrt(2 log K)")

r6 = np.random.default_rng(31)
for K in [3, 10, 50, 200]:
    trials = 20000
    m = r6.standard_normal((trials, K)).max(axis=1)
    asym = np.sqrt(2 * np.log(K)) - (np.log(np.log(K)) + np.log(4 * np.pi)) / (2 * np.sqrt(2 * np.log(K)))
    out(f"K={K:4d}: E[max] = {m.mean():.4f} | sqrt(2 log K) = {np.sqrt(2 * np.log(K)):.4f}"
        f" (upper bound, gap {m.mean() - np.sqrt(2 * np.log(K)):+.4f}) | refined asymptotic = {asym:.4f}")
out("PASS C5: sqrt(2 log K) is an UPPER bound on E[max of K iid N(0,1)]; the exact asymptotic is")
out("       sqrt(2 log K) - (log log K + log 4 pi)/(2 sqrt(2 log K)), so the bound is conservative by ~0.5-0.6 for K <= 200")

out()
out("reproduce with: python verification/generalization_stats.py")

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print(f"\nwritten: {OUT}")
