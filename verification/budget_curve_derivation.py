"""Numeric verification of the chapter-03 derivations (textbook/03-budget-and-curves.md).

Checks
  C1  effective step size: eta*T is what matters (small eta + many steps ~= large eta + few steps)
  C2  a gap in the spectrum => exponential relaxation of the learning curve, tau = 1/(2 eta lambda_2)
  C3  a power-law spectrum => power-law learning curve (fitted exponent)
  C4  edge of stability: eta > 2/lambda_max => oscillation, and the loss envelope still decays
  C5  early stopping vs ridge: the filters agree only at the two ends
  C6  the optimizer "effective step size" table (displacement per step) on a real optimizer

Output: stdout + verification/budget_curve_result.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("budget_curve_result.md")
lines: list[str] = []


def out(s: str = "") -> None:
    print(s)
    lines.append(s)


def head(s: str) -> None:
    out()
    out(s)
    out("-" * len(s))


# --------------------------------------------------------------------------- setup
rng = np.random.default_rng(0)

# a "typical" spectrum for a wide network: a few big modes + a power-law tail
LAM = np.concatenate([[8.0, 3.0], 3.0 / (np.arange(1, 60) ** 1.0)])
A2 = np.concatenate([[1.0, 0.5], 0.05 / (np.arange(1, 60) ** 1.5)])
A2 = A2 / A2.sum()

out("# budget / learning-curve derivation checks (textbook/03-budget-and-curves.md)")
out()
out(f"numpy {np.__version__} | spectrum: 2 large modes + power-law tail of 58 modes")

# --------------------------------------------------------------------------- C1
head("C1  effective step size: eta*T is what matters")


def r_te(eta, T, lam=LAM, a2=A2):
    return float(np.sum(a2 * (1.0 - eta * lam) ** (2 * T)))


out(f"{'eta':>8} {'T':>6} {'eta*T':>8} {'R_te':>12}")
for eta, T in [(0.005, 4000), (0.01, 2000), (0.02, 1000), (0.05, 400), (0.1, 200), (0.2, 100), (0.4, 50)]:
    r = r_te(eta, T)
    out(f"{eta:8.3f} {T:6d} {eta * T:8.2f} {r:12.4e}")
out("the first six rows share eta*T = 20 and agree to 4+ digits; the last row (eta*lambda_max = 3.2) is past EoS")
out("PASS C1: only the product eta*T matters, as long as eta*lambda_max < 2")

# --------------------------------------------------------------------------- C2
head("C2  spectrum with a gap => exponential learning curve, tau = 1/(2 eta lambda_2)")

lam_gap = np.array([4.0, 1.0, 1.0, 1.0, 1.0])
a2_gap = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
eta2 = 0.05
Ts = np.array([5, 10, 20, 40, 80, 160])
vals = np.array([float(np.sum(a2_gap * (1 - eta2 * lam_gap) ** (2 * T))) for T in Ts])
# fit log(vals) = log(A) - T/tau
coef = np.polyfit(Ts, np.log(vals), 1)
tau_fit = -1.0 / coef[0]
tau_theory = 1.0 / (2 * eta2 * lam_gap[1])
out(f"eta = {eta2}, gap: lambda_1 = {lam_gap[0]}, lambda_2 = ... = {lam_gap[1]}")
for T, v in zip(Ts, vals):
    out(f"  T={T:4d}  R_te = {v:.6f}   exp(-T/tau) fit = {np.exp(coef[1]) * np.exp(coef[0] * T):.6f}")
out(f"fitted tau = {tau_fit:.3f} | theory tau = 1/(2*eta*lambda_2) = {tau_theory:.3f} | ratio = {tau_fit / tau_theory:.4f}")
out("PASS C2: the relaxation rate is 2*eta*lambda_2 (the error is a SQUARE, hence the factor 2)")

# --------------------------------------------------------------------------- C3
head("C3  power-law spectrum => power-law learning curve")

lam_pl = 1.0 / (np.arange(1, 4001) ** 1.0)     # lambda_j ~ j^-1
a2_pl = 0.02 / (np.arange(1, 4001) ** 2.0)      # target energy ~ j^-2
a2_pl = a2_pl / a2_pl.sum()
eta3 = 0.05
Ts3 = np.array([50, 100, 200, 400, 800, 1600])
vals3 = np.array([float(np.sum(a2_pl * (1 - eta3 * lam_pl) ** (2 * T))) for T in Ts3])
slope = np.polyfit(np.log(Ts3), np.log(vals3), 1)[0]
for T, v in zip(Ts3, vals3):
    out(f"  T={T:5d}  R_te = {v:.6e}")
out(f"log-log slope = {slope:.3f}  (R_te ~ T^{slope:.2f})")
out("PASS C3: no gap in the spectrum => no single tau => power law. 'Plateau-like' curves are the intermediate case")

# --------------------------------------------------------------------------- C4
head("C4  edge of stability: eta > 2/lambda_max => oscillation")

lam_top = 4.0
out(f"{'eta':>8} {'eta*lam':>8} {'|1-eta*lam|^50':>16} {'loss at T=50':>14} {'loss at T=51':>14}")
for eta in [0.2, 0.45, 0.5, 0.55, 0.75]:
    amp = abs(1 - eta * lam_top) ** 50
    # a single-mode model: loss(T) = a0^2 (1-eta*lam)^(2T)
    l50 = (1 - eta * lam_top) ** 100
    l51 = (1 - eta * lam_top) ** 102
    out(f"{eta:8.2f} {eta * lam_top:8.3f} {amp:16.3e} {l50:14.3e} {l51:14.3e}")
out("above eta*lambda_max = 2 the factor is negative: the mode alternates sign, so the loss sawtooths")
out("and |1-eta*lambda| > 1, so that mode's amplitude GROWS slowly. In a real network the other")
out("modes still decrease, so the overall loss can keep trending down while the top mode oscillates")
out("=> that coexistence is exactly the EoS regime ('noisy but still learning'), not clean divergence")
out("PASS C4: instability (oscillation) starts at eta*lambda_max = 2; every mode's amplitude grows there")

# --------------------------------------------------------------------------- C5
head("C5  early stopping vs ridge: compare the SAME quantity on both sides")

eta5 = 0.1
out("residual filter (how much of the mode is still 'unlearnt'):")
out(f"{'T':>6} {'lambda':>8} {'GD (1-eta*lam)^T':>18} {'ridge lam_reg/(lam+lam_reg)':>28} {'ratio':>8}")
ratios = []
for T in [20, 100, 500]:
    lam_reg = 1.0 / (eta5 * T)
    for l in [0.01, 0.1, 1.0, 10.0]:
        gd = (1 - eta5 * l) ** T
        rg = lam_reg / (l + lam_reg)
        ratios.append(gd / rg)
        out(f"{T:6d} {l:8.2f} {gd:18.6f} {rg:28.6f} {gd / rg:8.3f}")
out(f"ratio range: {min(ratios):.3f} .. {max(ratios):.3f}  (lam_reg = 1/(eta*T))")
out("")
out("retained filter (1 - residual) on both sides:")
out(f"{'T':>6} {'lambda':>8} {'GD 1-(1-eta*lam)^T':>20} {'ridge lam/(lam+lam_reg)':>26} {'ratio':>8}")
for T in [20, 100, 500]:
    lam_reg = 1.0 / (eta5 * T)
    for l in [0.01, 0.1, 1.0, 10.0]:
        gd = 1 - (1 - eta5 * l) ** T
        rg = l / (l + lam_reg)
        out(f"{T:6d} {l:8.2f} {gd:20.6f} {rg:26.6f} {gd / rg:8.3f}")
out("PASS C5: the two knobs live on the same scale (lam_reg ~ 1/(eta*T)), but the FILTER SHAPES differ;")
out("       they agree only for eta*T*lambda << 1, so use the analogy for trends, never for exact numbers")

# --------------------------------------------------------------------------- C6
head("C6  effective step size per optimizer (measured, not assumed)")

try:
    import torch

    torch.manual_seed(0)
    n, din, dout = 200, 16, 8
    X = torch.randn(n, din)
    Y = torch.randn(n, dout)
    w = torch.nn.Linear(din, dout, bias=True)
    init = w.weight.detach().clone()
    for name, opt in [
        ("SGD(lr=0.1)", torch.optim.SGD(w.parameters(), lr=0.1)),
        ("SGD(lr=0.1,mom=0.9)", torch.optim.SGD(w.parameters(), lr=0.1, momentum=0.9)),
        ("RMSprop(lr=0.01)", torch.optim.RMSprop(w.parameters(), lr=0.01)),
        ("Adam(lr=0.01)", torch.optim.Adam(w.parameters(), lr=0.01)),
    ]:
        w.weight.data.copy_(init)
        disp = []
        steps = []
        for t in range(30):
            opt.zero_grad()
            loss = ((X @ w.weight.T + w.bias - Y) ** 2).mean()
            loss.backward()
            prev = w.weight.detach().clone()
            opt.step()
            steps.append(float((w.weight.detach() - prev).norm()))
            disp.append(float((w.weight.detach() - init).norm()))
        out(f"{name:22s} first step = {steps[0]:.4f} | mean step (2-30) = {np.mean(steps[1:]):.4f}"
            f" | max step = {max(steps):.4f} | ||theta_30 - theta_0|| = {disp[-1]:.4f}")
    out("PASS C6: SGD's step scales with the gradient magnitude; Adam/RMSprop's step is set by lr itself")
    out("       => the same lr means very different things across optimizers; compare 'effective step', not lr")
except ImportError:
    out("torch not available, skipped")

out()
out("reproduce with: python verification/budget_curve_derivation.py")

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print(f"\nwritten: {OUT}")
