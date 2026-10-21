# bigram_estimation.py — numerical verification of the bigram test-CE theory
# (notes/bigram-estimation.md). Pure numpy; no training of neural nets.
#
# What is verified:
#   P1  Exact identity: CE_test(phat) = H(Y|X) + sum_x pi_x KL(P_x || phat_x) + F,
#       with E[F] = 0 over test redraws and sd(F) matching the per-token variance formula.
#   P2  Estimation floor: expected population excess of the add-beta plug-in estimator,
#       compared against (a) first-order formula V(V-1)/(2N),
#       (b) exact per-cell binomial expectation. Checks where the formula holds/breaks.
#   P3  Optimization term: lookup-table softmax GD is per-context convex; excess(T) - excess(inf)
#       decays like 1/T and is bounded by Ls*||theta*||^2/(2T) up to a loose constant.
#   P4  Rank bottleneck: when d_model < V-1, the logit matrix cannot represent generic P;
#       SVD estimate of the induced approximation floor A (heuristic estimate).
#   P5  Real-data check on notes/bank-audit.csv: observed excess vs floor; bottleneck groups.
#
# Run:  python verification/bigram_estimation.py
# Output: stdout report + verification/bigram_estimation_result.json

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_JSON = os.path.join(HERE, "bigram_estimation_result.json")
BANK_CSV = os.path.join(HERE, "..", "notes", "bank-audit.csv")


# ---------------------------------------------------------------- v1.5 generator
def build_transition(vocab_size, alpha, table_seed):
    """Exact replica of the v1.5 bigram synthesizer."""
    rng = np.random.default_rng(table_seed)
    logits = rng.standard_normal((vocab_size, vocab_size)) * alpha
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = (probs / probs.sum(axis=1, keepdims=True)).astype(np.float64)
    pi = np.full(vocab_size, 1.0 / vocab_size, dtype=np.float64)
    for _ in range(256):
        pi = pi @ probs
        pi /= pi.sum()
    return probs, pi


def sample_sequences(n_seq, length, probs, pi, seed):
    """Exact replica of the v1.5 sequence sampler (length = ctx+1)."""
    rng = np.random.default_rng(seed)
    v = probs.shape[0]
    seq = np.empty((n_seq, length), dtype=np.int64)
    for i in range(n_seq):
        s = [int(rng.choice(v, p=pi))]
        while len(s) < length:
            s.append(int(rng.choice(v, p=probs[s[-1]])))
        seq[i] = np.asarray(s, dtype=np.int64)
    return seq


def cond_entropy(probs, pi):
    """H(Y|X) in nats, exact given the law."""
    with np.errstate(divide="ignore", invalid="ignore"):
        h_rows = -np.sum(np.where(probs > 0, probs * np.log(probs), 0.0), axis=1)
    return float(pi @ h_rows)


def pop_excess(probs, pi, phat):
    """sum_x pi_x KL(P_x || phat_x) in nats. phat must be strictly positive."""
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_rows = np.sum(np.where(probs > 0, probs * np.log(probs / phat), 0.0), axis=1)
    return float(pi @ kl_rows)


def seq_to_counts(seq, V):
    """Sequences [n_seq, L+1] -> count matrix C[x, y] over all L next-token pairs."""
    x = seq[:, :-1].ravel()
    y = seq[:, 1:].ravel()
    C = np.zeros((V, V), dtype=np.int64)
    np.add.at(C, (x, y), 1)
    return C


def add_beta_estimator(C, beta):
    n = C.sum(axis=1, keepdims=True)
    V = C.shape[0]
    return (C + beta) / (n + beta * V)


def test_ce(seq, phat):
    x = seq[:, :-1].ravel()
    y = seq[:, 1:].ravel()
    return float(-np.mean(np.log(phat[x, y])))


# ---------------------------------------------------------------- exact per-cell expectation
def binom_pmf(k, n, p):
    """log Binomial(k; n, p), vectorized over k (numpy array)."""
    k = np.asarray(k, dtype=np.float64)
    lg = math.lgamma(n + 1) - np.array([math.lgamma(kk + 1) for kk in k]) \
        - np.array([math.lgamma(n - kk + 1) for kk in k])
    with np.errstate(divide="ignore", invalid="ignore"):
        out = lg + k * np.log(p) + (n - k) * np.log1p(-p)
    out[~np.isfinite(out)] = -np.inf
    return out


def exact_excess_add_beta(probs, pi, N, beta):
    """E[ sum_x pi_x KL(P_x || phat_x) ] for phat = (C+beta)/(n_x+beta*V),
    with C_x, ~ Multinomial(n_x, P_x), n_x = round(N * pi_x) deterministic.
    Exact up to the deterministic-n_x approximation. Vectorized over k."""
    V = probs.shape[0]
    total = 0.0
    for x in range(V):
        n_x = int(round(N * pi[x]))
        if n_x <= 0:
            # unseen context: prediction stays at beta/(beta V) = 1/V
            total += pi[x] * float(np.sum(probs[x] * np.log(probs[x] * V)))
            continue
        k = np.arange(0, n_x + 1)
        for y in range(V):
            p = probs[x, y]
            if p <= 0:
                continue
            pmf = np.exp(binom_pmf(k, n_x, p))
            phat = (k + beta) / (n_x + beta * V)
            total += pi[x] * p * float(pmf @ np.log(p / phat))
    return total


# ---------------------------------------------------------------- P1: identity + F
def part1(V=32, alpha=0.8, L=16, table_seed=310777, n_train=800, n_test=200, redraws=200):
    probs, pi = build_transition(V, alpha, table_seed)
    H = cond_entropy(probs, pi)
    train = sample_sequences(n_train, L + 1, probs, pi, seed=1000)
    phat = add_beta_estimator(seq_to_counts(train, V), 0.5)
    e_pop = pop_excess(probs, pi, phat)

    ces, per_token_var = [], None
    all_scores = []
    for r in range(redraws):
        te = sample_sequences(n_test, L + 1, probs, pi, seed=5000 + r)
        x = te[:, :-1].ravel()
        y = te[:, 1:].ravel()
        s = -np.log(phat[x, y])
        all_scores.append(s)
        ces.append(float(s.mean()))
    ces = np.array(ces)
    M = n_test * L
    all_scores = np.stack(all_scores)  # [redraws, M]
    # lag-1 autocorrelation of per-token scores within the concatenated stream is
    # contaminated by sequence boundaries; estimate kappa via variance inflation instead.
    sd_iid = float(np.std(all_scores[0], ddof=1) / math.sqrt(M))
    sd_emp = float(np.std(ces, ddof=1))
    mean_ce = float(np.mean(ces))
    se_of_mean = sd_emp / math.sqrt(redraws)
    ok = abs(mean_ce - (H + e_pop)) <= 3 * se_of_mean
    return {
        "H": H, "e_pop_plugin": e_pop, "mean_ce": mean_ce,
        "identity_residual": mean_ce - (H + e_pop),
        "identity_tol_3se": 3 * se_of_mean,
        "sd_ce_empirical": sd_emp, "sd_ce_iid_formula": sd_iid,
        "sqrt_kappa": sd_emp / sd_iid,
        "pass": bool(ok),
    }


# ---------------------------------------------------------------- P2: estimation floor
def part2(configs, n_train=800, redraws=32, beta=0.5):
    out = []
    for (V, alpha, L) in configs:
        probs, pi = build_transition(V, alpha, table_seed=310777)
        H = cond_entropy(probs, pi)
        N = n_train * L
        sims = []
        for r in range(redraws):
            tr = sample_sequences(n_train, L + 1, probs, pi, seed=2000 + r)
            phat = add_beta_estimator(seq_to_counts(tr, V), beta)
            sims.append(pop_excess(probs, pi, phat))
        sims = np.array(sims)
        asym = V * (V - 1) / (2.0 * N)
        exact = exact_excess_add_beta(probs, pi, N, beta)
        mc_se = float(np.std(sims, ddof=1) / math.sqrt(redraws))
        ok = abs(float(np.mean(sims)) - exact) <= 3 * mc_se
        out.append({
            "V": V, "alpha": alpha, "L": L, "N": N, "H": H,
            "sim_mean": float(np.mean(sims)), "sim_mc_se": mc_se,
            "asymptotic": asym, "exact_binomial": exact,
            "ratio_asym_over_exact": asym / exact,
            "pass": bool(ok),
        })
    return out


# ---------------------------------------------------------------- P3: optimization term
def part3(V=32, alpha=0.8, L=16, table_seed=310777, n_train=800,
          gamma=1e-3, t_grid=(8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096), t_long=60000):
    """Lookup-table softmax GD with weight decay gamma (as in the v1.5 protocol).
    Objective: mean token CE + (gamma/2)||theta||^2. With gamma>0 the minimizer
    theta*_gamma is finite (wd = implicit smoothing toward uniform), the problem is
    Ls-smooth and gamma-strongly-convex, and the standard convex GD bound applies:
        L(theta_T) - L(theta*_gamma) <= Ls ||theta*_gamma||^2 / (2T),  eta = 1/Ls.
    We check: (a) the train-loss gap obeys the bound; (b) the *population* excess
    gap decays with log-log slope ~ -1; (c) the converged excess vs the add-beta floor."""
    probs, pi = build_transition(V, alpha, table_seed)
    train = sample_sequences(n_train, L + 1, probs, pi, seed=1000)
    C = seq_to_counts(train, V).astype(np.float64)
    n_x = C.sum(axis=1)
    N = float(C.sum())
    q = np.divide(C, n_x[:, None], out=np.full_like(C, 1.0 / V), where=n_x[:, None] > 0)

    Ls = float(n_x.max() / (2.0 * N))          # smoothness of mean CE (per-row Hessian <= n_x/(2N))
    eta = 0.04 / Ls                            # deliberately sub-optimal step: exposes the 1/T phase
    w = n_x / N

    def softmax_rows(theta):
        z = theta - theta.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    def train_loss(theta):
        p = softmax_rows(theta)
        ce = 0.0
        for x in range(V):
            if n_x[x] > 0:
                ce += w[x] * float(-(q[x] * np.log(p[x])).sum())
        return ce + 0.5 * gamma * float(np.sum(theta ** 2))

    theta = np.zeros((V, V))
    curve, train_gaps = {}, {}
    targets = set(t_grid) | {t_long}
    for t in range(1, t_long + 1):
        p_hat = softmax_rows(theta)
        grad = w[:, None] * (p_hat - q) + gamma * theta
        theta -= eta * grad
        if t in targets:
            curve[t] = pop_excess(probs, pi, softmax_rows(theta))
            train_gaps[t] = train_loss(theta)
    theta_star = theta  # proxy for theta*_gamma after 60k steps
    star_norm2 = float(np.sum(theta_star ** 2))
    l_star = train_gaps[t_long]
    e_inf = curve[t_long]
    # add-beta floor for comparison (same train set)
    e_addbeta = pop_excess(probs, pi, add_beta_estimator(C, 0.5))

    rows = []
    prev_t, prev_gap = None, None
    for t in t_grid:
        gap = curve[t] - e_inf
        train_gap = train_gaps[t] - l_star
        # bound with the actually used step: gap <= ||theta*||^2 / (2 eta T)
        bound = star_norm2 / (2.0 * eta * t)
        slope = (math.log(gap) - math.log(prev_gap)) / (math.log(prev_t) - math.log(t)) \
            if prev_t is not None and gap > 1e-9 and prev_gap > 1e-9 else None
        rows.append({
            "T": t, "excess_pop": curve[t], "gap_over_Tinf": gap,
            "train_gap": train_gap, "bound": bound,
            "train_bound_ok": bool(train_gap <= bound * 1.0001),
            "ratio_bound_over_actual": bound / gap if gap > 0 else None,
            "loglog_slope": slope,
        })
        prev_t, prev_gap = t, gap
    # slope over the visible (pre-saturation) phase only
    visible = [r for r in rows if r["gap_over_Tinf"] > 1e-3 and r["loglog_slope"] is not None]
    slopes = [r["loglog_slope"] for r in visible]
    mean_slope = float(np.mean(slopes)) if slopes else float("nan")
    train_ok = all(r["train_bound_ok"] for r in rows)
    gaps = [r["gap_over_Tinf"] for r in rows]
    monotone = all(g2 <= g1 + 1e-12 for g1, g2 in zip(gaps, gaps[1:]))
    converged = gaps[-1] < 1e-3 * max(e_inf, 1e-12)
    has_power_phase = any(0.8 <= s <= 2.0 for s in slopes)
    return {
        "gamma": gamma, "Ls": Ls, "eta": eta, "theta_star_norm2": star_norm2,
        "e_inf_wd": e_inf, "e_addbeta_half": e_addbeta,
        "curve": rows, "mean_loglog_slope_visible": mean_slope,
        "train_bound_ok_all": bool(train_ok), "monotone": bool(monotone),
        "converged": bool(converged), "has_power_phase": bool(has_power_phase),
        "pass": bool(train_ok and monotone and converged and has_power_phase),
    }


# ---------------------------------------------------------------- P4: rank bottleneck
def part4():
    out = []
    for V, alpha in [(48, 0.8), (48, 1.0), (48, 1.4), (32, 1.0)]:
        probs, pi = build_transition(V, alpha, table_seed=310777)
        C = np.log(probs)
        C -= C.mean(axis=1, keepdims=True)          # per-row gauge fixing
        W = np.sqrt(pi)[:, None]
        U, S, Vt = np.linalg.svd(W * C, full_matrices=False)
        for d in sorted({32, min(64, V)}):
            Cd = (U[:, :d] * S[:d]) @ Vt[:d]
            Cd = Cd / np.sqrt(pi)[:, None]
            z = Cd - Cd.max(axis=1, keepdims=True)
            phat = np.exp(z)
            phat /= phat.sum(axis=1, keepdims=True)
            a_est = pop_excess(probs, pi, phat)
            out.append({"V": V, "alpha": alpha, "d_model": d, "A_est_nats": a_est,
                        "sv_top8": [float(s) for s in S[:8]]})
    return out


# ---------------------------------------------------------------- P5: real-data check
def part5():
    import csv
    rows = []
    with open(BANK_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("fam") != "bigram":
                continue
            rows.append(r)
    res = {"n_candidates": len(rows)}
    floors, excesses = [], []
    for r in rows:
        V = float(r["vocab"]); N = float(r["n_train_tokens"])
        floor = V * (V - 1) / (2 * N)
        excess = float(r["mean_metric"]) - float(r["bigram_H_cond"])
        floors.append(floor); excesses.append(excess)
    floors = np.array(floors); excesses = np.array(excesses)
    res["floor_min_med_max"] = [float(floors.min()), float(np.median(floors)), float(floors.max())]
    res["excess_min_med_max"] = [float(excesses.min()), float(np.median(excesses)), float(excesses.max())]
    res["frac_excess_below_floor"] = float(np.mean(excesses < floors))
    res["min_excess_over_floor"] = float(np.min(excesses / floors))

    # bottleneck: V=48, group by d_model ('w') 32 vs >=64
    v48 = [r for r in rows if float(r["vocab"]) == 48.0]
    g32 = [float(r["mean_metric"]) - float(r["bigram_H_cond"]) for r in v48 if r["w"] == "32"]
    g64 = [float(r["mean_metric"]) - float(r["bigram_H_cond"]) for r in v48 if r["w"] in ("64", "128")]
    res["V48_excess_w32"] = {"n": len(g32), "median": float(np.median(g32)) if g32 else None}
    res["V48_excess_w64_128"] = {"n": len(g64), "median": float(np.median(g64)) if g64 else None}
    # control: V=24/32
    vsmall = [r for r in rows if float(r["vocab"]) in (24.0, 32.0)]
    res["V24_32_excess_median"] = float(np.median(
        [float(r["mean_metric"]) - float(r["bigram_H_cond"]) for r in vsmall]))
    return res


def main():
    report = {}
    print("P1 identity + F ...")
    report["P1"] = part1()
    print("P2 estimation floor ...")
    report["P2"] = part2([(32, 0.8, 16), (48, 1.4, 12), (24, 1.0, 24)])
    print("P3 optimization term (60k GD steps) ...")
    report["P3"] = part3()
    print("P4 rank bottleneck ...")
    report["P4"] = part4()
    print("P5 bank-audit check ...")
    report["P5"] = part5()

    with open(RESULT_JSON, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    p1 = report["P1"]
    print("\n=== P1: identity CE = H + KL + F ===")
    print(f"H={p1['H']:.4f}  E_pop(plugin)={p1['e_pop_plugin']:.4f}  mean CE={p1['mean_ce']:.4f}")
    print(f"residual={p1['identity_residual']:.5f} (tol {p1['identity_tol_3se']:.5f})  PASS={p1['pass']}")
    print(f"sd(CE) emp={p1['sd_ce_empirical']:.4f}  iid formula={p1['sd_ce_iid_formula']:.4f}  sqrt(kappa)={p1['sqrt_kappa']:.2f}")

    print("\n=== P2: estimation floor (add-0.5 plug-in) ===")
    for r in report["P2"]:
        print(f"V={r['V']} a={r['alpha']} L={r['L']} N={r['N']} H={r['H']:.3f}: "
              f"sim={r['sim_mean']:.4f}+-{r['sim_mc_se']:.4f}  exact={r['exact_binomial']:.4f}  "
              f"asym={r['asymptotic']:.4f} (asym/exact={r['ratio_asym_over_exact']:.2f})  PASS={r['pass']}")

    print("\n=== P3: lookup softmax GD (with weight decay) ===")
    p3 = report["P3"]
    print(f"gamma={p3['gamma']} Ls={p3['Ls']:.2e} eta={p3['eta']:.1f} ||theta*||^2={p3['theta_star_norm2']:.1f}")
    print(f"converged excess (wd)={p3['e_inf_wd']:.4f}  add-0.5 plug-in excess={p3['e_addbeta_half']:.4f} "
          f"mean visible log-log slope={p3['mean_loglog_slope_visible']:.2f} PASS={p3['pass']}")
    for r in p3["curve"]:
        s = f"{r['loglog_slope']:.2f}" if r['loglog_slope'] is not None else "  -- "
        print(f"T={r['T']:>5}  excess={r['excess_pop']:.4f}  gap={r['gap_over_Tinf']:.4f}  "
              f"train_gap={r['train_gap']:.5f}  bound={r['bound']:.4f}  "
              f"train_bound_ok={r['train_bound_ok']}  slope={s}")

    print("\n=== P4: rank bottleneck (SVD estimate of A) ===")
    for r in report["P4"]:
        print(f"V={r['V']} a={r['alpha']} d_model={r['d_model']}: A_est={r['A_est_nats']:.4f} nats")

    print("\n=== P5: bank-audit real data ===")
    p5 = report["P5"]
    print(json.dumps(p5, indent=2))


if __name__ == "__main__":
    main()
