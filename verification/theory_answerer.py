# theory_answerer.py — theory-based answerer for the v1.5 question bank.
#
# Idea (04-benchmark-testbed.md §4, "先算再比"): replace each candidate network by its
# *tangent linearization at initialization* (NTK/lazy model)
#
#     f_lin(theta)(x) = f(x; theta0) + J(x) (theta - theta0),   J = d f/d theta |_{theta0}
#
# then run the candidate's EXACT training protocol on this linearized model:
# same seed (same init, same minibatch sequence via torch.randint), same optimizer
# (SGD/Adam/AdamW/Adagrad/RMSprop with the spec's lr/wd/betas/momentum), same steps.
# The predicted test metric of the linearized model is the theory's answer to
# "what does this candidate score if it stays in the lazy regime".
#
# Where the real candidate IS lazy, this prediction should match the measured
# 10-seed mean; where it is not, |prediction - measurement| IS the feature-learning
# term (measurable, per-candidate). See 03-frontier.md §3.
#
# Families: univariate/multivariate regression (MSE) + tabular/xor/spiral (CE)
# use an explicit Jacobian (n x C x p).  bigram LM candidates (outputs B x L x V,
# explicit J too large) use the on-the-fly JVP(finite-difference)/VJP variant in
# answer_candidate_lm — same tangent model, same exact protocol replica.
# Companion analytic theory for bigram: notes/bigram-estimation.md.
#
# Usage:
#   python verification/theory_answerer.py --limit-per-family 2        # smoke test
#   python verification/theory_answerer.py --families univariate_regression,...
# Output: JSONL (one line per candidate), flushed incrementally.

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.func import functional_call, jacrev

ROOT = Path(__file__).resolve().parents[2]  # D:\research\Science of AI Textbook
DATA = ROOT / "v1.5" / "v15-500q-6d4ae9b" / "data"

FAM_DIRS = {
    "univariate_regression": "univariate_regression",
    "multivariate_regression": "multivariate_regression",
    "synthetic_tabular_classification": "synthetic_tabular_classification",
    "xor_classification": "xor_classification",
    "spiral_classification": "spiral_classification",
    "bigram_lm": "bigram_lm",
}
MAX_JAC_ELEMENTS = 7.5e8  # ~3 GB float32 for train J; big candidates use a smaller test subsample


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_optimizer_from_spec(theta: torch.Tensor, o: dict):
    typ = o["type"]
    kw = dict(lr=o["lr"], weight_decay=o.get("weight_decay", 0.0))
    if typ in ("AdamW", "Adam"):
        kw["betas"] = tuple(o.get("betas", (0.9, 0.999)))
        cls = torch.optim.AdamW if typ == "AdamW" else torch.optim.Adam
    elif typ == "SGD":
        kw["momentum"] = o.get("momentum", 0.0)
        cls = torch.optim.SGD
    elif typ == "RMSprop":
        cls = torch.optim.RMSprop
    elif typ == "Adagrad":
        cls = torch.optim.Adagrad
    else:
        raise ValueError(f"unsupported optimizer {typ}")
    return cls([theta], **kw)


def theta_vector(model: torch.nn.Module):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def make_f_of_theta(model: torch.nn.Module, shapes, sizes):
    names = [k for k, _ in model.named_parameters()]

    def f(theta, x):
        ps = {}
        off = 0
        for k, sh, sz in zip(names, shapes, sizes):
            ps[k] = theta[off:off + sz].view(sh)
            off += sz
        return functional_call(model, ps, (x,))

    return f


def loss_grad_pred(pred: torch.Tensor, y: torch.Tensor, loss_id: str) -> torch.Tensor:
    """dL/dpred for the official mean losses, on the current minibatch."""
    if loss_id == "mse":
        pr = pred.reshape(-1)
        yr = y.reshape(-1).to(pr.dtype)
        return 2.0 * (pr - yr) / pr.numel()
    if loss_id == "cross_entropy":
        n = pred.shape[0]
        probs = torch.softmax(pred, dim=1)
        probs[torch.arange(n), y.long()] -= 1.0
        return probs / n
    raise ValueError(loss_id)


def metric_of(pred: torch.Tensor, y: torch.Tensor, loss_id: str) -> float:
    if loss_id == "mse":
        return float(torch.mean((pred.reshape(-1) - y.reshape(-1).to(pred.dtype)) ** 2))
    return float(F.cross_entropy(pred, y.long()))


def answer_candidate_lm(model_cls, spec, train, test, seed, eps_rel=1e-3):
    """Linearized (lazy) replica for LM candidates (bigram): outputs are (B, L, V),
    so an explicit Jacobian is too large.  Instead the tangent model
        f_lin(theta) = f(theta0) + J (theta - theta0)
    is evaluated on the fly per minibatch:
      J d   ~= (f(theta0 + s d) - f(theta0)) / s     (finite difference, s tiny)
      J^T g = grad_theta <g, f(theta0)>              (reverse mode, exact)
    Consistency: <gp, J d> vs <J^T gp, d> checked once per candidate.
    Everything else (seed -> init -> randint minibatch stream -> optimizer -> steps)
    replicates the official runner exactly."""
    t0 = time.time()
    torch.manual_seed(seed)
    model = model_cls()
    shapes = [tuple(p.shape) for p in model.parameters()]
    sizes = [int(p.numel()) for p in model.parameters()]
    names = [k for k, _ in model.named_parameters()]
    p_total = sum(sizes)
    theta0 = theta_vector(model)
    f = make_f_of_theta(model, shapes, sizes)

    def params_of(theta):
        ps, off = {}, 0
        for k, sh, sz in zip(names, shapes, sizes):
            ps[k] = theta[off:off + sz].view(sh)
            off += sz
        return ps

    tx, ty = train["x"], train["y"]
    vx, vy = test["x"], test["y"]
    n = tx.shape[0]
    steps = int(spec["budget"]["training_steps"])
    bs = int(spec["budget"]["batch_size"])

    def ce_grad(pred, yb):
        pr = pred.reshape(-1, pred.shape[-1])
        yf = yb.reshape(-1).long()
        probs = torch.softmax(pr, dim=1)
        probs[torch.arange(pr.shape[0]), yf] -= 1.0
        return (probs / pr.shape[0]).reshape(pred.shape)

    f0_cache = {}

    def f0_of(idx):
        key = idx.tolist()
        key = tuple(key)
        if key not in f0_cache:
            with torch.no_grad():
                f0_cache[key] = f(theta0, tx[idx])
        return f0_cache[key]

    theta = theta0.clone().requires_grad_(True)
    opt = build_optimizer_from_spec(theta, spec["optimizer"])
    norm0 = max(float(theta0.norm()), 1e-3)
    fd_rel_err = None

    for step_i in range(steps):
        idx = torch.randint(0, n, (bs,))  # same RNG stream as the official runner
        d = theta.detach() - theta0
        f0b = f0_of(idx)
        with torch.no_grad():
            s = eps_rel * norm0 / max(float(d.norm()), 1e-12)
            jt = (f(theta0 + s * d, tx[idx]) - f0b) / s
            pred = f0b + jt
        gp = ce_grad(pred, ty[idx])
        ps = {k: v.clone().requires_grad_(True) for k, v in params_of(theta0).items()}
        out = functional_call(model, ps, (tx[idx],))
        dot = (out * gp).sum()
        grads = torch.autograd.grad(dot, list(ps.values()))
        theta.grad = torch.cat([g.reshape(-1) for g in grads])
        if step_i == 1:  # step 0 has d=0; check FD consistency on a real displacement
            lhs = float((gp * jt).sum())
            rhs = float((theta.grad * d).sum())
            den = float(gp.norm() * d.norm()) + 1e-30
            fd_rel_err = abs(lhs - rhs) / den
        opt.step()

    # test metric of the linearized model on all test windows
    d = theta.detach() - theta0
    s = eps_rel * norm0 / max(float(d.norm()), 1e-12)
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, vx.shape[0], 50):
            xb, yb = vx[i:i + 50], vy[i:i + 50]
            f0b = f(theta0, xb)
            pred = f0b + (f(theta0 + s * d, xb) - f0b) / s
            pr = pred.reshape(-1, pred.shape[-1])
            loss = F.cross_entropy(pr, yb.reshape(-1).long(), reduction="sum")
            tot += float(loss)
            cnt += pr.shape[0]
        # init metric
        init_tot = 0.0
        for i in range(0, vx.shape[0], 50):
            xb, yb = vx[i:i + 50], vy[i:i + 50]
            pr0 = f(theta0, xb).reshape(-1, pred.shape[-1])
            init_tot += float(F.cross_entropy(pr0, yb.reshape(-1).long(), reduction="sum"))

    return {
        "skipped": "",
        "p": p_total, "n": int(n), "C": int(pred.shape[-1]),
        "steps": steps, "bs": bs,
        "opt": spec["optimizer"]["type"], "lr": spec["optimizer"]["lr"],
        "wd": spec["optimizer"].get("weight_decay", 0.0),
        "loss": "cross_entropy",
        "lambda_max": float("nan"), "lambda_mean": float("nan"), "eff_rank": float("nan"),
        "eta_lambda_max": float("nan"),
        "init_metric": init_tot / cnt, "pred_metric": tot / cnt,
        "param_rel_disp": float(d.norm() / theta0.norm().clamp_min(1e-12)),
        "fd_consistency": fd_rel_err,
        "elapsed_s": round(time.time() - t0, 2),
    }


def answer_candidate(model_cls, spec, train, test, seed, lambda_probes=8):
    t0 = time.time()
    torch.manual_seed(seed)
    model = model_cls()
    shapes = [tuple(p.shape) for p in model.parameters()]
    sizes = [int(p.numel()) for p in model.parameters()]
    p_total = sum(sizes)
    theta0 = theta_vector(model)
    f = make_f_of_theta(model, shapes, sizes)

    tx, ty = train["x"], train["y"]
    vx, vy = test["x"], test["y"]
    n = tx.shape[0]
    loss_id = spec["loss"]["loss_id"]
    steps = int(spec["budget"]["training_steps"])
    bs = int(spec["budget"]["batch_size"])

    with torch.no_grad():
        f0_tr = f(theta0, tx)          # (n, C) or (n, 1)
    if f0_tr.ndim == 1:
        f0_tr = f0_tr[:, None]
    C = f0_tr.shape[1]

    if n * C * p_total > MAX_JAC_ELEMENTS:
        return {"skipped": f"jacobian too large: {n}x{C}x{p_total}"}

    # test-side subsample (test metric is a mean; a few hundred rows suffice for argmin
    # purposes and the full n_test jacrev otherwise dominates runtime/memory)
    nte = vx.shape[0]
    n_te_sub = 512 if n * C * p_total <= 4.2e8 else 256
    if nte > n_te_sub:
        gte = torch.Generator().manual_seed(0)
        te_idx = torch.randperm(nte, generator=gte)[:n_te_sub]
    else:
        te_idx = None
    vx_sub = vx[te_idx] if te_idx is not None else vx
    with torch.no_grad():
        f0_te = f(theta0, vx_sub)
    if f0_te.ndim == 1:
        f0_te = f0_te[:, None]

    J = jacrev(lambda th: f(th, tx).reshape(-1))(theta0).reshape(n, C, p_total)
    Jte = jacrev(lambda th: f(th, vx_sub).reshape(-1))(theta0).reshape(vx_sub.shape[0], C, p_total)

    # kernel diagnostics at init: K = J J^T / n via matvecs (never form K)
    def Kv(vec):  # vec: (n, C)
        w = torch.einsum("ncp,nc->p", J, vec)
        return torch.einsum("ncp,p->nc", J, w) / n

    g = torch.Generator().manual_seed(0)
    v = torch.randn(n, C, generator=g)
    v = v / v.norm()
    lam = 0.0
    for _ in range(12):
        u = Kv(v)
        nv = u.norm()
        if nv == 0:
            break
        v = u / nv
    lam = float((v * Kv(v)).sum())
    trK = float(J.pow(2).sum()) / n
    trK2 = float(torch.stack([Kv(torch.randn(n, C, generator=g)).pow(2).sum()
                              for _ in range(2)]).mean())  # Hutchinson: E||K g||^2 = tr K^2
    lam_mean = trK / (n * C)
    eff_rank = trK ** 2 / max(trK2, 1e-30)

    theta = theta0.clone().requires_grad_(True)
    opt = build_optimizer_from_spec(theta, spec["optimizer"])

    for _ in range(steps):
        idx = torch.randint(0, n, (bs,))  # same RNG stream as the official runner
        d = theta.detach() - theta0
        pred = f0_tr[idx] + torch.einsum("bcp,p->bc", J[idx], d)
        gp = loss_grad_pred(pred, ty[idx], loss_id).reshape(bs, C)
        theta.grad = torch.einsum("bcp,bc->p", J[idx], gp)
        opt.step()

    with torch.no_grad():
        d = theta.detach() - theta0
        pred_te = f0_te + torch.einsum("mcp,p->mc", Jte, d)
        vy_sub = vy[te_idx] if te_idx is not None else vy
        pred_metric = metric_of(pred_te, vy_sub, loss_id)
        init_metric = metric_of(f0_te, vy_sub, loss_id)
        disp = float(d.norm() / theta0.norm().clamp_min(1e-12))

    return {
        "skipped": "",
        "p": p_total, "n": int(n), "C": int(C),
        "steps": steps, "bs": bs,
        "opt": spec["optimizer"]["type"], "lr": spec["optimizer"]["lr"],
        "wd": spec["optimizer"].get("weight_decay", 0.0),
        "loss": loss_id,
        "lambda_max": lam, "lambda_mean": lam_mean, "eff_rank": eff_rank,
        "eta_lambda_max": float(spec["optimizer"]["lr"]) * lam,
        "init_metric": init_metric, "pred_metric": pred_metric,
        "param_rel_disp": disp,
        "elapsed_s": round(time.time() - t0, 2),
    }


def iter_questions(limit_per_family, families):
    coll = json.loads((ROOT / "v1.5" / "v15-500q-6d4ae9b" / "collection.json").read_text(encoding="utf-8"))
    per_fam = {}
    for qpath in coll["question_paths"]:
        fam = qpath.split("/")[1]
        if families and fam not in families:
            continue
        if limit_per_family and per_fam.get(fam, 0) >= limit_per_family:
            continue
        qdir = DATA / qpath
        q = json.loads((qdir / "question.json").read_text(encoding="utf-8"))
        per_fam[fam] = per_fam.get(fam, 0) + 1
        yield fam, qpath, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="")
    ap.add_argument("--limit-per-family", type=int, default=0)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", default=str(Path(__file__).with_name("theory_answerer.jsonl")))
    ap.add_argument("--skip-done", action="store_true",
                    help="skip (qid, letter, seed) already present in --out")
    args = ap.parse_args()
    families = [s for s in args.families.split(",") if s] or None
    seeds = [int(s) for s in args.seeds.split(",")]

    done = set()
    if args.skip_done and Path(args.out).exists():
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if not r.get("skipped"):
                        done.add((r["qid"], r["letter"], r["seed"]))
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"skip-done: {len(done)} records")

    out = open(args.out, "a", encoding="utf-8", newline="\n")
    done_q = 0
    for fam, qpath, q in iter_questions(args.limit_per_family, families):
        qid = q["question_id"]
        ds_id = q["dataset_id"]
        base = DATA / "datasets" / fam / ds_id
        train = torch.load(base / "train.pt", map_location="cpu", weights_only=True)
        test = torch.load(base / "test.pt", map_location="cpu", weights_only=True)
        for ch in q["choices"]:
            cdir = ROOT / "v1.5" / "v15-500q-6d4ae9b" / "data" / ch["candidate_path"]
            model_mod = load_module(cdir / "model.py", f"m_{ch['candidate_id']}")
            spec = json.loads((cdir / "candidate_spec.json").read_text(encoding="utf-8"))
            for seed in seeds:
                if (qid, ch["letter"], seed) in done:
                    continue
                try:
                    if fam == "bigram_lm":
                        r = answer_candidate_lm(model_mod.Model, spec, train, test, seed)
                    else:
                        r = answer_candidate(model_mod.Model, spec, train, test, seed)
                except Exception as e:  # keep the sweep going; record failure
                    r = {"skipped": f"error: {type(e).__name__}: {e}"}
                r.update({
                    "qid": qid, "family": fam, "letter": ch["letter"],
                    "candidate_id": ch["candidate_id"], "seed": seed,
                    "correct_letter": q["correct_letter"],
                    "qtype": q.get("type", ""),
                })
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                out.flush()
                print(qid, ch["letter"], "seed", seed,
                      "->", r.get("pred_metric", r.get("skipped")), flush=True)
        done_q += 1
        print(f"[done] {done_q} questions ({qid})", flush=True)
    out.close()


if __name__ == "__main__":
    main()
