"""Lazy/NTK probe on representative v1.5 questions.

Purpose: measure whether the finite-width network stays in the lazy/NTK regime
during its stated budget.  This is a *measurement*, not a proof; it tells us
whether the exact kernel formula in 02-exact-regimes.md can be trusted for a
given candidate.

Usage:
    python "Math of AI Textbook/verification/lazy_probe.py" --qid q_11b94d
    python "Math of AI Textbook/verification/lazy_probe.py" --qid q_aef977
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from torch.func import functional_call, jvp

ROOT = Path(__file__).resolve().parents[2]  # D:\research\Science of AI Textbook
DATA = ROOT / "v1.5" / "v15-500q-6d4ae9b" / "data" / "datasets"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_case(fam_dir: str, ds_id: str, set_id: str, cand_id: str):
    base = DATA / fam_dir / ds_id
    train = torch.load(base / "train.pt", map_location="cpu", weights_only=True)
    test = torch.load(base / "test.pt", map_location="cpu", weights_only=True)
    cdir = base / "candidates" / set_id / cand_id
    model_mod = load_module(cdir / "model.py", f"m_{cand_id}")
    spec = json.loads((cdir / "candidate_spec.json").read_text(encoding="utf-8"))
    return train, test, model_mod.Model, spec


def flatten_params(model: torch.nn.Module):
    return {k: v.detach().clone().requires_grad_(True) for k, v in model.named_parameters()}


def forward_fn(model, params, x):
    return functional_call(model, params, (x,)).squeeze(-1)


def ntk_probe(model, params, x, v):
    """Return K v = J J^T v / n, with K the empirical NTK of the mean loss."""
    f = forward_fn(model, params, x)
    grads = torch.autograd.grad((f * v).sum(), tuple(params.values()))
    g = {k: g for k, g in zip(params.keys(), grads)}
    _, jg = jvp(lambda p: forward_fn(model, p, x), (params,), (g,))
    return jg / x.shape[0]


def estimate_lambda_max(model, params, x, iters=12, seed=0, n_probes=2):
    g = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    vals = []
    for _ in range(n_probes):
        v = torch.randn(n, generator=g)
        v = v / v.norm()
        for _ in range(iters):
            v = ntk_probe(model, params, x, v)
            nv = v.norm()
            if nv == 0:
                break
            v = v / nv
        vals.append(float((v * ntk_probe(model, params, x, v)).sum()))
    return max(vals)


def relative_kernel_change(model, params0, paramsT, x, n_probes=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    rel, cos = [], []
    for _ in range(n_probes):
        v = torch.randn(n, generator=g)
        k0 = ntk_probe(model, params0, x, v).detach()
        kt = ntk_probe(model, paramsT, x, v).detach()
        denom = k0.norm().item()
        if denom == 0:
            continue
        rel.append(float((kt - k0).norm() / denom))
        cos.append(float(torch.nn.functional.cosine_similarity(k0, kt, dim=0)))
    return sum(rel) / len(rel), sum(cos) / len(cos)


def build_optimizer(model, spec):
    o = spec["optimizer"]
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
    return cls(model.parameters(), **kw)


def run_candidate(model_cls, train, test, spec, seed=0):
    torch.manual_seed(seed)
    model = model_cls()
    opt = build_optimizer(model, spec)
    tx, ty = train["x"], train["y"]
    vx, vy = test["x"], test["y"]
    n = tx.shape[0]
    steps = int(spec["budget"]["training_steps"])
    bs = int(spec["budget"]["batch_size"])
    params0 = flatten_params(model)
    with torch.no_grad():
        mse0 = float(torch.mean((model(vx) - vy) ** 2))
    for _ in range(steps):
        model.train()
        idx = torch.randint(0, n, (bs,))
        pred = model(tx[idx])
        loss = torch.mean((pred - ty[idx]) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        mseT = float(torch.mean((model(vx) - vy) ** 2))
    paramsT = flatten_params(model)
    with torch.no_grad():
        p0 = torch.cat([p.reshape(-1) for p in params0.values()])
        pT = torch.cat([p.reshape(-1) for p in paramsT.values()])
        disp = float((pT - p0).norm() / p0.norm())
    model.eval()
    rel, cos = relative_kernel_change(model, params0, paramsT, tx, n_probes=8)
    lam = estimate_lambda_max(model, params0, tx, n_probes=2)
    return dict(seed=seed, steps=steps, bs=bs, opt=spec["optimizer"]["type"],
                lr=spec["optimizer"]["lr"], test_mse_init=mse0, test_mse_final=mseT,
                param_rel_disp=disp, ntk_rel_change=rel, ntk_cosine=cos,
                lambda_max_init=lam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", default="q_11b94d")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    cases = {
        "q_11b94d": dict(
            fam_dir="univariate_regression", ds_id="sym_4d5411",
            set_id="set_4096_var_fix_fix_cecee3",
            cands={"A": "c_6c95ed", "B": "c_4d0869", "C": "c_d7ace4"},
        ),
        "q_aef977": dict(
            fam_dir="univariate_regression", ds_id="sym_801402",
            set_id="set_4096_var_fix_fix_46d8c0",
            cands={"A": "c_bae746", "B": "c_de78d2", "C": "c_ee0bd6"},
        ),
    }
    case = cases[args.qid]
    out = {"qid": args.qid, "candidates": {}}
    for letter, cid in case["cands"].items():
        train, test, model_cls, spec = load_case(case["fam_dir"], case["ds_id"], case["set_id"], cid)
        runs = []
        for s in [int(x) for x in args.seeds.split(",")]:
            r = run_candidate(model_cls, train, test, spec, seed=s)
            runs.append(r)
            print(letter, cid, r)
        agg = {}
        for k in runs[0]:
            if k in ("seed", "opt"):
                continue
            vals = [r[k] for r in runs]
            agg[k] = sum(vals) / len(vals)
        out["candidates"][letter] = {"candidate_id": cid, "runs": runs, "mean": agg}
    outpath = Path(args.out) if args.out else Path(__file__).with_name(f"lazy_probe_{args.qid}.json")
    outpath.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    print("wrote", outpath)
    print(json.dumps({k: v["mean"] for k, v in out["candidates"].items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
