# lazy_probe_generic.py — like lazy_probe.py but takes any qid (looks up question.json)
# and supports both MSE and CE losses.  Used for M3: measure ACTUAL NTK change during
# real short training on dev-half questions, to validate the init-only kappa_path meter.
#
# Usage:
#   python verification/lazy_probe_generic.py --qids q_xxx,q_yyy --seeds 0
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.func import functional_call, jvp

from lazy_probe import build_optimizer, flatten_params, load_case, load_module

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "v1.5" / "v15-500q-6d4ae9b" / "data"


def ntk_probe_g(model, params, x, v, fd_eps=1e-3):
    """K v for arbitrary output shape; v has the same shape as f(x).
    J^T v via reverse mode (exact); J g via finite difference (forward-AD is
    unavailable for some ops, e.g. transformer_encoder_layer)."""
    f0 = functional_call(model, params, (x,))  # grad-enabled
    if f0.ndim == 1:
        f0 = f0[:, None]
    grads = torch.autograd.grad((f0 * v).sum(), tuple(params.values()))
    p0 = torch.cat([p.detach().reshape(-1) for p in params.values()])
    gvec = torch.cat([gg.reshape(-1) for gg in grads])
    ps = p0 + fd_eps * gvec
    displaced = {}
    off = 0
    for k, p in params.items():
        n = p.numel()
        displaced[k] = ps[off:off + n].view_as(p)
        off += n
    with torch.no_grad():
        f0d = functional_call(model, params, (x,))
        if f0d.ndim == 1:
            f0d = f0d[:, None]
        f1 = functional_call(model, displaced, (x,))
        if f1.ndim == 1:
            f1 = f1[:, None]
    return (f1 - f0d) / (fd_eps * x.shape[0])


def kernel_stats(model, params0, paramsT, x, n_probes=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        f0 = functional_call(model, params0, (x,))
    shape = f0.shape if f0.ndim > 1 else (x.shape[0],)
    rel, cos = [], []
    for _ in range(n_probes):
        v = torch.randn(*shape, generator=g)
        k0 = ntk_probe_g(model, params0, x, v).detach()
        kt = ntk_probe_g(model, paramsT, x, v).detach()
        denom = k0.norm().item()
        if denom == 0:
            continue
        rel.append(float((kt - k0).norm() / denom))
        cos.append(float(torch.nn.functional.cosine_similarity(k0.reshape(-1), kt.reshape(-1), dim=0)))
    # lambda_max at init by power iteration
    v = torch.randn(*shape, generator=g)
    v = v / v.norm()
    lam = 0.0
    for _ in range(12):
        u = ntk_probe_g(model, params0, x, v).detach()
        nv = u.norm()
        if nv == 0:
            break
        v = u / nv
        lam = float((v * u).sum() / float((v * v).sum()))
    return sum(rel) / len(rel), sum(cos) / len(cos), lam


def find_question(qid: str):
    coll = json.loads((ROOT / "v1.5" / "v15-500q-6d4ae9b" / "collection.json").read_text(encoding="utf-8"))
    for qpath in coll["question_paths"]:
        if qpath.rstrip("/").endswith(qid):
            q = json.loads((DATA / qpath / "question.json").read_text(encoding="utf-8"))
            fam = qpath.split("/")[1]
            return fam, q
    raise KeyError(qid)


def loss_and_metric(pred, y, loss_id):
    if loss_id == "mse":
        return torch.mean((pred - y) ** 2)
    return torch.nn.functional.cross_entropy(pred.reshape(-1, pred.shape[-1]), y.reshape(-1).long())


def run_candidate(model_cls, train, test, spec, seed=0):
    torch.manual_seed(seed)
    model = model_cls()
    opt = build_optimizer(model, spec)
    tx, ty = train["x"], train["y"]
    vx, vy = test["x"], test["y"]
    n = tx.shape[0]
    steps = int(spec["budget"]["training_steps"])
    bs = int(spec["budget"]["batch_size"])
    loss_id = spec["loss"]["loss_id"]
    params0 = flatten_params(model)
    with torch.no_grad():
        m0 = float(loss_and_metric(model(vx), vy, loss_id))
    for _ in range(steps):
        model.train()
        idx = torch.randint(0, n, (bs,))
        loss = loss_and_metric(model(tx[idx]), ty[idx], loss_id)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        mT = float(loss_and_metric(model(vx), vy, loss_id))
    paramsT = flatten_params(model)
    rel, cos, lam = kernel_stats(model, params0, paramsT, tx)
    return dict(seed=seed, steps=steps, loss=loss_id,
                metric_init=m0, metric_final=mT,
                ntk_rel_change=rel, ntk_cosine=cos, lambda_max_init=lam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qids", required=True)
    ap.add_argument("--seeds", default="0")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    for qid in args.qids.split(","):
        fam, q = find_question(qid)
        ds_id = q["dataset_id"]
        out = {"qid": qid, "family": fam, "candidates": {}}
        for ch in q["choices"]:
            cpath = ch["candidate_path"]  # datasets/<fam>/<ds>/candidates/<set>/<cid>
            parts = cpath.split("/")
            set_id, cand_id = parts[-2], parts[-1]
            train, test, model_cls, spec = load_case(fam, ds_id, set_id, cand_id)
            runs = [run_candidate(model_cls, train, test, spec, seed=s) for s in seeds]
            for r in runs:
                print(qid, ch["letter"], {k: round(v, 4) if isinstance(v, float) else v
                                          for k, v in r.items()}, flush=True)
            agg = {k: sum(r[k] for r in runs) / len(runs)
                   for k in runs[0] if k not in ("seed", "loss")}
            out["candidates"][ch["letter"]] = {"candidate_id": cand_id, "runs": runs, "mean": agg}
        outpath = Path(__file__).with_name(f"lazy_probe_generic_{qid}.json")
        outpath.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
        print("wrote", outpath, flush=True)


if __name__ == "__main__":
    main()
