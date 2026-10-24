"""Moving second-order tangent models (M4): curvature plus re-linearization.

At each stage s, expand around the CURRENT theta_s:

    f(theta_s + d) ~= f(theta_s) + J_s d + 1/2 d^T H_s d.

The optimizer state and minibatch stream continue across stages.  Prediction is the REAL
network evaluated at the final parameters, never the quadratic extrapolation.  S=steps
reduces to real training (up to float64 arithmetic); smaller S asks whether second-order
information can match it with fewer refreshes than moving linearization.

The order-1 baseline is read from moving_linearization_pilot_result.json.  Dev half only;
nothing writes into the v1.5 tree.  Output: moving_second_order_pilot_result.md (+ .json).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.func import jacrev

sys.path.insert(0, str(Path(__file__).resolve().parent))
import second_order_pilot as sop
import theory_answerer as ta

HERE = Path(__file__).resolve().parent
OUT_MD = HERE / "moving_second_order_pilot_result.md"
OUT_JSON = HERE / "moving_second_order_pilot_result.json"
S_LIST = (4, 16, 64)
SEED = 0
EPS = 1e-3

lines: list[str] = []
result: dict = {}


def out(s: str = "") -> None:
    print(s, flush=True)
    lines.append(s)


def head(s: str) -> None:
    out()
    out(s)
    out("-" * len(s))


def run_moving_second_order(Model, spec, train, test, seed: int, S: int):
    t0 = time.time()
    sop.reset_rng(seed)
    model = Model().double()
    shapes = [tuple(p.shape) for p in model.parameters()]
    sizes = [int(p.numel()) for p in model.parameters()]
    p_total = sum(sizes)
    theta0 = ta.theta_vector(model).detach()
    f = ta.make_f_of_theta(model, shapes, sizes)

    tx = train["x"].double()
    ty = train["y"].double()
    vx = test["x"].double()
    vy = test["y"].double()
    n = tx.shape[0]
    loss_id = spec["loss"]["loss_id"]
    steps = int(spec["budget"]["training_steps"])
    bs = int(spec["budget"]["batch_size"])
    theta = theta0.clone().requires_grad_(True)
    opt = ta.build_optimizer_from_spec(theta, spec["optimizer"])
    chunks = [steps // S + (1 if i < steps % S else 0) for i in range(S)]

    for ns in chunks:
        theta_s = theta.detach().clone()
        with torch.no_grad():
            f_s = f(theta_s, tx).reshape(n, -1)
        C = f_s.shape[1]
        J_s = jacrev(lambda th: f(th, tx).reshape(-1))(theta_s).reshape(n, C, p_total)

        def quad_term(d, xb, f0b):
            if float(d.norm()) < 1e-12:
                return torch.zeros(xb.shape[0], C, dtype=torch.float64)
            with torch.no_grad():
                fp = f(theta_s + EPS * d, xb)
                fm = f(theta_s - EPS * d, xb)
            return ((fp - 2.0 * f0b + fm) / (2.0 * EPS * EPS)).reshape(xb.shape[0], C)

        def grad_hvp(d, xb, gp):
            if float(d.norm()) < 1e-12:
                return torch.zeros(p_total, dtype=torch.float64)

            def gu(th):
                t = th.clone().requires_grad_(True)
                return torch.autograd.grad((gp * f(t, xb)).sum(), t)[0]

            return (gu(theta_s + EPS * d) - gu(theta_s - EPS * d)) / (2.0 * EPS)

        for _ in range(ns):
            idx = torch.randint(0, n, (bs,))
            d = theta.detach() - theta_s
            pred = f_s[idx] + torch.einsum("bcp,p->bc", J_s[idx], d)
            pred = pred + quad_term(d, tx[idx], f_s[idx])
            gp = ta.loss_grad_pred(pred, ty[idx].reshape(bs, C), loss_id).reshape(bs, C)
            g = torch.einsum("bcp,bc->p", J_s[idx], gp)
            theta.grad = g + grad_hvp(d, tx[idx], gp)
            opt.step()

    with torch.no_grad():
        pred_te = f(theta.detach(), vx).reshape(vx.shape[0], -1)
        m = ta.metric_of(pred_te, vy.reshape(vx.shape[0], pred_te.shape[1]), loss_id)
        dnorm = float((theta.detach() - theta0).norm())
    return m, dnorm, time.time() - t0


def main() -> None:
    out("# moving second-order tangent models (M4)")
    out()
    out(f"torch {torch.__version__} | device cpu | seed {SEED} | dev half only")
    out(f"pilot questions: {sop.PILOT} | S in {S_LIST}")
    out("prediction = REAL network at the surrogate-reached parameters")
    out("first-order comparison = saved moving-linearization run, same seed and protocol")
    out()

    first = json.loads((HERE / "moving_linearization_pilot_result.json").read_text(
        encoding="utf-8"))["summary"]
    first_by_qid = {s["qid"]: s for s in first}
    wanted = sop.find_questions(sop.PILOT)
    summary = []
    for qid in sop.PILOT:
        if qid not in wanted:
            continue
        fam, qpath, q = wanted[qid]
        base = ta.DATA / "datasets" / fam / q["dataset_id"]
        train = torch.load(base / "train.pt", map_location="cpu", weights_only=True)
        test = torch.load(base / "test.pt", map_location="cpu", weights_only=True)
        truth = q["correct_letter"]
        head(f"{qid}  [{fam} / {q.get('type', '')}]  official answer = {truth}")
        first_rows = {r["letter"]: r for r in first_by_qid[qid]["rows"]}
        rows = []
        for ch in q["choices"]:
            cdir = ta.DATA / ch["candidate_path"]
            Model = ta.load_module(cdir / "model.py", f"ms_{ch['candidate_id']}").Model
            spec = json.loads((cdir / "candidate_spec.json").read_text(encoding="utf-8"))
            fr = first_rows[ch["letter"]]
            rec = dict(letter=ch["letter"], real=fr["real"], first={}, second={})
            line = f"  {ch['letter']}: real {fr['real']:.5f}"
            for S in S_LIST:
                rec["first"][f"S{S}"] = fr[f"S{S}"]
                m, dnorm, dt = run_moving_second_order(Model, spec, train, test, SEED, S)
                rec["second"][f"S{S}"] = dict(metric=m, dnorm=dnorm,
                                               seconds=round(dt, 1))
                line += (f" | S{S} first {fr[f'S{S}']:.4g} / second {m:.4g}"
                         f" (|d|={dnorm:.2f}, {dt:.0f}s)")
            rows.append(rec)
            out(line)

        def value(order, r, S):
            first_m = r["first"][f"S{S}"]
            second_m = r["second"][f"S{S}"]["metric"]
            if order == "first":
                return first_m
            if order == "second":
                return second_m if math.isfinite(second_m) else float("inf")
            return second_m if math.isfinite(second_m) else first_m

        def argmin(order, S):
            return min(rows, key=lambda r: value(order, r, S))["letter"]

        winners = {"real": min(rows, key=lambda r: r["real"])["letter"]}
        line = f"  winner: real {winners['real']}"
        for S in S_LIST:
            for order in ("first", "second", "hybrid"):
                winners[f"{order}_S{S}"] = argmin(order, S)
                line += f" | {order} S{S} {winners[f'{order}_S{S}']}"
        out(line + f"   (truth {truth})")
        summary.append(dict(qid=qid, family=fam, truth=truth, rows=rows, winners=winners))

    head("summary: first / second / nonfinite-fallback hybrid at equal S")
    out("  qid        | truth | " + " | ".join(f"S{S} f/s/h" for S in S_LIST))
    out("  " + "-" * 78)
    for s in summary:
        cells = []
        for S in S_LIST:
            marks = []
            for order in ("first", "second", "hybrid"):
                w = s["winners"][f"{order}_S{S}"]
                marks.append(f"{w}{'*' if w == s['truth'] else ' '}")
            cells.append("/".join(marks))
        out(f"  {s['qid']} |  {s['truth']}    | " + " | ".join(cells))
    out()
    out("  (* = matches the official answer; h = first-order fallback iff second is nonfinite)")
    for order in ("first", "second", "hybrid"):
        for S in S_LIST:
            hits = sum(s["winners"][f"{order}_S{S}"] == s["truth"] for s in summary)
            out(f"  {order}, S={S}: {hits}/{len(summary)}")
    out()
    out("  interpretation limits:")
    out("    * dev-half exploratory pilot, one seed; the official answer uses 10 seeds.")
    out("    * second order adds curvature within each stage, but its stage loss is still")
    out("      non-convex; re-linearization limits how far that approximation is used.")
    out("    * matching real training at large S is a consistency check, not evidence that")
    out("      the second-order term helps; the useful comparison is equal small S.")
    out("    * the nonfinite-fallback hybrid uses no labels, but was proposed AFTER seeing")
    out("      these dev results; it needs fresh questions before counting as an improvement.")

    result["S_list"] = S_LIST
    result["summary"] = summary
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
                        encoding="utf-8", newline="\n")
    print(f"\nwritten: {OUT_MD}")
    print(f"written: {OUT_JSON}")


if __name__ == "__main__":
    main()
