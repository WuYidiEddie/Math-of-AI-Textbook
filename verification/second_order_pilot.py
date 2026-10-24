"""Second-order tangent-model pilot (M4) -- can the quadratic surrogate fix non-lazy misses?

The first-order answerer (theory_answerer.py) replaces each candidate by its tangent
linearization at init,  f_lin(d) = f0 + J d , and runs the official protocol on it.
This script asks: does adding the second-order term,

    f_quad(d)(x) = f0(x) + J(x) d + (1/2) d^T H(x) d,    H(x) = d^2 f / d theta^2 |_{theta0}

flip the WRONG answers on a few non-lazy dev-half regression questions back to the truth?

The quadratic surrogate is a QUARTIC loss in d, so it is NON-convex: it is trained by the
official optimizer exactly like the real run (no closed form any more).  The H-terms are
never materialised:
  * forward  0.5 d^T H d  = 0.5 * (second directional derivative of f along d), via nested jvp
  * gradient sum_i gp_i H_i d = jvp( grad( sum_i gp_i f(x_i; .) ), theta0, d )   (forward-over-reverse HVP)

Per candidate we report three numbers, all under the SAME seed-0 protocol:
  real    : the actually-trained candidate's test MSE (ground truth)
  order1  : the linear  surrogate's test MSE  (should match theory_answerer's seed-0 value)
  order2  : the quadratic surrogate's test MSE
and per question whether argmin over candidates flips to the official answer.

Dev half only (md5(qid) % 2 == 0).  Nothing writes into the v1.5 tree.
Output: stdout + verification/second_order_pilot_result.md (+ .json)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from torch.func import jacrev

sys.path.insert(0, str(Path(__file__).resolve().parent))
import theory_answerer as ta  # reuse: load_module, make_f_of_theta, theta_vector, ...

HERE = Path(__file__).resolve().parent
OUT_MD = HERE / "second_order_pilot_result.md"
OUT_JSON = HERE / "second_order_pilot_result.json"

# dev-half regression questions: 3 wrong (large feature-learning gap) + 1 hit control.
# q_11b94d is the case-study NEAR-TIE wrong answer (tiny fl gap) -- the negative control,
# included only with --with-q11b94d because p=265k makes the HVPs slow.
PILOT = ["q_ab8774", "q_0b5a9f", "q_0525ce", "q_8f876c"]
if "--with-q11b94d" in sys.argv:
    PILOT.append("q_11b94d")
SEED = 0

lines: list[str] = []
result: dict = {}


def out(s: str = "") -> None:
    print(s, flush=True)
    lines.append(s)


def head(s: str) -> None:
    out()
    out(s)
    out("-" * len(s))


def find_questions(wanted: list[str]) -> dict:
    """qid -> (family, qpath, question dict), by scanning collection.json."""
    coll = json.loads((ta.DATA.parent / "collection.json").read_text(encoding="utf-8"))
    found = {}
    for qpath in coll["question_paths"]:
        qdir = ta.DATA / qpath
        q = json.loads((qdir / "question.json").read_text(encoding="utf-8"))
        if q["question_id"] in wanted:
            found[q["question_id"]] = (qpath.split("/")[1], qpath, q)
    return found


def reset_rng(seed: int):
    """Leave the global RNG at the official runner's post-init state.  Callers must build
    the model with THIS seed and NOT call Model() a second time (a double init would
    shift the minibatch stream)."""
    torch.manual_seed(seed)


def run_real(Model, spec, train, test, seed: int) -> float:
    """The actually-trained candidate's test MSE under the official protocol."""
    reset_rng(seed)
    model = Model()
    o = spec["optimizer"]
    params = list(model.parameters())
    typ = o["type"]
    kw = dict(lr=o["lr"], weight_decay=o.get("weight_decay", 0.0))
    if typ == "AdamW":
        opt = torch.optim.AdamW(params, betas=tuple(o.get("betas", (0.9, 0.999))), **kw)
    elif typ == "Adam":
        opt = torch.optim.Adam(params, betas=tuple(o.get("betas", (0.9, 0.999))), **kw)
    elif typ == "SGD":
        opt = torch.optim.SGD(params, momentum=o.get("momentum", 0.0), **kw)
    elif typ == "RMSprop":
        opt = torch.optim.RMSprop(params, **kw)
    elif typ == "Adagrad":
        opt = torch.optim.Adagrad(params, **kw)
    else:
        raise ValueError(typ)
    tx, ty = train["x"], train["y"]
    n = tx.shape[0]
    steps = int(spec["budget"]["training_steps"])
    bs = int(spec["budget"]["batch_size"])
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,))
        loss = torch.mean((model(tx[idx]).reshape(-1) - ty[idx].reshape(-1)) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    with torch.inference_mode():
        return float(torch.mean((model(test["x"]).reshape(-1) - test["y"].reshape(-1)) ** 2))


def run_surrogate(Model, spec, train, test, seed: int, order: int, lr_mult: float = 1.0,
                  track: bool = False, trust_radius: float | None = None):
    """Train the order-1 / order-2 tangent surrogate under the official protocol.

    order 1 runs in float32 (matches theory_answerer exactly, for validation).
    order 2 runs in float64 so the finite-difference second-derivative terms are
    resolvable; the H-terms use central finite differences (forward-mode AD through a
    second derivative is unsupported for SiLU, which these candidates use).
    lr_mult rescales the learning rate (a stabilization probe for the non-convex
    quartic surrogate).  trust_radius projects d back into a ball around theta0 after
    each optimizer step.  track=True also returns the test-metric curve + ||d||."""
    reset_rng(seed)
    model = Model()
    dtype = torch.float64 if order == 2 else torch.float32
    if order == 2:
        model = model.double()
    shapes = [tuple(p.shape) for p in model.parameters()]
    sizes = [int(p.numel()) for p in model.parameters()]
    p_total = sum(sizes)
    theta0 = ta.theta_vector(model).detach()
    f = ta.make_f_of_theta(model, shapes, sizes)
    EPS = 1e-3

    tx = train["x"].to(dtype)
    ty = train["y"]
    vx = test["x"].to(dtype)
    vy = test["y"]
    n = tx.shape[0]
    loss_id = spec["loss"]["loss_id"]
    steps = int(spec["budget"]["training_steps"])
    bs = int(spec["budget"]["batch_size"])

    with torch.no_grad():
        f0_tr = f(theta0, tx).reshape(n, -1)
        f0_te = f(theta0, vx).reshape(vx.shape[0], -1)
    C = f0_tr.shape[1]
    J = jacrev(lambda th: f(th, tx).reshape(-1))(theta0).reshape(n, C, p_total)
    Jte = jacrev(lambda th: f(th, vx).reshape(-1))(theta0).reshape(vx.shape[0], C, p_total)

    def quad_term(d, xb, f0b):
        """0.5 * d^T H d per point, central FD of the real net along d."""
        if float(d.norm()) < 1e-12:
            return torch.zeros(xb.shape[0], C, dtype=dtype)
        with torch.no_grad():
            fp = f(theta0 + EPS * d, xb)
            fm = f(theta0 - EPS * d, xb)
        return ((fp - 2.0 * f0b + fm) / (2.0 * EPS * EPS)).reshape(xb.shape[0], C)

    def grad_hvp(d, xb, gp):
        """sum_i gp_i H_i d, central FD of the gradient of theta -> sum(gp * f)."""
        if float(d.norm()) < 1e-12:
            return torch.zeros(p_total, dtype=dtype)
        def gu(th):
            t = th.clone().requires_grad_(True)
            return torch.autograd.grad((gp * f(t, xb)).sum(), t)[0]
        return (gu(theta0 + EPS * d) - gu(theta0 - EPS * d)) / (2.0 * EPS)

    def surrogate_pred(d, xb, f0b, Jb):
        out_ = f0b + torch.einsum("bcp,p->bc", Jb, d)
        if order == 2:
            out_ = out_ + quad_term(d, xb, f0b)
        return out_

    theta = theta0.clone().requires_grad_(True)
    o = dict(spec["optimizer"])
    o["lr"] = o["lr"] * lr_mult
    opt = ta.build_optimizer_from_spec(theta, o)
    curve = []
    cps = {int(steps * fr) for fr in (0.05, 0.1, 0.2, 0.4, 0.7, 1.0)} if track else set()
    vy_sub = vy.reshape(vx.shape[0], C)
    for step_i in range(steps):
        idx = torch.randint(0, n, (bs,))
        d = theta.detach() - theta0
        pred = f0_tr[idx] + torch.einsum("bcp,p->bc", J[idx], d)
        if order == 2:
            pred = pred + quad_term(d, tx[idx], f0_tr[idx])
        gp = ta.loss_grad_pred(pred, ty[idx].reshape(bs, C).to(dtype), loss_id).reshape(bs, C)
        g = torch.einsum("bcp,bc->p", J[idx], gp)
        if order == 2:
            g = g + grad_hvp(d, tx[idx], gp)
        theta.grad = g
        opt.step()
        if trust_radius is not None:
            with torch.no_grad():
                dd = theta - theta0
                dn = dd.norm()
                if dn > trust_radius:
                    theta.copy_(theta0 + dd * (trust_radius / dn))
        if (step_i + 1) in cps:
            with torch.no_grad():
                dd = theta.detach() - theta0
                m = ta.metric_of(surrogate_pred(dd, vx, f0_te, Jte), vy_sub, loss_id)
                curve.append([round((step_i + 1) / steps, 3), m, round(float(dd.norm()), 3)])

    with torch.no_grad():
        d = theta.detach() - theta0
        pred_te = surrogate_pred(d, vx, f0_te, Jte)
        m = ta.metric_of(pred_te, vy_sub, loss_id)
    return (m, p_total, curve) if track else (m, p_total)


def main() -> None:
    out("# second-order tangent-model pilot (M4)")
    out()
    out(f"torch {torch.__version__} | device cpu | seed {SEED} | dev half only")
    out(f"pilot questions: {PILOT}")
    out()
    wanted = find_questions(PILOT)
    missing = [q for q in PILOT if q not in wanted]
    if missing:
        out(f"!! not found in bank: {missing}")

    summary = []
    for qid in PILOT:
        if qid not in wanted:
            continue
        fam, qpath, q = wanted[qid]
        ds_id = q["dataset_id"]
        base = ta.DATA / "datasets" / fam / ds_id
        train = torch.load(base / "train.pt", map_location="cpu", weights_only=True)
        test = torch.load(base / "test.pt", map_location="cpu", weights_only=True)
        truth = q["correct_letter"]
        head(f"{qid}  [{fam} / {q.get('type','')}]  official answer = {truth}")

        rows = []
        for ch in q["choices"]:
            cdir = ta.DATA / ch["candidate_path"]
            Model = ta.load_module(cdir / "model.py", f"m_{ch['candidate_id']}").Model
            spec = json.loads((cdir / "candidate_spec.json").read_text(encoding="utf-8"))
            t0 = time.time()
            real = run_real(Model, spec, train, test, SEED)
            t1 = time.time()
            pred1, p_total = run_surrogate(Model, spec, train, test, SEED, order=1)
            t2 = time.time()
            pred2, _, curve2 = run_surrogate(Model, spec, train, test, SEED, order=2, track=True)
            t3 = time.time()
            rows.append(dict(letter=ch["letter"], real=real, order1=pred1, order2=pred2,
                             p=p_total, curve2=curve2, t_real=round(t1 - t0, 1),
                             t_o1=round(t2 - t1, 1), t_o2=round(t3 - t2, 1)))
            out(f"  {ch['letter']}: real {real:.5f} | order1 {pred1:.5f} | order2 {pred2:.5f}"
                f"   (p={p_total}, t real/o1/o2 = {t1-t0:.0f}/{t2-t1:.0f}/{t3-t2:.0f} s)")
            out(f"      order2 curve [frac, test_mse, ||d||]: "
                + "  ".join(f"{fr}:{m:.3g}(|d|={dn})" for fr, m, dn in curve2))

        def argmin(key):
            return min(rows, key=lambda r: r[key])["letter"]
        w_real, w1, w2 = argmin("real"), argmin("order1"), argmin("order2")
        hit1 = "HIT " if w1 == truth else "wrong"
        hit2 = "HIT " if w2 == truth else "wrong"
        flip = ""
        if w1 != w2:
            flip = f"  -> order2 flips {w1} -> {w2} ({'FIXED' if w2 == truth else 'now wrong'})"
        out(f"  winner: real {w_real} | order1 {w1} [{hit1}] | order2 {w2} [{hit2}]{flip}")
        summary.append(dict(qid=qid, family=fam, truth=truth,
                            w_real=w_real, w1=w1, w2=w2, rows=rows))

    head("stabilization probe: order2 at a smaller learning rate")
    out("  the order2 surrogate is a non-convex quartic; at the official lr it can diverge.")
    out("  re-run it at lr x {0.3, 0.1} to see whether the blow-ups are a stability artifact.")
    out()
    for s in summary:
        fam, qpath, q = wanted[s["qid"]]
        ds_id = q["dataset_id"]
        base = ta.DATA / "datasets" / fam / ds_id
        train = torch.load(base / "train.pt", map_location="cpu", weights_only=True)
        test = torch.load(base / "test.pt", map_location="cpu", weights_only=True)
        blew = any((r["order2"] != r["order2"]) or (r["order2"] > 10 * max(r["real"], 1e-12))
                   for r in s["rows"])
        if not blew:
            continue
        out(f"  {s['qid']} (official answer {s['truth']}):")
        for ch, r in zip(q["choices"], s["rows"]):
            cdir = ta.DATA / ch["candidate_path"]
            Model = ta.load_module(cdir / "model.py", f"m2_{ch['candidate_id']}").Model
            spec = json.loads((cdir / "candidate_spec.json").read_text(encoding="utf-8"))
            line = f"    {ch['letter']}: order2@1x {r['order2']:.4g}"
            for lm in (0.3, 0.1):
                m2, _ = run_surrogate(Model, spec, train, test, SEED, order=2, lr_mult=lm)
                line += f"  | order2@{lm}x {m2:.4g}"
                r[f"order2_lr{lm}"] = m2
            line += f"   (real {r['real']:.4g})"
            out(line)

    head("summary")
    out("  qid        | truth | real | order1 | order2 | order2 fixes?")
    out("  " + "-" * 70)
    n_fix = n_break = 0
    for s in summary:
        fix = ""
        if s["w1"] != s["truth"] and s["w2"] == s["truth"]:
            fix, n_fix = "YES (was wrong, now right)", n_fix + 1
        elif s["w1"] == s["truth"] and s["w2"] != s["truth"]:
            fix, n_break = "NO -- it BROKE a right answer", n_break + 1
        elif s["w1"] != s["truth"] and s["w2"] != s["truth"]:
            fix = "no (still wrong)"
        else:
            fix = "n/a (order1 already right)"
        out(f"  {s['qid']} |  {s['truth']}    |  {s['w_real']}   |   {s['w1']}    |   {s['w2']}    | {fix}")
    out()
    out(f"  order2 fixed {n_fix} wrong answer(s), broke {n_break} right answer(s), "
        f"out of {len(summary)} dev-half pilot questions.")
    out()
    out("  honest caveats:")
    out("    * 1 seed only; the official answer is a 10-seed mean, so near-ties are noise here.")
    out("    * the quadratic surrogate is NON-convex (quartic loss): it is trained, not solved,")
    out("      and could land in a different basin than the real run for reasons unrelated to")
    out("      the second-order term.")
    out("    * q_11b94d (if included) is a NEAR-TIE failure (feature-learning gap ~0): the")
    out("      second-order term is not expected to fix it -- it is the negative control.")

    result["summary"] = summary
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
                        encoding="utf-8", newline="\n")
    print(f"\nwritten: {OUT_MD}")
    print(f"written: {OUT_JSON}")


if __name__ == "__main__":
    main()
