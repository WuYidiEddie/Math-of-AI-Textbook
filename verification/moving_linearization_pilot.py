"""Moving-linearization pilot (M4 route a): a stable, convex way to capture feature learning.

The first-order answerer freezes the kernel at init (S = 1 linearization).  The quadratic
surrogate (second_order_pilot.py) frees the curvature but is a non-convex quartic that
diverges.  This script is the middle path: refresh the linearization S times along the
trajectory, so the kernel is allowed to move but every stage stays a CONVEX linear model.

    stage s:  linearize at theta_s  ->  train the linear model  g(d) = f(theta_s) + J_s d
              for steps/S steps  ->  theta <- theta_s + d ,  then re-linearize.

S is a dial:  S = 1 is the pure lazy surrogate;  S = steps reproduces the real training
exactly (each stage is one step with the current Jacobian).  The predicted test metric is
the REAL network evaluated at the surrogate-reached parameters (the real net is never
gradient-trained).  If S = 16 already matches the real winner, the question is near-lazy;
if the winner only converges at large S, the feature learning is genuinely needed.

Dev half only (md5(qid) % 2 == 0).  Nothing writes into the v1.5 tree.
Output: stdout + verification/moving_linearization_pilot_result.md (+ .json)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from torch.func import jacrev

sys.path.insert(0, str(Path(__file__).resolve().parent))
import theory_answerer as ta
import second_order_pilot as sop  # reuse find_questions / reset_rng / run_real / PILOT

HERE = Path(__file__).resolve().parent
OUT_MD = HERE / "moving_linearization_pilot_result.md"
OUT_JSON = HERE / "moving_linearization_pilot_result.json"

S_LIST = [1, 4, 16, 64]
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


def run_moving_linearization(Model, spec, train, test, seed: int, S: int):
    """S-stage moving linearization.  Returns (real-net test metric at theta_final, seconds)."""
    t0 = time.time()
    sop.reset_rng(seed)
    model = Model()
    shapes = [tuple(p.shape) for p in model.parameters()]
    sizes = [int(p.numel()) for p in model.parameters()]
    p_total = sum(sizes)
    theta0 = ta.theta_vector(model).detach()
    f = ta.make_f_of_theta(model, shapes, sizes)

    tx, ty = train["x"], train["y"]
    vx, vy = test["x"], test["y"]
    n = tx.shape[0]
    loss_id = spec["loss"]["loss_id"]
    steps = int(spec["budget"]["training_steps"])
    bs = int(spec["budget"]["batch_size"])

    theta = theta0.clone().requires_grad_(True)
    opt = ta.build_optimizer_from_spec(theta, spec["optimizer"])  # persistent across stages
    chunks = [steps // S + (1 if i < steps % S else 0) for i in range(S)]
    for ns in chunks:
        theta_s = theta.detach().clone()
        with torch.no_grad():
            f_s = f(theta_s, tx).reshape(n, -1)
        C = f_s.shape[1]
        J_s = jacrev(lambda th: f(th, tx).reshape(-1))(theta_s).reshape(n, C, p_total)
        for _ in range(ns):
            idx = torch.randint(0, n, (bs,))
            d = theta.detach() - theta_s
            pred = f_s[idx] + torch.einsum("bcp,p->bc", J_s[idx], d)
            gp = ta.loss_grad_pred(pred, ty[idx].reshape(bs, C), loss_id).reshape(bs, C)
            theta.grad = torch.einsum("bcp,bc->p", J_s[idx], gp)
            opt.step()
    with torch.no_grad():
        pred_te = f(theta.detach(), vx).reshape(vx.shape[0], -1)
        m = ta.metric_of(pred_te, vy.reshape(vx.shape[0], pred_te.shape[1]), loss_id)
    return m, time.time() - t0


def main() -> None:
    out("# moving-linearization pilot (M4 route a): the feature-learning dial S")
    out()
    out(f"torch {torch.__version__} | device cpu | seed {SEED} | dev half only")
    out(f"pilot questions: {sop.PILOT}   S in {S_LIST}")
    out()
    out("S = 1 is the lazy surrogate (kernel frozen at init); S = steps would be the real")
    out("training.  Prediction = the REAL net evaluated at the surrogate-reached parameters.")
    out()
    wanted = sop.find_questions(sop.PILOT)

    summary = []
    for qid in sop.PILOT:
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
            Model = ta.load_module(cdir / "model.py", f"ml_{ch['candidate_id']}").Model
            spec = json.loads((cdir / "candidate_spec.json").read_text(encoding="utf-8"))
            real = sop.run_real(Model, spec, train, test, SEED)
            rec = {"letter": ch["letter"], "real": real}
            line = f"  {ch['letter']}: real {real:.5f}"
            for S in S_LIST:
                m, dt = run_moving_linearization(Model, spec, train, test, SEED, S)
                rec[f"S{S}"] = m
                line += f" | S{S} {m:.5f}({dt:.0f}s)"
            rows.append(rec)
            out(line)

        def argmin(key):
            return min(rows, key=lambda r: r[key])["letter"]
        w_real = argmin("real")
        line = f"  winner: real {w_real}"
        rec_w = {"w_real": w_real}
        for S in S_LIST:
            w = argmin(f"S{S}")
            rec_w[f"w_S{S}"] = w
            line += f" | S{S} {w}{'(=truth)' if w == truth else ''}"
        out(line + f"   (truth {truth})")
        summary.append(dict(qid=qid, family=fam, truth=truth, rows=rows, **rec_w))

    head("summary: does the winner move toward the truth as S grows?")
    out("  qid        | truth | real | " + " | ".join(f"S{S}" for S in S_LIST))
    out("  " + "-" * (40 + 8 * len(S_LIST)))
    for s in summary:
        marks = []
        for S in S_LIST:
            w = s[f"w_S{S}"]
            marks.append(f"{w}{'*' if w == s['truth'] else ' '}")
        out(f"  {s['qid']} |  {s['truth']}    |  {s['w_real']}   | " + " | ".join(marks))
    out()
    out("  (* = matches the official answer)")
    out()
    out("  reading:")
    out("    * if S=1 already wins, the question is near-lazy (feature learning irrelevant);")
    out("      if the winner only appears at S=4/16, the kernel movement is what decides it.")
    out("    * unlike the quadratic surrogate, every stage here is a CONVEX linear model,")
    out("      so there is no divergence -- the cost is S Jacobians per candidate.")
    out("    * 1 seed; the official answer is a 10-seed mean.")

    result["summary"] = summary
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
                        encoding="utf-8", newline="\n")
    print(f"\nwritten: {OUT_MD}")
    print(f"written: {OUT_JSON}")


if __name__ == "__main__":
    main()
