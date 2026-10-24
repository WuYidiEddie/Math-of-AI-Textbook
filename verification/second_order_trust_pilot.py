"""Trust-region stabilization of the second-order tangent model (M4).

The unrestricted quadratic surrogate can leave the neighborhood where its Taylor
expansion is meaningful.  This pilot projects d = theta - theta0 back into a ball after
every optimizer step.  The radius is r = c ||d_1||, where d_1 is the FINAL displacement
of the order-1 surrogate for the same candidate; c is shared by every candidate.

This changes the optimization protocol deliberately: it tests whether restricting the
Taylor model to a trust region fixes its divergence without reducing the learning rate.
All radii and all candidates are reported; no per-candidate radius is selected by test MSE.

Dev half only (md5(qid) % 2 == 0).  Nothing writes into the v1.5 tree.
Output: verification/second_order_trust_pilot_result.md (+ .json)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import second_order_pilot as sop
import theory_answerer as ta

HERE = Path(__file__).resolve().parent
OUT_MD = HERE / "second_order_trust_pilot_result.md"
OUT_JSON = HERE / "second_order_trust_pilot_result.json"
RADII = (0.5, 1.0, 2.0)
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


def main() -> None:
    out("# second-order tangent model with a trust region (M4)")
    out()
    out(f"torch {torch.__version__} | device cpu | seed {SEED} | dev half only")
    out(f"pilot questions: {sop.PILOT}")
    out(f"trust radius r = c ||d_1||, c in {RADII}; the same c is used for every candidate")
    out("projection after every optimizer step; official lr, batches and optimizer otherwise")
    out()

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

        rows = []
        for ch in q["choices"]:
            cdir = ta.DATA / ch["candidate_path"]
            Model = ta.load_module(cdir / "model.py", f"tr_{ch['candidate_id']}").Model
            spec = json.loads((cdir / "candidate_spec.json").read_text(encoding="utf-8"))
            real = sop.run_real(Model, spec, train, test, SEED)
            pred1, p_total, curve1 = sop.run_surrogate(
                Model, spec, train, test, SEED, order=1, track=True)
            d1 = curve1[-1][2]
            pred2, _, curve2 = sop.run_surrogate(
                Model, spec, train, test, SEED, order=2, track=True)
            rec = dict(letter=ch["letter"], real=real, order1=pred1, order2=pred2,
                       p=p_total, d1=d1, curve1=curve1, curve2=curve2, trust={})
            out(f"  {ch['letter']}: real {real:.5f} | order1 {pred1:.5f} | "
                f"order2 {pred2:.5f} | ||d_1|| {d1:.3f}")
            for c in RADII:
                t0 = time.time()
                m, _, curve = sop.run_surrogate(
                    Model, spec, train, test, SEED, order=2, track=True,
                    trust_radius=c * d1)
                key = f"tr{c}"
                rec[key] = m
                rec["trust"][key] = dict(radius=c * d1, curve=curve,
                                         seconds=round(time.time() - t0, 1))
                out(f"      c={c}: test {m:.5f} | final ||d|| {curve[-1][2]:.3f} | "
                    f"r {c * d1:.3f} | {time.time() - t0:.0f}s")
            rows.append(rec)

        def argmin(key):
            return min(rows, key=lambda r: r[key])["letter"]

        keys = ["real", "order1", "order2"] + [f"tr{c}" for c in RADII]
        winners = {key: argmin(key) for key in keys}
        line = "  winner: " + " | ".join(f"{key} {winners[key]}" for key in keys)
        out(line + f"   (truth {truth})")
        summary.append(dict(qid=qid, family=fam, truth=truth, rows=rows,
                            winners=winners))

    keys = ["order1", "order2"] + [f"tr{c}" for c in RADII]
    head("summary")
    out("  qid        | truth | real | " + " | ".join(keys))
    out("  " + "-" * (36 + 10 * len(keys)))
    for s in summary:
        marks = []
        for key in keys:
            w = s["winners"][key]
            marks.append(f"{w}{'*' if w == s['truth'] else ' '}")
        out(f"  {s['qid']} |  {s['truth']}    |  {s['winners']['real']}   | "
            + " | ".join(marks))
    out()
    out("  (* = matches the official answer)")
    for key in keys:
        hits = sum(s["winners"][key] == s["truth"] for s in summary)
        out(f"  {key}: {hits}/{len(summary)}")
    out()
    out("  interpretation limits:")
    out("    * dev-half exploratory pilot, one seed; the official answer uses 10 seeds.")
    out("    * the radius grid is reported in full, not selected per candidate by test MSE.")
    out("    * ||d_1|| comes from the order-1 training trajectory only, never from test labels.")
    out("    * projection prevents large displacements but cannot repair Taylor truncation")
    out("      error inside the ball, or a trajectory that needs feature re-linearization.")

    result["radii"] = RADII
    result["summary"] = summary
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
                        encoding="utf-8", newline="\n")
    print(f"\nwritten: {OUT_MD}")
    print(f"written: {OUT_JSON}")


if __name__ == "__main__":
    main()
