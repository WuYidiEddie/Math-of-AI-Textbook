"""Full-bank evaluation of the dev-proposed moving-second-order hybrid.

Frozen rule before the full run:
  * non-bigram: moving second-order tangent model with S=16, seed 0;
  * iff its final real-network test metric is nonfinite, rerun moving FIRST order at S=16,
    then S=1; use finite v1 if available, otherwise moving first order at S=64 and S=steps;
  * bigram: retain the existing initialization-linearized theory_answerer predictions,
    because explicit output Jacobians/Hessian terms are too large for B x L x V outputs.

The full-bank run is evaluation, not another radius/S search.  Test-half results are for
aggregate reporting only.  Records are appended after every candidate and --skip-done
resumes an interrupted run.  Nothing writes into the v1.5 tree.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch

import theory_answerer as ta
import moving_linearization_pilot as mlp
import moving_second_order_pilot as mso

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "moving_second_order_s16_500.jsonl"
BASE_OUT = HERE / "theory_answerer.jsonl"
S = 16
SEED = 0
ONLINE_JAC_ELEMENTS = 1e8


def load_base_records(path):
    records = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if not r.get("skipped"):
                records[(r["qid"], r["letter"])].append(r)
    return records


def load_done(path):
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    if not r.get("skipped"):
                        done.add((r["qid"], r["letter"], r["seed"]))
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--base-jsonl", default=str(BASE_OUT))
    ap.add_argument("--pack", default="")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--families", default="")
    ap.add_argument("--limit-per-family", type=int, default=0)
    ap.add_argument("--qids", default="")
    args = ap.parse_args()
    families = [s for s in args.families.split(",") if s] or None
    wanted_qids = {s for s in args.qids.split(",") if s} or None
    out_path = Path(args.out)
    done = load_done(out_path) if args.skip_done else set()
    base_records = load_base_records(args.base_jsonl)
    pack = Path(args.pack) if args.pack else None
    data_root = pack if pack else ta.DATA
    print(f"rule: moving second order S={S}, seed={SEED}; nonfinite -> moving first S={S} -> S1 -> v1/S64/Ssteps")
    print(f"bigram: existing initialization-linearized predictions from {args.base_jsonl}")
    if pack:
        print(f"external question pack: {pack}")
    print(f"skip-done: {len(done)} candidate-seed records", flush=True)

    n_done = n_fallback = n_s1 = n_bigram = 0
    with open(out_path, "a", encoding="utf-8", newline="\n") as out:
        for fam, qpath, q in ta.iter_questions(args.limit_per_family, families, pack=pack):
            qid = q["question_id"]
            if wanted_qids and qid not in wanted_qids:
                continue
            base = data_root / "datasets" / fam / q["dataset_id"]
            train = torch.load(base / "train.pt", map_location="cpu", weights_only=True)
            test = torch.load(base / "test.pt", map_location="cpu", weights_only=True)
            for ch in q["choices"]:
                letter = ch["letter"]
                if fam == "bigram_lm":
                    records = base_records.get((qid, letter), [])
                    if not records:
                        r = dict(qid=qid, family=fam, letter=letter,
                                 candidate_id=ch["candidate_id"], seed=SEED,
                                 correct_letter=q["correct_letter"], qtype=q.get("type", ""),
                                 skipped="error: missing base bigram prediction")
                        out.write(json.dumps(r, ensure_ascii=False) + "\n")
                        out.flush()
                        continue
                    for old in records:
                        key = (qid, letter, old["seed"])
                        if key in done:
                            continue
                        r = dict(old)
                        r["prediction_source"] = "bigram_initialization_linearized"
                        r["second_order_applied"] = False
                        out.write(json.dumps(r, ensure_ascii=False) + "\n")
                        out.flush()
                        done.add(key)
                        n_bigram += 1
                    print(qid, letter, "-> bigram existing order1", flush=True)
                    continue

                if (qid, letter, SEED) in done:
                    continue
                cdir = data_root / ch["candidate_path"]
                Model = ta.load_module(cdir / "model.py", f"full_{ch['candidate_id']}").Model
                spec = json.loads((cdir / "candidate_spec.json").read_text(encoding="utf-8"))
                probe = Model()
                p_total = sum(p.numel() for p in probe.parameters())
                with torch.no_grad():
                    C = probe(train["x"][:1]).reshape(1, -1).shape[1]
                online = train["x"].shape[0] * C * p_total > ONLINE_JAC_ELEMENTS
                t0 = time.time()
                try:
                    if online:
                        m2, dnorm, _ = mso.run_moving_online(
                            Model, spec, train, test, SEED, S, order=2)
                    else:
                        m2, dnorm, _ = mso.run_moving_second_order(
                            Model, spec, train, test, SEED, S)
                    source = "moving_second_S16_online" if online else "moving_second_S16"
                    first_metric = None
                    v1_metric = None
                    second_metric = m2 if math.isfinite(m2) else None
                    if math.isfinite(m2):
                        pred = m2
                    else:
                        if online:
                            pred, _, _ = mso.run_moving_online(
                                Model, spec, train, test, SEED, S, order=1)
                        else:
                            pred, _ = mlp.run_moving_linearization(
                                Model, spec, train, test, SEED, S)
                        source = ("fallback_first_S16_online_nonfinite_second" if online
                                  else "fallback_first_S16_nonfinite_second")
                        first_metric = pred if math.isfinite(pred) else None
                        n_fallback += 1
                    if not math.isfinite(pred):
                        if online:
                            pred, _, _ = mso.run_moving_online(
                                Model, spec, train, test, SEED, 1, order=1)
                        else:
                            pred, _ = mlp.run_moving_linearization(
                                Model, spec, train, test, SEED, 1)
                        source = "fallback_first_S1_nonfinite_moving"
                        n_s1 += 1
                    if not math.isfinite(pred):
                        v1_records = [old for old in base_records.get((qid, letter), [])
                                      if old["seed"] == SEED and math.isfinite(old["pred_metric"])]
                        if v1_records:
                            pred = v1_records[0]["pred_metric"]
                            source = "fallback_v1_nonfinite_moving"
                            v1_metric = pred
                        else:
                            if online:
                                pred, _, _ = mso.run_moving_online(
                                    Model, spec, train, test, SEED, 64, order=1)
                            else:
                                pred, _ = mlp.run_moving_linearization(
                                    Model, spec, train, test, SEED, 64)
                            source = "fallback_first_S64_nonfinite_moving_and_v1"
                    if not math.isfinite(pred):
                        steps = int(spec["budget"]["training_steps"])
                        if online:
                            pred, _, _ = mso.run_moving_online(
                                Model, spec, train, test, SEED, steps, order=1)
                        else:
                            pred, _ = mlp.run_moving_linearization(
                                Model, spec, train, test, SEED, steps)
                        source = "fallback_first_S_steps_effective_real_training"
                    if not math.isfinite(pred):
                        raise FloatingPointError("all surrogate and per-step fallback metrics nonfinite")
                    r = dict(pred_metric=pred, second_metric=second_metric,
                             first_metric=first_metric, v1_metric=v1_metric, prediction_source=source,
                             second_order_applied=True, dnorm=dnorm if math.isfinite(dnorm) else None,
                             p=p_total,
                             opt=spec["optimizer"]["type"], lr=spec["optimizer"]["lr"],
                             wd=spec["optimizer"].get("weight_decay", 0.0),
                             loss=spec["loss"]["loss_id"], skipped="")
                except Exception as e:
                    r = dict(skipped=f"error: {type(e).__name__}: {e}",
                             prediction_source="failed", second_order_applied=True)
                r.update(qid=qid, family=fam, letter=letter,
                         candidate_id=ch["candidate_id"], seed=SEED,
                         correct_letter=q["correct_letter"], qtype=q.get("type", ""))
                r["elapsed_s"] = round(time.time() - t0, 2)
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                out.flush()
                print(qid, letter, "->", r.get("pred_metric", r["skipped"]),
                      r["prediction_source"], flush=True)
            n_done += 1
            print(f"[done] {n_done} questions ({qid}); fallbacks={n_fallback}; "
                  f"S1={n_s1}; bigram records={n_bigram}", flush=True)


if __name__ == "__main__":
    main()
