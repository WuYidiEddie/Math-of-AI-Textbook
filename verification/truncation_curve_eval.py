# truncation_curve_eval.py — predicted accuracy under truncated budgets.
#
# 王泽浩的「等价算力」基线：每个候选只训 ρ 倍预算，选当前领先者。
# 我们的答题器可以**零额外训练**地生成同一条曲线：读替身模拟在预算分数 ρ 处的
# 预测测试指标（curve 字段），取 argmin。对比实测基线（main 500：10% → 79.4%、
# 30% → ~90%；dataflip：10% → 44–53%、75% → ~90%）。
#
# Usage: python verification/truncation_curve_eval.py --jsonl verification/theory_answerer_dataflip.jsonl

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    args = ap.parse_args()

    per_q = defaultdict(dict)  # qid -> letter -> curve dict rho->metric
    meta = {}
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("skipped") or "curve" not in r:
                continue
            per_q[r["qid"]][r["letter"]] = dict(r["curve"])
            meta[r["qid"]] = (r["family"], r["correct_letter"])

    rhos = sorted({rho for c in per_q.values() for cur in c.values() for rho in cur})
    print(f"questions with curves: {len(per_q)}")
    print("rho_bin, accuracy, n")

    def binned(rho):
        return round(rho / 0.05) * 0.05

    bins = sorted({binned(r) for r in rhos})
    rows = []
    for b in bins:
        hit = tot = 0
        for qid, c in per_q.items():
            if len(c) < 3:
                continue
            vals = {}
            ok = True
            for l, cur in c.items():
                cand = [m for rho, m in cur.items() if binned(rho) == b]
                if not cand:
                    ok = False
                    break
                vals[l] = sum(cand) / len(cand)
            if not ok:
                continue
            pred = min(vals, key=vals.get)
            hit += pred == meta[qid][1]
            tot += 1
        rows.append((b, hit, tot))
        print(f"{b:.2f}  {hit / tot:.3f}  ({hit}/{tot})")
    return rows


if __name__ == "__main__":
    main()
