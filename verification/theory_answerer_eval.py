# theory_answerer_eval.py — evaluate theory_answerer.jsonl against the official keys.
#
# Metrics:
#   * top-1 accuracy of predicted argmin vs question.json correct_letter (overall, per family,
#     per qtype, per optimizer-set), vs 1/3 random baseline;
#   * Spearman rank correlation of predicted vs measured candidate metrics within question
#     (measured 10-seed means from notes/bank-audit.csv);
#   * feature-learning gap: |pred_metric - measured mean_metric| distribution per family;
#   * calibration on knife-edge questions (paired significance from bank-audit std_metric).
#
# Usage: python verification/theory_answerer_eval.py [--jsonl verification/theory_answerer.jsonl]

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
BANK = HERE.parent / "notes" / "bank-audit.csv"


def spearman(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0 + 1
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (vx * vy) if vx > 0 and vy > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=str(HERE / "theory_answerer.jsonl"))
    ap.add_argument("--out", default=str(HERE / "theory_answerer_result.md"))
    args = ap.parse_args()

    measured = {}  # (qid, letter) -> (mean_metric, std_metric)
    with open(BANK, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                measured[(r["qid"], r["letter"])] = (float(r["mean_metric"]), float(r["std_metric"]))
            except (ValueError, KeyError):
                pass

    preds = defaultdict(dict)  # qid -> letter -> {"pred_metric": mean over seeds, ...}
    raw = defaultdict(list)    # (qid, letter) -> [pred_metric per seed]
    meta = {}
    skipped = defaultdict(int)
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("skipped"):
                skipped[r["skipped"].split(":")[0]] += 1
                continue
            raw[(r["qid"], r["letter"])].append(r["pred_metric"])
            meta.setdefault(r["qid"], r)
    for (qid, letter), vals in raw.items():
        base = dict(meta[qid])
        base["letter"] = letter
        base["pred_metric"] = sum(vals) / len(vals)
        base["n_pred_seeds"] = len(vals)
        preds[qid][letter] = base

    fam_of = {}
    qtype_of = {}
    results = []
    for qid, cand in preds.items():
        if len(cand) < 3:
            continue
        recs = list(cand.values())
        fam = recs[0]["family"]
        fam_of[qid] = fam
        qtype_of[qid] = recs[0]["qtype"]
        pred_letter = min(cand, key=lambda l: cand[l]["pred_metric"])
        truth = recs[0]["correct_letter"]
        hit = pred_letter == truth
        # measured ranking + feature-learning gaps
        letters = sorted(cand)
        pm = [cand[l]["pred_metric"] for l in letters]
        mm = [measured.get((qid, l), (float("nan"), 0.0))[0] for l in letters]
        have_m = all(not math.isnan(x) for x in mm)
        rho = spearman(pm, mm) if have_m else float("nan")
        fl_gap = [abs(p - m) for p, m in zip(pm, mm)] if have_m else [float("nan")] * 3
        # paired significance: official std_metric on winner vs runner-up
        meas_sorted = sorted(letters, key=lambda l: measured.get((qid, l), (float("inf"), 0))[0])
        se_pair = None
        if have_m:
            w, ru = meas_sorted[0], meas_sorted[1]
            sw = measured[(qid, w)][1]
            sr = measured[(qid, ru)][1]
            se_pair = math.sqrt(sw * sw + sr * sr) / math.sqrt(10)
            gap = measured[(qid, ru)][0] - measured[(qid, w)][0]
            knife = gap < 2.26 * se_pair
        else:
            knife = None
        results.append({
            "qid": qid, "family": fam, "qtype": recs[0]["qtype"], "hit": hit,
            "pred_letter": pred_letter, "truth": truth, "rho": rho,
            "fl_gap_max": max(fl_gap), "knife": knife,
            "opts": tuple(sorted({cand[l]["opt"] for l in letters})),
        })

    def acc(rs):
        return sum(r["hit"] for r in rs) / len(rs) if rs else float("nan")

    # deterministic dev/test split by qid hash (fixed forever; failure analysis only on dev)
    import hashlib

    def split_of(qid):
        return "dev" if int(hashlib.md5(qid.encode()).hexdigest(), 16) % 2 == 0 else "test"

    for r in results:
        r["split"] = split_of(r["qid"])

    lines = []
    lines.append("# theory_answerer 评估结果\n")
    lines.append(
        "> 方法：每个候选 = 初始化处的切线性化模型（NTK/lazy），跑**官方原协议**（同 seed 初始化、同 "
        "minibatch 序列、同优化器配置、同步数），取其测试指标作为预测。回归/分类用显式 Jacobian"
        "（测试侧 256–512 点子采样）；bigram 用在线 FD-JVP/VJP 变体。多 seed 预测取均值。\n"
        "> **out-of-sample 纪律**：预测器零拟合（没有任何系数从题库学出）。为防止「看错题改方法」的"
        "自适应污染，题库按 qid 哈希固定切成 dev/test 两半——失败分析只在 dev 半区做，"
        "**headline 数字是 test 半区准确率**。\n"
    )
    lines.append(f"- 候选记录：{sum(len(c) for c in preds.values())}；题目（3 候选齐全）：{len(results)}")
    if skipped:
        lines.append(f"- 跳过：{dict(skipped)}")
    lines.append(f"- 随机基线：33.3%；KB 自报同分布约 74–88%（pack-01..04），标签天花板 98.5%\n")

    overall = acc(results)
    dev_rs = [r for r in results if r["split"] == "dev"]
    test_rs = [r for r in results if r["split"] == "test"]
    lines.append(f"## 总体 top-1 准确率：**{overall:.1%}**（{sum(r['hit'] for r in results)}/{len(results)}）\n")
    lines.append("### out-of-sample 切分（qid 哈希固定，失败分析只许在 dev 半区做）\n")
    lines.append(f"- **test 半区（headline OOS 数字）：{acc(test_rs):.1%}**（{sum(r['hit'] for r in test_rs)}/{len(test_rs)}）")
    lines.append(f"- dev 半区（允许迭代）：{acc(dev_rs):.1%}（{sum(r['hit'] for r in dev_rs)}/{len(dev_rs)}）\n")

    lines.append("## 按族\n")
    lines.append("| 族 | n 题 | top-1 | 平均 Spearman ρ（预测 vs 实测） | 特征学习差中位 | 刀刃题占比 |")
    lines.append("|---|---|---|---|---|---|")
    for fam in sorted({r["family"] for r in results}):
        rs = [r for r in results if r["family"] == fam]
        rhos = [r["rho"] for r in rs if not math.isnan(r["rho"])]
        flg = sorted(r["fl_gap_max"] for r in rs if not math.isnan(r["fl_gap_max"]))
        knives = [r["knife"] for r in rs if r["knife"] is not None]
        lines.append(
            f"| {fam} | {len(rs)} | {acc(rs):.1%} | "
            f"{(sum(rhos) / len(rhos)) if rhos else float('nan'):.2f} | "
            f"{flg[len(flg) // 2] if flg else float('nan'):.4g} | "
            f"{(sum(knives) / len(knives)) if knives else float('nan'):.0%} |"
        )

    lines.append("\n## 按题型\n")
    lines.append("| qtype | n 题 | top-1 |")
    lines.append("|---|---|---|")
    for qt in sorted({r["qtype"] for r in results}):
        rs = [r for r in results if r["qtype"] == qt]
        lines.append(f"| {qt} | {len(rs)} | {acc(rs):.1%} |")

    lines.append("\n## 刀刃题 vs 非刀刃题\n")
    nonknife = [r for r in results if r["knife"] is False]
    knife = [r for r in results if r["knife"] is True]
    lines.append(f"- 非刀刃题：{acc(nonknife):.1%}（{len(nonknife)} 题）")
    lines.append(f"- 刀刃题：{acc(knife):.1%}（{len(knife)} 题）——理论上不可分辨，预期接近随机")

    lines.append("\n## 按优化器集合\n")
    lines.append("| 优化器集合 | n 题 | top-1 |")
    lines.append("|---|---|---|")
    by_opt = defaultdict(list)
    for r in results:
        by_opt[r["opts"]].append(r)
    for opts, rs in sorted(by_opt.items(), key=lambda kv: -len(kv[1]))[:12]:
        lines.append(f"| {'/'.join(opts)} | {len(rs)} | {acc(rs):.1%} |")

    misses = [r for r in results if not r["hit"]]
    lines.append(f"\n## 错题（{len(misses)} 道）前 20\n")
    lines.append("| qid | 族 | qtype | 预测 | 答案 | ρ | 特征学习差 | 刀刃 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in misses[:20]:
        lines.append(f"| {r['qid']} | {r['family'][:18]} | {r['qtype']} | {r['pred_letter']} | {r['truth']} | "
                     f"{r['rho']:.2f} | {r['fl_gap_max']:.3g} | {r['knife']} |")

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines[:40]))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
