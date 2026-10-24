"""Build a clean full-bank JSONL after numerical retries.

Any candidate that ever produced a skipped record uses its latest finite retry, falling
back to seed-0 v1 only when no finite retry exists.  All attempts remain in the input file;
this script writes a separate final file and never edits the raw run.  Selection depends
only on numerical failure, not answers.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(HERE / "moving_second_order_s16_500.jsonl"))
    ap.add_argument("--baseline", default=str(HERE / "theory_answerer.jsonl"))
    ap.add_argument("--out", default=str(HERE / "moving_second_order_s16_500_final.jsonl"))
    args = ap.parse_args()
    raw = read_jsonl(args.input)
    baseline = defaultdict(list)
    for r in read_jsonl(args.baseline):
        if not r.get("skipped"):
            baseline[(r["qid"], r["letter"])].append(r)

    failed = {(r["qid"], r["letter"]) for r in raw if r.get("skipped")}
    retries = defaultdict(list)
    for r in raw:
        key = (r["qid"], r["letter"])
        if not r.get("skipped") and key in failed and math.isfinite(r["pred_metric"]):
            retries[key].append(r)
    final = []
    for r in raw:
        key = (r["qid"], r["letter"])
        if not r.get("skipped") and key not in failed:
            final.append(r)
    for key in sorted(failed):
        qid, letter = key
        if retries[key]:
            final.append(retries[key][-1])
            continue
        choices = [r for r in baseline[key]
                   if r["seed"] == 0 and math.isfinite(r["pred_metric"])]
        if len(choices) != 1:
            raise RuntimeError(f"no finite retry or seed-0 v1 record for {key}")
        r = dict(choices[0])
        r["prediction_source"] = "fallback_v1_nonfinite_moving"
        r["second_order_applied"] = True
        r["v1_metric"] = r["pred_metric"]
        r["fallback_reason"] = "moving second order and moving first order were nonfinite"
        final.append(r)

    candidates = defaultdict(set)
    for r in final:
        candidates[r["qid"]].add(r["letter"])
    assert len(candidates) == 500
    assert all(len(letters) == 3 for letters in candidates.values())
    assert len({(r["qid"], r["letter"]) for r in final}) == 1500
    assert all(not r.get("skipped") for r in final)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        for r in final:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")
    print(f"questions=500 candidates=1500 raw_records={len(final)} numerical_fallbacks={len(failed)}")
    print("sources:", dict(Counter(r.get("prediction_source") for r in final)))


if __name__ == "__main__":
    main()
