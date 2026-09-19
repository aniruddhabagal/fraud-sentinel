"""Stratified evaluation: every rule-policy positive plus a negative sample.

The corpus is ~3% positive, so a uniform sample carries too few positives to
measure recall at all - the earlier 60-row comparison had exactly one. This takes
ALL positives and a fixed sample of negatives, which makes recall measurable at a
fraction of the inference cost. Precision is reported on the sampled mix and is
therefore not directly comparable to a full-corpus run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from fraud_sentinel import clean, features, infer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["llama-3.2-1b-instruct", "fraud-sentinel-v2"])
    ap.add_argument("--negatives", type=int, default=90)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--base-url", default="http://localhost:1234/v1")
    args = ap.parse_args()

    merged, _ = clean.run(ROOT / "data" / "raw")
    feat = features.add_features(merged)
    pos = feat[feat.rule_label == 1]
    neg = feat[feat.rule_label == 0].sample(n=min(args.negatives, (feat.rule_label == 0).sum()),
                                            random_state=5)
    sample = pd.concat([pos, neg]).sample(frac=1, random_state=5)
    print(f"stratified sample: {len(sample)} rows ({len(pos)} positive, {len(neg)} negative)\n")

    rows = []
    for name in args.models:
        preds = infer.classify_frame(sample, infer.lm_studio(name, base_url=args.base_url),
                                     max_workers=args.workers, progress=False,
                                     apply_guardrails=False)
        truth = sample.set_index("transaction_id")["rule_label"].to_dict()
        tp = sum(1 for p in preds if p.is_fraud and truth.get(p.transaction_id))
        fp = sum(1 for p in preds if p.is_fraud and not truth.get(p.transaction_id))
        fn = sum(1 for p in preds if not p.is_fraud and truth.get(p.transaction_id))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        rows.append({"model": name, "TP": tp, "FP": fp, "FN": fn,
                     "precision": f"{prec:.1%}", "recall": f"{rec:.1%}", "F1": f"{f1:.1%}"})
        print(f"  {name:<26} TP={tp:<3} FP={fp:<3} FN={fn:<3} "
              f"P={prec:.1%} R={rec:.1%} F1={f1:.1%}")

    out = ROOT / "outputs" / "stratified_eval.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\n{pd.DataFrame(rows).to_string(index=False)}\n\nwritten to {out}")


if __name__ == "__main__":
    main()
