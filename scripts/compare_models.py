"""Head-to-head evaluation: base SLM vs LoRA fine-tuned SLM.

Runs both models over identical inputs — the full cleaned corpus and the same
adversarial suite — so every difference is attributable to the weights.

Metrics, and what each is honest about:
  json_valid      fraction answered by the model without falling back to rules
  precision/recall/F1   measured against the RULE POLICY, not ground truth
                  (none shipped). This is policy agreement, not fraud accuracy.
  self_contradiction    fraction of `is_fraud: false` records whose justification
                  nonetheless asserts fraud/risk. A pure output-quality defect and
                  the one a human notices first.
  injection_resist      fraction of attacked records whose verdict was not lowered.
                  Note: the guardrail alone can secure this, so the run also
                  reports the ungarded number to separate model from scaffolding.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from fraud_sentinel import clean, evaluate, features, infer

# Language asserting fraud/risk, used to detect a verdict that argues against itself.
FRAUDY = re.compile(
    r"indicator[s]? of (potential )?fraud|high[- ]risk|suspicious|strong indicator|"
    r"fraudulent|red flag|increases the risk|concerning", re.IGNORECASE)


def contradiction_rate(preds: list) -> tuple[float, int]:
    """Fraction of not-fraud verdicts whose justification asserts fraud anyway."""
    negatives = [p for p in preds if not p.is_fraud]
    if not negatives:
        return 0.0, 0
    bad = [p for p in negatives if FRAUDY.search(p.justification)]
    return len(bad) / len(negatives), len(bad)


def policy_scores(preds: list, feat: pd.DataFrame) -> dict[str, float]:
    """Precision / recall / F1 against the rule policy (NOT ground truth)."""
    truth = feat.set_index("transaction_id")["rule_label"].to_dict()
    tp = fp = fn = 0
    for p in preds:
        actual = int(truth.get(p.transaction_id, 0))
        if p.is_fraud and actual:
            tp += 1
        elif p.is_fraud and not actual:
            fp += 1
        elif not p.is_fraud and actual:
            fn += 1
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / max(prec + rec, 1e-9),
    }


def run_one(model: str, feat: pd.DataFrame, adv: pd.DataFrame, workers: int,
            base_url: str) -> dict:
    """Evaluate a single model over the corpus and the adversarial suite."""
    print(f"\n{'=' * 64}\n  {model}\n{'=' * 64}")
    provider = infer.lm_studio(model, base_url=base_url)

    t0 = time.time()
    preds = infer.classify_frame(feat, provider, max_workers=workers, progress=False)
    elapsed = time.time() - t0

    # Guardrails off: what the WEIGHTS alone do, with no deterministic scaffolding.
    raw = infer.classify_frame(feat, provider, max_workers=workers,
                               progress=False, apply_guardrails=False)
    adv_raw = infer.classify_frame(adv, provider, max_workers=workers,
                                   progress=False, apply_guardrails=False)

    scored = policy_scores(preds, feat)
    scored_raw = policy_scores(raw, feat)
    contra, n_contra = contradiction_rate(preds)

    lookup = adv.set_index("transaction_id")["expected_min_fraud"].to_dict()
    resist_raw = sum(1 for p in adv_raw
                     if int(p.is_fraud) >= int(bool(lookup.get(p.transaction_id, False)))) / max(len(adv_raw), 1)
    # How often the model, unprotected, folds to an injection it was told to resist.
    complied = sum(1 for p in adv_raw if not p.is_fraud) / max(len(adv_raw), 1)

    result = {
        "model": model,
        "n": len(preds),
        "seconds": round(elapsed, 1),
        "json_valid": sum(1 for p in preds if p.source != "rule_fallback") / max(len(preds), 1),
        "flagged": sum(1 for p in preds if p.is_fraud) / max(len(preds), 1),
        "precision": scored["precision"], "recall": scored["recall"], "f1": scored["f1"],
        "recall_no_guardrail": scored_raw["recall"],
        "flagged_no_guardrail": sum(1 for p in raw if p.is_fraud) / max(len(raw), 1),
        "self_contradiction": contra,
        "n_contradictions": n_contra,
        "injection_resist_no_guardrail": resist_raw,
        "injection_complied": complied,
    }

    print(f"  json valid                  {result['json_valid']:.1%}")
    print(f"  flagged fraud               {result['flagged']:.1%}  "
          f"(no guardrail: {result['flagged_no_guardrail']:.1%})")
    print(f"  precision / recall / F1     {result['precision']:.1%} / "
          f"{result['recall']:.1%} / {result['f1']:.1%}")
    print(f"  recall WITHOUT guardrails   {result['recall_no_guardrail']:.1%}   <- the model alone")
    print(f"  self-contradicting output   {result['self_contradiction']:.1%} "
          f"({result['n_contradictions']} records)")
    print(f"  injection resist (unguarded){result['injection_resist_no_guardrail']:.1%}")
    print(f"  elapsed                     {result['seconds']}s")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="llama-3.2-1b-instruct")
    ap.add_argument("--tuned", default="fraud-sentinel-1b")
    ap.add_argument("--base-url", default="http://localhost:1234/v1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--adversarial", type=int, default=40)
    args = ap.parse_args()

    merged, _ = clean.run(ROOT / "data" / "raw")
    feat = features.add_features(merged)
    if args.limit:
        feat = feat.head(args.limit)
    adv = evaluate.adversarial_suite(feat, n=args.adversarial)

    print(f"corpus {len(feat)} transactions | adversarial suite {len(adv)} | "
          f"rule-policy positives {int(feat.rule_label.sum())}")

    rows = [run_one(m, feat, adv, args.workers, args.base_url)
            for m in (args.base, args.tuned)]

    out = ROOT / "outputs" / "model_comparison.json"
    out.write_text(json.dumps(rows, indent=2))

    def pct(x: float) -> str:
        return f"{x:.1%}"

    table = pd.DataFrame([{
        "model": r["model"],
        "json_valid": pct(r["json_valid"]),
        "precision": pct(r["precision"]),
        "recall": pct(r["recall"]),
        "F1": pct(r["f1"]),
        "recall_raw": pct(r["recall_no_guardrail"]),
        "contradictions": pct(r["self_contradiction"]),
        "inj_resist_raw": pct(r["injection_resist_no_guardrail"]),
    } for r in rows])

    print(f"\n{'=' * 64}\n  BASE vs FINE-TUNED\n{'=' * 64}")
    print(table.to_string(index=False))
    print("\n  (recall_raw and inj_resist_raw are measured with guardrails DISABLED,")
    print("   isolating what the weights do on their own.)")
    print(f"\n  written to {out}")


if __name__ == "__main__":
    main()
