"""End-to-end pipeline: clean -> merge -> sanitize -> score -> classify -> JSON.

Runs top to bottom and writes three artefacts to outputs/:
  predictions.jsonl  one strict-schema record per surviving transaction
  quarantine.csv     every rejected row with the rule that rejected it
  run_report.json    cleaning stats, eval metrics, and provenance

Usage:
  python scripts/run_pipeline.py                     # base model via LM Studio
  python scripts/run_pipeline.py --model my-tuned    # a fine-tuned model
  python scripts/run_pipeline.py --limit 50          # quick smoke run
  python scripts/run_pipeline.py --no-slm            # rule engine only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_sentinel import clean, evaluate, features, infer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--out", type=Path, default=ROOT / "outputs")
    p.add_argument("--model", default="llama-3.2-1b-instruct")
    p.add_argument("--base-url", default="http://localhost:1234/v1")
    p.add_argument("--limit", type=int, default=None, help="classify only the first N rows")
    p.add_argument("--top-risk", type=int, default=None,
                   help="classify the N highest-risk rows instead of the first N. A "
                        "chronological slice holds almost no fraud (~3%% base rate), so "
                        "--limit cannot show whether a model detects anything")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--adversarial", type=int, default=40, help="size of the injected attack suite")
    p.add_argument("--no-slm", action="store_true", help="rule engine only, no model calls")
    p.add_argument("--no-guardrails", action="store_true", help="ablation: disable output guardrails")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # A partial run must never overwrite a full submission. --limit/--top-risk and
    # --no-guardrails produce diagnostics, not deliverables; writing them to the
    # default outputs/ has silently truncated predictions.jsonl more than once.
    partial = args.limit or args.top_risk or args.no_guardrails or args.no_slm
    default_out = args.out.resolve() == (ROOT / "outputs").resolve()
    if partial and default_out:
        raise SystemExit(
            "Refusing to write a partial or ablation run to outputs/, which holds the "
            "full submission.\n"
            "Pass --out outputs/<name> for diagnostics, or drop the flags for a full run."
        )

    t0 = time.time()

    # ---- 1. Clean and merge --------------------------------------------------
    print("[1/5] cleaning + merging three relational tables")
    merged, report = clean.run(args.data)
    print(report.summary())
    if not report.quarantine.empty:
        report.quarantine.to_csv(args.out / "quarantine.csv", index=False)
        print(f"  -> {len(report.quarantine)} rejected rows written to quarantine.csv")

    # ---- 2. Sanitize + engineer features ------------------------------------
    print("\n[2/5] sanitizing notes + engineering risk features")
    feat = features.add_features(merged)
    if args.top_risk:
        feat = feat.nlargest(args.top_risk, "rule_score")
        print(f"  selecting the {args.top_risk} highest-risk transactions "
              f"({int(feat.rule_label.sum())} are policy-positive)")
    n_inj = int(feat["injection_attempt"].sum())
    print(f"  prompt injections neutralized : {n_inj}")
    print(f"  rule-flagged fraud            : {int(feat.rule_label.sum())} "
          f"({feat.rule_label.mean():.1%})")

    # ---- 3. Classify ---------------------------------------------------------
    if args.no_slm:
        print("\n[3/5] SLM disabled - using rule engine only")
        preds = [infer.rule_fallback(r) for _, r in
                 (feat.head(args.limit) if args.limit else feat).iterrows()]
        provider_name = "rule_engine"
    else:
        print(f"\n[3/5] classifying via SLM '{args.model}'")
        provider = infer.lm_studio(args.model, base_url=args.base_url)
        provider_name = provider.name
        preds = infer.classify_frame(
            feat, provider, max_workers=args.workers, limit=args.limit,
            apply_guardrails=not args.no_guardrails,
        )

    # ---- 4. Adversarial robustness ------------------------------------------
    adv_result = None
    if args.adversarial and not args.no_slm:
        print(f"\n[4/5] adversarial suite: injecting {args.adversarial} held-out attacks")
        adv = evaluate.adversarial_suite(feat, n=args.adversarial)
        det = evaluate.detection_report(adv)
        print(f"  sanitizer caught {det['caught']}/{det['attacks']} "
              f"({det['detection_rate']:.0%}) before inference")
        adv_preds = infer.classify_frame(adv, infer.lm_studio(args.model, base_url=args.base_url),
                                         max_workers=args.workers, progress=False,
                                         apply_guardrails=not args.no_guardrails)
        adv_result = evaluate.score(adv_preds, adv, provider_name, adversarial=True)
        print(f"  injection resistance: {adv_result.injection_resistance:.1%}")
    else:
        print("\n[4/5] adversarial suite skipped")

    # ---- 5. Write outputs ----------------------------------------------------
    print("\n[5/5] writing outputs")
    scored = feat.head(args.limit) if args.limit else feat
    main_result = evaluate.score(preds, scored, provider_name)

    pred_path = args.out / "predictions.jsonl"
    with pred_path.open("w") as fh:
        for p in preds:
            fh.write(json.dumps(p.to_record()) + "\n")

    elapsed = time.time() - t0
    run_report = {
        "provider": provider_name,
        "guardrails_enabled": not args.no_guardrails,
        "elapsed_seconds": round(elapsed, 1),
        "cleaning": report.stats,
        "quarantined_rows": len(report.quarantine),
        "injections_neutralized": n_inj,
        "metrics": main_result.row(),
        "adversarial": adv_result.row() if adv_result else None,
    }
    (args.out / "run_report.json").write_text(json.dumps(run_report, indent=2))

    # ---- Summary -------------------------------------------------------------
    print(f"\n{'=' * 62}")
    print(f"  predictions      : {len(preds)} -> {pred_path}")
    print(f"  model answered   : {main_result.model_answered:.1%}")
    print(f"  flagged as fraud : {main_result.fraud_rate:.1%}")
    print(f"  agrees w/ rules  : {main_result.agreement:.1%}")
    print(f"  rule fallbacks   : {main_result.fallback_rate:.1%}")
    print(f"  guardrails fired : {main_result.guardrail_rate:.1%}")
    if adv_result:
        print(f"  injection resist : {adv_result.injection_resistance:.1%}")
    print(f"  elapsed          : {elapsed:.1f}s")
    print("=" * 62)


if __name__ == "__main__":
    main()
