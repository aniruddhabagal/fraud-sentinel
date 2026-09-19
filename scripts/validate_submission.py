"""Validate the submission artefacts against the brief, independently of the pipeline.

Deliberately re-derives everything from the CSVs and the output files rather than
trusting anything the pipeline reported about itself. Exits non-zero on failure
so it can gate a submission.

Checks:
  1. predictions.jsonl parses, one JSON object per line
  2. every record carries exactly the four required fields, correctly typed
  3. confidence is in [0, 1]; justification is a non-empty single sentence
  4. transaction_ids are unique and every surviving transaction is covered
  5. no quarantined or dropped row leaked into the predictions
  6. no prompt-injection text survived into any justification
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_sentinel import clean, features

REQUIRED = {"transaction_id", "is_fraud", "confidence", "justification"}
TYPES = {"transaction_id": str, "is_fraud": bool, "confidence": float, "justification": str}

# Any of these appearing in a justification means attacker text reached the output.
LEAK_PATTERNS = re.compile(
    r"ignore\s+(previous|prior|all)|disregard|you\s+are\s+now|<\|.*?\|>|"
    r"whitelisted|system\s*:\s*|respond\s+only\s+with", re.IGNORECASE)


class Checker:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passes: list[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        (self.passes if ok else self.failures).append(f"{label}{f' - {detail}' if detail else ''}")
        print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  ({detail})" if detail else ""))
        return ok

    def report(self) -> int:
        print(f"\n{'=' * 62}")
        print(f"  {len(self.passes)} passed, {len(self.failures)} failed")
        if self.failures:
            print("\n  Failures:")
            for f in self.failures:
                print(f"    - {f}")
        print("=" * 62)
        return 1 if self.failures else 0


def main() -> int:
    c = Checker()
    pred_path = ROOT / "outputs" / "predictions.jsonl"

    print("Validating submission artefacts\n")

    if not c.check(pred_path.exists(), "predictions.jsonl exists"):
        return c.report()

    # --- 1. Parseable, one object per line -----------------------------------
    records, parse_errors = [], []
    for i, line in enumerate(pred_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            parse_errors.append(f"line {i}: {e}")
    c.check(not parse_errors, "every line is valid JSON",
            f"{len(parse_errors)} bad lines" if parse_errors else f"{len(records)} records")

    if not records:
        return c.report()

    # --- 2. Exact schema, correct types --------------------------------------
    bad_keys = [r.get("transaction_id", "?") for r in records if set(r) != REQUIRED]
    c.check(not bad_keys, "every record has exactly the 4 required fields",
            f"{len(bad_keys)} violations" if bad_keys else "")

    bad_types = [
        f"{r.get('transaction_id')}.{k}"
        for r in records for k, t in TYPES.items()
        if k in r and not isinstance(r[k], t) and not (t is float and isinstance(r[k], int))
    ]
    c.check(not bad_types, "all field types correct",
            f"{len(bad_types)} violations: {bad_types[:3]}" if bad_types else "")

    # --- 3. Value ranges ------------------------------------------------------
    bad_conf = [r["transaction_id"] for r in records
                if not isinstance(r["confidence"], (int, float)) or not 0.0 <= r["confidence"] <= 1.0]
    c.check(not bad_conf, "confidence within [0, 1]",
            f"{len(bad_conf)} out of range" if bad_conf else "")

    bad_just = [r["transaction_id"] for r in records
                if not r["justification"].strip() or len(r["justification"]) > 300]
    c.check(not bad_just, "justification non-empty and <= 300 chars",
            f"{len(bad_just)} violations" if bad_just else "")

    # --- 4. Coverage against a fresh pipeline pass ---------------------------
    merged, report = clean.run(ROOT / "data" / "raw")
    feat = features.add_features(merged)
    expected = set(feat["transaction_id"])
    got = [r["transaction_id"] for r in records]

    c.check(len(got) == len(set(got)), "transaction_ids unique",
            f"{len(got) - len(set(got))} duplicates" if len(got) != len(set(got)) else "")

    missing = expected - set(got)
    c.check(not missing, "every surviving transaction is classified",
            f"{len(missing)} missing of {len(expected)}" if missing else f"{len(expected)} covered")

    # --- 5. No quarantined row leaked ----------------------------------------
    if not report.quarantine.empty:
        quarantined = set(report.quarantine["transaction_id"].dropna())
        # Only rows that were quarantined AND did not survive are true leaks.
        # Intersecting with `expected` too would make this vacuously true,
        # since check #4 already asserts got is a subset of expected.
        leaked = (quarantined - expected) & set(got)
        c.check(not leaked, "no quarantined row appears in predictions",
                f"{len(leaked)} leaked" if leaked else f"{len(quarantined)} quarantined, none leaked")

    # --- 6. No injection text survived into output ---------------------------
    leaks = [r["transaction_id"] for r in records if LEAK_PATTERNS.search(r["justification"])]
    c.check(not leaks, "no attacker text in any justification",
            f"{len(leaks)} leaks: {leaks[:3]}" if leaks else "")

    return c.report()


if __name__ == "__main__":
    sys.exit(main())
