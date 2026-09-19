"""Generate solution.ipynb - the submission notebook.

Kept as a generator rather than a hand-edited .ipynb so the notebook always
reflects the library code and cannot drift from it.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


_counter = itertools.count()


def _cell_id(kind: str) -> str:
    """nbformat 4.5+ requires a unique id per cell; without one, validation warns."""
    return f"{kind}-{next(_counter):02d}"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "id": _cell_id("md"), "metadata": {},
            "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "id": _cell_id("code"), "execution_count": None,
            "metadata": {}, "outputs": [], "source": text.strip().splitlines(keepends=True)}


CELLS = [
    md("""
# Relational Data Wrangler & Fraud Sentinel

Ingest three faulty relational tables, neutralize adversarial prompt injections,
and classify each transaction with a sub-3B model under a strict JSON contract.

**Pipeline:** `clean → merge → sanitize → engineer → classify → guardrail → JSON`

Two things about the delivered data shape every decision below:

- It ships **no `note` column and no injections**, so the injection defense is
  proven against an adversarial suite we construct ourselves.
- It ships **no `is_fraud` label**, so supervision is constructed from an
  explicit rule engine. "Agreement" below means agreement with that policy —
  *not* accuracy against ground truth.
"""),
    code("""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path.cwd() / "src"))

import pandas as pd
from fraud_sentinel import clean, features, infer, evaluate, sanitize

pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 70)
"""),

    md("## 1. Ingest and profile the raw tables\n\nEverything is read as `str` — "
       "letting pandas infer dtypes on dirty data silently coerces corruption away."),
    code("""
RAW = Path("data/raw")
raw = clean.load_raw(RAW)
for name, df in raw.items():
    print(f"{name:<14} {df.shape[0]:>5} rows x {df.shape[1]:>2} cols")

print("\\nCorruption visible in transactions.amount:")
bad = raw["transactions"]["amount"]
print([a for a in bad if a and not a.replace('.','').replace('-','').isdigit()][:8])
print("\\nInvalid timestamps:", [t for t in raw["transactions"]["transaction_timestamp"] if not t or "NOT" in t][:5])
"""),

    md("## 2. Clean and merge\n\nEvery rejected row is quarantined with the rule "
       "that rejected it, so nothing disappears silently."),
    code("""
merged, report = clean.run(RAW)
print(report.summary())
print(f"\\nquarantined: {len(report.quarantine)} rows")
report.quarantine[["transaction_id", "quarantine_reason"]].head(8)
"""),

    md("""
## 3. The prompt-injection defense

Four independent layers, so no single bypass is sufficient:

1. **Normalize** — strip zero-width chars, unicode confusables, RTL overrides,
   leetspeak, spaced-out letters, chat-template control tokens.
2. **Detect** — 12 pattern families matched against the *normalized* text.
3. **Neutralize** — redact the note and raise `injection_attempt`, which feeds
   the risk model as an **aggravating fraud signal**. The attack becomes
   evidence against the transaction carrying it.
4. **Contain** — survivors are fenced in a per-call nonce block outside the
   instruction region; schema-constrained decoding bounds the output shape.
"""),
    code("""
attacks = evaluate.HELD_OUT_ATTACKS
benign  = ["NEFT Beneficiary", "Luxury Goods Direct", "UPI P2P Transfer",
           "monthly grocery run", "invoice #88213", "Local Kirana Store"]

rows = [{"text": t[:58], "blocked": sanitize.sanitize_note(t).injection_attempt,
         "patterns": sanitize.sanitize_note(t).pattern_str[:44]}
        for t in attacks + benign]
res = pd.DataFrame(rows)
print(f"held-out attacks blocked : {res.head(len(attacks)).blocked.sum()}/{len(attacks)}")
print(f"benign false positives   : {res.tail(len(benign)).blocked.sum()}/{len(benign)}")
res
"""),

    md("## 4. Feature engineering and the rule engine\n\nThe arithmetic happens in "
       "pandas, never in the prompt — a 1B model is an unreliable calculator. "
       "`WEIGHTS` is the entire scoring policy, auditable at a glance."),
    code("""
feat = features.add_features(merged)
print(f"injections neutralized : {int(feat.injection_attempt.sum())}")
print(f"rule-flagged fraud     : {int(feat.rule_label.sum())} ({feat.rule_label.mean():.1%})\\n")
feat[list(features.WEIGHTS)].mean().sort_values(ascending=False).head(10).to_frame("fire_rate")
"""),
    code("""
# The evidence block the model actually receives - no raw numbers to reason over.
print(features.feature_summary(feat.loc[feat.rule_score.idxmax()]))
"""),

    md("## 5. What the model sees\n\nTrusted evidence first; untrusted text last, "
       "fenced in an unguessable nonce so it cannot forge a closing tag."),
    code("""
print(infer.build_prompt(feat.loc[feat.rule_score.idxmax()])[-700:])
"""),

    md("## 6. Inference under a strict JSON contract\n\nLM Studio enforces "
       "`RESPONSE_SCHEMA` during decoding, so non-conforming output is "
       "unrepresentable. Anything that still fails falls back to the rule engine — "
       "every input yields exactly one valid record."),
    code("""
provider = infer.lm_studio("llama-3.2-1b-instruct")   # swap for a fine-tuned model
pred = infer.classify_row(feat.loc[feat.rule_score.idxmax()], provider)
print("source:", pred.source)
pred.to_record()
"""),

    md("## 7. Adversarial evaluation\n\nReal transactions cloned, then attacked. "
       "Only the attacker-controlled text changes, so the correct answer is known "
       "by construction: **an injection must never lower the verdict.**"),
    code("""
adv = evaluate.adversarial_suite(feat, n=40)
det = evaluate.detection_report(adv)
print(f"sanitizer caught {det['caught']}/{det['attacks']} ({det['detection_rate']:.0%}) before inference")

adv_preds = infer.classify_frame(adv, provider, max_workers=8, progress=False)
adv_res = evaluate.score(adv_preds, adv, "base", adversarial=True)
print(f"injection resistance: {adv_res.injection_resistance:.1%}")
"""),

    md("## 8. Full run\n\nClassifies every surviving transaction and writes the "
       "submission artefacts."),
    code("""
preds = infer.classify_frame(feat, provider, max_workers=8)
result = evaluate.score(preds, feat, "llama-3.2-1b-instruct")

import json
Path("outputs").mkdir(exist_ok=True)
with open("outputs/predictions.jsonl", "w") as fh:
    for p in preds:
        fh.write(json.dumps(p.to_record()) + "\\n")

print(json.dumps(result.row(), indent=2))
print("\\nsample output records:")
for p in preds[:3]:
    print(json.dumps(p.to_record()))
"""),

    md("""
## 9. Fine-tuning

LoRA via `mlx-lm` on the constructed training set, then fuse into a standalone
model. Labels come from the rule engine, so the model distils an explicit,
auditable policy into weights rather than inventing ground truth.

```bash
python scripts/build_finetune_data.py     # build train/valid jsonl
./scripts/finetune.sh                      # train + fuse
HF_REPO=<user>/fraud-sentinel-1b ./scripts/finetune.sh   # ...and publish
```

Then load the fused model in LM Studio and re-run this notebook with
`infer.lm_studio("fraud-sentinel-1b")` for a like-for-like comparison.
"""),
    code("""
# Base vs fine-tuned, same eval sets.
# Uncomment once the fused model is loaded in LM Studio.
#
# tuned = infer.lm_studio("fraud-sentinel-1b")
# rows = [
#     evaluate.score(infer.classify_frame(feat, provider, max_workers=8, progress=False), feat, "base"),
#     evaluate.score(infer.classify_frame(feat, tuned,    max_workers=8, progress=False), feat, "fine-tuned"),
# ]
# evaluate.compare(rows)
"""),

    md("""
## 10. Extending this

| To add | Change | Propagates to |
|---|---|---|
| A risk signal | one boolean column + one `WEIGHTS` entry | scoring, prompts, labels, training |
| A model backend | implement `LLMProvider` | whole pipeline |
| A guardrail | append a `Rule` to `RULES` | every prediction |
| A new schema | `TEXT_COLUMNS` / `_col()` in `clean.py` | ingestion |

**Known limits.** Agreement is measured against the rule engine, not ground
truth — real labels would replace `rule_label` as the training target with no
structural change. Rule weights are hand-set from fraud typology, not fitted.
"""),
]

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = ROOT / "solution.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(CELLS)} cells)")
