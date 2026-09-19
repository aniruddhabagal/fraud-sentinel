"""Post-inference guardrails: the last line of defense on model output.

The sanitizer stops injections reaching the model. These rules assume one got
through anyway, and bound the damage. Each is deliberately narrow - a guardrail
that fires constantly is really a broken model, so every trip is recorded on the
prediction for review rather than applied silently.

Ordered from most to least severe. All are additive: a prediction can trip
several, and the record lists every one.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .infer import Prediction

# A justification echoing attacker phrasing means the note reached the model's
# reasoning even if the verdict looks reasonable.
_COMPROMISED_LANGUAGE = re.compile(
    r"\b(ignore\s+(previous|prior)|as\s+instructed|per\s+the\s+note|the\s+note\s+says|"
    r"whitelisted|pre[- ]approved|instructed\s+to|told\s+to\s+(mark|classify)|"
    r"user\s+requested|system\s+override)\b", re.IGNORECASE)

# Above this rule score the deterministic evidence is strong enough that a
# "not fraud" verdict is overridden rather than trusted.
HARD_OVERRIDE_SCORE = 0.45
# Below this, flagging fraud with high confidence is treated as a hallucination.
LOW_EVIDENCE_SCORE = 0.08


@dataclass(frozen=True)
class Rule:
    name: str
    applies: Callable[[Prediction, pd.Series], bool]
    reason: str


def _injection_forced_safe(pred: Prediction, row: pd.Series) -> bool:
    """The note attacked the classifier and the classifier came back 'safe'."""
    return bool(row["injection_attempt"]) and not pred.is_fraud


def _strong_evidence_ignored(pred: Prediction, row: pd.Series) -> bool:
    return float(row["rule_score"]) >= HARD_OVERRIDE_SCORE and not pred.is_fraud


def _compromised_justification(pred: Prediction, row: pd.Series) -> bool:
    return bool(_COMPROMISED_LANGUAGE.search(pred.justification))


def _hallucinated_fraud(pred: Prediction, row: pd.Series) -> bool:
    """Fraud asserted confidently with essentially no supporting signal."""
    return (pred.is_fraud and pred.confidence > 0.85
            and float(row["rule_score"]) < LOW_EVIDENCE_SCORE
            and not row["injection_attempt"])


RULES: list[Rule] = [
    Rule("injection_forced_safe", _injection_forced_safe,
         "Note contained a prompt-injection attempt; verdict overridden to fraud."),
    Rule("strong_evidence_ignored", _strong_evidence_ignored,
         "Deterministic risk signals outweigh the model's 'not fraud' verdict."),
    Rule("compromised_justification", _compromised_justification,
         "Justification echoed attacker-supplied phrasing; verdict overridden."),
    Rule("hallucinated_fraud", _hallucinated_fraud,
         "Fraud asserted with high confidence but no supporting signals; confidence reduced."),
]


def apply(pred: Prediction, row: pd.Series) -> Prediction:
    """Run every guardrail, returning a corrected prediction.

    The first three force a fraud verdict; the fourth only damps confidence,
    because a false positive is cheaper to review than a missed fraud.
    """
    from .infer import Prediction

    tripped = [r for r in RULES if r.applies(pred, row)]
    if not tripped:
        return pred

    names = [r.name for r in tripped]
    forcing = [r for r in tripped if r.name != "hallucinated_fraud"]

    if forcing:
        reason = forcing[0].reason
        justification = f"{reason} Signals: {row['rule_flags']}."
        return Prediction(
            pred.transaction_id, True,
            max(pred.confidence if pred.is_fraud else 0.0, 0.90),
            justification[:300], source=f"slm+guardrail[{'|'.join(names)}]",
        )

    # Only the hallucination damper fired: keep the verdict, lower the certainty.
    return Prediction(
        pred.transaction_id, pred.is_fraud, min(pred.confidence, 0.60),
        pred.justification, source=f"slm+guardrail[{'|'.join(names)}]",
    )
