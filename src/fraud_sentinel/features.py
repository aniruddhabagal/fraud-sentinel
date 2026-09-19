"""Deterministic risk features and a transparent rule score.

Two reasons the arithmetic lives here rather than in the prompt:

  1. A sub-3B model is an unreliable calculator. It cannot dependably compare an
     amount against an account's trailing average or count events in a window.
     We compute the numbers exactly and hand the model a pre-digested summary,
     leaving it to do what it is good at: weighing evidence and explaining it.
  2. The delivered data ships no `is_fraud` column, so supervision has to be
     constructed. `rule_score` is the weak-supervision signal that seeds the
     fine-tuning labels, and it doubles as the fallback whenever the SLM is
     unavailable or returns something unusable.

Weights are hand-set from standard card-fraud typology rather than fitted, since
there are no labels to fit against. They are declared in one dict so the whole
scoring policy is auditable at a glance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .sanitize import sanitize_note

HIGH_RISK_MCC = {
    "jewellery", "jewelry", "crypto", "cryptocurrency", "gambling", "casino",
    "fund_transfer", "wire_transfer", "money_transfer", "electronics", "duty_free",
}
HIGH_RISK_MERCHANTS = {"cryptoxchange_global", "luxury_goods_direct", "duty_free_electronics"}

WEIGHTS: dict[str, float] = {
    "injection_attempt":      0.30,   # an attack on the classifier is itself evidence
    "amount_ratio_extreme":   0.16,   # spend far above this account's own norm
    "new_device":             0.12,
    "foreign_txn":            0.10,
    "far_from_home":          0.10,
    "no_auth":                0.14,   # auth_method NONE / card-not-present
    "velocity_24h":           0.10,
    "odd_hour":               0.07,
    "high_risk_mcc":          0.09,
    "overdrawn":              0.09,   # balance driven negative
    "over_credit_limit":      0.08,
    "failed_or_reversed":     0.08,
    "rapid_repeat":           0.08,   # near-instant repeat on the same account
    "weak_kyc":               0.07,
    "high_risk_customer":     0.07,
    "pep":                    0.05,
    "many_devices":           0.05,
    "complaint_history":      0.04,
}

# Above this, a transaction is labelled fraud for weak supervision. Chosen so the
# positive rate lands near the 3-8% band typical of card-fraud datasets rather
# than at an arbitrary 0.5 cut.
FRAUD_THRESHOLD = 0.30


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach sanitized notes, behavioural features, and a blended rule_score."""
    out = df.copy().sort_values("timestamp").reset_index(drop=True)

    # --- Injection defense layers 1-3, applied row-wise ------------------------
    sanitized = out["note"].apply(sanitize_note)
    out["note_clean"] = [s.text for s in sanitized]
    out["injection_attempt"] = [s.injection_attempt for s in sanitized]
    out["injection_patterns"] = [s.pattern_str for s in sanitized]

    # --- Amount relative to the account's own behaviour -----------------------
    # Prefer the supplied ratio; recompute from history where it is missing.
    grp = out.groupby("account_id")["amount"]
    acct_mean = grp.transform("mean")
    ratio = out["amount_to_account_avg_ratio"]
    out["amount_ratio"] = ratio.fillna(out["amount"] / acct_mean.replace(0, np.nan)).fillna(1.0)
    out["amount_ratio_extreme"] = out["amount_ratio"] >= 5.0

    out["over_credit_limit"] = out["credit_limit"].notna() & (out["amount"] > out["credit_limit"])
    out["overdrawn"] = out["balance_after_txn"].notna() & (out["balance_after_txn"] < 0)

    # --- Device, location, authentication -------------------------------------
    out["new_device"] = out["is_new_device"]
    out["foreign_txn"] = out["is_foreign_transaction"] | out["merchant_country"].isin({"", "UNKNOWN"}).eq(False) & out["merchant_country"].ne("IN")
    out["far_from_home"] = out["distance_from_home_km"].fillna(0) > 500
    out["no_auth"] = out["auth_method"].isin({"none", "unknown"}) | (~out["is_card_present"] & out["channel"].isin({"pos", "atm"}))
    out["many_devices"] = out["num_linked_devices"].fillna(0) >= 4

    # --- Temporal behaviour ----------------------------------------------------
    out["hour"] = out["timestamp"].dt.hour
    out["odd_hour"] = out["hour"].between(0, 5)
    out["velocity_24h"] = out["txn_count_last_24h"].fillna(0) >= 5
    out["rapid_repeat"] = out["time_since_prev_txn_mins"].fillna(9999) < 3

    # --- Merchant / status risk ------------------------------------------------
    merch = df.get("merchant_name", pd.Series("", index=df.index)).fillna("")
    merch_key = merch.astype(str).str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    out["high_risk_mcc"] = out["merchant_category"].isin(HIGH_RISK_MCC) | merch_key.isin(HIGH_RISK_MERCHANTS)
    out["failed_or_reversed"] = out["status"].isin({"failed", "reversed", "declined"})

    # --- Customer standing -----------------------------------------------------
    out["weak_kyc"] = ~out["kyc_status"].isin({"verified"})
    out["high_risk_customer"] = out["risk_rating"].isin({"high"})
    out["pep"] = out["is_politically_exposed"]
    out["complaint_history"] = out["num_complaints_last_year"].fillna(0) >= 3

    # --- Blend -----------------------------------------------------------------
    score = sum(out[c].astype(float) * w for c, w in WEIGHTS.items())
    out["rule_score"] = (score / sum(WEIGHTS.values())).clip(0, 1)
    flags = out[list(WEIGHTS)].astype(bool)
    out["rule_flags"] = [", ".join(flags.columns[row]) or "none" for row in flags.to_numpy()]
    out["rule_label"] = (out["rule_score"] >= FRAUD_THRESHOLD).astype(int)
    return out


def feature_summary(row: pd.Series) -> str:
    """Compact, pre-computed evidence block handed to the SLM.

    Every number is formatted here so the model never performs arithmetic.
    """
    limit = "unknown" if pd.isna(row["credit_limit"]) else f"{row['credit_limit']:,.0f}"
    dist = "unknown" if pd.isna(row["distance_from_home_km"]) else f"{row['distance_from_home_km']:,.0f} km"
    bal = "unknown" if pd.isna(row["balance_after_txn"]) else f"{row['balance_after_txn']:,.2f}"
    gap = "unknown" if pd.isna(row["time_since_prev_txn_mins"]) else f"{row['time_since_prev_txn_mins']:.0f} min"
    # NaN is truthy, so `int(x or 0)` raises on a missing count - guard explicitly.
    txns_24h = 0 if pd.isna(row["txn_count_last_24h"]) else int(row["txn_count_last_24h"])
    return (
        f"amount: {row['amount']:,.2f} {row['currency'].upper()} "
        f"({row['amount_ratio']:.1f}x this account's average)\n"
        f"credit_limit: {limit} | balance_after: {bal}\n"
        f"merchant: {row['merchant_category']} | channel: {row['channel']} | type: {row['transaction_type']}\n"
        f"status: {row['status']} | auth_method: {row['auth_method']} | card_present: {bool(row['is_card_present'])}\n"
        f"hour_of_day: {int(row['hour'])} | weekend: {bool(row['is_weekend'])}\n"
        f"new_device: {bool(row['is_new_device'])} | foreign: {bool(row['is_foreign_transaction'])} "
        f"| distance_from_home: {dist}\n"
        f"txns_last_24h: {txns_24h} | time_since_prev_txn: {gap}\n"
        f"customer_kyc: {row['kyc_status']} | risk_rating: {row['risk_rating']} | pep: {bool(row['pep'])}\n"
        f"note_contained_prompt_injection: {bool(row['injection_attempt'])}\n"
        f"triggered_risk_signals: {row['rule_flags']}"
    )
