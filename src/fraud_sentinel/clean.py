"""Load, clean, and relationally merge the three faulty source tables.

Design rules:
  - Everything is read as `str`. Letting pandas infer dtypes on dirty data
    silently coerces "N/A" into NaN in some columns and keeps the literal string
    in others, hiding exactly the corruption we are meant to handle.
  - Nothing is dropped silently. Every rejected row is appended to a quarantine
    frame with the rule that rejected it, so the losses are auditable.
  - Column handling is defensive: the brief's schema and the delivered schema
    disagree, so every optional column is fetched through `_col`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_NULL_TOKENS = {"", "n/a", "na", "null", "none", "nan", "-", "unknown", "?",
                "missing", "not_available", "not available", "unavailable"}

# Free-text columns that could carry an adversarial payload. The delivered data
# has no `note` column, but the brief promises one, so we sanitize whichever of
# these is present and concatenate them into a single untrusted `note` field.
TEXT_COLUMNS = ["note", "notes", "description", "remarks", "memo", "comment",
                "transaction_note", "merchant_name"]


@dataclass
class CleanReport:
    """Audit trail for everything the cleaner changed or rejected."""
    quarantine: pd.DataFrame = field(default_factory=pd.DataFrame)
    stats: dict[str, int] = field(default_factory=dict)

    def note(self, key: str, value: int) -> None:
        self.stats[key] = int(value)

    def reject(self, rows: pd.DataFrame, reason: str) -> None:
        if rows.empty:
            return
        block = rows.copy()
        block["quarantine_reason"] = reason
        self.quarantine = pd.concat([self.quarantine, block], ignore_index=True)

    def summary(self) -> str:
        return "\n".join(f"  {k:.<52} {v:>6}" for k, v in self.stats.items())


def _col(df: pd.DataFrame, name: str, default: str = "") -> pd.Series:
    """Fetch a column that may not exist in this delivery of the schema."""
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index, dtype=object)


def _norm_key(s: pd.Series) -> pd.Series:
    """Join keys arrive padded and inconsistently cased; normalize before merging."""
    return s.fillna("").astype(str).str.strip().str.upper()


def _norm_cat(s: pd.Series) -> pd.Series:
    """Categoricals arrive as '  PURCHASE ', 'Pos', 'success' - fold them together."""
    out = s.fillna("").astype(str).str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    return out.where(~out.isin(_NULL_TOKENS), "unknown")


def _is_null_token(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.lower().isin(_NULL_TOKENS)


def parse_amount(s: pd.Series) -> pd.Series:
    """Handle 'INR 62146.26', '41,677.41', '$5,000', 'N/A', '' and bare numbers."""
    cleaned = (
        s.fillna("").astype(str).str.strip()
        .str.replace(r"[^\d.\-]", "", regex=True)   # drop currency codes, symbols, commas
        .str.replace(r"(?<=.)-", "", regex=True)    # keep only a leading minus
    )
    cleaned = cleaned.where(~_is_null_token(s), "")
    return pd.to_numeric(cleaned, errors="coerce")


def parse_timestamp(s: pd.Series) -> pd.Series:
    """Parse mixed date formats, leaving NaT for 'NOT_AVAILABLE' and friends."""
    raw = s.fillna("").astype(str).str.strip()
    raw = raw.where(~_is_null_token(s), "")
    out = pd.to_datetime(raw, errors="coerce", format="mixed", dayfirst=False)

    # Epoch seconds arrive as bare digit strings and must not be read as years.
    epoch = out.isna() & raw.str.fullmatch(r"\d{9,11}")
    if epoch.any():
        out.loc[epoch] = pd.to_datetime(raw[epoch].astype("int64"), unit="s", errors="coerce")
    return out


def parse_numeric(s: pd.Series, positive_only: bool = False) -> pd.Series:
    """Coerce a numeric column, treating null tokens and (optionally) <=0 as missing."""
    cleaned = s.fillna("").astype(str).str.strip().str.replace(r"[^\d.\-eE]", "", regex=True)
    cleaned = cleaned.where(~_is_null_token(s), "")
    out = pd.to_numeric(cleaned, errors="coerce")
    return out.where(out > 0, np.nan) if positive_only else out


def parse_bool(s: pd.Series) -> pd.Series:
    """'Y'/'N', 'TRUE'/'FALSE', '1'/'0' and blanks all appear in this data."""
    raw = s.fillna("").astype(str).str.strip().str.lower()
    return raw.isin({"y", "yes", "true", "1", "1.0", "t"})


def load_raw(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Read all three tables as strings so no corruption is silently coerced."""
    return {
        name: pd.read_csv(raw_dir / f"{name}.csv", dtype=str, keep_default_na=False)
        for name in ("transactions", "accounts", "customers")
    }


def clean_customers(df: pd.DataFrame, rep: CleanReport) -> pd.DataFrame:
    out = df.copy()
    out["customer_id"] = _norm_key(out["customer_id"])

    out["cust_country"] = _norm_cat(_col(out, "country")).str.upper()
    out["kyc_status"] = _norm_cat(_col(out, "kyc_status"))
    out["risk_rating"] = _norm_cat(_col(out, "risk_rating"))
    out["customer_segment"] = _norm_cat(_col(out, "customer_segment"))
    out["cust_age"] = parse_numeric(_col(out, "age"), positive_only=True)
    out["annual_income"] = parse_numeric(_col(out, "annual_income"), positive_only=True)
    out["is_politically_exposed"] = parse_bool(_col(out, "is_politically_exposed"))
    out["email_verified"] = parse_bool(_col(out, "email_verified"))
    out["phone_verified"] = parse_bool(_col(out, "phone_verified"))
    out["num_complaints_last_year"] = parse_numeric(_col(out, "num_complaints_last_year")).fillna(0)
    out["customer_since"] = parse_timestamp(_col(out, "customer_since"))

    bad = out["customer_id"].eq("")
    rep.reject(out[bad], "customer_id missing")
    rep.note("customers: rejected (no id)", bad.sum())
    out = out[~bad]

    dupes = out["customer_id"].duplicated().sum()
    rep.note("customers: duplicate ids collapsed", dupes)
    out = out.drop_duplicates(subset="customer_id", keep="first")

    keep = ["customer_id", "cust_country", "kyc_status", "risk_rating", "customer_segment",
            "cust_age", "annual_income", "is_politically_exposed", "email_verified",
            "phone_verified", "num_complaints_last_year", "customer_since"]
    rep.note("customers: clean rows", len(out))
    return out[keep]


def clean_accounts(df: pd.DataFrame, rep: CleanReport) -> pd.DataFrame:
    out = df.copy()
    out["account_id"] = _norm_key(out["account_id"])
    out["customer_id"] = _norm_key(out["customer_id"])

    out["account_type"] = _norm_cat(_col(out, "account_type"))
    out["account_status"] = _norm_cat(_col(out, "account_status"))
    out["account_tier"] = _norm_cat(_col(out, "account_tier"))
    out["card_type"] = _norm_cat(_col(out, "card_type"))

    # 0.0 is a null sentinel here, not a real limit - parse_numeric nulls it.
    out["credit_limit"] = parse_numeric(_col(out, "credit_limit"), positive_only=True)
    out["current_balance"] = parse_numeric(_col(out, "current_balance"))
    out["avg_monthly_balance_6m"] = parse_numeric(_col(out, "avg_monthly_balance_6m"))
    out["credit_utilization_pct"] = parse_numeric(_col(out, "credit_utilization_pct"))
    out["num_linked_devices"] = parse_numeric(_col(out, "num_linked_devices")).fillna(0)
    out["avg_monthly_txn_count"] = parse_numeric(_col(out, "avg_monthly_txn_count"))
    out["overdraft_enabled"] = parse_bool(_col(out, "overdraft_enabled"))
    out["is_joint_account"] = parse_bool(_col(out, "is_joint_account"))
    out["open_date"] = parse_timestamp(_col(out, "open_date"))
    out["close_date"] = parse_timestamp(_col(out, "close_date"))

    rep.note("accounts: corrupt/zero credit limits -> null", int(out["credit_limit"].isna().sum()))

    bad = out["account_id"].eq("")
    rep.reject(out[bad], "account_id missing")
    rep.note("accounts: rejected (no id)", bad.sum())
    out = out[~bad]

    # Duplicate account rows carry conflicting payloads. Keep the most complete
    # record per id rather than an arbitrary first, so we do not retain the row
    # whose credit_limit and status happened to be blank.
    before = len(out)
    completeness = out.notna().sum(axis=1) + out.ne("unknown").sum(axis=1)
    out = (out.assign(_c=completeness).sort_values("_c", ascending=False)
              .drop_duplicates(subset="account_id", keep="first").drop(columns="_c"))
    rep.note("accounts: duplicates collapsed", before - len(out))

    keep = ["account_id", "customer_id", "account_type", "account_status", "account_tier",
            "card_type", "credit_limit", "current_balance", "avg_monthly_balance_6m",
            "credit_utilization_pct", "num_linked_devices", "avg_monthly_txn_count",
            "overdraft_enabled", "is_joint_account", "open_date", "close_date"]
    rep.note("accounts: clean rows", len(out))
    return out[keep]


def clean_transactions(df: pd.DataFrame, rep: CleanReport) -> pd.DataFrame:
    out = df.copy()
    rep.note("transactions: raw rows", len(out))

    out["transaction_id"] = _norm_key(out["transaction_id"])
    out["account_id"] = _norm_key(out["account_id"])
    out["customer_id"] = _norm_key(out["customer_id"])
    out["amount"] = parse_amount(out["amount"])
    out["timestamp"] = parse_timestamp(_col(out, "transaction_timestamp"))

    for c in ("transaction_type", "channel", "status", "merchant_category",
              "merchant_city", "merchant_country", "device_type", "auth_method", "currency"):
        out[c] = _norm_cat(_col(out, c))
    out["merchant_country"] = out["merchant_country"].str.upper()

    out["is_new_device"] = parse_bool(_col(out, "is_new_device"))
    out["is_card_present"] = parse_bool(_col(out, "is_card_present"))
    out["is_foreign_transaction"] = parse_bool(_col(out, "is_foreign_transaction"))
    out["is_weekend"] = parse_bool(_col(out, "is_weekend"))
    for c in ("distance_from_home_km", "time_since_prev_txn_mins", "txn_count_last_24h",
              "txn_count_last_7d", "amount_to_account_avg_ratio", "balance_after_txn"):
        out[c] = parse_numeric(_col(out, c))
    out["device_id"] = _norm_key(_col(out, "device_id"))
    out["ip_address"] = _col(out, "ip_address").fillna("").astype(str).str.strip()
    out["merchant_id"] = _norm_key(_col(out, "merchant_id"))

    # Consolidate every free-text field into one untrusted `note` for sanitizing.
    present = [c for c in TEXT_COLUMNS if c in df.columns]
    rep.note("transactions: free-text columns sanitized", len(present))
    if present:
        out["note"] = (df[present].fillna("").astype(str)
                       .apply(lambda r: " | ".join(x.strip() for x in r if x.strip()), axis=1))
    else:
        out["note"] = ""

    checks = [
        (out["transaction_id"].eq(""), "transaction_id missing"),
        (out["account_id"].eq(""), "account_id missing"),
        (out["amount"].isna(), "amount unparsable (N/A, blank, non-numeric)"),
        (out["amount"] <= 0, "amount non-positive"),
        (out["timestamp"].isna(), "timestamp invalid or unparsable"),
    ]
    drop = pd.Series(False, index=out.index)
    for mask, reason in checks:
        mask = mask.fillna(False) & ~drop      # attribute each row to its first failure
        rep.reject(out[mask], reason)
        rep.note(f"transactions: dropped - {reason}", mask.sum())
        drop |= mask
    out = out[~drop]

    dupes = out["transaction_id"].duplicated(keep="first")
    rep.reject(out[dupes], "duplicate transaction_id")
    rep.note("transactions: duplicate ids dropped", dupes.sum())
    out = out[~dupes]

    rep.note("transactions: clean rows", len(out))
    return out


def merge_all(tx: pd.DataFrame, acc: pd.DataFrame, cust: pd.DataFrame,
              rep: CleanReport) -> pd.DataFrame:
    """Left-join transactions -> accounts -> customers, quarantining orphan FKs."""
    merged = tx.merge(acc, on="account_id", how="left", indicator="_acc",
                      validate="m:1", suffixes=("", "_acct"))

    orphans = merged["_acc"].eq("left_only")
    rep.reject(merged[orphans].drop(columns="_acc"), "orphan account_id (no matching account)")
    rep.note("merge: orphan transactions dropped", orphans.sum())
    merged = merged[~orphans].drop(columns="_acc")

    # The account table is the authority on ownership; the transaction's own
    # customer_id is redundant and sometimes blank, so prefer the account's.
    merged["customer_id"] = merged["customer_id_acct"].where(
        merged["customer_id_acct"].ne(""), merged["customer_id"])
    merged = merged.drop(columns=["customer_id_acct"])

    merged = merged.merge(cust, on="customer_id", how="left", indicator="_cust", validate="m:1")
    rep.note("merge: rows with no customer record", merged["_cust"].eq("left_only").sum())
    merged = merged.drop(columns="_cust")

    rep.note("merge: final joined rows", len(merged))
    return merged


def run(raw_dir: Path) -> tuple[pd.DataFrame, CleanReport]:
    """Full clean + merge. Returns the joined frame and its audit report."""
    rep = CleanReport()
    raw = load_raw(raw_dir)
    cust = clean_customers(raw["customers"], rep)
    acc = clean_accounts(raw["accounts"], rep)
    tx = clean_transactions(raw["transactions"], rep)
    return merge_all(tx, acc, cust, rep), rep
