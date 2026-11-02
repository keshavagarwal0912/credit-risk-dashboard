"""Anomaly detection: flag accounts whose repayment or bureau pattern is unusual vs their OWN history
or vs their PEER SEGMENT (same loan_type x income_band).

Three layers, each producing a human-readable reason:
  1. Own-history rules  (repayment months 10-12 vs months 1-9)
  2. Peer-segment rules (robust z-score vs the segment median/MAD)
  3. Isolation Forest   (multivariate outlier on repayment + bureau features)
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import PROCESSED, RANDOM_STATE

PEER_KEYS = ["loan_type", "income_band"]
PEER_Z = 3.5          # robust z threshold (≈ 3.5 sigma; Iglewicz-Hoaglin convention)
IF_CONTAMINATION = 0.02
DECISIONS = PROCESSED / "review_decisions.csv"

RULES = {
    "own_dpd_jump": "Repayment DPD jumped: avg DPD in months 10-12 is >=30 days after <=5 days in months 1-9",
    "own_payment_drop": "Payments collapsed: paid <50% of EMI in months 10-12 after paying >=95% in months 1-9",
    "own_consec_missed": "3 or more consecutive missed EMIs",
    "peer_emi_burden": "EMI-to-income far above peers (robust z > 3.5 within loan type x income band)",
    "peer_bureau_dpd": "3+ external lines 30+ DPD, above the peer segment's 99th percentile",
    "peer_bureau_writeoffs": "3+ external written-off lines, above the peer segment's 99th percentile",
    "peer_credit_hunger": "4+ new external credit lines in 12 months, above the peer segment's 99th percentile",
    "peer_utilization": "External utilization >=90% while the peer segment's median is below 50%",
    "isolation_forest": "Multivariate outlier: unusual combination of repayment + bureau features (top 2%)",
}


def robust_z(s, groups):
    med = s.groupby(groups).transform("median")
    mad = (s - med).abs().groupby(groups).transform("median")
    # Floor the MAD so near-constant peer groups don't make every small deviation 'infinite'.
    mad = mad.where(mad > 0, s.groupby(groups).transform("std").fillna(1) / 1.4826 + 1e-9)
    return 0.6745 * (s - med) / mad, med


def detect():
    df = pd.read_parquet(PROCESSED / "accounts.parquet")
    rep = pd.read_parquet(PROCESSED / "repayment.parquet")
    prior = rep[rep.payment_month <= 9].groupby("account_id")[["emi_paid", "emi_due"]].sum()
    df = df.merge((prior.emi_paid / prior.emi_due).rename("pay_ratio_prior"), left_on="account_id", right_index=True)

    groups = [df[k] for k in PEER_KEYS]
    flags = pd.DataFrame(index=df.index)
    flags["own_dpd_jump"] = (df.rep_dpd_last3 >= 30) & (df.rep_dpd_prior <= 5)
    flags["own_payment_drop"] = (df.rep_pay_ratio_last3 < 0.5) & (df.pay_ratio_prior >= 0.95)
    flags["own_consec_missed"] = df.rep_max_consec_missed >= 3

    z = {}
    z["peer_emi_burden"], df["emi_to_income_peer_median"] = robust_z(df.emi_to_income, groups)
    flags["peer_emi_burden"] = z["peer_emi_burden"] > PEER_Z
    # Bureau counts are small bounded integers, so a z-score never fires; use the peer 99th percentile + a floor.
    for name, col, floor in [("peer_bureau_dpd", "bureau_lines_dpd30", 3), ("peer_bureau_writeoffs", "bureau_written_off", 3),
                             ("peer_credit_hunger", "bureau_new_lines_12m", 4)]:
        df[f"{col}_peer_median"] = df[col].groupby(groups).transform("median")
        df[f"{col}_peer_p99"] = df[col].groupby(groups).transform(lambda s: s.quantile(0.99))
        flags[name] = (df[col] >= floor) & (df[col] >= df[f"{col}_peer_p99"])
    util_med = df.bureau_utilization.groupby(groups).transform("median")
    df["bureau_utilization_peer_median"] = util_med
    flags["peer_utilization"] = (df.bureau_utilization >= 0.90) & (util_med < 0.5) & (df.bureau_active >= 2)

    iso_cols = ["rep_avg_dpd", "rep_dpd_trend", "rep_pay_ratio", "rep_pay_ratio_last3", "rep_missed", "rep_late",
                "bureau_max_dpd", "bureau_utilization", "bureau_written_off", "bureau_new_lines_12m", "emi_to_income"]
    X = df[iso_cols].fillna(df[iso_cols].median())
    iso = IsolationForest(n_estimators=300, contamination=IF_CONTAMINATION, random_state=RANDOM_STATE).fit(X)
    iso_score = -iso.score_samples(X)               # higher = more anomalous
    flags["isolation_forest"] = iso.predict(X) == -1

    # Severity: rule count + scaled IF score, so analysts work the queue top-down.
    iso_pct = pd.Series(iso_score).rank(pct=True).values
    severity = flags.sum(axis=1) + iso_pct
    flagged = flags.any(axis=1)

    rows = []
    for i in np.where(flagged)[0]:
        r = df.iloc[i]
        hit = [k for k in RULES if flags.iloc[i][k]]
        reasons = [_reason(k, r, z, i) for k in hit]
        rows.append({"account_id": r.account_id, "customer_id": r.customer_id, "loan_type": r.loan_type,
                     "income_band": r.income_band, "anomaly_score": round(float(severity.iloc[i]), 3),
                     "anomaly_rules": ", ".join(hit), "anomaly_reasons": json.dumps(reasons),
                     "target_default": int(r.target_default)})
    out = pd.DataFrame(rows).sort_values("anomaly_score", ascending=False)
    out.to_parquet(PROCESSED / "anomalies.parquet", index=False)

    summary = {k: int(flags[k].sum()) for k in RULES}
    summary["flagged_total"] = int(flagged.sum())
    summary["flagged_default_rate"] = round(float(df.target_default[flagged].mean()), 4)
    summary["not_flagged_default_rate"] = round(float(df.target_default[~flagged].mean()), 4)
    (PROCESSED / "anomaly_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return out


def _reason(rule, r, z, i):
    if rule == "own_dpd_jump":
        return f"Avg DPD rose from {r.rep_dpd_prior:.0f} (months 1-9) to {r.rep_dpd_last3:.0f} days (months 10-12)."
    if rule == "own_payment_drop":
        return f"Paid {r.pay_ratio_prior:.0%} of EMI in months 1-9 but only {r.rep_pay_ratio_last3:.0%} in months 10-12."
    if rule == "own_consec_missed":
        return f"{int(r.rep_max_consec_missed)} consecutive missed EMIs."
    if rule == "peer_emi_burden":
        return (f"Annual EMI = {r.emi_to_income:.0%} of income vs peer median {r.emi_to_income_peer_median:.0%} "
                f"(z={z[rule].iloc[i]:.1f}).")
    if rule == "peer_bureau_dpd":
        return (f"{int(r.bureau_lines_dpd30)} external lines 30+ DPD vs peer median {r.bureau_lines_dpd30_peer_median:.0f} "
                f"(peer 99th pct {r.bureau_lines_dpd30_peer_p99:.0f}); bureau max DPD {r.bureau_max_dpd:.0f}.")
    if rule == "peer_bureau_writeoffs":
        return (f"{int(r.bureau_written_off)} written-off external lines vs peer median "
                f"{r.bureau_written_off_peer_median:.0f} (peer 99th pct {r.bureau_written_off_peer_p99:.0f}).")
    if rule == "peer_credit_hunger":
        return (f"{int(r.bureau_new_lines_12m)} new external lines in 12 months vs peer median "
                f"{r.bureau_new_lines_12m_peer_median:.0f} (peer 99th pct {r.bureau_new_lines_12m_peer_p99:.0f}).")
    if rule == "peer_utilization":
        return f"External utilization {r.bureau_utilization:.0%} vs peer median {r.bureau_utilization_peer_median:.0%}."
    return "Isolation Forest: this mix of repayment and bureau values is in the most unusual 2% of the portfolio."


def load_decisions():
    if DECISIONS.exists():
        return pd.read_csv(DECISIONS)
    return pd.DataFrame(columns=["account_id", "decision", "note", "reviewer", "decided_at"])


def record_decision(account_id, decision, note="", reviewer="analyst"):
    d = load_decisions()
    d = d[d.account_id != account_id]
    new = pd.DataFrame([{"account_id": account_id, "decision": decision, "note": note, "reviewer": reviewer,
                         "decided_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}])
    d = pd.concat([d, new], ignore_index=True) if len(d) else new
    d.to_csv(DECISIONS, index=False)
    return d


if __name__ == "__main__":
    detect()
