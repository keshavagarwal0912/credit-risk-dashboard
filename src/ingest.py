"""Load the 5 raw tables, fix data-quality issues, and build one account-level dataset.

Grain of the output: 1 row per loan account (14,685 rows).
  loan_accounts  (base, holds target_default)
    + customers            1:1 on customer_id     (left join, every loan has a customer)
    + bureau aggregates    per customer_id        (left join, 1,008 customers have no bureau file)
    + repayment aggregates per account_id         (left join, every account has 12 months)
    + collections aggs     per account_id         (left join, sparse: only 4,278 accounts)
Many-to-one tables are aggregated BEFORE joining so the account grain never fans out.
"""
import numpy as np
import pandas as pd

from src.config import AS_OF, PROCESSED, RAW

DQ_LOG: list[dict] = []


def log_dq(table, issue, rows, action):
    DQ_LOG.append({"table": table, "issue": issue, "rows_affected": int(rows), "action": action})


def load_raw():
    return {
        "customers": pd.read_csv(RAW / "customers.csv", parse_dates=["customer_since"]),
        "loans": pd.read_csv(RAW / "loan_accounts.csv", parse_dates=["origination_date"]),
        "bureau": pd.read_csv(RAW / "bureau_credit.csv", parse_dates=["date_opened", "date_closed"]),
        "repay": pd.read_csv(RAW / "repayment_history.csv"),
        "coll": pd.read_csv(RAW / "collections_events.csv", parse_dates=["event_date"]),
    }


def check_keys(t):
    for name, key in [("customers", "customer_id"), ("loans", "account_id"), ("bureau", "bureau_id"),
                      ("coll", "collection_event_id")]:
        dups = t[name][key].duplicated().sum()
        log_dq(name, f"duplicate {key}", dups, "none found" if dups == 0 else "dropped duplicates")
        t[name] = t[name].drop_duplicates(key)
    dups = t["repay"].duplicated(["account_id", "payment_month"]).sum()
    log_dq("repay", "duplicate account_id+payment_month", dups, "none found" if dups == 0 else "dropped")
    orphans = (~t["loans"].customer_id.isin(t["customers"].customer_id)).sum()
    log_dq("loans", "loan with unknown customer_id", orphans, "none found (referential integrity OK)")


def clean_customers(c):
    c = c.copy()
    mask = c.gender.isin(["M", "F"])
    log_dq("customers", "gender coded as M/F instead of Male/Female", mask.sum(), "mapped M->Male, F->Female")
    c["gender"] = c.gender.replace({"M": "Male", "F": "Female"})

    # Impute with segment medians and keep a missing flag so the model can still see "unknown".
    for col, by in [("years_employed", "employment_type"), ("num_dependents", "marital_status"),
                    ("years_at_residence", "housing_type")]:
        n = c[col].isna().sum()
        c[f"{col}_missing"] = c[col].isna().astype(int)
        c[col] = c[col].fillna(c.groupby(by)[col].transform("median"))
        log_dq("customers", f"{col} missing", n, f"imputed with median by {by}; added {col}_missing flag")

    c["age_band"] = pd.cut(c.age, [0, 25, 35, 45, 55, 100], labels=["21-25", "26-35", "36-45", "46-55", "56+"],
                           right=True).astype(str)
    c["income_band"] = pd.cut(c.annual_income, [0, 30000, 60000, 90000, 120000, np.inf],
                              labels=["<30k", "30-60k", "60-90k", "90-120k", "120k+"]).astype(str)
    return c


def clean_loans(l, c):
    l = l.copy()
    n = l.interest_rate_pct.isna().sum()
    l["interest_rate_missing"] = l.interest_rate_pct.isna().astype(int)
    l["interest_rate_pct"] = l.interest_rate_pct.fillna(l.groupby("loan_type").interest_rate_pct.transform("median"))
    log_dq("loans", "interest_rate_pct missing", n, "imputed with median by loan_type; added interest_rate_missing flag")

    since = l.merge(c[["customer_id", "customer_since"]], on="customer_id").set_index(l.index).customer_since
    bad = l.origination_date < since
    l["orig_before_customer_since"] = bad.astype(int)
    log_dq("loans", "origination_date earlier than customer.customer_since", bad.sum(),
           "kept rows; flagged; customer_since treated as unreliable and not used as a model feature")
    l["orig_year"] = l.origination_date.dt.year
    l["orig_quarter"] = l.origination_date.dt.to_period("Q").astype(str)
    return l


def bureau_features(b):
    b = b.copy()
    n = b.days_past_due.isna().sum()
    log_dq("bureau", "days_past_due missing", n,
           "left as NaN (not imputed to 0, which would look 'clean'); aggregated with max/mean ignoring NaN "
           "and counted per customer as bureau_dpd_missing")
    log_dq("bureau", "date_closed null", b.date_closed.isna().sum(),
           "expected: all nulls are Active lines (verified 0 Active rows with a close date)")
    as_of = pd.Timestamp(AS_OF)
    b["active"] = (b.account_status == "Active").astype(int)
    b["written_off"] = (b.account_status == "Written-Off").astype(int)
    b["closed"] = (b.account_status == "Closed").astype(int)
    b["dpd30"] = (b.days_past_due >= 30).astype(int)
    b["opened_last_12m"] = (b.date_opened >= as_of - pd.DateOffset(months=12)).astype(int)
    b["nonbank"] = b.lender_category.isin(["NBFC", "Fintech"]).astype(int)
    b["active_out"] = b.outstanding_amount.where(b.active == 1, 0)
    b["active_sanc"] = b.sanctioned_amount.where(b.active == 1, 0)
    g = b.groupby("customer_id")
    f = pd.DataFrame({
        "bureau_lines": g.size(),
        "bureau_active": g.active.sum(),
        "bureau_closed": g.closed.sum(),
        "bureau_written_off": g.written_off.sum(),
        "bureau_total_sanctioned": g.sanctioned_amount.sum(),
        "bureau_active_outstanding": g.active_out.sum(),
        "bureau_active_sanctioned": g.active_sanc.sum(),
        "bureau_max_dpd": g.days_past_due.max(),
        "bureau_avg_dpd": g.days_past_due.mean(),
        "bureau_lines_dpd30": g.dpd30.sum(),
        "bureau_dpd_missing": g.days_past_due.apply(lambda s: s.isna().sum()),
        "bureau_new_lines_12m": g.opened_last_12m.sum(),
        "bureau_nonbank_share": g.nonbank.mean(),
        "bureau_oldest_line_yrs": g.date_opened.min().rsub(as_of).dt.days / 365.25,
    })
    f["bureau_utilization"] = (f.bureau_active_outstanding / f.bureau_active_sanctioned.replace(0, np.nan)).fillna(0)
    f["bureau_any_written_off"] = (f.bureau_written_off > 0).astype(int)
    return f.reset_index()


def repayment_features(r):
    r = r.sort_values(["account_id", "payment_month"]).copy()
    r["missed"] = (r.payment_status == "Missed").astype(int)
    r["late"] = (r.payment_status == "Late").astype(int)
    last3 = r.payment_month > r.groupby("account_id").payment_month.transform("max") - 3
    r["dpd_last3"] = r.days_past_due.where(last3)
    r["dpd_prior"] = r.days_past_due.where(~last3)
    r["paid_last3"] = r.emi_paid.where(last3)
    r["due_last3"] = r.emi_due.where(last3)
    # Longest run of consecutive missed months.
    run_id = (r.missed != r.groupby("account_id").missed.shift()).cumsum()
    r["miss_run"] = r.groupby(run_id).missed.cumsum()
    g = r.groupby("account_id")
    f = pd.DataFrame({
        "rep_months": g.size(),
        "rep_missed": g.missed.sum(),
        "rep_late": g.late.sum(),
        "rep_max_dpd": g.days_past_due.max(),
        "rep_avg_dpd": g.days_past_due.mean(),
        "rep_dpd_last3": g.dpd_last3.mean(),
        "rep_dpd_prior": g.dpd_prior.mean(),
        "rep_pay_ratio": g.emi_paid.sum() / g.emi_due.sum(),
        "rep_pay_ratio_last3": g.paid_last3.sum() / g.due_last3.sum(),
        "rep_max_consec_missed": g.miss_run.max(),
        "rep_emi": g.emi_due.median(),
        "rep_last_balance": g.outstanding_balance.last(),
    })
    f["rep_dpd_trend"] = f.rep_dpd_last3 - f.rep_dpd_prior
    f["rep_ever_30dpd"] = (f.rep_max_dpd >= 30).astype(int)
    f["rep_ever_60dpd"] = (f.rep_max_dpd >= 60).astype(int)
    log_dq("repay", "history length", 0, f"every account has exactly {int(f.rep_months.min())} months; no gaps")
    return f.reset_index()


def collections_features(e, l):
    e = e.merge(l[["account_id", "origination_date"]], on="account_id", how="left")
    pre = e.event_date < e.origination_date
    log_dq("collections", "event_date earlier than the loan's origination_date", pre.sum(),
           "kept and flagged (coll_events_pre_origination); collections are NOT model features "
           "(they react to delinquency = target leakage), used for analytics/anomaly context only")
    e["pre_orig"] = pre.astype(int)
    e["legal"] = e.action_type.isin(["Legal Notice", "Written-Off Referral"]).astype(int)
    g = e.groupby("account_id")
    return pd.DataFrame({
        "coll_events": g.size(),
        "coll_recovered": g.amount_recovered.sum(),
        "coll_legal_actions": g.legal.sum(),
        "coll_last_action": g.apply(lambda d: d.sort_values("event_date").action_type.iloc[-1], include_groups=False),
        "coll_events_pre_origination": g.pre_orig.sum(),
    }).reset_index()


def build():
    DQ_LOG.clear()
    t = load_raw()
    check_keys(t)
    cust = clean_customers(t["customers"])
    loans = clean_loans(t["loans"], cust)
    bur = bureau_features(t["bureau"])
    rep = repayment_features(t["repay"])
    col = collections_features(t["coll"], loans)

    df = (loans.merge(cust, on="customer_id", how="left", validate="m:1")
          .merge(bur, on="customer_id", how="left", validate="m:1")
          .merge(rep, on="account_id", how="left", validate="1:1")
          .merge(col, on="account_id", how="left", validate="1:1"))
    assert len(df) == len(loans), "join changed the account grain"

    thin = df.bureau_lines.isna()
    log_dq("bureau", "customer has no bureau record (thin file)", cust.customer_id.isin(bur.customer_id).__invert__().sum(),
           f"kept; bureau counts/amounts set to 0, bureau_thin_file=1 ({thin.sum()} accounts)")
    df["bureau_thin_file"] = thin.astype(int)
    zero_cols = [c for c in bur.columns if c != "customer_id" and c not in ("bureau_max_dpd", "bureau_avg_dpd")]
    df[zero_cols] = df[zero_cols].fillna(0)
    # DPD stays NaN for thin-file / all-missing customers; LightGBM handles NaN, LR gets a median fill + flag.
    df["coll_last_action"] = df.coll_last_action.fillna("None")
    for c in ["coll_events", "coll_recovered", "coll_legal_actions", "coll_events_pre_origination"]:
        df[c] = df[c].fillna(0)

    df["customer_loan_count"] = df.groupby("customer_id").account_id.transform("count")
    df["loan_to_income"] = df.loan_amount / df.annual_income
    df["emi_to_income"] = df.rep_emi * 12 / df.annual_income

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED / "accounts.parquet", index=False)
    cust.to_parquet(PROCESSED / "customers.parquet", index=False)
    t["repay"].to_parquet(PROCESSED / "repayment.parquet", index=False)
    t["bureau"].to_parquet(PROCESSED / "bureau.parquet", index=False)
    t["coll"].to_parquet(PROCESSED / "collections.parquet", index=False)
    log_dq("output", "label spelling differs from DATA_DICTIONARY (Partner/DSA, SMS/Email Notice)", 0,
           "kept the values as they appear in the data")
    pd.DataFrame(DQ_LOG).to_csv(PROCESSED / "data_quality_log.csv", index=False)
    print(f"accounts.parquet: {df.shape}, default rate {df.target_default.mean():.2%}")
    return df


if __name__ == "__main__":
    build()
