"""Portfolio analytics shared by the dashboard and the LLM tool (one source of truth for every number)."""
import pandas as pd

from src.config import PROCESSED

DIMENSIONS = {
    "loan_type": "Loan type", "employment_type": "Employment type", "region": "Region",
    "income_band": "Income band", "age_band": "Age band", "city_tier": "City tier",
    "risk_band": "Model risk band", "housing_type": "Housing type", "disbursal_channel": "Disbursal channel",
    "loan_purpose": "Loan purpose", "education_level": "Education", "marital_status": "Marital status",
    "gender": "Gender", "orig_year": "Origination year", "bureau_any_written_off": "Prior bureau write-off (0/1)",
    "bureau_thin_file": "No bureau record (0/1)", "collateral_flag": "Secured (0/1)",
    "anomaly_flag": "Flagged by anomaly detection (0/1)",
}
ORDER = {
    "income_band": ["<30k", "30-60k", "60-90k", "90-120k", "120k+"],
    "age_band": ["21-25", "26-35", "36-45", "46-55", "56+"],
    "risk_band": ["Low", "Medium", "High", "Very High"],
}
METRICS = {
    "accounts": "Number of loan accounts",
    "customers": "Number of distinct customers",
    "defaults": "Number of defaulted accounts (target_default=1)",
    "default_rate": "Share of accounts with target_default=1",
    "exposure": "Sum of original loan_amount (USD)",
    "exposure_share": "Share of total exposure in the current filter",
    "defaulted_exposure": "Sum of loan_amount on defaulted accounts (USD)",
    "outstanding": "Sum of outstanding balance after month 12 (USD)",
    "avg_loan_amount": "Mean loan_amount (USD)",
    "avg_income": "Mean customer annual_income (USD)",
    "avg_pd": "Mean model probability of default (out-of-fold LightGBM)",
    "avg_interest_rate": "Mean interest_rate_pct",
    "anomalies": "Number of accounts flagged by anomaly detection",
}


def load_portfolio():
    df = pd.read_parquet(PROCESSED / "accounts.parquet")
    scores = pd.read_parquet(PROCESSED / "scores.parquet")
    df = df.merge(scores, on="account_id", how="left")
    anom = PROCESSED / "anomalies.parquet"
    if anom.exists():
        a = pd.read_parquet(anom)[["account_id", "anomaly_score", "anomaly_reasons", "anomaly_rules"]]
        df = df.merge(a, on="account_id", how="left")
        df["anomaly_flag"] = df.anomaly_score.notna().astype(int)
    else:
        df["anomaly_flag"] = 0
    return df


def apply_filters(df, filters: dict | None):
    for col, values in (filters or {}).items():
        if col not in DIMENSIONS or values in (None, [], ""):
            continue
        values = values if isinstance(values, list) else [values]
        df = df[df[col].astype(str).isin([str(v) for v in values])]
    return df


def summarize(df):
    total_exp = df.loan_amount.sum()
    return {
        "accounts": len(df),
        "customers": df.customer_id.nunique(),
        "defaults": int(df.target_default.sum()),
        "default_rate": df.target_default.mean() if len(df) else float("nan"),
        "exposure": total_exp,
        "defaulted_exposure": df.loan_amount[df.target_default == 1].sum(),
        "outstanding": df.rep_last_balance.sum(),
        "avg_loan_amount": df.loan_amount.mean(),
        "avg_income": df.annual_income.mean(),
        "avg_pd": df.pd_score.mean(),
        "avg_interest_rate": df.interest_rate_pct.mean(),
        "anomalies": int(df.anomaly_flag.sum()),
    }


def segment_table(df, dims: list[str]):
    g = df.groupby([df[d].astype(str) for d in dims], observed=True)
    out = g.agg(accounts=("account_id", "size"), customers=("customer_id", "nunique"),
                defaults=("target_default", "sum"), default_rate=("target_default", "mean"),
                exposure=("loan_amount", "sum"), outstanding=("rep_last_balance", "sum"),
                avg_loan_amount=("loan_amount", "mean"), avg_income=("annual_income", "mean"),
                avg_pd=("pd_score", "mean"), avg_interest_rate=("interest_rate_pct", "mean"),
                anomalies=("anomaly_flag", "sum")).reset_index()
    out["defaulted_exposure"] = g.apply(lambda d: d.loan_amount[d.target_default == 1].sum(),
                                        include_groups=False).values
    out["exposure_share"] = out.exposure / out.exposure.sum()
    if len(dims) == 1 and dims[0] in ORDER:
        out[dims[0]] = pd.Categorical(out[dims[0]], ORDER[dims[0]], ordered=True)
        out = out.sort_values(dims[0])
    return out


def insights(df):
    """Top portfolio insights. Every figure is computed live from the data, nothing is hard-coded."""
    lt = segment_table(df, ["loan_type"]).set_index("loan_type")
    bad_exp = lt.defaulted_exposure / lt.defaulted_exposure.sum()
    inc = segment_table(df, ["income_band"]).set_index("income_band")
    emp = segment_table(df, ["employment_type"]).set_index("employment_type")
    ie = df.pivot_table(index="income_band", columns="employment_type", values="target_default", aggfunc="mean")
    wo = df.pivot_table(index="loan_type", columns="bureau_any_written_off", values="target_default", aggfunc="mean")
    late = df.groupby(pd.cut(df.rep_late, [-1, 0, 2, 12], labels=["0", "1-2", "3+"]), observed=True).target_default
    late_dr, late_n = late.mean(), late.size()
    reg = segment_table(df, ["region"]).set_index("region")
    return [
        {
            "title": "Mortgages are the loss concentration",
            "finding": f"Mortgages default at {lt.default_rate['Mortgage']:.1%} (portfolio {df.target_default.mean():.1%}). "
                       f"They hold {lt.exposure_share['Mortgage']:.0%} of exposure but {bad_exp['Mortgage']:.0%} of defaulted exposure.",
            "so_what": "Mortgage underwriting is the biggest lever on losses. Review LTV/affordability cut-offs and "
                       "put mortgages first in the early-warning queue, even though they are secured.",
        },
        {
            "title": "Income is the strongest single driver",
            "finding": f"Default rate falls from {inc.default_rate['<30k']:.1%} under $30k to {inc.default_rate['120k+']:.1%} "
                       f"above $120k, a {inc.default_rate['<30k'] / inc.default_rate['120k+']:.1f}x gap. "
                       f"It is also the top SHAP feature in the model.",
            "so_what": "Use income-based affordability limits (loan-to-income, EMI-to-income) in approval, "
                       "and verify income for applicants under $60k.",
        },
        {
            "title": "Cross-cut: employment type changes risk inside the same income band",
            "finding": f"In the $30-60k band, Contract workers default at {ie.loc['30-60k', 'Contract']:.1%} and Unemployed "
                       f"at {ie.loc['30-60k', 'Unemployed']:.1%}, vs Salaried {ie.loc['30-60k', 'Salaried']:.1%}. "
                       f"Unemployed overall: {emp.default_rate['Unemployed']:.1%} on {emp.accounts['Unemployed']:,} accounts.",
            "so_what": "Price and set limits on income x employment together, not income alone. Unemployed "
                       "applicants need a documented alternative repayment source.",
        },
        {
            "title": "A prior bureau write-off lifts risk in every product",
            "finding": f"Default rate with vs without a prior write-off: Business Loan {wo.loc['Business Loan', 1]:.1%} vs "
                       f"{wo.loc['Business Loan', 0]:.1%}, Credit Card {wo.loc['Credit Card', 1]:.1%} vs {wo.loc['Credit Card', 0]:.1%}, "
                       f"Mortgage {wo.loc['Mortgage', 1]:.1%} vs {wo.loc['Mortgage', 0]:.1%}.",
            "so_what": "Make prior write-off a hard referral rule (manual underwriting) rather than just a score input.",
        },
        {
            "title": "Early repayment behaviour is an early-warning signal",
            "finding": f"Accounts with 0 late payments in their first 12 months default at {late_dr['0']:.1%}; with 3+ late "
                       f"payments at {late_dr['3+']:.1%} ({late_n['3+']:,} accounts). Region barely matters "
                       f"({reg.default_rate.min():.1%}-{reg.default_rate.max():.1%}).",
            "so_what": "Trigger proactive collections contact after the 2nd late payment. Do not spend effort on "
                       "region-based pricing.",
        },
    ]
