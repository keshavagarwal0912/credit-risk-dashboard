"""Default model: logistic regression baseline vs LightGBM, with AUC / KS / Gini and SHAP.

Scoring assumption: a behavioural scorecard evaluated at month 12 of the loan, so 12 months of
repayment history plus the bureau snapshot are known at score time. Collections events are
excluded (they are triggered by delinquency -> leakage).
"""
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import ARTIFACTS, PROCESSED, RANDOM_STATE, RISK_BANDS

NUMERIC = [
    # application / customer
    "age", "annual_income", "years_employed", "num_dependents", "years_at_residence",
    "loan_amount", "interest_rate_pct", "tenure_months", "collateral_flag", "loan_to_income", "emi_to_income",
    "customer_loan_count", "years_employed_missing", "interest_rate_missing",
    # bureau (engineered)
    "bureau_lines", "bureau_active", "bureau_written_off", "bureau_any_written_off", "bureau_utilization",
    "bureau_max_dpd", "bureau_avg_dpd", "bureau_lines_dpd30", "bureau_dpd_missing", "bureau_new_lines_12m",
    "bureau_nonbank_share", "bureau_oldest_line_yrs", "bureau_active_outstanding", "bureau_thin_file",
    # repayment (engineered)
    "rep_missed", "rep_late", "rep_max_dpd", "rep_avg_dpd", "rep_dpd_last3", "rep_dpd_trend", "rep_pay_ratio",
    "rep_pay_ratio_last3", "rep_max_consec_missed", "rep_ever_30dpd", "rep_ever_60dpd",
]
CATEGORICAL = ["loan_type", "employment_type", "region", "city_tier", "housing_type", "education_level",
               "marital_status", "disbursal_channel", "loan_purpose"]
FEATURES = NUMERIC + CATEGORICAL


def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def metrics(y, p):
    auc = roc_auc_score(y, p)
    return {"AUC": round(auc, 4), "Gini": round(2 * auc - 1, 4), "KS": round(ks_stat(y, p), 4)}


def make_lr():
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]),
         NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=2000, C=0.5))])


def make_gbm():
    # Shallow, heavily regularised trees: picked by 5-fold CV on the train split (deeper trees overfit,
    # e.g. num_leaves=15 gave train AUC 0.89 vs test 0.66).
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=3, min_child_samples=300,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
                              random_state=RANDOM_STATE, verbose=-1)


def to_gbm_frame(X):
    X = X.copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    return X


def band(p):
    for lo, hi, name in RISK_BANDS:
        if lo <= p < hi:
            return name
    return RISK_BANDS[-1][2]


def train():
    df = pd.read_parquet(PROCESSED / "accounts.parquet")
    X, y = df[FEATURES], df.target_default
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)

    lr = make_lr().fit(X_tr, y_tr)
    gbm = make_gbm().fit(to_gbm_frame(X_tr), y_tr)
    p_lr, p_gbm = lr.predict_proba(X_te)[:, 1], gbm.predict_proba(to_gbm_frame(X_te))[:, 1]
    results = {
        "logistic_regression": {"train": metrics(y_tr, lr.predict_proba(X_tr)[:, 1]), "test": metrics(y_te, p_lr)},
        "lightgbm": {"train": metrics(y_tr, gbm.predict_proba(to_gbm_frame(X_tr))[:, 1]), "test": metrics(y_te, p_gbm)},
        "test_rows": int(len(y_te)), "train_rows": int(len(y_tr)), "test_default_rate": round(float(y_te.mean()), 4),
    }

    # Out-of-fold PD for every account, so dashboard scores are never in-sample.
    oof = np.zeros(len(df))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE).split(X, y):
        m = make_gbm().fit(to_gbm_frame(X.iloc[tr]), y.iloc[tr])
        oof[te] = m.predict_proba(to_gbm_frame(X.iloc[te]))[:, 1]
    results["lightgbm"]["oof_5fold"] = metrics(y, oof)

    # Final model on all data -> SHAP for global importance and per-account drivers.
    final = make_gbm().fit(to_gbm_frame(X), y)
    sv = shap.TreeExplainer(final).shap_values(to_gbm_frame(X))
    sv = sv[1] if isinstance(sv, list) else sv
    importance = (pd.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(sv).mean(0)})
                  .sort_values("mean_abs_shap", ascending=False))
    lr_coef = pd.DataFrame({"feature": lr.named_steps["pre"].get_feature_names_out(),
                            "coef": lr.named_steps["lr"].coef_[0]}).sort_values("coef", key=abs, ascending=False)

    top = np.argsort(-np.abs(sv), axis=1)[:, :5]
    drivers = [
        [{"feature": FEATURES[j], "value": _fmt(X.iloc[i, j]), "shap": round(float(sv[i, j]), 4)} for j in top[i]]
        for i in range(len(df))
    ]
    scores = pd.DataFrame({"account_id": df.account_id, "pd_score": oof.round(4),
                           "risk_band": [band(p) for p in oof], "drivers": [json.dumps(d) for d in drivers]})

    # Calibration-style check per band (used in the dashboard + monitoring baseline).
    band_tbl = scores.assign(y=y.values).groupby("risk_band").agg(accounts=("y", "size"), actual_default=("y", "mean"),
                                                                    avg_pd=("pd_score", "mean")).round(4)
    results["bands"] = band_tbl.reset_index().to_dict("records")

    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump({"lr": lr, "gbm": final}, ARTIFACTS / "models.joblib")
    scores.to_parquet(PROCESSED / "scores.parquet", index=False)
    importance.to_csv(ARTIFACTS / "shap_importance.csv", index=False)
    lr_coef.to_csv(ARTIFACTS / "lr_coefficients.csv", index=False)
    # Baseline score distribution for PSI monitoring after deployment.
    np.save(ARTIFACTS / "baseline_scores.npy", oof)
    (ARTIFACTS / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(importance.head(15).to_string(index=False))
    return results


def _fmt(v):
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else round(float(v), 2)
    return str(v)


def psi(expected, actual, bins=10):
    """Population Stability Index between a baseline and a new score distribution."""
    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    e = np.histogram(expected, cuts)[0] / len(expected)
    a = np.histogram(np.clip(actual, cuts[0], cuts[-1]), cuts)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


if __name__ == "__main__":
    train()
