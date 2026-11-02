"""Credit Risk Portfolio Monitor - Streamlit dashboard entry point: streamlit run app/app.py"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from src import anomaly, ask  # noqa: E402
from src.analytics import DIMENSIONS, ORDER, apply_filters, insights, load_portfolio, segment_table, summarize  # noqa: E402
from src.config import ARTIFACTS, PROCESSED  # noqa: E402

st.set_page_config(page_title="Credit Risk Portfolio Monitor", page_icon="📉", layout="wide")
BAND_COLORS = {"Low": "#2e7d32", "Medium": "#f9a825", "High": "#ef6c00", "Very High": "#c62828"}


@st.cache_resource(show_spinner="First run: building dataset, model and anomaly flags (~1 min)...")
def ensure_built():
    if not (PROCESSED / "anomalies.parquet").exists() or not (ARTIFACTS / "metrics.json").exists():
        from src import ingest, model
        ingest.build(); model.train(); anomaly.detect()
    return True


@st.cache_data
def data():
    df = load_portfolio()
    rep = pd.read_parquet(PROCESSED / "repayment.parquet")
    bur = pd.read_parquet(PROCESSED / "bureau.parquet")
    col = pd.read_parquet(PROCESSED / "collections.parquet")
    anom = pd.read_parquet(PROCESSED / "anomalies.parquet")
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    shap_imp = pd.read_csv(ARTIFACTS / "shap_importance.csv")
    lr_coef = pd.read_csv(ARTIFACTS / "lr_coefficients.csv")
    dq = pd.read_csv(PROCESSED / "data_quality_log.csv")
    anom_sum = json.loads((PROCESSED / "anomaly_summary.json").read_text())
    return df, rep, bur, col, anom, metrics, shap_imp, lr_coef, dq, anom_sum


def api_key():
    """GROQ_API_KEY from Streamlit secrets, falling back to the environment / .env."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")


def money(v):
    return f"${v/1e6:,.1f}M" if abs(v) >= 1e6 else f"${v:,.0f}"


def sort_dim(t, dim):
    if dim in ORDER:
        t[dim] = pd.Categorical(t[dim].astype(str), ORDER[dim], ordered=True)
        return t.sort_values(dim)
    return t.sort_values("default_rate", ascending=False)


ensure_built()
df, rep, bur, col, anom, metrics, shap_imp, lr_coef, dq, anom_sum = data()

# ---------------- Sidebar filters (apply to the Portfolio view) ----------------
st.sidebar.title("📉 Portfolio filters")
FILTER_DIMS = ["loan_type", "employment_type", "region", "income_band", "age_band", "risk_band", "city_tier"]
filters = {}
for d in FILTER_DIMS:
    opts = ORDER.get(d) or sorted(df[d].astype(str).unique())
    sel = st.sidebar.multiselect(DIMENSIONS[d], opts, key=f"f_{d}")
    if sel:
        filters[d] = sel
fdf = apply_filters(df, filters)
st.sidebar.caption(f"{len(fdf):,} of {len(df):,} accounts selected")
st.sidebar.divider()
st.sidebar.caption("Synthetic data: 10,000 customers, 14,685 loans. Risk score = out-of-fold LightGBM PD.")

st.title("Credit Risk Portfolio Monitor")
tab_port, tab_acct, tab_ask, tab_anom, tab_model = st.tabs(
    ["📊 Portfolio", "🔎 Account view", "💬 Ask the Portfolio (AI)", "🚩 Anomaly review (AI)", "🧠 Model & data quality"])

# ================= PORTFOLIO =================
with tab_port:
    if fdf.empty:
        st.warning("No accounts match the filters; showing the full portfolio.")
        fdf = df
    s = summarize(fdf)
    k = st.columns(6)
    k[0].metric("Accounts", f"{s['accounts']:,}")
    k[1].metric("Exposure", money(s["exposure"]))
    k[2].metric("Default rate", f"{s['default_rate']:.1%}", f"{s['default_rate'] - df.target_default.mean():+.1%} vs portfolio",
                delta_color="inverse")
    k[3].metric("Defaulted exposure", money(s["defaulted_exposure"]))
    k[4].metric("Avg model PD", f"{s['avg_pd']:.1%}")
    k[5].metric("Anomaly flags", f"{s['anomalies']:,}")

    st.subheader("Risk segments")
    c1, c2 = st.columns([3, 2])
    bt = segment_table(fdf, ["risk_band"])
    with c1:
        fig = go.Figure()
        fig.add_bar(x=bt.risk_band.astype(str), y=bt.exposure, name="Exposure", marker_color=[BAND_COLORS[b] for b in bt.risk_band.astype(str)],
                    text=[money(v) for v in bt.exposure], textposition="outside")
        fig.add_scatter(x=bt.risk_band.astype(str), y=bt.default_rate, name="Actual default rate", yaxis="y2", mode="lines+markers+text",
                        text=[f"{v:.0%}" for v in bt.default_rate], textposition="top center", line_color="#333")
        fig.update_layout(height=340, yaxis_title="Exposure", yaxis2=dict(overlaying="y", side="right", tickformat=".0%", title="Default rate"),
                          legend=dict(orientation="h", y=1.15), margin=dict(t=30, b=10))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.dataframe(bt[["risk_band", "accounts", "default_rate", "avg_pd", "exposure", "exposure_share"]].style.format(
            {"default_rate": "{:.1%}", "avg_pd": "{:.1%}", "exposure": "${:,.0f}", "exposure_share": "{:.1%}", "accounts": "{:,}"}),
            hide_index=True, width="stretch")
        st.caption("Bands on model PD: Low <10%, Medium 10-25%, High 25-50%, Very High ≥50%. "
                   "Actual default rate rising across bands shows the score ranks risk correctly.")

    st.subheader("Risk across dimensions")
    c1, c2 = st.columns([1, 3])
    with c1:
        dim = st.radio("Dimension", ["loan_type", "income_band", "employment_type", "age_band", "region", "city_tier",
                                     "disbursal_channel", "loan_purpose", "housing_type"], format_func=DIMENSIONS.get)
    with c2:
        t = sort_dim(segment_table(fdf, [dim]), dim)
        fig = go.Figure()
        fig.add_bar(x=t[dim].astype(str), y=t.exposure, name="Exposure", marker_color="#90a4ae")
        fig.add_scatter(x=t[dim].astype(str), y=t.default_rate, name="Default rate", yaxis="y2", mode="lines+markers", line_color="#c62828")
        fig.add_scatter(x=t[dim].astype(str), y=t.avg_pd, name="Avg model PD", yaxis="y2", mode="lines+markers", line=dict(dash="dot", color="#1565c0"))
        fig.update_layout(height=360, yaxis_title="Exposure", yaxis2=dict(overlaying="y", side="right", tickformat=".0%"),
                          legend=dict(orientation="h", y=1.12), margin=dict(t=30, b=10))
        st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Cross-dimensional cut: default rate heatmap**")
        a, b = st.columns(2)
        rdim = a.selectbox("Rows", list(DIMENSIONS)[:7], index=3, format_func=DIMENSIONS.get)
        cdim = b.selectbox("Columns", list(DIMENSIONS)[:7], index=1, format_func=DIMENSIONS.get)
        if rdim != cdim:
            hm = fdf.pivot_table(index=rdim, columns=cdim, values="target_default", aggfunc="mean", observed=True)
            cnt = fdf.pivot_table(index=rdim, columns=cdim, values="target_default", aggfunc="size", observed=True)
            if rdim in ORDER:
                hm, cnt = hm.reindex([v for v in ORDER[rdim] if v in hm.index]), cnt.reindex([v for v in ORDER[rdim] if v in cnt.index])
            if cdim in ORDER:
                hm, cnt = hm[[v for v in ORDER[cdim] if v in hm.columns]], cnt[[v for v in ORDER[cdim] if v in cnt.columns]]
            fig = px.imshow(hm, text_auto=".0%", color_continuous_scale="Reds", aspect="auto")
            fig.update_traces(customdata=cnt.values, hovertemplate="%{y} × %{x}<br>default rate %{z:.1%}<br>n=%{customdata}<extra></extra>")
            fig.update_layout(height=360, margin=dict(t=10, b=10), coloraxis_colorbar_tickformat=".0%")
            st.plotly_chart(fig, width="stretch")
            st.caption("Hover a cell for its account count; treat cells with small n with caution.")
    with c2:
        st.markdown("**Trend by origination quarter**")
        tr = fdf.groupby("orig_quarter").agg(accounts=("account_id", "size"), default_rate=("target_default", "mean"),
                                             exposure=("loan_amount", "sum"), avg_pd=("pd_score", "mean")).reset_index()
        fig = go.Figure()
        fig.add_bar(x=tr.orig_quarter, y=tr.exposure, name="Exposure originated", marker_color="#b0bec5")
        fig.add_scatter(x=tr.orig_quarter, y=tr.default_rate, name="Default rate", yaxis="y2", line_color="#c62828")
        fig.add_scatter(x=tr.orig_quarter, y=tr.avg_pd, name="Avg model PD", yaxis="y2", line=dict(dash="dot", color="#1565c0"))
        fig.update_layout(height=410, yaxis2=dict(overlaying="y", side="right", tickformat=".0%"),
                          legend=dict(orientation="h", y=1.12), margin=dict(t=30, b=10))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Drill-down: segment → accounts")
    a, b, c = st.columns(3)
    ddim = a.selectbox("Segment by", ["loan_type", "income_band", "employment_type", "region", "risk_band", "age_band"],
                       format_func=DIMENSIONS.get, key="ddim")
    dval = b.selectbox("Segment", ORDER.get(ddim) or sorted(fdf[ddim].astype(str).unique()), key="dval")
    sort_col = c.selectbox("Sort accounts by", ["pd_score", "loan_amount", "rep_max_dpd", "anomaly_score"])
    seg = fdf[fdf[ddim].astype(str) == dval]
    ss = summarize(seg) if len(seg) else None
    if ss:
        st.caption(f"{dval}: {ss['accounts']:,} accounts · default rate {ss['default_rate']:.1%} · exposure {money(ss['exposure'])} · "
                   f"avg PD {ss['avg_pd']:.1%} · {ss['anomalies']} anomaly flags")
        cols = ["account_id", "customer_id", "loan_type", "loan_amount", "pd_score", "risk_band", "rep_max_dpd", "rep_missed",
                "bureau_written_off", "anomaly_flag", "target_default"]
        st.dataframe(seg.sort_values(sort_col, ascending=False)[cols].head(200).style.format(
            {"pd_score": "{:.1%}", "loan_amount": "${:,.0f}"}), hide_index=True, width="stretch", height=280)
        st.caption("Top 200 shown. Open any account in the 🔎 Account view tab.")

    with st.expander("💡 Portfolio insights (computed live from the full portfolio)", expanded=True):
        for i, ins in enumerate(insights(df), 1):
            st.markdown(f"**{i}. {ins['title']}**  \n{ins['finding']}  \n➡️ *So what:* {ins['so_what']}")

# ================= ACCOUNT VIEW =================
with tab_acct:
    ids = df.sort_values("pd_score", ascending=False).account_id.tolist()
    default_id = st.session_state.get("acct", ids[0])
    acc_id = st.selectbox("Account (sorted by risk score, type to search)", ids, index=ids.index(default_id) if default_id in ids else 0)
    r = df[df.account_id == acc_id].iloc[0]
    k = st.columns(5)
    k[0].metric("Risk score (PD)", f"{r.pd_score:.1%}")
    k[1].metric("Risk band", r.risk_band)
    k[2].metric("Loan", f"{r.loan_type}", money(r.loan_amount), delta_color="off")
    k[3].metric("Customer", f"{r.customer_id}", f"{r.employment_type}, {money(r.annual_income)}/yr", delta_color="off")
    k[4].metric("Actual outcome", "Defaulted" if r.target_default else "Performing")
    if r.anomaly_flag:
        st.error("🚩 Flagged by anomaly detection: " + " ".join(json.loads(r.anomaly_reasons)))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Key drivers of this score (SHAP)**")
        drv = pd.DataFrame(json.loads(r.drivers))
        drv["label"] = drv.feature + " = " + drv.value.astype(str)
        drv = drv.iloc[::-1]
        fig = go.Figure(go.Bar(x=drv.shap, y=drv.label, orientation="h",
                               marker_color=["#c62828" if v > 0 else "#2e7d32" for v in drv.shap]))
        fig.update_layout(height=300, margin=dict(t=10, b=10), xaxis_title="Contribution to log-odds of default (red raises risk)")
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("**Repayment history (12 months)**")
        h = rep[rep.account_id == acc_id].sort_values("payment_month")
        fig = go.Figure()
        fig.add_bar(x=h.payment_month, y=h.emi_due, name="EMI due", marker_color="#cfd8dc")
        fig.add_bar(x=h.payment_month, y=h.emi_paid, name="EMI paid", marker_color="#1565c0")
        fig.add_scatter(x=h.payment_month, y=h.days_past_due, name="Days past due", yaxis="y2", line_color="#c62828")
        fig.update_layout(height=300, barmode="overlay", margin=dict(t=10, b=10), xaxis_title="Month since origination",
                          yaxis2=dict(overlaying="y", side="right", title="DPD"), legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, width="stretch")

    c1, c2, c3 = st.columns([2, 2, 1.4])
    with c1:
        st.markdown("**Customer & loan profile**")
        st.dataframe(pd.DataFrame({"field": ["Age", "Income band", "Region / tier", "Years employed", "Loan purpose", "Interest rate",
                                             "Tenure", "Originated", "Loan-to-income", "EMI-to-income", "Other loans at bank"],
                                   "value": [r.age, r.income_band, f"{r.region} / {r.city_tier}", f"{r.years_employed:.1f}", r.loan_purpose,
                                             f"{r.interest_rate_pct:.2f}%", f"{r.tenure_months} m", str(r.origination_date.date()),
                                             f"{r.loan_to_income:.2f}", f"{r.emi_to_income:.0%}", int(r.customer_loan_count) - 1]}).astype(str),
                     hide_index=True, width="stretch")
    with c2:
        st.markdown("**External bureau lines**")
        bl = bur[bur.customer_id == r.customer_id][["credit_type", "lender_category", "sanctioned_amount", "outstanding_amount",
                                                    "account_status", "days_past_due"]]
        if len(bl):
            st.dataframe(bl, hide_index=True, width="stretch")
        else:
            st.info("No bureau record (thin file).")
    with c3:
        st.markdown("**Collections events**")
        cl = col[col.account_id == acc_id].sort_values("event_date")[["event_date", "action_type", "outcome", "amount_recovered"]]
        if len(cl):
            st.dataframe(cl, hide_index=True, width="stretch")
        else:
            st.info("None.")

# ================= ASK THE PORTFOLIO =================
with tab_ask:
    st.subheader("💬 Ask the Portfolio")
    with st.expander("How this works · trigger · false positive / negative · human-in-the-loop", expanded=False):
        st.markdown("""
- **How it works:** your question goes to the LLM (`openai/gpt-oss-120b` on Groq) together with 4 tool definitions. The LLM can't see the data; it can only call
  `query_portfolio`, `list_dimension_values`, `get_account` and `get_model_info`, which compute figures with pandas from the same tables
  as this dashboard. **Data sent to the LLM:** the question + aggregated tool results (no raw tables, no PII).
- **Trigger:** an analyst submits a question.
- **Grounding check:** every number in the answer is matched against the tool outputs. Unmatched numbers are shown as ⚠️ UNVERIFIED.
- **False positive:** a confident answer to the wrong question, e.g. the tool was called with the wrong filter or metric. The
  *Source* line and the tool-call panel below make this visible.
- **False negative:** a question the data *can* answer gets declined, or is refused because of an unusual label spelling.
- **Human-in-the-loop:** answers are decision support, not decisions. The analyst checks the Source / tool trace (and the ⚠️ badge)
  before reusing a figure. Out-of-scope questions (profit, forecasts, names) are declined by design.""")

    key = api_key()
    examples = ["Which loan type has the highest default rate?",
                "What share of exposure is in the Very High risk band?",
                "Compare default rates of Unemployed vs Salaried customers earning 30-60k",
                "Why is account ACC202884 flagged?",
                "What is the AUC of the model?",
                "What will the default rate be next year?"]
    if not key:
        st.warning("No `GROQ_API_KEY` set, so the LLM is disabled. You can still query the same grounded engine manually below.")
    else:
        st.caption("LLM: Groq · openai/gpt-oss-120b")
        cols = st.columns(3)
        for i, ex in enumerate(examples):
            if cols[i % 3].button(ex, key=f"ex{i}", width="stretch"):
                st.session_state["question"] = ex
        q = st.chat_input("Ask a question about the portfolio...")
        q = q or st.session_state.pop("question", None)
        if q:
            st.session_state.setdefault("history", [])
            with st.spinner("Querying the data..."):
                try:
                    res = ask.ask(q, df, key)
                except Exception as e:  # surface API problems instead of crashing the app
                    res = {"answer": f"LLM call failed: {type(e).__name__}: {e}", "trace": [], "unverified": []}
            st.session_state["history"].insert(0, (q, res))
        for q, res in st.session_state.get("history", []):
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.markdown(res["answer"])
                if res["unverified"]:
                    st.warning(f"⚠️ UNVERIFIED figure(s) not found in any tool result: {res['unverified']}. Do not reuse without checking.")
                elif res["trace"]:
                    st.success("✅ Every figure in this answer matches a tool result.", icon="✅")
                if res["trace"]:
                    with st.expander(f"Where the numbers came from ({len(res['trace'])} tool call(s))"):
                        for t in res["trace"]:
                            st.code(f"{t['tool']}({json.dumps(t['args'])})", language="python")
                            out = t["result"]
                            if "rows" in out:
                                st.dataframe(pd.DataFrame(out["rows"]), hide_index=True)
                                st.caption(f"based on {out['based_on_accounts']} accounts, filters={out['filters']}")
                            else:
                                st.json(out, expanded=False)

    with st.expander("Manual query (same engine, no LLM)", expanded=not key):
        a, b, c = st.columns(3)
        mets = a.multiselect("Metrics", list(ask.METRICS), default=["accounts", "default_rate", "exposure"])
        gb = b.multiselect("Group by (max 2)", list(DIMENSIONS), default=["loan_type"], max_selections=2)
        fd = c.selectbox("Filter dimension", ["(none)"] + list(DIMENSIONS))
        fv = c.multiselect("Filter values", sorted(df[fd].astype(str).unique()) if fd != "(none)" else [])
        out = ask.query_portfolio(df, mets, gb, {fd: fv} if fd != "(none)" and fv else None)
        st.dataframe(pd.DataFrame(out["rows"]) if "rows" in out else pd.DataFrame([out.get("result", out)]), hide_index=True)

# ================= ANOMALY REVIEW =================
with tab_anom:
    st.subheader("🚩 Anomaly review queue")
    with st.expander("How this works · trigger · false positive / negative · human-in-the-loop", expanded=False):
        st.markdown(f"""
- **Trigger:** an account is flagged if **any** of these fire (recomputed on every data refresh):
  - *Own history* (months 10-12 vs months 1-9): DPD jump to ≥30 days from ≤5; payments fall below 50% of EMI after ≥95%; 3+ consecutive missed EMIs.
  - *Peer segment* (same loan type × income band): EMI-to-income robust z-score > 3.5; bureau 30+DPD lines, written-off lines or new
    lines in 12 months ≥ the peer 99th percentile (with a minimum floor); external utilization ≥90% where the peer median is <50%.
  - *Isolation Forest* on 11 repayment and bureau features (top 2% most unusual).
- **Queue order:** severity = number of rules hit + the Isolation Forest percentile.
- **Result:** {anom_sum['flagged_total']:,} accounts flagged. Their historical default rate is {anom_sum['flagged_default_rate']:.1%}
  vs {anom_sum['not_flagged_default_rate']:.1%} for unflagged accounts, so the flags carry real risk signal.
- **False positive:** an unusual but benign pattern, e.g. a one-off missed EMI from a payment-system error, or a high EMI-to-income ratio on a business
  loan serviced from business revenue rather than personal income. The analyst dismisses it with a note.
- **False negative:** a borrower deteriorating slowly *in line with* their peers, or whose stress only shows in data we don't hold (e.g. bank
  statements). It stays unflagged; the PD model and portfolio monitoring are the backstop.
- **Human-in-the-loop:** nothing is actioned automatically. An analyst reviews each flag, then **Confirms** (send to early collections / limit
  review), **Dismisses** (false positive, with a reason), or **Escalates**. Decisions are logged with a timestamp and can be used to tune thresholds.""")
    dec = anomaly.load_decisions()
    q = anom.merge(df[["account_id", "pd_score", "risk_band", "loan_amount"]], on="account_id").merge(
        dec[["account_id", "decision"]], on="account_id", how="left")
    q["decision"] = q.decision.fillna("Pending")
    k = st.columns(4)
    k[0].metric("Flagged accounts", f"{len(q):,}")
    k[1].metric("Pending review", f"{(q.decision == 'Pending').sum():,}")
    k[2].metric("Confirmed", f"{(q.decision == 'Confirm').sum():,}")
    k[3].metric("Dismissed", f"{(q.decision == 'Dismiss').sum():,}")

    a, b, c = st.columns(3)
    rule_f = a.multiselect("Rule", list(anomaly.RULES))
    lt_f = b.multiselect("Loan type", sorted(q.loan_type.unique()))
    st_f = c.multiselect("Status", ["Pending", "Confirm", "Dismiss", "Escalate"], default=["Pending"])
    view = q
    if rule_f:
        view = view[view.anomaly_rules.apply(lambda s: any(x in s for x in rule_f))]
    if lt_f:
        view = view[view.loan_type.isin(lt_f)]
    if st_f:
        view = view[view.decision.isin(st_f)]
    st.dataframe(view[["account_id", "customer_id", "loan_type", "income_band", "anomaly_score", "anomaly_rules", "pd_score",
                       "risk_band", "decision"]].style.format({"pd_score": "{:.1%}"}),
                 hide_index=True, width="stretch", height=260)

    if len(view):
        sel = st.selectbox("Review account", view.account_id.tolist())
        row = q[q.account_id == sel].iloc[0]
        acct = df[df.account_id == sel].iloc[0]
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(f"**Why {sel} was flagged** (severity {row.anomaly_score:.2f}, model PD {row.pd_score:.1%}, {row.risk_band})")
            for reason in json.loads(row.anomaly_reasons):
                st.markdown(f"- {reason}")
            h = rep[rep.account_id == sel].sort_values("payment_month")
            fig = go.Figure()
            fig.add_bar(x=h.payment_month, y=h.emi_paid / h.emi_due, name="Share of EMI paid", marker_color="#1565c0")
            fig.add_scatter(x=h.payment_month, y=h.days_past_due, name="DPD", yaxis="y2", line_color="#c62828")
            fig.add_vrect(x0=9.5, x1=12.5, fillcolor="#ffcdd2", opacity=0.3, line_width=0, annotation_text="recent 3 months")
            fig.update_layout(height=260, margin=dict(t=10, b=10), yaxis=dict(tickformat=".0%"),
                              yaxis2=dict(overlaying="y", side="right", title="DPD"), legend=dict(orientation="h", y=1.2))
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.markdown("**Record decision**")
            with st.form(f"dec_{sel}"):
                choice = st.radio("Decision", ["Confirm", "Dismiss", "Escalate"], horizontal=True,
                                  help="Confirm = real risk, send to early collections / limit review. Dismiss = false positive.")
                note = st.text_input("Reason / note", placeholder="e.g. one-off bank-side payment failure, verified with customer")
                reviewer = st.text_input("Reviewer", value="analyst")
                if st.form_submit_button("Save decision", type="primary"):
                    anomaly.record_decision(sel, choice, note, reviewer)
                    st.success(f"Saved: {sel} → {choice}")
                    st.rerun()
            if row.decision != "Pending":
                d = dec[dec.account_id == sel].iloc[0]
                st.info(f"Current decision: **{d.decision}** by {d.reviewer} at {d.decided_at}. {d.note if isinstance(d.note, str) else ''}")
            st.caption(f"Loan {money(acct.loan_amount)} · {acct.employment_type} · income {money(acct.annual_income)} · "
                       f"bureau lines {int(acct.bureau_lines)} ({int(acct.bureau_written_off)} written off)")
    dec = anomaly.load_decisions()
    if len(dec):
        with st.expander(f"Decision log ({len(dec)})"):
            st.dataframe(dec, hide_index=True, width="stretch")
            st.download_button("Download decisions CSV", dec.to_csv(index=False), "review_decisions.csv")
    st.caption("Decisions are stored in data/processed/review_decisions.csv. On Streamlit Cloud that file resets when the app restarts; "
               "production would write to a database table.")

# ================= MODEL & DATA QUALITY =================
with tab_model:
    st.subheader("Default model")
    mt = pd.DataFrame([{"model": "Logistic regression (baseline)", **metrics["logistic_regression"]["test"]},
                       {"model": "LightGBM (chosen)", **metrics["lightgbm"]["test"]},
                       {"model": "LightGBM, 5-fold out-of-fold (all accounts)", **metrics["lightgbm"]["oof_5fold"]}])
    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(mt, hide_index=True, width="stretch")
        st.caption(f"Hold-out test set: {metrics['test_rows']:,} accounts (30%, stratified), default rate "
                   f"{metrics['test_default_rate']:.1%}. Train AUC LightGBM {metrics['lightgbm']['train']['AUC']} → small overfit gap.")
        bands = pd.DataFrame(metrics["bands"])
        bands["risk_band"] = pd.Categorical(bands.risk_band, ORDER["risk_band"], ordered=True)
        st.markdown("**Calibration by band** (out-of-fold)")
        st.dataframe(bands.sort_values("risk_band").style.format({"actual_default": "{:.1%}", "avg_pd": "{:.1%}"}), hide_index=True)
        st.markdown("""**Approach.** 1 row per account; 40 features (application + engineered bureau + repayment aggregates).
Collections events excluded (they are triggered by delinquency = leakage). Scoring point = month 12 of the loan.
LightGBM hyper-parameters chosen by 5-fold CV on the train split; shallow trees because deeper ones overfit.
The signal is mostly monotonic, so the LR baseline is close behind; LightGBM adds ~+0.01 AUC.""")
    with c2:
        st.markdown("**Global feature importance (mean |SHAP|, LightGBM)**")
        top = shap_imp.head(15).iloc[::-1]
        fig = px.bar(top, x="mean_abs_shap", y="feature", orientation="h")
        fig.update_layout(height=430, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        with st.expander("Logistic regression coefficients (standardised)"):
            st.dataframe(lr_coef.head(20), hide_index=True)
    st.markdown("""**Monitoring after deployment**
- **Stability:** monthly PSI of the score distribution vs the training baseline (alert >0.10, retrain review >0.25) plus feature-level PSI/CSI on the top-10 SHAP features.
- **Performance:** AUC/KS/Gini on each cohort once its 12-month outcome window matures; alert if Gini drops >5 points. Track actual vs predicted default by band (calibration).
- **Data quality:** null rates, label drift (e.g. new gender codes), bureau feed latency, join coverage (thin-file share).
- **Fairness and governance:** default and approval rates by age band and gender; annual independent validation; documented challenger model; the anomaly-review dismiss rate as a feedback signal.""")
    st.subheader("Data-quality log (every decision taken in ingestion)")
    st.dataframe(dq, hide_index=True, width="stretch")
