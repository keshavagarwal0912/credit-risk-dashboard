"""'Ask the Portfolio': grounded natural-language Q&A.

How grounding works:
  - The LLM never sees raw rows. It can only call the deterministic tools below, which compute figures
    with pandas from the same tables the dashboard uses, and return pre-formatted values.
  - The system prompt forbids numbers that are not in a tool result and requires declining questions
    the tools cannot answer.
  - After the answer comes back, verify_numbers() checks every number in it against the tool outputs;
    anything unmatched is shown to the analyst as UNVERIFIED.
Data sent to the LLM: the question, tool schemas, and aggregated tool results (segment counts, rates,
sums, and single-account fields when asked for by account_id). No names/PII exist in this dataset.
"""
import json
import numbers
import re

import pandas as pd

from src.analytics import DIMENSIONS, METRICS, ORDER, apply_filters, segment_table, summarize
from src.config import ARTIFACTS, GROQ_FALLBACK_MODEL, GROQ_MODEL, PROCESSED

MAX_TOOL_ROUNDS = 6

SYSTEM = f"""You are "Ask the Portfolio", an assistant for credit-risk analysts at a bank.
You answer questions about a loan portfolio of 14,685 loan accounts / 10,000 customers (synthetic data).

Rules (strict):
1. Every number you state MUST be copied exactly, character for character, from a tool result in this
   conversation (e.g. "$24,215,330", not "$24.2 million"; "24.7%", not "about 25%"). Never estimate, round,
   convert units, recall, or do your own arithmetic. If a comparison needs a number no tool returned, describe it in words.
2. Always call a tool before answering a data question.
3. If the question cannot be answered with the available tools and fields (e.g. profit, revenue, forecasts,
   macroeconomics, customer names, anything outside this dataset), reply starting with "I can't answer that
   from this data" and say briefly what the data does cover. Do not guess.
4. Keep the answer short: 1-4 sentences of plain English, then a line starting "Source:" naming the tool,
   the grouping and the filters used (e.g. "Source: query_portfolio - loan_accounts joined to customers,
   grouped by loan_type, no filters, n=14,685 accounts").
5. "Default" means target_default=1 in loan_accounts (historical outcome). "PD"/"risk score" means the model's
   predicted probability of default.

Available dimensions: {json.dumps(DIMENSIONS)}
Available metrics: {json.dumps(METRICS)}
"""

TOOLS = [
    {
        "name": "query_portfolio",
        "description": "Compute portfolio metrics, optionally grouped by up to 2 dimensions and filtered. "
                       "Returns formatted figures plus the row count they are based on. Use for any "
                       "aggregate question (rates, counts, exposure, comparisons across segments, rankings).",
        "input_schema": {
            "type": "object",
            "properties": {
                "metrics": {"type": "array", "items": {"type": "string", "enum": list(METRICS)},
                            "description": "Metrics to return."},
                "group_by": {"type": "array", "items": {"type": "string", "enum": list(DIMENSIONS)}, "maxItems": 2,
                             "description": "Dimensions to group by; empty for a portfolio total."},
                "filters": {"type": "object", "description": "Map of dimension -> list of allowed values, "
                            "e.g. {\"loan_type\": [\"Mortgage\"], \"region\": [\"South\"]}.",
                            "additionalProperties": {"type": "array", "items": {"type": "string"}}},
                "sort_by": {"type": "string", "enum": list(METRICS), "description": "Sort groups descending by this metric."},
                "top_n": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["metrics"],
        },
    },
    {
        "name": "list_dimension_values",
        "description": "List the valid values of a dimension (use before filtering if unsure of spelling).",
        "input_schema": {"type": "object", "properties": {"dimension": {"type": "string", "enum": list(DIMENSIONS)}},
                         "required": ["dimension"]},
    },
    {
        "name": "get_account",
        "description": "Details for one loan account: loan, customer profile, repayment summary, bureau summary, "
                       "model PD + top drivers, and anomaly flags.",
        "input_schema": {"type": "object", "properties": {"account_id": {"type": "string"}}, "required": ["account_id"]},
    },
    {
        "name": "get_model_info",
        "description": "Default-model performance (AUC, KS, Gini for logistic regression and LightGBM), "
                       "top features by SHAP, and the anomaly-detection rule counts.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def fmt(metric, v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    if metric in ("default_rate", "exposure_share", "avg_pd"):
        return f"{v:.1%}"
    if metric in ("exposure", "defaulted_exposure", "outstanding", "avg_loan_amount", "avg_income"):
        return f"${v:,.0f}"
    if metric == "avg_interest_rate":
        return f"{v:.2f}%"
    return f"{int(v):,}"


def query_portfolio(df, metrics, group_by=None, filters=None, sort_by=None, top_n=None):
    bad = [f for f in (filters or {}) if f not in DIMENSIONS]
    if bad:
        return {"error": f"Unknown filter dimension(s): {bad}. Valid: {list(DIMENSIONS)}"}
    sub = apply_filters(df, filters)
    if sub.empty:
        return {"error": "No accounts match these filters. Check values with list_dimension_values."}
    metrics = [m for m in metrics if m in METRICS] or ["accounts", "default_rate"]
    out = {"based_on_accounts": f"{len(sub):,}", "filters": filters or {}, "group_by": group_by or []}
    if not group_by:
        s = summarize(sub)
        s["exposure_share"] = s["exposure"] / df.loan_amount.sum()
        out["result"] = {m: fmt(m, s[m]) for m in metrics}
        return out
    t = segment_table(sub, group_by)
    if sort_by:
        t = t.sort_values(sort_by, ascending=False)
    elif len(group_by) == 1 and group_by[0] not in ORDER:
        t = t.sort_values(metrics[0], ascending=False)
    if top_n:
        t = t.head(top_n)
    rows = []
    for _, r in t.iterrows():
        row = {d: str(r[d]) for d in group_by}
        row.update({m: fmt(m, r[m]) for m in metrics})
        row["n_accounts"] = f"{int(r.accounts):,}"
        rows.append(row)
    out["rows"] = rows
    return out


def list_dimension_values(df, dimension):
    if dimension not in DIMENSIONS:
        return {"error": f"Unknown dimension {dimension}"}
    return {"dimension": dimension, "values": sorted(df[dimension].astype(str).unique().tolist())}


def get_account(df, account_id):
    r = df[df.account_id == account_id.strip().upper()]
    if r.empty:
        return {"error": f"No account {account_id}"}
    r = r.iloc[0]
    keep = ["account_id", "customer_id", "loan_type", "loan_purpose", "loan_amount", "interest_rate_pct", "tenure_months",
            "origination_date", "target_default", "age", "employment_type", "annual_income", "region", "income_band",
            "rep_missed", "rep_late", "rep_max_dpd", "rep_pay_ratio", "bureau_lines", "bureau_written_off",
            "bureau_max_dpd", "bureau_utilization", "coll_events", "risk_band"]
    out = {k: (round(float(r[k]), 4) if isinstance(r[k], numbers.Number) else str(r[k])) for k in keep}
    out["origination_date"] = str(r.origination_date.date())
    out["pd_score"] = f"{r.pd_score:.1%}"
    out["top_drivers"] = [f"{d['feature']}={d['value']} ({'raises' if d['shap'] > 0 else 'lowers'} risk)"
                          for d in json.loads(r.drivers)[:3]]
    out["anomaly_flagged"] = bool(r.anomaly_flag)
    if r.anomaly_flag:
        out["anomaly_reasons"] = json.loads(r.anomaly_reasons)
    return out


def get_model_info():
    m = json.loads((ARTIFACTS / "metrics.json").read_text())
    imp = pd.read_csv(ARTIFACTS / "shap_importance.csv").head(10)
    anom = json.loads((PROCESSED / "anomaly_summary.json").read_text())
    return {"test_set_metrics": {k: m[k]["test"] for k in ("logistic_regression", "lightgbm")},
            "test_rows": m["test_rows"], "top_features_by_mean_abs_shap": imp.feature.tolist(),
            "anomaly_rule_counts": anom}


def run_tool(df, name, args):
    try:
        if name == "query_portfolio":
            return query_portfolio(df, **args)
        if name == "list_dimension_values":
            return list_dimension_values(df, **args)
        if name == "get_account":
            return get_account(df, **args)
        if name == "get_model_info":
            return get_model_info()
        return {"error": f"Unknown tool {name}"}
    except TypeError as e:
        return {"error": f"Bad arguments: {e}"}


NUM = re.compile(r"(?<![A-Za-z_])\$?-?\d[\d,]*\.?\d*%?")


def _numbers(text):
    vals = []
    for tok in NUM.findall(text):
        t = tok.replace("$", "").replace(",", "").rstrip("%").rstrip(".")
        try:
            vals.append(float(t))
        except ValueError:
            pass
    return vals


def verify_numbers(answer, tool_outputs, question):
    """Return numbers in the answer that do not appear in any tool output (or the question)."""
    known = set()
    for o in tool_outputs:
        known.update(_numbers(json.dumps(o)))
    known.update(_numbers(question))
    body = answer.split("Source:")[0]
    unverified = []
    for v in _numbers(body):
        if v in known or any(abs(v - k) <= max(0.051, abs(k) * 0.001) for k in known):
            continue
        unverified.append(v)
    return unverified


def _finish(answer, trace, question):
    return {"answer": answer, "trace": trace, "unverified": verify_numbers(answer, [t["result"] for t in trace], question)}


GAVE_UP = "I couldn't complete this within the tool-call limit; try a narrower question."


def _groq_call(client, messages, tools):
    # gpt-oss occasionally emits a malformed tool call (Groq 400 "tool_use_failed"): retry, then fall back to Qwen.
    from groq import BadRequestError
    for model in [GROQ_MODEL, GROQ_MODEL, GROQ_MODEL, GROQ_FALLBACK_MODEL]:
        try:
            return client.chat.completions.create(model=model, messages=messages, tools=tools, tool_choice="auto",
                                                  temperature=0, max_completion_tokens=2000).choices[0].message
        except BadRequestError as e:
            if "tool_use_failed" not in str(e):
                raise
            err = e
    raise err


def ask(question, df, api_key):
    """Run the tool loop on Groq; returns answer, tool trace, unverified numbers."""
    from groq import Groq
    client = Groq(api_key=api_key, timeout=60.0)
    tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                               "parameters": t["input_schema"]}} for t in TOOLS]
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    trace = []
    for _ in range(MAX_TOOL_ROUNDS):
        msg = _groq_call(client, messages, tools)
        if not msg.tool_calls:
            return _finish((msg.content or "").strip(), trace, question)
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                        for c in msg.tool_calls]})
        for c in msg.tool_calls:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args, out = {}, {"error": "Arguments were not valid JSON"}
            else:
                out = run_tool(df, c.function.name, args)
            trace.append({"tool": c.function.name, "args": args, "result": out})
            messages.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(out)})
    return {"answer": GAVE_UP, "trace": trace, "unverified": []}
