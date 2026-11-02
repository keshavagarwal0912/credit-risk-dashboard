# Credit Risk Portfolio Monitor

An interactive credit-risk monitoring dashboard built from 5 synthetic bank tables (10,000 customers, 14,685 loans). It includes a default model (logistic regression baseline + LightGBM) and two embedded AI features:

- **Ask the Portfolio**: grounded LLM Q&A with tool use (`openai/gpt-oss-120b` on Groq's free tier).
- **Anomaly detection**: a review queue with a human-in-the-loop decision for each flagged account.

**Python version:** 3.11 (developed on 3.11; on Streamlit Community Cloud pick 3.11 under *Advanced settings*).

## Run it (3 commands)

```bash
pip install -r requirements.txt     # 1. install (every package pinned)
python -m src.build                 # 2. build the clean dataset, train the models, run anomaly detection
streamlit run app/app.py            # 3. start the dashboard
```

Build outputs (`data/processed/`, `artifacts/`) are committed so the hosted app starts instantly. If they are missing, the app rebuilds them on first start (about 1 minute). Optional: `pytest` runs the tests, and `python docs/make_docs.py` regenerates the two PDFs.

## LLM API key

The "Ask the Portfolio" feature calls the Groq API (`openai/gpt-oss-120b`, with `qwen/qwen3.8-27b` as a fallback if a tool call is malformed). The free-tier key comes from https://console.groq.com. It is read from `st.secrets` first, then from the environment.

- **Local:** copy `.env.example` to `.env` and set `GROQ_API_KEY=...`, or export it in your shell.
- **Streamlit Cloud:** *App settings → Secrets* → `GROQ_API_KEY = "gsk_..."`.
- **No key set:** the rest of the dashboard works normally. The Ask tab shows a notice that the LLM is disabled, and it offers a **manual query builder** that runs the same grounded query engine without an LLM.

No key or secret is stored in this repository (`.env` and `.streamlit/secrets.toml` are git-ignored).

## Where each item in the brief lives

| Brief item | Location |
|---|---|
| 1. Data ingestion, join logic, data-quality decisions | `src/ingest.py` (module docstring describes the joins). Every DQ decision is written to `data/processed/data_quality_log.csv` and shown in the dashboard's *Model & data quality* tab |
| 2. Portfolio analytics + 4-5 insights with "so what" (incl. a cross-dimensional cut) | `src/analytics.py` → `insights()` (figures computed live). Shown in *Portfolio → Portfolio insights* |
| 3. Default model: LR baseline + LightGBM, AUC/KS/Gini, SHAP, monitoring | `src/model.py`. Results in `artifacts/metrics.json`, `artifacts/shap_importance.csv`, `artifacts/lr_coefficients.csv`. Monitoring plan + `psi()` helper in *Model & data quality* |
| 4. Dashboard: portfolio view, account view, filters + drill-down | `app/app.py`: sidebar filters, *Portfolio* tab (risk segments, exposure, trends, heatmap, segment→account drill-down), *Account view* tab (PD, band, SHAP drivers, history) |
| AI feature 1: Ask the Portfolio (LLM) | `src/ask.py` (tools, system prompt, number verifier) + the *Ask the Portfolio* tab |
| AI feature 2: Anomaly detection + review decisions | `src/anomaly.py` (rules, Isolation Forest, decision log) + the *Anomaly review* tab |
| Trigger / false positive / false negative / human-in-the-loop for each AI feature | "How this works" expander at the top of each AI tab |
| Architecture diagram | `docs/architecture.pdf` |
| Solution overview | `docs/solution_overview.pdf` |
| Tests | `tests/test_app.py` (grain, cleaning, tool correctness, verifier, headless app render) |

## Key design choices

- **Join logic:** `loan_accounts` is the base table (1 row per account, holds the target). `customers` is joined 1:1 on `customer_id`. Bureau lines are aggregated per customer, and repayment months and collections events per account, **before** the left joins. This means the grain never fans out (asserted in code).
- **Data quality:**
  - M/F gender labels are normalised.
  - Missing values are imputed with segment medians, plus a `_missing` flag.
  - Missing bureau DPD stays as NaN (unknown is not the same as 0).
  - 1,008 customers with no bureau record are kept and flagged as `bureau_thin_file`.
  - Date inconsistencies are flagged and kept: 7,293 loans originated before `customer_since`, and 3,875 collection events are dated before origination.
- **Model:**
  - It is a behavioural score at month 12.
  - Collections are excluded as target leakage.
  - LightGBM is shallow and tuned with cross-validation.
  - Dashboard scores are 5-fold out-of-fold, so they are never in-sample.
- **LLM grounding:**
  - The LLM never sees raw data. It can only call 4 deterministic pandas tools that return pre-formatted figures.
  - It must cite a *Source* and decline out-of-scope questions.
  - `verify_numbers()` badges any figure in the answer that doesn't appear in a tool result.
- **Anomaly decisions** are saved to `data/processed/review_decisions.csv`. On Streamlit Cloud the filesystem is ephemeral, so decisions reset when the app restarts; a CSV download is provided. Production would write them to a database table.
