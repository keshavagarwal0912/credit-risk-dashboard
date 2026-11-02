"""Generate docs/architecture.pdf and docs/solution_overview.pdf (one page each) from the built artifacts.
Run: python docs/make_docs.py   (after python -m src.build)"""
import json
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analytics import insights, load_portfolio  # noqa: E402

AUTHOR = "Keshav Agarwal"
M = json.loads((ROOT / "artifacts/metrics.json").read_text())
A = json.loads((ROOT / "data/processed/anomaly_summary.json").read_text())
INK, GREY, RED, BLUE, GREEN, AMBER = (colors.HexColor(c) for c in ("#263238", "#eceff1", "#c62828", "#1565c0", "#2e7d32", "#ef6c00"))


# ---------------------------------------------------------------- architecture.pdf
def box(c, x, y, w, h, title, lines, fill=GREY, edge=INK, title_color=INK):
    c.setFillColor(fill); c.setStrokeColor(edge); c.setLineWidth(1)
    c.roundRect(x, y, w, h, 6, fill=1, stroke=1)
    c.setFillColor(title_color); c.setFont("Helvetica-Bold", 9.5)
    c.drawString(x + 6, y + h - 13, title)
    c.setFillColor(INK); c.setFont("Helvetica", 7.4)
    for i, ln in enumerate(lines):
        c.drawString(x + 6, y + h - 25 - i * 9.4, ln)


def arrow(c, x1, y1, x2, y2, label=None, color=INK, dash=False):
    import math
    c.setStrokeColor(color); c.setFillColor(color); c.setLineWidth(1.3)
    c.setDash(3, 2) if dash else c.setDash()
    c.line(x1, y1, x2, y2); c.setDash()
    a = math.atan2(y2 - y1, x2 - x1)
    p = c.beginPath(); p.moveTo(x2, y2)
    p.lineTo(x2 - 7 * math.cos(a - 0.4), y2 - 7 * math.sin(a - 0.4))
    p.lineTo(x2 - 7 * math.cos(a + 0.4), y2 - 7 * math.sin(a + 0.4)); p.close()
    c.drawPath(p, fill=1, stroke=0)
    if label:
        c.setFont("Helvetica-Oblique", 6.8)
        c.drawCentredString((x1 + x2) / 2, (y1 + y2) / 2 + 4, label)


def architecture(path):
    W, H = landscape(A4)
    c = canvas.Canvas(str(path), pagesize=(W, H))
    c.setTitle("Credit Risk Portfolio Monitor - Architecture"); c.setAuthor(AUTHOR)
    c.setFont("Helvetica-Bold", 15); c.setFillColor(INK)
    c.drawString(30, H - 36, "Credit Risk Portfolio Monitor: architecture")
    c.setFont("Helvetica", 8.5)
    c.drawString(30, H - 50, "Python 3.11 · batch build (python -m src.build) writes parquet/joblib artifacts · Streamlit app reads them · "
                             "hosted on Streamlit Community Cloud (API key in Secrets)")

    y1, h1 = 330, 150
    box(c, 25, y1, 135, h1, "1. Raw data  (data/raw)", ["customers.csv  10,000", "loan_accounts.csv  14,685", "bureau_credit.csv  22,979",
        "repayment_history.csv  176,220", "collections_events.csv  8,635", "", "Tool: CSV files as provided"])
    box(c, 185, y1, 175, h1, "2. Ingestion  src/ingest.py", ["pandas: key + referential checks", "fix: M/F gender, median+flag imputes,",
        "  bureau DPD kept NaN, thin-file flag,", "  date-inconsistency flags", "aggregate many-to-one tables FIRST", "  (bureau→customer, repay/coll→account)",
        "left-join → 1 row per account (asserted)", "→ accounts.parquet + data_quality_log.csv"])
    box(c, 385, y1 + 80, 190, 70, "3a. Analytics  src/analytics.py", ["pandas segment tables (one source of truth", "  for dashboard + LLM tools)",
        "insights computed live, no hard-coded figures"])
    box(c, 385, y1 - 25, 190, 95, "3b. Default model  src/model.py", ["scikit-learn LogisticRegression (baseline)", "LightGBM (chosen), CV-tuned, 5-fold OOF scores",
        f"test AUC {M['lightgbm']['test']['AUC']} / KS {M['lightgbm']['test']['KS']} / Gini {M['lightgbm']['test']['Gini']}",
        "SHAP TreeExplainer: global + per-account drivers", "→ scores.parquet, models.joblib, metrics.json"])
    box(c, 385, y1 - 140, 190, 100, "3c. AI feature 2: anomaly detection", ["src/anomaly.py: pandas + scikit-learn", "own history: months 10-12 vs 1-9 (DPD, pay ratio,",
        "  consecutive misses)", "peer: loan type × income band (robust z, p99)", "IsolationForest (2%) on repayment+bureau",
        f"→ anomalies.parquet ({A['flagged_total']:,} flags + reasons)"], fill=colors.HexColor("#fff3e0"), edge=AMBER, title_color=AMBER)

    box(c, 600, y1 - 140, 215, 290, "4. Dashboard  app/app.py", ["Streamlit + Plotly; sidebar filters (7 dims)", "",
        "Portfolio: KPIs, risk bands, dimension chart,", "  cross-dim heatmap, quarterly trend,", "  segment → account drill-down, insights",
        "Account view: PD, band, SHAP drivers,", "  repayment timeline, bureau, collections",
        "Ask the Portfolio (AI feature 1)", "Anomaly review: queue, reasons, chart,", "  Confirm / Dismiss / Escalate + note",
        "  → review_decisions.csv (HITL log)", "Model & data quality: metrics, SHAP,", "  calibration, monitoring plan, DQ log",
        "", "No key set → LLM disabled, manual", "  query builder on the same engine"])

    # LLM lane
    box(c, 385, 40, 190, 105, "AI feature 1: Ask the Portfolio  src/ask.py", ["1. analyst question → LLM API", "2. LLM picks a tool; app runs it in pandas:",
        "   query_portfolio · list_dimension_values ·", "   get_account · get_model_info", "3. results back to LLM → short answer + Source",
        "4. verify_numbers(): every figure must match", "   a tool result, else badged UNVERIFIED"], fill=colors.HexColor("#e3f2fd"), edge=BLUE, title_color=BLUE)
    box(c, 600, 40, 215, 105, "LLM: Groq API, openai/gpt-oss-120b", ["(fallback qwen/qwen3.8-27b on malformed tool calls)",
        "called ONLY from src/ask.py → ask()", "SENT: question, system prompt, 4 tool schemas,",
        "  aggregated tool results (formatted counts,", "  rates, sums; 1 account's fields if asked)", "NOT SENT: raw tables, rows in bulk, the API key",
        "  (key from st.secrets / env) · max 6 tool rounds"], fill=colors.HexColor("#e3f2fd"), edge=BLUE, title_color=BLUE)

    arrow(c, 160, y1 + 75, 185, y1 + 75)
    arrow(c, 360, y1 + 110, 385, y1 + 115)
    arrow(c, 360, y1 + 60, 385, y1 + 30)
    arrow(c, 360, y1 + 30, 385, y1 - 70)
    arrow(c, 575, y1 + 115, 600, y1 + 115)
    arrow(c, 575, y1 + 25, 600, y1 + 25)
    arrow(c, 575, y1 - 90, 600, y1 - 90)
    arrow(c, 480, y1 + 80, 480, y1 + 70, color=INK)
    arrow(c, 600, 120, 575, 120, color=BLUE)
    arrow(c, 575, 75, 600, 75, color=BLUE)
    c.setFont("Helvetica-Oblique", 6.8); c.setFillColor(BLUE)
    c.drawCentredString(587, 126, "tool_use"); c.drawCentredString(587, 64, "tool_result")
    arrow(c, 707, y1 - 140, 707, 145, color=BLUE, dash=True)
    c.drawString(712, 168, "question in / answer + trace out")
    box(c, 185, 185, 175, 75, "data/processed  (parquet)", ["accounts · scores · anomalies · repayment ·", "bureau · collections · data_quality_log",
        "", "read by dashboard AND by the LLM tools", "(same functions → same numbers)"])
    arrow(c, 272, y1, 272, 260)
    arrow(c, 385, 92, 300, 185, color=BLUE, dash=True)
    c.setFillColor(BLUE); c.drawString(250, 140, "tools query in pandas")

    c.setFillColor(INK); c.setFont("Helvetica-Bold", 8.5)
    c.drawString(30, 120, "Human-in-the-loop")
    c.setFont("Helvetica", 7.4)
    for i, ln in enumerate(["Ask: analyst checks Source line + tool", "  trace; UNVERIFIED badge blocks reuse.",
                            "Anomaly: nothing auto-actioned; analyst", "  confirms / dismisses / escalates with", "  a note; log tunes thresholds.",
                            "", "Tests: pytest (grain, cleaning, tool", "  correctness, verifier, app render)"]):
        c.drawString(30, 108 - i * 9.4, ln)
    c.showPage(); c.save()


# ---------------------------------------------------------------- solution_overview.pdf
def overview(path):
    df = load_portfolio()
    ins = insights(df)
    s = ParagraphStyle("b", fontName="Helvetica", fontSize=8.2, leading=10.3, alignment=TA_LEFT, textColor=INK)
    h = ParagraphStyle("h", parent=s, fontName="Helvetica-Bold", fontSize=9.8, leading=12, spaceBefore=5, spaceAfter=1.5, textColor=RED)
    t = ParagraphStyle("t", parent=s, fontName="Helvetica-Bold", fontSize=13, leading=15)
    P = lambda x, st=s: Paragraph(x, st)  # noqa: E731
    lr, gb = M["logistic_regression"]["test"], M["lightgbm"]["test"]
    story = [
        P("Credit Risk Portfolio Monitor: solution overview", t),
        P("A Streamlit dashboard over 14,685 loans / 10,000 customers. It has a portfolio view, an account view, a default model and two embedded AI "
          "features. Built from 5 synthetic tables, 1 row per account after aggregating bureau, repayment and collections data."),
        P("What the dashboard shows and how an analyst uses it", h),
        P("<b>Sidebar filters</b> (loan type, employment, region, income band, age band, risk band, city tier) drive the portfolio view. "
          "<b>Portfolio:</b> KPIs (exposure, default rate vs portfolio, defaulted exposure, avg PD, flags), risk segments (exposure + actual default "
          "by model band), a dimension explorer, a cross-dimension heatmap, a quarterly origination trend, and a segment → account <b>drill-down</b>. "
          "The analyst uses it to see where losses concentrate and which segments drift. <b>Account view:</b> PD, band, top-5 SHAP drivers, 12-month "
          "repayment timeline, bureau lines and collections. Used to explain a score before a limit or collections decision. <b>Ask the Portfolio</b> and "
          "<b>Anomaly review</b> (below). <b>Model & data quality:</b> metrics, SHAP, calibration by band, monitoring plan and the full DQ log."),
        P("AI feature 1: Ask the Portfolio (LLM)", h),
        P("<b>How:</b> the LLM (openai/gpt-oss-120b on Groq's free tier) gets the question plus 4 tools (query_portfolio, "
          "list_dimension_values, get_account, get_model_info). It never sees "
          "raw tables. The tools compute in pandas with the same functions as the dashboard and return formatted figures. The LLM answers in 1-4 sentences "
          "with a <i>Source</i> line. A post-check matches every number in the answer to a tool result; misses are badged UNVERIFIED, and the tool "
          "trace is shown. Out-of-scope questions (forecasts, profit, names) are declined. <b>Trigger:</b> analyst submits a question. "
          "<b>False positive:</b> a confident answer built on the wrong filter or metric (visible in the trace). <b>False negative:</b> an answerable "
          "question declined, e.g. an odd label spelling. <b>Human step:</b> analyst checks the Source/trace before reusing a figure. "
          "<b>No API key:</b> LLM disabled; a manual query builder runs the same engine."),
        P("AI feature 2: Anomaly detection", h),
        P(f"<b>Trigger:</b> any of (a) <i>own history</i>, months 10-12 vs 1-9: DPD jump to ≥30 from ≤5, payment ratio drop from ≥95% to &lt;50%, "
          f"3+ consecutive misses; (b) <i>peer segment</i> (loan type × income band): EMI-to-income robust z &gt; 3.5, bureau 30+DPD / written-off / "
          f"new lines at or above the peer 99th percentile, utilization ≥90% where the peer median is &lt;50%; (c) Isolation Forest top 2%. "
          f"<b>{A['flagged_total']:,} flagged</b>, ranked by severity, each with plain-English reasons. Flagged default rate is "
          f"{A['flagged_default_rate']:.1%} vs {A['not_flagged_default_rate']:.1%} unflagged. <b>False positive:</b> a benign one-off (bank-side "
          "payment failure; a business loan serviced from revenue). <b>False negative:</b> slow deterioration in line with peers, or stress outside our "
          "data. <b>Human step:</b> nothing is auto-actioned. The analyst Confirms / Dismisses / Escalates with a note, logged with a timestamp "
          "to tune thresholds."),
        P("Model results and approach", h),
    ]
    tbl = Table([[f"Model (30% hold-out, n={M['test_rows']:,})", "AUC", "KS", "Gini"],
                 ["Logistic regression (baseline)", lr["AUC"], lr["KS"], lr["Gini"]],
                 ["LightGBM (chosen)", gb["AUC"], gb["KS"], gb["Gini"]],
                 ["LightGBM 5-fold out-of-fold (all accounts)", *(M["lightgbm"]["oof_5fold"][k] for k in ("AUC", "KS", "Gini"))]],
                colWidths=[190, 45, 45, 45])
    tbl.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 8), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
                             ("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), GREY),
                             ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    story += [tbl, Spacer(1, 2),
              P("40 features: application, engineered bureau (write-offs, DPD, utilization, new lines, thin-file) and repayment (DPD level and trend, "
                "pay ratio, misses, late count). Collections are excluded as leakage. Scoring point = month 12. LightGBM is shallow and CV-tuned "
                "(deeper trees overfit: train 0.89 vs test 0.66). The signal is largely monotonic, so LR is close behind. Top SHAP features: income, "
                "years employed, repayment DPD, bureau DPD, age. Bands are well calibrated (actual default rises Low→Very High). <b>Monitoring:</b> monthly "
                "score/feature PSI (alert 0.10 / act 0.25), Gini on matured cohorts (alert at -5 pts), calibration by band, DQ checks, fairness by age/gender."),
              P("Top 3 portfolio insights", h)]
    for i in ins[:3]:
        story.append(P(f"<b>{i['title']}.</b> {i['finding']} <i>So what:</i> {i['so_what']}"))
    story += [
        P("Key assumptions and what production needs from the bank's IT and data teams", h),
        P("<b>Assumptions:</b> target_default is a fixed historical outcome observed after month 12. Collections dates that precede origination and "
          "customer_since dates after origination are data errors (flagged and kept, not used as features). Bureau DPD nulls mean unknown, not zero. "
          "<b>Needed for production:</b> governed source feeds (core banking loans/repayments daily, bureau pull API, collections system) with keys and "
          "timestamps; a point-in-time feature store to avoid leakage; a scheduled pipeline (Airflow/dbt) and model registry; a database table for review "
          "decisions in place of the CSV; SSO + role-based access; an approved LLM endpoint (private / VPC, zero data retention) with logging; model risk "
          "management validation; data-privacy sign-off on which aggregates may leave the bank."),
    ]
    doc = SimpleDocTemplate(str(path), title="Credit Risk Portfolio Monitor - Solution Overview", author=AUTHOR,
                            pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm, topMargin=10 * mm, bottomMargin=8 * mm)
    pages = []
    doc.build(story, onLaterPages=lambda c, d: pages.append(d.page))
    assert not pages, "solution overview overflowed onto a 2nd page"


if __name__ == "__main__":
    architecture(ROOT / "docs/architecture.pdf")
    overview(ROOT / "docs/solution_overview.pdf")
    print("wrote docs/architecture.pdf, docs/solution_overview.pdf")
