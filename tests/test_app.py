"""Smoke + correctness tests. Run: pytest -q (after python -m src.build)."""
import json
from pathlib import Path

import pandas as pd
import pytest

from src import ask
from src.analytics import load_portfolio, segment_table
from src.config import PROCESSED

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def df():
    return load_portfolio()


def test_grain_and_target(df):
    raw = pd.read_csv(ROOT / "data/raw/loan_accounts.csv")
    assert len(df) == len(raw) and df.account_id.is_unique
    assert df.target_default.sum() == raw.target_default.sum()


def test_cleaning(df):
    assert set(df.gender.unique()) == {"Male", "Female"}
    assert df.interest_rate_pct.isna().sum() == 0 and df.interest_rate_missing.sum() == 294
    assert df.pd_score.between(0, 1).all()


def test_segment_table_matches_raw(df):
    raw = pd.read_csv(ROOT / "data/raw/loan_accounts.csv")
    t = segment_table(df, ["loan_type"]).set_index("loan_type")
    assert t.default_rate["Mortgage"] == pytest.approx(raw[raw.loan_type == "Mortgage"].target_default.mean())


def test_query_tool_and_filters(df):
    out = ask.query_portfolio(df, ["default_rate"], ["loan_type"], {"region": ["South"]})
    assert out["based_on_accounts"] == f"{(df.region == 'South').sum():,}"
    assert "error" in ask.query_portfolio(df, ["accounts"], filters={"region": ["Mars"]})


def test_verifier_catches_invented_numbers(df):
    out = ask.query_portfolio(df, ["default_rate"], ["loan_type"])
    good = "Mortgage has the highest default rate at 24.7%. Source: query_portfolio"
    bad = "Mortgage has the highest default rate at 31.2%. Source: query_portfolio"
    assert ask.verify_numbers(good, [out], "q") == []
    assert ask.verify_numbers(bad, [out], "q") == [31.2]


def test_anomaly_reasons_present():
    a = pd.read_parquet(PROCESSED / "anomalies.parquet")
    assert len(a) > 0 and all(len(json.loads(r)) >= 1 for r in a.anomaly_reasons)


def test_app_renders_without_errors():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
