"""Shared paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"

# Snapshot date for "age" style features: day after the latest date in the data.
AS_OF = "2024-01-01"

RANDOM_STATE = 42
GROQ_MODEL = "openai/gpt-oss-120b"        # LLM for "Ask the Portfolio" (Groq free tier)
GROQ_FALLBACK_MODEL = "qwen/qwen3.8-27b"  # if gpt-oss keeps emitting malformed tool calls

# Risk segments on the model's predicted probability of default (PD).
RISK_BANDS = [(0.0, 0.10, "Low"), (0.10, 0.25, "Medium"), (0.25, 0.50, "High"), (0.50, 1.01, "Very High")]
