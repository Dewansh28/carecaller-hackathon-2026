"""Extract features from call metadata and numeric columns."""
import pandas as pd
import numpy as np
from config import SAFE_OUTCOMES, ALL_OUTCOMES, DAYS_OF_WEEK


# Numeric columns to pass through directly
NUMERIC_COLS = [
    "call_duration",
    "attempt_number",
    "whisper_mismatch_count",
    "answered_count",
    "response_completeness",
    "turn_count",
    "user_turn_count",
    "agent_turn_count",
    "user_word_count",
    "agent_word_count",
    "avg_user_turn_words",
    "avg_agent_turn_words",
    "interruption_count",
    "max_time_in_call",
    "hour_of_day",
]


def extract(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of metadata features, one row per call."""
    out = pd.DataFrame(index=df.index)

    # ── Numeric pass-through ───────────────────────────────────────────
    for col in NUMERIC_COLS:
        out[col] = df[col].fillna(0).astype(float)

    # ── Engineered numeric features ────────────────────────────────────
    out["has_whisper_mismatch"] = (df["whisper_mismatch_count"] > 0).astype(int)
    out["duration_per_question"] = df["call_duration"] / np.maximum(df["answered_count"], 1)
    out["user_agent_word_ratio"] = df["user_word_count"] / np.maximum(df["agent_word_count"], 1)
    out["is_safe_outcome"] = df["outcome"].isin(SAFE_OUTCOMES).astype(int)
    out["words_per_second"] = (
        (df["user_word_count"] + df["agent_word_count"]) / np.maximum(df["call_duration"], 1)
    )
    out["completeness_x_duration"] = df["response_completeness"] * df["call_duration"]

    # ── Categorical one-hot encoding ───────────────────────────────────
    for outcome in ALL_OUTCOMES:
        out[f"outcome_{outcome}"] = (df["outcome"] == outcome).astype(int)

    out["direction_outbound"] = (df["direction"] == "outbound").astype(int)
    out["whisper_completed"] = (df["whisper_status"] == "completed").astype(int)
    out["form_submitted"] = df["form_submitted"].astype(int)

    for day in DAYS_OF_WEEK:
        out[f"day_{day}"] = (df["day_of_week"] == day).astype(int)

    return out
