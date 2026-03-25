"""Layer 1: Rule-based heuristic classifier.

High-precision rules derived from training data patterns.
Each rule targets a specific ticket type with near-zero false positives.
"""
import pandas as pd
import numpy as np
from config import SAFE_OUTCOMES


def predict(df: pd.DataFrame) -> pd.DataFrame:
    """Apply rules and return a DataFrame with rule predictions.

    Returns columns:
        rule_prediction: bool  — whether any rule flagged this call
        rule_confidence: float — max confidence across fired rules (0.0-1.0)
        rule_reasons: str      — semicolon-separated list of reasons
    """
    n = len(df)
    predictions = np.zeros(n, dtype=bool)
    confidences = np.zeros(n, dtype=float)
    reasons = [""] * n

    vn = df["validation_notes"].fillna("").str.lower()

    def _fire(mask, confidence, reason):
        nonlocal predictions, confidences, reasons
        for i in mask[mask].index:
            pos = df.index.get_loc(i)
            predictions[pos] = True
            confidences[pos] = max(confidences[pos], confidence)
            if reasons[pos]:
                reasons[pos] += "; " + reason
            else:
                reasons[pos] = reason

    # ── Rule 1: Whisper mismatch (STT errors) ─────────────────────────
    _fire(
        df["whisper_mismatch_count"] > 0,
        0.99,
        "whisper_mismatch > 0",
    )

    # ── Rule 2: Dosage guidance in validation notes (medical advice) ──
    _fire(
        vn.str.contains("dosage guidance", regex=False),
        0.98,
        "vn_dosage_guidance",
    )

    # ── Rule 3: Opted-out miscategorization ────────────────────────────
    _fire(
        vn.str.contains("not right now", regex=False) & (df["outcome"] == "opted_out"),
        0.95,
        "opted_out_miscategorization",
    )

    # ── Rule 4: Outcome corrected by validation ───────────────────────
    _fire(
        vn.str.contains("corrected by validation", regex=False),
        0.97,
        "outcome_corrected",
    )

    # ── Rule 5: Possible miscategorization ─────────────────────────────
    _fire(
        vn.str.contains("possible miscategorization", regex=False),
        0.95,
        "possible_miscategorization",
    )

    # ── Rule 6: Wrong number with answers (misclassification) ──────────
    _fire(
        (df["outcome"] == "wrong_number") & (df["answered_count"] > 0),
        0.95,
        "wrong_number_with_answers",
    )

    # ── Rule 7: Weight differs between sources ─────────────────────────
    _fire(
        vn.str.contains("differs between sources", regex=False),
        0.97,
        "weight_stt_error",
    )

    # ── Rule 8: Skipped questions flagged in VN ────────────────────────
    _fire(
        vn.str.contains("fabricated responses", regex=False),
        0.95,
        "fabricated_responses",
    )

    # ── Rule 9: Stop calling + wrong_number ────────────────────────────
    _fire(
        vn.str.contains("stop calling", regex=False) & (df["outcome"] == "wrong_number"),
        0.95,
        "wrong_number_stop_calling",
    )

    # ── Rule 10: Inconsistency in validation notes ─────────────────────
    _fire(
        vn.str.contains("inconsisten", regex=False),
        0.90,
        "vn_inconsistency",
    )

    # ── Safe outcome override (negative rule) ──────────────────────────
    safe_mask = df["outcome"].isin(SAFE_OUTCOMES)
    for i in safe_mask[safe_mask].index:
        pos = df.index.get_loc(i)
        if not predictions[pos]:  # only override if no positive rule fired
            confidences[pos] = 0.0

    result = pd.DataFrame({
        "rule_prediction": predictions,
        "rule_confidence": confidences,
        "rule_reasons": reasons,
    }, index=df.index)

    return result
