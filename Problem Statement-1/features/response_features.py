"""Extract features from the structured Q&A responses."""
import re
import pandas as pd
import numpy as np


def _safe_responses(resp) -> list[dict]:
    """Ensure responses is a list of dicts."""
    if isinstance(resp, list):
        return resp
    return []


def _get_answer(responses: list[dict], idx: int) -> str:
    """Get the answer at a given index, empty string if missing."""
    if idx < len(responses):
        return str(responses[idx].get("answer", ""))
    return ""


def _extract_weight(answer: str) -> float:
    """Extract numeric weight from an answer string."""
    m = re.search(r"(\d+(?:\.\d+)?)", answer)
    if m:
        return float(m.group(1))
    return np.nan


def extract(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of response-derived features."""
    out = pd.DataFrame(index=df.index)

    responses_list = df["responses"].apply(_safe_responses)

    # ── Basic counts ───────────────────────────────────────────────────
    out["num_empty_answers"] = responses_list.apply(
        lambda rs: sum(1 for r in rs if not str(r.get("answer", "")).strip())
    )
    out["num_answered"] = responses_list.apply(
        lambda rs: sum(1 for r in rs if str(r.get("answer", "")).strip())
    )

    # ── Weight analysis (Q2, index 1) ──────────────────────────────────
    weight_answers = responses_list.apply(lambda rs: _get_answer(rs, 1))
    out["weight_value"] = weight_answers.apply(_extract_weight)
    out["weight_is_outlier"] = (
        (out["weight_value"] < 80) | (out["weight_value"] > 500)
    ).astype(int)
    out["weight_is_outlier"] = out["weight_is_outlier"].fillna(0).astype(int)

    # ── Average answer length ──────────────────────────────────────────
    out["avg_answer_length"] = responses_list.apply(
        lambda rs: np.mean([len(str(r.get("answer", ""))) for r in rs]) if rs else 0
    )

    # ── Specific health signals ────────────────────────────────────────
    # Q5 (idx 4): side effects
    out["has_side_effects"] = responses_list.apply(
        lambda rs: int(bool(
            _get_answer(rs, 4).strip()
            and _get_answer(rs, 4).strip().lower() not in ("no", "none", "no side effects", "")
        ))
    )
    # Q9 (idx 8): new medications
    out["has_new_medications"] = responses_list.apply(
        lambda rs: int(bool(
            _get_answer(rs, 8).strip()
            and _get_answer(rs, 8).strip().lower() not in ("no", "none", "")
        ))
    )
    # Q10 (idx 9): new conditions
    out["has_new_conditions"] = responses_list.apply(
        lambda rs: int(bool(
            _get_answer(rs, 9).strip()
            and _get_answer(rs, 9).strip().lower() not in ("no", "none", "")
        ))
    )
    # Q11 (idx 10): new allergies
    out["has_new_allergies"] = responses_list.apply(
        lambda rs: int(bool(
            _get_answer(rs, 10).strip()
            and _get_answer(rs, 10).strip().lower() not in ("no", "none", "")
        ))
    )
    # Q12 (idx 11): surgeries
    out["has_surgeries"] = responses_list.apply(
        lambda rs: int(bool(
            _get_answer(rs, 11).strip()
            and _get_answer(rs, 11).strip().lower() not in ("no", "none", "")
        ))
    )

    # ── Outcome-response consistency ───────────────────────────────────
    out["completed_but_few_answers"] = (
        (df["outcome"] == "completed") & (df["answered_count"] < 14)
    ).astype(int)
    out["wrong_number_with_answers"] = (
        (df["outcome"] == "wrong_number") & (df["answered_count"] > 0)
    ).astype(int)
    out["incomplete_but_fully_answered"] = (
        (df["outcome"] == "incomplete") & (df["response_completeness"] >= 0.9)
    ).astype(int)

    out = out.fillna(0)
    return out
