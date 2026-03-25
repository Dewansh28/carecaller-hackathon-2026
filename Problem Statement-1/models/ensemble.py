"""Ensemble: combine rule-based and gradient boosting predictions."""
import numpy as np
import pandas as pd
from config import SAFE_OUTCOMES


def combine(
    df: pd.DataFrame,
    rule_results: pd.DataFrame,
    gb_proba: np.ndarray,
    threshold: float = 0.35,
) -> np.ndarray:
    """Cascading ensemble logic.

    Priority:
    1. Hard rule override (confidence >= 0.95) → use rule prediction
    2. Safe outcomes (scheduled/voicemail) → predict False
    3. Weighted combination of rule confidence + GB probability

    Args:
        df: original call data (needs 'outcome' column)
        rule_results: output from rule_based.predict()
        gb_proba: probability array from gradient boost model
        threshold: decision threshold for the final prediction

    Returns:
        Boolean array of final predictions.
    """
    n = len(df)
    final_proba = np.zeros(n, dtype=float)

    rule_conf = rule_results["rule_confidence"].values
    rule_pred = rule_results["rule_prediction"].values
    outcomes = df["outcome"].values

    for i in range(n):
        # 1. Hard rule override
        if rule_pred[i] and rule_conf[i] >= 0.95:
            final_proba[i] = rule_conf[i]
            continue

        # 2. Safe outcome → predict False
        if outcomes[i] in SAFE_OUTCOMES:
            final_proba[i] = 0.0
            continue

        # 3. Weighted combination
        # Give more weight to GB, but boost if rules triggered
        rule_signal = rule_conf[i]
        gb_signal = gb_proba[i]

        if rule_signal > 0:
            # Rule fired but below override threshold — blend
            final_proba[i] = 0.4 * rule_signal + 0.6 * gb_signal
        else:
            # No rule fired — rely on GB
            final_proba[i] = gb_signal

    return final_proba >= threshold
