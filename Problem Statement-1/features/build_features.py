"""Orchestrate all feature extractors into a single feature matrix."""
import pandas as pd
from features import metadata_features, text_features, response_features


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Build the complete feature matrix for a DataFrame of calls.

    Returns a numeric DataFrame ready for model input (no string columns).
    """
    parts = [
        metadata_features.extract(df),
        text_features.extract(df),
        response_features.extract(df),
    ]
    features = pd.concat(parts, axis=1)
    features = features.fillna(0)
    return features


def get_feature_names(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names."""
    return build(df.head(1)).columns.tolist()
