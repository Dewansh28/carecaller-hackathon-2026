"""Load and prepare hackathon datasets."""
import json
import pandas as pd
from config import TRAIN_FILE, VAL_FILE, TEST_FILE


def load_split(path) -> pd.DataFrame:
    """Load a single JSON split into a DataFrame.

    Keeps 'transcript' (list of dicts) and 'responses' (list of dicts)
    as native Python objects for downstream feature extraction.
    """
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data["calls"])
    return df


def load_train() -> pd.DataFrame:
    return load_split(TRAIN_FILE)


def load_val() -> pd.DataFrame:
    return load_split(VAL_FILE)


def load_test() -> pd.DataFrame:
    return load_split(TEST_FILE)


def load_train_val() -> pd.DataFrame:
    """Combined train + val for final model training."""
    return pd.concat([load_train(), load_val()], ignore_index=True)


if __name__ == "__main__":
    for name, loader in [("train", load_train), ("val", load_val), ("test", load_test)]:
        df = loader()
        print(f"{name}: {df.shape[0]} rows, {df.shape[1]} cols")
        if "has_ticket" in df.columns:
            print(f"  tickets: {df['has_ticket'].sum()}")
