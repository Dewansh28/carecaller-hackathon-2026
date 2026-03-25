"""Configuration: paths, constants, and hyperparameters."""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR.parent / "Datasets"
JSON_DIR = DATA_DIR / "json"
CSV_DIR = DATA_DIR / "csv"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_FILE = JSON_DIR / "hackathon_train.json"
VAL_FILE = JSON_DIR / "hackathon_val.json"
TEST_FILE = JSON_DIR / "hackathon_test.json"

SUBMISSION_FILE = OUTPUT_DIR / "submission.csv"
MODEL_FILE = OUTPUT_DIR / "lgbm_model.txt"

# ── Dataset Constants ──────────────────────────────────────────────────
NUM_QUESTIONS = 14
SAFE_OUTCOMES = {"scheduled", "voicemail"}
ALL_OUTCOMES = [
    "completed", "incomplete", "opted_out", "scheduled",
    "escalated", "wrong_number", "voicemail",
]
DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# ── LightGBM Hyperparameters ──────────────────────────────────────────
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "min_child_samples": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 10.6,
    "random_state": 42,
}

# ── Ensemble ───────────────────────────────────────────────────────────
RULE_OVERRIDE_CONFIDENCE = 0.95
ENSEMBLE_WEIGHTS = {"rule": 0.3, "gb": 0.4, "llm": 0.3}
DEFAULT_THRESHOLD = 0.35  # tuned on val set
