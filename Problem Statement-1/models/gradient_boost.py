"""Layer 2: LightGBM gradient boosting classifier."""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from config import LGBM_PARAMS, MODEL_FILE


class GradientBoostModel:
    """Wrapper around LightGBM for the ticket prediction task."""

    def __init__(self, params: dict | None = None):
        self.params = params or LGBM_PARAMS.copy()
        self.model: lgb.LGBMClassifier | None = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "GradientBoostModel":
        """Train the model with optional early stopping on validation set."""
        self.model = lgb.LGBMClassifier(**self.params)

        callbacks = []
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            callbacks = [lgb.early_stopping(50, verbose=False)]

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks,
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of has_ticket=True for each row."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions at the given threshold."""
        return (self.predict_proba(X) >= threshold).astype(bool)

    def feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        """Return feature importances sorted descending."""
        importances = self.model.feature_importances_
        fi = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)
        return fi

    def save(self, path=MODEL_FILE):
        """Save the model to disk."""
        self.model.booster_.save_model(str(path))

    def load(self, path=MODEL_FILE) -> "GradientBoostModel":
        """Load a saved model."""
        booster = lgb.Booster(model_file=str(path))
        self.model = lgb.LGBMClassifier(**self.params)
        self.model._Booster = booster
        self.model.fitted_ = True
        self.model._n_features = booster.num_feature()
        self.model._n_features_in = booster.num_feature()
        self.model.classes_ = np.array([False, True])
        return self


def tune_threshold(
    y_true: pd.Series,
    y_proba: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """Sweep thresholds and return (best_threshold, best_f1)."""
    if thresholds is None:
        thresholds = np.arange(0.10, 0.90, 0.01)

    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return round(best_t, 2), round(best_f1, 4)
