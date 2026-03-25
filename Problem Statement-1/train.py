"""Training pipeline: build features, train models, evaluate on val."""
import numpy as np
from data_loader import load_train, load_val
from features.build_features import build
from models.rule_based import predict as rule_predict
from models.gradient_boost import GradientBoostModel, tune_threshold
from models.ensemble import combine
from evaluation import evaluate, evaluate_by_outcome


def main():
    # ── Load data ──────────────────────────────────────────────────────
    print("Loading data...")
    train_df = load_train()
    val_df = load_val()

    y_train = train_df["has_ticket"].astype(bool)
    y_val = val_df["has_ticket"].astype(bool)

    # ── Build features ─────────────────────────────────────────────────
    print("Building features...")
    X_train = build(train_df)
    X_val = build(val_df)
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")

    # ── Layer 1: Rule-based ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LAYER 1: RULE-BASED")
    rule_train = rule_predict(train_df)
    rule_val = rule_predict(val_df)

    evaluate(y_train, rule_train["rule_prediction"], "Rules-Train")
    evaluate(y_val, rule_val["rule_prediction"], "Rules-Val")

    # Show which rules fired
    fired = rule_train[rule_train["rule_prediction"]]
    print(f"\n  Rules fired on {len(fired)} train calls")
    for _, row in fired.iterrows():
        if row["rule_reasons"]:
            pass  # avoid printing all 59

    # ── Layer 2: Gradient Boosting ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("LAYER 2: LIGHTGBM")
    gb = GradientBoostModel()
    gb.train(X_train, y_train, X_val, y_val)

    gb_proba_train = gb.predict_proba(X_train)
    gb_proba_val = gb.predict_proba(X_val)

    # Tune threshold on val
    best_t, best_f1 = tune_threshold(y_val, gb_proba_val)
    print(f"  Best threshold: {best_t}, Best val F1: {best_f1}")

    gb_pred_val = gb.predict(X_val, threshold=best_t)
    evaluate(y_train, gb.predict(X_train, threshold=best_t), "GB-Train")
    evaluate(y_val, gb_pred_val, "GB-Val")

    # Feature importance
    print("\n  Top 15 features:")
    fi = gb.feature_importance(X_train.columns.tolist())
    for _, row in fi.head(15).iterrows():
        print(f"    {row['feature']:<40s} {row['importance']:5.0f}")

    # ── Layer 3: Ensemble ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ENSEMBLE")

    # Tune ensemble threshold on val
    best_ens_t = best_t
    best_ens_f1 = 0.0
    for t in np.arange(0.10, 0.90, 0.01):
        ens_pred = combine(val_df, rule_val, gb_proba_val, threshold=t)
        f1 = np.mean(
            2 * (np.sum(ens_pred & y_val)) /
            max(np.sum(ens_pred) + np.sum(y_val), 1)
        )
        if f1 > best_ens_f1:
            best_ens_f1 = f1
            best_ens_t = t

    print(f"  Best ensemble threshold: {best_ens_t:.2f}")
    ens_pred_val = combine(val_df, rule_val, gb_proba_val, threshold=best_ens_t)
    evaluate(y_val, ens_pred_val, "Ensemble-Val")
    evaluate_by_outcome(y_val, ens_pred_val, val_df["outcome"])

    ens_pred_train = combine(train_df, rule_train, gb_proba_train, threshold=best_ens_t)
    evaluate(y_train, ens_pred_train, "Ensemble-Train")

    # ── Save model and threshold ───────────────────────────────────────
    gb.save()
    print(f"\n  Model saved. Ensemble threshold: {best_ens_t:.2f}")
    print(f"  Use this threshold in predict.py")

    return best_ens_t


if __name__ == "__main__":
    main()
