"""Generate submission CSV for the test set."""
import sys
import pandas as pd
from data_loader import load_test, load_train, load_val
from features.build_features import build
from models.rule_based import predict as rule_predict
from models.gradient_boost import GradientBoostModel
from models.ensemble import combine
from config import SUBMISSION_FILE, MODEL_FILE


def main(threshold: float = 0.35):
    # ── Retrain on train+val for final submission ──────────────────────
    print("Loading data...")
    train_df = load_train()
    val_df = load_val()
    test_df = load_test()

    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    y_combined = combined_df["has_ticket"].astype(bool)

    print("Building features...")
    X_combined = build(combined_df)
    X_test = build(test_df)

    # ── Train final model on all labeled data ──────────────────────────
    print("Training final model on train+val...")
    gb = GradientBoostModel()
    gb.train(X_combined, y_combined)

    # ── Predict on test ────────────────────────────────────────────────
    rule_test = rule_predict(test_df)
    gb_proba_test = gb.predict_proba(X_test)

    predictions = combine(test_df, rule_test, gb_proba_test, threshold=threshold)

    # ── Build submission ───────────────────────────────────────────────
    submission = pd.DataFrame({
        "call_id": test_df["call_id"],
        "predicted_ticket": predictions,
    })

    submission.to_csv(SUBMISSION_FILE, index=False)
    print(f"\nSubmission saved to {SUBMISSION_FILE}")
    print(f"  Total predictions: {len(submission)}")
    print(f"  Predicted tickets: {submission['predicted_ticket'].sum()}")
    print(f"  Ticket rate: {submission['predicted_ticket'].mean():.3f}")

    # Show rule-flagged calls
    flagged = rule_test[rule_test["rule_prediction"]]
    print(f"  Rule-flagged: {len(flagged)}")


if __name__ == "__main__":
    t = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35
    main(threshold=t)
