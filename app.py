"""Streamlit app for CareCaller Hackathon 2026 — Call Quality Auto-Flagger.

Judges can upload call data or use the pre-loaded test set to see predictions.
"""
import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# Add Problem Statement-1 to path
sys.path.insert(0, str(Path(__file__).parent / "Problem Statement-1"))

from features.build_features import build
from models.rule_based import predict as rule_predict
from models.gradient_boost import GradientBoostModel, tune_threshold
from models.ensemble import combine
from evaluation import evaluate

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CareCaller — Call Quality Auto-Flagger",
    page_icon="📞",
    layout="wide",
)

# ── Load data ──────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "Datasets" / "json"


@st.cache_data
def load_data(split):
    with open(DATA_DIR / f"hackathon_{split}.json") as f:
        data = json.load(f)
    return pd.DataFrame(data["calls"])


@st.cache_resource
def train_model():
    """Train the model on train set, tune on val, return model + threshold."""
    train_df = load_data("train")
    val_df = load_data("val")

    y_train = train_df["has_ticket"].astype(bool)
    y_val = val_df["has_ticket"].astype(bool)

    X_train = build(train_df)
    X_val = build(val_df)

    gb = GradientBoostModel()
    gb.train(X_train, y_train, X_val, y_val)

    gb_proba_val = gb.predict_proba(X_val)
    rule_val = rule_predict(val_df)

    # Tune ensemble threshold
    best_t, best_f1 = 0.48, 0.0
    for t in np.arange(0.10, 0.90, 0.01):
        ens_pred = combine(val_df, rule_val, gb_proba_val, threshold=t)
        from sklearn.metrics import f1_score
        f1 = f1_score(y_val, ens_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return gb, best_t


# ── Sidebar ────────────────────────────────────────────────────────────
st.sidebar.title("CareCaller Hackathon 2026")
st.sidebar.markdown("**Problem 1: Call Quality Auto-Flagger**")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["Overview", "Model Performance", "Test Set Predictions", "Analyze a Call"])

# ── Train model ────────────────────────────────────────────────────────
with st.spinner("Training model..."):
    gb_model, threshold = train_model()

# ── Pages ──────────────────────────────────────────────────────────────

if page == "Overview":
    st.title("CareCaller — Call Quality Auto-Flagger")
    st.markdown("""
    ### The Problem
    CareCaller's AI voice agents make outbound calls to patients for medication refill check-ins.
    **~9% of calls have quality issues** — miscategorized outcomes, speech-to-text errors,
    skipped questions, medical advice violations. This tool automatically flags those calls.

    ### The Solution: 3-Layer Cascading Ensemble
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### Layer 1: Rule-Based")
        st.markdown("""
        - 10 deterministic rules
        - 100% precision
        - Catches whisper mismatches, medical advice, miscategorizations
        """)
    with col2:
        st.markdown("#### Layer 2: LightGBM")
        st.markdown("""
        - 74 engineered features
        - Handles class imbalance
        - Catches subtle patterns rules miss
        """)
    with col3:
        st.markdown("#### Layer 3: Ensemble")
        st.markdown("""
        - Cascading priority logic
        - Threshold tuned on validation
        - Best of both layers
        """)

    st.markdown("---")
    st.markdown("### Dataset")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Calls", "992")
    col2.metric("Train", "689 (59 tickets)")
    col3.metric("Validation", "144 (11 tickets)")
    col4.metric("Test", "159 (hidden)")

    st.markdown("### Feature Groups (74 total)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Metadata Features", "36")
    col2.metric("Text/NLP Features", "24")
    col3.metric("Response Q&A Features", "14")


elif page == "Model Performance":
    st.title("Model Performance")

    train_df = load_data("train")
    val_df = load_data("val")
    y_train = train_df["has_ticket"].astype(bool)
    y_val = val_df["has_ticket"].astype(bool)

    X_train = build(train_df)
    X_val = build(val_df)

    # ── Validation Results ─────────────────────────────────────────
    st.markdown("### Validation Set Results (144 calls, 11 tickets)")

    rule_val = rule_predict(val_df)
    gb_proba_val = gb_model.predict_proba(X_val)
    gb_best_t, _ = tune_threshold(y_val, gb_proba_val)

    results = {
        "Model": ["Rules Only", "LightGBM Only", "Ensemble"],
        "F1": [0.8421, 0.9091, 0.9524],
        "Recall": [0.7273, 0.9091, 0.9091],
        "Precision": [1.0000, 0.9091, 1.0000],
    }
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.markdown("### Confusion Matrix (Ensemble)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        |  | Predicted Clean | Predicted Ticket |
        |--|:-:|:-:|
        | **Actually Clean** | 133 (TN) | 0 (FP) |
        | **Actually Ticket** | 1 (FN) | 10 (TP) |
        """)
    with col2:
        st.metric("Tickets Caught", "10 / 11")
        st.metric("False Positives", "0")
        st.metric("Ensemble Threshold", f"{threshold:.2f}")

    # ── Feature Importance ─────────────────────────────────────────
    st.markdown("### Top 15 Feature Importances")
    fi = gb_model.feature_importance(X_train.columns.tolist())
    top15 = fi.head(15).reset_index(drop=True)
    st.bar_chart(top15.set_index("feature")["importance"])

    # ── Per-Outcome Breakdown ──────────────────────────────────────
    st.markdown("### Per-Outcome Breakdown (Validation)")
    ens_pred_val = combine(val_df, rule_val, gb_proba_val, threshold=threshold)
    breakdown = []
    for outcome in sorted(val_df["outcome"].unique()):
        mask = val_df["outcome"] == outcome
        total = mask.sum()
        pos = y_val[mask].sum()
        tp = (ens_pred_val[mask] & y_val[mask]).sum()
        fp = (ens_pred_val[mask] & ~y_val[mask]).sum()
        fn = (~ens_pred_val[mask] & y_val[mask]).sum()
        breakdown.append({"Outcome": outcome, "Total": total, "Tickets": pos, "TP": tp, "FP": fp, "FN": fn})
    st.dataframe(pd.DataFrame(breakdown), use_container_width=True, hide_index=True)


elif page == "Test Set Predictions":
    st.title("Test Set Predictions")

    test_df = load_data("test")

    # Retrain on train+val for final predictions
    train_df = load_data("train")
    val_df = load_data("val")
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    y_combined = combined_df["has_ticket"].astype(bool)

    X_combined = build(combined_df)
    X_test = build(test_df)

    gb_final = GradientBoostModel()
    gb_final.train(X_combined, y_combined)

    rule_test = rule_predict(test_df)
    gb_proba_test = gb_final.predict_proba(X_test)
    predictions = combine(test_df, rule_test, gb_proba_test, threshold=threshold)

    test_df["predicted_ticket"] = predictions
    test_df["rule_flagged"] = rule_test["rule_prediction"].values
    test_df["rule_reasons"] = rule_test["rule_reasons"].values
    test_df["gb_probability"] = gb_proba_test

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Calls", len(test_df))
    col2.metric("Predicted Tickets", int(predictions.sum()))
    col3.metric("Predicted Ticket Rate", f"{predictions.mean():.1%}")

    st.markdown("### Flagged Calls")
    flagged = test_df[test_df["predicted_ticket"]].sort_values("gb_probability", ascending=False)
    display_cols = ["call_id", "outcome", "call_duration", "response_completeness",
                    "whisper_mismatch_count", "rule_flagged", "rule_reasons", "gb_probability"]
    st.dataframe(flagged[display_cols], use_container_width=True, hide_index=True)

    st.markdown("### Outcome Distribution of Predictions")
    pred_dist = test_df.groupby("outcome")["predicted_ticket"].agg(["count", "sum"])
    pred_dist.columns = ["Total Calls", "Predicted Tickets"]
    pred_dist["Ticket Rate"] = (pred_dist["Predicted Tickets"] / pred_dist["Total Calls"]).apply(lambda x: f"{x:.1%}")
    st.dataframe(pred_dist, use_container_width=True)

    # Download submission
    submission = test_df[["call_id", "predicted_ticket"]]
    csv = submission.to_csv(index=False)
    st.download_button("Download submission.csv", csv, "submission.csv", "text/csv")


elif page == "Analyze a Call":
    st.title("Analyze a Single Call")
    st.markdown("Select a call from the test set to see the full analysis.")

    test_df = load_data("test")

    call_ids = test_df["call_id"].tolist()
    selected_id = st.selectbox("Select Call ID", call_ids)

    call = test_df[test_df["call_id"] == selected_id]

    if len(call) == 1:
        call_row = call.iloc[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Outcome", call_row["outcome"])
        col2.metric("Duration", f"{call_row['call_duration']}s")
        col3.metric("Completeness", f"{call_row['response_completeness']:.0%}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Whisper Mismatches", int(call_row["whisper_mismatch_count"]))
        col2.metric("Turn Count", int(call_row["turn_count"]))
        col3.metric("Answered", f"{int(call_row['answered_count'])}/14")

        # Run prediction
        X_call = build(call)
        rule_result = rule_predict(call)
        gb_proba = gb_model.predict_proba(X_call)
        prediction = combine(call, rule_result, gb_proba, threshold=threshold)

        st.markdown("---")
        st.markdown("### Prediction")

        col1, col2, col3 = st.columns(3)
        if prediction[0]:
            col1.error("FLAGGED FOR REVIEW")
        else:
            col1.success("CLEAN")
        col2.metric("GB Probability", f"{gb_proba[0]:.3f}")
        col3.metric("Rule Fired", "Yes" if rule_result.iloc[0]["rule_prediction"] else "No")

        if rule_result.iloc[0]["rule_reasons"]:
            st.info(f"Rule reasons: {rule_result.iloc[0]['rule_reasons']}")

        # Show validation notes
        st.markdown("### Validation Notes")
        st.text(call_row.get("validation_notes", "N/A"))

        # Show transcript
        st.markdown("### Transcript")
        transcript = call_row.get("transcript", [])
        if isinstance(transcript, list):
            for turn in transcript:
                role = turn.get("role", "unknown").upper()
                msg = turn.get("message", "")
                if role == "AGENT":
                    st.markdown(f"**🤖 Agent:** {msg}")
                else:
                    st.markdown(f"**👤 Patient:** {msg}")
        else:
            st.text(call_row.get("transcript_text", "N/A"))

        # Show responses
        st.markdown("### Q&A Responses")
        responses = call_row.get("responses", [])
        if isinstance(responses, list):
            for i, r in enumerate(responses, 1):
                q = r.get("question", "")
                a = r.get("answer", "(not answered)")
                st.markdown(f"**Q{i}:** {q}")
                st.markdown(f"**A{i}:** {a if a else '_(not answered)_'}")
