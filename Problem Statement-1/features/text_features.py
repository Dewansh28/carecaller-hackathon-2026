"""Extract features from validation_notes and transcript_text."""
import re
import pandas as pd
import numpy as np


# ── Validation notes keyword signals ───────────────────────────────────
# Each tuple: (feature_name, pattern, case_sensitive)
VN_SIGNALS = [
    ("vn_dosage_guidance", r"dosage guidance", False),
    ("vn_not_right_now", r"not right now", False),
    ("vn_erroneously", r"erroneously", False),
    ("vn_fabricated", r"fabricated", False),
    ("vn_outcome_corrected", r"corrected by validation", False),
    ("vn_weight_differs", r"differs between sources", False),
    ("vn_possible_miscategorization", r"possible miscategorization", False),
    ("vn_inconsistency", r"inconsisten", False),
    ("vn_stop_calling", r"stop calling", False),
    ("vn_wrong_person", r"wrong person|not me|not the patient", False),
    ("vn_interested_but", r"interested but|interest but", False),
    ("vn_whisper_mismatch", r"whisper.*(?:differs|mismatch|discrepan)", False),
    ("vn_skipped_questions", r"questions? (?:were|was) not asked|not asked", False),
    ("vn_medical_advice", r"medical advice|clinical recommendation", False),
]

# ── Transcript agent-turn patterns (medical advice detection) ──────────
AGENT_ADVICE_PATTERNS = [
    r"\brecommend\b",
    r"\bsuggest(?:ing|ed|s)?\b",
    r"\btry taking\b",
    r"\byou (?:could|should|might)(?: want to)? (?:try|take|consider)\b",
    r"\bvitamin\b",
    r"\bsupplement\b",
    r"\bincrease.*(?:dose|dosage)\b",
    r"\bdecrease.*(?:dose|dosage)\b",
]

# ── Transcript user-turn patterns (outcome consistency) ────────────────
USER_OUTCOME_PATTERNS = {
    "user_says_not_interested": r"\bnot interested\b|\bno thanks\b|\bdon'?t want\b",
    "user_says_wrong_number": r"\bwrong number\b|\bnot me\b|\bwrong person\b",
    "user_says_reschedule": r"\breschedule\b|\bcall (?:me )?back\b|\banother time\b",
    "user_confirmed_identity": r"\bthat'?s me\b|\byes.*speaking\b|\bthis is\b",
}


def _safe_str(val) -> str:
    if pd.isna(val):
        return ""
    return str(val)


def _extract_agent_turns(transcript) -> list[str]:
    """Extract agent messages from the transcript list."""
    if not isinstance(transcript, list):
        return []
    return [t["message"] for t in transcript if t.get("role") == "agent"]


def _extract_user_turns(transcript) -> list[str]:
    """Extract user messages from the transcript list."""
    if not isinstance(transcript, list):
        return []
    return [t["message"] for t in transcript if t.get("role") == "user"]


def _extract_question_count_from_vn(vn: str) -> float:
    """Extract the 'X of 14 questions' count from validation_notes."""
    m = re.search(r"(\d+)\s+of\s+14\s+(?:questionnaire\s+)?questions?\s+were\s+asked", vn, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+)\s+of\s+14\s+(?:mapped\s+)?questions?\s+were\s+asked", vn, re.I)
    if m:
        return float(m.group(1))
    # "All 14 questions were asked"
    if re.search(r"all\s+14\s+(?:mapped\s+)?questions?\s+were\s+asked", vn, re.I):
        return 14.0
    return np.nan


def extract(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of text-derived features."""
    out = pd.DataFrame(index=df.index)

    vn_series = df["validation_notes"].apply(_safe_str)

    # ── Validation notes keyword signals ───────────────────────────────
    for feat_name, pattern, _ in VN_SIGNALS:
        out[feat_name] = vn_series.str.contains(pattern, case=False, regex=True).astype(int)

    # ── Validation notes question count discrepancy ────────────────────
    out["vn_question_count"] = vn_series.apply(_extract_question_count_from_vn)
    out["vn_answered_discrepancy"] = df["answered_count"] - out["vn_question_count"]
    out["vn_answered_discrepancy_abs"] = out["vn_answered_discrepancy"].abs()

    # ── Sum of all VN signal flags ─────────────────────────────────────
    signal_cols = [name for name, _, _ in VN_SIGNALS]
    out["vn_signal_count"] = out[signal_cols].sum(axis=1)

    # ── Validation notes length ────────────────────────────────────────
    out["vn_length"] = vn_series.str.len()

    # ── Transcript: agent medical advice detection ─────────────────────
    agent_turns_list = df["transcript"].apply(_extract_agent_turns)
    advice_counts = []
    for turns in agent_turns_list:
        agent_text = " ".join(turns).lower()
        count = sum(1 for p in AGENT_ADVICE_PATTERNS if re.search(p, agent_text))
        advice_counts.append(count)
    out["agent_advice_score"] = advice_counts

    # ── Transcript: user outcome consistency signals ───────────────────
    user_turns_list = df["transcript"].apply(_extract_user_turns)
    for feat_name, pattern in USER_OUTCOME_PATTERNS.items():
        flags = []
        for turns in user_turns_list:
            user_text = " ".join(turns).lower()
            flags.append(1 if re.search(pattern, user_text) else 0)
        out[feat_name] = flags

    # ── Transcript length ──────────────────────────────────────────────
    out["transcript_char_len"] = df["transcript_text"].apply(lambda x: len(_safe_str(x)))

    # Fill NaN
    out = out.fillna(0)
    return out
