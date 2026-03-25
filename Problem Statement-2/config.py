"""Configuration for the TrimRX Voice Agent."""
import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI Realtime API ───────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
REALTIME_MODEL = "gpt-4o-realtime-preview"
REALTIME_URL = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"

# ── Audio Settings ─────────────────────────────────────────────────────
SAMPLE_RATE = 24000       # 24 kHz required by OpenAI Realtime
CHANNELS = 1              # Mono
CHUNK_DURATION_MS = 50    # 50ms chunks
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 1200 samples

# ── Voice Agent Settings ──────────────────────────────────────────────
AGENT_NAME = "Jessica"
COMPANY_NAME = "TrimRX"
VOICE = "shimmer"         # Female voice matching "Jessica"

# ── Default Patient Info (can be overridden at runtime) ───────────────
DEFAULT_PATIENT = {
    "name": "John Smith",
    "medication": "Tirzepatide",
    "dosage": "2.5mg weekly injection",
}
