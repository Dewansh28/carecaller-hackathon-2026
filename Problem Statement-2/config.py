"""Configuration for the TrimRX Voice Agent."""
import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI Realtime API ───────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
REALTIME_MODEL = "gpt-4o-realtime-preview"
REALTIME_URL = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"

# ── Twilio ────────────────────────────────────────────────────────────
TWILIO_ACCOUNT_SID  = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN   = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "")  # e.g. "+12345678901"
SERVER_URL          = os.environ.get("SERVER_URL", "")           # ngrok https URL

# ── Audio Settings ─────────────────────────────────────────────────────
SAMPLE_RATE = 24000       # 24 kHz — OpenAI Realtime output rate
TWILIO_RATE = 8000        # 8 kHz  — Twilio Media Streams rate
CHANNELS = 1              # Mono
CHUNK_DURATION_MS = 20    # 20ms chunks (lower latency than 50ms)
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 480 samples

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
