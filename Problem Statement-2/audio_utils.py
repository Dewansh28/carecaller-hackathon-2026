"""Audio format conversion between Twilio and OpenAI Realtime."""
import audioop
from config import SAMPLE_RATE, TWILIO_RATE

# Persistent state for audioop.ratecv (reduces clicking at chunk boundaries)
_to_openai_state = None
_to_twilio_state = None


def mulaw_to_pcm16(mulaw_bytes: bytes) -> bytes:
    """Twilio → OpenAI: μ-law 8kHz → PCM16 24kHz."""
    global _to_openai_state
    # Step 1: μ-law → PCM16 at 8kHz
    pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
    # Step 2: 8kHz → 24kHz (with stateful resampler to avoid boundary clicks)
    pcm_24k, _to_openai_state = audioop.ratecv(
        pcm_8k, 2, 1, TWILIO_RATE, SAMPLE_RATE, _to_openai_state
    )
    return pcm_24k


def pcm16_to_mulaw(pcm_24k_bytes: bytes) -> bytes:
    """OpenAI → Twilio: PCM16 24kHz → μ-law 8kHz."""
    global _to_twilio_state
    # Step 1: 24kHz → 8kHz (stateful)
    pcm_8k, _to_twilio_state = audioop.ratecv(
        pcm_24k_bytes, 2, 1, SAMPLE_RATE, TWILIO_RATE, _to_twilio_state
    )
    # Step 2: PCM16 → μ-law
    return audioop.lin2ulaw(pcm_8k, 2)
