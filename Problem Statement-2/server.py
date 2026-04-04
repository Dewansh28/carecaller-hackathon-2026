"""
FastAPI server — bridges Twilio Media Streams ↔ OpenAI Realtime API.

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000

Expose publicly:
    ngrok http 8000
    export SERVER_URL=https://xxxx.ngrok-free.app

Make a call:
    python3 call.py --to "+91XXXXXXXXXX" --name "Dewansh"
"""
import asyncio
import base64
import json
from urllib.parse import quote

import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import Response, JSONResponse
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream

from config import (
    OPENAI_API_KEY, REALTIME_URL, VOICE,
    TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER,
    SERVER_URL, DEFAULT_PATIENT,
)
from audio_utils import mulaw_to_pcm16, pcm16_to_mulaw
from prompts import build_system_prompt
from questionnaire import CallRecord, QUESTIONS

app = FastAPI()
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


# ── Outbound call ──────────────────────────────────────────────────────

@app.post("/make-call")
async def make_call(request: Request):
    """Initiate an outbound call via Twilio."""
    body         = await request.json()
    to_number    = body["to"]
    patient_name = body.get("name",       DEFAULT_PATIENT["name"])
    medication   = body.get("medication", DEFAULT_PATIENT["medication"])
    dosage       = body.get("dosage",     DEFAULT_PATIENT["dosage"])

    twiml_url = (
        f"{SERVER_URL}/twiml"
        f"?name={quote(patient_name)}"
        f"&medication={quote(medication)}"
        f"&dosage={quote(dosage)}"
    )
    call = twilio_client.calls.create(
        to=to_number,
        from_=TWILIO_PHONE_NUMBER,
        url=twiml_url,
        method="GET",
    )
    print(f"[Server] Call initiated → {to_number}  SID: {call.sid}")
    return JSONResponse({"call_sid": call.sid, "status": call.status})


# ── TwiML webhook ──────────────────────────────────────────────────────

@app.get("/twiml")
async def twiml(
    name:       str = DEFAULT_PATIENT["name"],
    medication: str = DEFAULT_PATIENT["medication"],
    dosage:     str = DEFAULT_PATIENT["dosage"],
):
    """Return TwiML that opens a Media Stream to our WebSocket."""
    ws_url = SERVER_URL.replace("https://", "wss://").replace("http://", "ws://")

    response = VoiceResponse()
    connect  = Connect()
    stream   = Stream(url=f"{ws_url}/audio-stream")
    stream.parameter(name="patient_name", value=name)
    stream.parameter(name="medication",   value=medication)
    stream.parameter(name="dosage",       value=dosage)
    connect.append(stream)
    response.append(connect)

    return Response(content=str(response), media_type="text/xml")


# ── WebSocket bridge ───────────────────────────────────────────────────

@app.websocket("/audio-stream")
async def audio_stream(twilio_ws: WebSocket):
    """Bridge between Twilio Media Stream and OpenAI Realtime."""
    await twilio_ws.accept()

    call_record = None
    openai_ws   = None

    try:
        # ── Wait for Twilio "start" event to get call metadata ─────────
        patient_name = DEFAULT_PATIENT["name"]
        medication   = DEFAULT_PATIENT["medication"]
        dosage       = DEFAULT_PATIENT["dosage"]
        stream_sid   = None
        call_sid     = None

        while True:
            raw  = await twilio_ws.receive_text()
            data = json.loads(raw)
            if data["event"] == "start":
                stream_sid = data["start"]["streamSid"]
                call_sid   = data["start"].get("callSid")
                params     = data["start"].get("customParameters", {})
                patient_name = params.get("patient_name", patient_name)
                medication   = params.get("medication",   medication)
                dosage       = params.get("dosage",       dosage)
                break
            if data["event"] == "stop":
                return

        print(f"\n[Stream] Connected — Patient: {patient_name} | CallSID: {call_sid}")
        call_record = CallRecord(patient_name, medication, dosage)

        # ── Connect to OpenAI Realtime ─────────────────────────────────
        openai_ws = await websockets.connect(
            REALTIME_URL,
            additional_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta":   "realtime=v1",
            },
            ping_interval=20,
            ping_timeout=20,
        )

        event = json.loads(await openai_ws.recv())
        if event["type"] != "session.created":
            print(f"[OpenAI] Unexpected first event: {event['type']}")

        # Configure session — tighter VAD for snappier responses
        await openai_ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": build_system_prompt(patient_name, medication, dosage),
                "voice": VOICE,
                "input_audio_format":  "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {
                    "type":                "server_vad",
                    "threshold":           0.4,   # sensitive enough for phone audio
                    "prefix_padding_ms":   100,   # start capturing 100ms before speech
                    "silence_duration_ms": 400,   # respond after 400ms silence (was 600)
                },
                "tools":       _tools(),
                "tool_choice": "auto",
            },
        }))
        await openai_ws.recv()  # session.updated confirmation

        # Send initial trigger
        await openai_ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message", "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": "[System: Call connected. Start with your standard greeting.]",
                }],
            },
        }))
        await openai_ws.send(json.dumps({"type": "response.create"}))
        print("[Stream] Bridge active — Jessica is live.\n")

        # ── Shared stop signal between the two bridge tasks ────────────
        stop_event = asyncio.Event()

        task_in  = asyncio.create_task(
            _twilio_to_openai(twilio_ws, openai_ws, stop_event)
        )
        task_out = asyncio.create_task(
            _openai_to_twilio(openai_ws, twilio_ws, stream_sid, call_sid, call_record, stop_event)
        )

        # Wait for whichever finishes first, then cancel the other
        done, pending = await asyncio.wait(
            [task_in, task_out],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

    except Exception as e:
        print(f"[Stream] Error: {type(e).__name__}: {e}")
    finally:
        if openai_ws:
            await openai_ws.close()
        if call_record:
            print("\n" + "=" * 60)
            print("CALL SUMMARY")
            print("=" * 60)
            print(call_record.summary())
            print("=" * 60)
            print("STRUCTURED JSON")
            print("=" * 60)
            print(call_record.to_json())


# ── Bridge tasks ───────────────────────────────────────────────────────

async def _twilio_to_openai(
    twilio_ws: WebSocket,
    openai_ws,
    stop_event: asyncio.Event,
):
    """Patient audio: Twilio mulaw 8kHz → OpenAI PCM16 24kHz."""
    try:
        while not stop_event.is_set():
            try:
                raw = await asyncio.wait_for(twilio_ws.receive_text(), timeout=1.0)
            except asyncio.TimeoutError:
                continue  # check stop_event and retry

            data = json.loads(raw)

            if data["event"] == "media":
                mulaw  = base64.b64decode(data["media"]["payload"])
                pcm24k = mulaw_to_pcm16(mulaw)
                await openai_ws.send(json.dumps({
                    "type":  "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm24k).decode(),
                }))

            elif data["event"] == "stop":
                print("[Stream] Twilio stream stopped.")
                stop_event.set()
                break

    except Exception as e:
        if not stop_event.is_set():
            print(f"[Twilio→OpenAI] {type(e).__name__}: {e}")
        stop_event.set()


async def _openai_to_twilio(
    openai_ws,
    twilio_ws: WebSocket,
    stream_sid: str,
    call_sid:   str,
    call_record: CallRecord,
    stop_event: asyncio.Event,
):
    """Jessica audio: OpenAI PCM16 24kHz → Twilio mulaw 8kHz + function calls."""
    call_ending       = False
    current_call_id   = None
    current_func_name = None
    current_func_args = ""

    try:
        async for message in openai_ws:
            if stop_event.is_set():
                break

            event      = json.loads(message)
            event_type = event.get("type", "")

            # ── Jessica audio → Twilio ─────────────────────────────────
            if event_type == "response.audio.delta":
                pcm24k = base64.b64decode(event["delta"])
                mulaw  = pcm16_to_mulaw(pcm24k)
                await twilio_ws.send_text(json.dumps({
                    "event":     "media",
                    "streamSid": stream_sid,
                    "media":     {"payload": base64.b64encode(mulaw).decode()},
                }))

            elif event_type == "response.audio.done":
                if call_ending:
                    # Give Twilio ~2s to finish playing the goodbye
                    await asyncio.sleep(2.0)
                    # Explicitly hang up via Twilio REST API
                    await _hangup(call_sid)
                    stop_event.set()
                    break

            # ── Live transcripts ───────────────────────────────────────
            elif event_type == "conversation.item.input_audio_transcription.completed":
                t = event.get("transcript", "").strip()
                if t:
                    print(f"  [Patient ]: {t}")

            elif event_type == "response.audio_transcript.done":
                t = event.get("transcript", "").strip()
                if t:
                    print(f"  [Jessica ]: {t}")

            # ── Patient interrupts → clear Twilio buffer ───────────────
            elif event_type == "input_audio_buffer.speech_started":
                await twilio_ws.send_text(json.dumps({
                    "event":     "clear",
                    "streamSid": stream_sid,
                }))

            # ── Function calls ─────────────────────────────────────────
            elif event_type == "response.output_item.added":
                item = event.get("item", {})
                if item.get("type") == "function_call":
                    current_call_id   = item.get("call_id")
                    current_func_name = item.get("name")
                    current_func_args = ""

            elif event_type == "response.function_call_arguments.delta":
                current_func_args += event.get("delta", "")

            elif event_type == "response.function_call_arguments.done":
                if current_call_id and current_func_name:
                    result, ended = _handle_function(
                        current_func_name, current_func_args, call_record
                    )
                    if ended:
                        call_ending = True

                    await openai_ws.send(json.dumps({
                        "type": "conversation.item.create",
                        "item": {
                            "type":    "function_call_output",
                            "call_id": current_call_id,
                            "output":  json.dumps(result),
                        },
                    }))
                    await openai_ws.send(json.dumps({"type": "response.create"}))
                    current_call_id = current_func_name = None
                    current_func_args = ""

            elif event_type == "error":
                print(f"  [OpenAI Error] {event.get('error', {}).get('message')}")

    except Exception as e:
        if not stop_event.is_set():
            print(f"[OpenAI→Twilio] {type(e).__name__}: {e}")
        stop_event.set()


async def _hangup(call_sid: str):
    """End the Twilio call via REST API (runs sync client in thread pool)."""
    if not call_sid:
        return
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            None,
            lambda: twilio_client.calls(call_sid).update(status="completed"),
        )
        print(f"[Server] Call {call_sid} hung up.")
    except Exception as e:
        print(f"[Server] Hangup error: {e}")


# ── Helpers ────────────────────────────────────────────────────────────

def _handle_function(name: str, arguments: str, call_record: CallRecord):
    """Execute a function call. Returns (result_dict, call_ended)."""
    args = json.loads(arguments)

    if name == "record_response":
        idx    = int(args["question_index"])
        answer = args["answer"]
        call_record.set_answer(idx, answer)
        q = QUESTIONS[idx] if idx < len(QUESTIONS) else f"Q{idx}"
        print(f"  [Recorded] Q{idx + 1}: {q}")
        print(f"             A{idx + 1}: {answer}")
        return {"status": "recorded", "question_index": idx}, False

    if name == "end_call":
        outcome = args["outcome"]
        notes   = args.get("notes", "")
        call_record.set_outcome(outcome)
        print(f"\n[Agent] Call ended: {outcome}")
        if notes:
            print(f"  Notes: {notes}")
        return {"status": "call_ended", "outcome": outcome}, True

    return {"error": f"Unknown: {name}"}, False


def _tools():
    return [
        {
            "type": "function",
            "name": "record_response",
            "description": "Record a patient's answer to a health check-in question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question_index": {
                        "type": "integer",
                        "description": (
                            "0-based index (0-13). Q0=feeling, Q1=weight, Q2=height, "
                            "Q3=weight_lost, Q4=side_effects, Q5=satisfaction, Q6=goal_weight, "
                            "Q7=dosage_request, Q8=new_meds, Q9=new_conditions, Q10=allergies, "
                            "Q11=surgeries, Q12=doctor_questions, Q13=address_changed"
                        ),
                    },
                    "answer": {"type": "string", "description": "Patient's answer."},
                },
                "required": ["question_index", "answer"],
            },
        },
        {
            "type": "function",
            "name": "end_call",
            "description": "End the call and record the outcome.",
            "parameters": {
                "type": "object",
                "properties": {
                    "outcome": {
                        "type": "string",
                        "enum": [
                            "completed", "incomplete", "opted_out",
                            "scheduled", "escalated", "wrong_number", "voicemail",
                        ],
                    },
                    "notes": {"type": "string"},
                },
                "required": ["outcome"],
            },
        },
    ]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
