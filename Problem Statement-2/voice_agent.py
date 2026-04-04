"""TrimRX Voice Agent using OpenAI Realtime API.

Handles WebSocket connection, microphone input, speaker output,
and the full medication refill check-in conversation.
"""
import asyncio
import base64
import json
import queue
import threading
import numpy as np
import sounddevice as sd
import websockets

from config import (
    OPENAI_API_KEY,
    REALTIME_URL,
    SAMPLE_RATE,
    CHANNELS,
    CHUNK_SIZE,
    VOICE,
)
from prompts import build_system_prompt
from questionnaire import CallRecord, QUESTIONS


class VoiceAgent:
    """Real-time voice agent for medication refill check-in calls."""

    def __init__(self, patient_name: str, medication: str, dosage: str):
        self.patient_name = patient_name
        self.medication = medication
        self.dosage = dosage
        self.call_record = CallRecord(patient_name, medication, dosage)
        self.ws = None
        # Thread-safe queue for audio playback (sounddevice runs in a thread)
        self.playback_queue = queue.Queue(maxsize=200)
        # Thread-safe queue for mic audio to send via WebSocket
        self.mic_queue = queue.Queue(maxsize=200)
        self.is_running = False
        self._call_ending = False    # set when end_call tool fires
        self.agent_speaking = False  # mute mic while agent talks (echo suppression)
        self.output_stream = None
        self.input_stream = None
        self._loop = None  # will be set to the running asyncio loop

    async def connect(self):
        """Establish WebSocket connection to OpenAI Realtime API."""
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1",
        }

        print("[Agent] Connecting to OpenAI Realtime API...")
        self.ws = await websockets.connect(
            REALTIME_URL,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=20,
        )

        # Wait for session.created
        event = json.loads(await self.ws.recv())
        if event["type"] == "session.created":
            print("[Agent] Session created successfully.")
        else:
            print(f"[Agent] Unexpected event: {event['type']}")

        # Configure session
        system_prompt = build_system_prompt(
            self.patient_name, self.medication, self.dosage
        )

        await self.ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": system_prompt,
                "voice": VOICE,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1",
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 700,
                },
                "tools": [
                    {
                        "type": "function",
                        "name": "record_response",
                        "description": (
                            "Record a patient's answer to a health check-in question. "
                            "Call this after the patient answers each of the 14 questions."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "question_index": {
                                    "type": "integer",
                                    "description": (
                                        "The 0-based index of the question (0-13). "
                                        "Q0=feeling, Q1=weight, Q2=height, Q3=weight_lost, "
                                        "Q4=side_effects, Q5=satisfaction, Q6=goal_weight, "
                                        "Q7=dosage_request, Q8=new_meds, Q9=new_conditions, "
                                        "Q10=allergies, Q11=surgeries, Q12=doctor_questions, "
                                        "Q13=address_changed"
                                    ),
                                },
                                "answer": {
                                    "type": "string",
                                    "description": "The patient's answer, captured accurately.",
                                },
                            },
                            "required": ["question_index", "answer"],
                        },
                    },
                    {
                        "type": "function",
                        "name": "end_call",
                        "description": (
                            "End the call and record the outcome. Call this when the "
                            "conversation is finished — after all questions are answered, "
                            "or if the patient opts out, is a wrong number, wants to "
                            "reschedule, or needs escalation."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "outcome": {
                                    "type": "string",
                                    "enum": [
                                        "completed", "incomplete", "opted_out",
                                        "scheduled", "escalated", "wrong_number",
                                        "voicemail",
                                    ],
                                    "description": "The call outcome.",
                                },
                                "notes": {
                                    "type": "string",
                                    "description": "Brief notes about why the call ended.",
                                },
                            },
                            "required": ["outcome"],
                        },
                    },
                ],
                "tool_choice": "auto",
            },
        }))

        # Wait for session.updated confirmation
        event = json.loads(await self.ws.recv())
        if event["type"] == "session.updated":
            print("[Agent] Session configured.")

        # List available audio devices for debugging
        print(f"[Agent] Input device: {sd.query_devices(kind='input')['name']}")
        print(f"[Agent] Output device: {sd.query_devices(kind='output')['name']}")
        print("[Agent] Ready. Speak into your microphone.\n")

    async def send_initial_greeting(self):
        """Trigger the agent to start the conversation."""
        await self.ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": (
                        "[System: The call has just connected. Start the conversation "
                        "by greeting the patient with your standard opening.]"
                    ),
                }],
            },
        }))
        await self.ws.send(json.dumps({"type": "response.create"}))

    def _start_audio_output(self):
        """Start the speaker output stream."""
        def output_callback(outdata, frames, time_info, status):
            try:
                data = self.playback_queue.get_nowait()
                outdata_bytes = bytes(outdata)
                out_len = len(outdata_bytes)
                if len(data) < out_len:
                    outdata[:len(data)] = data
                    outdata[len(data):] = b'\x00' * (out_len - len(data))
                else:
                    outdata[:] = data[:out_len]
            except queue.Empty:
                outdata[:] = b'\x00' * len(bytes(outdata))

        self.output_stream = sd.RawOutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=CHUNK_SIZE,
            callback=output_callback,
        )
        self.output_stream.start()

    def _start_audio_input(self):
        """Start the microphone input stream."""
        def input_callback(indata, frames, time_info, status):
            if not self.is_running:
                return
            # Echo suppression: don't capture mic while agent is speaking
            if self.agent_speaking:
                return
            audio_bytes = bytes(indata)
            try:
                self.mic_queue.put_nowait(audio_bytes)
            except queue.Full:
                pass  # drop frame if queue is full

        self.input_stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=CHUNK_SIZE,
            callback=input_callback,
        )
        self.input_stream.start()

    async def _mic_sender(self):
        """Async task: drain mic_queue and send audio to WebSocket."""
        while self.is_running:
            try:
                audio_bytes = self.mic_queue.get_nowait()
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                await self.ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": audio_b64,
                }))
            except queue.Empty:
                await asyncio.sleep(0.01)  # yield, check again shortly
            except websockets.exceptions.ConnectionClosed:
                break

    async def _handle_function_call(self, call_id: str, name: str, arguments: str):
        """Handle tool calls from the agent."""
        args = json.loads(arguments)

        if name == "record_response":
            idx = args["question_index"]
            answer = args["answer"]
            self.call_record.set_answer(idx, answer)
            q = QUESTIONS[idx] if idx < len(QUESTIONS) else f"Q{idx}"
            print(f"  [Recorded] Q{idx + 1}: {q}")
            print(f"             A{idx + 1}: {answer}")
            result = {"status": "recorded", "question_index": idx}

        elif name == "end_call":
            outcome = args["outcome"]
            notes = args.get("notes", "")
            self.call_record.set_outcome(outcome)
            print(f"\n[Agent] Call ended: {outcome}")
            if notes:
                print(f"  Notes: {notes}")
            result = {"status": "call_ended", "outcome": outcome}
            self._call_ending = True

        else:
            result = {"error": f"Unknown function: {name}"}

        # Send function output back
        await self.ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result),
            },
        }))
        # Request the model to continue
        await self.ws.send(json.dumps({"type": "response.create"}))

    def _unmute_mic(self):
        """Re-enable microphone after agent finishes speaking."""
        self.agent_speaking = False

    def _stop(self):
        """Signal the agent to stop."""
        self.is_running = False

    async def _listen(self):
        """Main event loop: listen for server events."""
        current_call_id = None
        current_func_name = None
        current_func_args = ""

        try:
            async for message in self.ws:
                if not self.is_running:
                    break

                event = json.loads(message)
                event_type = event.get("type", "")

                # ── Audio output ───────────────────────────────────
                if event_type == "response.audio.delta":
                    self.agent_speaking = True  # mute mic
                    # Flush any mic audio already buffered — it's pre-speech noise
                    while not self.mic_queue.empty():
                        try:
                            self.mic_queue.get_nowait()
                        except queue.Empty:
                            break
                    audio_bytes = base64.b64decode(event["delta"])
                    # Split into chunks matching the output stream blocksize
                    chunk_bytes = CHUNK_SIZE * 2  # 2 bytes per int16 sample
                    for i in range(0, len(audio_bytes), chunk_bytes):
                        chunk = audio_bytes[i:i + chunk_bytes]
                        try:
                            self.playback_queue.put_nowait(chunk)
                        except queue.Full:
                            pass  # drop if buffer is full

                elif event_type == "response.audio.done":
                    # Unmute after 1.5s — gives speaker echo time to die out
                    self._loop.call_later(1.5, self._unmute_mic)
                    # If the call is ending, wait for playback to drain then stop
                    if self._call_ending:
                        await asyncio.sleep(2.0)  # let goodbye audio play out
                        self.is_running = False
                        break

                # ── Transcription of user speech ───────────────────
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    transcript = event.get("transcript", "").strip()
                    if transcript:
                        print(f"  [Patient]: {transcript}")

                # ── Agent transcript (for logging) ─────────────────
                elif event_type == "response.audio_transcript.done":
                    transcript = event.get("transcript", "").strip()
                    if transcript:
                        print(f"  [Jessica]: {transcript}")

                # ── Function calls ─────────────────────────────────
                elif event_type == "response.function_call_arguments.delta":
                    current_func_args += event.get("delta", "")

                elif event_type == "response.output_item.added":
                    item = event.get("item", {})
                    if item.get("type") == "function_call":
                        current_call_id = item.get("call_id")
                        current_func_name = item.get("name")
                        current_func_args = ""

                elif event_type == "response.function_call_arguments.done":
                    if current_call_id and current_func_name:
                        await self._handle_function_call(
                            current_call_id,
                            current_func_name,
                            current_func_args,
                        )
                        current_call_id = None
                        current_func_name = None
                        current_func_args = ""

                # ── Speech detection (interruption handling) ───────
                elif event_type == "input_audio_buffer.speech_started":
                    # Only treat as real interruption if mic is already unmuted
                    # If agent_speaking=True, this is echo from the speakers — ignore
                    if not self.agent_speaking:
                        while not self.playback_queue.empty():
                            try:
                                self.playback_queue.get_nowait()
                            except queue.Empty:
                                break

                # ── Errors ─────────────────────────────────────────
                elif event_type == "error":
                    error = event.get("error", {})
                    print(f"  [Error] {error.get('message', 'Unknown error')}")

        except websockets.exceptions.ConnectionClosed:
            print("[Agent] Connection closed.")

    async def run(self):
        """Start the voice agent."""
        self.is_running = True
        self._loop = asyncio.get_running_loop()

        try:
            await self.connect()
            self._start_audio_output()
            self._start_audio_input()
            await self.send_initial_greeting()

            # Run mic sender and event listener concurrently
            await asyncio.gather(
                self._mic_sender(),
                self._listen(),
            )
        except KeyboardInterrupt:
            print("\n[Agent] Interrupted by user.")
            self.call_record.set_outcome("incomplete")
        finally:
            self.is_running = False
            if self.input_stream:
                self.input_stream.stop()
                self.input_stream.close()
            if self.output_stream:
                self.output_stream.stop()
                self.output_stream.close()
            if self.ws:
                await self.ws.close()

            # Print final summary
            print("\n" + "=" * 60)
            print("CALL SUMMARY")
            print("=" * 60)
            print(self.call_record.summary())

            return self.call_record
