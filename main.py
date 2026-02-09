import os
import json
import base64
import asyncio
import audioop
import logging
import uvicorn
import numpy as np
from google import genai
from google.genai import types
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form
from fastapi.responses import Response, JSONResponse
from twilio.rest import Client as TwilioClient
from dotenv import load_dotenv

load_dotenv()

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("voice-agent")

# --- Configuration ---
PORT = int(os.getenv("PORT", "8520"))
DOMAIN = os.getenv("DOMAIN")
if not DOMAIN:
    raise ValueError("DOMAIN environment variable not set.")
DOMAIN = DOMAIN.replace("https://", "").replace("http://", "").rstrip("/")
WS_URL = f"wss://{DOMAIN}/ws"

SYSTEM_PROMPT = (
    "You are a helpful and friendly voice assistant on a phone call. "
    "Keep responses short and natural. "
    "Spell out all numbers (e.g. say 'one thousand two hundred' instead of 1200). "
    "Do not use special characters, bullet points, or emojis."
)

WELCOME_PROMPT = "Greet the caller warmly. Say hi and ask how you can help them today."

GEMINI_MODEL = "gemini-2.5-flash-live"
GEMINI_VOICE = "Aoede"

# Audio format constants
TWILIO_SAMPLE_RATE = 8000       # Twilio sends/expects mulaw at 8 kHz
GEMINI_INPUT_RATE = 16000       # Gemini Live expects PCM16 at 16 kHz
GEMINI_OUTPUT_RATE = 24000      # Gemini Live outputs PCM16 at 24 kHz
MULAW_FRAME_SIZE = 160          # 20 ms of mulaw at 8 kHz (one Twilio frame)

# --- Twilio Initialization ---
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
    raise ValueError(
        "TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_PHONE_NUMBER must be set."
    )
twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# --- Gemini Initialization ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
gemini_client = genai.Client(api_key=GOOGLE_API_KEY)

# Active call sessions keyed by stream_sid
active_sessions: dict[str, dict] = {}

# --- FastAPI App ---
app = FastAPI()


# ──────────────────────────────────────────────
# Audio conversion helpers
# ──────────────────────────────────────────────

def resample_audio(data: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Resample PCM16 audio via numpy linear interpolation."""
    if src_rate == dst_rate or len(data) < 2:
        return data
    samples = np.frombuffer(data, dtype=np.int16)
    if len(samples) == 0:
        return data
    num_out = int(len(samples) * dst_rate / src_rate)
    if num_out == 0:
        return b""
    indices = np.linspace(0, len(samples) - 1, num_out)
    resampled = np.interp(indices, np.arange(len(samples)), samples.astype(np.float64))
    return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()


def twilio_to_gemini_audio(mulaw_bytes: bytes) -> bytes:
    """Twilio mulaw 8 kHz ➜ Gemini PCM16 16 kHz."""
    pcm16_8k = audioop.ulaw2lin(mulaw_bytes, 2)
    return resample_audio(pcm16_8k, TWILIO_SAMPLE_RATE, GEMINI_INPUT_RATE)


def gemini_to_twilio_audio(pcm16_bytes: bytes) -> bytes:
    """Gemini PCM16 24 kHz ➜ Twilio mulaw 8 kHz."""
    pcm16_8k = resample_audio(pcm16_bytes, GEMINI_OUTPUT_RATE, TWILIO_SAMPLE_RATE)
    return audioop.lin2ulaw(pcm16_8k, 2)


# ──────────────────────────────────────────────
# HTTP endpoints
# ──────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health probe for Docker / load balancers."""
    return JSONResponse({
        "status": "healthy",
        "active_sessions": len(active_sessions),
    })


@app.post("/twiml")
async def twiml_endpoint():
    """Return TwiML that opens a bidirectional Media Stream."""
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{WS_URL}" />
    </Connect>
</Response>"""
    return Response(content=xml, media_type="text/xml")


@app.post("/call")
async def call_endpoint(phone_number: str = Form(...)):
    """Initiate an outbound call."""
    call = twilio_client.calls.create(
        to=phone_number,
        from_=TWILIO_PHONE_NUMBER,
        url=f"https://{DOMAIN}/twiml",
    )
    return JSONResponse({"callSid": call.sid, "status": call.status})


# ──────────────────────────────────────────────
# WebSocket bridge: Twilio ↔ Gemini Live
# ──────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Bridge Twilio Media Streams to Gemini Live in real time."""
    await websocket.accept()

    stream_sid: str | None = None
    call_sid: str = "unknown"
    tasks: list[asyncio.Task] = []

    try:
        # ── Phase 1: wait for the Twilio "start" event ──
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            if msg["event"] == "start":
                stream_sid = msg["start"]["streamSid"]
                call_sid = msg["start"].get("callSid", "unknown")
                logger.info("Stream started  stream_sid=%s  call_sid=%s", stream_sid, call_sid)
                break

        # ── Phase 2: open Gemini Live session ──
        gemini_config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=types.Content(
                parts=[types.Part(text=SYSTEM_PROMPT)]
            ),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=GEMINI_VOICE
                    )
                )
            ),
        )

        async with gemini_client.aio.live.connect(
            model=GEMINI_MODEL,
            config=gemini_config,
        ) as gemini_session:

            active_sessions[stream_sid] = {
                "call_sid": call_sid,
                "stream_sid": stream_sid,
            }

            # Send a welcome greeting so Gemini speaks first
            await gemini_session.send(
                input=WELCOME_PROMPT,
                end_of_turn=True,
            )

            # ── Phase 3: run bidirectional audio bridge ──
            twilio_task = asyncio.create_task(
                _listen_twilio(websocket, gemini_session, stream_sid),
                name=f"twilio-{stream_sid}",
            )
            gemini_task = asyncio.create_task(
                _listen_gemini(gemini_session, websocket, stream_sid),
                name=f"gemini-{stream_sid}",
            )
            tasks = [twilio_task, gemini_task]

            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Log errors from completed tasks
            for task in done:
                exc = task.exception()
                if exc:
                    logger.error("Task %s raised: %s", task.get_name(), exc)

    except WebSocketDisconnect:
        logger.info("Twilio disconnected  stream_sid=%s", stream_sid)
    except Exception as exc:
        logger.error("WebSocket error  stream_sid=%s: %s", stream_sid, exc)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if stream_sid and stream_sid in active_sessions:
            del active_sessions[stream_sid]
        logger.info("Session cleaned up  stream_sid=%s", stream_sid)


async def _listen_twilio(
    websocket: WebSocket,
    gemini_session,
    stream_sid: str,
) -> None:
    """Forward Twilio audio ➜ Gemini Live (mulaw 8 kHz ➜ PCM16 16 kHz)."""
    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            if msg["event"] == "media":
                mulaw_bytes = base64.b64decode(msg["media"]["payload"])
                pcm16_16k = twilio_to_gemini_audio(mulaw_bytes)

                await gemini_session.send(
                    input=types.LiveClientRealtimeInput(
                        media_chunks=[
                            types.Blob(
                                data=pcm16_16k,
                                mime_type="audio/pcm;rate=16000",
                            )
                        ]
                    )
                )

            elif msg["event"] == "stop":
                logger.info("Twilio stream stopped  stream_sid=%s", stream_sid)
                return

            elif msg["event"] == "mark":
                mark_name = msg.get("mark", {}).get("name", "")
                logger.debug("Mark received: %s", mark_name)

    except WebSocketDisconnect:
        logger.info("Twilio disconnected in listener  stream_sid=%s", stream_sid)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("listen_twilio error  stream_sid=%s: %s", stream_sid, exc)


async def _listen_gemini(
    gemini_session,
    websocket: WebSocket,
    stream_sid: str,
) -> None:
    """Forward Gemini Live audio ➜ Twilio (PCM16 24 kHz ➜ mulaw 8 kHz)."""
    try:
        while True:
            turn = gemini_session.receive()
            async for response in turn:
                # ── Audio data ──
                if response.data:
                    mulaw_bytes = gemini_to_twilio_audio(response.data)

                    # Send in Twilio-sized frames (20 ms each)
                    for i in range(0, len(mulaw_bytes), MULAW_FRAME_SIZE):
                        chunk = mulaw_bytes[i : i + MULAW_FRAME_SIZE]
                        if not chunk:
                            continue
                        payload = base64.b64encode(chunk).decode("utf-8")
                        await websocket.send_text(json.dumps({
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": payload},
                        }))

                # ── Turn complete ──
                server = getattr(response, "server_content", None)
                if server and getattr(server, "turn_complete", False):
                    await websocket.send_text(json.dumps({
                        "event": "mark",
                        "streamSid": stream_sid,
                        "mark": {"name": "turn-complete"},
                    }))

    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("listen_gemini error  stream_sid=%s: %s", stream_sid, exc)


# ──────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting server on port %d", PORT)
    logger.info("WebSocket URL for Twilio: %s", WS_URL)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
