"""Thin adapters over Gnani Prisma (STT, websocket) and Timbre (TTS, SSE).

Interfaces (see PRD section 5):
    transcribe_stream(pcm, clock) -> STTResult     (audio in, transcript + timestamps out)
    synthesize_stream(text)       -> async iterator of raw s16le mono PCM bytes

Protocol details come from docs.gnani.ai (fetched 2026-11-10):
  * STT: wss, `x-api-key-id` header, config as upgrade headers, 1024-byte PCM frames sent at
    real-time cadence, `commit` message to close an open utterance, JSON `transcript` events.
    The docs describe NO partial/interim transcripts: each segment gives one final `transcript`.
  * TTS: POST SSE, `X-API-Key-ID`, base64 audio in `chunk` events, ONE active request per key (429).
"""
import asyncio
import base64
import json
import time
from dataclasses import dataclass, field

import httpx
import websockets

from . import config
from .audio import FRAME_SECONDS, frames


@dataclass
class STTResult:
    text: str = ""
    t_end_of_speech: float = 0.0
    t_asr_final: float = 0.0
    t_asr_first_event: float = 0.0  # first speech_start/processing/transcript message after audio started
    events: list = field(default_factory=list)
    model_info: dict = field(default_factory=dict)


class PrismaSTT:
    """Streams a pre-recorded PCM buffer at real-time cadence. In benchmark mode,
    t_end_of_speech = moment the last audio frame has been sent (no live endpointing)."""

    def __init__(self, timeout=15.0):
        self.timeout = timeout

    def _headers(self):
        return {
            "x-api-key-id": config.GNANI_API_KEY,
            "model": config.STT_MODEL,
            "language_code": config.STT_LANGUAGE,
            "format": config.STT_FORMAT,
            "x-sample-rate": "16000",
        }

    _closing = set()  # background close tasks (kept referenced so they are not GC'd)

    async def transcribe_stream(self, pcm: bytes, now, on_eos=None) -> STTResult:
        res = STTResult()
        ws = await websockets.connect(config.STT_WS_URL, additional_headers=self._headers(), max_size=None)
        try:
            transcripts, committed = [], asyncio.Event()
            state = {"open": False}  # True between speech_start and its transcript
            got_final = asyncio.Event()

            async def reader():
                try:
                    await _read(ws)
                except websockets.ConnectionClosed as e:
                    res.events.append((round(now(), 4), f"closed:{e.rcvd.code if e.rcvd else '?'}"))
                    got_final.set()

            async def _read(ws):
                async for raw in ws:
                    msg = json.loads(raw)
                    t = now()
                    typ = msg.get("type")
                    res.events.append((round(t, 4), typ))
                    if typ == "connected":
                        res.model_info = msg.get("config", {})
                    elif typ in ("speech_start", "processing", "transcript") and not res.t_asr_first_event:
                        res.t_asr_first_event = t
                    if typ in ("speech_start", "processing"):
                        state["open"] = True
                    if typ == "transcript":
                        state["open"] = False
                        if msg.get("text", "").strip():
                            transcripts.append(msg["text"].strip())
                        res.t_asr_final = t
                        if committed.is_set():
                            got_final.set()
                    elif typ == "error":
                        res.events.append((round(t, 4), "error:" + str(msg.get("message"))))
                        got_final.set()

            rtask = asyncio.create_task(reader())
            t_next = time.perf_counter()
            for f in frames(pcm):  # real-time cadence; bursting degrades the server VAD
                if got_final.is_set():  # server closed or errored early
                    break
                try:
                    await ws.send(f)
                except websockets.ConnectionClosed:
                    await asyncio.sleep(0.05)  # let the reader record the close
                    break
                t_next += FRAME_SECONDS
                await asyncio.sleep(max(0.0, t_next - time.perf_counter()))
            res.t_end_of_speech = now()
            if on_eos:
                on_eos()
            if not got_final.is_set() and transcripts and not state["open"]:
                # VAD already closed the last segment before end of audio: nothing left to commit.
                res.events.append((round(now(), 4), "final_before_eos"))
                got_final.set()
            if not got_final.is_set():
                await ws.send(json.dumps({"type": "commit"}))
                committed.set()
                try:
                    await asyncio.wait_for(got_final.wait(), self.timeout)
                except asyncio.TimeoutError:
                    res.events.append((round(now(), 4), "timeout_waiting_for_final"))
            rtask.cancel()
            res.text = " ".join(transcripts)
            if not res.t_asr_final:
                res.t_asr_final = now()
        finally:
            # Closing handshake took ~1 s in live tests; do it off the critical path.
            t = asyncio.create_task(ws.close())
            self._closing.add(t)
            t.add_done_callback(self._closing.discard)
        return res

    @classmethod
    async def drain(cls):
        """Await pending background closes (call before the event loop exits)."""
        if cls._closing:
            await asyncio.gather(*cls._closing, return_exceptions=True)


class TimbreTTS:
    """Streaming synthesis via SSE. Serialised with a lock: the API allows one active request."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._client = None  # one keep-alive HTTP client: v1 paid a TCP+TLS handshake on every chunk

    def _http(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60, limits=httpx.Limits(keepalive_expiry=120))
        return self._client

    async def warm(self):
        """Open the connection before the first request (startup, off the critical path)."""
        try:
            await self._http().get(config.TTS_SSE_URL.split("/api/")[0] + "/")
        except Exception:
            pass

    async def synthesize_stream(self, text: str):
        body = {
            "text": text, "model": config.TTS_MODEL, "voice": config.TTS_VOICE,
            "language": config.TTS_LANGUAGE, "speed": 1.0,
            "audio_config": {"sample_rate": config.TTS_SAMPLE_RATE, "num_channels": 1,
                             "sample_width": 2, "encoding": "linear_pcm", "container": "raw"},
        }
        headers = {"Content-Type": "application/json", "X-API-Key-ID": config.GNANI_API_KEY}
        async with self._lock:
            if True:  # keeps the v1 indentation of the block below
                async with self._http().stream("POST", config.TTS_SSE_URL, json=body, headers=headers) as r:
                    if r.status_code != 200:
                        raise RuntimeError(f"TTS HTTP {r.status_code}: {(await r.aread())[:300]!r}")
                    event = None
                    async for line in r.aiter_lines():
                        if line.startswith("event:"):
                            event = line[6:].strip()
                        elif line.startswith("data:"):
                            d = json.loads(line[5:].strip() or "{}")
                            if event == "chunk" and d.get("audio"):
                                yield base64.b64decode(d["audio"])
                            elif event == "error" or d.get("type") == "error":
                                raise RuntimeError(f"TTS stream error: {d}")
