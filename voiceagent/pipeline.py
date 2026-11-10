"""One request through the pipeline. Two modes share all components; only orchestration differs.

baseline: wait for final transcript -> wait for the full LLM response -> one TTS call for the
          whole text -> playback starts when the full audio has returned.
streamed: LLM output is split into sentence chunks; chunk 1 goes to TTS as soon as it is
          complete; playback starts on first audio bytes while later chunks synthesise.

Both modes stream audio into Prisma at real-time cadence (the API has no partial results, so
ASR is identical in both modes by construction; the A/B isolates the LLM+TTS overlap).
All timestamps: time.perf_counter() relative to request start (t0), one machine.
"""
import asyncio
import datetime
import time

from . import config
from .audio import load_wav_16k_mono
from .chunker import SentenceChunker
from .playback import SimPlayback
from .rollno import normalize_roll_numbers


def _d(ts, a, b):
    return round(ts[b] - ts[a], 4) if a in ts and b in ts else None


async def run_request(wav_path, mode, stt, tts, llm, *, mock=False, extra=None, ack_pcm=None) -> dict:
    """ack_pcm: optional pre-synthesised acknowledgement ("Haan ji, ek second") played the moment the
    user stops speaking. Reported separately as ttfa_ack; ttfa always means first ANSWER audio."""
    assert mode in ("baseline", "streamed")
    t0 = time.perf_counter()
    now = lambda: time.perf_counter() - t0
    ts, errors = {}, []
    pcm = load_wav_16k_mono(wav_path)
    pb = SimPlayback(config.TTS_SAMPLE_RATE, now)
    chunk_log = []  # per TTS chunk: text, t_request, t_first_byte, t_done
    all_audio = []  # every TTS byte in order (for --save-audio; not used in timing)

    # ---- 1. ASR ----
    stt_res = await stt.transcribe_stream(pcm, now, on_eos=(lambda: pb.push(ack_pcm, ack=True)) if ack_pcm else None)
    ts["t_end_of_speech"] = stt_res.t_end_of_speech
    ts["t_asr_final"] = stt_res.t_asr_final
    if stt_res.t_asr_first_event:
        ts["t_asr_first_event"] = stt_res.t_asr_first_event

    # ---- 2+3+4. LLM -> chunker -> TTS -> playback ----
    chunker = SentenceChunker()
    q: asyncio.Queue = asyncio.Queue()
    first_sentence_seen = False

    async def synth(text):
        cl = {"text": text, "t_request": now(), "t_first_byte": None}
        parts = []
        try:
            async for b in tts.synthesize_stream(text):
                if cl["t_first_byte"] is None:
                    cl["t_first_byte"] = now()
                all_audio.append(b)
                if mode == "streamed":
                    pb.push(b)
                else:
                    parts.append(b)
        except Exception as e:  # keep the record; a failed request is data too
            errors.append(f"tts: {e!r}")
        cl["t_done"] = now()
        chunk_log.append(cl)
        return parts

    async def tts_worker():  # streamed mode: sequential (API allows one active request)
        while True:
            text = await q.get()
            if text is None:
                return
            await synth(text)

    worker = asyncio.create_task(tts_worker()) if mode == "streamed" else None

    def emit(chunks):
        nonlocal first_sentence_seen
        for c in chunks:
            if not first_sentence_seen:
                ts["t_llm_first_sentence"] = now()
                first_sentence_seen = True
            if mode == "streamed":
                q.put_nowait(c)

    async def on_delta(d):
        emit(chunker.feed(d))

    llm_out = {"text": "", "tool_calls": [], "model": getattr(llm, "model", "?")}
    llm_input = ""
    if stt_res.text:
        try:
            llm_input = normalize_roll_numbers(stt_res.text)  # "S one zero zero four" -> "S1004" (v2)
            llm_out = await llm.respond(llm_input, on_delta, ts, now)
            emit(chunker.flush())
        except Exception as e:
            errors.append(f"llm: {e!r}")
    else:
        errors.append("asr: empty transcript")
    ts["t_llm_done"] = now()

    if mode == "streamed":
        q.put_nowait(None)
        await worker
    elif llm_out["text"]:
        parts = await synth(llm_out["text"])  # full text, one call; play after full audio returns
        pb.push(b"".join(parts))

    if chunk_log:
        first = min(chunk_log, key=lambda c: c["t_request"])
        ts["t_tts_request"] = first["t_request"]
        if first["t_first_byte"] is not None:
            ts["t_tts_first_byte"] = first["t_first_byte"]
    if pb.t_first is not None:
        ts["t_first_audio_out"] = pb.t_first
        ts["t_last_audio_end"] = pb.t_last_audio_end

    eos = ts["t_end_of_speech"]
    metrics = {
        "asr_latency": _d(ts, "t_end_of_speech", "t_asr_final"),
        "handoff": _d(ts, "t_asr_final", "t_llm_request"),  # our own overhead; should be ~0
        "llm_ttft": _d(ts, "t_llm_request", "t_llm_first_token"),
        "tool_latency": _d(ts, "t_tool_start", "t_tool_end"),
        "llm_first_sentence": _d(ts, "t_llm_request", "t_llm_first_sentence"),
        "llm_total": _d(ts, "t_llm_request", "t_llm_done"),
        "tts_ttfb": _d(ts, "t_tts_request", "t_tts_first_byte"),
        "ttfa": round(ts["t_first_audio_out"] - eos, 4) if "t_first_audio_out" in ts else None,
        "ttfa_ack": round(pb.t_ack - eos, 4) if pb.t_ack is not None else None,
        "total_response_time": round(ts["t_last_audio_end"] - eos, 4) if "t_last_audio_end" in ts else None,
        "playback_gap_total": round(sum(pb.gaps), 4),
        "playback_n_gaps": len(pb.gaps),
        "n_tts_chunks": len(chunk_log),
    }
    return {
        "ts_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "wav": str(wav_path), "mode": mode, "mock": mock, **(extra or {}),
        "transcript": stt_res.text, "llm_input": llm_input, "reply": llm_out["text"], "tool_calls": llm_out["tool_calls"],
        "llm_model": llm_out["model"], "stt_model": config.STT_MODEL, "tts_model": config.TTS_MODEL,
        "timestamps": {k: round(v, 4) for k, v in ts.items()},
        "metrics": metrics, "tts_chunks": [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()} for c in chunk_log],
        "asr_events": stt_res.events, "errors": errors,
        "_audio": b"".join(all_audio),  # stripped before logging
    }
