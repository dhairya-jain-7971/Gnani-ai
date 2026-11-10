"""Run one wav through the pipeline and print the latency breakdown.
  python run_one.py testset/clean_01.wav --mode streamed
  python run_one.py anything.wav --mode baseline --mock      # offline smoke test only
"""
import argparse
import asyncio
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")  # Devanagari transcripts on a Windows console

from voiceagent import config
from voiceagent.pipeline import run_request


def build(mock):
    if mock:
        from voiceagent.mock import MockLLM, MockSTT, MockTTS
        return MockSTT(), MockTTS(), MockLLM()
    if not config.GNANI_API_KEY:
        raise SystemExit("GNANI_API_KEY not set (see .env.example)")
    from voiceagent.adapters import PrismaSTT, TimbreTTS
    from voiceagent.llm import LLM
    return PrismaSTT(), TimbreTTS(), LLM()


async def warm(tts):
    if hasattr(tts, "warm"):
        await tts.warm()


ACK_TEXT = "Haan ji, ek second."


async def load_ack(tts):
    """Acknowledgement audio, synthesised once with Timbre and cached (raw s16le at TTS_SAMPLE_RATE)."""
    p = config.ROOT / "assets" / f"ack_{config.TTS_MODEL}_{config.TTS_VOICE}_{config.TTS_SAMPLE_RATE}.raw"
    if p.exists():
        return p.read_bytes()
    audio = b"".join([b async for b in tts.synthesize_stream(ACK_TEXT)])
    p.parent.mkdir(exist_ok=True)
    p.write_bytes(audio)
    return audio


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--mode", choices=["baseline", "streamed"], default="streamed")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--save-audio", action="store_true", help="write the spoken reply to reply_<mode>.wav")
    ap.add_argument("--ack", action="store_true", help='play "Haan ji, ek second." the moment the user stops')
    a = ap.parse_args()
    stt, tts, llm = build(a.mock)
    await warm(tts)
    ack = await load_ack(tts) if a.ack else None
    rec = await run_request(a.wav, a.mode, stt, tts, llm, mock=a.mock, ack_pcm=ack)
    if ack:
        rec["_audio"] = ack + rec["_audio"]
    audio = rec.pop("_audio", b"")
    if a.save_audio and audio:
        from voiceagent.audio import write_wav
        out = f"reply_{a.mode}.wav"
        write_wav(out, audio, config.TTS_SAMPLE_RATE)
        print(f"saved spoken reply -> {out}")
    print("transcript:", rec["transcript"])
    print("reply     :", rec["reply"])
    print("tools     :", json.dumps(rec["tool_calls"], ensure_ascii=False))
    for k, v in rec["metrics"].items():
        print(f"  {k:22s} {v}")
    if rec["errors"]:
        print("ERRORS:", rec["errors"])
    if a.mock:
        print("(MOCK run: not a measurement)")
    config.RESULTS_DIR.mkdir(exist_ok=True)
    with open(config.RESULTS_DIR / "requests.jsonl", "a", encoding="utf-8") as f:  # f6: one jsonl record per request
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    from voiceagent.adapters import PrismaSTT
    await PrismaSTT.drain()


if __name__ == "__main__":
    asyncio.run(main())
