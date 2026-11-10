"""MOCK adapters for offline smoke tests of the pipeline logic ONLY.
Sleeps stand in for network time. Records produced with these carry mock=true and must never
be reported as measurements."""
import asyncio

from . import config, tools
from .adapters import STTResult
from .audio import FRAME_SECONDS, frames


class MockSTT:
    async def transcribe_stream(self, pcm, now, on_eos=None):
        n = len(list(frames(pcm)))
        await asyncio.sleep(n * FRAME_SECONDS * 0.05)  # sped up; irrelevant for smoke test
        r = STTResult(t_end_of_speech=now())
        if on_eos:
            on_eos()
        await asyncio.sleep(0.15)
        r.text, r.t_asr_final, r.t_asr_first_event = "mera roll number S1004 hai, scholarship ka status batao", now(), now()
        return r


class MockTTS:
    def __init__(self):
        self._lock = asyncio.Lock()

    async def synthesize_stream(self, text):
        async with self._lock:
            await asyncio.sleep(0.25)  # first-byte latency
            n = max(1, len(text) // 12)
            for _ in range(n):
                await asyncio.sleep(0.05)
                yield b"\x00\x00" * int(config.TTS_SAMPLE_RATE * 0.4)  # 0.4 s of silence


class MockLLM:
    model = "mock-llm"

    async def respond(self, user_text, on_delta, ts, now):
        ts["t_llm_request"] = now()
        await asyncio.sleep(0.3)
        ts["t_tool_start"] = now()
        r = tools.get_scholarship_status("S1004")
        ts["t_tool_end"] = now()
        ts["t_llm_final_request"] = now()
        await asyncio.sleep(0.2)
        text = ("S ek zero zero chaar ke liye aapki scholarship abhi under review hai. "
                "Agla step document verification hai, jo paccees hazaar rupaye ki hai.")
        ts["t_llm_first_token"] = now()
        for w in text.split(" "):
            await asyncio.sleep(0.04)
            await on_delta(w + " ")
        return {"text": text, "tool_calls": [{"name": "get_scholarship_status",
                                              "args": {"roll_no": "S1004"}, "result": r}], "model": self.model}
