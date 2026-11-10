"""Sentence chunker for streamed LLM text. Emits a chunk at a sentence end once it is
long enough; short fragments are merged into the next sentence to avoid choppy TTS."""
import re

_END = re.compile(r"[.!?।\n]+(?=\s|$)")


class SentenceChunker:
    def __init__(self, min_chars=20):
        self.buf = ""
        self.min_chars = min_chars

    def feed(self, delta: str):
        self.buf += delta
        out = []
        while True:
            for m in _END.finditer(self.buf):
                if m.end() >= self.min_chars:
                    out.append(self.buf[:m.end()].strip())
                    self.buf = self.buf[m.end():]
                    break
            else:
                break
        return [c for c in out if c]

    def flush(self):
        rest, self.buf = self.buf.strip(), ""
        return [rest] if rest else []
