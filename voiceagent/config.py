"""Configuration. Reads a .env file (KEY=VALUE lines) then the process environment."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv():
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def env(name, default=None):
    return os.environ.get(name, default)


# --- Gnani (endpoints verified against docs.gnani.ai on 2026-11-10) ---
GNANI_API_KEY = env("GNANI_API_KEY")
STT_WS_URL = env("STT_WS_URL", "wss://api.vachana.ai/stt/v3/stream")
TTS_SSE_URL = env("TTS_SSE_URL", "https://api.vachana.ai/api/v1/tts/sse")
STT_MODEL = env("STT_MODEL", "prisma-v3.0")
# Live server (2026-11-10) rejects lists despite the docs: "exactly one locale is required".
STT_LANGUAGE = env("STT_LANGUAGE", "hi-IN")
STT_FORMAT = env("STT_FORMAT", "codemix")  # codemix is v3.0-only; verify it suits roll numbers
TTS_MODEL = env("TTS_MODEL", "timbre-v2.5")
TTS_VOICE = env("TTS_VOICE", "Nalini")
TTS_LANGUAGE = env("TTS_LANGUAGE", "hi-en")  # Hinglish
TTS_SAMPLE_RATE = int(env("TTS_SAMPLE_RATE", "24000"))

# --- LLM (any one hosted model with tool calling) ---
LLM_PROVIDER = env("LLM_PROVIDER")  # "anthropic" | "openai"; auto-detected from keys if unset
LLM_MODEL = env("LLM_MODEL")
LLM_REASONING_EFFORT = env("LLM_REASONING_EFFORT", "low")  # only sent to reasoning models (gpt-oss)
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 800  # includes hidden reasoning tokens on gpt-oss
ANTHROPIC_API_KEY = env("ANTHROPIC_API_KEY")
OPENAI_API_KEY = env("OPENAI_API_KEY")
OPENAI_BASE_URL = env("OPENAI_BASE_URL")  # set for OpenAI-compatible providers (Groq, Gemini, ...)

DATA_PATH = ROOT / "data" / "students.json"
RESULTS_DIR = ROOT / "results"
