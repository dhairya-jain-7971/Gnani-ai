"""WAV loading and framing for Prisma (16 kHz, mono, s16le, 1024-byte frames)."""
import wave

import numpy as np

STT_RATE = 16000
FRAME_BYTES = 1024  # 512 samples -> 32 ms at 16 kHz (per Prisma docs)
FRAME_SECONDS = FRAME_BYTES / 2 / STT_RATE


def load_wav_16k_mono(path) -> bytes:
    with wave.open(str(path), "rb") as w:
        ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError(f"{path}: need 16-bit PCM, got {8*sw}-bit")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != STT_RATE:  # linear resample; fine for speech benchmarking
        t_new = np.arange(int(len(x) * STT_RATE / sr)) * (sr / STT_RATE)
        x = np.interp(t_new, np.arange(len(x)), x)
    return np.clip(x, -32768, 32767).astype("<i2").tobytes()


def frames(pcm: bytes):
    for i in range(0, len(pcm), FRAME_BYTES):
        f = pcm[i:i + FRAME_BYTES]
        if len(f) < FRAME_BYTES:
            f = f + b"\x00" * (FRAME_BYTES - len(f))  # docs: frames are exactly 1024 bytes
        yield f


def pcm_seconds(nbytes: int, sample_rate: int) -> float:
    return nbytes / 2 / sample_rate


def write_wav(path, pcm: bytes, sample_rate: int):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
