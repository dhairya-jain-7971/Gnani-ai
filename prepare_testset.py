"""Convert the raw recordings in testset/originals/ to benchmark wavs and print a QC table.

  python prepare_testset.py            # originals/Clean_07.ogg -> testset/clean_07.wav

Each file: ffmpeg -> 16 kHz mono s16le, then leading/trailing silence trimmed to fixed pads
(LEAD_PAD before speech, TRAIL_PAD after). Why: in benchmark mode t_end_of_speech is "last audio
frame sent", so variable trailing silence would shift that clock by up to ~1.5 s per file, and
silence long enough for Prisma's VAD to close the segment before the commit would make the
utterance end before t_end_of_speech. With trimming, end-of-file ~= end-of-speech for every file.
Originals are never modified.
"""
import glob
import os
import subprocess
import sys
import wave

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testset")
SR, FRAME = 16000, 320  # 20 ms analysis frames
LEAD_PAD, TRAIL_PAD = 0.30, 0.20
REL_DB, FLOOR_DB = 35, -55  # speech = frames within 35 dB of the loudest frame (and above -55 dBFS)


def decode(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, "<i2")


def speech_bounds(x):
    fr = x[: len(x) // FRAME * FRAME].astype(float).reshape(-1, FRAME) / 32768
    e = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-9)
    v = np.where(e > max(e.max() - REL_DB, FLOOR_DB))[0]
    return (v[0] * FRAME, (v[-1] + 1) * FRAME) if len(v) else (0, len(x))


def main():
    srcs = sorted(glob.glob(os.path.join(DIR, "originals", "*")))
    print(f"{'file':14s} {'src':6s} {'orig_s':>6s} {'trim_s':>6s} {'cut_lead':>8s} {'cut_trail':>9s} {'peak':>5s} {'clip%':>5s}")
    for src in srcs:
        stem = os.path.splitext(os.path.basename(src))[0].lower()  # Clean_07 -> clean_07
        x = decode(src)
        a, b = speech_bounds(x)
        s, e = max(0, a - int(LEAD_PAD * SR)), min(len(x), b + int(TRAIL_PAD * SR))
        y = x[s:e]
        with wave.open(os.path.join(DIR, stem + ".wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(y.tobytes())
        pk = np.abs(y.astype(float)).max() / 32768
        print(f"{stem + '.wav':14s} {os.path.splitext(src)[1][1:]:6s} {len(x)/SR:6.2f} {len(y)/SR:6.2f} "
              f"{s/SR:8.2f} {(len(x)-e)/SR:9.2f} {pk:5.2f} {100*(np.abs(y) > 32440).mean():5.2f}")
    print(f"\n{len(srcs)} files written to {DIR}")


if __name__ == "__main__":
    main()
