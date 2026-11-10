"""Noisy condition (PRD section 9: "noise mixed in digitally"): testset/clean_XX.wav -> testset/noisy_XX.wav
and testset/manifest_noisy.csv.

Noise = synthetic fan-like noise: brown (1/f^2) noise + broadband white component + a 100 Hz motor hum
with harmonics, mixed at a fixed SNR measured against the speech portion of each file. Deterministic
(seeded), so the noisy set is reproducible. It is NOT real room noise; say so in the writeup.

  python make_noisy.py [--snr 10]
"""
import argparse
import csv
import glob
import os
import wave

import numpy as np

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testset")
SR = 16000


def fan_noise(n, rng):
    brown = np.cumsum(rng.standard_normal(n))
    brown -= np.convolve(brown, np.ones(1601) / 1601, mode="same")  # remove drift (high-pass ~10 Hz)
    brown /= brown.std() + 1e-9
    white = rng.standard_normal(n)
    t = np.arange(n) / SR
    hum = sum(np.sin(2 * np.pi * 100 * k * t + rng.uniform(0, 6.28)) / k for k in (1, 2, 3))
    x = brown + 0.35 * white + 0.25 * hum / np.std(hum)
    return x / x.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=float, default=10.0)
    a = ap.parse_args()
    rng = np.random.default_rng(42)
    for f in sorted(glob.glob(os.path.join(DIR, "clean_*.wav"))):
        with wave.open(f) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(float)
        fr = x[: len(x) // 320 * 320].reshape(-1, 320)
        e = (fr ** 2).mean(1)
        speech_pow = e[e > e.max() * 10 ** (-3.5)].mean()  # power of speech frames (within 35 dB of peak)
        noise = fan_noise(len(x), rng) * np.sqrt(speech_pow / 10 ** (a.snr / 10))
        y = x + noise
        y *= min(1.0, 32000 / np.abs(y).max())  # avoid clipping
        out = f.replace("clean_", "noisy_")
        with wave.open(out, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes(y.astype("<i2").tobytes())
    rows = list(csv.DictReader(open(os.path.join(DIR, "manifest.csv"), encoding="utf-8")))
    with open(os.path.join(DIR, "manifest_noisy.csv"), "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=rows[0].keys())
        w.writeheader()
        for r in rows:
            w.writerow({**r, "file": r["file"].replace("clean_", "noisy_"), "condition": f"noisy_fan_snr{a.snr:g}"})
    print(f"wrote {len(rows)} noisy files at SNR {a.snr:g} dB + manifest_noisy.csv")


if __name__ == "__main__":
    main()
