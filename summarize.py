"""Print p50/p95 per stage per mode (plus n and min/max) from a results.csv, and the paired
per-utterance comparison needed for H1. Small n: look at the raw distribution, not only p95.
  python summarize.py results/<run>/results.csv [--condition clean]
"""
import argparse
import csv
import statistics as st
from collections import defaultdict

STAGES = ["asr_latency", "handoff", "llm_ttft", "tool_latency", "llm_first_sentence", "llm_total", "tts_ttfb",
          "ttfa", "total_response_time", "playback_gap_total", "n_tts_chunks"]


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--condition")
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.csv, encoding="utf-8"))
            if not a.condition or r["condition"] == a.condition]
    ok = [r for r in rows if r["n_errors"] == "0"]
    print(f"{len(rows)} requests, {len(ok)} error-free (errors excluded from latency stats; report them)\n")
    print(f"{'stage':22s} {'mode':9s} {'n':>4s} {'p50':>8s} {'p95':>8s} {'min':>8s} {'max':>8s}")
    for s in STAGES:
        for mode in ("baseline", "streamed"):
            xs = [float(r[s]) for r in ok if r["mode"] == mode and r[s] not in ("", "None")]
            if xs:
                print(f"{s:22s} {mode:9s} {len(xs):4d} {pct(xs,50):8.3f} {pct(xs,95):8.3f} {min(xs):8.3f} {max(xs):8.3f}")
    # H1: per (file, repeat) pair, streamed ttfa < baseline ttfa ?
    pairs = defaultdict(dict)
    for r in ok:
        if r["ttfa"] not in ("", "None"):
            pairs[(r["file"], r["repeat"])][r["mode"]] = float(r["ttfa"])
    both = [v for v in pairs.values() if len(v) == 2]
    if both:
        wins = sum(v["streamed"] < v["baseline"] for v in both)
        d = [v["baseline"] - v["streamed"] for v in both]
        print(f"\nH1: streamed ttfa < baseline in {wins} of {len(both)} pairs; "
              f"saving p50={st.median(d):.3f}s min={min(d):.3f}s max={max(d):.3f}s")
    n = len(rows)
    if n:
        for c in ("transcript_nonempty", "roll_correct", "tool_call_correct", "answer_spoken", "end_to_end_ok"):
            vals = [r[c] for r in rows if r[c] not in ("", "None")]
            print(f"{c}: {sum(v == 'True' for v in vals)} of {len(vals)}")


if __name__ == "__main__":
    main()
