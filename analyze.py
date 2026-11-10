"""Post-hoc analysis of one benchmark run from its requests.jsonl (+ the manifest for expected answers).
Re-scores correctness with the current checks and decomposes TTFA for H1-H3.

  python analyze.py results/<run> [--manifest testset/manifest.csv]
"""
import argparse
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
from bench import correctness  # noqa: E402
from summarize import pct  # noqa: E402


def med(xs):
    return st.median(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--manifest", default="testset/manifest.csv")
    a = ap.parse_args()
    man = {r["file"]: r for r in csv.DictReader(open(a.manifest, encoding="utf-8"))}
    R = [json.loads(l) for l in open(Path(a.run) / "requests.jsonl", encoding="utf-8")]
    meta = json.loads((Path(a.run) / "meta.json").read_text(encoding="utf-8"))
    for r in R:
        r["file"] = Path(r["wav"]).name
        r["c"] = correctness(man[r["file"]], r)

    print(f"run {meta['run_id']}  {meta['started']}  network={meta['network']}  llm={meta['llm_model']}  "
          f"stt={meta['stt_model']}  tts={meta['tts_model']}/{meta['tts_voice']}")
    files = sorted({r["file"] for r in R})
    print(f"requests={len(R)}  utterances={len(files)}  errors={sum(bool(r['errors']) for r in R)}")
    reps = defaultdict(int)
    for r in R:
        reps[r["file"]] += 1
    print("runs per utterance (both modes):", dict(sorted(defaultdict(int, {k: v for k, v in reps.items()}).items())))

    # ---------- correctness, as counts ----------
    print("\n== correctness (counts over requests; and over utterances where all runs agree)")
    for k in ("transcript_nonempty", "roll_correct", "tool_call_correct", "answer_spoken", "end_to_end_ok"):
        v = [r["c"][k] for r in R if r["c"][k] is not None]
        per_utt = defaultdict(list)
        for r in R:
            if r["c"][k] is not None:
                per_utt[r["file"]].append(r["c"][k])
        always = sum(all(x) for x in per_utt.values())
        print(f"  {k:20s} {sum(v):3d} of {len(v):3d} requests | utterances always ok: {always} of {len(per_utt)}")
    print("  failing utterances:")
    seen = set()
    for r in R:
        if not r["c"]["end_to_end_ok"] and r["file"] not in seen:
            seen.add(r["file"])
            bad = [x for x in R if x["file"] == r["file"]]
            nbad = sum(not x["c"]["end_to_end_ok"] for x in bad)
            print(f"    {r['file']} ({nbad}/{len(bad)} runs) heard: {r['transcript']} | tools: "
                  f"{[(c['name'], c['args'].get('roll_no')) for c in r['tool_calls']]} | reply: {r['reply'][:60]!r}")

    # ---------- latency table (requests that produced audio) ----------
    ok = [r for r in R if r["metrics"]["ttfa"] is not None]
    print(f"\n== latency, seconds (n = requests with audio out; {len(R) - len(ok)} without audio excluded)")
    stages = ["asr_latency", "llm_ttft", "llm_first_sentence", "llm_total", "tts_ttfb", "ttfa",
              "total_response_time", "playback_gap_total"]
    print(f"  {'stage':22s} {'mode':9s} {'n':>3s} {'p50':>6s} {'p95':>6s} {'min':>6s} {'max':>6s}")
    for s in stages:
        for m in ("baseline", "streamed"):
            xs = [r["metrics"][s] for r in ok if r["mode"] == m and r["metrics"][s] is not None]
            print(f"  {s:22s} {m:9s} {len(xs):3d} {pct(xs,50):6.2f} {pct(xs,95):6.2f} {min(xs):6.2f} {max(xs):6.2f}")

    # ---------- TTFA decomposition ----------
    # baseline ttfa = [eos->llm start] + llm_total + full TTS (request -> all audio back)
    # streamed ttfa = [eos->llm start] + llm_first_sentence + [queue] + tts_ttfb (first chunk)
    def parts(r):
        t, m = r["timestamps"], r["metrics"]
        pre = t["t_llm_request"] - t["t_end_of_speech"]
        if r["mode"] == "baseline":
            return {"to_llm_start": pre, "llm": t["t_llm_done"] - t["t_llm_request"],
                    "tts": t["t_first_audio_out"] - t["t_tts_request"]}
        return {"to_llm_start": pre, "llm": t["t_llm_first_sentence"] - t["t_llm_request"],
                "tts": t["t_first_audio_out"] - t["t_llm_first_sentence"]}

    print("\n== TTFA decomposition, medians (s) and share of median TTFA")
    for m in ("baseline", "streamed"):
        P = [parts(r) for r in ok if r["mode"] == m]
        tt = med([r["metrics"]["ttfa"] for r in ok if r["mode"] == m])
        line = "  ".join(f"{k}={med([p[k] for p in P]):.2f} ({100*med([p[k] for p in P])/tt:.0f}%)" for k in P[0])
        print(f"  {m:9s} ttfa={tt:.2f}  {line}")
    print("  (to_llm_start = ASR finalisation after end of audio; tts = baseline: full synthesis; "
          "streamed: first chunk incl. queueing)")

    # ---------- paired comparison ----------
    pairs = defaultdict(dict)
    for r in ok:
        pairs[(r["file"], r["repeat"])][r["mode"]] = r
    both = [v for v in pairs.values() if len(v) == 2]
    d = [v["baseline"]["metrics"]["ttfa"] - v["streamed"]["metrics"]["ttfa"] for v in both]
    llm_saved = [(v["baseline"]["metrics"]["llm_total"] - v["baseline"]["metrics"]["llm_first_sentence"]) for v in both]
    tts_saved = [parts(v["baseline"])["tts"] - v["baseline"]["metrics"]["tts_ttfb"] for v in both]
    print(f"\n== H1: streamed TTFA lower in {sum(x > 0 for x in d)} of {len(both)} pairs; "
          f"saving p50={med(d):.2f}s p95={pct(d,95):.2f} min={min(d):.2f} max={max(d):.2f}")
    print(f"   within baseline runs, what streaming could remove (medians): waiting for the rest of the LLM "
          f"response after sentence 1 = {med(llm_saved):.2f}s; waiting for full TTS audio instead of first "
          f"byte = {med(tts_saved):.2f}s")
    print("   slower pairs:")
    for v in both:
        b, s = v["baseline"]["metrics"], v["streamed"]["metrics"]
        if s["ttfa"] > b["ttfa"]:
            print(f"    {v['baseline']['file']} rep{v['baseline']['repeat']}: ttfa b={b['ttfa']:.2f} s={s['ttfa']:.2f}; "
                  f"llm_first_sentence b={b['llm_first_sentence']:.2f} s={s['llm_first_sentence']:.2f}")
    neg = [r for r in R if r["metrics"]["asr_latency"] is not None and r["metrics"]["asr_latency"] < 0]
    print(f"\n== ASR final before end of audio (negative asr_latency): {len(neg)} requests, files "
          f"{sorted({r['file'] for r in neg})}")
    chunks = [r["metrics"]["n_tts_chunks"] for r in ok if r["mode"] == "streamed"]
    gaps = [r["metrics"]["playback_gap_total"] for r in ok if r["mode"] == "streamed"]
    print(f"== streamed: replies split into >1 TTS chunk in {sum(c > 1 for c in chunks)} of {len(chunks)}; "
          f"playback gap>0 in {sum(g > 0 for g in gaps)} of {len(gaps)}, gap p50={med(gaps):.2f}s max={max(gaps):.2f}s")


if __name__ == "__main__":
    main()
