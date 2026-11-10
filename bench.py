"""Benchmark: every utterance x both modes x N repeats, baseline/streamed interleaved
(order alternates per pair so network drift favours neither). Writes results/<run>/{requests.jsonl,
results.csv, meta.json}.

  python bench.py --manifest testset/manifest.csv --repeats 3 --network "home wifi" --region "Bengaluru"
  python bench.py --manifest testset/manifest.csv --repeats 1 --limit 2     # check credit cost first
  python bench.py --manifest testset/manifest.csv --mock                    # offline smoke test only

manifest.csv columns: file,condition,intent,roll_no,note
  intent: scholarship | fee | none (agent should ask for roll number); roll_no: expected, may be empty
"""
import argparse
import asyncio
import csv
import datetime
import json
import platform
import random
import sys
from pathlib import Path

from voiceagent import config
from voiceagent.pipeline import run_request
from run_one import build, warm

sys.stdout.reconfigure(encoding="utf-8")
TOOL_FOR = {"scholarship": "get_scholarship_status", "fee": "get_fee_status"}
METRICS = ["asr_latency", "handoff", "llm_ttft", "tool_latency", "llm_first_sentence", "llm_total", "tts_ttfb",
           "ttfa", "ttfa_ack", "total_response_time", "playback_gap_total", "playback_n_gaps", "n_tts_chunks"]


def correctness(row, rec):
    """Automatic parts of the small correctness check. Report as counts ('x of n'), not accuracy."""
    calls = rec["tool_calls"]
    exp_tool, exp_roll = TOOL_FOR.get(row["intent"]), row["roll_no"].strip().upper()
    # Prisma writes spoken roll numbers as words, so the check is whether the LLM recovered the right
    # number from the transcript (None when the utterance has no roll number).
    roll_ok = None
    if exp_roll:
        rolls = {c["args"].get("roll_no", "").upper() for c in calls}
        roll_ok = rolls == {exp_roll}
    if row["intent"] == "none" or not exp_roll:
        tool_ok = len(calls) == 0  # should have asked instead of guessing
    else:
        tool_ok = any(c["name"] == exp_tool and c["args"].get("roll_no", "").upper() == exp_roll for c in calls)
    spoken = bool(rec["reply"].strip())
    return {"transcript_nonempty": bool(rec["transcript"].strip()), "roll_correct": roll_ok,
            "tool_call_correct": tool_ok, "answer_spoken": spoken, "end_to_end_ok": tool_ok and spoken}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--network", default="unknown")
    ap.add_argument("--region", default="unknown")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.manifest, encoding="utf-8")))[: a.limit]
    base = Path(a.manifest).parent
    stt, tts, llm = build(a.mock)
    await warm(tts)
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ("-MOCK" if a.mock else "")
    out = config.RESULTS_DIR / run_id
    out.mkdir(parents=True)
    rng = random.Random(a.seed)

    meta = {"run_id": run_id, "started": datetime.datetime.now().astimezone().isoformat(),
            "machine": platform.platform(), "python": platform.python_version(),
            "network": a.network, "region": a.region, "mock": a.mock, "repeats": a.repeats,
            "n_utterances": len(rows), "stt_model": config.STT_MODEL, "tts_model": config.TTS_MODEL,
            "tts_voice": config.TTS_VOICE, "llm_model": getattr(llm, "model", None),
            "llm_temperature": config.LLM_TEMPERATURE,
            "note": "t_end_of_speech = last audio frame sent (wav benchmark, no live endpointing). "
                    "Playback is simulated (real-time clock), no audio device latency."}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    jf = open(out / "requests.jsonl", "w", encoding="utf-8")
    cf = open(out / "results.csv", "w", newline="", encoding="utf-8")
    cols = ["run_id", "repeat", "file", "condition", "intent", "mode"] + METRICS + \
           ["transcript_nonempty", "roll_correct", "tool_call_correct", "answer_spoken", "end_to_end_ok", "n_errors", "transcript", "reply"]
    cw = csv.DictWriter(cf, fieldnames=cols)
    cw.writeheader()

    total = len(rows) * 2 * a.repeats
    done = 0
    for rep in range(a.repeats):
        order = rows[:]
        rng.shuffle(order)
        for i, row in enumerate(order):
            modes = ["baseline", "streamed"] if (i + rep) % 2 == 0 else ["streamed", "baseline"]
            for mode in modes:  # adjacent in time -> same network conditions
                wav = base / row["file"]
                rec = await run_request(wav, mode, stt, tts, llm, mock=a.mock,
                                        extra={"run_id": run_id, "repeat": rep, "condition": row["condition"],
                                               "intent": row["intent"]})
                rec.pop("_audio", None)
                c = correctness(row, rec)
                rec["correctness"] = c
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n"); jf.flush()
                cw.writerow({"run_id": run_id, "repeat": rep, "file": row["file"], "condition": row["condition"],
                             "intent": row["intent"], "mode": mode, **rec["metrics"], **c,
                             "n_errors": len(rec["errors"]), "transcript": rec["transcript"], "reply": rec["reply"]})
                cf.flush()
                done += 1
                print(f"[{done}/{total}] {row['file']} {mode:8s} ttfa={rec['metrics']['ttfa']} err={len(rec['errors'])}")
    from voiceagent.adapters import PrismaSTT
    await PrismaSTT.drain()
    print("wrote", out)


if __name__ == "__main__":
    asyncio.run(main())
