# Where does the time go in a Hinglish voice agent?

*Gnani.ai Great Indian AI Internship Challenge entry. Gnani models used: **Prisma v3.0** (streaming
speech-to-text) and **Timbre v2.5** (text-to-speech). All student data is synthetic.*

**TL;DR:** Over 172 measured requests on 20 real recordings:
- **Streaming the reply audio** cut the median silence before the answer from **5.36 s to 3.29 s**.
- **An instant spoken acknowledgement** fills the remaining gap from the first moment.
- **A second iteration**, driven by the first benchmark's failures, raised end-to-end correctness from
  **83% to 95%** of requests.

## 1. Problem and user

A college student would rather ask *"meri scholarship ka status kya hai?"* out loud than dig through
a portal. Voice agents feel slow when every stage waits for the previous one to finish, and most demos
hide that dead air. This project measures it, then reduces it.

## 2. What it does and how

The agent answers scholarship and fee status questions in Hinglish, over 10 synthetic students. One
code path runs in two modes, chosen by one flag:

```
wav (real time) ─► Prisma STT ─► roll-no parser ─► LLM + tools ─► sentence chunker ─► Timbre TTS ─► playback
   ("Haan ji, ek second" plays here: --ack)         get_scholarship_status / get_fee_status
t_end_of_speech   t_asr_final    t_llm_request / first_token / first_sentence   t_tts_first_byte   t_first_audio_out
```

- **baseline:** wait for the final transcript, then the full LLM reply, then the full TTS audio, then play.
- **streamed:** the reply is split into sentences. Sentence 1 goes to TTS as soon as it is complete, and
  playback starts on its first audio bytes while later sentences synthesise.

The LLM is Groq `openai/gpt-oss-120b` (reasoning effort low, temperature 0). It decides the intent and
calls one of two tools, and if no roll number was given it asks once instead of guessing. Reading the
roll number back is a demo mechanism, not authentication.

## 3. What was measured

Every request writes one JSON record with monotonic timestamps from one machine.

| metric | definition |
|---|---|
| asr_latency | t_asr_final − t_end_of_speech |
| llm_ttft | first answer token − first LLM request (includes the tool-deciding round trip) |
| llm_first_sentence | first complete sentence − first LLM request |
| tts_ttfb | first audio byte − TTS request (first chunk) |
| **ttfa** (headline) | first **answer** audio out − t_end_of_speech |
| ttfa_ack | acknowledgement audio out − t_end_of_speech (only with `--ack`, reported separately) |
| total_response_time | t_last_audio_end − t_end_of_speech |
| playback_gap_total | total underrun (stutter) time once the answer has started |

Caveats on the definitions:
- **Benchmark input is wav files, so `t_end_of_speech` is the moment the last audio frame is sent.**
  Files were streamed at real-time pace and trimmed to 0.2 s of silence after speech. This is not
  live-microphone latency.
- **Playback is simulated on a real-time clock**, so there is no audio-device latency.
- **Prisma's streaming API returns no partial transcripts**, so ASR behaves the same in both modes.

**Test set:** 20 utterances, recorded by one speaker: 14 Hinglish, 3 Hindi, 3 English, 5 without a
roll number. Recordings came from mixed apps (WAV, Opus voice notes, MP3) and were converted to
16 kHz mono.

**Runs:** 10 Nov 2026, Windows 11 laptop, mobile hotspot, India. Baseline and streamed ran
back-to-back per utterance, with the order alternating, so network drift does not favour either mode.

| run | code | requests | when |
|---|---|---|---|
| v1 `20261110-181229` | first version | 92 of 120 planned (stopped by low memory on the machine) | 18:12 IST |
| v2 `20261110-203730` | after fixing v1's failures (section 6) | 80 (20 × 2 modes × 2 repeats) | 20:37 IST |

**v1 and v2 ran about 2.5 hours apart on the same mobile hotspot, so network conditions were not
identical.** Treat the v1-to-v2 latency comparison as indicative. The baseline-vs-streamed comparison
within each run is controlled.

## 4. Results

| stage, p50 (p95), seconds | v1 baseline | v1 streamed | **v2 baseline** | **v2 streamed** |
|---|---|---|---|---|
| asr_latency | 0.65 (1.11) | 0.64 (0.83) | 0.65 (1.18) | 0.65 (1.09) |
| llm_first_sentence | 1.78 (3.88) | 1.51 (3.60) | 1.64 (3.75) | 1.56 (4.80) |
| tts_ttfb | 1.37 (1.63) | 1.37 (1.85) | **0.87 (1.20)** | **0.87 (1.12)** |
| **ttfa** | 5.66 (8.71) | 3.64 (6.07) | 5.36 (8.65) | **3.29 (6.23)** |
| total_response_time | 12.25 (16.74) | 10.66 (14.56) | 11.14 (17.17) | 9.98 (14.53) |
| playback stutter | 0.00 | 0.15 (0.53) | 0.00 | 0.29 (0.70) |
| n (requests with audio) | 43 | 43 | 40 | 40 |

**With `--ack`,** a pre-synthesised *"Haan ji, ek second."* starts playing at end of speech, at
**ttfa_ack ≈ 0.00 s**, so the user hears a response immediately while the real answer is prepared. It
does not make the answer itself faster, so it is reported apart from ttfa.

**Where baseline TTFA goes** (v2 medians): ASR finalisation 0.65 s (12%), LLM 1.67 s (31%), full TTS
synthesis 3.02 s (56%).

**Correctness,** reported as counts, not as a general accuracy claim:

| check | v1 | **v2** |
|---|---|---|
| roll number recovered correctly | 65 of 72 | **60 of 60** |
| correct tool and roll number (or correctly asked for a missing one) | 82 of 92 | **76 of 80** |
| a reply was spoken | 86 of 92 | **80 of 80** |
| **end-to-end correct** | **76 of 92 (83%)** | **76 of 80 (95%)** |
| utterances correct in every run | 16 of 20 | **19 of 20** |

Raw per-request records and the full analysis are in `results/<run>/requests.jsonl` and
`analysis.txt`.

## 5. What the numbers say

- **H1, "streamed has lower TTFA on every utterance": false as stated, true in 35 of 40 pairs (v2)
  and 39 of 43 pairs (v1).** The median saving was **2.23 s** in v2. In every pair where streamed lost,
  that run's LLM happened to be slower, with a first sentence of up to 4.8 s against 1.1 s. Variance in
  the hosted LLM outweighs the pipeline difference on an unlucky request.
- **H2, "the biggest share of baseline TTFA is LLM plus full TTS, not ASR": supported.** Together they
  are 87% of baseline TTFA, and ASR is 12%.
- **H3, "chunking helps mainly by removing the wait for the full LLM response": false.** The time
  between the first sentence and the end of the LLM reply was **0.01 s at the median**, because Groq
  emits the short reply in one burst after hidden reasoning. The saving comes almost entirely from
  **playing TTS audio as it streams in**, which removes 2.11 s of waiting for the full file.
- **Reusing one TTS connection** instead of a new TCP and TLS handshake per call cut Timbre's
  time-to-first-byte from 1.37 s to 0.87 s at the median (−36%). Network drift between runs may
  account for part of this.
- **Streaming is not free.** Playback stuttered in 38 of 40 streamed runs, by 0.29 s at the median in
  v2, up from 0.15 s in v1. Timbre's stream sometimes arrives slower than real time, and starting
  playback sooner leaves less buffer.

## 6. Failures, and what v2 changed

| v1 failure | cause | v2 fix | v2 result |
|---|---|---|---|
| "scholarship **and** fees" gave **no reply** (6 of 6 runs) | my tool loop allowed only 2 LLM rounds | 4 rounds | answered, 4 of 4 |
| "S one thousand two" was looked up as **S1004** (4 of 4) | the prompt's example roll number biased the LLM | **deterministic roll-number parser in code** (English and Hindi number words, Latin and Devanagari) | 60 of 60 roll numbers correct |
| "S. One thousand four" not recognised (3 of 4) | the LLM could not parse it | same parser | correct |
| reply said "**aath** hazaar" (8,000) for 18,000 | the LLM converts numbers to words unreliably | **tools return amounts and dates already in spoken Hinglish words**, and the LLM copies them | correct in all spot checks |
| a new TTS connection on every call | implementation | one keep-alive client, warmed at start-up | TTFB −36% |

Still failing, or not done:
1. **ASR mis-hearing, 4 of 4 v2 runs:** *"fee ka status"* is transcribed as **"FIFA status"**, and the
   agent then answers about the scholarship. The cause is in speech recognition, not the agent.
   Prisma's keyword boosting (`keyterms`) is the next thing to try.
2. **Endpointing:** for 2 recordings with background noise to the end of the file, Prisma's VAD closed
   the utterance 1–2 s before the audio ended, giving a negative `asr_latency`. For those files TTFA
   understates the speaker's real wait. Both modes are affected equally.
3. **Not done:** the noisy condition (`make_noisy.py` prepares it; not run for lack of time), the
   browser microphone page, and a confirmation turn before the lookup.

## 7. What I would change with more time

- A small **jitter buffer**, waiting 200–300 ms of audio before playback, to remove the stutter at a
  measured cost in TTFA.
- **Prisma `keyterms`** for domain words (fees, scholarship, roll number) to fix "FIFA status".
- **Start the LLM the moment Prisma's final transcript arrives**, and test a smaller or more consistent
  LLM. The LLM is now the largest stage in streamed mode (47%), and its variance causes every case
  where streaming lost.
- Run the noisy condition, more speakers, and live-microphone timing.

---
*Synthetic-data statement: all student names, roll numbers, scholarship and fee records in this
project are synthetic. No real personal data was used. Recordings are of the author reading the
synthetic test lines.*
