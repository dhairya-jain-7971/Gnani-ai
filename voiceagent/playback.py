"""Simulated playback queue. Benchmark runs have no speaker: audio 'plays' at real time from
the moment it is available, gapless when the next audio is already queued. This yields
t_first_audio_out (first ANSWER audio), t_ack_audio_out (optional instant acknowledgement),
t_last_audio_end and underrun gaps (choppiness) without device-dependent latency."""
from .audio import pcm_seconds


class SimPlayback:
    def __init__(self, sample_rate, now):
        self.sr, self.now = sample_rate, now
        self.t_first = None   # first answer audio
        self.t_ack = None     # acknowledgement audio, if any
        self.end = None       # time at which currently queued audio finishes
        self.gaps = []        # underruns (s) after the answer started playing
        self.bytes = 0

    def push(self, pcm: bytes, ack: bool = False):
        if not pcm:
            return
        t = self.now()
        start = t if self.end is None or t > self.end else self.end  # queued audio waits its turn
        if self.end is not None and t > self.end and self.t_first is not None:
            self.gaps.append(t - self.end)  # underrun while the answer is playing
        if ack and self.t_ack is None:
            self.t_ack = start
        elif not ack and self.t_first is None:
            self.t_first = start  # audible start of the answer
        self.end = start + pcm_seconds(len(pcm), self.sr)
        self.bytes += len(pcm)

    @property
    def t_answer_start(self):
        return self.t_first

    @property
    def t_last_audio_end(self):
        return self.end
