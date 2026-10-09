"""CPU cost and accuracy of Moonshine Tiny vs Small for passive transcription.

Speech is synthesised in RAM with the Piper voice Kin ships (never written to disk), resampled to
16 kHz, and streamed into each Moonshine model in 32 ms frames, as the passive pipeline does.
Reported per model: CPU seconds per second of audio (1.00 = one core fully busy in real time),
wall time, RAM added by loading it, and word error rate against the script.

Usage:
  uv run python scripts/stt_cpu_probe.py                 # tiny + small, laptop thread settings
  uv run python scripts/stt_cpu_probe.py --threads 4     # restrict onnxruntime threads (Pi-like)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SENTENCES = [
    "Priyanka is going to be my mentor for the new analytics project.",
    "Let's meet tomorrow at four at the library cafe.",
    "The office wifi password is saffron twenty twenty six.",
    "I paid eighteen hundred and fifty rupees for the electricity bill this month.",
    "Rahul said he is vegan now, so no paneer or ghee for him.",
    "My dentist appointment with Doctor Mehta moved to next Wednesday at six.",
    "The quarterly review got moved to Tuesday the thirteenth at three.",
    "Kavya is a doctor at AIIMS Delhi and she is allergic to peanuts.",
    "Ananya's flight to Edinburgh leaves on the fifth of January at six in the morning.",
    "We could turn the spare room into a small home office.",
]


def _words(t: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", t.lower())


def wer(ref: str, hyp: str) -> tuple[int, int]:
    r, h = _words(ref), _words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="tiny-streaming,small-streaming")
    ap.add_argument("--threads", type=int, default=None, help="OMP/onnxruntime threads (Pi 5: 4)")
    args = ap.parse_args()
    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)

    import numpy as np
    import psutil
    from scipy.signal import resample_poly

    from kin.audio.stt import MoonshineSTT, find_moonshine_model
    from kin.config import load_config
    from kin.io.tts import PiperTTS

    cfg = load_config("laptop")
    tts = PiperTTS(cfg.model_path(cfg.tts.voice))
    clips = []
    for s in SENTENCES:
        a = tts.synthesize(s)
        a = resample_poly(a, 160, tts.sample_rate // 100).astype(np.float32)
        a = 0.5 * a / (np.abs(a).max() or 1.0)
        clips.append(np.concatenate([np.zeros(4800, np.float32), a, np.zeros(9600, np.float32)]))
    audio_s = sum(len(c) for c in clips) / 16000
    proc = psutil.Process()
    results = {}
    for arch in args.models.split(","):
        rss0 = proc.memory_info().rss
        stt = MoonshineSTT(find_moonshine_model(cfg.model_path(cfg.stt.models_dir), arch), arch, cfg.stt.update_interval_s)
        rss1 = proc.memory_info().rss
        errors = words = 0
        heard = []
        cpu0, t0 = proc.cpu_times(), time.perf_counter()
        for ref, clip in zip(SENTENCES, clips):
            lines: list[str] = []
            stt.start(lambda line: lines.append(line.text) if line.final else None)
            for i in range(0, len(clip), 512):
                stt.feed(clip[i : i + 512])
            stt.stop()
            hyp = " ".join(lines)
            heard.append(hyp)
            e, n = wer(ref, hyp)
            errors, words = errors + e, words + n
        cpu1, wall = proc.cpu_times(), time.perf_counter() - t0
        cpu_s = (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)
        stt.close()
        results[arch] = {
            "audio_s": round(audio_s, 1),
            "cpu_s": round(cpu_s, 1),
            "cpu_per_audio_s": round(cpu_s / audio_s, 3),
            "wall_s": round(wall, 1),
            "real_time_factor": round(wall / audio_s, 3),
            "ram_mb": round((rss1 - rss0) / 2**20),
            "wer": round(errors / words, 3),
            "sample": heard[:3],
        }
        print(f"{arch}: {results[arch]}", flush=True)
    print("RESULT " + json.dumps({"cpu_count": psutil.cpu_count(), "threads": args.threads, "models": results}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
