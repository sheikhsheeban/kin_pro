"""Wake word probe: what does openWakeWord score for "Hey Jarvis" on this mic?

Opens the mic at 16 kHz, cuts the audio into 1280-sample (80 ms) chunks, converts each to
int16 with (x * 32767), feeds EVERY chunk to the stock hey_jarvis model, and prints the
highest score and the mic level for each second. Audio stays in RAM; nothing is saved.

  uv run python scripts/wake_probe.py                 # 30 s; say "Hey Jarvis" a few times
  uv run python scripts/wake_probe.py --seconds 60 --device 2
  uv run python scripts/wake_probe.py --play 10       # no voice needed: Piper says it through the speakers

Reading the output: a working setup shows ~0.00 in silence and 0.8-1.0 for "Hey Jarvis".
Peaks that stay under 0.5 mean the model hears you but not well enough (distance, mic, accent):
lower `wake.threshold` in the profile. A mic level below about -45 dBFS while you speak means
the mic gain is too low for any model.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kin.config import load_config  # noqa: E402

SR = 16000
CHUNK = 1280  # 80 ms, openWakeWord's native step


def to_int16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)


def dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0
    return 20 * np.log10(rms + 1e-9)


def speaker_says(times: int, gap_s: float, start_delay_s: float, stop: threading.Event) -> None:
    """Play "Hey Jarvis" through the default speakers, for a test without a human voice."""
    import sounddevice as sd

    from kin.io.tts import PiperTTS

    cfg = load_config("laptop")
    tts = PiperTTS(cfg.model_path(cfg.tts.voice))
    if stop.wait(start_delay_s):
        return
    for i in range(times):
        if stop.is_set():
            return
        audio = tts.synthesize("Hey Jarvis.")
        print(f"   [speaker] Hey Jarvis ({i + 1}/{times})", flush=True)
        sd.play(audio, tts.sample_rate, blocking=True)
        if stop.wait(gap_s):
            return


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="laptop")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--device", default=None, help="input device index or name (see `uv run kin devices`)")
    ap.add_argument("--threshold", type=float, default=None, help="defaults to the profile's wake.threshold")
    ap.add_argument("--play", type=int, default=0, metavar="N", help="say 'Hey Jarvis' N times through the speakers")
    args = ap.parse_args()

    import sounddevice as sd
    from openwakeword.model import Model

    cfg = load_config(args.profile)
    threshold = args.threshold if args.threshold is not None else cfg.wake.threshold
    device = int(args.device) if args.device is not None and str(args.device).isdigit() else args.device
    model = Model(
        wakeword_models=[str(cfg.model_path(cfg.wake.model))],
        inference_framework="onnx",
        melspec_model_path=str(cfg.model_path(cfg.wake.melspec_model)),
        embedding_model_path=str(cfg.model_path(cfg.wake.embedding_model)),
        ncpu=cfg.runtime.threads or 1,
    )
    key = next(iter(model.models))

    blocks: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, time_info, status) -> None:
        blocks.put(indata[:, 0].copy())

    name = sd.query_devices(device if device is not None else sd.default.device[0], "input")["name"]
    print(f"mic: {name} · 16 kHz mono · threshold {threshold} · {args.seconds:.0f} s")
    if not args.play:
        print('say "Hey Jarvis" at normal distance a few times, with pauses\n')
    print(" sec   max score   mic level (dBFS, mean / peak)")

    stop = threading.Event()
    player = None
    if args.play:
        player = threading.Thread(target=speaker_says, args=(args.play, 2.5, 2.0, stop), daemon=True)

    pending = np.zeros(0, np.int16)
    sec_scores: list[float] = []
    sec_audio: list[np.ndarray] = []
    peaks: list[float] = []
    above = 0
    cpu = 0.0
    t_end = time.monotonic() + args.seconds
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=CHUNK, device=device, callback=callback):
        if player is not None:
            player.start()
        second = 0
        while time.monotonic() < t_end:
            try:
                block = blocks.get(timeout=0.5)
            except queue.Empty:
                continue
            sec_audio.append(block)
            pending = np.concatenate([pending, to_int16(block)])
            while len(pending) >= CHUNK:
                chunk, pending = pending[:CHUNK], pending[CHUNK:]
                t0 = time.perf_counter()
                score = float(model.predict(chunk)[key])
                cpu += time.perf_counter() - t0
                sec_scores.append(score)
            if sum(len(a) for a in sec_audio) >= SR:
                second += 1
                audio = np.concatenate(sec_audio)
                best = max(sec_scores, default=0.0)
                peaks.append(best)
                above += best >= threshold
                bar = "█" * int(best * 20)
                mark = "  ← WAKE" if best >= threshold else ""
                print(f"{second:4d}   {best:6.3f}  {bar:<20}  {dbfs(audio):6.1f} / {20 * np.log10(np.abs(audio).max() + 1e-9):6.1f}{mark}",
                      flush=True)
                sec_scores, sec_audio = [], []
    stop.set()

    print(f"\nseconds at or above {threshold}: {above} · best score {max(peaks, default=0):.3f} · "
          f"wake model CPU {100 * cpu / max(1e-9, args.seconds):.1f}% of one core")
    return 0


if __name__ == "__main__":
    sys.exit(main())
