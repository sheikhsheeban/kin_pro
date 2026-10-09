"""Scripted end-to-end demo: the umbrella scenario from the architecture doc (section 5).

  1. Morning: Kin overhears "I'm meeting Priya at 4 pm at the library cafe for coffee." and remembers it.
  2. "Hey Kin, do I need an umbrella for the coffee with Priya I mentioned this morning?"
     -> meeting time/place from local memory, forecast from the weather API via the gateway
        process, decision made locally by Gemma. LED goes blue during the lookup.
  3. Network Monitor: the payload that left the device contains only a city and a date.
  4. Wi-Fi off (offline toggle): the lookup is refused honestly; recall still works.
  5. "Remind me to email the professor in 10 seconds" -> confirmation, then the reminder is spoken.

Usage:
  uv run python scripts/demo.py                 # real weather (needs internet), text-driven, silent
  uv run python scripts/demo.py --weather fake  # canned forecast, works anywhere
  uv run python scripts/demo.py --voice         # speak each line with Piper into the microphone pipeline
  uv run python scripts/demo.py --speak         # play Kin's replies through the speakers

Exit code 0 when every acceptance check passed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kin.app import KinApp  # noqa: E402
from kin.audio import build_engines  # noqa: E402
from kin.brain import build_brain  # noqa: E402
from kin.config import load_config  # noqa: E402
from kin.core.bus import EventBus  # noqa: E402
from kin.core.events import (  # noqa: E402
    ActivationEvent,
    Announcement,
    AttendingEnded,
    LedChanged,
    LookupPerformed,
    MemoryStored,
    ReplyCompleted,
    TranscriptEvent,
)

MORNING = "I'm meeting Priya at 4 pm at the library cafe for coffee."
UMBRELLA = "Do I need an umbrella for the coffee with Priya I mentioned this morning?"
RECALL = "When am I meeting Priya?"
REMIND = "Remind me to email the professor in 10 seconds"

BOLD, DIM, BLUE, GREEN, RED, END = "\033[1m", "\033[2m", "\033[94m", "\033[92m", "\033[91m", "\033[0m"


class ScriptedMic:
    """A microphone stream we can push audio into (the demo's 'room')."""

    def __init__(self, *, samplerate, blocksize, device, callback) -> None:
        self.callback, self.blocksize = callback, blocksize
        self.active = self.closed = False

    def start(self) -> None:
        self.active = True

    def abort(self) -> None:
        self.active = False

    def close(self) -> None:
        self.active, self.closed = False, True

    def play(self, audio: np.ndarray) -> None:
        n = self.blocksize
        t0 = time.monotonic()
        for i in range(0, len(audio) - n + 1, n):
            if self.active:
                self.callback(audio[i : i + n].reshape(-1, 1), n, None, None)
            delay = t0 + (i + n) / 16000 - time.monotonic()
            if delay > 0:
                time.sleep(delay)


class Demo:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.checks: list[tuple[str, bool, str]] = []
        self.mics: list[ScriptedMic] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((name, ok, detail))
        print(f"   {GREEN + '✔' if ok else RED + '✘'}{END} {name}{DIM + ' · ' + detail + END if detail else ''}")

    def step(self, n: int, title: str) -> None:
        print(f"\n{BOLD}{n}. {title}{END}")

    async def run(self) -> int:
        os.environ.setdefault("KIN_DB_KEY", "ab" * 32)
        cfg = load_config(self.args.profile)
        cfg.data_dir = Path(tempfile.mkdtemp(prefix="kin-demo-"))  # a fresh memory for every rehearsal
        cfg.tts.sink = "speaker" if self.args.speak else "null"
        cfg.gateway.providers = self.args.weather
        bus = EventBus()
        self.events: list = []
        for cls in (LookupPerformed, LedChanged, MemoryStored, Announcement, TranscriptEvent):
            bus.subscribe(cls, self.events.append)

        def mic_factory(**kw):
            self.mics.append(ScriptedMic(**kw))
            return self.mics[-1]

        print(f"{BOLD}Kin demo{END} · profile {cfg.profile} · weather {self.args.weather} · "
              f"{'voice' if self.args.voice else 'text'} input · data in {cfg.data_dir}")
        print(f"{DIM}loading models and starting the gateway process…{END}")
        app = KinApp(cfg, bus=bus, engines_factory=build_engines, stream_factory=mic_factory,
                     device_namer=lambda _d: "scripted room", brain_factory=build_brain)
        await app.start()
        try:
            await app.wait_gateway()
            await app.wait_brain()
            g = app.gateway_info
            print(f"   gateway pid {app.gateway.proc.pid if app.gateway and app.gateway.proc else '?'} · "
                  f"brain pid {os.getpid()} · allowed hosts: {', '.join(g.get('hosts', [])) or 'none (fake)'}")
            self.check("gateway runs as a separate process", bool(g.get("running")) and app.gateway.proc.pid != os.getpid())
            self.check("brain process has the socket guard", app.snapshot()["network_guard"])
            await self.scenario(app)
        finally:
            await app.stop()
        failed = [c for c in self.checks if not c[1]]
        print(f"\n{BOLD}{len(self.checks) - len(failed)}/{len(self.checks)} checks passed{END}")
        return 1 if failed else 0

    # --- input helpers ---------------------------------------------------------------------

    async def overhear(self, app: KinApp, text: str) -> None:
        print(f"   {DIM}room:{END} “{text}”")
        if self.args.voice:
            await asyncio.to_thread(self.mics[-1].play, self.speech(app, text))
            await asyncio.sleep(1.0)
        else:
            app.bus.publish(TranscriptEvent(line_id=f"demo-{time.monotonic()}", text=text, final=True, mode="passive"))
            await asyncio.sleep(0.1)
        await app.brain.drain()

    async def ask(self, app: KinApp, text: str) -> ReplyCompleted:
        print(f"   {BOLD}you:{END} “Hey Kin, {text[0].lower() + text[1:]}”")
        waiter = asyncio.create_task(app.bus.wait_for(ReplyCompleted, timeout=90))
        await asyncio.sleep(0)
        if self.args.voice:
            app.hardware.simulate_button(True)
            await app.bus.wait_for(ActivationEvent, timeout=5)
            await asyncio.to_thread(self.mics[-1].play, self.speech(app, text))
            app.hardware.simulate_button(False)
        else:
            app.bus.publish(ActivationEvent(source="wake_word", score=0.9))
            app.bus.publish(TranscriptEvent(line_id=f"demo-q-{time.monotonic()}", text=text, final=True, mode="query"))
            app.bus.publish(AttendingEnded(reason="query_complete"))
        reply = await waiter
        cites = f" {DIM}[from memory #{reply.citations}]{END}" if reply.citations else ""
        print(f"   {BOLD}kin:{END} “{reply.text}”{cites}")
        print(f"   {DIM}{reply.outcome} · first audio {reply.time_to_first_audio_ms} ms · total {reply.total_ms} ms{END}")
        return reply

    def speech(self, app: KinApp, text: str) -> np.ndarray:
        from scipy.signal import resample_poly

        tts = app.brain.speaker.tts
        audio = resample_poly(tts.synthesize(text), 16000, tts.sample_rate).astype(np.float32) * 0.8
        pad = np.zeros(8000, np.float32)
        return np.concatenate([pad, audio, pad, pad])  # in RAM only, like everything else

    # --- the scenario ------------------------------------------------------------------------

    async def scenario(self, app: KinApp) -> None:
        self.step(1, "Morning: Kin overhears a plan")
        await self.overhear(app, MORNING)
        stored = [e for e in self.events if isinstance(e, MemoryStored)]
        self.check("the plan is remembered (encrypted, on-device)", bool(stored),
                   f"memory #{stored[0].id} · {stored[0].kind}" if stored else "nothing stored")

        self.step(2, "Afternoon: the umbrella question")
        mark = len(self.events)
        reply = await self.ask(app, UMBRELLA)
        new = self.events[mark:]
        lookups = [e for e in new if isinstance(e, LookupPerformed)]
        leds = [e.color.value for e in new if isinstance(e, LedChanged)]
        self.check("Kin looked the weather up through the gateway", bool(lookups) and lookups[-1].status == "ok",
                   lookups[-1].status if lookups else "no lookup")
        self.check("the spoken answer says it checked online", reply.text.startswith("I checked the weather online."))
        self.check("a decision about the umbrella was made locally", "umbrella" in reply.text.lower())
        self.check("LED turned blue during the lookup", "blue" in leds, " → ".join(leds))

        self.step(3, "Network Monitor: what actually left the device")
        if lookups:
            look = lookups[-1]
            print(f"   {BLUE}{look.request}{END}")
            for line in look.upstream.splitlines():
                print(f"   {DIM}→ {line}{END}")
            params = json.loads(look.request)["params"]
            self.check("payload holds only a city and a date", set(params) == {"city", "date"}, json.dumps(params))
            leaked = [w for w in ("priya", "coffee", "library", "umbrella", "meeting") if w in (look.request + look.upstream).lower()]
            self.check("no transcript or memory text left the device", not leaked, ", ".join(leaked) or "clean")

        self.step(4, "Wi-Fi off")
        app.hardware.simulate_offline(True)
        await asyncio.sleep(0.3)
        offline = await self.ask(app, UMBRELLA)
        self.check("the lookup fails honestly", "offline mode" in offline.text)
        recall = await self.ask(app, RECALL)
        self.check("recall still works offline", recall.outcome == "answered" and "Priya" in recall.text)
        app.hardware.simulate_offline(False)

        self.step(5, "A reminder")
        conf = await self.ask(app, REMIND)
        self.check("reminder confirmed out loud", conf.text.startswith("Okay, I'll remind you to email the professor"))
        print(f"   {DIM}waiting for it…{END}")
        for _ in range(300):
            fired = [e for e in self.events if isinstance(e, Announcement)]
            if fired:
                print(f"   {BOLD}kin:{END} “{fired[0].text}”")
                break
            await asyncio.sleep(0.1)
        self.check("the reminder fired and was spoken", any(isinstance(e, Announcement) for e in self.events))


def main() -> int:
    parser = argparse.ArgumentParser(description="Kin umbrella demo")
    parser.add_argument("--profile", choices=["laptop", "pi5"], default="laptop")
    parser.add_argument("--weather", choices=["real", "fake"], default="real", help="real needs internet (gateway only)")
    parser.add_argument("--voice", action="store_true", help="speak the user's lines into the microphone pipeline")
    parser.add_argument("--speak", action="store_true", help="play Kin's replies through the speakers")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if sys.platform == "win32":
        os.system("")  # enable ANSI colours in the Windows console
    return asyncio.run(Demo(args).run())


if __name__ == "__main__":
    sys.exit(main())
