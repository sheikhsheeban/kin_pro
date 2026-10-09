"""Composition root: builds and wires the modules for one profile.

Modules never reach into each other; they talk over the event bus and
through `Hardware`. This is the only place that knows about all of them.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import Any

from kin.audio import build_engines
from kin.audio.capture import MicCapture, StreamFactory, sounddevice_stream_factory
from kin.audio.pipeline import Engines, PerceptionPipeline
from kin.audio.ring_buffer import RingBuffer
from kin.brain import build_brain
from kin.config import KinConfig, load_config
from kin.core.bus import EventBus
from kin.core.events import (
    BrainState,
    BrainStateChanged,
    ButtonChanged,
    GatewayStatus,
    LookupPerformed,
    MuteChanged,
    OfflineChanged,
    PipelineState,
    PipelineStateChanged,
    ProfileChanged,
    ReplyCompleted,
    StreamStateChanged,
    TranscriptEvent,
)
from kin.core.metrics import ResourceMonitor
from kin.core.netguard import guard_installed, install_network_guard
from kin.core.privacy import install_audio_write_guard
from kin.tools.gateway_process import GatewayProcess
from kin.core.state import derive_led
from kin.hal import Hardware, create_hardware

log = logging.getLogger(__name__)

EnginesFactory = Callable[[KinConfig], Engines]
BrainFactory = Callable[[KinConfig, EventBus, ResourceMonitor], Awaitable[Any]]


class KinApp:
    def __init__(
        self,
        cfg: KinConfig,
        *,
        bus: EventBus | None = None,
        hardware: Hardware | None = None,
        engines_factory: EnginesFactory = build_engines,
        stream_factory: StreamFactory = sounddevice_stream_factory,
        device_namer: Callable[[Any], str] | None = None,
        brain_factory: BrainFactory | None = build_brain,
        gateway_factory: Callable[[KinConfig], GatewayProcess] | None = GatewayProcess,
    ) -> None:
        self.cfg = cfg
        self.bus = bus or EventBus()
        self.hardware = hardware or create_hardware(cfg.hardware, self.bus)
        self._engines_factory = engines_factory
        self._stream_factory = stream_factory
        self._device_namer = device_namer
        self.pipeline: PerceptionPipeline | None = None
        self.capture: MicCapture | None = None
        self.ring: RingBuffer | None = None
        self._brain_factory = brain_factory
        self.brain = None  # kin.brain.orchestrator.Brain once loaded
        self.brain_state = BrainState.IDLE
        self.brain_status = "disabled" if brain_factory is None else "loading"
        self._brain_task: asyncio.Task | None = None
        self.monitor = ResourceMonitor(cfg.runtime.ram_target_mb)
        self.last_reply: ReplyCompleted | None = None
        self._gateway_factory = gateway_factory if cfg.gateway.enabled else None
        self.gateway: GatewayProcess | None = None
        self.gateway_info: dict[str, Any] = {"running": False, "detail": "disabled" if self._gateway_factory is None else "starting"}
        self._gateway_task: asyncio.Task | None = None
        self.lookups: deque[LookupPerformed] = deque(maxlen=200)
        self.state = PipelineState.IDLE
        self.loading = False
        self.last_error: str | None = None
        # Recent transcript text for the UI (RAM only, text only).
        self.transcripts: deque[TranscriptEvent] = deque(maxlen=200)
        self._switch_lock = asyncio.Lock()

    # --- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        install_audio_write_guard()
        if self.cfg.network_guard:
            install_network_guard()  # from here on this process cannot open an outbound socket
        self.bus.subscribe(OfflineChanged, self._on_offline, name="app.offline")
        self.bus.subscribe(LookupPerformed, self.lookups.append, name="app.lookups")
        self.bus.subscribe(MuteChanged, self._on_mute, name="app.mute")
        self.bus.subscribe(ButtonChanged, self._on_button, name="app.button")
        self.bus.subscribe(PipelineStateChanged, self._on_state, name="app.state")
        self.bus.subscribe(TranscriptEvent, self._on_transcript, name="app.transcript")
        self.bus.subscribe(BrainStateChanged, self._on_brain_state, name="app.brain_state")
        self.bus.subscribe(ReplyCompleted, self._on_reply, name="app.reply")
        await self.bus.start()
        await self.hardware.start()
        await self._start_perception()
        self._start_gateway()
        self._start_brain()

    async def stop(self) -> None:
        await self._stop_brain()
        await self._stop_perception()
        await self._stop_gateway()
        await self.hardware.stop()
        await self.bus.stop()

    # --- gateway (separate process) -------------------------------------------

    def _start_gateway(self) -> None:
        if self._gateway_factory is None:
            return

        async def launch() -> None:
            self.gateway = self._gateway_factory(self.cfg)
            try:
                info = await asyncio.to_thread(self.gateway.start)
                if self.hardware.offline:
                    info = await asyncio.to_thread(self.gateway.client.set_offline, True) | info
                self.gateway_info = {"running": True, "detail": None, **info, "offline": self.hardware.offline}
            except Exception as exc:
                log.error("gateway failed to start: %s", exc)
                self.gateway_info = {"running": False, "detail": str(exc)}
            self.bus.publish(GatewayStatus(running=self.gateway_info["running"], offline=self.hardware.offline,
                                           detail=self.gateway_info.get("detail")))

        self._gateway_task = asyncio.create_task(launch(), name="app.gateway")

    async def _stop_gateway(self) -> None:
        if self._gateway_task is not None:
            await asyncio.gather(self._gateway_task, return_exceptions=True)
        if self.gateway is not None:
            await asyncio.to_thread(self.gateway.stop)
            self.gateway = None

    async def wait_gateway(self) -> None:
        if self._gateway_task is not None:
            await self._gateway_task

    async def _on_offline(self, e: OfflineChanged) -> None:
        """The simulated Wi-Fi switch: the gateway refuses every lookup while offline."""
        self.gateway_info["offline"] = e.offline
        if self.gateway is not None and self.gateway_info.get("running"):
            try:
                await asyncio.to_thread(self.gateway.client.set_offline, e.offline)
            except Exception as exc:
                log.warning("could not tell the gateway about offline mode: %s", exc)
        self.bus.publish(GatewayStatus(running=bool(self.gateway_info.get("running")), offline=e.offline))

    async def network_log(self, limit: int = 100) -> list[dict[str, Any]]:
        """Gateway audit entries (the gateway's own record), or what this session saw if it is down."""
        if self.gateway is not None and self.gateway_info.get("running"):
            try:
                return await asyncio.to_thread(self.gateway.client.audit, limit)
            except Exception:
                pass
        return [
            {"ts": e.ts, "id": e.request_id, "kind": e.kind, "request": e.request, "upstream": e.upstream.splitlines(),
             "status": e.status, "reason": e.reason, "provider": e.provider, "response_bytes": e.response_bytes,
             "reply_bytes": e.reply_bytes, "duration_ms": e.duration_ms, "scrubbed": e.scrubbed}
            for e in list(self.lookups)[-limit:]
        ]

    async def wait_brain(self) -> None:
        if self._brain_task is not None:
            await self._brain_task

    def _start_brain(self) -> None:
        """Load the brain in the background so the UI and listening work while models load."""
        if self._brain_factory is None:
            return
        self.brain_status = "loading"

        async def load() -> None:
            try:
                self.brain = await self._brain_factory(self.cfg, self.bus, self.monitor)
                self.brain_status = "ready"
            except Exception as exc:
                log.exception("brain failed to load")
                self.brain_status = f"error: {exc}"
            self.bus.publish(ProfileChanged(profile=self.cfg.profile))

        self._brain_task = asyncio.create_task(load(), name="app.brain_load")

    async def _stop_brain(self) -> None:
        if self._brain_task is not None:
            await asyncio.gather(self._brain_task, return_exceptions=True)
            self._brain_task = None
        if self.brain is not None:
            brain, self.brain = self.brain, None
            await brain.stop()
        for name in ("embedder + memory", "tts", "llm"):
            self.monitor.forget(name)
        self.brain_state = BrainState.IDLE

    async def switch_profile(self, profile: str) -> None:
        """Hot-swap to another profile (the UI's "Pi 5 mode" toggle)."""
        async with self._switch_lock:
            if profile == self.cfg.profile:
                return
            new_cfg = load_config(profile, overrides={"ui": self.cfg.ui.model_dump()})
            self.bus.publish(ProfileChanged(profile=profile, loading=True))
            await self._stop_brain()
            await self._stop_perception()
            old_cfg, self.cfg = self.cfg, new_cfg
            self.monitor.ram_target_mb = new_cfg.runtime.ram_target_mb
            try:
                await self._start_perception()
            except Exception as exc:
                log.exception("profile switch failed; reverting")
                self.cfg = old_cfg
                self.monitor.ram_target_mb = old_cfg.runtime.ram_target_mb
                await self._start_perception()
                self._start_brain()
                self.bus.publish(ProfileChanged(profile=old_cfg.profile, error=str(exc)))
                return
            self._start_brain()
            self.bus.publish(ProfileChanged(profile=profile))

    async def _start_perception(self) -> None:
        self.loading = True
        try:
            with self.monitor.measure("speech (VAD, wake word, STT)"):
                engines = await asyncio.to_thread(self._engines_factory, self.cfg)
        finally:
            self.loading = False
        self.ring = RingBuffer(self.cfg.audio.ring_buffer_seconds, self.cfg.audio.sample_rate)
        self.pipeline = PerceptionPipeline(self.cfg, engines, self.ring, self.bus.publish_threadsafe)
        self.pipeline.start()
        kwargs = {"device_namer": self._device_namer} if self._device_namer else {}
        self.capture = MicCapture(self.cfg.audio, self.pipeline.submit_frame, self._stream_factory, **kwargs)
        if not self.hardware.muted:
            self._open_stream()
        self._update_led()

    async def _stop_perception(self) -> None:
        if self.capture is not None:
            self._close_stream()
        if self.pipeline is not None:
            pipeline, self.pipeline = self.pipeline, None
            await asyncio.to_thread(pipeline.stop)
        if self.ring is not None:
            self.ring.clear()
        self.state = PipelineState.IDLE
        self._update_led()

    # --- stream control ------------------------------------------------------

    @property
    def stream_open(self) -> bool:
        return self.capture is not None and self.capture.is_open

    def _open_stream(self) -> None:
        assert self.capture is not None and self.pipeline is not None
        try:
            self.capture.open()
        except Exception as exc:
            self.last_error = f"Could not open microphone: {exc}"
            log.error(self.last_error)
            self.bus.publish(StreamStateChanged(open=False, error=self.last_error))
            return
        self.last_error = None
        self.pipeline.stream_opened()
        self.bus.publish(
            StreamStateChanged(open=True, device=self.capture.device_name, sample_rate=self.cfg.audio.sample_rate)
        )
        self._update_led()

    def _close_stream(self) -> None:
        assert self.capture is not None
        self.capture.close()
        if self.pipeline is not None:
            self.pipeline.stream_closed()
        self.bus.publish(StreamStateChanged(open=False))
        self._update_led()

    @property
    def effective_state(self) -> PipelineState:
        """Perception state, unless the brain is busy and the user is not mid-query."""
        if self.state is PipelineState.ATTENDING or self.brain_state is BrainState.IDLE:
            return self.state
        return {
            BrainState.SPEAKING: PipelineState.SPEAKING,
            BrainState.LOOKUP: PipelineState.LOOKUP,
        }.get(self.brain_state, PipelineState.THINKING)

    def _update_led(self) -> None:
        self.hardware.set_led(derive_led(self.stream_open, self.effective_state))

    # --- bus handlers -------------------------------------------------------

    def _on_mute(self, e: MuteChanged) -> None:
        if self.capture is None:
            return
        if e.muted:
            self._close_stream()
        elif not self.stream_open:
            self._open_stream()

    def _on_button(self, e: ButtonChanged) -> None:
        if self.pipeline is not None:
            self.pipeline.button(e.pressed)

    def _on_state(self, e: PipelineStateChanged) -> None:
        self.state = e.state
        self._update_led()

    def _on_transcript(self, e: TranscriptEvent) -> None:
        if e.final:
            self.transcripts.append(e)

    def _on_brain_state(self, e: BrainStateChanged) -> None:
        self.brain_state = e.state
        if self.pipeline is not None:
            self.pipeline.duck(e.state is BrainState.SPEAKING)
        self._update_led()

    def _on_reply(self, e: ReplyCompleted) -> None:
        self.last_reply = e

    # --- UI snapshot ----------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        cfg = self.cfg
        engines = self.pipeline.engines if self.pipeline else None
        return {
            "type": "Snapshot",
            "profile": cfg.profile,
            "hardware": self.hardware.name,
            "muted": self.hardware.muted,
            "offline": self.hardware.offline,
            "led": self.hardware.led.value,
            "stream_open": self.stream_open,
            "device": self.capture.device_name if self.capture else None,
            "state": self.effective_state.value,
            "loading": self.loading,
            "brain": self.brain_status,
            "gateway": {**self.gateway_info, "offline": self.hardware.offline},
            "network_guard": guard_installed(),
            "home_city": cfg.user.home_city,
            "error": self.last_error,
            "wake_phrase": cfg.wake.phrase if cfg.wake.enabled else None,
            "wake_threshold": cfg.wake.threshold if cfg.wake.enabled else None,
            "models": {
                "vad": "Silero VAD v5 (ONNX)",
                "wake": f"openWakeWord · {cfg.wake.phrase}" if cfg.wake.enabled else None,
                "passive_stt": f"Moonshine v2 {cfg.stt.passive.arch}",
                "query_stt": f"Moonshine v2 {cfg.stt.query.arch}",
                "query_stt_loaded": bool(engines and engines.query_loaded),
                "llm": self.brain.llm_name if self.brain else None,
                "embedder": f"EmbeddingGemma 300M {cfg.memory.embedding_variant} · {cfg.memory.dim}-d",
                "tts": Path(cfg.tts.voice).stem if cfg.tts.enabled else None,
                "router": f"semantic · EmbeddingGemma k={cfg.router.k} · accept ≥{cfg.router.accept_score:.2f}, margin ≥{cfg.router.accept_margin:.2f}",
            },
            "limits": {
                "threads": cfg.runtime.threads,
                "llm_context_tokens": cfg.llm.context_tokens,
                "prompt_token_budget": cfg.memory.prompt_token_budget,
                "answer_floor": cfg.memory.retrieval.answer_floor,
                "strong_score": cfg.memory.retrieval.strong_score,
                "reranker": cfg.memory.retrieval.rerank,
                "llm_timeout_s": cfg.llm.timeout_s,
                "ram_target_mb": cfg.runtime.ram_target_mb,
                "ring_buffer_seconds": cfg.audio.ring_buffer_seconds,
            },
            "transcripts": [t.to_dict() for t in self.transcripts],
            "last_reply": self.last_reply.to_dict() if self.last_reply else None,
        }

    def system_stats(self) -> dict[str, Any]:
        stats = self.monitor.snapshot()
        r = self.last_reply
        stats["last_query"] = (
            {
                "time_to_first_audio_ms": r.time_to_first_audio_ms,
                "llm_ttft_ms": r.llm_ttft_ms,
                "tokens_per_s": r.tokens_per_s,
                "prompt_tokens": r.prompt_tokens,
                "total_ms": r.total_ms,
                "backend": r.backend,
                "path": r.path,
                "llm_calls": r.llm_calls,
            }
            if r
            else None
        )
        stats["brain"] = self.brain_status
        b = self.brain
        stats["router"] = (
            {
                "mode": "semantic",
                "accept_score": self.cfg.router.accept_score,
                "accept_margin": self.cfg.router.accept_margin,
                "pending": b.pending_count,  # episodes waiting for fact extraction
                "llm_queue": b.queue.queued,
                "extraction": [
                    {"episode": e.episode_id, "lines": e.lines, "outcome": e.outcome, "added": e.added,
                     "updated": e.updated, "duplicates": e.duplicates, "ms": e.duration_ms, "error": e.error}
                    for e in reversed(b.extraction_log)
                ],
                "recent": [
                    {"text": t.text, "label": t.label, "score": t.score, "margin": t.margin, "source": t.source,
                     "decision": t.decision}
                    for t in reversed(b.recent_tags)
                ],
            }
            if b is not None and hasattr(b, "recent_tags")
            else None
        )
        return stats
