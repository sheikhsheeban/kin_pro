"""Reasoning: orchestrator, safety guard, tool calling, semantic router, LLM backends, confidence gate."""

from __future__ import annotations

import asyncio
import logging

from kin.config import KinConfig
from kin.core.bus import EventBus
from kin.core.metrics import ResourceMonitor

log = logging.getLogger(__name__)


async def build_brain(cfg: KinConfig, bus: EventBus, monitor: ResourceMonitor):
    """Load the real models one at a time (so per-component RAM can be measured) and wire the Brain."""
    from kin.brain.llm import create_llm
    from kin.brain.orchestrator import Brain
    from kin.brain.semantic_router import SemanticRouter, load_examples
    from kin.io.tts import NullSink, PiperTTS, SoundDeviceSink, Speaker
    from kin.memory.embed import GemmaEmbedder, find_embedding_model
    from kin.memory.service import MemoryService
    from kin.memory.store import MemoryStore, load_or_create_key

    m = cfg.memory
    key = load_or_create_key(cfg.data_path(m.key_file))
    from kin.memory.rerank import CrossEncoder, find_reranker

    reranker = find_reranker(cfg.model_path(m.retrieval.rerank_model)) if m.retrieval.rerank else None
    if m.retrieval.rerank and reranker is None:
        log.warning("memory.retrieval.rerank is on but the model is missing: run `kin download-models --reranker`")
    memory = MemoryService(
        bus,
        embedder_factory=lambda: GemmaEmbedder(
            find_embedding_model(cfg.model_path(m.embedding_model)), m.embedding_variant, m.dim
        ),
        # a new `memory.dim` re-embeds every fact and line once at start instead of refusing to open
        store_factory=lambda dim: MemoryStore(cfg.data_path(m.db_path), key, dim, reembed=True),
        name_match=cfg.extract.name_match,
        retrieval=m.retrieval,
        only_about_me=m.only_about_me,
        reranker_factory=(lambda: CrossEncoder(reranker, threads=cfg.runtime.threads)) if reranker else None,
    )
    with monitor.measure("embedder + memory"):
        await memory.start()

    tts = None
    if cfg.tts.enabled:
        with monitor.measure("tts"):
            tts = await asyncio.to_thread(PiperTTS, cfg.model_path(cfg.tts.voice))
    use_speaker = tts is not None and cfg.tts.sink == "speaker"
    speaker = Speaker(tts, SoundDeviceSink(cfg.tts.output_device) if use_speaker else NullSink())

    r = cfg.router
    router = SemanticRouter(
        load_examples(r.examples_path()), k=r.k, accept_score=r.accept_score, accept_margin=r.accept_margin,
        min_words=m.min_words,
    )
    with monitor.measure("semantic router"):  # example vectors: embedded once, then cached on disk
        await router.fit(memory.embed_documents, memory.embedder_name, cfg.data_path("data/cache"))

    from kin.tools.gateway_process import make_client
    from kin.tools.reminders import ReminderScheduler
    from kin.tools.tasks import TaskService

    tasks = TaskService(bus, memory)
    brain = Brain(
        cfg, bus,
        llm_factory=lambda: create_llm(cfg),
        memory=memory,
        speaker=speaker,
        router=router,
        lookup_client=make_client(cfg) if cfg.gateway.enabled else None,
        tasks=tasks,
    )
    brain.reminders = ReminderScheduler(
        bus, tasks, brain.announce, check_s=cfg.tasks.reminder_check_s, missed_grace_h=cfg.tasks.missed_grace_h
    )
    with monitor.measure("llm"):
        await brain.load_llm()
    await brain.start()
    return brain
