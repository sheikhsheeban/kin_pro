"""Audio front end: capture -> RAM ring buffer -> VAD -> wake word -> streaming STT."""

from __future__ import annotations

from kin.config import KinConfig


def build_engines(cfg: KinConfig):
    """Load the real models for a profile. Imported lazily so tests can run with fakes."""
    from kin.audio.pipeline import Engines
    from kin.audio.stt import MoonshineSTT, find_moonshine_model
    from kin.audio.vad import SileroVAD
    from kin.audio.wake import OpenWakeWordDetector

    vad = SileroVAD(cfg.model_path(cfg.vad.model), cfg.runtime.threads)
    wake = None
    if cfg.wake.enabled:
        wake = OpenWakeWordDetector(
            cfg.model_path(cfg.wake.model),
            cfg.model_path(cfg.wake.melspec_model),
            cfg.model_path(cfg.wake.embedding_model),
            cfg.runtime.threads,
        )
    stt_dir = cfg.model_path(cfg.stt.models_dir)

    def make_stt(arch: str, language: str) -> MoonshineSTT:
        return MoonshineSTT(find_moonshine_model(stt_dir, arch, language), arch, cfg.stt.update_interval_s)

    passive = make_stt(cfg.stt.passive.arch, cfg.stt.passive.language)
    query = cfg.stt.query
    return Engines(vad, wake, passive, lambda: make_stt(query.arch, query.language), lazy_query=cfg.stt.lazy_load_query)
