"""Profile configuration. Laptop and Pi builds differ only here."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

CONFIG_DIR = Path(__file__).parent
PROFILES = ("laptop", "pi5")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AudioConfig(_Strict):
    sample_rate: Literal[16000] = 16000
    frame_samples: int = 512  # 32 ms at 16 kHz; Silero VAD's native window
    ring_buffer_seconds: float = 30.0
    device: str | int | None = None  # None = system default input
    preroll_ms: int = 300  # audio before VAD onset handed to STT/wake word
    level_meter_hz: float = 15.0


class VadConfig(_Strict):
    model: str = "silero_vad.onnx"
    threshold: float = 0.5
    min_speech_ms: int = 96
    min_silence_ms: int = 600


class WakeConfig(_Strict):
    enabled: bool = True
    phrase: str = "Hey Jarvis"  # stock openWakeWord model until "Hey Kin" is trained
    model: str = "openwakeword/hey_jarvis_v0.1.onnx"
    melspec_model: str = "openwakeword/melspectrogram.onnx"
    embedding_model: str = "openwakeword/embedding_model.onnx"
    threshold: float = Field(0.5, ge=0.0, le=1.0)  # wake when a score reaches this; see scripts/wake_probe.py
    cooldown_s: float = 2.0


class SttModelConfig(_Strict):
    arch: Literal["tiny-streaming", "base-streaming", "small-streaming", "medium-streaming"]
    language: str = "en"


class SttConfig(_Strict):
    models_dir: str = "moonshine"
    passive: SttModelConfig = SttModelConfig(arch="tiny-streaming")
    query: SttModelConfig = SttModelConfig(arch="small-streaming")
    update_interval_s: float = 0.3
    lazy_load_query: bool = False
    query_timeout_s: float = 6.0  # attending with no speech -> back to listening


class RuntimeConfig(_Strict):
    threads: int | None = None  # None = let runtimes decide
    ram_target_mb: int | None = None


class LlmConfig(_Strict):
    backend: Literal["litert", "llamacpp"] = "litert"
    model: str = "llm/gemma-4-E2B-it.litertlm"
    fallback_backend: Literal["llamacpp", "none"] = "llamacpp"
    fallback_model: str = "llm/Qwen3.5-2B-Q4_K_M.gguf"
    context_tokens: int = 4096  # hard cap on the KV cache
    max_output_tokens: int = 160
    timeout_s: float = 20.0
    warm_session: bool = True  # prefill system prompt + tools into a spare session while idle
    # A second, larger model only for writing recall answers (tool choice stays on `model`). Laptop only.
    # Prompt 5 oracle: Gemma 4 E4B answers 100% of answerable questions from the gold facts (E2B 93.8%), but
    # its first sentence takes 7.5 s instead of 2.7 s and it adds ~420 MB, so it is off by default.
    answer_model: str | None = None  # e.g. llm/gemma-4-E4B-it.litertlm


class RouterConfig(_Strict):
    """Cheap pre-filter for overheard speech: EmbeddingGemma k-NN over labelled examples.

    It no longer decides what is stored: it only marks episodes in which every line is confidently
    chatter, so the LLM extractor can skip them."""

    examples: str = "router_examples.yaml"  # in kin/config/, or an absolute path
    k: int = 5  # nearest examples per label
    accept_score: float = 0.70  # best label's mean similarity must reach this ...
    accept_margin: float = 0.08  # ... and beat the runner-up by this much; else the line counts as maybe worth keeping

    def examples_path(self) -> Path:
        p = Path(self.examples)
        return p if p.is_absolute() else CONFIG_DIR / p


class ExtractConfig(_Strict):
    """Background fact extraction: overheard lines -> episodes -> self-contained facts (Gemma 4 E2B)."""

    enabled: bool = True
    episode_gap_s: float = 90.0  # a new episode starts after this much silence ...
    episode_max_s: float = 600.0  # ... or when the current one is this long
    idle_s: float = 0.0  # only extract after this long without a question (Pi: 60 s)
    max_output_tokens: int = 400
    dedup_similarity: float = 0.75  # an active fact this similar (same subject/people) -> ask ADD/UPDATE/DUPLICATE
    dedup_candidates: int = 5
    name_match: float = 80.0  # rapidfuzz ratio for "Pria" -> "Priya"; "Priya" vs "Priyanka" is 77
    retry_failed_after_s: float = 3600.0  # an episode whose extraction failed twice is retried after this


class RetrievalConfig(_Strict):
    """Hybrid recall (memory/retrieval.py): dense + BM25 fused with RRF, filters, time boost, optional reranker."""

    candidates: int = 50  # per list: nearest vectors and BM25 matches gathered for fusion
    rrf_k: float = 60.0
    dense_weight: float = 1.0
    lexical_weight: float = 1.0
    alt_weight: float = 0.5  # the model's rewording of the question, relative to the question itself
    # Weights below were chosen with scripts/eval_retrieval.py on the recall benchmark (clean / noisy, 256-d, reranker on):
    # time boosts change nothing once the reranker orders the top 10; without it a x1.1 recency boost cost
    # top-1 0.68 -> 0.56, so recency only breaks ties. A person tier (facts naming the person first) cost
    # top-1 0.68 -> 0.62: extracted facts often say "Someone said ..." when the speaker is unknown.
    future_boost: float = 0.1  # "when is ...?": upcoming events x(1 + this)
    past_event_penalty: float = 0.05  # "when is ...?": events already past x(1 - this)
    recency_boost: float = 0.01  # other questions: x(1 + this * 0.5^(age / half-life)), a tie-breaker
    recency_half_life_days: float = 7.0
    person_mode: Literal["tier", "boost"] = "boost"  # tier: facts naming the person rank first; boost: x(1 + person_boost)
    person_boost: float = 0.0  # names already count through BM25 and the vectors
    rerank: bool = False  # cross-encoder over the fused top `rerank_top` (download: kin download-models --reranker)
    rerank_top: int = 10  # recall@5 0.771 / 0.820 at 10; 0.755 / 0.788 at 7; 0.750 / 0.808 at 15
    rerank_model: str = "reranker"
    confidence: Literal["cos", "rerank", "blend"] = "blend"  # what the abstention threshold applies to
    coverage_weight: float = 0.3  # blend: cosine + this * share of the question's content words found in the fact
    answer_floor: float = 0.375  # best confidence under this -> "I don't remember" without asking the LLM
    strong_score: float = 0.75  # best confidence at or over this -> answer in one LLM call (no tool-choice call)
    utterance_floor: float = 0.8  # raw-transcript fallback: a line this close is offered as "heard on <time>"
    conflict_similarity: float = 0.8  # two active facts this close that disagree (time, numbers) -> say both


class MaintenanceConfig(_Strict):
    """Nightly upkeep: retention purges, FTS optimize, VACUUM (secure_delete is on), integrity check."""

    enabled: bool = True
    hour: int = Field(3, ge=0, le=23)  # run once a night after this local hour, when no question is running ...
    when_charging: bool = True  # ... or earlier, once the device is on mains power (a battery reports it)
    min_interval_h: float = 20.0  # never more often than this


class MemoryConfig(_Strict):
    db_path: str = "data/kin.db"
    key_file: str = "data/kin.key"  # used when KIN_DB_KEY is not set
    embedding_model: str = "embedding"
    embedding_variant: Literal["q4", "q8"] = "q4"
    dim: int = 256  # EmbeddingGemma is Matryoshka-trained; 768 -> 256
    top_k: int = 5
    answer_facts: int = 5  # facts offered to the answer prompt. 3 (Prompt 5 B2) cut off supporting facts ranked 4th-5th: dev 66.7% -> 63.6%
    list_facts: int = 6  # "what do I know about Rahul": matching facts listed for the summary (pinned, then newest)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    prompt_token_budget: int = 1000
    min_words: int = 3  # the router's pre-filter treats shorter lines as chatter (they are still kept as utterances)
    utterance_retention_days: float = 14.0  # raw transcript lines are purged after this (nightly); facts stay
    fact_retention_days: float | None = None  # facts heard longer ago are deleted (pinned ones stay); None: keep forever
    only_about_me: bool = False  # default for the UI setting: skip extracted facts whose subject is another person
    maintenance: MaintenanceConfig = Field(default_factory=lambda: MaintenanceConfig())


class TtsConfig(_Strict):
    enabled: bool = True
    voice: str = "tts/en_US-lessac-medium.onnx"
    output_device: str | int | None = None
    sink: Literal["speaker", "null"] = "speaker"  # null: synthesise but do not play (tests, headless)
    ack_text: str = "Let me check."
    ack_delay_s: float = 0.8  # say the acknowledgement if no answer is ready by then


class GatewayConfig(_Strict):
    enabled: bool = True
    autostart: bool = True  # `kin run` spawns the gateway process (on the Pi: its own systemd unit)
    providers: Literal["real", "fake"] = "real"
    key_file: str = "data/gateway.key"
    audit_log: str = "data/gateway-audit.jsonl"
    upstream_timeout_s: float = 6.0
    client_timeout_s: float = 10.0


class UserConfig(_Strict):
    home_city: str = "London"  # default place for weather questions; set yours in the profile


class TasksConfig(_Strict):
    reminder_check_s: float = 1.0
    missed_grace_h: float = 12.0  # reminders missed while Kin was off are still spoken within this window


class UiConfig(_Strict):
    host: str = "127.0.0.1"
    port: int = 8000


class KinConfig(_Strict):
    profile: str
    hardware: Literal["simulated", "pi_gpio"] = "simulated"
    models_dir: Path = Path("models")
    data_dir: Path = Path(".")
    demo: bool = False  # `kin run --demo`: data_dir is demo/, seeded with a made-up week, resettable from the UI
    audio: AudioConfig = Field(default_factory=AudioConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    wake: WakeConfig = Field(default_factory=WakeConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    extract: ExtractConfig = Field(default_factory=ExtractConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    user: UserConfig = Field(default_factory=UserConfig)
    tasks: TasksConfig = Field(default_factory=TasksConfig)
    network_guard: bool = True  # block outbound sockets in the brain process
    ui: UiConfig = Field(default_factory=UiConfig)

    def model_path(self, relative: str) -> Path:
        return (self.models_dir / relative).resolve()

    def data_path(self, relative: str) -> Path:
        return (self.data_dir / relative).resolve()


def project_root() -> Path:
    return Path(os.environ.get("KIN_HOME", Path(__file__).resolve().parents[2]))


def load_config(profile: str = "laptop", overrides: dict | None = None) -> KinConfig:
    path = CONFIG_DIR / f"{profile}.yaml"
    if not path.exists():
        raise ValueError(f"Unknown profile {profile!r}. Available: {', '.join(PROFILES)}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for key, value in (overrides or {}).items():
        data[key] = value
    cfg = KinConfig(**data)
    if not cfg.models_dir.is_absolute():
        cfg.models_dir = project_root() / cfg.models_dir
    if not cfg.data_dir.is_absolute():
        cfg.data_dir = project_root() / cfg.data_dir
    return cfg
