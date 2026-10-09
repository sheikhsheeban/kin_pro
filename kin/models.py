"""Model download helper (`kin download-models`).

Run once during setup, never by the running app: at runtime Kin only reads
models from ./models. Downloads go to a .part file and are renamed only when
complete, so an interrupted download is never mistaken for a model.
"""

from __future__ import annotations

import os
import shutil
import sys
import urllib.request
from pathlib import Path

from kin.config import KinConfig, load_config

SILERO_VAD_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
HF = "https://huggingface.co"
LLM_URLS = {
    "gemma-4-E2B-it.litertlm": f"{HF}/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm",
    "Qwen3.5-2B-Q4_K_M.gguf": f"{HF}/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf",
}
PIPER_BASE = f"{HF}/rhasspy/piper-voices/resolve/main/en/en_US"
RERANKER_BASE = f"{HF}/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main"


def _size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else path.stat().st_size


def _fetch(url: str, dest: Path) -> None:
    if dest.exists():
        return
    print(f"downloading {dest.name} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB", end="", file=sys.stderr)
        print(file=sys.stderr)
    os.replace(part, dest)


def download_models(cfg: KinConfig | None = None, *, fallback: bool = False, reranker: bool = False) -> dict[str, Path]:
    cfg = cfg or load_config("laptop")
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    # --- Phase 1: ears -------------------------------------------------------------
    out["vad"] = cfg.model_path(cfg.vad.model)
    _fetch(SILERO_VAD_URL, out["vad"])

    from openwakeword.utils import download_models as oww_download

    wake = cfg.model_path(cfg.wake.model)
    if not wake.exists():
        print(f"downloading openWakeWord {wake.name} ...")
        oww_download(model_names=[wake.name.split("_v")[0]], target_directory=str(wake.parent))
    out["wake"] = wake

    from moonshine_voice import get_embedding_model, get_model_for_language, string_to_model_arch

    from kin.audio.stt import find_moonshine_model

    stt_dir = cfg.model_path(cfg.stt.models_dir)
    for role, m in (("passive_stt", cfg.stt.passive), ("query_stt", cfg.stt.query)):
        try:
            out[role] = find_moonshine_model(stt_dir, m.arch, m.language)
        except FileNotFoundError:
            print(f"downloading Moonshine {m.arch}-{m.language} ...")
            get_model_for_language(m.language, string_to_model_arch(m.arch), cache_root=stt_dir)
            out[role] = find_moonshine_model(stt_dir, m.arch, m.language)

    # --- Phase 2: brain and memory -------------------------------------------------------
    from kin.memory.embed import find_embedding_model

    emb_dir = cfg.model_path(cfg.memory.embedding_model)
    get_embedding_model("embeddinggemma-300m", cfg.memory.embedding_variant, cache_root=emb_dir)
    out["embedder"] = find_embedding_model(emb_dir)

    voice = cfg.model_path(cfg.tts.voice)
    name = voice.stem  # en_US-lessac-medium
    speaker, quality = name.split("-")[1], name.split("-")[2]
    for suffix in (".onnx", ".onnx.json"):
        _fetch(f"{PIPER_BASE}/{speaker}/{quality}/{name}{suffix}", voice.with_name(name + suffix))
    out["tts"] = voice

    llm = cfg.model_path(cfg.llm.model)
    _fetch(LLM_URLS[llm.name], llm)
    out["llm"] = llm

    if fallback:
        gguf = cfg.model_path(cfg.llm.fallback_model)
        _fetch(LLM_URLS[gguf.name], gguf)
        out["llm_fallback"] = gguf

    if reranker:  # optional cross-encoder for recall (memory.retrieval.rerank); int8 for this CPU
        import platform

        from kin.memory.rerank import MODEL_NAME

        rr = cfg.models_dir / "reranker" / MODEL_NAME
        arm = platform.machine().lower() in ("arm64", "aarch64")
        onnx = "model_qint8_arm64.onnx" if arm else "model_quint8_avx2.onnx"
        _fetch(f"{RERANKER_BASE}/onnx/{onnx}", rr / onnx)
        _fetch(f"{RERANKER_BASE}/vocab.txt", rr / "vocab.txt")
        out["reranker"] = rr

    print()
    for role, path in out.items():
        print(f"  {role:13s} {_size(path) / 1e6:8.1f} MB  {path}")
    free = shutil.disk_usage(cfg.models_dir).free / 1e9
    print(f"\n  free disk: {free:.1f} GB")
    return out
