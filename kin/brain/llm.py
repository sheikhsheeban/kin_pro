"""Local LLM backends behind one interface.

  LiteRTBackend    Gemma 4 E2B (int4, .litertlm) on LiteRT-LM. The Pi target.
  LlamaCppBackend  any GGUF on llama.cpp. Fallback (Qwen3.5-2B by default).

Both run fully on-device. There is no cloud backend and there must never be one.

A `session` is one conversation: `send(prompt)` streams text and tool calls;
`send(ToolResult(...))` feeds a tool's answer back so the model can finish.
Backends are synchronous and are always driven from a single `ModelWorker` thread.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any, Literal

log = logging.getLogger(__name__)

StopReason = Literal["complete", "cancelled", "max_tokens", "error"]


@dataclass(frozen=True)
class ToolSpec:
    """A function the model may call. `parameters` is a JSON schema object."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})

    def as_function(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """The answer to a ToolCall, as plain text for the model to reason over."""

    name: str
    content: str


@dataclass(frozen=True)
class LlmChunk:
    text: str = ""
    tool_call: ToolCall | None = None


@dataclass
class LlmStats:
    prompt_tokens: int | None = None
    output_tokens: int = 0
    ttft_s: float | None = None
    duration_s: float = 0.0
    stop_reason: StopReason = "complete"

    @property
    def tokens_per_s(self) -> float | None:
        gen = self.duration_s - (self.ttft_s or 0.0)
        return round(self.output_tokens / gen, 1) if self.output_tokens > 1 and gen > 0 else None


class LlmSession(ABC):
    def __init__(self, backend: LLMBackend) -> None:
        self.backend = backend

    @abstractmethod
    def send(self, message: str | ToolResult, *, cancel: Event | None = None) -> Iterator[LlmChunk]: ...

    def close(self) -> None:  # noqa: B027 - optional hook
        pass


class LLMBackend(ABC):
    name: str = "abstract"

    def __init__(self, context_tokens: int) -> None:
        self.context_tokens = context_tokens
        self.last_stats = LlmStats()

    @abstractmethod
    def session(self, system: str, *, tools: Sequence[ToolSpec] = (), max_output_tokens: int) -> LlmSession: ...

    def stream(
        self,
        system: str,
        prompt: str,
        *,
        max_output_tokens: int,
        tools: Sequence[ToolSpec] = (),
        cancel: Event | None = None,
    ) -> Iterator[LlmChunk]:
        """Single-turn convenience: one session, one message."""
        s = self.session(system, tools=tools, max_output_tokens=max_output_tokens)
        try:
            yield from s.send(prompt, cancel=cancel)
        finally:
            s.close()

    def count_tokens(self, text: str) -> int:
        """Exact where the runtime exposes a tokenizer; otherwise a safe over-estimate."""
        return max(1, int(len(text) / 3.2) + 1)

    def generate(self, system: str, prompt: str, *, max_output_tokens: int) -> str:
        return "".join(c.text for c in self.stream(system, prompt, max_output_tokens=max_output_tokens))

    def close(self) -> None:  # noqa: B027 - optional hook
        pass


class _StatsTimer:
    def __init__(self, stats: LlmStats) -> None:
        self.stats = stats
        self.t0 = time.monotonic()

    def first_token(self) -> None:
        if self.stats.ttft_s is None:
            self.stats.ttft_s = time.monotonic() - self.t0

    def done(self) -> None:
        self.stats.duration_s = time.monotonic() - self.t0


# --- LiteRT-LM ------------------------------------------------------------------


class LiteRTBackend(LLMBackend):
    """Gemma 4 E2B via LiteRT-LM. On a 4 GB Pi this is the runtime that fits (see architecture §3)."""

    def __init__(self, model_path: Path, *, context_tokens: int, threads: int | None, cache_dir: Path) -> None:
        super().__init__(context_tokens)
        import litert_lm as lm

        if not Path(model_path).exists():
            raise FileNotFoundError(f"LLM model not found at {model_path}. Run: uv run kin download-models")
        lm.set_min_log_severity(lm.LogSeverity.ERROR)
        cache_dir.mkdir(parents=True, exist_ok=True)  # XNNPACK weight cache: weights only, never user data
        self._lm = lm
        self._engine = lm.Engine(
            str(model_path),
            backend=lm.Backend.CPU(thread_count=threads),
            max_num_tokens=context_tokens,
            cache_dir=str(cache_dir),
        )
        self.name = f"litert:{Path(model_path).stem}"

    def count_tokens(self, text: str) -> int:
        return len(self._engine.tokenize(text))

    def session(self, system, *, tools=(), max_output_tokens):
        return _LiteRTSession(self, system, tools, max_output_tokens)

    def close(self) -> None:
        engine, self._engine = self._engine, None
        if engine is not None and hasattr(engine, "close"):
            engine.close()


class _LiteRTSession(LlmSession):
    def __init__(self, backend: LiteRTBackend, system: str, tools: Sequence[ToolSpec], max_output_tokens: int) -> None:
        super().__init__(backend)
        lm = backend._lm
        self._conv = backend._engine.create_conversation(
            system_message=system,
            tools=[_lite_rt_tool(t) for t in tools] or None,
            automatic_tool_calling=False,
            max_output_tokens=max_output_tokens,
            sampler_config=lm.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0),
        )

    def send(self, message, *, cancel=None):
        b: LiteRTBackend = self.backend  # type: ignore[assignment]
        if isinstance(message, ToolResult):
            message = {"role": "tool", "content": [{"type": "tool_response", "name": message.name, "response": message.content}]}
        b.last_stats = stats = LlmStats()
        timer = _StatsTimer(stats)
        before = self._conv.token_count or 0
        text_out: list[str] = []
        # LiteRT-LM's stream blocks on a queue with no timeout, and the check in the loop below only runs when a
        # chunk arrives, so a cancel during the prefill (no chunk yet) waited for the first token. A watcher
        # cancels the native process as soon as the flag is set; the stream then ends with a CANCELLED chunk
        # (about 1 s after the cancel on the laptop, measured with Gemma 4 E2B mid-prefill).
        finished = threading.Event()

        def watch() -> None:
            while not finished.wait(0.05):
                if cancel.is_set():
                    with contextlib.suppress(Exception):
                        self._conv.cancel_process()
                    return

        if cancel is not None:
            threading.Thread(target=watch, name="kin-llm-cancel", daemon=True).start()
        try:
            for msg in self._conv.send_message_async(message):
                if cancel is not None and cancel.is_set():
                    self._conv.cancel_process()
                    stats.stop_reason = "cancelled"
                    break
                for call in msg.get("tool_calls", []) or []:
                    fn = call.get("function", call)
                    timer.first_token()
                    yield LlmChunk(tool_call=ToolCall(fn.get("name", ""), _as_dict(fn.get("arguments"))))
                content = msg.get("content", [])
                for part in content if isinstance(content, list) else [content]:
                    if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                        timer.first_token()
                        text_out.append(part["text"])
                        yield LlmChunk(text=part["text"])
            if cancel is not None and cancel.is_set():
                stats.stop_reason = "cancelled"  # the watcher ended the stream
        finally:
            finished.set()
            timer.done()
            # Stream chunks are not tokens; count properly for tokens/s.
            stats.output_tokens = b.count_tokens("".join(text_out)) if text_out else 0
            total = self._conv.token_count
            stats.prompt_tokens = max(0, total - before - stats.output_tokens) if total else None

    def close(self) -> None:
        self._conv.close()


def _lite_rt_tool(spec: ToolSpec):
    """Adapter from ToolSpec to LiteRT-LM's Tool ABC (execution stays with the orchestrator)."""
    import litert_lm as lm

    class _Tool(lm.Tool):
        def get_tool_description(self) -> dict[str, Any]:
            return {"type": "function", "function": spec.as_function()}

        def execute(self, param):  # never called: automatic_tool_calling=False
            raise RuntimeError("tool calls are executed by the orchestrator")

    return _Tool()


# --- llama.cpp ------------------------------------------------------------------

_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


class LlamaCppBackend(LLMBackend):
    """Any chat GGUF via llama-cpp-python (install with: uv sync --extra fallback).

    Tools use the Qwen convention (<tool_call>{json}</tool_call> in the text), which the
    model's own chat template understands; calls are parsed out of the stream.
    """

    def __init__(self, model_path: Path, *, context_tokens: int, threads: int | None) -> None:
        super().__init__(context_tokens)
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError("llama-cpp-python is not installed. Run: uv sync --extra fallback") from exc
        if not Path(model_path).exists():
            raise FileNotFoundError(f"GGUF model not found at {model_path}")
        self._llm = Llama(str(model_path), n_ctx=context_tokens, n_threads=threads, verbose=False)
        self.name = f"llamacpp:{Path(model_path).stem}"
        # Qwen3.x thinks by default; a spoken assistant needs the answer, not the reasoning.
        self.no_think = "qwen3" in Path(model_path).name.lower()

    def count_tokens(self, text: str) -> int:
        return len(self._llm.tokenize(text.encode("utf-8"), add_bos=False))

    def session(self, system, *, tools=(), max_output_tokens):
        return _LlamaSession(self, system, tools, max_output_tokens)

    def close(self) -> None:
        llm, self._llm = self._llm, None
        if llm is not None and hasattr(llm, "close"):
            llm.close()


class _LlamaSession(LlmSession):
    def __init__(self, backend: LlamaCppBackend, system: str, tools: Sequence[ToolSpec], max_output_tokens: int) -> None:
        super().__init__(backend)
        if tools:
            listing = "\n".join(json.dumps({"type": "function", "function": t.as_function()}) for t in tools)
            system += (
                "\n\n# Tools\nYou may call one of these functions:\n<tools>\n" + listing + "\n</tools>\n"
                'To call one, reply with only: <tool_call>{"name": <name>, "arguments": <object>}</tool_call>'
            )
        self._messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        self._max = max_output_tokens

    def send(self, message, *, cancel=None):
        b: LlamaCppBackend = self.backend  # type: ignore[assignment]
        if isinstance(message, ToolResult):
            self._messages.append({"role": "user", "content": f"<tool_response>\n{message.content}\n</tool_response>"})
        else:
            self._messages.append({"role": "user", "content": f"{message} /no_think" if b.no_think else message})
        b.last_stats = stats = LlmStats()
        timer = _StatsTimer(stats)
        raw: list[str] = []
        pending = ""  # text held back while it might be the start of a <tool_call> or <think>
        in_think = False
        try:
            for part in b._llm.create_chat_completion(self._messages, max_tokens=self._max, temperature=0.0, stream=True):
                if cancel is not None and cancel.is_set():
                    stats.stop_reason = "cancelled"
                    break
                choice = part["choices"][0]
                text = choice.get("delta", {}).get("content") or ""
                if choice.get("finish_reason") == "length":
                    stats.stop_reason = "max_tokens"
                if not text:
                    continue
                raw.append(text)
                stats.output_tokens += 1
                pending += text
                if "<think>" in pending:
                    in_think = True
                if in_think:
                    if "</think>" not in pending:
                        continue
                    in_think, pending = False, pending.split("</think>", 1)[1]
                if _may_be_tool_call(pending):
                    continue  # hold back until we know whether this is a call or plain text
                timer.first_token()
                yield LlmChunk(text=pending)
                pending = ""
            full = "".join(raw)
            calls = _TOOL_CALL.findall(full)
            for c in calls:
                try:
                    obj = json.loads(c)
                except json.JSONDecodeError:
                    continue
                timer.first_token()
                yield LlmChunk(tool_call=ToolCall(obj.get("name", ""), _as_dict(obj.get("arguments"))))
            if not calls and pending.strip() and "<tool_call>" not in pending:
                timer.first_token()
                yield LlmChunk(text=pending)
            self._messages.append({"role": "assistant", "content": full})
            stats.prompt_tokens = sum(b.count_tokens(m["content"]) for m in self._messages[:-1])
        finally:
            timer.done()


def _may_be_tool_call(pending: str) -> bool:
    """True if the held text contains a tool-call tag or could still turn into one."""
    head = pending.lstrip()
    return "<tool_call>" in pending or (bool(head) and "<tool_call>".startswith(head[:11]))


def _as_dict(args: Any) -> dict[str, Any]:
    if isinstance(args, dict):
        return args
    if isinstance(args, str) and args.strip():
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"raw": args}
    return {}


def create_llm(cfg) -> LLMBackend:
    """Build the configured backend, falling back to the GGUF backend if the primary cannot load."""
    threads = cfg.runtime.threads
    ctx = cfg.llm.context_tokens

    def build(kind: str, model: str) -> LLMBackend:
        path = cfg.model_path(model)
        if kind == "litert":
            return LiteRTBackend(path, context_tokens=ctx, threads=threads, cache_dir=cfg.model_path("llm/cache"))
        return LlamaCppBackend(path, context_tokens=ctx, threads=threads)

    try:
        return build(cfg.llm.backend, cfg.llm.model)
    except Exception as exc:
        if cfg.llm.fallback_backend == "none":
            raise
        log.warning("primary LLM %s failed to load (%s); using fallback %s", cfg.llm.model, exc, cfg.llm.fallback_model)
        return build(cfg.llm.fallback_backend, cfg.llm.fallback_model)
