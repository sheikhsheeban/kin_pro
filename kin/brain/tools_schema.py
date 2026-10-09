"""The tools Gemma 4 E2B may call while answering a query, and strict validation of its calls.

One fixed system prompt plus these definitions, byte-identical on every query, so the
prefix can be prefilled ahead of time (see `Brain._warm_session`). The model either answers
directly (chit-chat) or calls one tool; the orchestrator executes the call and feeds the
result back. Nothing here executes anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from kin.brain.llm import ToolCall, ToolSpec
from kin.brain.prompt import load_prompt

LOOKUP_QUERY_MAX = 80
LOOKUP_QUERY_WORDS = 12  # the gateway refuses longer search queries
LOOKUP_KINDS = ("weather", "news", "search")


def _obj(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required}


_STR = {"type": "string"}
_OPT_STR = {"type": ["string", "null"]}

TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "recall",
        "Search the user's own memories: their plans, belongings, people they know, and what someone said or shared. "
        "Use it for every question about the user's life, even when it mentions news, weather or reminding.",
        _obj({"question": _STR | {"description": "The user's question"}}, ["question"]),
    ),
    ToolSpec(
        "remember",
        "Save something the user explicitly asks you to remember.",
        _obj({"text": _STR | {"description": "The fact, as a full sentence"}}, ["text"]),
    ),
    ToolSpec(
        "add_task",
        "Add a new item to the user's to-do list, or set a reminder when a time is given. Not for marking items done.",
        _obj({"text": _STR | {"description": "What to do, e.g. email the professor"},
              "due": _OPT_STR | {"description": "When, in the user's words (at 5, tomorrow at 9, in 10 minutes), or null"}},
             ["text"]),
    ),
    ToolSpec("list_tasks", "Read the user's to-do list and reminders.", _obj({}, [])),
    ToolSpec(
        "complete_task",
        "Mark an existing item on the user's to-do list as done (\"mark … as done\", \"tick off …\").",
        _obj({"text": _STR | {"description": "The task"}}, ["text"]),
    ),
    ToolSpec(
        "cancel_reminder",
        "Cancel one of the user's reminders.",
        _obj({"text": _STR | {"description": "Which reminder"}}, ["text"]),
    ),
    ToolSpec(
        "correct_fact",
        "The user says something Kin just said from memory is wrong and gives the right version "
        "(\"no, it's 5 pm\", \"Rahul isn't vegan any more\", \"actually the code is 4471\").",
        _obj({"correction": _STR | {"description": "The right version, in the user's words"},
              "which": _OPT_STR | {"description": "What the wrong memory was about, if the user said; else null"}},
             ["correction"]),
    ),
    ToolSpec(
        "retract_fact",
        "The user says a memory is wrong but gives no right version (\"that's wrong\", \"that's not true\", "
        "\"I never said that\"). Kin stops using it; nothing is deleted.",
        _obj({"which": _OPT_STR | {"description": "What the wrong memory was about, if the user said; else null"}}, []),
    ),
    ToolSpec(
        "review_memory",
        "The user asks what Kin remembered today or yesterday (\"what did you remember today?\").",
        _obj({"day": {"type": ["string", "null"], "enum": ["today", "yesterday", None]}}, []),
    ),
    ToolSpec(
        "lookup",
        "Look up public information online: a weather forecast (rain, umbrella, temperature, what to wear), news headlines, or a general fact. "
        "The user's relevant memories are added to the result automatically. Never put a person's name in it.",
        _obj({"kind": {"type": "string", "enum": list(LOOKUP_KINDS)},
              "query": _STR | {"description": "A city for weather (empty for the home city), a news topic, or a short search"},
              "when": _OPT_STR | {"description": "Weather only: the day, e.g. today, tomorrow, Friday; else null"}},
             ["kind", "query"]),
    ),
)
TOOL_NAMES = frozenset(t.name for t in TOOLS)
_SPECS = {t.name: t for t in TOOLS}


def system_prompt() -> str:
    return load_prompt("agent_system")


@dataclass(frozen=True)
class ValidCall:
    name: str
    args: dict[str, Any]

    @property
    def label(self) -> str:
        return f"tool: {self.name} ({self.args['kind']})" if self.name == "lookup" else f"tool: {self.name}"


class InvalidCall(ValueError):
    pass


def _type_ok(value: Any, schema: dict[str, Any]) -> bool:
    types = schema.get("type", "string")
    types = types if isinstance(types, list) else [types]
    if value is None:
        return "null" in types
    if isinstance(value, str):
        return "string" in types and ("enum" not in schema or value in schema["enum"])
    return False


def validate(call: ToolCall) -> ValidCall:
    """Strict: a known tool, an object of string (or null) arguments, every required field, nothing extra.

    Raises InvalidCall with a short reason the model can act on.
    """
    spec = _SPECS.get(call.name)
    if spec is None:
        raise InvalidCall(f"unknown tool {call.name!r}")
    args = call.arguments
    if not isinstance(args, dict) or "raw" in args and len(args) == 1:
        raise InvalidCall("arguments must be a JSON object")
    props: dict[str, Any] = spec.parameters["properties"]
    extra = set(args) - set(props)
    if extra:
        raise InvalidCall(f"unexpected argument(s) {sorted(extra)} for {call.name}")
    out: dict[str, Any] = {}
    for name, schema in props.items():
        if name not in args:
            if name in spec.parameters["required"] and "null" not in str(schema.get("type")):
                raise InvalidCall(f"{call.name} needs {name!r}")
            out[name] = None
            continue
        value = args[name]
        if isinstance(value, str):
            value = value.strip()
        if not _type_ok(value, schema):
            allowed = schema.get("enum")
            raise InvalidCall(f"{name!r} must be {' or '.join(allowed) if allowed else 'a string'}")
        out[name] = value
    for name in ("question", "text"):
        if name in out and not out[name]:
            raise InvalidCall(f"{call.name} needs a non-empty {name!r}")
    return ValidCall(call.name, out)


def correction_message(error: str) -> str:
    return f"That tool call was invalid ({error}). Call one of the listed tools with valid arguments, or answer directly."


_WORD = re.compile(r"[\w']+")


def scrub_lookup_query(query: str, names: set[str], cap: int = LOOKUP_QUERY_MAX, max_words: int = LOOKUP_QUERY_WORDS) -> str:
    """Drop every known person name (one or two words), collapse spaces, cap words and length at a word boundary.

    The gateway scrubs again with salted hashes; this is the brain-side first line.
    """
    lowered = {n.lower() for n in names}
    words = list(_WORD.finditer(query))
    drop: set[int] = set()
    for i, w in enumerate(words):
        if w.group().lower() in lowered or w.group().lower().removesuffix("'s") in lowered:
            drop.add(i)
        if i + 1 < len(words) and f"{w.group()} {words[i + 1].group()}".lower() in lowered:
            drop.update({i, i + 1})
    out, last = [], 0
    for i, w in enumerate(words):
        if i in drop:
            out.append(query[last : w.start()])
            last = w.end()
    out.append(query[last:])
    cleaned = re.sub(r"\s{2,}", " ", "".join(out)).strip(" ,.'?!")
    cleaned = " ".join(cleaned.split()[:max_words])
    if len(cleaned) > cap:
        cleaned = cleaned[:cap].rsplit(" ", 1)[0].strip(" ,.'")
    return cleaned
