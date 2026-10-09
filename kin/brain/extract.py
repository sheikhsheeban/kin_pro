"""Fact extraction: one finished episode (a conversation) -> self-contained facts, with Gemma 4 E2B.

The model gets the episode's lines with their times and speakers, today's date with a small
calendar, and the people Kin already knows. It answers with a strict JSON array:

  [{"text", "kind", "subject", "people", "event_time", "lines"}]

which is validated here (one corrective retry; a second failure leaves the episode for later).

What a 2B model does unreliably is then made deterministic:
  - dates: "tomorrow at 4" in a cited line is resolved by `kin.tools.timeparse` against the time
    that line was heard; relative words left in the text are replaced by the absolute date.
  - speakers: first-person words are rewritten to "the user" only when every cited line was
    spoken by the owner. Otherwise nobody is guessed: the fact says "Someone said: ...".

Also here: the single ADD / UPDATE / DUPLICATE / CONTRADICTS question asked when a new fact is
close to an active one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from kin.memory.store import Hit, Utterance
from kin.tools.timeparse import parse_when

KINDS = ("plan", "fact", "idea", "note")
MAX_FACT_CHARS = 300

EXTRACT_SYSTEM = (
    "You turn an overheard conversation into short memory facts for the owner of a private voice assistant. "
    "Reply with only a JSON array and nothing else."
)

EXTRACT_PROMPT = """Find what is worth remembering in an overheard conversation: plans and appointments, facts about people, numbers, codes, passwords, prices, places, ideas.
Most overheard lines are small talk and are NOT worth remembering: feelings, complaints, jokes, greetings, questions, and comments on food, drinks, weather, TV, sport or traffic. Never turn small talk into a fact.

Reply with a JSON array, one object per thing worth remembering:
{{"text": "...", "kind": "plan|fact|idea|note", "subject": "...", "people": ["..."], "event_time": "YYYY-MM-DDTHH:MM" or null, "lines": [line numbers used]}}
- text: one self-contained sentence with absolute dates ("on Wednesday 30 September 2026 at 16:00"), never tomorrow, tonight or next Friday. Resolve times from when the line was heard.
- Short lines that belong together become one fact: "Let's meet at 4." + "Where?" + "Library cafe." is one plan with its place.
- Speakers: "owner" is the user; "other" is someone else; "unknown" means nobody knows who spoke. For "owner" write "The user" instead of I/me/my. For "other" or "unknown" never guess who spoke: write "Someone said ...", unless the words name the speaker.
- subject: who the fact is about (a name, "owner", or null if unknown). people: every person named. event_time: when the planned thing happens, else null.
- Keep names, numbers and codes exactly as heard. Never add anything that was not said. If nothing is worth remembering, reply [].

Example 1
[1] Fri 18:02 unknown: Leo's flat is in Hill Road now.
[2] Fri 18:03 unknown: Let's have dinner there next Friday at 8.
Reply: [{{"text": "Leo's flat is in Hill Road.", "kind": "fact", "subject": "Leo", "people": ["Leo"], "event_time": null, "lines": [1]}}, {{"text": "Someone said: let's have dinner at Leo's flat in Hill Road on Friday 16 October 2026 at 20:00.", "kind": "plan", "subject": null, "people": ["Leo"], "event_time": "2026-10-16T20:00", "lines": [1, 2]}}]

Example 2
[1] Sat 09:10 unknown: Ugh, this tea is way too sweet.
[2] Sat 09:11 unknown: What do you want to watch tonight?
[3] Sat 09:11 unknown: Anything but football.
Reply: []

Now the real conversation. Today is {today}. Calendar: {calendar}. Known people: {people}.
{lines}
Reply:"""

RETRY_PROMPT = "That was not valid ({error}). Reply with only the JSON array described above."

KEEP_SYSTEM = "Answer only yes or no."
KEEP_PROMPT = (
    "Would someone want to recall anything in this overheard conversation later (a plan, a fact about a person, "
    "a number, a code, a price, a place, or an idea)? Small talk is no.\nConversation:\n{text}"
)

DEDUP_SYSTEM = "You compare a new memory with stored memories. Reply with one line only."
DEDUP_PROMPT = """Stored memories:
{existing}

New memory: {new}

Reply with exactly one of:
ADD (it is about something different, or only adds a detail)
DUPLICATE <id> (it says the same thing as that memory, with nothing new)
UPDATE <id> (the same plan or fact has CHANGED: a new time, date, place, number or value replaces the old one)
CONTRADICTS <id> (it says that memory is no longer true)"""


class ExtractionError(ValueError):
    pass


@dataclass
class ExtractedFact:
    text: str
    kind: str
    subject: str | None
    people: list[str]
    event_ts: float | None
    lines: list[int]  # 1-based line numbers in the episode
    evidence: list[int] = field(default_factory=list)  # utterance ids
    heard_ts: float = 0.0
    notes: list[str] = field(default_factory=list)  # what the deterministic passes changed
    small_talk: bool = False  # looks like a copy of small talk (see is_small_talk_text)


# --- prompt --------------------------------------------------------------------------------------


def _day(d: datetime) -> str:
    return d.strftime("%a %d %b %Y").replace(" 0", " ")


def calendar(now: datetime) -> str:
    days = [f"today {_day(now)}", f"tomorrow {_day(now + timedelta(days=1))}"]
    days += [f"{(now + timedelta(days=i)).strftime('%a')} {(now + timedelta(days=i)).day}" for i in range(2, 8)]
    return ", ".join(days)


def speaker_label(u: Utterance) -> str:
    return u.speaker or "unknown"


def build_prompt(lines: list[Utterance], now: datetime, people: list[str]) -> str:
    convo = "\n".join(
        f"[{i}] {datetime.fromtimestamp(u.ts).strftime('%a %H:%M')} {speaker_label(u)}: {u.text}" for i, u in enumerate(lines, 1)
    )
    return EXTRACT_PROMPT.format(
        today=now.strftime("%A %d %B %Y, %H:%M"),
        calendar=calendar(datetime.fromtimestamp(lines[0].ts) if lines else now),
        people=", ".join(people) if people else "none yet",
        lines=convo,
    )


# --- validation ----------------------------------------------------------------------------------


def _json_array(raw: str) -> list:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        raise ExtractionError("no JSON array in the reply")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(data, list):
        raise ExtractionError("the reply is not a JSON array")
    return data


def _event_time(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    if not isinstance(value, str):
        raise ExtractionError("event_time must be a string or null")
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), fmt).timestamp()
        except ValueError:
            continue
    raise ExtractionError(f"event_time {value!r} is not YYYY-MM-DDTHH:MM")


def parse_facts(raw: str, n_lines: int) -> list[ExtractedFact]:
    """Validate the model's reply against the schema. Raises ExtractionError with a short reason."""
    out = []
    for i, item in enumerate(_json_array(raw)):
        if not isinstance(item, dict):
            raise ExtractionError(f"item {i} is not an object")
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ExtractionError(f"item {i} has no text")
        kind = str(item.get("kind") or "note").strip().lower()
        if kind not in KINDS:
            kind = "note"
        subject = item.get("subject")
        if subject is not None and not isinstance(subject, str):
            raise ExtractionError(f"item {i}: subject must be a string or null")
        people = item.get("people") or []
        if not isinstance(people, list) or not all(isinstance(p, str) for p in people):
            raise ExtractionError(f"item {i}: people must be a list of names")
        lines = item.get("lines") or []
        if not isinstance(lines, list):
            lines = [lines]
        lines = [int(x) for x in lines if isinstance(x, (int, float, str)) and str(x).strip().isdigit() and 1 <= int(x) <= n_lines]
        out.append(
            ExtractedFact(
                text=" ".join(text.split())[:MAX_FACT_CHARS],
                kind=kind,
                subject=_clean_subject(subject),
                people=[p.strip() for p in people if p.strip() and p.strip().lower() not in _NOT_PEOPLE
                        and not re.search(r"\b(someone|user|speaker)\b", p, re.I)][:8],
                event_ts=_event_time(item.get("event_time")),
                lines=lines,
            )
        )
    return out


_NOT_PEOPLE = {"owner", "user", "the user", "someone", "unknown", "speaker", "i", "me", "you", "null", "none"}


def _clean_subject(subject: str | None) -> str | None:
    if subject is None:
        return None
    s = subject.strip()
    low = s.lower()
    if low in ("owner", "user", "the user", "you"):
        return "owner"
    if not s or low in _NOT_PEOPLE:
        return None
    return s


# --- deterministic passes ----------------------------------------------------------------------------

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_REL_DAY = re.compile(
    r"\b(?:(today|tonight|tomorrow|yesterday)|(?:(?:on|this|next)\s+)?(" + "|".join(_WEEKDAYS) + r"))\b", re.I
)
_NO_ON = frozenset({"to", "by", "until", "till", "from", "before", "after", "for", "since", "than"})
_RECURRING_AFTER = re.compile(r"\s+(nights|mornings|evenings|afternoons)\b", re.I)
# a date right after the weekday: "Friday 2 October", "Monday the 12th", "Tuesday 13th"
_HAS_DATE = re.compile(
    r"(?:the\s+)?\d{1,2}(?:st|nd|rd|th)\b|\d{1,2}\s+(?:of\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.I
)


def _date_words(d: datetime) -> str:
    return f"{d.strftime('%A')} {d.day} {d.strftime('%B %Y')}"


def absolutise(text: str, heard: datetime) -> tuple[str, bool]:
    """Replace relative day words with the absolute date, relative to when the line was heard.

    "Let's meet tomorrow at 4" heard Tue 29 Sep 2026 -> "Let's meet on Wednesday 30 September 2026 at 4".
    A weekday that already has a date next to it ("Friday 2 October") is left alone.
    """
    changed = False

    def sub(m: re.Match) -> str:
        nonlocal changed
        after = text[m.end() : m.end() + 24]
        before = text[max(0, m.start() - 7) : m.start()].lower()
        if _HAS_DATE.match(after.lstrip(" ,")) or before.endswith(("every ", "each ")) or _RECURRING_AFTER.match(after):
            return m.group(0)  # already dated, or recurring ("every Thursday", "Wednesday nights")
        word = (m.group(1) or "").lower()
        if word in ("today", "tonight"):
            d = heard
        elif word == "tomorrow":
            d = heard + timedelta(days=1)
        elif word == "yesterday":
            d = heard - timedelta(days=1)
        else:
            wd = _WEEKDAYS.index(m.group(2).lower())
            d = heard + timedelta(days=(wd - heard.weekday()) % 7 or 7)
        changed = True
        if word == "tonight":
            return f"on the evening of {_date_words(d)}"
        prev = re.search(r"(\w+)\s*$", text[: m.start()])
        if not m.group(0).lower().startswith("on") and prev and prev.group(1).lower() in _NO_ON:
            return _date_words(d)  # "moved to next Tuesday" -> "moved to Tuesday 6 October 2026"
        return f"on {_date_words(d)}"

    out = _REL_DAY.sub(sub, text)
    out = re.sub(r"\bon on\b", "on", out)
    return out, changed


_FIRST_PERSON = re.compile(r"\b(I|I'm|I've|I'll|I'd|(?i:me|my|mine|myself))\b")
_OWNER_SUBS = [
    (re.compile(r"\bI'm\b"), "the user is"), (re.compile(r"\bI've\b"), "the user has"),
    (re.compile(r"\bI'll\b"), "the user will"), (re.compile(r"\bI'd\b"), "the user would"),
    (re.compile(r"\bI am\b"), "the user is"), (re.compile(r"\bI\b"), "the user"),
    (re.compile(r"\bmyself\b", re.I), "the user"), (re.compile(r"\bmine\b", re.I), "the user's"),
    (re.compile(r"\bmy\b", re.I), "the user's"), (re.compile(r"\bme\b", re.I), "the user"),
]


_SECOND_PERSON = re.compile(r"\b(you|your|yours|you're|you've|you'll)\b", re.I)
_THE_USER = re.compile(r"\b[Tt]he user('s)?(?=\W|$)")


def _unquoted(text: str) -> str:
    return re.sub(r'"[^"]*"|“[^”]*”', " ", text)


def resolve_speaker(fact: ExtractedFact, cited: list[Utterance], episode: list[Utterance] | None = None) -> None:
    """First person -> "the user" only when the owner provably spoke every cited line; never guess.

    The first person may sit in a neighbouring line of the episode ("I'm vegan now." / "Since last
    month."), so the whole episode counts when deciding whether "the user" was a guess.
    """
    speakers = {u.speaker for u in cited}
    owner_only = bool(cited) and speakers == {"owner"}
    context = episode or cited
    heard = " ".join(u.text for u in cited)
    first_person = bool(_FIRST_PERSON.search(" ".join(u.text for u in context if u.speaker != "owner")))
    addressed = bool(_SECOND_PERSON.search(heard))
    if fact.subject == "owner" and not owner_only and first_person:
        fact.subject = None  # "I'm vegan" from an unknown speaker is not about the owner
        fact.notes.append("subject unknown: first person from an unidentified speaker")
    if not owner_only and first_person and not addressed and _THE_USER.search(fact.text):
        # The model decided an unidentified "I" was the user. Nobody knows that: undo the guess.
        # (With "you/your" in the lines the owner was being addressed, so "the user" may be right.)
        fact.text = _THE_USER.sub(lambda m: "Someone's" if m.group(1) else "Someone", fact.text)
        fact.text = fact.text[:1].upper() + fact.text[1:]
        fact.notes.append("'the user' guessed for an unidentified speaker -> someone")
    if not _FIRST_PERSON.search(_unquoted(fact.text)):
        return
    if owner_only:
        t = fact.text
        for pat, rep in _OWNER_SUBS:
            t = pat.sub(rep, t)
        fact.text = t[:1].upper() + t[1:]
        fact.notes.append("first person -> the user (owner spoke)")
    elif not fact.text.lower().startswith("someone said"):
        fact.text = f"Someone said: {fact.text}"
        fact.notes.append("speaker unknown: kept as 'Someone said'")


def resolve_time(fact: ExtractedFact, cited: list[Utterance]) -> None:
    """event_ts from the cited line's own time words, relative to when that line was heard."""
    for u in cited:
        if re.search(r"\b(every|each)\b", u.text, re.I):
            continue  # recurring: no single event time
        heard = datetime.fromtimestamp(u.ts)
        when = parse_when(u.text, heard)
        if when.due is not None and not when.relative:
            if fact.event_ts is not None and abs(abs(fact.event_ts - when.due.timestamp()) - 12 * 3600) < 60:
                break  # same day, am/pm differs: the model saw the context ("dinner at 8" is 20:00), keep its time
            if fact.event_ts is None or abs(fact.event_ts - when.due.timestamp()) > 60:
                fact.notes.append(f"event time from line: {when.due:%Y-%m-%d %H:%M}")
            fact.event_ts = when.due.timestamp()
            break
    base = datetime.fromtimestamp(cited[0].ts) if cited else None
    if base is not None:
        fact.text, changed = absolutise(fact.text, base)
        if changed:
            fact.notes.append("relative day words replaced")


_CAPWORD = re.compile(r"\b[A-Z][a-z']+")
_STARTERS = frozenset("Someone The I I'm I've I'll Let's Let Do Does Did Is Are Was Were Nice Time Lunch Dinner Okay Ok Good "
                      "Great That That's This It It's No Yes So Oh Wow Ugh Hmm Thanks Thank Sorry What How Why When Where Who "
                      "Can Could Would Will Maybe We You My Our Your Just Not".split())


def is_small_talk_text(f: ExtractedFact, cited: list[Utterance]) -> bool:
    """Text half of the small-talk check: no number, no event time, no name, and a question or a near copy of
    the line it came from ("Let's get lunch.", "Do you want to come over for dinner sometime?"). Text alone
    cannot tell "I keep forgetting to drink water" from "I'm allergic to peanuts", so the fact is dropped
    only if the semantic router also calls it chatter (Brain._store_fact)."""
    from rapidfuzz import fuzz

    if not cited or f.event_ts is not None:
        return False
    body = re.sub(r"^someone said:\s*", "", f.text.strip(), flags=re.I)
    if any(ch.isdigit() for ch in body) or [w for w in _CAPWORD.findall(body) if w not in _STARTERS]:
        return False
    heard = " ".join(u.text for u in cited)
    return heard.rstrip().endswith("?") or fuzz.token_set_ratio(body.lower(), heard.lower()) >= 85


def finish(facts: list[ExtractedFact], lines: list[Utterance]) -> list[ExtractedFact]:
    """Evidence ids, heard time, dates and speakers for each validated fact; small talk is flagged."""
    out = []
    for f in facts:
        cited = [lines[i - 1] for i in f.lines] or list(lines)
        f.evidence = [u.id for u in cited]
        f.heard_ts = max(u.ts for u in cited)
        resolve_time(f, cited)
        resolve_speaker(f, cited, lines)
        f.small_talk = is_small_talk_text(f, cited)  # confirmed by the router when the fact is stored
        out.append(f)
    return out


# --- running the model ---------------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    facts: list[ExtractedFact]
    llm_calls: int
    raw: list[str]


class Extractor:
    """Runs on the LLM worker thread with the shared main LLM (never owns or closes it)."""

    def __init__(self, load_llm, max_output_tokens: int = 400) -> None:
        self._load = load_llm
        self.max_output_tokens = max_output_tokens

    def extract(self, lines: list[Utterance], now: datetime, people: list[str]) -> ExtractionResult:
        llm = self._load()
        session = llm.session(EXTRACT_SYSTEM, max_output_tokens=self.max_output_tokens)
        raws: list[str] = []
        try:
            message = build_prompt(lines, now, people)
            for attempt in range(2):
                raw = "".join(c.text for c in session.send(message))
                raws.append(raw)
                try:
                    return ExtractionResult(finish(parse_facts(raw, len(lines)), lines), attempt + 1, raws)
                except ExtractionError as exc:
                    if attempt:
                        raise ExtractionError(f"{exc} (after one retry)") from exc
                    message = RETRY_PROMPT.format(error=exc)
        finally:
            session.close()
        raise AssertionError("unreachable")

    def worth(self, lines: list[Utterance]) -> bool:
        """A 3-token yes/no before the full extraction: is anything in this episode worth keeping?

        The same question measured 22/24 keep/drop on single lines with E2B (v3 pending tagger). It
        costs ~1 s, so small talk no longer pays for a ~10 s JSON extraction, and E2B (which turns
        almost any line into a "fact" when asked for JSON) gets a second, simpler chance to say no.
        """
        llm = self._load()
        convo = "\n".join(u.text for u in lines)
        out = llm.generate(KEEP_SYSTEM, KEEP_PROMPT.format(text=convo), max_output_tokens=3)
        return out.strip().lower().startswith("yes")

    def classify(self, new_text: str, candidates: list[Hit]) -> tuple[str, int | None]:
        """ADD / UPDATE / DUPLICATE / CONTRADICTS for a new fact against close active ones."""
        llm = self._load()
        existing = "\n".join(f"[{h.fact.id}] {h.fact.text}" for h in candidates)
        out = llm.generate(DEDUP_SYSTEM, DEDUP_PROMPT.format(existing=existing, new=new_text), max_output_tokens=12)
        return parse_decision(out, {h.fact.id for h in candidates})


_DECISION = re.compile(r"\b(ADD|UPDATE|DUPLICATE|CONTRADICTS)\b\s*(?:<|\[|#)?\s*(\d+)?", re.I)


def parse_decision(out: str, allowed: set[int]) -> tuple[str, int | None]:
    m = _DECISION.search(out or "")
    if not m:
        return "ADD", None
    verb = m.group(1).upper()
    if verb == "ADD":
        return "ADD", None
    fid = int(m.group(2)) if m.group(2) else (next(iter(allowed)) if len(allowed) == 1 else None)
    if fid not in allowed:
        return "ADD", None  # an invented id is never acted on
    return verb, fid
