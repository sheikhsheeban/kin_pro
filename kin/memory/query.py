"""Query understanding for recall, without an LLM call: when, who, and which way in time.

A question becomes a `QueryPlan`:
  window   a time range from the words ("yesterday", "last week", "on Friday", "tomorrow", "on 14 November",
           "3 days ago", "later in October"), relative to `now` in local time
  field    which timestamp the window applies to: past-tense questions ("what did Mum tell me on Monday?")
           match what was heard OR happened then; others ("what's on tomorrow?") match when things happen
  people   known people named in the question (exact, alias, or a near-miss spelling via rapidfuzz)
  future   "when am I ...", "what's next", "coming up": upcoming events are preferred
  history  "what was it before", "originally": superseded facts are searched too

Regexes cover the common spoken forms; explicit dates go through dateparser with `now` as the base,
so the result never depends on the wall clock.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from kin.memory.people import NameIndex

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
MONTHS = tuple(m.lower() for m in calendar.month_name[1:])
_MONTH_ABBR = {m[:3]: i + 1 for i, m in enumerate(MONTHS)} | {m: i + 1 for i, m in enumerate(MONTHS)}
_NUMWORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "a couple of": 2, "couple of": 2, "a few": 3, "few": 3}
_NUMW = "|".join(sorted(map(re.escape, _NUMWORDS), key=len, reverse=True))

_PAST = re.compile(
    r"\b(did|was|were|said|told|tell me|met|had|heard|planned|mentioned|booked|submitted|spent|bought|gave|paid|"
    r"talked|spoke|asked|happened|went|ago|last)\b", re.I)
_FUTURE = re.compile(
    r"\b(when (is|am|are|do|does|will|should)|what'?s (next|on|coming|up)|upcoming|coming up|next|later|"
    r"tomorrow|tonight|soon|plans? for|do i have (to|on|any)|have i got|am i (meeting|seeing|going|doing))\b", re.I)
_HISTORY = re.compile(
    r"\b(before( it| that)? (changed|moved)|originally|original|used to|previous(ly)?|first (said|planned|time)|"
    r"old (time|plan|date|number)|what was it before|was it (before|first)|changed? (from|it)|history of)\b", re.I)

_DAY = re.compile(r"\b(?:(last|this|next|on|on this|on last|on next|coming)\s+)?(" + "|".join(WEEKDAYS) + r")s?\b", re.I)
_AGO = re.compile(rf"\b(\d+|{_NUMW})\s+(day|week)s?\s+ago\b", re.I)
_IN_DAYS = re.compile(rf"\bin\s+(\d+|{_NUMW})\s+(day|week)s?\b", re.I)
_ORD = r"(\d{1,2})(?:st|nd|rd|th)?"
_MON = r"(" + "|".join(sorted(_MONTH_ABBR, key=len, reverse=True)) + r")\.?"
_DATE_DM = re.compile(rf"\b(?:on\s+)?(?:the\s+)?{_ORD}\s+(?:of\s+)?{_MON}\b", re.I)
_DATE_MD = re.compile(rf"\b(?:on\s+)?{_MON}\s+(?:the\s+)?{_ORD}\b", re.I)
_DATE_NUM = re.compile(r"\b(?:on\s+)?(\d{1,2})[/.](\d{1,2})(?:[/.](\d{2,4}))?\b")
_THE_NTH = re.compile(r"\bon\s+the\s+(\d{1,2})(?:st|nd|rd|th)\b", re.I)
_MONTH = re.compile(r"(?<!from )(?<!since )(?<!until )(?<!till )(?<!by )(?<!before )(?<!after )"  # "rent from November" is content
                    r"\b(?:(later|earlier|early|late|end of|rest of)\s+)?(?:(?:in|this|next|last|of)\s+)*("
                    + "|".join(MONTHS) + r")\b(?!\s+(?:the\s+)?\d)", re.I)
_REL_MONTH = re.compile(r"\b(?:(later|earlier|rest of)\s+)?(this|next|last)\s+month\b", re.I)


@dataclass(frozen=True)
class QueryPlan:
    text: str  # the question, misheard names normalised
    window: tuple[float, float] | None = None
    time_field: str = "either"  # heard | event | either: which timestamp `window` applies to
    label: str = ""  # the phrase that set the window, for the "why this answer" panel
    people: tuple[str, ...] = ()  # canonical names of known people in the question
    unknown_names: tuple[str, ...] = ()  # capitalised names in the question that nobody in memory has
    future: bool = False
    past: bool = False
    history: bool = False
    alt: str | None = None  # the model's rewording (recall tool argument), searched as a second query
    fixes: dict = field(default_factory=dict)  # {heard spelling: canonical name}
    original: str = ""  # the question exactly as heard, before names were normalised

    @property
    def search_original(self) -> str | None:
        """The question as heard (time phrase removed), when normalising changed a name: searched too, so a
        fact stored with the heard spelling ("Raul") is still found for "Rahul" and the other way round."""
        if not self.fixes or not self.original or self.original == self.text:
            return None
        cut = self.original
        if self.label:
            cut = re.sub(re.escape(self.label), " ", cut, count=1, flags=re.I)
        return re.sub(r"\s{2,}", " ", cut).strip()

    @property
    def search(self) -> str:
        """What is searched: the question without its time phrase, which the window already covers
        ("What did Priyanka tell me on Thursday?" must not prefer facts that merely mention a Thursday)."""
        if not self.label:
            return self.text
        cut = re.sub(re.escape(self.label), " ", self.text, count=1, flags=re.I)
        cut = re.sub(r"\s+([?.!,])", r"\1", re.sub(r"\s{2,}", " ", cut)).strip()
        return cut if len(cut.split()) >= 2 else self.text

    def with_alt(self, alt: str | None) -> "QueryPlan":
        from dataclasses import replace

        alt = (alt or "").strip()
        return replace(self, alt=alt if alt and alt.lower() != self.text.lower() else None)

    def describe(self) -> dict:
        d = {"query": self.text, "searched": self.search, "alt": self.alt, "people": list(self.people), "future": self.future,
             "history": self.history, "time": None}
        if self.window:
            a, b = (datetime.fromtimestamp(t) for t in self.window)
            d["time"] = {"label": self.label, "field": self.time_field,
                         "from": a.isoformat(timespec="minutes"), "to": b.isoformat(timespec="minutes")}
        return d


def _start(d: datetime) -> datetime:
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


def _span(a: datetime, b: datetime) -> tuple[float, float]:
    return a.timestamp(), b.timestamp()


def _days(d: datetime, n: int = 1) -> tuple[float, float]:
    s = _start(d)
    return _span(s, s + timedelta(days=n))


def _num(tok: str) -> int:
    t = tok.lower()
    return int(t) if t.isdigit() else _NUMWORDS[t]


def _month_window(month: int, now: datetime, modifier: str | None, past: bool, rel: str | None = None) -> tuple[float, float]:
    year = now.year
    if rel == "next" or (rel is None and not past and month < now.month):
        year += rel is None  # "in March" asked in October: next March
    if rel == "last" or (rel is None and past and month > now.month):
        year -= 1
    first = datetime(year, month, 1)
    last = datetime(year + (month == 12), month % 12 + 1, 1)
    mod = (modifier or "").lower()
    if mod in ("later", "rest of") and first <= now < last:
        return _span(now, last)
    if mod in ("earlier",) and first <= now < last:
        return _span(first, now)
    if mod in ("early",):
        return _span(first, first + timedelta(days=10))
    if mod in ("late", "end of"):
        return _span(last - timedelta(days=10), last)
    return _span(first, last)


def time_window(question: str, now: datetime) -> tuple[tuple[float, float] | None, str, bool]:
    """(window, the phrase that set it, past tense?). Past-tense questions look back; others look ahead."""
    q = question.lower()
    past = bool(_PAST.search(q))

    def found(w: tuple[float, float], m: re.Match | str) -> tuple[tuple[float, float], str, bool]:
        return w, (m if isinstance(m, str) else m.group(0)).strip(), past

    if m := re.search(r"\bday after tomorrow\b", q):
        return found(_days(now + timedelta(days=2)), m)
    if m := re.search(r"\bday before yesterday\b", q):
        return found(_days(now - timedelta(days=2)), m)
    if m := re.search(r"\byesterday\b", q):
        return found(_days(now - timedelta(days=1)), m)
    if m := re.search(r"\blast night\b", q):
        y = _start(now) - timedelta(days=1)
        return found(_span(y + timedelta(hours=12), _start(now) + timedelta(hours=6)), m)
    if m := re.search(r"\btomorrow\b", q):
        return found(_days(now + timedelta(days=1)), m)
    if m := re.search(r"\b(today|this morning|this afternoon|this evening|tonight|earlier today|so far today)\b", q):
        return found(_days(now), m)
    if m := _AGO.search(q):
        n = _num(m.group(1)) * (7 if m.group(2).lower() == "week" else 1)
        if m.group(1).lower() in ("few", "a few", "couple of", "a couple of"):
            return found(_span(_start(now - timedelta(days=n + 2)), _start(now)), m)
        return found(_days(now - timedelta(days=n), 7 if m.group(2).lower() == "week" else 1), m)
    if m := _IN_DAYS.search(q):
        n = _num(m.group(1)) * (7 if m.group(2).lower() == "week" else 1)
        return found(_days(now + timedelta(days=n)), m)
    monday = _start(now) - timedelta(days=now.weekday())
    if m := re.search(r"\blast weekend\b", q):
        sat = monday - timedelta(days=2)
        return found(_span(sat, sat + timedelta(days=2)), m)
    if m := re.search(r"\b(this|the|next|coming)\s+weekend\b", q):
        sat = monday + timedelta(days=5) + timedelta(days=7 if m.group(1).lower() == "next" else 0)
        if past and m.group(1).lower() in ("the", "this") and now < sat:
            sat -= timedelta(days=7)  # "what did we do at the weekend?" asked on a weekday
        return found(_span(sat, sat + timedelta(days=2)), m)
    if m := re.search(r"\blast week\b", q):
        return found(_span(_start(now - timedelta(days=7)), now), m)
    if m := re.search(r"\bnext week\b", q):
        return found(_span(monday + timedelta(days=7), monday + timedelta(days=14)), m)
    if m := re.search(r"\b(this week|later this week|rest of the week)\b", q):
        return found(_span(monday, now) if past else _span(now, monday + timedelta(days=7)), m)
    for rx in (_DATE_DM, _DATE_MD):
        if m := rx.search(q):
            a, b = m.group(1), m.group(2)
            day, mon = (int(a), _MONTH_ABBR[b.lower()]) if rx is _DATE_DM else (int(b), _MONTH_ABBR[a.lower()])
            d = _date(now, mon, day, past)
            if d is not None:
                return found(_days(d), m)
    if m := _DATE_NUM.search(q):
        d = _parse_numeric(m.group(0), now, past)
        if d is not None:
            return found(_days(d), m)
    if m := _THE_NTH.search(q):
        day = int(m.group(1))
        month = now.month
        if not past and day < now.day:
            month = month % 12 + 1
        elif past and day > now.day:
            month = (month - 2) % 12 + 1
        year = now.year + (not past and month < now.month) - (past and month > now.month)
        try:
            return found(_days(datetime(year, month, day)), m)
        except ValueError:
            pass
    m = _DAY.search(q)
    if m:
        which, wd = (m.group(1) or "").lower(), WEEKDAYS.index(m.group(2).lower())
        if "last" in which or (("next" not in which and "coming" not in which) and past):
            back = (now.weekday() - wd) % 7 or 7
            return found(_days(now - timedelta(days=back)), m)
        ahead = (wd - now.weekday()) % 7
        if ahead == 0 and "next" not in which and "coming" not in which:
            return found(_days(now), m)  # "on Monday", asked on a Monday: today
        return found(_days(now + timedelta(days=ahead or 7)), m)
    if m := _REL_MONTH.search(q):
        mod, rel = m.group(1), m.group(2).lower()
        month = now.month if rel == "this" else (now.month % 12 + 1 if rel == "next" else (now.month - 2) % 12 + 1)
        return found(_month_window(month, now, mod, past, rel if rel != "this" else None), m)
    if m := _MONTH.search(q):
        mon = _MONTH_ABBR[m.group(2).lower()]
        if m.group(2).lower() == "may" and not re.search(r"\b(in|of|this|next|last|early|late|later)\s+may\b", q):
            return None, "", past  # "may" as a verb
        return found(_month_window(mon, now, m.group(1), past), m)
    return None, "", past


def _date(now: datetime, month: int, day: int, past: bool) -> datetime | None:
    for year in ((now.year, now.year - 1) if past else (now.year, now.year + 1)):
        try:
            d = datetime(year, month, day)
        except ValueError:
            return None
        if (past and _start(d) <= now) or (not past and d >= _start(now)):
            return d
    return None


def _parse_numeric(text: str, now: datetime, past: bool) -> datetime | None:
    """"5/10" or "05.10.2026": day first (India / UK), through dateparser so odd forms still parse."""
    import dateparser

    settings = {"RELATIVE_BASE": now, "DATE_ORDER": "DMY", "PREFER_DATES_FROM": "past" if past else "future",
                "RETURN_AS_TIMEZONE_AWARE": False, "PREFER_DAY_OF_MONTH": "first"}
    try:
        d = dateparser.parse(text.replace("on ", ""), languages=["en"], settings=settings)
    except Exception:
        return None
    return d


# --- people ------------------------------------------------------------------------------------------

_CAPWORD = re.compile(r"\b([A-Z][a-z]{2,20})(?='s\b|\b)")
_NOT_PEOPLE = {w.lower() for w in WEEKDAYS} | set(MONTHS) | {
    "what", "when", "where", "who", "whom", "which", "why", "how", "does", "did", "the", "and", "can", "could", "tell",
    "remind", "is", "are", "was", "were", "will", "would", "should", "have", "has", "had", "do", "am", "my", "any",
    "today", "tomorrow", "yesterday", "kin", "hey", "jarvis", "please",
    # relations and titles are not names (and never a reason to doubt the question)
    "mum", "mom", "mummy", "dad", "daddy", "papa", "aunty", "auntie", "uncle", "sister", "brother", "grandma", "grandpa",
    "nani", "dadi", "boss", "manager", "doctor", "dr", "mr", "mrs", "ms", "sir", "madam",
}


def people_in(question: str, idx: NameIndex) -> tuple[list[str], list[str], dict[str, str]]:
    """(known people named, capitalised names nobody in memory has, {heard: canonical})."""
    known: list[str] = []
    unknown: list[str] = []
    fixes: dict[str, str] = {}
    lowered = question.lower()
    for p in idx.people:  # exact names and aliases, also when the STT wrote them in lower case
        for spelling in (p.name, *p.aliases):
            if len(spelling) >= 3 and re.search(r"(?<![a-z])" + re.escape(spelling.lower()) + r"(?![a-z])", lowered):
                known.append(p.name)
                break
    for m in _CAPWORD.finditer(question):
        word = m.group(1)
        if word.lower() in _NOT_PEOPLE:
            continue
        canon = idx.canonical(word)
        if canon is not None:
            if canon != word:
                fixes[word] = canon
            known.append(canon)
        else:
            unknown.append(word)
    known = list(dict.fromkeys(known))
    unknown = [u for u in dict.fromkeys(unknown) if u not in known]
    return known, unknown, fixes


def understand(question: str, now: datetime, idx: NameIndex | None = None) -> QueryPlan:
    idx = idx or NameIndex([])
    text, fixes = idx.normalise_text(question)
    window, label, past = time_window(text, now)
    people, unknown, more = people_in(text, idx)
    future = not past and bool(_FUTURE.search(text))
    history = bool(_HISTORY.search(text))
    if window is None:
        fld = "either"
    elif past:
        fld = "either"  # what was heard then, or what happened then
    elif window[1] <= now.timestamp():
        fld = "either"
    else:
        fld = "event"  # "what's on tomorrow?": when things happen
    return QueryPlan(
        text=text, window=window, time_field=fld, label=label, people=tuple(people), unknown_names=tuple(unknown),
        future=future or (window is not None and fld == "event"), past=past, history=history, fixes={**fixes, **more},
        original=question,
    )
