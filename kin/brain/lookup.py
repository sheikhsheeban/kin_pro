"""The `lookup` tool's arguments -> the gateway's `lookup(kind, params)`, and its results back to text.

The model calls lookup(kind, query, when). Only the fields the gateway allows are built here:
  weather -> {"city", "date"}   city only if the query looks like a place, else the home city
  news    -> {"topic"}          names scrubbed, capped
  search  -> {"query"}          names scrubbed, capped
Values are not otherwise "fixed": the gateway validates them and refuses (and audits) anything odd.
Results go back to the model as one compact line of facts (small models reason better over short
text than JSON).
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any

from kin.brain.tools_schema import LOOKUP_QUERY_MAX, scrub_lookup_query

NEWS_TOPIC_MAX = 60  # the gateway's own cap for news topics

CHECKED = {
    "weather": "I checked the weather online.",
    "news": "I checked the news online.",
    "search": "I looked that up online.",
}
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_DAY_WORDS = re.compile(
    r"\b(?:today|tonight|tomorrow|now|this (?:morning|afternoon|evening|weekend)|next week|weather|forecast|"
    r"in|for|at|on|the|(?:next\s+)?(?:" + "|".join(_WEEKDAYS) + r"))\b",
    re.I,
)
_PLACE_WORD = re.compile(r"^(?:[A-Z][\w'.<>/-]*|of|upon|on|de|la|le|am|-)$")


def failure_sentence(kind: str, reason: str | None) -> str:
    what = {"weather": "the weather", "news": "the news", "search": "that"}.get(kind, "that")
    if reason == "offline":
        return f"I can't check {what} online right now, because I'm in offline mode."
    if reason == "gateway_unavailable":
        return f"I can't check {what} online right now, because my lookup service isn't running."
    if reason == "personal_name":
        return "I won't send a person's name online, so I didn't look that up."
    if reason in ("invalid", "too_large"):
        return f"I couldn't form a safe lookup for {what}, so I didn't go online."
    return f"I tried to check {what} online, but the service didn't answer."


def to_date(value: Any, today: date) -> date:
    """"today" / "tomorrow" / a weekday / YYYY-MM-DD, then dateparser for anything else; today if unknown."""
    s = str(value or "").strip().lower().rstrip(".")
    s = re.sub(r"^(?:on|this|for)\s+", "", s)
    if s in ("", "today", "now", "this afternoon", "this evening", "tonight", "this morning", "later"):
        return today
    if s == "tomorrow" or s.startswith("tomorrow "):
        return today + timedelta(days=1)
    if s in _WEEKDAYS:
        return today + timedelta(days=(_WEEKDAYS.index(s) - today.weekday()) % 7)
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        pass
    import dateparser

    base = datetime.combine(today, datetime.min.time()).replace(hour=12)
    parsed = dateparser.parse(s, settings={"PREFER_DATES_FROM": "future", "RELATIVE_BASE": base})
    return parsed.date() if parsed else today


def place_in(query: str, names: set[str]) -> str | None:
    """The place named in a weather query ("Nagpur", "New York tomorrow"), or None.

    Only capitalised words count, so "coffee with Priya" or "tomorrow" fall back to the home city.
    """
    rest = _DAY_WORDS.sub(" ", scrub_lookup_query(query, names, cap=LOOKUP_QUERY_MAX))
    words = rest.split()
    if not words or len(words) > 4 or any(ch.isdigit() for ch in rest):
        return None
    if not words[0][:1].isupper() or not all(_PLACE_WORD.match(w) for w in words):
        return None
    return " ".join(words).strip(" ,.")


def to_request(args: dict[str, Any], *, home_city: str, names: set[str], today: date | None = None) -> tuple[str, dict[str, Any]]:
    """Validated lookup arguments -> (kind, params) with only the fields the gateway allows."""
    kind = args["kind"]
    query = str(args.get("query") or "")
    today = today or date.today()
    if kind == "weather":
        when = args.get("when")
        day = (_day_hint(when) or when) if when else _day_hint(query)  # "weather tomorrow" may carry the day in the query
        return kind, {"city": place_in(query, names) or home_city, "date": to_date(day, today).isoformat()}
    if kind == "news":
        return kind, {"topic": scrub_lookup_query(query, names, cap=NEWS_TOPIC_MAX)}
    return kind, {"query": scrub_lookup_query(query, names, cap=LOOKUP_QUERY_MAX)}


def _day_hint(text: str) -> str:
    """The day phrase inside free text ("weather in Pune tomorrow" -> "tomorrow")."""
    m = re.search(r"\b(today|tonight|tomorrow|(?:on\s+)?(?:" + "|".join(_WEEKDAYS) + r")|\d{4}-\d{2}-\d{2})\b", str(text), re.I)
    return m.group(1) if m else ""


def render(kind: str, data: dict[str, Any]) -> str:
    """Compact, model-friendly facts. Cite them as [web]."""
    if kind == "weather":
        def hour(h: dict) -> str:
            return " ".join(str(p) for p in (h["time"], f"{h['rain_chance_pct']}%", h.get("sky"), h.get("temp_c")) if p is not None)

        day = [data.get("sky"), f"{data.get('min_temp_c')}-{data.get('max_temp_c')} °C",
               f"wind up to {data['max_wind_kmh']} km/h" if data.get("max_wind_kmh") is not None else None,
               f"humidity {data['humidity_pct']}%" if data.get("humidity_pct") is not None else None,
               f"UV up to {data['uv_index_max']}" if data.get("uv_index_max") is not None else None,
               f"rain {data.get('rain_mm_total')} mm"]
        hours = "; ".join(hour(h) for h in data.get("hourly", []))
        return (
            f"Forecast for {data.get('city')} on {data.get('date')}: {', '.join(p for p in day if p)}. "
            f"By hour (rain chance, sky, °C): {hours}. Cite as [web]. Answer what was asked: the sky and temperature, "
            "plus wind, UV or humidity if notable. Mention rain or an umbrella only if asked, or if the rain chance "
            "is 40% or more when it matters (then advise an umbrella)."
        )
    if kind == "news":
        return kind, {"topic": scrub_lookup_query(query, names, cap=NEWS_TOPIC_MAX)}
    return kind, {"query": scrub_lookup_query(query, names, cap=LOOKUP_QUERY_MAX)}


def _day_hint(text: str) -> str:
    """The day phrase inside free text ("weather in Pune tomorrow" -> "tomorrow")."""
    m = re.search(r"\b(today|tonight|tomorrow|(?:on\s+)?(?:" + "|".join(_WEEKDAYS) + r")|\d{4}-\d{2}-\d{2})\b", str(text), re.I)
    return m.group(1) if m else ""


def render(kind: str, data: dict[str, Any]) -> str:
    """Compact, model-friendly facts. Cite them as [web]."""
    if kind == "weather":
        def hour(h: dict) -> str:
            parts = [f"{h['rain_chance_pct']}% rain chance"]
            parts += [p for p in (h.get("sky"), f"{h['temp_c']} °C" if h.get("temp_c") is not None else None) if p]
            return f"{h['time']} {', '.join(parts)}"

        day = [f"{data.get('sky')}" if data.get("sky") else None,
               f"{data.get('min_temp_c')} to {data.get('max_temp_c')} °C",
               f"wind up to {data['max_wind_kmh']} km/h" if data.get("max_wind_kmh") is not None else None,
               f"humidity {data['humidity_pct']}%" if data.get("humidity_pct") is not None else None,
               f"UV index up to {data['uv_index_max']}" if data.get("uv_index_max") is not None else None,
               f"total rain {data.get('rain_mm_total')} mm"]
        hours = "; ".join(hour(h) for h in data.get("hourly", []))
        return (
            f"Forecast for {data.get('city')} on {data.get('date')}: {', '.join(p for p in day if p)}. By hour: {hours}. "
            "Cite as [web]. Answer what the user asked, in the words of this forecast: the sky and the temperature "
            "(hot above 32 °C, cold below 10 °C), and wind, UV or humidity when they stand out. Mention rain or an "
            "umbrella only if the user asked about it, or the rain chance reaches 40% at the time that matters "
            "(then: take an umbrella). Never say there is no rain unless they asked."
        )
    if kind == "news":
        items = "; ".join(f"{h['headline']} ({h.get('source') or 'unknown'})" for h in data.get("headlines", [])[:5])
        return f"Headlines about {data.get('topic')}: {items or 'none found'}. Cite as [web]."
    results = " | ".join(f"{r['title']}: {r['snippet']}" for r in data.get("results", [])[:3])
    return f"Search results for {data.get('query')}: {results or 'nothing found'}. Cite as [web]."

