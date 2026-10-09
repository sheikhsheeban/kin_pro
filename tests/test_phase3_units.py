"""Unit tests: time parsing, due parsing, name extraction, web citations, lookup mapping."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from kin.brain import lookup as lk
from kin.brain.gate import Gate, sources_in, strip_citations
from kin.brain.names import extract_names
from kin.tools.timeparse import describe, parse_due, parse_when

NOW = datetime(2026, 10, 1, 10, 30)  # a Thursday morning


@pytest.mark.parametrize(
    ("text", "due", "rest"),
    [
        ("email the professor at 5", datetime(2026, 10, 1, 17, 0), "email the professor"),
        ("call mom at 5:30 p.m.", datetime(2026, 10, 1, 17, 30), "call mom"),
        ("take pills at 9", datetime(2026, 10, 1, 21, 0), "take pills"),  # 9 am already passed
        ("take pills at 11", datetime(2026, 10, 1, 11, 0), "take pills"),
        ("check the oven in 10 minutes", datetime(2026, 10, 1, 10, 40), "check the oven"),
        ("stretch in an hour", datetime(2026, 10, 1, 11, 30), "stretch"),
        ("water the plants in half an hour", datetime(2026, 10, 1, 11, 0), "water the plants"),
        ("buy milk tomorrow", datetime(2026, 10, 2, 9, 0), "buy milk"),
        ("tomorrow at 8 am buy milk", datetime(2026, 10, 2, 8, 0), "buy milk"),
        ("submit the report on Friday at 3", datetime(2026, 10, 2, 15, 0), "submit the report"),
        ("pick up the kids at noon", datetime(2026, 10, 1, 12, 0), "pick up the kids"),
        ("call dad this evening", datetime(2026, 10, 1, 18, 0), "call dad"),
        ("email the professor at five", datetime(2026, 10, 1, 17, 0), "email the professor"),
        ("book flights at 7 in the morning", datetime(2026, 10, 2, 7, 0), "book flights"),
    ],
)
def test_parse_when(text, due, rest):
    when = parse_when(text, NOW)
    assert when.due == due and when.rest == rest


def test_parse_when_without_time_and_describe():
    assert parse_when("do the laundry", NOW).due is None
    assert describe(datetime(2026, 10, 1, 17, 0), NOW) == "at 5 PM today"
    assert describe(datetime(2026, 10, 1, 10, 50), NOW) == "at 10:50 AM today"  # spoken as a clock time
    assert describe(datetime(2026, 10, 1, 10, 40), NOW, relative=True) == "in 10 minutes"
    assert describe(datetime(2026, 10, 2, 9, 0), NOW) == "at 9 AM tomorrow"
    assert describe(datetime(2026, 10, 5, 9, 0), NOW) == "at 9 AM on Monday"


@pytest.mark.parametrize(
    ("due", "expected", "relative"),
    [
        ("at 5", datetime(2026, 10, 1, 17, 0), False),
        ("5", datetime(2026, 10, 1, 17, 0), False),  # a bare hour from the model
        ("5 PM", datetime(2026, 10, 1, 17, 0), False),
        ("17:00", datetime(2026, 10, 1, 17, 0), False),  # dateparser
        ("tomorrow at 9", datetime(2026, 10, 2, 9, 0), False),
        ("in 10 minutes", datetime(2026, 10, 1, 10, 40), True),
        ("in half an hour", datetime(2026, 10, 1, 11, 0), True),
        ("October 12", datetime(2026, 10, 12, 9, 0), False),  # dateparser, a date without a time -> 9 am
        (None, None, False),
        ("", None, False),
        ("whenever", None, False),
    ],
)
def test_parse_due(due, expected, relative):
    w = parse_due(due, NOW)
    assert w.due == expected and (w.relative == relative or expected is None)


def test_name_extraction_is_conservative():
    assert extract_names("I'm meeting Priya at 4 pm at the library cafe.") == {"Priya"}
    assert extract_names("Rahul is allergic to peanuts.") == {"Rahul"}
    assert extract_names("Call Sam and Alex tomorrow.") == {"Sam", "Alex"}
    assert extract_names("I live in London and work in Cambridge.") == set()
    assert extract_names("The Library is closed on Monday.") == set()
    assert extract_names("Coffee with London friends.", {"London"}) == set()


def test_web_citations_in_every_spelling():
    g = Gate(0.5)
    for s in ["Take an umbrella [web].", "Take an umbrella [#web].", "Take one [#7, [web]].", "Take one [#7][web]."]:
        assert g.check(s, {7}, web_ok=True).speakable, s
        assert not g.check(s, {7}, web_ok=False).speakable, s
    assert sources_in("Rain at 4 [#7, web].").ids == (7,) and sources_in("Rain at 4 [#7, web].").web
    assert strip_citations("Prices [in GBP] rose [web].") == "Prices [in GBP] rose."


def test_lookup_mapping_copies_only_allowed_fields_and_scrubs_names():
    today = date(2026, 10, 1)
    names = {"Priya", "Rahul"}

    def req(**args):
        return lk.to_request({"query": "", "when": None} | args, home_city="London", names=names, today=today)

    assert req(kind="weather", query="Paris", when="tomorrow") == ("weather", {"city": "Paris", "date": "2026-10-02"})
    assert req(kind="weather") == ("weather", {"city": "London", "date": "2026-10-01"})
    assert req(kind="weather", query="coffee with Priya") == ("weather", {"city": "London", "date": "2026-10-01"})
    assert req(kind="weather", query="New York on Friday") == ("weather", {"city": "New York", "date": "2026-10-02"})
    assert req(kind="news", query="what Priya said about Raspberry Pi") == ("news", {"topic": "what said about Raspberry Pi"})
    assert req(kind="search", query="Rahul's favourite band") == ("search", {"query": "favourite band"})
    _, params = req(kind="search", query="word " * 40)
    assert len(params["query"]) <= 80
    _, params = req(kind="news", query="topic " * 40)
    assert len(params["topic"]) <= 60  # the gateway's own cap for news


def test_weather_facts_render_hourly_table():
    data = {"city": "London", "date": "2026-10-01", "min_temp_c": 9, "max_temp_c": 16, "rain_mm_total": 6.2,
            "hourly": [{"time": "14:00", "rain_chance_pct": 65}, {"time": "16:00", "rain_chance_pct": 80}]}
    text = lk.render("weather", data)
    assert "16:00 80%" in text and "London" in text and "[web]" in text
