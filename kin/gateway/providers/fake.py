"""Canned providers for demos without internet and for tests (`kin gateway --providers fake`).

Deterministic: an afternoon of rain, a stock search result, two headlines.
They still go through validation, scrubbing, offline mode and the audit log.
"""

from __future__ import annotations

from typing import Any

from kin.gateway.http import Exchange, HttpClient

_RAIN = {6: 5, 8: 10, 10: 20, 12: 35, 14: 65, 16: 80, 18: 70, 20: 40, 22: 15}


class FakeWeather:
    name = "fake-weather"
    hosts: frozenset[str] = frozenset()

    def fetch(self, params, http: HttpClient, ex: Exchange) -> dict[str, Any]:
        ex.requests.append(f"FAKE weather city={params.city} date={params.date.isoformat()}")
        hourly = [
            {"time": f"{h:02d}:00", "rain_chance_pct": p, "rain_mm": round(p / 50, 1), "temp_c": 11 + h // 4,
             "sky": "rain" if p >= 60 else "overcast" if p >= 30 else "partly cloudy"}
            for h, p in _RAIN.items()
        ]
        ex.bytes_received = 640
        return {"city": params.city.title(), "country": None, "date": params.date.isoformat(), "max_temp_c": 16,
                "min_temp_c": 9, "rain_mm_total": 6.2, "max_rain_chance_pct": 80, "sky": "rain", "max_wind_kmh": 18,
                "uv_index_max": 2, "humidity_pct": 81, "hourly": hourly}


class FakeSearch:
    name = "fake-search"
    hosts: frozenset[str] = frozenset()

    def fetch(self, params, http: HttpClient, ex: Exchange) -> dict[str, Any]:
        ex.requests.append(f"FAKE search q={params.query}")
        ex.bytes_received = 300
        return {"query": params.query, "results": [{"title": params.query.title(), "snippet": f"{params.query} is a topic with a stock answer.", "source": "fake"}]}


class FakeNews:
    name = "fake-news"
    hosts: frozenset[str] = frozenset()

    def fetch(self, params, http: HttpClient, ex: Exchange) -> dict[str, Any]:
        ex.requests.append(f"FAKE news topic={params.topic}")
        ex.bytes_received = 400
        return {"topic": params.topic, "headlines": [
            {"headline": f"{params.topic.title()} makes headlines again", "source": "Fake Times", "published": None},
            {"headline": f"Experts discuss {params.topic}", "source": "Fake Post", "published": None},
        ]}


def fake_providers() -> dict[str, Any]:
    return {"weather": FakeWeather(), "search": FakeSearch(), "news": FakeNews()}
