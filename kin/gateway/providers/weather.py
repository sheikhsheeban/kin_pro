"""Weather from Open-Meteo (no API key). City -> geocode -> forecast for one day.

Returns the day summary plus an hourly table (every 2 hours, 06:00-22:00), so the
model can line the forecast up with a time it finds in memory ("4 pm").
"""

from __future__ import annotations

import json
from typing import Any

from kin.gateway.http import Exchange, HttpClient, UpstreamError
from kin.gateway.schemas import WeatherParams

GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST = "https://api.open-meteo.com/v1/forecast"
HOURS = range(6, 23, 2)

# WMO weather interpretation codes (Open-Meteo "weather_code") -> plain words.
_SKY = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "fog",
        51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
        61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
        71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light showers", 81: "showers",
        82: "heavy showers", 85: "snow showers", 86: "heavy snow showers", 95: "thunderstorm",
        96: "thunderstorm with hail", 99: "thunderstorm with hail"}


def sky(code: Any) -> str | None:
    try:
        return _SKY.get(int(code))
    except (TypeError, ValueError):
        return None


class OpenMeteo:
    name = "open-meteo"
    hosts = frozenset({"geocoding-api.open-meteo.com", "api.open-meteo.com"})

    def fetch(self, params: WeatherParams, http: HttpClient, ex: Exchange) -> dict[str, Any]:
        geo = json.loads(http.get(GEOCODE, {"name": params.city, "count": 1, "language": "en", "format": "json"}, ex))
        if not geo.get("results"):
            raise UpstreamError(f"unknown city {params.city!r}")
        place = geo["results"][0]
        day = params.date.isoformat()
        fc = json.loads(
            http.get(
                FORECAST,
                {
                    "latitude": round(place["latitude"], 2),
                    "longitude": round(place["longitude"], 2),
                    "hourly": "precipitation_probability,precipitation,temperature_2m,weather_code",
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,"
                             "weather_code,wind_speed_10m_max,uv_index_max,relative_humidity_2m_mean",
                    "timezone": "auto",
                    "start_date": day,
                    "end_date": day,
                },
                ex,
            )
        )
        h, d = fc["hourly"], fc["daily"]
        hourly = []
        for i, stamp in enumerate(h["time"]):
            hour = int(stamp[11:13])
            if hour in HOURS:
                hourly.append(
                    {
                        "time": f"{hour:02d}:00",
                        "rain_chance_pct": h["precipitation_probability"][i],
                        "rain_mm": h["precipitation"][i],
                        "temp_c": round(h["temperature_2m"][i]),
                        "sky": sky((h.get("weather_code") or [None] * (i + 1))[i]),
                    }
                )
        return {
            "city": place["name"],
            "country": place.get("country"),
            "date": day,
            "max_temp_c": round(d["temperature_2m_max"][0]),
            "min_temp_c": round(d["temperature_2m_min"][0]),
            "rain_mm_total": d["precipitation_sum"][0],
            "max_rain_chance_pct": d["precipitation_probability_max"][0],
            "sky": sky(_first(d, "weather_code")),
            "max_wind_kmh": _round(_first(d, "wind_speed_10m_max")),
            "uv_index_max": _round(_first(d, "uv_index_max")),
            "humidity_pct": _round(_first(d, "relative_humidity_2m_mean")),
            "hourly": hourly,
        }


def _first(block: dict, key: str) -> Any:
    values = block.get(key) or [None]
    return values[0]


def _round(x: Any) -> int | None:
    return None if x is None else round(x)
