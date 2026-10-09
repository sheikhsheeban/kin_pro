"""The only shapes of request the gateway accepts.

`{"kind": "weather" | "search" | "news", "params": {...}}`, with strict,
length-capped params. Extra keys are rejected, so no transcript, prompt or
memory text can ride along in an unexpected field. The free-text fields are
short and limited to a plain character set.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter, field_validator

MAX_REQUEST_BYTES = 512

_PLAIN = r"^[\w][\w .,'&()\-/]*$"  # letters (any script), digits, spaces, basic punctuation

City = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64, pattern=_PLAIN)]
Query = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120, pattern=_PLAIN)]
Topic = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=60, pattern=_PLAIN)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WeatherParams(_Strict):
    city: City
    date: date

    @field_validator("date")
    @classmethod
    def _near(cls, d: date) -> date:
        today = date.today()
        if not today - timedelta(days=1) <= d <= today + timedelta(days=15):
            raise ValueError("forecasts are only available from yesterday to 15 days ahead")
        return d


class SearchParams(_Strict):
    query: Query

    @field_validator("query")
    @classmethod
    def _words(cls, q: str) -> str:
        if len(q.split()) > 12:
            raise ValueError("search queries are limited to 12 words")
        return q


class NewsParams(_Strict):
    topic: Topic


class WeatherRequest(_Strict):
    kind: Literal["weather"]
    params: WeatherParams


class SearchRequest(_Strict):
    kind: Literal["search"]
    params: SearchParams


class NewsRequest(_Strict):
    kind: Literal["news"]
    params: NewsParams


LookupRequest = Annotated[WeatherRequest | SearchRequest | NewsRequest, Field(discriminator="kind")]
LOOKUP_ADAPTER: TypeAdapter[WeatherRequest | SearchRequest | NewsRequest] = TypeAdapter(LookupRequest)
KINDS = ("weather", "search", "news")


def canonical(req: WeatherRequest | SearchRequest | NewsRequest) -> str:
    """The exact JSON that represents a lookup in the audit log and the Network Monitor."""
    return req.model_dump_json()


def tokens(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower())
