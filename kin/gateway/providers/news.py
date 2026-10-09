"""News headlines: NewsAPI with NEWSAPI_KEY, otherwise the Google News RSS search feed (no key)."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from typing import Any

from kin.gateway.http import Exchange, HttpClient, UpstreamError
from kin.gateway.providers.search import _clean
from kin.gateway.schemas import NewsParams


class GoogleNewsRss:
    name = "google-news-rss"
    hosts = frozenset({"news.google.com"})

    def fetch(self, params: NewsParams, http: HttpClient, ex: Exchange) -> dict[str, Any]:
        body = http.get("https://news.google.com/rss/search", {"q": params.topic, "hl": "en-GB", "gl": "GB", "ceid": "GB:en"}, ex)
        try:
            root = ET.fromstring(body)  # bundled expat limits entity expansion
        except ET.ParseError as e:
            raise UpstreamError("news feed was not valid XML") from e
        items = []
        for item in root.iter("item"):
            source = item.find("source")
            items.append(
                {
                    "headline": _clean(item.findtext("title"), 160),
                    "source": source.text if source is not None else None,
                    "published": item.findtext("pubDate"),
                }
            )
            if len(items) == 5:
                break
        return {"topic": params.topic, "headlines": items}


class NewsApi:
    name = "newsapi"
    hosts = frozenset({"newsapi.org"})

    def __init__(self, api_key: str) -> None:
        self._key = api_key

    def fetch(self, params: NewsParams, http: HttpClient, ex: Exchange) -> dict[str, Any]:
        body = http.get(
            "https://newsapi.org/v2/everything",
            {"q": params.topic, "pageSize": 5, "sortBy": "publishedAt", "language": "en", "apiKey": self._key},
            ex,
            redact={"apiKey"},
        )
        arts = json.loads(body).get("articles", [])[:5]
        return {
            "topic": params.topic,
            "headlines": [{"headline": _clean(a.get("title"), 160), "source": (a.get("source") or {}).get("name"), "published": a.get("publishedAt")} for a in arts],
        }
