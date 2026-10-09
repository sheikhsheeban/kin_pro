"""Web search: Brave Search API with BRAVE_API_KEY, otherwise Wikipedia (no key)."""

from __future__ import annotations

import html
import json
import re
from typing import Any

from kin.gateway.http import Exchange, HttpClient
from kin.gateway.schemas import SearchParams

_TAGS = re.compile(r"<[^>]+>")


def _clean(text: str | None, limit: int = 280) -> str:
    t = html.unescape(_TAGS.sub("", text or "")).strip()
    return t if len(t) <= limit else t[: limit - 1].rstrip() + "…"


class WikipediaSearch:
    name = "wikipedia"
    hosts = frozenset({"en.wikipedia.org"})

    def fetch(self, params: SearchParams, http: HttpClient, ex: Exchange) -> dict[str, Any]:
        found = json.loads(http.get("https://en.wikipedia.org/w/rest.php/v1/search/page", {"q": params.query, "limit": 3}, ex))
        pages = found.get("pages", [])
        results = []
        if pages:
            top = pages[0]
            summary = json.loads(http.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{top['key']}", None, ex))
            results.append({"title": top["title"], "snippet": _clean(summary.get("extract"), 500), "source": "Wikipedia"})
            results += [
                {"title": p["title"], "snippet": _clean(p.get("description") or p.get("excerpt")), "source": "Wikipedia"}
                for p in pages[1:]
            ]
        return {"query": params.query, "results": results}


class BraveSearch:
    name = "brave"
    hosts = frozenset({"api.search.brave.com"})

    def __init__(self, api_key: str) -> None:
        self._key = api_key

    def fetch(self, params: SearchParams, http: HttpClient, ex: Exchange) -> dict[str, Any]:
        body = http.get(
            "https://api.search.brave.com/res/v1/web/search",
            {"q": params.query, "count": 3, "safesearch": "moderate"},
            ex,
            headers={"X-Subscription-Token": self._key, "Accept": "application/json"},
        )
        web = json.loads(body).get("web", {}).get("results", [])[:3]
        return {
            "query": params.query,
            "results": [{"title": _clean(r.get("title"), 120), "snippet": _clean(r.get("description")), "source": r.get("profile", {}).get("name", "web")} for r in web],
        }
