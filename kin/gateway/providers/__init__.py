"""Lookup providers. Each turns validated params into a few compact facts.

Providers return small, plain data (never raw pages), so the LLM gets facts
to reason over and the response the brain receives stays tiny.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

from kin.gateway.http import Exchange, HttpClient


class Provider(Protocol):
    name: str
    hosts: frozenset[str]

    def fetch(self, params: Any, http: HttpClient, ex: Exchange) -> dict[str, Any]: ...


def default_providers() -> dict[str, Provider]:
    """Keyless providers by default; keyed upgrades when keys are present (.env)."""
    from kin.gateway.providers.news import GoogleNewsRss, NewsApi
    from kin.gateway.providers.search import BraveSearch, WikipediaSearch
    from kin.gateway.providers.weather import OpenMeteo

    return {
        "weather": OpenMeteo(),
        "search": BraveSearch(os.environ["BRAVE_API_KEY"]) if os.environ.get("BRAVE_API_KEY") else WikipediaSearch(),
        "news": NewsApi(os.environ["NEWSAPI_KEY"]) if os.environ.get("NEWSAPI_KEY") else GoogleNewsRss(),
    }
